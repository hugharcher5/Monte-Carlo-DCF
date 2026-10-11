"""
Monte Carlo simulation layer for the DCF engine.

Each simulated path draws five inputs once and holds them for the whole
forecast: starting revenue growth, the EBIT-margin level (one shock moves the
starting and the mature margin together), terminal growth, WACC and CapEx
intensity.  Their central values and standard deviations come from
priors.py; each is drawn from a PERT distribution whose standard deviation
matches the estimate, and they are tied together with a Gaussian copula using
correlations estimated from the cross-section of forecast errors.  Share count
and net debt are also sampled when the data sources disagree about them by
more than 2%.  All draws are valued in one vectorised call (dcf.value_many).
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

from .data      import fetch_raw
from .drivers   import compute_drivers
from .wacc      import compute_wacc
from .dcf       import (Assumptions, value_many, detailed_value, HIGH_GROWTH_THRESHOLD,
                        G_CEILING_A, G_CEILING_B)
from .result    import ValuationResult
from .reconcile import FIELD_TYPE, ALL_FIELDS
from .filings   import apply_filing_data
from .industry_margins import industry_margin
from .priors    import (forecast_inputs, size_of, terminal_g_params, correlation_matrix,
                        load_base_rates, PERT_HALF_PER_SD, WACC_TG_GAP, VARIABLES)
from .gates     import check_not_financial, output_flags
from .reverse_dcf import implied_inputs

log = logging.getLogger(__name__)

# ─────────────────────────── Editable constants ───────────────────────────────

MC_RANDOM_SEED: int     = 42
PROMOTE_THRESHOLD: float = 0.02    # sample shares / net debt when sources disagree by > 2%
CAPEX_PCT_CAP: float    = 0.60
WACC_FLOOR: float       = 0.04

# Plausibility clamps on sampled values (decimal)
CLAMPS = {
    "revenue_growth": (-0.30, 1.00),
    "ebit_margin":    (-0.30, 0.75),
    "target_margin":  (-0.20, 0.75),
    "capex_pct":      (0.0, CAPEX_PCT_CAP),
}


# ─────────────────────────────── Helpers ──────────────────────────────────────

def _ensure_psd(C: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return C, or the nearest positive-definite correlation matrix if C is not."""
    C = (C + C.T) / 2.0
    lam, V = np.linalg.eigh(C)
    if np.all(lam > eps):
        return C
    C2 = V @ np.diag(np.maximum(lam, eps)) @ V.T
    d = np.sqrt(np.diag(C2))
    return C2 / np.outer(d, d)


def pert_ppf(u: np.ndarray, lo: float, mode: float, hi: float) -> np.ndarray:
    """Inverse PERT CDF at uniform quantiles u.  Point mass when lo == hi; the
    mode is clamped into [lo, hi] so the Beta parameters stay valid."""
    u = np.asarray(u, dtype=float)
    if abs(hi - lo) < 1e-12:
        return np.full(u.shape, mode, dtype=float)
    mode = min(max(mode, lo), hi)
    r = hi - lo
    a = 1.0 + 4.0 * (mode - lo) / r
    b = 1.0 + 4.0 * (hi - mode) / r
    return lo + r * scipy.stats.beta.ppf(np.clip(u, 1e-9, 1.0 - 1e-9), a, b)


def pert_bounds(mode: float, sd: float, clamp: tuple[float, float] = (-np.inf, np.inf)) -> tuple[float, float]:
    """Bounds whose symmetric PERT has standard deviation sd, clipped to clamp."""
    h = PERT_HALF_PER_SD * max(sd, 0.0)
    return max(mode - h, clamp[0]), min(mode + h, clamp[1])


def gaussian_uniforms(n: int, C: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(n, k) correlated uniforms via a Gaussian copula."""
    L = np.linalg.cholesky(C)
    return scipy.stats.norm.cdf(rng.standard_normal((n, C.shape[0])) @ L.T)


def is_unproven(drivers) -> bool:
    """No profitable history (average EBIT margin at or below zero).  A short
    history alone does not make a company unproven: the estimated uncertainty
    already widens when there are fewer years to average (priors.predict_sd)."""
    return drivers.ebit_margin <= 0


def target_margin_for(drivers, industry: str = "") -> tuple[float, str, bool]:
    """Mature EBIT margin implied by history, how it was chosen, and whether it is
    an industry figure (used as is) rather than the company's own (shrunk later).

    Profitable history: midpoint of average and best historical EBIT margin.
    No profitable history: the industry's average operating margin (Damodaran,
    January 2026), or the US market average (about 13%) if unknown.
    """
    if drivers.ebit_margin > 0:
        tm = 0.5 * (drivers.ebit_margin + drivers.best_ebit_margin)
        return tm, "midpoint of historical average and best EBIT margin, shrunk toward the base rate", False
    tm, label = industry_margin(industry)
    return tm, label, True


def stub_fraction(fiscal_year_end, valuation_date) -> float:
    """Share of the first forecast year still to come on the valuation date."""
    if fiscal_year_end is None:
        return 1.0
    elapsed = (pd.Timestamp(valuation_date) - pd.Timestamp(fiscal_year_end)).days / 365.25
    return float(min(max(1.0 - elapsed, 0.0), 1.0))


def build_base(raw, drivers, wacc_r, fi, tg_mode: float, stub: float) -> Assumptions:
    net_debt = raw.total_debt - raw.cash
    fy = 10 if drivers.revenue_growth > HIGH_GROWTH_THRESHOLD else 5
    ebitda = ((float(raw.ebit.iloc[-1]) if not raw.ebit.empty else 0.0) +
              (float(raw.da.iloc[-1]) if not raw.da.empty else 0.0))
    return Assumptions(
        ticker=raw.ticker, currency=raw.currency,
        start_revenue=float(raw.revenue.iloc[-1]),
        revenue_growth=float(np.clip(fi.revenue_growth, *CLAMPS["revenue_growth"])),
        ebit_margin=float(np.clip(fi.ebit_margin, *CLAMPS["ebit_margin"])),
        target_margin=float(np.clip(fi.target_margin, *CLAMPS["target_margin"])),
        tax_rate=drivers.tax_rate, da_pct=drivers.da_pct,
        capex_pct=float(np.clip(fi.capex_pct, *CLAMPS["capex_pct"])),
        nwc_pct=drivers.nwc_pct, terminal_g=tg_mode, wacc=wacc_r.wacc,
        net_debt=net_debt, diluted_shares=raw.diluted_shares, fx_rate=raw.fx_rate,
        forecast_years=fy,
        rf=wacc_r.rf, erp=wacc_r.erp, beta_adj=wacc_r.beta_adj,
        cost_of_equity=wacc_r.cost_of_equity, cost_of_debt_pretax=wacc_r.cost_of_debt_pretax,
        cost_of_debt_aftertax=wacc_r.cost_of_debt_aftertax,
        equity_weight=wacc_r.equity_weight, debt_weight=wacc_r.debt_weight,
        implied_rating=wacc_r.implied_rating,
        current_price_usd=raw.current_price_usd, current_ebitda=ebitda,
        market_ev_local=raw.market_cap_local + net_debt, stub=stub,
    )


def prepare(raw, reconcile_result=None, sigma_cross=None, as_of=None, base_rates=None):
    """Everything up to the base case: filings, drivers, WACC, priors, base Assumptions."""
    sigma_cross = sigma_cross or {}
    provenance = apply_filing_data(raw, reconcile_result)
    drvrs = compute_drivers(raw)
    wacc_r = compute_wacc(raw, drvrs, as_of=as_of)
    unproven = is_unproven(drvrs)
    tm_trail, tm_basis, industry_target = target_margin_for(drvrs, getattr(raw, "industry", ""))
    fi = forecast_inputs(drvrs, size_of(float(raw.revenue.iloc[-1]), raw.fx_rate), tm_trail,
                         unproven, sigma_cross, base_rates or load_base_rates(), industry_target)
    tg = terminal_g_params(wacc_r.rf, wacc_r.wacc)
    vdate = pd.Timestamp(as_of or getattr(raw, "valuation_date", None) or pd.Timestamp.today()).normalize()
    fye = raw.revenue.index[-1]
    base = build_base(raw, drvrs, wacc_r, fi, tg[0], stub_fraction(fye, vdate))
    return dict(provenance=provenance, drivers=drvrs, wacc=wacc_r, unproven=unproven,
                tm_basis=tm_basis, fi=fi, tg=tg, base=base, valuation_date=vdate, fye=fye)


# ──────────────────────────── Core simulation ─────────────────────────────────

def simulation_bounds(base: Assumptions, fi, tg: tuple, sw: float, zero_width: bool = False) -> dict:
    """PERT (lo, mode, hi) for every copula variable.  The margin entry is the
    shock added to both the starting and the mature margin."""
    if zero_width:
        return {"revenue_growth": (base.revenue_growth,) * 3, "margin": (0.0, 0.0, 0.0),
                "terminal_g": (base.terminal_g,) * 3, "wacc": (base.wacc,) * 3,
                "capex_pct": (base.capex_pct,) * 3}
    g_lo, g_hi = pert_bounds(base.revenue_growth, fi.revenue_growth_sd, CLAMPS["revenue_growth"])
    h_m = PERT_HALF_PER_SD * fi.margin_sd
    w_lo, w_hi = pert_bounds(base.wacc, sw, (WACC_FLOOR, np.inf))
    c_lo, c_hi = pert_bounds(base.capex_pct, fi.capex_sd, CLAMPS["capex_pct"])
    return {"revenue_growth": (g_lo, base.revenue_growth, g_hi),
            "margin": (-h_m, 0.0, h_m),
            "terminal_g": (tg[1], base.terminal_g, tg[2]),
            "wacc": (w_lo, base.wacc, w_hi),
            "capex_pct": (c_lo, base.capex_pct, c_hi)}


def draw_inputs(base: Assumptions, bounds: dict, corr: np.ndarray, n: int,
                rng: np.random.Generator, shares_sd: float = 0.0, net_debt_sd: float = 0.0) -> dict:
    """Correlated draws of every sampled input, shape (n,) each."""
    U = gaussian_uniforms(n, corr, rng)
    x = {v: pert_ppf(U[:, i], *bounds[v]) for i, v in enumerate(VARIABLES)}
    shock = x.pop("margin")
    x["ebit_margin"] = np.clip(base.ebit_margin + shock, *CLAMPS["ebit_margin"])
    x["target_margin"] = np.clip(base.target_margin + shock, *CLAMPS["target_margin"])
    x["terminal_g"] = np.minimum(x["terminal_g"], x["wacc"] - WACC_TG_GAP)
    if shares_sd > 0:
        lo, hi = pert_bounds(base.diluted_shares, shares_sd, (base.diluted_shares * 0.5, np.inf))
        x["diluted_shares"] = pert_ppf(rng.uniform(size=n), lo, base.diluted_shares, hi)
    if net_debt_sd > 0:
        lo, hi = pert_bounds(base.net_debt, net_debt_sd)
        x["net_debt"] = pert_ppf(rng.uniform(size=n), lo, base.net_debt, hi)
    return x


def simulate(base: Assumptions, bounds: dict, corr: np.ndarray, n: int, rng: np.random.Generator,
             shares_sd: float = 0.0, net_debt_sd: float = 0.0):
    """(values, hits, draws) for n simulated paths, NaN draws removed."""
    x = draw_inputs(base, bounds, corr, n, rng, shares_sd, net_debt_sd)
    vals, hits = value_many(base, return_hits=True, **x)
    ok = np.isfinite(vals)
    if not ok.all():
        log.info("%s: %d of %d draws invalid (WACC <= terminal growth)", base.ticker, (~ok).sum(), n)
    return vals[ok], hits[ok], {k: v[ok] for k, v in x.items()}


# ───────────────────────────── Public functions ───────────────────────────────

def run_valuation(
    ticker: str,
    n_sims: int = 10_000,
    sigma_cross: dict[str, float] | None = None,
    reconcile_result=None,
    raw=None,
    as_of=None,
    base_rates: dict | None = None,
    light: bool = False,
) -> ValuationResult:
    """Value one company.

    raw: an already-fetched RawData (fetched from Yahoo when None).
    as_of: valuation date for backtests (point-in-time risk-free rate, stub period).
    light: skip the reverse DCF and transparency payload (faster, for backtests).
    """
    sigma_cross = dict(sigma_cross or {})
    if raw is None:
        raw = fetch_raw(ticker)
    check_not_financial(raw.ticker, getattr(raw, "sector", ""), getattr(raw, "industry", ""))
    P = prepare(raw, reconcile_result, sigma_cross, as_of, base_rates)
    drvrs, wacc_r, fi, tg, base = P["drivers"], P["wacc"], P["fi"], P["tg"], P["base"]
    unproven = P["unproven"]
    dcf_r = detailed_value(base)
    sw = wacc_r.std_wacc * (1.5 if unproven else 1.0)

    terminal_fcff = float(dcf_r.forecast["FCFF"].iloc[-1])
    bc_value = dcf_r.value_per_share_usd
    dcf_applicable = terminal_fcff > 0 and bc_value > 0
    gate_triggers = []
    if terminal_fcff <= 0:
        gate_triggers.append("terminal_fcff_non_positive")
    if base.ebit_margin * base.start_revenue < 0:
        gate_triggers.append("current_ebit_negative")
    n_neg = int((dcf_r.forecast["FCFF"] < 0).sum())
    if n_neg:
        gate_triggers.append(f"forecast_fcff_negative_in_{n_neg}_of_{base.forecast_years}_years")
    if bc_value < 0:
        gate_triggers.append("base_case_intrinsic_negative")

    shares_sd = sigma_cross.get("diluted_shares", 0.0)
    shares_sd = shares_sd if base.diluted_shares > 0 and shares_sd / base.diluted_shares > PROMOTE_THRESHOLD else 0.0
    nd_sd = sigma_cross.get("net_debt", 0.0)
    nd_sd = nd_sd if abs(base.net_debt) > 1e-9 and nd_sd / abs(base.net_debt) > PROMOTE_THRESHOLD else 0.0

    recon_fields, recon_sigma = {}, {}
    if reconcile_result is not None:
        for f in ALL_FIELDS:
            d = reconcile_result.disagreement.get(f, float("nan"))
            recon_fields[f] = {"source": reconcile_result.field_sources.get(f, "Yahoo"),
                               "disagree_pct": d * 100 if math.isfinite(d) else None,
                               "field_type": FIELD_TYPE.get(f, "DATA")}
    recon_sigma = {
        "revenue_growth": {"sigma_hist": math.sqrt(max(fi.revenue_growth_sd ** 2 - fi.growth_sd_cross ** 2, 0)),
                           "sigma_cross": fi.growth_sd_cross, "sigma_eff": fi.revenue_growth_sd},
        "margin": {"sigma_hist": math.sqrt(max(fi.margin_sd ** 2 - fi.margin_sd_cross ** 2, 0)),
                   "sigma_cross": fi.margin_sd_cross, "sigma_eff": fi.margin_sd},
        "capex_pct": {"sigma_hist": math.sqrt(max(fi.capex_sd ** 2 - fi.capex_sd_cross ** 2, 0)),
                      "sigma_cross": fi.capex_sd_cross, "sigma_eff": fi.capex_sd},
    }
    if shares_sd:
        recon_sigma["diluted_shares"] = {"sigma_hist": 0.0, "sigma_cross": shares_sd, "sigma_eff": shares_sd}
    if nd_sd:
        recon_sigma["net_debt"] = {"sigma_hist": 0.0, "sigma_cross": nd_sd, "sigma_eff": nd_sd}

    nan = float("nan")
    common = dict(
        ticker=ticker.upper(), currency=base.currency, current_price_usd=base.current_price_usd,
        drivers=drvrs, wacc_result=wacc_r, assumptions=base, dcf=dcf_r,
        sigma_cross=sigma_cross, recon_fields=recon_fields, recon_sigma=recon_sigma, sw=sw,
        dcf_applicable=dcf_applicable, data_provenance=P["provenance"], unproven=unproven,
        target_margin_basis=P["tm_basis"], forecast_inputs=fi, terminal_g_range=tg,
        valuation_date=P["valuation_date"], fiscal_year_end=P["fye"],
    )
    if n_sims <= 0:
        return ValuationResult(sims=np.array([]), n_valid=0, copula_label="", consistency_pass=False,
                               cc_p50=nan, p10=nan, p25=nan, p50=nan, p75=nan, p90=nan,
                               mean_val=nan, std_val=nan, pct_undervalued=nan, **common)

    corr = _ensure_psd(correlation_matrix(base_rates))
    # Zero-width check: with every range collapsed the simulation must return the base case.
    cc_vals, _, _ = simulate(base, simulation_bounds(base, fi, tg, sw, zero_width=True), corr, 50,
                             np.random.default_rng(0))
    cc_p50 = float(np.median(cc_vals)) if len(cc_vals) else nan
    cc_ok = abs(cc_p50 - bc_value) < 0.01

    bounds = simulation_bounds(base, fi, tg, sw)
    sims, hits, draws = simulate(base, bounds, corr, n_sims, np.random.default_rng(MC_RANDOM_SEED),
                                 shares_sd, nd_sd)
    if len(sims) == 0:
        raise ValueError(f"{ticker.upper()}: every Monte Carlo draw was invalid.")

    p10, p25, p50, p75, p90 = (float(x) for x in np.percentile(sims, [10, 25, 50, 75, 90]))
    price = base.current_price_usd
    flags = list(output_flags(price, p10, p50, base.stub, wacc_r.rf_is_local, base.currency, unproven,
                              base.wacc, base.terminal_g))
    implied = {} if light else implied_inputs(base, price)

    transparency = {}
    if not light:
        transparency = _transparency(base, fi, tg, sw, bounds, corr, draws, hits, n_sims, cc_ok,
                                     cc_p50, bc_value, terminal_fcff, gate_triggers, dcf_applicable,
                                     wacc_r, shares_sd, nd_sd)

    return ValuationResult(
        sims=sims, n_valid=len(sims), copula_label="Gaussian", consistency_pass=cc_ok, cc_p50=cc_p50,
        p10=p10, p25=p25, p50=p50, p75=p75, p90=p90,
        mean_val=float(np.mean(sims)), std_val=float(np.std(sims)),
        pct_undervalued=float(np.mean(sims > price)) * 100,
        transparency=transparency, flags=flags, implied=implied, **common)


def _transparency(base, fi, tg, sw, bounds, corr, draws, hits, n_sims, cc_ok, cc_p50, bc_value,
                  terminal_fcff, gate_triggers, dcf_applicable, wacc_r, shares_sd, nd_sd) -> dict:
    h_m = bounds["margin"][2]
    dist = {
        "revenue_growth": {"lo": bounds["revenue_growth"][0], "mode": base.revenue_growth,
                           "hi": bounds["revenue_growth"][2], "sd": fi.revenue_growth_sd,
                           "clamp": CLAMPS["revenue_growth"]},
        "ebit_margin": {"lo": max(base.ebit_margin - h_m, CLAMPS["ebit_margin"][0]),
                        "mode": base.ebit_margin,
                        "hi": min(base.ebit_margin + h_m, CLAMPS["ebit_margin"][1]),
                        "sd": fi.margin_sd, "clamp": CLAMPS["ebit_margin"]},
        "target_margin": {"lo": max(base.target_margin - h_m, CLAMPS["target_margin"][0]),
                          "mode": base.target_margin,
                          "hi": min(base.target_margin + h_m, CLAMPS["target_margin"][1]),
                          "sd": fi.margin_sd, "clamp": CLAMPS["target_margin"]},
        "terminal_g": {"lo": bounds["terminal_g"][0], "mode": base.terminal_g,
                       "hi": bounds["terminal_g"][2], "sd": None, "clamp": (tg[1], tg[2])},
        "wacc": {"lo": bounds["wacc"][0], "mode": base.wacc, "hi": bounds["wacc"][2], "sd": sw,
                 "clamp": (WACC_FLOOR, float("inf"))},
        "capex_pct": {"lo": bounds["capex_pct"][0], "mode": base.capex_pct,
                      "hi": bounds["capex_pct"][2], "sd": fi.capex_sd, "clamp": CLAMPS["capex_pct"]},
    }
    if shares_sd:
        lo, hi = pert_bounds(base.diluted_shares, shares_sd, (base.diluted_shares * 0.5, np.inf))
        dist["diluted_shares"] = {"lo": lo, "mode": base.diluted_shares, "hi": hi, "sd": shares_sd, "clamp": None}
    if nd_sd:
        lo, hi = pert_bounds(base.net_debt, nd_sd)
        dist["net_debt"] = {"lo": lo, "mode": base.net_debt, "hi": hi, "sd": nd_sd, "clamp": None}
    names = ["revenue_growth", "ebit_margin", "terminal_g", "wacc", "capex_pct"]
    realised = np.corrcoef(np.vstack([draws[k] for k in names])) if len(draws["wacc"]) > 2 else np.eye(5)
    return {
        "samples": draws,
        "distribution_params": dist,
        "correlation_target": corr.copy(),
        "correlation_realized": realised,
        "dcf_applicability": {"dcf_applicable": dcf_applicable, "terminal_fcff": terminal_fcff,
                              "current_ebit": base.ebit_margin * base.start_revenue,
                              "base_case_intrinsic": bc_value, "triggers": gate_triggers},
        "g_ceiling": {"A": G_CEILING_A, "b": G_CEILING_B,
                      "pct_paths_bound": float(np.mean(hits > 0)) * 100 if len(hits) else 0.0,
                      "avg_years_bound": float(np.mean(hits)) if len(hits) else 0.0},
        "validation": {"zero_width_test_passed": cc_ok, "mc_p50_at_zero_width": cc_p50,
                       "deterministic_base": bc_value, "abs_diff": abs(cc_p50 - bc_value)},
        "n_sims": n_sims, "random_seed": MC_RANDOM_SEED,
    }


# ───────────────────────────── Terminal output ────────────────────────────────

def _save_histogram(result: ValuationResult) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir = Path(__file__).parent.parent / "output"
    os.makedirs(output_dir, exist_ok=True)
    path = str(output_dir / f"{result.ticker}_montecarlo.png")
    sims, price = result.sims, result.current_price_usd
    p1, p99 = np.percentile(sims, [1, 99])
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(sims[(sims >= p1) & (sims <= p99)], bins=120, color="steelblue", alpha=0.75)
    ax.axvline(price, color="crimson", lw=2, label=f"Market price ${price:.2f}")
    ax.axvline(result.p50, color="seagreen", lw=2, ls="--", label=f"P50 ${result.p50:.2f}")
    ax.axvline(result.p10, color="darkorange", lw=1.2, ls=":", label=f"P10 / P90 ${result.p10:.2f} / ${result.p90:.2f}")
    ax.axvline(result.p90, color="darkorange", lw=1.2, ls=":")
    ax.set_xlabel("Intrinsic value per share (USD)")
    ax.set_title(f"{result.ticker}: Monte Carlo DCF ({result.n_valid:,} simulations)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def print_valuation(result: ValuationResult) -> None:
    """Terminal formatter for run_mc.py."""
    from .report import print_drivers, print_wacc
    print_drivers(result)
    print_wacc(result)
    dp = result.transparency.get("distribution_params", {})
    print(f"\n  PERT ranges (standard deviation matched to the estimate):")
    for k, d in dp.items():
        if k in ("diluted_shares", "net_debt"):
            print(f"  {k:<16} {d['lo']/1e9:>10.3f}B {d['mode']/1e9:>10.3f}B {d['hi']/1e9:>10.3f}B")
        else:
            print(f"  {k:<16} {d['lo']:>10.2%} {d['mode']:>10.2%} {d['hi']:>10.2%}")
    print(f"\n  Zero-width check: {'PASS' if result.consistency_pass else 'FAIL'} "
          f"(base ${result.dcf.value_per_share_usd:.4f}, MC ${result.cc_p50:.4f})")
    print(f"\n  {result.ticker}: price ${result.current_price_usd:.2f}  base case ${result.dcf.value_per_share_usd:.2f}")
    for lbl, v in (("P10", result.p10), ("P25", result.p25), ("P50", result.p50),
                   ("P75", result.p75), ("P90", result.p90)):
        print(f"  {lbl}: ${v:,.2f}")
    print(f"  Share of simulated scenarios above the price: {result.pct_undervalued:.1f}%")
    g = result.implied.get("revenue_growth")
    if g is not None:
        print(f"  Starting growth the price implies: {g:.1%} (model: {result.assumptions.revenue_growth:.1%})")
    for f in result.flags:
        print(f"  ! {f}")
    print(f"\n  Histogram -> {_save_histogram(result)}")


def run_monte_carlo(ticker: str, n_sims: int = 10_000) -> None:
    from .pipeline import value_ticker
    print_valuation(value_ticker(ticker, n_sims=n_sims)[0])
