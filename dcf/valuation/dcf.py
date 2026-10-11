"""
Pure DCF engine.

value(assumptions)          -> float     intrinsic value per share in USD, no I/O
value_many(assumptions, **draws) -> ndarray  the same maths for many draws at once
detailed_value(assumptions) -> DCFResult  the same maths plus the year-by-year table

All three share one vectorised core (_core), so the Monte Carlo, the base case
and the forecast table can never disagree.

Timing.  Cash flows are discounted to the valuation date (today, or the
rebalance date in a backtest), not to the last fiscal year end.  `stub` is the
fraction of the first forecast year still to come on the valuation date: if
the last 10-K ended four months ago, stub = 8/12, only that share of year 1's
cash flow is counted, and it arrives half-way through the remaining stub.
Every later year's cash flow arrives mid-year (the mid-year convention), and
the terminal value is discounted from the middle of the final year, because
the perpetuity's own cash flows also arrive mid-year.

Currency.  Everything is computed in the reporting currency, discounted at a
WACC built on that currency's own risk-free rate, and converted to USD once at
the spot exchange rate.  Net debt is a balance-sheet figure today, so it is
also converted at spot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ── Module-level constants ────────────────────────────────────────────────────
TERMINAL_G: float            = 0.025   # long-run nominal growth used as the terminal-growth mode
HIGH_GROWTH_THRESHOLD: float = 0.15    # use a 10-year horizon if trailing growth > this

# Size-dependent revenue growth ceiling: g_ceiling(R) = max(terminal_g, A / R**B)
# R = revenue in USD billions at the start of the forecast year.  A company with
# $1T of revenue cannot keep growing 50% a year; the ceiling tightens as revenue
# compounds inside each path.  The two anchors are judgement calls, not fitted
# to data:  R = $250B -> about 25%;  R = $1T -> about 6%.
#   b = ln(25/6) / ln(4) ≈ 1.029;  A = 0.25 × 250^1.029 ≈ 74.0
G_CEILING_A: float = 74.0
G_CEILING_B: float = 1.029


@dataclass
class Assumptions:
    ticker: str
    currency: str               # reporting currency label

    # ── Core DCF inputs ───────────────────────────────────────────────────────
    start_revenue: float        # most recent annual revenue in reporting currency
    revenue_growth: float       # starting growth rate (decimal, e.g. 0.12)
    ebit_margin: float          # starting EBIT margin (year 1 of forecast)
    target_margin: float        # mature EBIT margin (fades linearly to this by year N)
    tax_rate: float
    da_pct: float               # D&A / Revenue
    capex_pct: float            # CapEx / Revenue in year 1
    nwc_pct: float              # operating NWC / Revenue
    terminal_g: float
    wacc: float
    net_debt: float             # total debt − cash, in reporting currency
    diluted_shares: float       # share count the equity value is divided by
    fx_rate: float              # USD per 1 unit of reporting currency (spot)
    forecast_years: int         # 5 (normal) or 10 (high growth)

    # ── WACC decomposition — carried for display; not used in value() ─────────
    rf: float
    erp: float
    beta_adj: float
    cost_of_equity: float
    cost_of_debt_pretax: float
    cost_of_debt_aftertax: float
    equity_weight: float
    debt_weight: float
    implied_rating: str

    # ── Market data — for cross-checks ────────────────────────────────────────
    current_price_usd: float
    current_ebitda: float       # most recent actual EBITDA in local currency
    market_ev_local: float      # market cap + net debt, in local currency

    # ── Timing ────────────────────────────────────────────────────────────────
    stub: float = 1.0           # share of forecast year 1 still to come on the valuation date
    mid_year: bool = True       # cash flows arrive mid-year (False = end of year)


@dataclass
class DCFResult:
    value_per_share_usd: float
    value_per_share_local: float
    equity_value_local: float
    ev_local: float
    pv_explicit: float          # PV of explicit forecast FCFFs
    pv_tv: float                # PV of terminal value
    terminal_value_local: float
    implied_ev_ebitda: float
    forecast: pd.DataFrame      # year-by-year table
    ceiling_hits: int = 0       # forecast years where the size ceiling was binding


def capex_path(capex_pct, da_pct, revenue_growth, terminal_g, n_years):
    """CapEx share of revenue for each forecast year, shape (n, N).

    Net reinvestment (CapEx% − D&A%) fades linearly to the level the terminal
    growth rate needs.  Reinvestment = growth / return on capital, so with the
    company's historical return on capital held fixed, net reinvestment scales
    by terminal_g / revenue_growth.  It is never scaled up: companies already
    growing at or below terminal_g keep their own rate, and a company whose
    CapEx is below D&A keeps its CapEx share unchanged.
    """
    capex_pct, da_pct, revenue_growth, terminal_g = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(x, dtype=float)) for x in (capex_pct, da_pct, revenue_growth, terminal_g)))
    net = capex_pct - da_pct
    scale = np.clip(terminal_g / np.maximum(np.maximum(revenue_growth, terminal_g), 1e-9), 0.0, 1.0)
    end = np.where(net > 0, da_pct + net * scale, capex_pct)
    frac = np.linspace(0.0, 1.0, n_years) if n_years > 1 else np.zeros(1)
    return capex_pct[:, None] + (end - capex_pct)[:, None] * frac[None, :]


def discount_times(stub: float, n_years: int, mid_year: bool = True) -> np.ndarray:
    """Years from the valuation date to each forecast year's cash flow."""
    t = np.arange(1, n_years + 1, dtype=float)
    end = stub + t - 1.0                      # end of each forecast year
    if not mid_year:
        return end
    mid = end - 0.5
    mid[0] = stub / 2.0                       # only the remaining stub of year 1
    return mid


def _core(a: Assumptions, rg, em, tm, tg, wacc, capex, shares, net_debt, keep_paths=False):
    """Vectorised DCF over n draws.  Every input array has shape (n,)."""
    rg, em, tm, tg, wacc, capex, shares, net_debt = (
        np.atleast_1d(np.asarray(x, dtype=float)) for x in (rg, em, tm, tg, wacc, capex, shares, net_debt))
    n = max(len(x) for x in (rg, em, tm, tg, wacc, capex, shares, net_debt))
    rg, em, tm, tg, wacc, capex, shares, net_debt = (
        np.broadcast_to(x, (n,)) for x in (rg, em, tm, tg, wacc, capex, shares, net_debt))
    N = a.forecast_years
    frac = np.linspace(0.0, 1.0, N) if N > 1 else np.zeros(1)

    growth = rg[:, None] + (tg - rg)[:, None] * frac[None, :]
    margin = em[:, None] + (tm - em)[:, None] * frac[None, :]
    cx = capex_path(capex, a.da_pct, rg, tg, N)

    revenue = np.empty((n, N))
    applied_g = np.empty((n, N))
    hits = np.zeros(n, dtype=int)
    prev = np.full(n, float(a.start_revenue))
    for t in range(N):
        g = growth[:, t]
        if a.fx_rate > 0:
            r_usd_b = np.maximum(prev * a.fx_rate / 1e9, 1e-12)
            ceiling = np.maximum(tg, G_CEILING_A / r_usd_b ** G_CEILING_B)
            bound = (g > ceiling) & (prev > 0)
            hits += bound
            g = np.where(bound, ceiling, g)
        applied_g[:, t] = g
        prev = prev * (1.0 + g)
        revenue[:, t] = prev

    ebit = revenue * margin
    nopat = ebit * (1.0 - a.tax_rate)
    da = revenue * a.da_pct
    capex_amt = revenue * cx
    nwc = revenue * a.nwc_pct
    prev_nwc = np.concatenate([np.full((n, 1), a.start_revenue * a.nwc_pct), nwc[:, :-1]], axis=1)
    dnwc = nwc - prev_nwc
    fcff = nopat + da - capex_amt - dnwc

    times = discount_times(a.stub, N, a.mid_year)
    disc = (1.0 + wacc[:, None]) ** (-times[None, :])
    counted = fcff.copy()
    counted[:, 0] *= a.stub                   # only the part of year 1 still to come
    pv = counted * disc
    pv_explicit = pv.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        tv = fcff[:, -1] * (1.0 + tg) / (wacc - tg)
    pv_tv = tv * disc[:, -1]
    ev = pv_explicit + pv_tv
    equity = ev - net_debt
    with np.errstate(divide="ignore", invalid="ignore"):
        vps_local = equity / shares
    vps_local = np.where((wacc > tg) & (shares > 0), vps_local, np.nan)
    out = {"vps_usd": vps_local * a.fx_rate, "vps_local": vps_local, "equity": equity, "ev": ev,
           "pv_explicit": pv_explicit, "pv_tv": pv_tv, "tv": tv, "hits": hits}
    if keep_paths:
        out.update(growth=applied_g, margin=margin, revenue=revenue, ebit=ebit, nopat=nopat, da=da,
                   capex_pct=cx, capex=capex_amt, dnwc=dnwc, fcff=fcff, disc=disc, pv=pv,
                   times=times, counted=counted)
    return out


def value_many(a: Assumptions, *, revenue_growth=None, ebit_margin=None, target_margin=None,
               terminal_g=None, wacc=None, capex_pct=None, diluted_shares=None, net_debt=None,
               return_hits: bool = False):
    """Value per share (USD) for arrays of draws; any input left as None uses the base case.
    Draws where WACC <= terminal growth, or the share count is not positive, return NaN."""
    def pick(x, base):
        return base if x is None else x
    r = _core(a, pick(revenue_growth, a.revenue_growth), pick(ebit_margin, a.ebit_margin),
              pick(target_margin, a.target_margin), pick(terminal_g, a.terminal_g),
              pick(wacc, a.wacc), pick(capex_pct, a.capex_pct),
              pick(diluted_shares, a.diluted_shares), pick(net_debt, a.net_debt))
    return (r["vps_usd"], r["hits"]) if return_hits else r["vps_usd"]


def _check(a: Assumptions) -> None:
    if a.wacc <= a.terminal_g:
        raise ValueError(f"WACC ({a.wacc:.2%}) must exceed terminal_g ({a.terminal_g:.2%}); "
                         "terminal value would be undefined.")
    if not a.diluted_shares > 0:
        raise ValueError("Diluted share count is zero or missing; cannot compute value per share.")


def value(a: Assumptions) -> float:
    """Pure, fast path: intrinsic value per share in USD."""
    _check(a)
    return float(value_many(a)[0])


def detailed_value(a: Assumptions) -> DCFResult:
    """Full computation including the forecast table and cross-checks."""
    _check(a)
    r = _core(a, a.revenue_growth, a.ebit_margin, a.target_margin, a.terminal_g, a.wacc,
              a.capex_pct, a.diluted_shares, a.net_debt, keep_paths=True)
    N = a.forecast_years
    forecast = pd.DataFrame({
        "Year": np.arange(1, N + 1),
        "Growth": r["growth"][0], "Margin%": r["margin"][0], "Revenue": r["revenue"][0],
        "EBIT": r["ebit"][0], "NOPAT": r["nopat"][0], "D&A": r["da"][0],
        "CapEx%": r["capex_pct"][0], "CapEx": r["capex"][0], "ΔNWC": r["dnwc"][0],
        "FCFF": r["fcff"][0], "Years to cash": r["times"], "Disc.": r["disc"][0],
        "PV(FCFF)": r["pv"][0],
    }).set_index("Year")
    ev = float(r["ev"][0])
    ebitda_ok = a.current_ebitda and a.current_ebitda > 0
    return DCFResult(
        value_per_share_usd=float(r["vps_usd"][0]),
        value_per_share_local=float(r["vps_local"][0]),
        equity_value_local=float(r["equity"][0]),
        ev_local=ev,
        pv_explicit=float(r["pv_explicit"][0]),
        pv_tv=float(r["pv_tv"][0]),
        terminal_value_local=float(r["tv"][0]),
        implied_ev_ebitda=ev / a.current_ebitda if ebitda_ok else float("nan"),
        forecast=forecast,
        ceiling_hits=int(r["hits"][0]),
    )
