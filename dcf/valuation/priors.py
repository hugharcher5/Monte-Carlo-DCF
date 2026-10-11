"""
Turn a company's history into forecasts and parameter uncertainty.

The Monte Carlo draws each input once per simulated path and holds it for the
whole forecast, so what it needs is uncertainty about each input's long-run
level, not the year-to-year noise of four annual figures.  This module gets
both the central value and that uncertainty from a cross-sectional regression
fitted on about 1,700 US companies (scripts/estimate_base_rates.py):

    forecast  = a + b x (company's trailing average) + c x size
    variance  = v0 x exp(k x size) + v1 x (year-to-year sd)^2 / (years of history)

b below 1 pulls the company toward the base rate (shrinkage).  The variance
falls with size and rises when the company's own history is short or erratic.
Cross-source disagreement (sigma_cross) is added in quadrature on top.

Terminal growth does not depend on the company: it is a long-run nominal growth
assumption tied to the reporting currency's risk-free rate.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_RATES_LIVE = DATA_DIR / "base_rates.json"
BASE_RATES_PRE2015 = DATA_DIR / "base_rates_pre2015.json"

# ── Terminal growth ───────────────────────────────────────────────────────────
TERM_G_MODE: float = 0.025     # long-run nominal growth: about 2% inflation plus a little real growth
TERM_G_HALF: float = 0.010     # PERT runs from mode − 1pp to mode + 1pp (1.5% to 3.5% in USD)
WACC_TG_GAP: float = 0.010     # terminal growth always at least 1pp below WACC

# ── PERT sizing ───────────────────────────────────────────────────────────────
# A symmetric PERT on [mode − h, mode + h] has standard deviation h / sqrt(7),
# so h = sqrt(7) × sd makes the sampled spread match the estimated uncertainty.
PERT_HALF_PER_SD: float = math.sqrt(7.0)

# Floors and caps on the estimated standard deviations (decimal).
GROWTH_SD_MIN: float = 0.02
MARGIN_SD_MIN: float = 0.015
MARGIN_SD_MAX: float = 0.15
CAPEX_SD_MIN: float = 0.002
UNPROVEN_SD_MULT: float = 1.5    # extra width for companies with no profitable history
UNPROVEN_MARGIN_SD_MIN: float = 0.04


@lru_cache(maxsize=4)
def load_base_rates(path: str | Path = BASE_RATES_LIVE) -> dict:
    return json.loads(Path(path).read_text())


def size_of(revenue_local: float, fx_rate: float) -> float:
    """log10 of revenue in USD minus 9 (0 = $1bn)."""
    usd = abs(revenue_local) * (fx_rate if fx_rate > 0 else 1.0)
    return math.log10(max(usd, 1e6)) - 9.0


def _clip_size(fit: dict, size: float) -> float:
    lo, hi = fit.get("size_range", [-np.inf, np.inf])
    return float(min(max(size, lo), hi))


def predict(fit: dict, trailing: float, size: float) -> float:
    s = _clip_size(fit, size)
    return fit["a"] + fit["b"] * trailing + fit.get("c", 0.0) * s


def predict_sd(fit: dict, hist_sd: float, n_years: int, size: float) -> float:
    s = _clip_size(fit, size)
    hist_sd = hist_sd if math.isfinite(hist_sd) else 0.0
    var = fit["v0"] * math.exp(fit.get("k", 0.0) * s) + fit["v1"] * hist_sd ** 2 / max(n_years, 1)
    return math.sqrt(max(var, 0.0))


@dataclass
class ForecastInputs:
    """Central values and standard deviations of the sampled inputs."""
    revenue_growth: float
    revenue_growth_sd: float
    ebit_margin: float           # starting margin (average of forecast years 1-3)
    target_margin: float         # mature margin by the last forecast year
    margin_sd: float             # one margin-level shock moves both start and target
    capex_pct: float
    capex_sd: float
    # decomposition, for display
    growth_trailing: float
    margin_trailing: float
    growth_sd_hist: float
    margin_sd_hist: float
    growth_sd_cross: float
    margin_sd_cross: float
    capex_sd_cross: float
    size: float
    shrink_growth: float         # regression slope b for growth
    shrink_margin: float


def forecast_inputs(drivers, size: float, target_trailing: float, unproven: bool,
                    sigma_cross: dict | None = None, base_rates: dict | None = None,
                    industry_target: bool = False) -> ForecastInputs:
    """Shrink the history toward base rates and attach parameter uncertainty.

    target_trailing: the target margin implied by the company's own history
    (midpoint of average and best margin) or an industry margin for unproven
    companies (industry_target=True, used as is, not shrunk).
    """
    br = base_rates or load_base_rates()
    sc = sigma_cross or {}
    g_fit, m_fit = br["growth"], br["margin"]
    n_g = max(int(drivers.years_used), 1)
    n_m = max(int(getattr(drivers, "n_margin_years", 0) or n_g), 1)

    g0 = predict(g_fit, drivers.revenue_growth, size)
    g_sd_model = predict_sd(g_fit, drivers.std_revenue_growth, n_g, size)
    g_cross = float(sc.get("revenue_growth", 0.0))
    g_sd = max(math.hypot(g_sd_model, g_cross), GROWTH_SD_MIN)

    m0 = predict(m_fit, drivers.ebit_margin, size)
    tm = target_trailing if industry_target else predict(m_fit, target_trailing, size)
    m_sd_model = predict_sd(m_fit, drivers.std_ebit_margin, n_m, size)
    m_cross = float(sc.get("ebit_margin", 0.0))
    m_sd = min(max(math.hypot(m_sd_model, m_cross), MARGIN_SD_MIN), MARGIN_SD_MAX)

    n_c = max(int(getattr(drivers, "n_capex_years", 0) or n_g), 1)
    c_hist = drivers.std_capex_pct if math.isfinite(drivers.std_capex_pct) else 0.0
    c_cross = float(sc.get("capex_pct", 0.0))
    c_sd = max(math.hypot(c_hist / math.sqrt(n_c), c_cross), CAPEX_SD_MIN)

    if unproven:
        g_sd *= UNPROVEN_SD_MULT
        m_sd = max(m_sd * UNPROVEN_SD_MULT, UNPROVEN_MARGIN_SD_MIN)
        c_sd *= UNPROVEN_SD_MULT

    return ForecastInputs(
        revenue_growth=g0, revenue_growth_sd=g_sd, ebit_margin=m0, target_margin=tm,
        margin_sd=m_sd, capex_pct=drivers.capex_pct, capex_sd=c_sd,
        growth_trailing=drivers.revenue_growth, margin_trailing=drivers.ebit_margin,
        growth_sd_hist=drivers.std_revenue_growth, margin_sd_hist=drivers.std_ebit_margin,
        growth_sd_cross=g_cross, margin_sd_cross=m_cross, capex_sd_cross=c_cross,
        size=size, shrink_growth=g_fit["b"], shrink_margin=m_fit["b"],
    )


def terminal_g_params(rf_local: float, wacc: float) -> tuple[float, float, float]:
    """(mode, low, high) for terminal growth.

    The mode is 2.5%, or the reporting currency's risk-free rate if that is
    lower: no business can grow faster than the economy forever, and a
    currency's long-term yield already reflects its expected inflation.  The
    range is the mode ± 1pp, and everything stays at least 1pp below WACC.
    """
    mode = min(TERM_G_MODE, rf_local) if math.isfinite(rf_local) else TERM_G_MODE
    cap = wacc - WACC_TG_GAP
    mode = min(mode, cap)
    lo = mode - TERM_G_HALF
    hi = max(min(mode + TERM_G_HALF, cap), mode)
    return float(mode), float(lo), float(hi)


VARIABLES = ["revenue_growth", "margin", "terminal_g", "wacc", "capex_pct"]


def correlation_matrix(base_rates: dict | None = None) -> np.ndarray:
    """Copula correlations.  Growth, margin and CapEx entries are estimated from
    the cross-section of forecast errors; terminal growth and WACC are market
    and macro inputs with nothing company-level to estimate from, so they are
    independent (0)."""
    c = (base_rates or load_base_rates())["corr"]
    C = np.eye(5)
    for (i, j), v in {(0, 1): c["growth_margin"], (0, 4): c["growth_capex"],
                      (1, 4): c["margin_capex"]}.items():
        C[i, j] = C[j, i] = v
    return C
