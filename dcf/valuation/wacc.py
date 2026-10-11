"""Compute WACC via CAPM (cost of equity) + Damodaran synthetic rating (cost of debt).

The risk-free rate is the 10-year government bond yield of the company's
reporting currency, because the cash flows are forecast and discounted in that
currency.  US dollars use FRED's DGS10; other currencies use the OECD long-term
government bond yields on FRED (series IRLTLT01xxM156N).  Currencies without a
series fall back to the US yield, and the result says so.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .data import RawData
from .drivers import Drivers

load_dotenv()
log = logging.getLogger(__name__)

_WACC_SIGMA_FALLBACK: float = 0.015
_RF_FALLBACK: float = 0.043

# ── Equity risk premium ──────────────────────────────────────────────────────
# A fixed assumption, not estimated here.  4.5% is close to Damodaran's implied
# US ERP at the start of 2026 (it has ranged roughly 4% to 6% since 2008).  No
# country risk premium is added for companies based outside the US.
ERP: float = 0.045

# ── Damodaran synthetic-rating table (large non-financial companies, January 2024) ─
# Rows: (coverage_min_inclusive, coverage_max_exclusive, rating_label, spread)
_SYNTHETIC_RATING: list[tuple[float, float, str, float]] = [
    (12.50, float('inf'), 'Aaa/AAA', 0.0063),
    ( 9.50,       12.50, 'Aa2/AA',  0.0078),
    ( 7.50,        9.50, 'A1/A+',   0.0098),
    ( 6.00,        7.50, 'A2/A',    0.0113),
    ( 4.50,        6.00, 'A3/A-',   0.0128),
    ( 4.00,        4.50, 'Baa2/BBB',0.0163),
    ( 3.50,        4.00, 'Ba1/BB+', 0.0238),
    ( 3.00,        3.50, 'Ba2/BB',  0.0313),
    ( 2.50,        3.00, 'B1/B+',   0.0388),
    ( 2.00,        2.50, 'B2/B',    0.0463),
    ( 1.50,        2.00, 'B3/B-',   0.0563),
    ( 1.25,        1.50, 'Caa/CCC', 0.0713),
    ( 0.80,        1.25, 'Ca2/CC',  0.0863),
    ( 0.50,        0.80, 'C2/C',    0.1113),
    (float('-inf'), 0.50, 'D2/D',   0.1413),
]

# FRED series for each currency's 10-year government bond yield (percent).
RF_SERIES: dict[str, str] = {
    'USD': 'DGS10',
    'EUR': 'IRLTLT01EZM156N', 'GBP': 'IRLTLT01GBM156N', 'JPY': 'IRLTLT01JPM156N',
    'CHF': 'IRLTLT01CHM156N', 'CAD': 'IRLTLT01CAM156N', 'AUD': 'IRLTLT01AUM156N',
    'SEK': 'IRLTLT01SEM156N', 'NOK': 'IRLTLT01NOM156N', 'DKK': 'IRLTLT01DKM156N',
    'KRW': 'IRLTLT01KRM156N', 'NZD': 'IRLTLT01NZM156N', 'MXN': 'IRLTLT01MXM156N',
    'INR': 'INDIRLTLT01STM', 'ZAR': 'IRLTLT01ZAM156N', 'ILS': 'IRLTLT01ILM156N',
}
# Currencies pegged to the dollar use the US yield.
USD_PEGGED = {'HKD'}

# Network and parsing errors a FRED or Yahoo fetch can raise.
_FETCH_ERRORS = (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError)


@dataclass
class WACCResult:
    rf: float
    rf_source: str
    beta_raw: float
    beta_adj: float
    erp: float
    cost_of_equity: float
    coverage_ratio: float
    implied_rating: str
    credit_spread: float
    cost_of_debt_pretax: float
    cost_of_debt_aftertax: float
    equity_value: float     # market cap in local currency
    debt_value: float       # total debt in local currency
    equity_weight: float
    debt_weight: float
    wacc: float
    std_wacc:            float   # historical σ; equals _WACC_SIGMA_FALLBACK when < 3 years
    wacc_sigma_fallback: bool    # True when insufficient history for reconstruction
    rf_is_local: bool = True     # False when the US yield stood in for a missing local one


_SERIES_CACHE: dict[str, pd.Series] = {}


def _fred_series(series_id: str) -> pd.Series:
    """Daily or monthly FRED series as decimals, memoised per process.
    Empty when FRED is unavailable."""
    if series_id in _SERIES_CACHE:
        return _SERIES_CACHE[series_id]
    s = pd.Series(dtype=float)
    api_key = os.getenv('FRED_API_KEY', '').strip()
    if api_key and api_key != 'your_api_key_here':
        try:
            from fredapi import Fred
            s = Fred(api_key=api_key).get_series(series_id).dropna() / 100.0
        except _FETCH_ERRORS as e:
            log.warning("FRED fetch of %s failed (%s)", series_id, e)
    _SERIES_CACHE[series_id] = s
    return s


def rate_known_at(series: pd.Series, as_of) -> float:
    """Last observation on or before as_of: the rate an investor knew that day."""
    if series.empty:
        return float('nan')
    known = series[series.index <= pd.Timestamp(as_of)]
    return float(known.iloc[-1]) if not known.empty else float('nan')


def get_rf(currency: str = 'USD', as_of=None) -> tuple[float, str, bool]:
    """(rate, source label, is_local).  as_of=None gives today's rate."""
    ccy = (currency or 'USD').upper()
    use = 'USD' if ccy in USD_PEGGED else ccy
    series_id = RF_SERIES.get(use)
    if series_id is not None:
        s = _fred_series(series_id)
        r = rate_known_at(s, as_of) if as_of is not None else (float(s.iloc[-1]) if not s.empty else float('nan'))
        if np.isfinite(r):
            when = f" on {pd.Timestamp(as_of).date()}" if as_of is not None else ""
            label = f"FRED {series_id}{when}" + (" (USD peg)" if ccy in USD_PEGGED else "")
            return r, label, True
    if ccy != 'USD' and series_id is None and ccy not in USD_PEGGED:
        r, label, _ = get_rf('USD', as_of)
        return r, f"{label}; no {ccy} yield available, US yield used", False
    if as_of is None:
        try:
            import yfinance as yf
            return float(yf.Ticker('^TNX').fast_info.last_price) / 100, '^TNX (yfinance)', True
        except _FETCH_ERRORS:
            pass
    return _RF_FALLBACK, f'fallback {_RF_FALLBACK:.2%} (FRED unavailable)', ccy == 'USD'


def _synthetic_rating(coverage: float) -> tuple[str, float]:
    for lo, hi, rating, spread in _SYNTHETIC_RATING:
        if lo <= coverage < hi:
            return rating, spread
    return 'D2/D', 0.1413


def _annual_rf(currency: str) -> pd.Series:
    ccy = 'USD' if (currency or 'USD').upper() in USD_PEGGED else (currency or 'USD').upper()
    s = _fred_series(RF_SERIES.get(ccy, 'DGS10'))
    if s.empty:
        return s
    return s.groupby(s.index.year).mean()


def _reconstruct_wacc_history(raw, drivers, beta_adj, eq_w, dt_w, rf_annual, as_of=None):
    """WACC for each historical year with EBIT and interest expense, holding beta
    and capital-structure weights fixed; only the risk-free rate and the
    coverage-implied credit spread vary.  Only years before as_of are used."""
    common_idx = raw.ebit.index.intersection(raw.interest_expense.index)
    out = []
    for ts in common_idx:
        ebit_val = float(raw.ebit[ts])
        int_val = float(raw.interest_expense[ts])
        if ebit_val <= 0 or rf_annual.empty:
            continue
        yr = ts.year
        if as_of is not None and yr >= pd.Timestamp(as_of).year:
            continue
        if yr in rf_annual.index:
            rf_yr = float(rf_annual[yr])
        else:
            rf_yr = float(rf_annual[min(rf_annual.index, key=lambda y: abs(y - yr))])
        cov_yr = ebit_val / int_val if int_val > 0 else float('inf')
        _, spread_yr = _synthetic_rating(cov_yr)
        coe_yr = rf_yr + beta_adj * ERP
        cod_yr = (rf_yr + spread_yr) * (1.0 - drivers.tax_rate)
        out.append(eq_w * coe_yr + dt_w * cod_yr)
    return out


def compute_wacc(raw: RawData, drivers: Drivers, as_of=None) -> WACCResult:
    """as_of: optional date for point-in-time (backtest) use; the risk-free rate
    is then the one published on or before that date."""
    rf, rf_source, rf_local = get_rf(raw.currency, as_of)

    beta_adj = 0.67 * raw.beta + 0.33
    cost_of_equity = rf + beta_adj * ERP

    ebit_last = float(raw.ebit.iloc[-1]) if not raw.ebit.empty else 0.0
    int_exp_last = float(raw.interest_expense.iloc[-1]) if not raw.interest_expense.empty else 0.0
    coverage = ebit_last / int_exp_last if int_exp_last > 0 else float('inf')
    implied_rating, credit_spread = _synthetic_rating(coverage)
    cost_of_debt_pretax = rf + credit_spread
    cost_of_debt_aftertax = cost_of_debt_pretax * (1 - drivers.tax_rate)

    E = raw.market_cap_local
    D = raw.total_debt
    V = E + D
    eq_w = E / V if V > 0 else 1.0
    dt_w = D / V if V > 0 else 0.0
    wacc = eq_w * cost_of_equity + dt_w * cost_of_debt_aftertax

    hist = _reconstruct_wacc_history(raw, drivers, beta_adj, eq_w, dt_w, _annual_rf(raw.currency), as_of)
    if len(hist) >= 3:
        sigma_w, w_fallback = float(np.std(hist)), False
    else:
        sigma_w, w_fallback = _WACC_SIGMA_FALLBACK, True
    log.info("%s: WACC %.3f%% (rf %.3f%% %s), sigma %.3f%%%s", raw.ticker, wacc * 100, rf * 100,
             rf_source, sigma_w * 100, " [fallback]" if w_fallback else "")

    return WACCResult(
        rf=rf, rf_source=rf_source, beta_raw=raw.beta, beta_adj=beta_adj, erp=ERP,
        cost_of_equity=cost_of_equity, coverage_ratio=coverage, implied_rating=implied_rating,
        credit_spread=credit_spread, cost_of_debt_pretax=cost_of_debt_pretax,
        cost_of_debt_aftertax=cost_of_debt_aftertax, equity_value=E, debt_value=D,
        equity_weight=eq_w, debt_weight=dt_w, wacc=wacc, std_wacc=sigma_w,
        wacc_sigma_fallback=w_fallback, rf_is_local=rf_local,
    )
