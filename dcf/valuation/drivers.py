"""Compute base-case DCF drivers from historical financials."""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from .data import RawData

log = logging.getLogger(__name__)

# ── Plausibility bands ────────────────────────────────────────────────────────
_BANDS = {
    'revenue_growth':  (-0.50,  0.80),
    'ebit_margin':     (-0.30,  0.60),
    'tax_rate':        ( 0.00,  0.60),
    'da_pct':          ( 0.00,  0.30),
    'capex_pct':       ( 0.00,  0.50),
    'nwc_pct':         (-0.30,  0.50),
    'sbc_pct':         ( 0.00,  0.30),
}


@dataclass
class Drivers:
    """Historical averages and year-to-year spreads.  These are the raw history;
    priors.forecast_inputs() turns them into forecasts and parameter uncertainty."""
    # Mean of available history
    revenue_growth: float
    ebit_margin: float
    tax_rate: float
    da_pct: float
    capex_pct: float
    nwc_pct: float
    sbc_pct: float

    # Best (peak) historical EBIT margin — anchor for target_margin
    best_ebit_margin: float

    # Year-to-year standard deviations of the annual figures
    std_revenue_growth: float
    std_ebit_margin: float
    std_target_margin: float
    std_fcf_pct: float         # std of historical (EBIT×(1−t) + D&A − CapEx) / Revenue

    years_used: int

    # Historical year-by-year DataFrame for display
    hist_df: pd.DataFrame = None

    # σ of historical CapEx / Revenue
    std_capex_pct: float = 0.0
    n_margin_years: int = 0      # years with an EBIT margin (one more than growth years)
    n_capex_years: int = 0
    flags: list = field(default_factory=list)   # plausibility warnings


def compute_drivers(raw: RawData) -> Drivers:
    # ── Align all income-stmt + cashflow series ───────────────────────────────
    series = {
        'revenue': raw.revenue,
        'ebit':    raw.ebit,
        'pretax':  raw.pretax_income,
        'tax':     raw.tax_provision,
        'da':      raw.da,
        'capex':   raw.capex,
        'sbc':     raw.sbc if not raw.sbc.empty
                   else pd.Series(0.0, index=raw.revenue.index),
    }
    df = pd.concat(series, axis=1).dropna(subset=['revenue', 'ebit']).sort_index()

    if raw.ebit.empty and not raw.revenue.empty:
        raise ValueError(
            f"{raw.ticker} reports no operating income (EBIT) line in its filings or on "
            "Yahoo Finance. This is usual for banks and insurers, whose cash flows a "
            "free-cash-flow-to-the-firm DCF cannot value."
        )
    if len(df) < 2:
        raise ValueError(
            f"Need ≥2 years of financial history for {raw.ticker}; "
            f"got {len(df)}. Check yfinance data availability."
        )

    # ── Year-over-year drivers ────────────────────────────────────────────────
    df['revenue_growth'] = df['revenue'].pct_change()
    df['ebit_margin']    = df['ebit']  / df['revenue']
    df['tax_rate']       = (df['tax']  / df['pretax']).clip(0.0, 0.99)
    df['da_pct']         = df['da']    / df['revenue']
    df['capex_pct']      = df['capex'] / df['revenue']
    df['sbc_pct']        = df['sbc']   / df['revenue']

    # NWC from most-recent balance sheet (single observation, applied uniformly)
    # cash_nwc is the cash figure from the same balance sheet as current assets
    # (filings.py sets it; raw.cash itself may come from a later 10-Q).
    cash_nwc   = getattr(raw, 'cash_nwc', float('nan'))
    if not np.isfinite(cash_nwc):
        cash_nwc = raw.cash
    nwc_latest = (raw.current_assets - cash_nwc) - (raw.current_liabilities - raw.current_debt)
    rev_latest  = float(df['revenue'].iloc[-1])
    nwc_pct = nwc_latest / rev_latest if rev_latest != 0 else 0.0

    # Drop first row (NaN growth from pct_change)
    df = df.dropna(subset=['revenue_growth'])
    years_used = len(df)

    rev_growth   = float(df['revenue_growth'].mean())
    ebit_margin  = float(df['ebit_margin'].mean())
    best_margin  = float(df['ebit_margin'].max())
    tax_rate     = float(df['tax_rate'].mean())
    da_pct       = float(df['da_pct'].mean())
    capex_pct    = float(df['capex_pct'].mean())
    sbc_pct      = float(df['sbc_pct'].mean())
    std_growth   = float(df['revenue_growth'].std())
    std_margin   = float(df['ebit_margin'].std())
    std_capex    = float(df['capex_pct'].std()) if df['capex_pct'].notna().sum() >= 2 else 0.0
    def _latest_ratio(series: pd.Series) -> float:
        """Latest year with both the item and revenue reported, as a share of revenue."""
        both = pd.concat([series.abs(), raw.revenue], axis=1).dropna()
        both = both[both.iloc[:, 1] != 0]
        return float(both.iloc[-1, 0] / both.iloc[-1, 1]) if len(both) else float('nan')

    # Yahoo sometimes drops a line for recent years (ABNB CapEx after 2022).
    # Use the latest year that has it; fail clearly only if it is missing everywhere.
    if not np.isfinite(da_pct):
        da_pct = _latest_ratio(raw.da)
    if not np.isfinite(capex_pct):
        capex_pct = _latest_ratio(raw.capex)
    for label, v in (("depreciation and amortisation", da_pct), ("capital expenditure", capex_pct)):
        if not np.isfinite(v):
            raise ValueError(
                f"{raw.ticker}: no {label} figure found in the cash flow statement, "
                "so free cash flow cannot be calculated."
            )

    df['fcf_pct'] = (df['ebit'] * (1.0 - df['tax_rate']) + df['da'] - df['capex']) / df['revenue']
    _fcf_clean    = df['fcf_pct'].dropna()
    std_fcf_pct   = float(_fcf_clean.std()) if len(_fcf_clean) >= 2 else 0.0

    drivers = Drivers(
        revenue_growth=rev_growth,
        ebit_margin=ebit_margin,
        best_ebit_margin=best_margin,
        tax_rate=max(0.0, min(tax_rate, 0.6)),
        da_pct=da_pct,
        capex_pct=capex_pct,
        nwc_pct=nwc_pct,
        sbc_pct=sbc_pct,
        std_revenue_growth=std_growth,
        std_ebit_margin=std_margin,
        std_target_margin=std_margin,
        std_fcf_pct=std_fcf_pct,
        years_used=years_used,
        hist_df=df,
        std_capex_pct=std_capex,
        n_margin_years=int(df['ebit_margin'].notna().sum()),
        n_capex_years=int(df['capex_pct'].notna().sum()),
    )

    # ── Plausibility checks ───────────────────────────────────────────────────
    flags = []
    for name, val in {
        'revenue_growth': rev_growth,
        'ebit_margin':    ebit_margin,
        'tax_rate':       tax_rate,
        'da_pct':         da_pct,
        'capex_pct':      capex_pct,
        'nwc_pct':        nwc_pct,
        'sbc_pct':        sbc_pct,
    }.items():
        lo, hi = _BANDS[name]
        if not (lo <= val <= hi):
            flags.append(f"  ⚠  {name} = {val:.2%}  outside expected [{lo:.0%}, {hi:.0%}]")

    for f in flags:
        log.info("%s: plausibility flag %s", raw.ticker, f.strip())
    drivers.flags = flags
    return drivers
