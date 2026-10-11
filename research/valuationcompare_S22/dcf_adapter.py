"""
Point-in-time inputs for S22's DCF signals.

build_pit_rawdata() turns SEC filings published on or before the rebalance
date into the same RawData object the live tool gets from Yahoo, so the live
engine (dcf/valuation) values it unchanged.

pit_dcf_signals() runs the current model on it:
    mos_base  (base-case value - price) / price
    mos_p50   (Monte Carlo P50 - price) / price
    p_above   share of simulated scenarios above the price
The Monte Carlo uses 1,000 scenarios per stock (enough to rank on), the
risk-free rate published on or before the date, and base rates fitted only on
outcomes known by the end of 2014 (dcf/data/base_rates_pre2015.json), so
nothing from inside the 2015-2020 test window leaks in.

pit_dcf_margin_of_safety() keeps the original July 2026 call signature for
run.py and run_intersection.py; it now returns the current model's base case.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pit_facts as PF

_DCF_DIR = Path(__file__).resolve().parents[2] / "dcf"
sys.path.insert(0, str(_DCF_DIR))
from valuation.data import RawData          # noqa: E402
from valuation.gates import NotApplicable   # noqa: E402
from valuation.montecarlo import run_valuation  # noqa: E402
from valuation.priors import BASE_RATES_PRE2015, load_base_rates  # noqa: E402

N_SIMS = 1_000
# SIC codes for banks, insurers, brokers and financial holding companies.  The
# live tool refuses these (valuation/gates.py), so the backtest does too.
FINANCIAL_SIC = range(6000, 6800)


def build_pit_rawdata(ticker: str, cik: int, as_of: pd.Timestamp,
                       price: float, shares: float, beta: float) -> RawData | None:
    as_of_str = as_of.strftime("%Y-%m-%d")
    facts = PF.get_facts(cik)
    if facts is None:
        return None

    revenue = PF.annual_series(facts, PF.REVENUE_CONCEPTS, as_of_str)
    ebit = PF.annual_series(facts, PF.EBIT_CONCEPTS, as_of_str)
    interest_expense = PF.annual_series(facts, PF.INTEREST_EXPENSE_CONCEPTS, as_of_str, abs_val=True)
    pretax_income = PF.annual_series(facts, PF.PRETAX_CONCEPTS, as_of_str)
    tax_provision = PF.annual_series(facts, PF.TAX_CONCEPTS, as_of_str, abs_val=True)
    sbc = PF.annual_series(facts, PF.SBC_CONCEPTS, as_of_str, abs_val=True)
    da = PF.da_series(facts, as_of_str)
    capex = PF.annual_series(facts, PF.CAPEX_CONCEPTS, as_of_str, abs_val=True)

    if revenue.empty or ebit.empty or len(revenue) < 2 or len(ebit) < 2:
        return None   # compute_drivers() requires >=2 years of revenue+ebit history

    cash = PF.scalar_pit(facts, PF.CASH_CONCEPTS, as_of_str)
    total_debt = PF.total_debt_pit(facts, as_of_str)
    current_assets = PF.scalar_pit(facts, PF.CURRENT_ASSETS_CONCEPTS, as_of_str)
    current_liabilities = PF.scalar_pit(facts, PF.CURRENT_LIAB_CONCEPTS, as_of_str)
    current_debt = PF.scalar_pit(facts, PF.STD_CONCEPTS, as_of_str)

    # Balance-sheet scalars: 0.0 fallback when a smaller filer doesn't tag the
    # concept at all (matches fetch_edgar()'s missing-field tolerance -- a
    # missing current_debt/current_liabilities is far more common than a
    # genuinely-zero one for small caps, but treating as 0 is the same
    # convention this codebase already uses rather than dropping the name).
    cash = 0.0 if np.isnan(cash) else cash
    total_debt = 0.0 if np.isnan(total_debt) else total_debt
    current_assets = 0.0 if np.isnan(current_assets) else current_assets
    current_liabilities = 0.0 if np.isnan(current_liabilities) else current_liabilities
    current_debt = 0.0 if np.isnan(current_debt) else current_debt

    if np.isnan(price) or price <= 0 or np.isnan(shares) or shares <= 0:
        return None
    market_cap = price * shares

    return RawData(
        ticker=ticker, currency="USD",
        current_price_usd=float(price),
        market_cap_usd=float(market_cap),
        market_cap_local=float(market_cap),
        fx_rate=1.0,
        beta=float(beta) if not np.isnan(beta) else 1.0,
        diluted_shares=float(shares),
        revenue=revenue, ebit=ebit,
        interest_expense=interest_expense,
        pretax_income=pretax_income,
        tax_provision=tax_provision,
        sbc=sbc, da=da, capex=capex,
        cash=float(cash), total_debt=float(total_debt),
        current_assets=float(current_assets),
        current_liabilities=float(current_liabilities),
        current_debt=float(current_debt),
        op_lease_liab=0.0, price_vol=0.0,
        missing_fields=[],
    )


def is_financial(sic) -> bool:
    try:
        return int(float(sic)) in FINANCIAL_SIC
    except (TypeError, ValueError):
        return False


def pit_dcf_signals(ticker: str, cik: int, as_of: pd.Timestamp, price: float, shares: float,
                    beta: float, sic=None, n_sims: int = N_SIMS) -> dict | None:
    """The three DCF signals, or None when the model does not apply (financials,
    missing inputs, non-positive terminal cash flow or base-case value)."""
    if is_financial(sic):
        return None
    raw = build_pit_rawdata(ticker, cik, as_of, price, shares, beta)
    if raw is None:
        return None
    raw.valuation_date = as_of
    try:
        r = run_valuation(ticker, n_sims=n_sims, raw=raw, as_of=as_of,
                          base_rates=load_base_rates(BASE_RATES_PRE2015), light=True)
    except (NotApplicable, ValueError, ZeroDivisionError, FloatingPointError, KeyError, IndexError):
        return None
    if not r.dcf_applicable:
        return None
    if n_sims <= 0:
        return {"mos_base": (r.dcf.value_per_share_usd - price) / price}
    if not np.isfinite(r.p50):
        return None
    return {"mos_base": (r.dcf.value_per_share_usd - price) / price,
            "mos_p50": (r.p50 - price) / price,
            "p_above": r.pct_undervalued / 100.0}


def pit_dcf_margin_of_safety(ticker: str, cik: int, as_of: pd.Timestamp,
                              price: float, shares: float, beta: float) -> float | None:
    """Base-case margin of safety from the current model (no simulation)."""
    s = pit_dcf_signals(ticker, cik, as_of, price, shares, beta, n_sims=0)
    return None if s is None else s["mos_base"]
