"""Synthetic companies for the tests.  Nothing here touches the network."""
from __future__ import annotations

import numpy as np
import pandas as pd
from valuation.data import RawData
from valuation.dcf import Assumptions


def make_assumptions(**overrides) -> Assumptions:
    """A plain USD company: $100bn revenue, 25% margin, 9% WACC."""
    d = dict(
        ticker="TEST", currency="USD", start_revenue=100e9, revenue_growth=0.10,
        ebit_margin=0.25, target_margin=0.25, tax_rate=0.21, da_pct=0.05, capex_pct=0.06,
        nwc_pct=0.05, terminal_g=0.025, wacc=0.09, net_debt=10e9, diluted_shares=1e9,
        fx_rate=1.0, forecast_years=5, rf=0.04, erp=0.045, beta_adj=1.0, cost_of_equity=0.085,
        cost_of_debt_pretax=0.05, cost_of_debt_aftertax=0.04, equity_weight=0.85,
        debt_weight=0.15, implied_rating="A", current_price_usd=150.0, current_ebitda=30e9,
        market_ev_local=160e9,
    )
    d.update(overrides)
    return Assumptions(**d)


def make_raw(ticker="TEST", currency="USD", fx_rate=1.0, years=5, growth=0.08,
             margin=0.20, price=50.0, shares=1e9, **overrides) -> RawData:
    """Five years of steady history, fiscal years ending 31 December."""
    idx = pd.to_datetime([f"{2025 - years + 1 + i}-12-31" for i in range(years)])
    rev = pd.Series(10e9 * (1 + growth) ** np.arange(years), index=idx)
    d = dict(
        ticker=ticker, currency=currency, current_price_usd=price,
        market_cap_usd=price * shares, market_cap_local=price * shares / fx_rate,
        fx_rate=fx_rate, beta=1.0, diluted_shares=shares,
        revenue=rev, ebit=rev * margin, interest_expense=rev * 0.005,
        pretax_income=rev * (margin - 0.005), tax_provision=rev * (margin - 0.005) * 0.21,
        sbc=rev * 0.01, da=rev * 0.04, capex=rev * 0.05,
        cash=2e9, total_debt=4e9, current_assets=5e9, current_liabilities=4e9, current_debt=0.5e9,
        sector="Industrials", industry="Specialty Industrial Machinery", country="United States",
        valuation_date=pd.Timestamp("2026-06-30"),
    )
    d.update(overrides)
    return RawData(**d)
