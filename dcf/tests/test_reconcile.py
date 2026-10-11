"""Cross-source reconciliation, using SBUX- and MCD-like synthetic sources."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from valuation.reconcile import plausible_levels, reconcile, robust_spread, FIELD_TYPE
from valuation.sources import SourceData

IDX = pd.to_datetime(["2023-09-30", "2024-09-30", "2025-09-30"])


def src(name, revenue=(36e9, 36.2e9, 37.2e9), ebit_m=0.10, shares=1.14e9, debt=13.3e9,
        cash=3.0e9, capex_m=0.07):
    rev = pd.Series(revenue, index=IDX)
    return SourceData(source_name=name, currency="USD", revenue=rev, ebit=rev * ebit_m,
                      dep_amort=rev * 0.05, capex=rev * capex_m, diluted_shares=shares,
                      total_debt=debt, cash=cash, tax_rate=0.24)


def test_robust_spread_ignores_one_wild_value():
    assert robust_spread([1.0, 1.0, 1.0]) == 0.0
    assert robust_spread([1.0, 1.01, 50.0]) == pytest.approx(1.4826 * 0.01)
    assert robust_spread([1.0]) == 0.0
    assert robust_spread([1.0, float("nan")]) == 0.0


def test_lease_inclusive_debt_does_not_widen_net_debt():
    # SBUX-like: one source counts $10.5bn of leases as debt, the others do not.
    sources = [src("Yahoo", debt=16.1e9), src("FMP", debt=26.6e9), src("EDGAR", debt=13.3e9)]
    r = reconcile(sources)
    assert FIELD_TYPE["total_debt"] == "DEFINITIONAL"
    assert r.sigma_cross["net_debt"] == pytest.approx(0.0)


def test_cash_disagreement_still_moves_net_debt():
    sources = [src("Yahoo", cash=3.0e9), src("EDGAR", cash=4.0e9)]
    assert reconcile(sources).sigma_cross["net_debt"] > 0


def test_mis_scaled_share_count_is_dropped():
    # MCD-like: EDGAR's weighted-average count is tagged in millions (712.3).
    sources = [src("Yahoo", shares=712.0e6), src("FMP", shares=715.0e6), src("EDGAR", shares=712.3)]
    assert plausible_levels(sources, "diluted_shares") == [712.0e6, 715.0e6]
    sd = reconcile(sources).sigma_cross["diluted_shares"]
    assert sd < 0.01 * 712e6


def test_missing_share_count_is_dropped():
    sources = [src("Yahoo"), src("EDGAR", shares=float("nan"))]
    assert "diluted_shares" not in reconcile(sources).sigma_cross


def test_definitional_fields_never_enter_sigma_cross():
    sources = [src("Yahoo"), src("EDGAR")]
    sources[1].dep_amort = sources[1].dep_amort * 1.5
    sc = reconcile(sources).sigma_cross
    assert not {"dep_amort", "total_debt", "tax_rate"} & set(sc)


def test_preferred_source_is_edgar_with_cash_flow_da():
    y, e = src("Yahoo"), src("EDGAR")
    e.dep_amort = e.dep_amort * 0.8
    r = reconcile([y, e])
    assert r.preferred.source_name == "EDGAR"
    assert r.field_sources["dep_amort"] == "Yahoo"
    assert r.preferred.dep_amort.iloc[-1] == pytest.approx(y.dep_amort.iloc[-1])


def test_growth_disagreement_measured_on_latest_year():
    a = src("Yahoo", revenue=(36e9, 36e9, 39.6e9))   # +10%
    b = src("EDGAR", revenue=(36e9, 36e9, 37.8e9))   # +5%
    sc = reconcile([a, b])
    assert sc.sigma_cross["revenue_growth"] == pytest.approx(1.4826 * 0.025)
    assert math.isfinite(sc.disagreement["revenue"])
