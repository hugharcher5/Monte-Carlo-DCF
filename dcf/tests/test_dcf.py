"""DCF arithmetic, timing, CapEx path and currency conversion."""
from __future__ import annotations

import numpy as np
import pytest

from tests.factories import make_assumptions
from valuation.dcf import (capex_path, detailed_value, discount_times, value, value_many,
                           G_CEILING_A, G_CEILING_B)


def test_one_year_value_matches_hand_calculation():
    a = make_assumptions(forecast_years=1, stub=1.0, mid_year=False)
    rev = 100e9 * 1.10
    fcff = rev * 0.25 * (1 - 0.21) + rev * 0.05 - rev * 0.06 - (rev - 100e9) * 0.05
    tv = fcff * 1.025 / (0.09 - 0.025)
    expected = ((fcff + tv) / 1.09 - 10e9) / 1e9
    assert value(a) == pytest.approx(expected, rel=1e-12)


def test_value_many_matches_value_for_each_draw():
    a = make_assumptions()
    growth = np.array([0.02, 0.10, 0.25])
    wacc = np.array([0.08, 0.09, 0.11])
    many = value_many(a, revenue_growth=growth, wacc=wacc)
    one_by_one = [value(make_assumptions(revenue_growth=g, wacc=w)) for g, w in zip(growth, wacc)]
    np.testing.assert_allclose(many, one_by_one, rtol=1e-12)


def test_detailed_value_agrees_with_value():
    a = make_assumptions(stub=0.4)
    assert detailed_value(a).value_per_share_usd == pytest.approx(value(a), rel=1e-12)


def test_invalid_draws_are_nan():
    a = make_assumptions()
    out = value_many(a, wacc=np.array([0.02, 0.09]), diluted_shares=np.array([1e9, 1e9]))
    assert np.isnan(out[0]) and np.isfinite(out[1])
    assert np.isnan(value_many(a, diluted_shares=np.array([0.0]))[0])


def test_value_raises_when_wacc_not_above_terminal_growth():
    with pytest.raises(ValueError):
        value(make_assumptions(wacc=0.02))


def test_discount_times_mid_year_and_stub():
    np.testing.assert_allclose(discount_times(1.0, 3, mid_year=False), [1, 2, 3])
    np.testing.assert_allclose(discount_times(1.0, 3), [0.5, 1.5, 2.5])
    # Four months into the year: 8 months of year 1 left, arriving after 4 months.
    np.testing.assert_allclose(discount_times(8 / 12, 3), [4 / 12, 8 / 12 + 0.5, 8 / 12 + 1.5])


def test_shorter_stub_counts_less_of_year_one_but_discounts_less():
    full = detailed_value(make_assumptions(stub=1.0))
    part = detailed_value(make_assumptions(stub=0.5))
    assert part.pv_explicit < full.pv_explicit     # half of year 1's cash flow is gone
    assert part.pv_tv > full.pv_tv                 # the terminal value is six months closer


def test_mid_year_convention_raises_value():
    assert value(make_assumptions(mid_year=True)) > value(make_assumptions(mid_year=False))


def test_capex_path_fades_net_reinvestment_toward_terminal_need():
    path = capex_path(0.10, 0.04, 0.20, 0.025, 5)[0]
    assert path[0] == pytest.approx(0.10)
    # Net reinvestment 6% scaled by 2.5% / 20% = 0.75%, so CapEx ends at 4.75%.
    assert path[-1] == pytest.approx(0.04 + 0.06 * 0.025 / 0.20)
    assert np.all(np.diff(path) <= 0)


def test_capex_path_unchanged_for_slow_growers_and_capex_below_da():
    np.testing.assert_allclose(capex_path(0.06, 0.04, 0.01, 0.025, 5)[0], 0.06)
    np.testing.assert_allclose(capex_path(0.03, 0.04, 0.20, 0.025, 5)[0], 0.03)


def test_size_ceiling_caps_growth_for_huge_companies():
    big = make_assumptions(start_revenue=1_000e9, revenue_growth=0.50, target_margin=0.25)
    r = detailed_value(big)
    ceiling = G_CEILING_A / (1_000.0 ** G_CEILING_B)
    assert r.forecast["Growth"].iloc[0] == pytest.approx(ceiling, rel=1e-9)
    assert r.ceiling_hits > 0


def test_foreign_currency_is_converted_once_at_spot():
    local = make_assumptions(currency="JPY", fx_rate=1.0)
    yen = make_assumptions(currency="JPY", fx_rate=0.0067)
    assert value(yen) == pytest.approx(value(local) * 0.0067, rel=1e-12)
    r = detailed_value(yen)
    assert r.value_per_share_usd == pytest.approx(r.value_per_share_local * 0.0067, rel=1e-12)
