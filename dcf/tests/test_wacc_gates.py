"""Risk-free rate lookup, WACC and output gates."""
from __future__ import annotations

import pandas as pd
import pytest

from tests.factories import make_raw
from valuation.drivers import compute_drivers
from valuation.gates import NotApplicable, check_not_financial, output_flags
from valuation.wacc import compute_wacc, get_rf, rate_known_at, _synthetic_rating


def test_rate_known_at_never_looks_ahead():
    s = pd.Series([0.01, 0.02, 0.03], index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]))
    assert rate_known_at(s, "2020-02-15") == 0.02
    assert rate_known_at(s, "2020-02-01") == 0.02
    assert pd.isna(rate_known_at(s, "2019-12-31"))


def test_currency_without_a_yield_falls_back_to_us_and_says_so():
    r, label, local = get_rf("TWD")
    assert r == pytest.approx(0.04) and not local and "TWD" in label


def test_dollar_peg_uses_us_yield():
    assert get_rf("HKD")[2] is True


def test_wacc_weights_and_cost_of_equity():
    raw = make_raw()
    w = compute_wacc(raw, compute_drivers(raw))
    assert w.equity_weight + w.debt_weight == pytest.approx(1.0)
    assert w.cost_of_equity == pytest.approx(0.04 + (0.67 * 1.0 + 0.33) * 0.045)
    assert w.wacc == pytest.approx(w.equity_weight * w.cost_of_equity + w.debt_weight * w.cost_of_debt_aftertax)


def test_synthetic_rating_edges():
    assert _synthetic_rating(20)[0] == "Aaa/AAA"
    assert _synthetic_rating(0.1)[0] == "D2/D"


@pytest.mark.parametrize("industry", ["Banks - Regional", "Insurance - Diversified", "Capital Markets"])
def test_financials_are_not_applicable(industry):
    with pytest.raises(NotApplicable):
        check_not_financial("X", "Financial Services", industry)


def test_non_financials_pass():
    check_not_financial("X", "Technology", "Software - Infrastructure")


def test_flags_far_from_price_and_negative_p10():
    flags = output_flags(100.0, -5.0, 10.0, 0.5, True, "USD", False)
    assert any("10x" not in f and "0.1x" in f for f in flags)
    assert any("negative" in f for f in flags)
    assert output_flags(100.0, 60.0, 110.0, 0.5, True, "USD", False) == []


def test_flag_when_wacc_close_to_terminal_growth():
    flags = output_flags(100.0, 60.0, 110.0, 0.5, True, "JPY", False, 0.040, 0.025)
    assert len(flags) == 1 and "less than 2 points" in flags[0]
