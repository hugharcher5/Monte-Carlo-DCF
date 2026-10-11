"""Share-count choice: treasury-stock method and the mis-scaling rule."""
from __future__ import annotations

import pytest

from valuation.filings import _plausible, tsm_diluted_shares


def test_tsm_adds_only_in_the_money_options_net_of_buyback():
    # 100 options at $40 with the price at $50: exercise brings in $4,000, which
    # buys back 80 shares, so dilution is 20.
    count, dil = tsm_diluted_shares(1_000, 50.0, 100, 40.0, 0)
    assert dil == pytest.approx(20) and count == pytest.approx(1_020)


def test_tsm_ignores_out_of_the_money_options_and_adds_rsus():
    count, dil = tsm_diluted_shares(1_000, 30.0, 100, 40.0, 15)
    assert dil == pytest.approx(15) and count == pytest.approx(1_015)


def test_tsm_handles_missing_option_data():
    count, dil = tsm_diluted_shares(1_000, 50.0, float("nan"), float("nan"), float("nan"))
    assert (count, dil) == (1_000, 0.0)


@pytest.mark.parametrize("count, ok", [(712.3, False), (712e6, True), (9e9, False),
                                        (float("nan"), False), (-5.0, False)])
def test_counts_more_than_10x_from_yahoo_are_rejected(count, ok):
    assert _plausible(count, 712e6) is ok
