"""Shared fixtures.  No test in this folder touches the network."""
from __future__ import annotations

import pandas as pd
import pytest

from tests.factories import make_assumptions
from valuation import wacc as wacc_mod


@pytest.fixture
def assumptions():
    return make_assumptions()


@pytest.fixture(autouse=True)
def offline_rates(monkeypatch):
    """Replace FRED with a fixed yield curve so nothing goes to the network."""
    idx = pd.date_range("2000-01-01", "2026-09-30", freq="MS")

    def fake(series_id: str) -> pd.Series:
        level = 0.010 if series_id == wacc_mod.RF_SERIES["JPY"] else 0.040
        return pd.Series(level, index=idx)

    monkeypatch.setattr(wacc_mod, "_fred_series", fake)
