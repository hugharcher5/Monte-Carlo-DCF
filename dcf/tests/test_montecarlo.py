"""Monte Carlo: zero-width check, sampling, end-to-end run on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.factories import make_assumptions, make_raw
from valuation.gates import NotApplicable
from valuation.montecarlo import (pert_bounds, pert_ppf, run_valuation, simulate,
                                  simulation_bounds, stub_fraction, _ensure_psd)
from valuation.priors import correlation_matrix
from valuation.dcf import value


def fi(**kw):
    from types import SimpleNamespace
    d = dict(revenue_growth_sd=0.05, margin_sd=0.03, capex_sd=0.01)
    d.update(kw)
    return SimpleNamespace(**d)


def test_pert_bounds_match_requested_sd():
    lo, hi = pert_bounds(0.10, 0.02)
    u = (np.arange(200_000) + 0.5) / 200_000
    draws = pert_ppf(u, lo, 0.10, hi)
    assert draws.std() == pytest.approx(0.02, rel=0.01)
    assert draws.mean() == pytest.approx(0.10, abs=1e-4)


def test_pert_point_mass_when_range_is_zero():
    assert np.all(pert_ppf(np.array([0.1, 0.9]), 0.05, 0.05, 0.05) == 0.05)


def test_zero_width_run_reproduces_base_case():
    base = make_assumptions()
    b = simulation_bounds(base, fi(), (0.025, 0.015, 0.035), 0.01, zero_width=True)
    vals, _, _ = simulate(base, b, correlation_matrix(), 100, np.random.default_rng(0))
    np.testing.assert_allclose(vals, value(base), rtol=1e-12)


def test_more_input_uncertainty_widens_the_band():
    base = make_assumptions()
    tg = (0.025, 0.015, 0.035)
    C = correlation_matrix()
    narrow, _, _ = simulate(base, simulation_bounds(base, fi(), tg, 0.01), C, 20_000,
                            np.random.default_rng(1))
    wide, _, _ = simulate(base, simulation_bounds(base, fi(revenue_growth_sd=0.15), tg, 0.01), C,
                          20_000, np.random.default_rng(1))
    w = lambda s: np.subtract(*np.percentile(s, [90, 10]))
    assert w(wide) > w(narrow)


def test_terminal_growth_draws_stay_below_wacc():
    base = make_assumptions(wacc=0.05)
    b = simulation_bounds(base, fi(), (0.025, 0.015, 0.035), 0.02)
    from valuation.montecarlo import draw_inputs
    x = draw_inputs(base, b, correlation_matrix(), 5_000, np.random.default_rng(2))
    assert np.all(x["terminal_g"] <= x["wacc"] - 0.01 + 1e-12)


def test_ensure_psd_repairs_a_bad_matrix():
    bad = np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1.0]])
    good = _ensure_psd(bad)
    assert np.all(np.linalg.eigvalsh(good) > 0) and np.allclose(np.diag(good), 1)


def test_stub_fraction():
    assert stub_fraction(pd.Timestamp("2025-12-31"), pd.Timestamp("2026-06-30")) == pytest.approx(0.504, abs=0.002)
    assert stub_fraction(pd.Timestamp("2024-01-31"), pd.Timestamp("2026-06-30")) == 0.0
    assert stub_fraction(None, pd.Timestamp("2026-06-30")) == 1.0


def test_end_to_end_valuation_on_synthetic_company():
    r = run_valuation("TEST", 2_000, raw=make_raw())
    assert r.consistency_pass
    assert r.p10 < r.p50 < r.p90
    assert 0 <= r.pct_undervalued <= 100
    assert r.n_valid == 2_000
    assert 0 < r.assumptions.stub < 1
    # reverse DCF: plugging the implied growth back in gives the price
    g = r.implied["revenue_growth"]
    import copy
    a = copy.copy(r.assumptions)
    a.revenue_growth = g
    assert value(a) == pytest.approx(r.current_price_usd, rel=1e-4)


def test_same_seed_gives_same_answer():
    a = run_valuation("TEST", 1_000, raw=make_raw(), light=True)
    b = run_valuation("TEST", 1_000, raw=make_raw(), light=True)
    assert a.p50 == b.p50


def test_banks_are_refused():
    with pytest.raises(NotApplicable):
        run_valuation("BANK", 100, raw=make_raw(industry="Banks - Diversified", sector="Financial Services"))


def test_foreign_company_uses_local_rate():
    r = run_valuation("TM", 500, raw=make_raw(currency="JPY", fx_rate=0.0067, price=0.0067 * 3000),
                      light=True)
    assert r.wacc_result.rf == pytest.approx(0.010)
    assert r.wacc_result.rf_is_local


def test_correlations_change_the_range_only_modestly():
    """Shown in the app's methodology: the estimated correlations widen the
    P10-P90 range a little and barely move the median."""
    base = make_assumptions(revenue_growth=0.10)
    b = simulation_bounds(base, fi(revenue_growth_sd=0.08, margin_sd=0.04), (0.025, 0.015, 0.035), 0.012)
    with_c, _, _ = simulate(base, b, correlation_matrix(), 20_000, np.random.default_rng(3))
    without, _, _ = simulate(base, b, np.eye(5), 20_000, np.random.default_rng(3))
    w = lambda s: np.subtract(*np.percentile(s, [90, 10]))
    assert 1.0 < w(with_c) / w(without) < 1.15
    assert abs(np.median(with_c) / np.median(without) - 1) < 0.02
