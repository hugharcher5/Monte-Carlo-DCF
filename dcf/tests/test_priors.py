"""Shrinkage toward base rates, parameter uncertainty and terminal growth."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from valuation.priors import (correlation_matrix, forecast_inputs, load_base_rates, predict,
                              predict_sd, size_of, terminal_g_params, TERM_G_HALF)


def drivers(**kw):
    d = dict(revenue_growth=0.30, std_revenue_growth=0.10, ebit_margin=0.20,
             std_ebit_margin=0.03, capex_pct=0.05, std_capex_pct=0.01, years_used=4,
             n_margin_years=5, n_capex_years=5)
    d.update(kw)
    return SimpleNamespace(**d)


def test_size_is_log_revenue_in_billions():
    assert size_of(1e9, 1.0) == pytest.approx(0.0)
    assert size_of(100e9, 1.0) == pytest.approx(2.0)
    assert size_of(150e9, 0.0067) == pytest.approx(math.log10(150e9 * 0.0067) - 9)


def test_fast_growth_is_pulled_toward_the_base_rate():
    br = load_base_rates()
    g = predict(br["growth"], 0.30, 0.0)
    assert 0.0 < g < 0.30
    # slope below one means shrinkage
    assert br["growth"]["b"] < 1.0 and br["margin"]["b"] < 1.0


def test_uncertainty_falls_with_more_history():
    fit = load_base_rates()["growth"]
    assert predict_sd(fit, 0.10, 10, 0.0) < predict_sd(fit, 0.10, 2, 0.0)


def test_cross_source_disagreement_adds_in_quadrature():
    base = forecast_inputs(drivers(), 1.0, 0.22, False, {})
    wider = forecast_inputs(drivers(), 1.0, 0.22, False, {"revenue_growth": 0.05})
    assert wider.revenue_growth_sd == pytest.approx(math.hypot(base.revenue_growth_sd, 0.05))
    assert wider.revenue_growth == base.revenue_growth


def test_unproven_companies_get_wider_ranges_and_keep_industry_margin():
    p = forecast_inputs(drivers(ebit_margin=-0.05), 0.0, 0.13, False, {}, industry_target=False)
    u = forecast_inputs(drivers(ebit_margin=-0.05), 0.0, 0.13, True, {}, industry_target=True)
    assert u.revenue_growth_sd > p.revenue_growth_sd
    assert u.target_margin == 0.13


def test_terminal_growth_range_is_narrow_and_centred_on_mode():
    mode, lo, hi = terminal_g_params(0.043, 0.09)
    assert (mode, lo, hi) == pytest.approx((0.025, 0.015, 0.035))


def test_terminal_growth_does_not_depend_on_company_growth():
    # Nothing company-specific goes in, so a 40% grower gets the same range.
    assert terminal_g_params(0.043, 0.09) == terminal_g_params(0.043, 0.09)


def test_terminal_growth_follows_low_local_yields_and_stays_below_wacc():
    mode, lo, hi = terminal_g_params(0.010, 0.06)
    assert mode == pytest.approx(0.010) and lo == pytest.approx(0.010 - TERM_G_HALF)
    mode, lo, hi = terminal_g_params(0.043, 0.030)
    assert hi <= 0.030 - 0.010 + 1e-12 and mode <= hi


def test_correlation_matrix_is_valid():
    C = correlation_matrix()
    assert np.allclose(C, C.T) and np.allclose(np.diag(C), 1.0)
    assert np.all(np.linalg.eigvalsh(C) > 0)
    # terminal growth and WACC are not tied to the company inputs
    assert C[2, [0, 1, 4]].tolist() == [0, 0, 0] and C[3, [0, 1, 4]].tolist() == [0, 0, 0]
