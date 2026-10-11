"""Reverse DCF: which single input value makes the base-case DCF equal the price?

Everything else is held at the base case.  The headline is the starting
revenue growth the market price implies, which can be compared directly with
the company's own history and the model's forecast.
"""
from __future__ import annotations

import copy
import math

from scipy.optimize import brentq

from .dcf import Assumptions, value

# Search ranges (decimal).
GROWTH_RANGE = (-0.50, 2.00)
MARGIN_RANGE = (-0.50, 0.95)
WACC_MAX = 0.60


def solve(base: Assumptions, target: float, param: str, lo: float, hi: float) -> float | None:
    """Value of `param` in [lo, hi] at which value(base) == target, or None
    when no value in the range reaches the target."""
    def f(x: float) -> float:
        a = copy.copy(base)
        setattr(a, param, x)
        if param == "target_margin":
            a.ebit_margin = base.ebit_margin   # only the mature margin moves
        try:
            return value(a) - target
        except ValueError:
            return float("nan")

    f_lo, f_hi = f(lo), f(hi)
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)) or f_lo * f_hi > 0:
        return None
    return float(brentq(f, lo, hi, xtol=1e-5, maxiter=200))


def implied_inputs(base: Assumptions, price_usd: float) -> dict[str, float | None]:
    """Implied starting growth, mature margin and WACC (each solved alone)."""
    if not price_usd > 0:
        return {}
    return {
        "revenue_growth": solve(base, price_usd, "revenue_growth", *GROWTH_RANGE),
        "target_margin": solve(base, price_usd, "target_margin", *MARGIN_RANGE),
        "wacc": solve(base, price_usd, "wacc", base.terminal_g + 0.002, WACC_MAX),
    }
