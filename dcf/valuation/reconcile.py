"""
Reconcile data from multiple SourceData objects into a single preferred estimate.

Field disagreement = (max − min) / |median| across sources with finite values.
NaN when fewer than two sources have data.

Fields are classified as DATA or DEFINITIONAL:
  DATA        — genuine uncertainty between independent sources; σ_cross drives MC widening
  DEFINITIONAL — difference is a reporting convention (for example whether leases count
                 as debt); it never enters σ_cross

σ_cross is a robust spread: 1.4826 × the median absolute deviation from the
median across sources (equal to the standard deviation for normal data, but one
wild source cannot dominate it).  Before it is computed, a source's value is
dropped when it is missing or more than 10x away from Yahoo's (the same rule
filings.py uses to spot share counts tagged in millions).
"""

from __future__ import annotations
import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .sources import SourceData


# Fields that are annual series (we compare on the last available value)
SERIES_FIELDS: list[str] = ["revenue", "ebit", "dep_amort", "capex"]
# Fields that are scalar point estimates
SCALAR_FIELDS: list[str] = ["diluted_shares", "total_debt", "cash", "tax_rate"]
ALL_FIELDS: list[str]    = SERIES_FIELDS + SCALAR_FIELDS

# DATA  → genuine cross-source uncertainty → feeds σ_cross into Monte Carlo
# DEFINITIONAL → convention-resolved → disagreement is expected and suppressed in σ_cross
FIELD_TYPE: dict[str, str] = {
    "revenue":        "DATA",
    "ebit":           "DATA",
    "capex":          "DATA",
    "cash":           "DATA",
    "diluted_shares": "DATA",
    # D&A: EDGAR reconstructs from per-tag components and misses untagged amortisation;
    #      the cash-flow-statement total (Yahoo / FMP) is authoritative — see override below.
    "dep_amort":  "DEFINITIONAL",
    # Debt: standardised to bonds + finance leases by sources.py; any residual diff is rounding.
    "total_debt": "DEFINITIONAL",
    # Tax rate: different averaging windows and line-item conventions across data providers.
    "tax_rate":   "DEFINITIONAL",
}


@dataclass
class ReconcileResult:
    preferred:    SourceData             # normalised point-estimate source for DCF
    disagreement: dict[str, float]       # field → relative spread (NaN = can't compute)
    sources:      list[SourceData]       # all sources for reference
    # σ_cross for DATA MC-driver variables (std of per-source derived estimate).
    # Only DATA fields appear here; DEFINITIONAL fields are absent.
    sigma_cross:  dict[str, float] = field(default_factory=dict)
    # Which source won for each field. dep_amort may differ from preferred.source_name
    # due to the D&A override (EDGAR preferred, but cash-flow total from FMP/Yahoo used).
    field_sources: dict[str, str] = field(default_factory=dict)


def _relative_spread(values: list[float]) -> float:
    """(max − min) / |median|.  Returns NaN when < 2 finite values or median ≈ 0."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if len(finite) < 2:
        return float("nan")
    med = abs(float(np.median(finite)))
    if med < 1e-9:
        return float("nan")
    return (max(finite) - min(finite)) / med


MAD_SCALE: float = 1.4826
MAX_RATIO_VS_YAHOO: float = 10.0


def robust_spread(values: list[float]) -> float:
    """1.4826 × median absolute deviation of the finite values (0 when < 2)."""
    v = np.array([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    if len(v) < 2:
        return 0.0
    return float(MAD_SCALE * np.median(np.abs(v - np.median(v))))


def plausible_levels(sources: list[SourceData], field: str) -> list[float]:
    """Finite, positive values of a level field, dropping any more than 10x away
    from Yahoo's (a scaling error such as a share count tagged in millions)."""
    vals = {s.source_name: getattr(s, field) for s in sources}
    ref = vals.get("Yahoo")
    out = []
    for name, v in vals.items():
        if v is None or not math.isfinite(v) or v <= 0:
            continue
        if (ref is not None and math.isfinite(ref) and ref > 0 and name != "Yahoo"
                and not 1.0 / MAX_RATIO_VS_YAHOO < v / ref < MAX_RATIO_VS_YAHOO):
            continue
        out.append(float(v))
    return out


def _last_val(s: pd.Series) -> float:
    """Most recent (last index) value, or NaN if empty."""
    return float(s.iloc[-1]) if not s.empty else float("nan")


def reconcile(sources: list[SourceData]) -> ReconcileResult:
    """
    Compare all sources field by field and return a normalised preferred estimate.

    Preferred hierarchy: EDGAR > FMP > Yahoo for all fields except dep_amort.
    D&A override: dep_amort on the preferred source is replaced with the
    Yahoo / FMP cash-flow-statement total, which captures untagged amortisation
    that EDGAR's per-tag reconstruction misses.

    σ_cross is computed only for DATA-classified fields and expressed as the
    standard deviation of the per-source derived estimate (growth rate or level).
    """
    if not sources:
        raise ValueError("reconcile() requires at least one source")

    # ── Field-by-field disagreement ───────────────────────────────────────────
    disagreement: dict[str, float] = {}

    for f in SERIES_FIELDS:
        vals = [_last_val(getattr(s, f)) for s in sources]
        disagreement[f] = _relative_spread(vals)

    for f in SCALAR_FIELDS:
        vals = [getattr(s, f) for s in sources]
        disagreement[f] = _relative_spread(vals)

    # ── Preferred source: EDGAR > FMP > Yahoo ─────────────────────────────────
    _PREF_ORDER = {"EDGAR": 0, "FMP": 1, "Yahoo": 2}
    preferred = min(sources, key=lambda s: _PREF_ORDER.get(s.source_name, 99))

    # ── D&A override ─────────────────────────────────────────────────────────
    # For dep_amort, use the cash-flow-statement total from Yahoo or FMP instead
    # of EDGAR's tag reconstruction, which omits untagged amortisation items.
    # This override only applies when EDGAR is preferred; if Yahoo/FMP is already
    # preferred, no change is needed.
    da_source_name = preferred.source_name  # default: same as preferred
    if preferred.source_name == "EDGAR":
        cf_source = next(
            (s for name in ("FMP", "Yahoo") for s in sources if s.source_name == name),
            None,
        )
        if cf_source is not None and not cf_source.dep_amort.empty:
            preferred = dataclasses.replace(preferred, dep_amort=cf_source.dep_amort)
            da_source_name = cf_source.source_name

    # ── field_sources: track winning source per field ─────────────────────────
    field_sources: dict[str, str] = {f: preferred.source_name for f in ALL_FIELDS}
    field_sources["dep_amort"] = da_source_name  # may differ when D&A override applied

    # ── σ_cross for DATA fields ───────────────────────────────────────────────
    # Only DATA-classified fields produce σ_cross entries; DEFINITIONAL fields
    # are expected to disagree (by convention) and must not widen MC distributions.
    sigma_cross: dict[str, float] = {}

    def _ratio_last(num: str, s: SourceData, growth: bool = False) -> float:
        rev = s.revenue
        if growth:
            if len(rev) < 2 or abs(float(rev.iloc[-2])) < 1e-9:
                return float("nan")
            return float(rev.iloc[-1]) / float(rev.iloc[-2]) - 1.0
        ser = getattr(s, num)
        if ser.empty or rev.empty or abs(float(rev.iloc[-1])) < 1e-9:
            return float("nan")
        return float(ser.iloc[-1]) / float(rev.iloc[-1])

    # Revenue growth, EBIT margin and CapEx intensity: the latest-year ratio from
    # each source (the same form the model uses), then the robust spread.
    for key, num, growth in (("revenue_growth", "revenue", True), ("ebit_margin", "ebit", False),
                             ("capex_pct", "capex", False)):
        vals = [_ratio_last(num, s, growth) for s in sources]
        if sum(math.isfinite(v) for v in vals) >= 2:
            sigma_cross[key] = robust_spread(vals)

    # Share count: drop missing or mis-scaled counts first.
    shares_vals = plausible_levels(sources, "diluted_shares")
    if len(shares_vals) >= 2:
        sigma_cross["diluted_shares"] = robust_spread(shares_vals)

    # Net debt: debt is DEFINITIONAL (sources disagree mainly on whether leases
    # count), so every source's cash is set against the preferred source's debt.
    # Only genuine disagreement about cash moves net debt.
    pref_debt = preferred.total_debt
    if math.isfinite(pref_debt):
        cash_vals = [s.cash for s in sources if math.isfinite(s.cash) and s.cash >= 0]
        if len(cash_vals) >= 2:
            sigma_cross["net_debt"] = robust_spread([pref_debt - c for c in cash_vals])

    return ReconcileResult(
        preferred=preferred,
        disagreement=disagreement,
        sources=sources,
        sigma_cross=sigma_cross,
        field_sources=field_sources,
    )
