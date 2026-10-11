"""Structured result object shared by terminal and Streamlit consumers."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .drivers   import Drivers
from .wacc      import WACCResult
from .dcf       import Assumptions, DCFResult


@dataclass
class ValuationResult:
    # ── Identity ──────────────────────────────────────────────────────────────
    ticker:            str
    currency:          str
    current_price_usd: float

    # ── Computation artefacts ─────────────────────────────────────────────────
    drivers:     Drivers
    wacc_result: WACCResult
    assumptions: Assumptions
    dcf:         DCFResult

    # ── Monte Carlo outputs (sims is an empty array when n_sims=0) ────────────
    sims:            np.ndarray   # full array of valid draws, shape (n_valid,)
    n_valid:         int
    copula_label:    str
    consistency_pass: bool
    cc_p50:          float        # zero-width run median

    # ── Summary statistics (NaN when no sims) ─────────────────────────────────
    p10:             float
    p25:             float
    p50:             float
    p75:             float
    p90:             float
    mean_val:        float
    std_val:         float
    pct_undervalued: float        # % of simulated scenarios above the current price

    # ── Cross-source uncertainty ──────────────────────────────────────────────
    sigma_cross: dict = field(default_factory=dict)
    recon_fields: dict = field(default_factory=dict)   # field → {source, disagree_pct, field_type}
    recon_sigma:  dict = field(default_factory=dict)   # variable → {sigma_hist, sigma_cross, sigma_eff}

    sw: float = 0.015             # WACC standard deviation used

    # ── Transparency payload (empty when n_sims=0) ────────────────────────────
    transparency: dict = field(default_factory=dict)

    # ── Applicability and warnings ────────────────────────────────────────────
    dcf_applicable: bool = True   # False when terminal FCFF <= 0 or base value <= 0
    flags: list = field(default_factory=list)          # plain-English warnings

    # ── Data provenance and company profile ──────────────────────────────────
    data_provenance: dict = field(default_factory=dict)
    unproven: bool = False
    target_margin_basis: str = ""

    # ── Forecast inputs, terminal growth and reverse DCF ──────────────────────
    forecast_inputs: object = None                     # priors.ForecastInputs
    terminal_g_range: tuple = (float("nan"),) * 3      # (mode, low, high)
    implied: dict = field(default_factory=dict)        # reverse DCF: param → value or None
    valuation_date: object = None
    fiscal_year_end: object = None
