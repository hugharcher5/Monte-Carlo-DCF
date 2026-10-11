"""Checks that decide whether the DCF should be run at all, and flags on its output."""
from __future__ import annotations

import math


class NotApplicable(ValueError):
    """Raised when a free-cash-flow-to-the-firm DCF does not fit the business."""


# Yahoo industry labels for businesses whose debt is part of operations (deposits,
# policy reserves, trading liabilities).  For them "free cash flow to the firm"
# and "net debt" have no clean meaning, so the model refuses to value them.
FINANCIAL_INDUSTRY_WORDS = ("bank", "insurance", "capital markets", "mortgage finance",
                            "financial conglomerates", "shell companies")


def check_not_financial(ticker: str, sector: str, industry: str) -> None:
    ind = (industry or "").lower()
    if any(w in ind for w in FINANCIAL_INDUSTRY_WORDS):
        raise NotApplicable(
            f"{ticker} is a {industry} business ({sector}). Banks, insurers and brokers use "
            "debt and customer money as part of their operations, so free cash flow to the "
            "firm and net debt have no clean meaning for them. This model does not value "
            "them. A dividend-discount or excess-return model is the usual tool."
        )


FAR_FROM_PRICE: float = 5.0   # flag outputs more than 5x above or below the price
THIN_SPREAD: float = 0.02     # flag when WACC is less than 2 points above terminal growth


def output_flags(price: float, p10: float, p50: float, stub: float, rf_is_local: bool,
                 currency: str, unproven: bool, wacc: float = float("nan"),
                 terminal_g: float = float("nan")) -> list[str]:
    """Plain-English warnings attached to a finished valuation."""
    flags = []
    if math.isfinite(wacc) and math.isfinite(terminal_g) and wacc - terminal_g < THIN_SPREAD:
        flags.append(
            f"WACC ({wacc:.1%}) is less than 2 points above terminal growth ({terminal_g:.1%}). "
            "The terminal value is then very sensitive to small changes in either, so the "
            "range is much less reliable than usual. A low WACC usually comes from a low local "
            "bond yield or a large share of cheap debt (for example a car maker's finance arm).")
    if price > 0 and math.isfinite(p50) and p50 > 0:
        ratio = p50 / price
        if ratio > FAR_FROM_PRICE or ratio < 1.0 / FAR_FROM_PRICE:
            flags.append(
                f"The median value is {ratio:.1f}x the share price. A gap this large usually "
                "means the company's recent history (a loss year, a one-off margin, an "
                "investment cycle) is a poor guide to its future, so treat the number with "
                "suspicion.")
    elif math.isfinite(p50) and p50 <= 0:
        flags.append("The median value is zero or negative, which is not a meaningful share value.")
    if math.isfinite(p10) and p10 < 0:
        flags.append(
            f"At least 10% of simulated scenarios give a negative equity value (P10 is "
            f"${p10:,.2f}). Shares cannot be worth less than zero, so those scenarios mean the "
            "model's assumptions break down there, not that the stock could have negative value.")
    if stub <= 0.0:
        flags.append("The latest annual accounts are more than a year old, so the first forecast "
                     "year has already passed and is not counted.")
    if not rf_is_local:
        flags.append(f"No {currency} government bond yield was available, so the US 10-year "
                     "yield was used. This overstates the discount rate for low-inflation "
                     "currencies and understates it for high-inflation ones.")
    if unproven:
        flags.append("Unproven company: no profitable history, so the mature margin is the "
                     "industry average and every input has been given wider ranges.")
    return flags
