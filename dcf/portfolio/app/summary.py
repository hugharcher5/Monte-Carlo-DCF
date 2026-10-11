"""Headline numbers, the reverse DCF and the plain-English reading of the result."""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from . import charts, details, methodology
from .common import pct, usd


def _reverse_dcf(r) -> None:
    """What the price implies, shown near the top because it is often more useful
    than the value itself."""
    a, d, imp = r.assumptions, r.drivers, r.implied or {}
    g, tm, w = imp.get("revenue_growth"), imp.get("target_margin"), imp.get("wacc")
    head = (f"**What the price of {usd(r.current_price_usd)} implies.** Changing one input at a time "
            "and holding the rest at the base case, the price needs ")
    options = []
    if g is not None:
        options.append(f"starting revenue growth of **{g:.1%}** a year, fading to {a.terminal_g:.1%} over "
                       f"{a.forecast_years} years (the model forecasts {a.revenue_growth:.1%}; the company "
                       f"averaged {d.revenue_growth:.1%} over the last {d.years_used} years)")
    if tm is not None:
        options.append(f"a mature operating margin of **{tm:.1%}** (model: {a.target_margin:.1%})")
    if w is not None:
        options.append(f"a WACC of **{w:.1%}** (model: {a.wacc:.1%})")
    text = head + ", or ".join(options) + "." if options else ""
    if g is None:
        text += (" No starting growth rate between -50% and +200% a year reaches the price on its own, "
                 "partly because the size limit caps growth for very large companies.")
    if not options:
        text = (f"**What the price of {usd(r.current_price_usd)} implies.** No single change to growth, "
                "mature margin or WACC within sensible limits gets the base case to the price.")
    st.markdown(text.strip())


def _reading(r) -> None:
    price, p10, p50, p90 = r.current_price_usd, r.p10, r.p50, r.p90
    gap = (p50 - price) / price if price > 0 else float("nan")
    if abs(gap) < 0.10:
        s1 = f"The median value of {usd(p50)} is within 10% of the price, so the model sees the stock as roughly fair."
    else:
        s1 = (f"The median value of {usd(p50)} is {abs(gap):.0%} {'above' if gap > 0 else 'below'} "
              f"the price of {usd(price)}.")
    if p10 <= price <= p90:
        s2 = (f"The price is inside the P10 to P90 range ({usd(p10, 0)} to {usd(p90, 0)}), so it fits "
              "the model's input ranges without unusual assumptions.")
    elif price > p90:
        s2 = (f"The price is above P90 ({usd(p90, 0)}): fewer than 10% of simulated scenarios reach it. "
              "The market is expecting more growth or profit than the company's history, pulled "
              "toward averages, supports.")
    else:
        s2 = (f"The price is below P10 ({usd(p10, 0)}): more than 90% of simulated scenarios are above "
              "it. The market is pricing in a risk or decline the history does not show.")
    st.info(f"{s1} {s2}\n\n"
            f"**{r.pct_undervalued:.1f}%** of the {r.n_valid:,} simulated scenarios give a value above "
            "today's price. That is a summary of the model's input ranges, not a forecast of returns.")


def _not_applicable(r, info: dict) -> None:
    st.error(
        "**DCF not applicable.** This company does not produce positive free cash flow by the end of "
        "the forecast, or the base case gives a negative equity value. A DCF cannot value a business "
        "that does not generate cash: its value depends on whether it gets there, how much new money "
        "it needs, and outcomes a single forecast cannot capture.")
    labels = {
        "terminal_fcff_non_positive": "Free cash flow in the final forecast year is zero or negative.",
        "current_ebit_negative": "Operating profit is negative today.",
        "base_case_intrinsic_negative": "The base-case value per share is negative.",
    }
    lines = []
    for t in info.get("triggers", []):
        if t.startswith("forecast_fcff_negative_in_"):
            n = t.removeprefix("forecast_fcff_negative_in_").replace("_", " ")
            lines.append(f"Free cash flow is negative in {n} forecast years.")
        else:
            lines.append(labels.get(t, t.replace("_", " ")))
    if lines:
        st.markdown("\n".join(f"- {x}" for x in lines))
    c1, c2 = st.columns(2)
    c1.metric("Price", f"${r.current_price_usd:,.2f}")
    c2.metric("Base case (not a valuation)", f"${info.get('base_case_intrinsic', float('nan')):,.2f}")


def render(r) -> None:
    info = (r.transparency or {}).get("dcf_applicability", {})
    if not info.get("dcf_applicable", True):
        _not_applicable(r, info)
        st.divider()
        methodology.render()
        details.provenance(r)
        details.history(r)
        details.forecast(r)
        return

    price, base = r.current_price_usd, r.dcf.value_per_share_usd
    c = st.columns(5)
    c[0].metric("Price", f"${price:,.2f}")
    c[1].metric("Median value (P50)", f"${r.p50:,.2f}", delta=pct((r.p50 - price) / price, 1, True))
    c[2].metric("P10 to P90", f"\\${r.p10:,.0f} to \\${r.p90:,.0f}")
    c[3].metric("Scenarios above price", f"{r.pct_undervalued:.1f}%")
    c[4].metric("Base case", f"${base:,.2f}", help="One DCF run with every input at its most likely value.")

    for f in r.flags:
        st.warning(f.replace("$", "\\$"))
    _reverse_dcf(r)
    _reading(r)
    st.plotly_chart(charts.value_density(r), width='stretch')
    st.dataframe(pd.DataFrame({
        "": ["P10", "P25", "P50 (median)", "P75", "P90", "Mean", "Std dev"],
        "Value per share": [f"${v:,.2f}" for v in (r.p10, r.p25, r.p50, r.p75, r.p90, r.mean_val, r.std_val)],
    }), hide_index=True, width='content')
    st.caption("P10 means 10% of simulated scenarios give a lower value.")

    st.divider()
    methodology.render()
    details.provenance(r)
    details.history(r)
    details.wacc(r)
    details.forecast(r)
    details.terminal_and_bridge(r)
    details.simulation_inputs(r)
    details.checks(r)
