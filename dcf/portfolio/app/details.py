"""The detail expanders under the headline valuation."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from valuation.filings import FIELD_LABELS
from valuation.priors import VARIABLES

from . import charts
from .common import pct, table


def _disagree(r, field: str) -> str:
    info = (r.recon_fields or {}).get(field)
    if not info or info["disagree_pct"] is None:
        return "n/a"
    d = info["disagree_pct"]
    if info["field_type"] == "DEFINITIONAL":
        return f"{d:.1f}% (definition)"
    return f"{d:.1f}%"


def provenance(r) -> None:
    prov = r.data_provenance or {}
    if not prov:
        return
    with st.expander("Where the data comes from", expanded=False):
        st.markdown(
            "Each input comes from the company's SEC filings where they give it cleanly, from "
            "FMP when the filings have no cash, debt or share figure, and from Yahoo Finance "
            "otherwise. Each row says which source was used and why.")
        table([(FIELD_LABELS.get(k, k), v["source"], v["detail"]) for k, v in prov.items()],
              ["Input", "Source", "Detail"])
        n = sum(1 for v in prov.values() if v["source"] == "SEC EDGAR")
        st.caption(f"{n} of {len(prov)} inputs come from SEC filings.")


def history(r) -> None:
    d, fi = r.drivers, r.forecast_inputs
    with st.expander("History and forecast inputs", expanded=False):
        st.markdown(
            "The company's own figures for each year, then how they become forecast inputs. "
            "Growth and margin are pulled toward the averages of similar-sized US companies "
            "(see \"How the model works\", section 3), and the standard deviation is the "
            "uncertainty about the long-run level that the simulation samples.")
        if d.hist_df is not None:
            cols = [c for c in ["revenue_growth", "ebit_margin", "tax_rate", "da_pct", "capex_pct", "sbc_pct"]
                    if c in d.hist_df.columns]
            h = d.hist_df[cols].copy()
            h.index = [str(i.year) for i in h.index]
            h.columns = ["Revenue growth", "EBIT margin", "Tax rate", "D&A %", "CapEx %", "SBC %"][:len(cols)]
            st.dataframe(h.map(lambda x: f"{x:.1%}" if pd.notna(x) else ""), width='stretch')
        rows = [
            ("Revenue growth", pct(d.revenue_growth), pct(fi.revenue_growth), pct(fi.revenue_growth_sd),
             f"each point of past growth adds {fi.shrink_growth:.2f} of a point", _disagree(r, "revenue")),
            ("EBIT margin (start)", pct(d.ebit_margin), pct(r.assumptions.ebit_margin), pct(fi.margin_sd),
             f"slope {fi.shrink_margin:.2f}", _disagree(r, "ebit")),
            ("EBIT margin (mature)", pct(d.best_ebit_margin) + " best year", pct(r.assumptions.target_margin),
             pct(fi.margin_sd), r.target_margin_basis, ""),
            ("CapEx / revenue", pct(d.capex_pct), pct(r.assumptions.capex_pct), pct(fi.capex_sd),
             "own average", _disagree(r, "capex")),
            ("D&A / revenue", pct(d.da_pct), pct(d.da_pct), "", "held", _disagree(r, "dep_amort")),
            ("Tax rate", pct(d.tax_rate), pct(d.tax_rate), "", "held", _disagree(r, "tax_rate")),
            ("Working capital / revenue", pct(d.nwc_pct), pct(d.nwc_pct), "", "latest balance sheet", ""),
            ("SBC / revenue", pct(d.sbc_pct), "", "", "already an expense in EBIT", ""),
        ]
        table(rows, ["Input", "History", "Forecast", "Std dev", "Note", "Source gap"])
        st.caption(f"{d.years_used} years of growth history. \"Source gap\" is how far apart the data "
                   "sources are on the latest year's figure.")


def wacc(r) -> None:
    w, ccy = r.wacc_result, r.currency
    with st.expander("Discount rate (WACC)", expanded=False):
        st.markdown(
            "WACC is the return that lenders and shareholders together require, and it is the rate "
            "every future cash flow is discounted at. Cost of equity uses CAPM; cost of debt uses "
            "the credit spread implied by interest cover, less the tax saving on interest.")
        st.latex(r"\text{WACC} = w_E K_E + w_D K_D (1-t)")
        cover = f"{w.coverage_ratio:.2f}x" if math.isfinite(w.coverage_ratio) else "no interest"
        table([
            ("Risk-free rate", f"{w.rf:.3%}", w.rf_source),
            ("Beta (raw / adjusted)", f"{w.beta_raw:.2f} / {w.beta_adj:.2f}", "adjusted = 0.67 x raw + 0.33"),
            ("Equity risk premium", f"{w.erp:.2%}", "fixed assumption"),
            ("Cost of equity", f"{w.cost_of_equity:.3%}", "risk-free + beta x premium"),
            ("Interest cover", cover, "EBIT / interest expense"),
            ("Implied rating / spread", f"{w.implied_rating} / {w.credit_spread:.2%}", "Damodaran table, Jan 2024"),
            ("Cost of debt after tax", f"{w.cost_of_debt_aftertax:.3%}", f"before tax {w.cost_of_debt_pretax:.3%}"),
            ("Equity weight", f"{w.equity_weight:.1%}", f"{ccy} {w.equity_value / 1e9:,.1f}bn market value"),
            ("Debt weight", f"{w.debt_weight:.1%}", f"{ccy} {w.debt_value / 1e9:,.1f}bn debt"),
            ("WACC", f"{w.wacc:.3%}", f"simulated with sd {r.sw:.2%}" + (" (fallback)" if w.wacc_sigma_fallback else "")),
        ], ["Part", "Value", "Note"])


def forecast(r) -> None:
    a, res = r.assumptions, r.dcf
    ccy = a.currency
    with st.expander("Year-by-year forecast", expanded=False):
        fye = pd.Timestamp(r.fiscal_year_end).date() if r.fiscal_year_end is not None else "n/a"
        vd = pd.Timestamp(r.valuation_date).date() if r.valuation_date is not None else "today"
        st.markdown(
            f"Free cash flow to the firm is after-tax operating profit, plus D&A, minus CapEx, minus "
            f"the cash tied up in working capital. Revenue growth and margin move in straight lines "
            f"to their end values. The last annual accounts end on {fye} and the valuation date is "
            f"{vd}, so {a.stub:.0%} of year 1 is still to come and only that part is counted. Each "
            f"year's cash is discounted from the middle of the year.")
        st.latex(r"\text{FCFF}_t = \text{EBIT}_t (1-t) + \text{D\&A}_t - \text{CapEx}_t - \Delta\text{NWC}_t")
        df = res.forecast
        bn = lambda s: (s / 1e9).map("{:,.2f}".format)
        st.caption(f"Money in {ccy} billions.")
        st.dataframe(pd.DataFrame({
            "Year": df.index, "Growth": df["Growth"].map("{:.1%}".format),
            "Margin": df["Margin%"].map("{:.1%}".format), "Revenue": bn(df["Revenue"]),
            "EBIT": bn(df["EBIT"]), "After-tax EBIT": bn(df["NOPAT"]), "D&A": bn(df["D&A"]),
            "CapEx %": df["CapEx%"].map("{:.1%}".format), "CapEx": bn(df["CapEx"]),
            "Change in NWC": bn(df["ΔNWC"]), "FCFF": bn(df["FCFF"]),
            "Years to cash": df["Years to cash"].map("{:.2f}".format),
            "Discount factor": df["Disc."].map("{:.4f}".format), "Present value": bn(df["PV(FCFF)"]),
        }), width='stretch', hide_index=True)
        if df["CapEx%"].iloc[0] - df["CapEx%"].iloc[-1] > 1e-4:
            st.caption(f"CapEx falls from {df['CapEx%'].iloc[0]:.1%} to {df['CapEx%'].iloc[-1]:.1%} of revenue "
                       f"because net reinvestment shrinks toward what {a.terminal_g:.1%} terminal growth needs.")


def terminal_and_bridge(r) -> None:
    a, res = r.assumptions, r.dcf
    ccy = a.currency
    with st.expander("Terminal value and equity bridge", expanded=False):
        st.markdown(
            "The terminal value covers every year after the forecast, with the final year's cash flow "
            "growing at a constant rate. Enterprise value minus net debt is the equity value, which "
            "is divided by the share count.")
        st.latex(r"TV = \frac{\text{FCFF}_N (1+g)}{\text{WACC} - g}")
        tv_share = res.pv_tv / res.ev_local if res.ev_local else float("nan")
        prov = r.data_provenance or {}
        rows = [
            ("Final-year FCFF", f"{ccy} {res.forecast['FCFF'].iloc[-1] / 1e9:,.2f}bn", ""),
            ("Terminal growth / WACC", f"{a.terminal_g:.2%} / {a.wacc:.2%}", ""),
            ("Present value of forecast years", f"{ccy} {res.pv_explicit / 1e9:,.2f}bn", ""),
            ("Present value of terminal value", f"{ccy} {res.pv_tv / 1e9:,.2f}bn", f"{tv_share:.0%} of enterprise value"),
            ("Enterprise value", f"{ccy} {res.ev_local / 1e9:,.2f}bn", ""),
            ("less net debt" if a.net_debt >= 0 else "plus net cash", f"{ccy} {abs(a.net_debt) / 1e9:,.2f}bn",
             f"debt: {prov.get('total_debt', {}).get('source', '')}; cash: {prov.get('cash', {}).get('source', '')}"),
            ("Equity value", f"{ccy} {res.equity_value_local / 1e9:,.2f}bn", ""),
            ("Shares", f"{a.diluted_shares / 1e9:,.3f}bn", prov.get("diluted_shares", {}).get("source", "")),
        ]
        if ccy != "USD":
            rows += [(f"Value per share ({ccy})", f"{res.value_per_share_local:,.2f}", ""),
                     (f"x {ccy} to USD (spot)", f"{a.fx_rate:.6f}", "net debt converted at the same rate")]
        rows.append(("Value per share (USD)", f"{res.value_per_share_usd:,.2f}", "base case"))
        table(rows, ["Step", "Value", "Note"])
        mkt = a.market_ev_local / a.current_ebitda if a.current_ebitda and a.current_ebitda > 0 else float("nan")
        st.caption(f"Cross-check: the base case implies {res.implied_ev_ebitda:.1f}x EV/EBITDA; "
                   f"the market price implies {mkt:.1f}x." if math.isfinite(mkt) else
                   "EV/EBITDA cross-check not available (EBITDA is zero or negative).")


_LABELS = {"revenue_growth": "Revenue growth", "ebit_margin": "EBIT margin (start)",
           "target_margin": "EBIT margin (mature)", "terminal_g": "Terminal growth", "wacc": "WACC",
           "capex_pct": "CapEx / revenue", "diluted_shares": "Share count", "net_debt": "Net debt"}


def simulation_inputs(r) -> None:
    t = r.transparency or {}
    dp, samples = t.get("distribution_params", {}), t.get("samples", {})
    if not dp:
        return
    with st.expander("Simulation inputs", expanded=False):
        st.markdown(
            "The range each input is drawn from. Each is a PERT distribution whose standard "
            "deviation matches the estimate, kept inside plausibility limits. The margin draw "
            "moves the starting and mature margin together. Open a chart to compare the "
            "intended shape with the actual draws.")
        rows = []
        for var, d in dp.items():
            money = var in ("diluted_shares", "net_debt")
            f = (lambda v: f"{v / 1e9:,.3f}bn") if money else (lambda v: f"{v:.2%}")
            s = samples.get(var)
            rows.append((_LABELS.get(var, var), f(d["lo"]), f(d["mode"]), f(d["hi"]),
                         f(d["sd"]) if d.get("sd") is not None else "fixed range",
                         f(float(np.median(s))) if s is not None and len(s) else ""))
        table(rows, ["Input", "Low", "Most likely", "High", "Std dev", "Median draw"])
        st.caption(f"Terminal growth runs {pct(dp['terminal_g']['lo'])} to {pct(dp['terminal_g']['hi'])} "
                   "and is never closer than 1 point to the WACC drawn in the same scenario.")
        cols = st.columns(4)
        for i, var in enumerate(v for v in dp if v in samples and len(samples[v]) > 10):
            d = dp[var]
            if d["hi"] - d["lo"] > 1e-12:
                with cols[i % 4].popover(f"Chart: {_LABELS.get(var, var)}"):
                    st.plotly_chart(charts.pert_vs_draws(samples[var], d["lo"], d["mode"], d["hi"],
                                                         _LABELS.get(var, var),
                                                         var not in ("diluted_shares", "net_debt")),
                                    width='stretch')

        gc = t.get("g_ceiling", {})
        if gc and gc.get("pct_paths_bound", 0) > 1:
            st.caption(f"The size limit on growth cut the growth rate in {gc['pct_paths_bound']:.1f}% of "
                       f"scenarios (on average {gc['avg_years_bound']:.1f} years each), which trims the "
                       "top of the range.")

        corr_t, corr_r = t.get("correlation_target"), t.get("correlation_realized")
        if corr_t is not None and corr_r is not None:
            names = ["Growth", "Margin", "Terminal g", "WACC", "CapEx"]
            st.markdown("**Correlations.** Estimated from how forecast errors move together across "
                        "companies. The second chart shows the correlations in the actual draws.")
            c1, c2 = st.columns(2)
            c1.plotly_chart(charts.corr_heatmap(corr_t, names, "Target"), width='stretch')
            c2.plotly_chart(charts.corr_heatmap(corr_r, names, f"Actual ({t.get('n_sims', 0):,} draws)"),
                            width='stretch')


def checks(r) -> None:
    t = r.transparency or {}
    val = t.get("validation", {})
    with st.expander("Sources and checks", expanded=False):
        status = getattr(r, "source_status", None) or {}
        if r.recon_fields:
            st.markdown(
                "**How far apart the sources are** on the latest figures. Fields marked "
                "\"definition\" differ because the sources define them differently (for example "
                "whether leases count as debt), so they do not widen the ranges.")
            table([(FIELD_LABELS.get({"dep_amort": "da"}.get(f, f), f), i["source"], _disagree(r, f),
                    "definition" if i["field_type"] == "DEFINITIONAL" else "data")
                   for f, i in r.recon_fields.items()], ["Field", "Source used", "Gap", "Type"])
        if r.recon_sigma:
            rows = []
            for var, i in r.recon_sigma.items():
                money = var in ("diluted_shares", "net_debt")
                f = (lambda v: f"{v / 1e9:,.3f}bn") if money else (lambda v: f"{v:.2%}")
                rows.append((_LABELS.get(var, var), f(i["sigma_hist"]), f(i["sigma_cross"]), f(i["sigma_eff"])))
            table(rows, ["Input", "Model std dev", "Source spread", "Combined"])
            st.caption("Combined = square root of (model std dev squared + source spread squared).")
        if val:
            ok = val.get("zero_width_test_passed", False)
            st.markdown(f"**Zero-width check: {'passed' if ok else 'FAILED'}.** With every range set to zero, "
                        f"the simulation gives \\${val.get('mc_p50_at_zero_width', float('nan')):,.4f}, against "
                        f"\\${val.get('deterministic_base', float('nan')):,.4f} from the base case. They must "
                        f"agree to within a cent. Random seed {t.get('random_seed')}, "
                        f"{t.get('n_sims', 0):,} scenarios.")
