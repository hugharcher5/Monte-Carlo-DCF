"""
S22 on its own page: does the DCF engine pick better stocks than a simple
P/E, P/S or P/B ratio? Self-contained (only streamlit and pandas), and the
Phase 3 registry page imports its card from here, so the numbers live in one place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ── S22 group card (long-only valuation-signal comparison, 4 sub-trials) ──────

_S22_VARIANTS = [
    {
        "id": "DCF", "label": "DCF-implied margin of safety",
        "ic": "+0.0014", "ic_t": "0.072",
        "gross": "+12.54%/yr", "cost_bps": "180", "net": "+10.59%/yr",
        "sharpe": "0.452", "calmar": "0.233", "max_dd": "−45.5%", "beta": "1.010 (t=24.5)",
        "dsr_obs": "0.226", "dsr_threshold": "0.368",
        "coverage": "63.4%", "alpha_ew": "−2.74%/yr", "alpha_tbill": "+15.53%/yr",
    },
    {
        "id": "P/E", "label": "Price / TTM net income",
        "ic": "+0.0132", "ic_t": "0.566",
        "gross": "+9.85%/yr", "cost_bps": "186", "net": "+7.88%/yr",
        "sharpe": "0.388", "calmar": "0.127", "max_dd": "−61.9%", "beta": "1.221 (t=20.7)",
        "dsr_obs": "0.194", "dsr_threshold": "0.368",
        "coverage": "58.9%", "alpha_ew": "−5.77%/yr", "alpha_tbill": "+16.32%/yr",
    },
    {
        "id": "P/S", "label": "Price / TTM revenue",
        "ic": "+0.0210", "ic_t": "0.651",
        "gross": "+14.49%/yr", "cost_bps": "144", "net": "+12.91%/yr",
        "sharpe": "0.493", "calmar": "0.225", "max_dd": "−57.3%", "beta": "1.279 (t=19.3)",
        "dsr_obs": "0.246", "dsr_threshold": "0.368",
        "coverage": "90.8%", "alpha_ew": "−1.49%/yr", "alpha_tbill": "+22.57%/yr",
    },
    {
        "id": "P/B", "label": "Price / book equity",
        "ic": "+0.0040", "ic_t": "0.156",
        "gross": "+9.64%/yr", "cost_bps": "179", "net": "+7.72%/yr",
        "sharpe": "0.389", "calmar": "0.132", "max_dd": "−58.5%", "beta": "1.246 (t=26.8)",
        "dsr_obs": "0.194", "dsr_threshold": "0.368",
        "coverage": "94.5%", "alpha_ew": "−6.03%/yr", "alpha_tbill": "+16.54%/yr",
    },
]


def _render_s22_group(expanded: bool = False) -> None:
    with st.expander(
        "S22 — Long-Only Quarterly Valuation Portfolios: DCF vs P/E vs P/S vs P/B",
        expanded=expanded,
    ):
        st.markdown(
            "Top-20%-most-undervalued, equal-weight, **long-only** portfolios (genuinely "
            "new construction vs. every other trial in this registry, which are beta-"
            "neutral long/short) — four valuation signals tested head-to-head on the same "
            "underlying question. All four share DSR trial #28 (N_TRIALS=28, quarterly "
            "threshold 0.368). Beta ≈ 1.0–1.28 (all t > 19) confirms the long-only "
            "construction carries genuine market beta by design, not a defect. Window "
            "2015-01-01→2021-01-01 (24 quarters)."
        )

        comp_rows = []
        for v in _S22_VARIANTS:
            comp_rows.append({
                "Signal":       v["id"],
                "IC (t)":       f"{v['ic']} ({v['ic_t']})",
                "Gross":        v["gross"],
                "Cost (bps/yr)": v["cost_bps"],
                "Net":          v["net"],
                "Sharpe":       v["sharpe"],
                "CALMAR":       v["calmar"],
                "Max DD":       v["max_dd"],
                "Beta (t)":     v["beta"],
                "DSR obs":      v["dsr_obs"],
                "DSR thresh":   v["dsr_threshold"],
                "Coverage":     v["coverage"],
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        st.divider()

        st.markdown(
            "**All four FAIL DSR at N=28.** Alpha vs. the equal-weight universe is "
            "NEGATIVE for every signal (DCF −2.74%, P/E −5.77%, P/S −1.49%, P/B −6.03%/yr) "
            "despite strongly positive alpha vs. the 3-month T-bill (+15.5% to +22.6%/yr) — "
            "absolute returns are substantially explained by high-beta exposure to a rising "
            "2015–2020 small-cap market, not genuine alpha over that market."
        )

        st.markdown(
            "**Original (unequal-universe) ranking:** P/S > DCF > P/E ≈ P/B, with P/S "
            "best on Sharpe/IC/alpha. **This ranking did NOT survive scrutiny.** Signal "
            "coverage was very unequal each quarter (mean universe ~422 names): "
            "P/B 94.5%, P/S 90.8%, DCF 63.4%, P/E 58.9% — DCF and P/E were effectively "
            "ranking from a smaller, more-established/profitable subset (P/E structurally "
            "excludes loss-making names; DCF needs ~12 XBRL concepts plus an "
            "applicability gate vs. P/B/P/S's single-concept dependency)."
        )
        st.markdown(
            "**CORRECTION (2026-07-15) — intersection-universe re-run:** restricting all "
            "four signals to the common subset where all four have valid values each "
            "quarter (mean 179/quarter, down from ~422) made the original ranking "
            "**evaporate**: Sharpe clusters 0.41–0.43 for all four (vs. the original "
            "0.39–0.49 spread), alpha vs. market clusters −3.7% to −4.2% (all four now "
            "negative and close together), and the IC ranking **inverts** — P/E is now "
            "best (0.017), P/B flips to negative (−0.0025), P/S (the original 'winner') "
            "drops to third. **Corrected conclusion: none of the four signals is "
            "distinguishable from the others on a fair, equal-universe basis** — DCF's "
            "substantial added complexity bought no measurable edge, but also no "
            "underperformance, once the universe is held fixed. Re-run was cross-checked "
            "byte-for-byte against the original computation first (max diff 0.0 across all "
            "96 signal-periods) before trusting the corrected numbers. This is a "
            "*stronger* negative finding than the original ('simple beats complex'), not a "
            "weaker one — treat any future multi-signal head-to-head in this registry with "
            "the same coverage-parity check before ranking."
        )
        st.markdown(
            "Two infrastructure fixes made along the way, both reusable and unrelated to "
            "the ranking correction: (1) a new PIT adapter "
            "(`research/valuationcompare_S22/pit_facts.py`, `dcf_adapter.py`) feeds the "
            "existing DCF engine's already-pure computation core "
            "(`compute_drivers`/`compute_wacc`/`_build_base`/`dcf.value`) via one additive "
            "`as_of` parameter on `wacc.py`'s `_get_rf()` — no parallel WACC calculator "
            "needed. (2) `compute_wacc()` was calling a live FRED fetch on every single "
            "invocation, uncached — harmless for the live tool's one-valuation-at-a-time "
            "use but made backtesting ~65x slower than necessary; fixed with module-level "
            "memoization (3.85s/name → 0.059s/name), which also speeds up the live tool."
        )


# ── Main render: the experiment write-up ──────────────────────────────────────

_RESULTS = Path(__file__).resolve().parents[2] / "research" / "valuationcompare_S22" / "results"
_SIGNALS = [("dcf", "DCF margin of safety", "seagreen"), ("pe", "P/E", "steelblue"),
            ("ps", "P/S", "darkorange"), ("pb", "P/B", "mediumpurple")]


@st.cache_data(show_spinner=False)
def _load(kind: str) -> dict[str, pd.DataFrame]:
    """kind: 'intersection' (same stocks for every signal) or 'period_returns' (first run)."""
    return {s: pd.read_csv(_RESULTS / f"s22_{s}_{kind}.csv", parse_dates=["rebalance_date"])
            for s, _, _ in _SIGNALS}


def _metrics(df: pd.DataFrame, col: str = "ret_net") -> dict:
    """Quarterly returns → annual return, Sharpe, worst fall and Calmar.
    col='mkt_ret' gives the equal-weight universe, which ranks nothing."""
    r = df[col]
    growth = pd.concat([pd.Series([1.0]), (1 + r).cumprod()], ignore_index=True)
    annual = growth.iloc[-1] ** (4 / len(r)) - 1
    worst = (growth / growth.cummax() - 1).min()
    ranked = col == "ret_net"
    return {
        "Return a year": f"{annual:+.1%}",
        "Sharpe ratio": f"{r.mean() / r.std() * 2:.2f}",
        "Worst fall": f"{worst:.0%}",
        "Calmar ratio": f"{annual / abs(worst):.2f}" if worst < 0 else "n/a",
        "Ranking accuracy (IC)": f"{df['ic'].mean():+.3f}" if ranked else "n/a",
        "Quarters beating the market": f"{(r > df['mkt_ret']).mean():.0%}" if ranked else "n/a",
    }


def render_valuation_compare_page() -> None:
    st.markdown("### Does a DCF pick better stocks than a simple ratio?")

    st.markdown(
        "**Aim.** The valuation tool runs a full discounted cash flow model: around a dozen "
        "inputs from each company's filings, a cost of capital, a forecast and a terminal "
        "value. A P/E, P/S or P/B ratio takes one line of a spreadsheet. If a free ratio "
        "finds undervalued stocks just as well, the extra work buys nothing. This experiment "
        "tests whether it does."
    )
    st.markdown(
        "**Hypothesis.** A portfolio of the stocks the DCF calls most undervalued earns a "
        "higher risk-adjusted return than portfolios picked the same way by P/E, P/S or P/B."
    )

    st.markdown("**Variables**")
    st.dataframe(pd.DataFrame([
        {"Type": "Independent (what changes)",
         "Variable": "The signal used to rank stocks: DCF margin of safety "
                     "((DCF value − price) ÷ price), P/E, P/S or P/B"},
        {"Type": "Dependent (what is measured)",
         "Variable": "Return after trading costs, Sharpe ratio, worst fall, ranking accuracy "
                     "(information coefficient), and return compared with the whole universe"},
        {"Type": "Constant: universe",
         "Variable": "US companies worth $100M to $2B, price at least $1, at least $1M traded a day. "
                     "Includes companies that later went bankrupt or delisted (Tiingo prices), "
                     "held to their last trading day"},
        {"Type": "Constant: data",
         "Variable": "Point in time: only filings published before each rebalance date (SEC XBRL)"},
        {"Type": "Constant: period",
         "Variable": "January 2015 to January 2021, rebalanced every quarter (24 quarters)"},
        {"Type": "Constant: portfolio rule",
         "Variable": "Buy the 20% most undervalued, equal weight, long only, at least 20 stocks"},
        {"Type": "Constant: costs",
         "Variable": "Bid-ask spread of 0.2% to 1% a trade depending on company size, on every trade"},
        {"Type": "Constant: stocks compared",
         "Variable": "In the fair run, only stocks all four signals can value that quarter"},
    ]), hide_index=True, use_container_width=True)

    st.markdown(
        "**Method.**\n"
        "1. Each quarter, build the universe from prices and filings available on that date.\n"
        "2. Score every stock four ways. P/E only works when earnings are positive, P/B when "
        "book equity is positive, and the DCF when the filings have every input it needs.\n"
        "3. Keep only stocks with all four scores, so each signal chooses from the same list.\n"
        "4. Buy the top 20% by each signal, hold for a quarter, subtract trading costs, repeat.\n"
        "5. Compare the four portfolios with each other and with all stocks in the universe "
        "held equally."
    )

    st.markdown(
        "**How we made it a fair test.**\n"
        "- **No survivorship bias.** A backtest that only uses companies still listed today "
        "quietly drops every firm that went bankrupt, was delisted or was taken over, so it "
        "only sees the survivors and its returns look better than they were. To avoid that, "
        "this project paid for a Tiingo data subscription, whose price history includes "
        "companies that no longer trade. Each quarter's universe is rebuilt from the stocks "
        "that existed then, and a stock that later delisted is held until its last trading day.\n"
        "- **No look-ahead.** Each quarter uses only filings the SEC had published by that date.\n"
        "- **Same stocks for every signal.** All four signals rank the same list, so no signal "
        "gains from choosing among easier companies.\n"
        "- **Same rules.** The same rebalance dates, portfolio size, equal weighting and "
        "trading-cost model apply to all four.\n"
        "- **Adjusted for repeated testing.** This was one of 28 strategies tested, so results "
        "are judged with a deflated Sharpe ratio that raises the bar for each extra test.\n\n"
        "One caveat: this test used the DCF as it was in July 2026, valued once per stock "
        "without the Monte Carlo, with a 20% floor on long-run margins and capital spending "
        "falling to depreciation. The live tool has changed since, so these results describe "
        "that earlier version."
    )

    first = _load("period_returns")
    fair = _load("intersection")
    st.markdown("**Sample size** (average per quarter)")
    st.dataframe(pd.DataFrame([
        {"Test": label,
         "Stocks in universe": f"{first[s]['n_universe'].mean():.0f}",
         "Stocks the signal could score": f"{first[s]['n_with_signal'].mean():.0f}",
         "Stocks scored by all four (fair run)": f"{fair[s]['n_universe'].mean():.0f}",
         "Stocks held (fair run)": f"{fair[s]['n_long'].mean():.0f}"}
        for s, label, _ in _SIGNALS
    ]), hide_index=True, use_container_width=True)
    st.caption("825 different companies passed through the universe over the 24 quarters.")

    st.markdown("**Results**")
    def _growth(d: pd.DataFrame, col: str) -> tuple[list, list]:
        """$1 at the first rebalance, then its value at the end of each quarter."""
        x = [d["rebalance_date"].iloc[0]] + list(d["rebalance_date"] + pd.DateOffset(months=3))
        return x, [1.0] + list((1 + d[col]).cumprod())

    fig = go.Figure()
    for s, label, color in _SIGNALS:
        x, y = _growth(fair[s], "ret_net")
        fig.add_trace(go.Scatter(x=x, y=y,
                                 mode="lines", name=label, line=dict(color=color, width=2),
                                 hovertemplate="%{x|%b %Y}: $%{y:.2f}<extra></extra>"))
    x, y = _growth(fair["dcf"], "mkt_ret")
    fig.add_trace(go.Scatter(x=x, y=y,
                             mode="lines", name="Whole universe, equal weight",
                             line=dict(color="grey", dash="dash"),
                             hovertemplate="%{x|%b %Y}: $%{y:.2f}<extra></extra>"))
    fig.update_layout(title="Growth of $1, same stocks for every signal, after costs",
                      yaxis_title="Value of $1", height=420, margin=dict(t=50, b=40),
                      legend=dict(orientation="h", yanchor="top", y=-0.12))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame(
        [{"Signal": f"{label} (after costs)", **_metrics(fair[s])} for s, label, _ in _SIGNALS]
        + [{"Signal": "Whole universe, equal weight (before costs)", **_metrics(fair["dcf"], "mkt_ret")}]
    ), hide_index=True, use_container_width=True)
    st.caption(
        "Calmar ratio = return a year ÷ worst fall. The whole universe holds all ~422 stocks "
        "equally with no trading costs, so it ranks nothing and is a slightly generous benchmark."
    )

    st.markdown(
        "**Conclusion.** The hypothesis is rejected. On the same stocks, every \\$1 grew to "
        "between \\$1.70 and \\$1.73 whichever signal picked them, with Sharpe ratios between 0.41 "
        "and 0.43. Holding every stock in the universe equally grew \\$1 to \\$2.13 (before costs), "
        "so none of the four added value over the market. The DCF did no worse than the "
        "ratios, but its extra complexity bought no measurable edge. None of the four passed "
        "the deflated Sharpe test, which adjusts for the 28 strategies tested in this research "
        "programme. Most of the return came from small caps rising over the period."
    )
    st.markdown(
        "**Limitations.** Six years and 24 quarters is a short sample. The test covers US "
        "small caps only, where a DCF has the least reliable inputs. The DCF could value only "
        "about two thirds of the universe, so the fair run uses fewer stocks."
    )

    with st.expander("First run, before the comparison was made fair"):
        st.markdown(
            "The first run let each signal choose from every stock it could score. P/S came "
            "out on top and the DCF second. That ranking was misleading: P/S and P/B could "
            "score over 90% of the universe, while P/E and the DCF only scored about 60%, "
            "mostly larger and profitable companies. Restricting all four to the same stocks "
            "made the differences disappear, which is why the fair run above is the result."
        )
        st.dataframe(pd.DataFrame([{"Signal": label, **_metrics(first[s])} for s, label, _ in _SIGNALS]),
                     hide_index=True, use_container_width=True)
