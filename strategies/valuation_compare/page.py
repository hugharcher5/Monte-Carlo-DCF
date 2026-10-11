"""
S22 on its own page: does the DCF engine pick better stocks than a simple
P/E, P/S or P/B ratio?

render_valuation_compare_page() shows the October 2026 rerun on the current
model (research/valuationcompare_S22/rerun_v2.py and analyze_v2.py).
_render_s22_group() is the archived July 2026 card, kept for the registry.
"""

from __future__ import annotations

import json
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
        st.dataframe(pd.DataFrame(comp_rows), width='stretch', hide_index=True)

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

_S22 = Path(__file__).resolve().parents[2] / "research" / "valuationcompare_S22" / "results"
_V2 = _S22 / "v2"
_SIGNALS = [("dcf", "DCF base case", "seagreen"),
            ("dcf_mc", "DCF Monte Carlo median", "teal"),
            ("dcf_pu", "DCF share of scenarios above price", "olive"),
            ("pe", "P/E", "steelblue"), ("ps", "P/S", "darkorange"), ("pb", "P/B", "mediumpurple")]
_JULY = [("dcf", "DCF (July 2026 model)"), ("pe", "P/E"), ("ps", "P/S"), ("pb", "P/B")]


@st.cache_data(show_spinner=False)
def _load_v2() -> tuple[dict, pd.DataFrame, dict]:
    runs = {s: pd.read_csv(_V2 / f"s22v2_{s}.csv", parse_dates=["rebalance_date"]) for s, _, _ in _SIGNALS}
    bench = pd.read_csv(_V2 / "s22v2_benchmark.csv", parse_dates=["rebalance_date"])
    return runs, bench, json.loads((_V2 / "s22v2_metrics.json").read_text())


@st.cache_data(show_spinner=False)
def _load_july(kind: str) -> dict[str, pd.DataFrame]:
    return {s: pd.read_csv(_S22 / f"s22_{s}_{kind}.csv", parse_dates=["rebalance_date"]) for s, _ in _JULY}


def _metrics(r: pd.Series, ic: pd.Series | None = None, bench: pd.Series | None = None) -> dict:
    """Quarterly returns -> annual return (3 dp), Sharpe, worst fall, Calmar."""
    growth = pd.concat([pd.Series([1.0]), (1 + r).cumprod()], ignore_index=True)
    annual = growth.iloc[-1] ** (4 / len(r)) - 1
    worst = (growth / growth.cummax() - 1).min()
    return {
        "Return a year": f"{annual:+.3%}",
        "Sharpe ratio": f"{r.mean() / r.std() * 2:.2f}",
        "Worst fall": f"{worst:.0%}",
        "Calmar ratio": f"{annual / abs(worst):.2f}" if worst < 0 else "n/a",
        "Ranking accuracy (IC)": f"{ic.mean():+.3f}" if ic is not None else "n/a",
        "Quarters beating benchmark": f"{(r.values > bench.values).mean():.0%}" if bench is not None else "n/a",
    }


def _growth(dates: pd.Series, r: pd.Series) -> tuple[list, list]:
    """$1 at the first rebalance, then its value at the end of each quarter."""
    return [dates.iloc[0]] + list(dates + pd.DateOffset(months=3)), [1.0] + list((1 + r).cumprod())


def render_valuation_compare_page() -> None:
    runs, bench, m = _load_v2()
    sig, bm = m["signals"], m["benchmark_common_after_costs"]
    st.markdown("### Does a DCF pick better stocks than a simple ratio?")
    st.caption(f"Rerun on 11 October 2026 with the current model (code version {m['commit']}). "
               "The July 2026 results are in the expander at the bottom.")

    st.markdown(
        "**Aim.** The valuation tool runs a full discounted cash flow model: around a dozen inputs "
        "from each company's filings, a cost of capital, a forecast and a terminal value. A P/E, P/S "
        "or P/B ratio takes one line of a spreadsheet. If a free ratio finds undervalued stocks just "
        "as well, the extra work buys nothing. This experiment tests whether it does.")
    st.markdown(
        "**Hypothesis.** A portfolio of the stocks the DCF calls most undervalued earns a higher "
        "risk-adjusted return than portfolios picked the same way by P/E, P/S or P/B.")

    st.markdown("**Variables**")
    st.dataframe(pd.DataFrame([
        {"Type": "Independent (what changes)",
         "Variable": "The signal used to rank stocks. Three from the DCF: the base-case margin of safety "
                     "((value − price) ÷ price), the Monte Carlo median's margin of safety, and the share "
                     "of 1,000 simulated scenarios above the price. Three ratios: P/E, P/S and P/B"},
        {"Type": "Dependent (what is measured)",
         "Variable": "Return after trading costs, Sharpe ratio, worst fall, Calmar ratio, ranking accuracy "
                     "(information coefficient) and return compared with the benchmark"},
        {"Type": "Constant: universe",
         "Variable": "US companies worth $100M to $2B, price at least $1, at least $1M traded a day. "
                     "Includes companies that later went bankrupt or delisted (Tiingo prices), held to "
                     "their last trading day. Banks and insurers are left out, because the DCF does not "
                     "value them"},
        {"Type": "Constant: data",
         "Variable": "Point in time: only filings published before each rebalance date (SEC XBRL), the "
                     "bond yield published on or before that date, and model base rates fitted only on "
                     "results known by the end of 2014"},
        {"Type": "Constant: period",
         "Variable": "January 2015 to January 2021, rebalanced every quarter (24 quarters)"},
        {"Type": "Constant: portfolio rule",
         "Variable": "Buy the 20% most undervalued, equal weight, long only, at least 20 stocks"},
        {"Type": "Constant: costs",
         "Variable": "Bid-ask spread of 0.2% to 1% a trade depending on company size, on every trade"},
        {"Type": "Constant: stocks compared",
         "Variable": "Only stocks every signal can score that quarter"},
    ]), hide_index=True, width='stretch')

    st.markdown(
        "**Method.**\n"
        "1. Each quarter, build the universe from prices and filings available on that date.\n"
        "2. Score every stock six ways. P/E only works when earnings are positive, P/B when book "
        "equity is positive, and the DCF when the filings have every input it needs and the company "
        "produces cash.\n"
        "3. Keep only stocks with all six scores, so each signal chooses from the same list.\n"
        "4. Buy the top 20% by each signal, hold for a quarter, subtract trading costs, repeat.\n"
        "5. Compare the portfolios with each other and with a benchmark that holds every stock on "
        "the same list equally, with the same trading costs.\n"
        "6. Test the differences: a paired bootstrap of the quarterly returns, and a regression on "
        "the standard market, size, value and momentum factors.")

    st.markdown(
        "**How we made it a fair test.**\n"
        "- **No survivorship bias.** A backtest that only uses companies still listed today quietly "
        "drops every firm that went bankrupt, was delisted or was taken over, so it only sees the "
        "survivors and its returns look better than they were. To avoid that, this project paid for "
        "a Tiingo data subscription, whose price history includes companies that no longer trade. "
        "Each quarter's universe is rebuilt from the stocks that existed then, and a stock that later "
        "delisted is held until its last trading day.\n"
        "- **No look-ahead.** Each quarter uses only filings the SEC had published by that date and "
        "the bond yield published by that date. The model's base rates (how much past growth "
        "carries forward) were refitted on results known by the end of 2014, before the test starts.\n"
        "- **Correct returns.** Returns use prices adjusted for stock splits and dividends. The July "
        f"run used raw prices, which got {m['split_cases']} stock-quarters badly wrong (one stock showed "
        "+290% over a quarter in which it actually fell 2%).\n"
        "- **Same stocks for every signal.** All six signals rank the same list, so no signal gains "
        "from choosing among easier companies.\n"
        "- **A fair benchmark.** The benchmark holds that same list equally and pays the same "
        "trading costs, so it differs from the portfolios only in not choosing.\n"
        "- **Adjusted for repeated testing.** This was one of 28 strategies tested, so results are "
        "judged with a deflated Sharpe ratio that raises the bar for each extra test.")

    st.markdown("**Sample size** (average per quarter)")
    st.dataframe(pd.DataFrame([{
        "Stocks in universe": f"{bench['n_universe'].mean():.0f}",
        "Scored by all six signals": f"{bench['n'].mean():.0f}",
        "Held by each portfolio": f"{runs['dcf']['n_long'].mean():.0f}",
        "Quarters": f"{len(bench)}",
    }]), hide_index=True, width='content')

    st.markdown("**Results**")
    fig = go.Figure()
    for s, label, color in _SIGNALS:
        x, y = _growth(runs[s]["rebalance_date"], runs[s]["ret_net"])
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label, line=dict(color=color, width=2),
                                 hovertemplate="%{x|%b %Y}: $%{y:.2f}<extra></extra>"))
    x, y = _growth(bench["rebalance_date"], bench["ret_net"])
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Benchmark: same stocks, equal weight",
                             line=dict(color="grey", dash="dash"),
                             hovertemplate="%{x|%b %Y}: $%{y:.2f}<extra></extra>"))
    fig.update_layout(title="Growth of $1, same stocks for every signal, after costs",
                      yaxis_title="Value of $1", height=440, margin=dict(t=50, b=40),
                      legend=dict(orientation="h", yanchor="top", y=-0.12))
    st.plotly_chart(fig, width='stretch')
    st.dataframe(pd.DataFrame(
        [{"Signal": label, **_metrics(runs[s]["ret_net"], runs[s]["ic"], runs[s]["bench_net"])}
         for s, label, _ in _SIGNALS]
        + [{"Signal": "Benchmark: same stocks, equal weight (after costs)", **_metrics(bench["ret_net"])},
           {"Signal": "Whole universe, equal weight (before costs)", **_metrics(bench["univ_gross"])}]
    ), hide_index=True, width='stretch')
    st.caption("All rows after trading costs except the last. Calmar ratio = return a year ÷ worst fall. "
               "The whole-universe row holds all ~420 stocks with no costs, so it is a generous "
               "comparison, shown for continuity with the July run.")

    st.markdown("**Is any difference real?**")
    rows = []
    names = dict((s, l) for s, l, _ in _SIGNALS)
    for k, v in m["bootstrap"].items():
        a, b = k.split("_minus_")
        rows.append({"Comparison": f"{names[a]} minus {names[b]}",
                     "Average gap a year": f"{v['mean_diff_annual']:+.2%}",
                     "95% range": f"{v['ci_low']:+.1%} to {v['ci_high']:+.1%}",
                     "Share of resamples where the DCF did no better": f"{v['p_no_better']:.0%}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption("Paired bootstrap: the 24 quarters are resampled with replacement 10,000 times, keeping "
               "each quarter's two returns together. Every 95% range includes zero.")
    st.dataframe(pd.DataFrame([
        {"Portfolio": label, "Alpha a year": f"{sig[s]['ff_alpha_annual']:+.2%}",
         "Alpha t-stat": f"{sig[s]['ff_alpha_t']:.2f}", "Market": f"{sig[s]['ff_mkt']:.2f}",
         "Size": f"{sig[s]['ff_smb']:.2f}", "Value": f"{sig[s]['ff_hml']:.2f}",
         "Momentum": f"{sig[s]['ff_mom']:.2f}", "R²": f"{sig[s]['ff_r2']:.2f}"}
        for s, label, _ in _SIGNALS
    ] + [{"Portfolio": "Benchmark", "Alpha a year": f"{bm['ff_alpha_annual']:+.2%}",
          "Alpha t-stat": f"{bm['ff_alpha_t']:.2f}", "Market": f"{bm['ff_mkt']:.2f}",
          "Size": f"{bm['ff_smb']:.2f}", "Value": f"{bm['ff_hml']:.2f}",
          "Momentum": f"{bm['ff_mom']:.2f}", "R²": f"{bm['ff_r2']:.2f}"}]),
        hide_index=True, width='stretch')
    st.caption("Factor regression: each portfolio's quarterly return above the T-bill rate, explained by "
               "the Fama-French market, size and value factors and momentum (Ken French data library). "
               "Alpha is what is left over. A t-stat below about 2 means it cannot be told apart from zero.")

    d, ps, pb = sig["dcf"], sig["ps"], sig["pb"]
    st.markdown(
        f"**Conclusion.** The hypothesis is not supported. On the same stocks, the DCF base case did "
        f"best on paper ({d['annual']:+.1%} a year after costs, against {ps['annual']:+.1%} for P/S, "
        f"{sig['pe']['annual']:+.1%} for P/E and {pb['annual']:+.1%} for P/B, and "
        f"{bm['annual']:+.1%} for the benchmark), and it had the smallest worst fall of the six. But the "
        f"gaps are small next to the noise: every bootstrap range includes zero, and no portfolio has a "
        f"factor alpha that can be told apart from zero. The returns of all six portfolios move almost "
        f"together (correlations 0.98 to 0.998), because they hold many of the same stocks and all "
        f"ride the small-cap market. The Monte Carlo signals did no better than the plain base case. "
        f"None of the six passes the deflated Sharpe test for 28 strategies (best quarterly Sharpe "
        f"{d['per_period_sr']:.3f} against a bar of 0.368). The fair reading is that the DCF does at "
        f"least as well as the ratios on this sample, but this test cannot show it does better.")
    st.markdown(
        "**Limitations.** Six years and 24 quarters is a short sample, so only large differences "
        "could be detected. The test covers US small caps only, where a DCF has the least reliable "
        "inputs. The DCF can score about two thirds of the universe, so the fair comparison uses "
        "about 180 stocks a quarter. The backtest uses filings only, so it has no cross-source "
        "comparison, and the Monte Carlo uses 1,000 scenarios per stock rather than the app's 10,000.")

    with st.expander("July 2026 run (earlier model, raw prices)"):
        st.markdown(
            "The first version of this test (July 2026) used an earlier DCF: one deterministic value "
            "per stock, a 20% floor on long-run margins and capital spending falling to depreciation. "
            "It also measured returns on raw prices, which mishandles stock splits, and compared with "
            "the whole universe before costs. Its first run let each signal choose from every stock it "
            "could score, and P/S came out on top; restricting all four to the same stocks made the "
            "differences disappear. Its fair-run results are below for reference.")
        july = _load_july("intersection")
        st.dataframe(pd.DataFrame([{"Signal": label, **_metrics(july[s]["ret_net"], july[s]["ic"])}
                                   for s, label in _JULY]), hide_index=True, width='stretch')
