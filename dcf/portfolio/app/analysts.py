"""The "Analyst estimates" sub-tab: one stock's analyst targets, the 100-stock
comparison with its calibration check, and the weekly price tracking."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats
import streamlit as st

from valuation.analysts import AnalystBenchmarks, fetch_benchmarks

DATA = Path(__file__).resolve().parent.parent.parent / "data"
BENCHMARK_CSV = DATA / "analyst_benchmark.csv"
VERDICTS_CSV = DATA / "tracking_verdicts.csv"
PRICES_CSV = DATA / "price_history.csv"

GROUPS = [
    ("Undervalued by both", "seagreen"),
    ("Undervalued by our DCF only", "steelblue"),
    ("Undervalued by analysts only", "darkorange"),
    ("Overvalued by both", "crimson"),
]
EVAL_WEEKS = (26, 52)
EVIDENCE_GAP = 0.15   # about two standard errors of the one-year gap; see _render_tracking


@st.cache_data(show_spinner=False, ttl=3600)
def cached_benchmarks(ticker: str) -> AnalystBenchmarks:
    return fetch_benchmarks(ticker)


def _target_mode(vals: np.ndarray) -> tuple[float, str]:
    """Most common target (analysts favour round numbers); with no repeats, the
    peak of a smoothed density."""
    counts = pd.Series(vals).round(2).value_counts()
    if counts.iloc[0] > 1:
        top = counts[counts == counts.iloc[0]].index
        return float(min(top)), f"{int(counts.iloc[0])} firms"
    if len(vals) > 2 and vals.std() > 0:
        grid = np.linspace(vals.min(), vals.max(), 400)
        return float(grid[np.argmax(scipy.stats.gaussian_kde(vals)(grid))]), "density peak"
    return float(np.median(vals)), "no repeats"


def render_stock(r, b: AnalystBenchmarks) -> None:
    price, targets = r.current_price_usd, b.targets
    ours = r.p50 if r.dcf_applicable and math.isfinite(r.p50) else None
    st.subheader(f"{r.ticker}: analyst price targets")
    s = b.summary or {}
    if targets.empty and s.get("mean"):
        st.info(f"Yahoo Finance has no firm-by-firm targets for {r.ticker} from the last 12 months, "
                "only its consensus summary, so no distribution can be drawn.")
        c = st.columns(4)
        c[0].metric("Mean", f"${s['mean']:,.2f}", delta=f"{(s['mean'] - price) / price * 100:+.1f}% vs price")
        c[1].metric("Median", f"${s.get('median', float('nan')):,.2f}")
        c[2].metric("High", f"${s.get('high', float('nan')):,.2f}")
        c[3].metric("Low", f"${s.get('low', float('nan')):,.2f}")
    elif targets.empty:
        st.info(f"Yahoo Finance lists no analyst price targets for {r.ticker} in the last 12 months.")
    else:
        vals = targets["Target"].to_numpy(dtype=float)
        mu, med = float(vals.mean()), float(np.median(vals))
        sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        mode, mode_note = _target_mode(vals)
        c = st.columns(6)
        c[0].metric("Mean", f"${mu:,.2f}", delta=f"{(mu - price) / price * 100:+.1f}% vs price")
        c[1].metric("Std deviation", f"${sd:,.2f}")
        c[2].metric("Median", f"${med:,.2f}")
        c[3].metric("Mode", f"${mode:,.2f}", help=f"Most common target ({mode_note}).")
        c[4].metric("High", f"${vals.max():,.2f}")
        c[5].metric("Low", f"${vals.min():,.2f}")
        if ours is not None:
            gap = (ours - mu) / mu
            st.markdown(f"**{len(vals)} firms.** Our DCF median of **\\${ours:,.2f}** is "
                        f"**{abs(gap):.1%} {'below' if gap < 0 else 'above'}** the analyst mean"
                        + (f", and FMP's free DCF puts the stock at **\\${b.fmp_dcf:,.2f}**."
                           if b.fmp_dcf is not None else "."))

        refs = [p for p in (price, ours, b.fmp_dcf, vals.min(), vals.max()) if p is not None]
        lo, hi = min(refs), max(refs)
        if sd > 0:
            lo, hi = min(lo, mu - 3.5 * sd), max(hi, mu + 3.5 * sd)
        pad = 0.05 * ((hi - lo) or hi or 1.0)
        x = np.linspace(max(lo - pad, 0.0), hi + pad, 400)
        fig, peak = go.Figure(), 1.0
        if sd > 0:
            y = scipy.stats.norm.pdf(x, mu, sd)
            peak = float(y.max())
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="tozeroy",
                                     fillcolor="rgba(70, 130, 180, 0.25)", line=dict(color="steelblue", width=2),
                                     name=f"Normal fit (mean ${mu:,.0f}, sd ${sd:,.0f})",
                                     hovertemplate="$%{x:,.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=vals, y=np.full(len(vals), peak * 0.03), mode="markers",
            marker=dict(symbol="line-ns-open", size=18, color="steelblue", line=dict(width=2)),
            text=targets["Firm"], customdata=targets["Date"].astype(str),
            hovertemplate="%{text}: $%{x:,.2f}<br>%{customdata}<extra></extra>", name="Each firm's target"))
        lines = [(price, "crimson", "solid", f"Market ${price:,.2f}", "top right")]
        if ours is not None:
            lines.append((ours, "seagreen", "dash", f"Our DCF P50 ${ours:,.2f}", "top left"))
        if b.fmp_dcf is not None:
            lines.append((b.fmp_dcf, "darkorange", "dot", f"FMP DCF ${b.fmp_dcf:,.2f}", "bottom right"))
        for xv, color, dash, label, pos in lines:
            fig.add_vline(x=xv, line_color=color, line_width=2, line_dash=dash,
                          annotation_text=label, annotation_position=pos)
        fig.update_layout(title=f"{r.ticker}: analyst 12-month price targets (latest per firm, last 12 months)",
                          xaxis_title="Price per share (USD)", yaxis_title="Density", height=420,
                          margin=dict(t=50, b=40), legend=dict(orientation="h", yanchor="top", y=-0.2))
        st.plotly_chart(fig, width='stretch')
        with st.expander(f"All {len(vals)} analyst targets"):
            st.dataframe(targets.assign(**{"vs price": (targets["Target"] / price - 1).map("{:+.1%}".format),
                                           "Target": targets["Target"].map("${:,.2f}".format)}),
                         hide_index=True, width='stretch')
    st.caption(
        "Analyst targets come from Yahoo Finance's record of rating changes: the latest price target each "
        "firm published in the last 12 months. They are 12-month price targets, usually built from a mix "
        "of DCF and trading multiples, so they sit closer to the market than a pure intrinsic value. The "
        "FMP DCF is Financial Modeling Prep's free discounted cash flow value, a second DCF to compare with.")
    for name, err in b.errors.items():
        st.warning(f"{name} unavailable: {err}")


@st.cache_data(show_spinner=False, ttl=3600)
def _read(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _calibration(v: pd.DataFrame) -> dict:
    inside = ((v["price"] >= v["p10"]) & (v["price"] <= v["p90"])).mean()
    return {"n": len(v), "inside": inside, "above": (v["price"] > v["p90"]).mean(),
            "below": (v["price"] < v["p10"]).mean(), "median_gap": (v["p50"] / v["price"] - 1).median()}


def _render_calibration(v: pd.DataFrame) -> None:
    st.markdown("#### Calibration: is the price usually inside our range?")
    now = _calibration(v)
    rows = [("This version", now)]
    if VERDICTS_CSV.exists():
        old = _read(str(VERDICTS_CSV)).rename(columns={"start_price": "price", "our_p10": "p10",
                                                       "our_p50": "p50", "our_p90": "p90"})
        rows.insert(0, (f"Previous version ({old['start_date'].iloc[0]})", _calibration(old)))
    st.dataframe(pd.DataFrame([{
        "Model": name, "Stocks": c["n"], "Price inside P10 to P90": f"{c['inside']:.0%}",
        "Price above P90": f"{c['above']:.0%}", "Price below P10": f"{c['below']:.0%}",
        "Median of P50 vs price": f"{c['median_gap']:+.0%}"} for name, c in rows]),
        hide_index=True, width='content')
    st.markdown(
        f"If the ranges were well calibrated and the market price were a fair guess at value, about 80% "
        f"of prices would fall inside P10 to P90, with about 10% on each side. This version gets "
        f"{now['inside']:.0%} inside and {now['above']:.0%} above P90. The model still sits below the "
        f"market for most of these large, popular stocks: its median is {abs(now['median_gap']):.0%} "
        f"below the price for the typical stock. That is partly the model's caution (growth is pulled "
        f"toward the average company) and partly the market paying for growth the history does not "
        f"yet show. It is a check on the ranges, not proof either side is right.")


def render_study() -> None:
    st.subheader("How our DCF compares with analysts across 100 stocks")
    if not BENCHMARK_CSV.exists():
        st.info("The 100-stock comparison has not been generated yet.")
        return
    df = _read(str(BENCHMARK_CSV))
    run_date = str(df["run_date"].iloc[0]) if "run_date" in df else ""
    v = df[df["status"] == "valued"].dropna(subset=["p50", "price", "analyst_mean"]).copy()
    if v.empty:
        st.info("No valued stocks in the comparison file.")
        return
    v["ours_vs_price"] = v["p50"] / v["price"] - 1
    v["analyst_vs_price"] = v["analyst_mean"] / v["price"] - 1
    v["ours_vs_analyst"] = v["p50"] / v["analyst_mean"] - 1
    ours_under, an_under = (v["ours_vs_price"] > 0).mean(), (v["analyst_vs_price"] > 0).mean()
    below_an = (v["ours_vs_analyst"] < 0).mean()
    agree = ((v["ours_vs_price"] > 0) == (v["analyst_vs_price"] > 0)).mean()
    med_gap = v["ours_vs_analyst"].median()

    st.caption(f"Run on {run_date} with the same steps as the app, except that FMP's financial "
               "statements are skipped (its free plan allows about 250 calls a day), so each stock "
               "compares Yahoo with SEC filings. 3,000 scenarios per stock.")
    c = st.columns(4)
    c[0].metric("Stocks valued", f"{len(v)} of {len(df)}")
    c[1].metric("Our median vs analyst mean", f"{med_gap:+.0%}", help="Median of (our P50 / analyst mean − 1).")
    c[2].metric("We are below analysts on", f"{below_an:.0%}")
    c[3].metric("Same verdict as analysts", f"{agree:.0%}",
                help="Both say undervalued, or both say overvalued, against today's price.")
    st.dataframe(pd.DataFrame({
        "": ["Our DCF (P50 vs price)", "Analysts (mean target vs price)"],
        "Undervalued": [f"{ours_under:.0%}", f"{an_under:.0%}"],
        "Overvalued": [f"{1 - ours_under:.0%}", f"{1 - an_under:.0%}"],
    }), hide_index=True, width='content')

    if below_an >= 0.7:
        lean = (f"The model is more conservative than analysts: it is below the analyst mean on "
                f"{below_an:.0%} of stocks, by a median of {abs(med_gap):.0%}.")
    elif below_an <= 0.3:
        lean = (f"The model is more optimistic than analysts: it is above the analyst mean on "
                f"{1 - below_an:.0%} of stocks, by a median of {abs(med_gap):.0%}.")
    else:
        lean = (f"The model has no consistent lean against analysts: it is below the analyst mean on "
                f"{below_an:.0%} of stocks and above on {1 - below_an:.0%}.")
    fmp_note = ""
    if "fmp_dcf" in v and v["fmp_dcf"].notna().sum() >= 10:
        f = v.dropna(subset=["fmp_dcf"])
        fmp_note = (f" Against FMP's DCF ({len(f)} stocks) the median gap is {(f['p50'] / f['fmp_dcf'] - 1).median():+.0%}, "
                    f"and FMP's DCF itself sits a median {(f['fmp_dcf'] / f['price'] - 1).median():+.0%} from the "
                    "price, so much of the gap to analysts is a DCF-versus-price-target gap.")
    st.markdown(f"{lean}{fmp_note} Analyst targets are 12-month price targets that lean on trading "
                "multiples, while this DCF only credits growth the company's history and the base rates support.")

    _render_calibration(v)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=v["analyst_vs_price"] * 100, y=v["ours_vs_price"] * 100, mode="markers+text", text=v["ticker"],
        textposition="top center", textfont=dict(size=9), marker=dict(size=8, color="steelblue"),
        hovertemplate="%{text}<br>Analysts %{x:+.0f}%<br>Our DCF %{y:+.0f}%<extra></extra>", showlegend=False))
    both = pd.concat([v["analyst_vs_price"], v["ours_vs_price"]]) * 100
    lim = [float(both.min()), float(both.max())]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="grey", dash="dash"),
                             name="Same as analysts", hoverinfo="skip"))
    fig.add_hline(y=0, line_color="crimson", line_width=1)
    fig.add_vline(x=0, line_color="crimson", line_width=1)
    fig.update_layout(title="Upside to fair value: our DCF median vs analyst mean target (one point per stock)",
                      xaxis_title="Analyst mean target vs price (%)", yaxis_title="Our DCF P50 vs price (%)",
                      height=520, margin=dict(t=50, b=40), legend=dict(orientation="h", yanchor="top", y=-0.15))
    st.plotly_chart(fig, width='stretch')
    st.caption("Points below the dashed line are stocks where our DCF is more conservative than analysts.")

    t = df.copy()
    for col in ("price", "p10", "p50", "p90", "analyst_mean", "fmp_dcf"):
        t[col] = t[col].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
    t["Our P50 vs analysts"] = (df["p50"] / df["analyst_mean"] - 1).map(lambda x: f"{x:+.0%}" if pd.notna(x) else "")
    if "implied_growth" in df:
        t["Growth the price implies"] = df["implied_growth"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
        t["Model growth"] = df["model_growth"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
    t = t.rename(columns={"ticker": "Ticker", "price": "Price", "p10": "Our P10", "p50": "Our P50",
                          "p90": "Our P90", "analyst_mean": "Analyst mean", "analyst_n": "Analysts",
                          "fmp_dcf": "FMP DCF", "status": "Status", "note": "Note"})
    cols = ["Ticker", "Price", "Our P10", "Our P50", "Our P90", "Analyst mean", "Analysts",
            "Our P50 vs analysts", "Model growth", "Growth the price implies", "FMP DCF", "Status", "Note"]
    with st.expander(f"All {len(df)} stocks" + (f" (run {run_date})" if run_date else "")):
        st.dataframe(t[[c for c in cols if c in t]], hide_index=True, width='stretch')

    _render_tracking()


def _render_tracking() -> None:
    """Weekly returns since the start date, grouped by the verdicts frozen on that date."""
    if not (PRICES_CSV.exists() and VERDICTS_CSV.exists()):
        st.subheader("How these stocks have done")
        st.info("Weekly price tracking has not started yet.")
        return
    frozen = _read(str(VERDICTS_CSV)).set_index("ticker")
    wide = _read(str(PRICES_CSV)).pivot_table(index="ticker", columns="date", values="close").sort_index(axis=1)
    dates = list(wide.columns)
    start = str(frozen["start_date"].iloc[0])
    st.subheader(f"How these stocks have done since {start}")
    group = frozen["verdict"]
    n_ours = int(group.isin(["Undervalued by both", "Undervalued by our DCF only"]).sum())
    st.markdown(
        f"Each stock's verdict was recorded on {start} and frozen in `dcf/data/tracking_verdicts.csv`: "
        f"undervalued means our DCF median, or the analyst mean target, was above the price that day. "
        f"The verdicts come from the model as it stood on {start}, before the October 2026 changes, and "
        f"later model changes do not alter them. A GitHub Action records Friday's close every Saturday. "
        f"Group returns are equal weighted, and SPY is the market.")
    st.markdown(
        f"**When we will judge it, and what would count.** The test is read at {EVAL_WEEKS[0]} and "
        f"{EVAL_WEEKS[1]} weeks. The question is whether the {n_ours} stocks our DCF called undervalued "
        f"beat the {len(group) - n_ours} it called overvalued. Single stocks typically move about 30% a "
        f"year, so with groups this size the gap between the two averages has a standard error of very "
        f"roughly 7 points at one year. We will count it as evidence for the model only if the "
        f"undervalued group leads by more than {EVIDENCE_GAP * 100:.0f} percentage points at {EVAL_WEEKS[1]} weeks. A "
        f"smaller gap either way is inconclusive. One year of one market is a weak test whatever the result.")

    rets = wide.div(wide[start], axis=0) - 1 if start in wide else wide * np.nan
    if len(dates) < 2:
        st.info(f"Only the starting prices from {start} are recorded so far. The first weekly close "
                "arrives on Saturday, and this section fills in from then.")
        return
    fig = go.Figure()
    for name, color in GROUPS:
        members = group[group == name].index.intersection(rets.index)
        if len(members):
            fig.add_trace(go.Scatter(x=dates, y=rets.loc[members].mean() * 100, mode="lines+markers",
                                     name=f"{name} ({len(members)})", line=dict(color=color, width=2),
                                     hovertemplate="%{x}: %{y:+.1f}%<extra></extra>"))
    if "SPY" in rets.index:
        fig.add_trace(go.Scatter(x=dates, y=rets.loc["SPY"] * 100, mode="lines", name="SPY (market)",
                                 line=dict(color="grey", dash="dash"), hovertemplate="%{x}: %{y:+.1f}%<extra></extra>"))
    fig.add_hline(y=0, line_color="grey", line_width=1)
    fig.update_layout(title=f"Average return since {start} by verdict", xaxis_title="Week",
                      yaxis_title="Return since start (%)", height=420, margin=dict(t=50, b=40),
                      legend=dict(orientation="h", yanchor="top", y=-0.2))
    st.plotly_chart(fig, width='stretch')

    latest = rets[dates[-1]]
    counts = group.value_counts()
    st.dataframe(pd.DataFrame(
        [{"Group": name, "Stocks": int(counts.get(name, 0)),
          "Average return": f"{latest.reindex(group[group == name].index).mean():+.1%}"} for name, _ in GROUPS]
        + ([{"Group": "SPY (market)", "Stocks": 1, "Average return": f"{latest['SPY']:+.1%}"}]
           if "SPY" in latest.index else [])), hide_index=True, width='content')

    table = (rets[dates[1:]] * 100).round(1).reindex(group.index)
    table.columns = [f"{d} (%)" for d in dates[1:]]
    table.insert(0, "Start price", frozen["start_price"].map("${:,.2f}".format))
    table.insert(0, "Verdict", group)
    order = {g: i for i, (g, _) in enumerate(GROUPS)}
    table = table.sort_values(["Verdict", table.columns[-1]],
                              key=lambda s: s.map(order) if s.name == "Verdict" else -s)
    with st.expander(f"Weekly change for each stock since {start}"):
        st.dataframe(table.rename_axis("Ticker").reset_index(), hide_index=True, width='stretch')
