"""Plotly charts for the valuation page."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import scipy.stats


def value_density(r) -> go.Figure:
    """Smoothed distribution of simulated values, with the price and percentiles marked.
    The shaded green area is the share of scenarios above the price."""
    sims, price = r.sims, r.current_price_usd
    p1, p99 = np.percentile(sims, [1, 99])
    x = np.linspace(p1, p99, 300)
    kde = scipy.stats.gaussian_kde(sims)
    y = kde(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="tozeroy",
                             fillcolor="rgba(70, 130, 180, 0.25)", line=dict(color="steelblue", width=2),
                             hovertemplate="$%{x:,.2f}<extra></extra>", name="Simulated values"))
    edge = max(p1, min(p99, price))
    above = x >= edge
    if above.any():
        fig.add_trace(go.Scatter(x=np.concatenate([[edge], x[above]]), y=np.concatenate([kde([edge]), y[above]]),
                                 mode="lines", fill="tozeroy", fillcolor="rgba(46, 204, 113, 0.22)",
                                 line=dict(width=0), hoverinfo="skip", showlegend=False))
    for xv, color, width, dash, label, pos in (
        (price, "crimson", 2, None, f"Market ${price:,.2f}", "top right"),
        (r.p50, "seagreen", 2, "dash", f"P50 ${r.p50:,.2f}", "top left"),
        (r.p10, "darkorange", 1.5, "dot", f"P10 ${r.p10:,.2f}", "bottom right"),
        (r.p90, "darkorange", 1.5, "dot", f"P90 ${r.p90:,.2f}", "top right"),
    ):
        fig.add_vline(x=xv, line_color=color, line_width=width, line_dash=dash,
                      annotation_text=label, annotation_position=pos)
    fig.update_layout(
        title=f"{r.ticker}: value per share across {r.n_valid:,} simulated scenarios",
        xaxis_title="Value per share (USD)", yaxis_title="Density (smoothed)",
        showlegend=False, height=420, margin=dict(t=50, b=40),
    )
    return fig


def pert_vs_draws(samps: np.ndarray, lo: float, mode: float, hi: float, label: str,
                  is_pct: bool = True) -> go.Figure:
    """The intended PERT curve against the smoothed actual draws."""
    p005, p995 = np.percentile(samps, [0.5, 99.5])
    x_lo, x_hi = min(p005, lo), max(p995, hi)
    pad = max((x_hi - x_lo) * 0.08, 1e-6)
    x = np.linspace(x_lo - pad, x_hi + pad, 400)
    y_kde = scipy.stats.gaussian_kde(samps)(x)
    if hi - lo > 1e-12:
        a = 1.0 + 4.0 * (mode - lo) / (hi - lo)
        b = 1.0 + 4.0 * (hi - mode) / (hi - lo)
        y_pert = scipy.stats.beta.pdf(np.clip((x - lo) / (hi - lo), 0, 1), a, b) / (hi - lo)
    else:
        y_pert = np.zeros_like(x)
    fmt = ".1%" if is_pct else ".3s"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_pert, mode="lines", name="Intended PERT",
                             line=dict(color="#4a9eff", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=y_kde, mode="lines", fill="tozeroy", name="Actual draws (smoothed)",
                             fillcolor="rgba(70,130,180,0.18)", line=dict(color="steelblue", width=2)))
    fig.add_vline(x=mode, line_color="gold", line_dash="dot", annotation_text="mode",
                  annotation_position="top left", annotation_font_size=11)
    fig.update_layout(title=dict(text=f"{label}: intended vs {len(samps):,} actual draws", font_size=13),
                      xaxis_title=label, yaxis_title="Density", xaxis_tickformat=fmt, height=300,
                      margin=dict(t=40, b=36, l=46, r=16),
                      legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font_size=11))
    return fig


def corr_heatmap(matrix: np.ndarray, names: list[str], title: str) -> go.Figure:
    n = len(names)
    fig = go.Figure(go.Heatmap(
        z=matrix, x=names, y=names, text=[[f"{matrix[i, j]:.2f}" for j in range(n)] for i in range(n)],
        texttemplate="%{text}", textfont=dict(size=11), colorscale="RdBu", zmin=-1, zmax=1,
        colorbar=dict(thickness=10, len=0.85, tickformat=".1f"),
    ))
    fig.update_layout(title=dict(text=title, font_size=13), yaxis=dict(autorange="reversed"),
                      height=290, margin=dict(t=38, b=8, l=100, r=16))
    return fig
