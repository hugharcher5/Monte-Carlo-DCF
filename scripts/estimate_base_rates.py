"""
Estimate the base rates the DCF shrinks each company's history toward.

A company's average revenue growth over the last four years is a noisy guide to
its growth over the next few: fast growers usually slow down and shrinking
firms often recover.  Instead of assuming the trailing average persists, the
model forecasts it with a cross-sectional regression fitted on US companies:

    next-3-year average growth      = a + b x trailing average growth + c x size
    next-3-year average EBIT margin = a + b x trailing average EBIT margin + c x size

where size = log10(latest revenue in USD) - 9 (0 at $1bn of revenue).

The slope b (below 1) is the shrinkage toward the base rate.  The spread of
the regression residuals is the uncertainty about the company's future average,
which is what the Monte Carlo samples.  Residual variance is allowed to grow
with the sampling variance of the company's own trailing mean:

    residual^2 = v0 x exp(k x size) + v1 x (historical sd)^2 / (number of years)

Big companies are both slower growing and more predictable, so size enters
both equations.  Size is clipped to the range seen in the data when the model
applies the fit.

The same panel gives the correlations between growth, margin and CapEx-intensity
surprises, which the copula uses.

Data: the SEC XBRL company-facts files cached for the research trials
(research/mgmt_pit/cache/sec_facts, about 2,100 US small and mid caps).  Each
fiscal year uses the most recently filed 10-K figure.  Two versions are written:

    dcf/data/base_rates.json          all years (used by the live tool)
    dcf/data/base_rates_pre2015.json  only outcomes known by 31 Dec 2014
                                      (used by the S22 backtest, which starts
                                      in January 2015, so it has no look-ahead)

    venv/bin/python scripts/estimate_base_rates.py
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
FACTS_DIR = REPO / "research" / "mgmt_pit" / "cache" / "sec_facts"
PANEL_CACHE = REPO / "research" / "valuationcompare_S22" / "cache" / "annual_panel.pkl"
OUT_ALL = REPO / "dcf" / "data" / "base_rates.json"
OUT_PRE2015 = REPO / "dcf" / "data" / "base_rates_pre2015.json"

REVENUE = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
           "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"]
EBIT = ["OperatingIncomeLoss"]
CAPEX = ["PaymentsToAcquirePropertyPlantAndEquipment"]

TRAIL_YEARS = 4          # growth observations in the trailing window (the model uses up to 4)
MIN_TRAIL = 2            # the model needs at least 2
AHEAD = 3                # forecast target: average over the next 3 fiscal years
MIN_REVENUE = 20e6       # skip shells and tiny filers
CLIP_G = (-0.5, 1.0)
CLIP_M = (-0.5, 0.6)
CLIP_C = (0.0, 0.6)


def _annual(ug: dict, tags: list[str], combine_max: bool) -> pd.Series:
    """Full-year 10-K values per fiscal-year end; latest filing wins; across tags
    take the max (revenue) or the first tag that reports that year."""
    per_tag = {}
    for tag in tags:
        rows = ug.get(tag, {}).get("units", {}).get("USD", [])
        pts: dict = {}
        for r in rows:
            if r.get("form") not in ("10-K", "10-K/A") or "start" not in r:
                continue
            try:
                days = (pd.Timestamp(r["end"]) - pd.Timestamp(r["start"])).days
            except (KeyError, ValueError):
                continue
            if not 350 <= days <= 380:
                continue
            end = r["end"]
            if end not in pts or r.get("filed", "") >= pts[end][0]:
                pts[end] = (r.get("filed", ""), float(r["val"]))
        if pts:
            per_tag[tag] = pd.Series({pd.Timestamp(k): v[1] for k, v in pts.items()})
    if not per_tag:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(per_tag).sort_index()
    s = frame.max(axis=1) if combine_max else frame.bfill(axis=1).iloc[:, 0]
    return s.dropna()


def _load_one(path: Path) -> pd.DataFrame | None:
    try:
        with open(path) as f:
            ug = json.load(f).get("facts", {}).get("us-gaap", {})
    except (OSError, ValueError):
        return None
    rev = _annual(ug, REVENUE, True)
    if len(rev) < MIN_TRAIL + 1 + AHEAD:
        return None
    ebit = _annual(ug, EBIT, False)
    capex = _annual(ug, CAPEX, False).abs()
    df = pd.concat({"revenue": rev, "ebit": ebit, "capex": capex}, axis=1, sort=True)
    df.index = pd.DatetimeIndex(df.index)
    df = df[~df.index.year.duplicated(keep="last")]   # one row per fiscal year
    df["cik"] = int(path.stem)
    return df


def build_panel() -> pd.DataFrame:
    if PANEL_CACHE.exists():
        return pd.read_pickle(PANEL_CACHE)
    files = sorted(FACTS_DIR.glob("*.json"))
    print(f"reading {len(files)} company-facts files...", flush=True)
    with Pool(4) as pool:
        frames = [f for f in pool.imap_unordered(_load_one, files, chunksize=8) if f is not None]
    panel = pd.concat(frames).rename_axis("fy_end").reset_index()
    PANEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_pickle(PANEL_CACHE)
    return panel


def observations(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (company, anchor year): trailing stats and next-3-year outcomes."""
    rows = []
    for cik, g in panel.groupby("cik"):
        g = g.sort_values("fy_end").reset_index(drop=True)
        yrs = g["fy_end"].dt.year.to_numpy()
        rev = g["revenue"].to_numpy(float)
        ebit = g["ebit"].to_numpy(float)
        capex = g["capex"].to_numpy(float)
        growth = np.full(len(g), np.nan)
        ok = (yrs[1:] - yrs[:-1] == 1) & (rev[:-1] > 0)
        growth[1:][ok] = rev[1:][ok] / rev[:-1][ok] - 1
        pos = rev > 0
        margin = np.full(len(g), np.nan); margin[pos] = ebit[pos] / rev[pos]
        cxp = np.full(len(g), np.nan); cxp[pos] = np.clip(capex[pos] / rev[pos], *CLIP_C)
        for i in range(len(g)):
            if not rev[i] >= MIN_REVENUE or i + AHEAD >= len(g):
                continue
            if yrs[i + AHEAD] - yrs[i] != AHEAD:
                continue
            tg = growth[max(1, i - TRAIL_YEARS + 1):i + 1]
            tg = tg[np.isfinite(tg)]
            if len(tg) < MIN_TRAIL:
                continue
            fg = growth[i + 1:i + 1 + AHEAD]
            tm = margin[max(0, i - TRAIL_YEARS):i + 1]
            tm = tm[np.isfinite(tm)]
            fm = margin[i + 1:i + 1 + AHEAD]
            tc = cxp[max(0, i - TRAIL_YEARS):i + 1]
            tc = tc[np.isfinite(tc)]
            fc = cxp[i + 1:i + 1 + AHEAD]
            tgc = np.clip(tg, *CLIP_G)
            tmc = np.clip(tm, *CLIP_M)
            rows.append({
                "cik": cik, "size": np.log10(rev[i]) - 9.0, "anchor": g["fy_end"].iloc[i], "outcome_end": g["fy_end"].iloc[i + AHEAD],
                "n_trail": len(tg), "g_trail": tgc.mean(), "g_sd": tgc.std(ddof=1),
                "g_next": np.clip(fg, *CLIP_G).mean() if np.isfinite(fg).all() else np.nan,
                "n_m": len(tm),
                "m_trail": tmc.mean() if len(tm) >= 2 else np.nan,
                "m_sd": tmc.std(ddof=1) if len(tm) >= 2 else np.nan,
                "m_next": np.clip(fm, *CLIP_M).mean() if np.isfinite(fm).all() else np.nan,
                "c_trail": tc.mean() if len(tc) >= 2 else np.nan,
                "c_next": fc.mean() if np.isfinite(fc).all() else np.nan,
            })
    return pd.DataFrame(rows)


def _fit(x, y, sd, n, size):
    """y = a + b x + c size by OLS; then residual^2 = v0 exp(k size) + v1 sd^2/n
    by non-linear least squares (the exponential keeps the variance positive)."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(sd) & (n > 0) & np.isfinite(size)
    x, y, sd, n, size = x[m], y[m], sd[m], n[m], size[m]
    (a, b, c), *_ = np.linalg.lstsq(np.column_stack([np.ones_like(x), x, size]), y, rcond=None)
    resid = y - (a + b * x + c * size)
    se2 = sd ** 2 / n
    from scipy.optimize import curve_fit
    (v0, k, v1), _ = curve_fit(lambda X, v0, k, v1: v0 * np.exp(k * X[0]) + v1 * X[1],
                               np.vstack([size, se2]), resid ** 2, p0=[0.01, -0.3, 0.3],
                               bounds=([1e-6, -5.0, 0.0], [1.0, 5.0, 10.0]))
    fit = {"a": float(a), "b": float(b), "c": float(c),
           "v0": float(v0), "k": float(k), "v1": float(v1),
           "size_range": [float(np.percentile(size, 1)), float(np.percentile(size, 99))],
           "resid_sd": float(resid.std(ddof=3)), "n_obs": int(len(y)),
           "r2": float(1 - resid.var() / y.var())}
    return fit, resid, m


def estimate(obs: pd.DataFrame, label: str) -> dict:
    size = obs["size"].to_numpy()
    g, g_res, g_mask = _fit(obs["g_trail"].to_numpy(), obs["g_next"].to_numpy(),
                            obs["g_sd"].to_numpy(), obs["n_trail"].to_numpy(float), size)
    mg, m_res, m_mask = _fit(obs["m_trail"].to_numpy(), obs["m_next"].to_numpy(),
                             obs["m_sd"].to_numpy(), obs["n_m"].to_numpy(float), size)
    c_x, c_y = obs["c_trail"].to_numpy(), obs["c_next"].to_numpy()
    cm = np.isfinite(c_x) & np.isfinite(c_y)
    (ca, cb), *_ = np.linalg.lstsq(np.column_stack([np.ones(cm.sum()), c_x[cm]]), c_y[cm], rcond=None)
    c_res = np.full(len(obs), np.nan); c_res[cm] = c_y[cm] - (ca + cb * c_x[cm])
    gr = np.full(len(obs), np.nan); gr[g_mask] = g_res
    mr = np.full(len(obs), np.nan); mr[m_mask] = m_res

    def _corr(u, v):
        k = np.isfinite(u) & np.isfinite(v)
        return float(np.corrcoef(u[k], v[k])[0, 1]), int(k.sum())

    rho_gm, n1 = _corr(gr, mr)
    rho_gc, n2 = _corr(gr, c_res)
    rho_mc, n3 = _corr(mr, c_res)
    return {
        "label": label,
        "sample": f"{obs['cik'].nunique()} companies, anchor years {obs['anchor'].min().date()} to "
                  f"{obs['anchor'].max().date()}, outcomes up to {obs['outcome_end'].max().date()}",
        "target": f"average of the next {AHEAD} fiscal years",
        "growth": g, "margin": mg,
        "capex": {"a": float(ca), "b": float(cb), "n_obs": int(cm.sum())},
        "corr": {"growth_margin": rho_gm, "growth_capex": rho_gc, "margin_capex": rho_mc,
                 "n_obs": min(n1, n2, n3)},
    }


def main() -> None:
    panel = build_panel()
    obs = observations(panel).dropna(subset=["g_trail", "g_next"])
    print(f"{len(obs):,} company-years from {obs['cik'].nunique():,} companies")
    for out_path, cutoff, label in ((OUT_ALL, None, "all years"),
                                    (OUT_PRE2015, pd.Timestamp("2014-12-31"),
                                     "outcomes known by 2014-12-31")):
        o = obs if cutoff is None else obs[obs["outcome_end"] <= cutoff]
        est = estimate(o, label)
        out_path.write_text(json.dumps(est, indent=2) + "\n")
        print(json.dumps(est, indent=2))


if __name__ == "__main__":
    sys.exit(main())
