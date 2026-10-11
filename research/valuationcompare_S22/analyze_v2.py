"""
Statistics for the October 2026 S22 rerun (reads results/v2 from rerun_v2.py).

  * Return a year, Sharpe, worst fall and Calmar for each signal and for the
    equal-weight benchmark after costs.
  * Paired bootstrap of each DCF signal minus each ratio: resample the 24
    quarters with replacement 10,000 times, keeping each quarter's pair
    together.  Reports the annualised mean difference, a 95% interval and the
    share of resamples where the DCF did no better.
  * Factor alpha: quarterly excess return regressed on the Fama-French market,
    size and value factors plus momentum (Ken French data library, monthly
    factors compounded to each holding quarter).
  * Per-quarter Sharpe against the registry's deflated-Sharpe threshold.

    venv/bin/python research/valuationcompare_S22/analyze_v2.py
"""
from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V2 = HERE / "results" / "v2"
FF = HERE / "cache" / "ff"
SIGNALS = {"dcf": "DCF base case", "dcf_mc": "DCF Monte Carlo P50", "dcf_pu": "DCF share of scenarios above price",
           "pe": "P/E", "ps": "P/S", "pb": "P/B"}
DCF_SIGNALS, RATIOS = ["dcf", "dcf_mc", "dcf_pu"], ["pe", "ps", "pb"]
N_BOOT, SEED = 10_000, 7
DSR_THRESHOLD_Q = 0.3680   # registry threshold for N = 28 trials, quarterly


def _ff_monthly() -> pd.DataFrame:
    def read(name: str) -> pd.DataFrame:
        text = (FF / name).read_text()
        lines = text.splitlines()
        start = next(i for i, l in enumerate(lines) if l.strip().startswith(","))
        block = []
        for l in lines[start:]:
            if block and not l.strip()[:1].isdigit():
                break
            block.append(l)
        df = pd.read_csv(io.StringIO("\n".join(block)), index_col=0)
        df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m")
        return df.apply(pd.to_numeric) / 100.0
    f = read("F-F_Research_Data_Factors.csv").join(read("F-F_Momentum_Factor.csv"))
    f.columns = [c.strip() for c in f.columns]
    return f


def ff_quarterly(dates: pd.Series) -> pd.DataFrame:
    m = _ff_monthly()
    rows = []
    for d in pd.to_datetime(dates):
        q = m[(m.index >= d) & (m.index < d + pd.DateOffset(months=3))]
        rows.append((1 + q).prod() - 1)
    return pd.DataFrame(rows, index=pd.to_datetime(dates).values)


def metrics(r: pd.Series) -> dict:
    growth = np.concatenate([[1.0], (1 + r).cumprod().values])
    annual = growth[-1] ** (4 / len(r)) - 1
    worst = float((growth / np.maximum.accumulate(growth) - 1).min())
    return {"annual": float(annual), "sharpe": float(r.mean() / r.std() * 2), "worst_fall": worst,
            "calmar": float(annual / abs(worst)) if worst < 0 else None,
            "per_period_sr": float(r.mean() / r.std()), "end_value": float(growth[-1])}


def paired_bootstrap(a: pd.Series, b: pd.Series, rng) -> dict:
    d = (a.values - b.values)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[idx].mean(axis=1) * 4
    return {"mean_diff_annual": float(d.mean() * 4), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)), "p_no_better": float((means <= 0).mean()),
            "t_stat": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))}


def factor_alpha(r: pd.Series, f: pd.DataFrame) -> dict:
    y = r.values - f["RF"].values
    X = np.column_stack([np.ones(len(y)), f[["Mkt-RF", "SMB", "HML", "Mom"]].values])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    cov = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    names = ["alpha", "mkt", "smb", "hml", "mom"]
    out = {n: float(b) for n, b in zip(names, beta)}
    out.update({f"{n}_t": float(b / s) for n, b, s in zip(names, beta, se)})
    out["alpha_annual"] = float((1 + beta[0]) ** 4 - 1)
    out["r2"] = float(1 - resid.var() / y.var())
    return out


def main() -> None:
    data = {s: pd.read_csv(V2 / f"s22v2_{s}.csv", parse_dates=["rebalance_date"]) for s in SIGNALS}
    bench = pd.read_csv(V2 / "s22v2_benchmark.csv", parse_dates=["rebalance_date"])
    f = ff_quarterly(bench["rebalance_date"])
    out = {"commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                    text=True, cwd=HERE).stdout.strip(),
           "quarters": len(bench), "signals": {}, "bootstrap": {}}
    for s, d in data.items():
        m = metrics(d["ret_net"])
        m.update(mean_ic=float(d["ic"].mean()), ic_t=float(d["ic"].mean() / d["ic"].std(ddof=1) * np.sqrt(len(d))),
                 beat_bench=float((d["ret_net"] > d["bench_net"]).mean()), cost_bps=float(d["cost"].mean() * 4e4),
                 turnover=float(d["turnover"].mean()), n_long=float(d["n_long"].mean()),
                 n_common=float(d["n_universe"].mean()), clears_dsr=bool(m["per_period_sr"] > DSR_THRESHOLD_Q),
                 alpha_vs_bench_annual=float(((1 + d["ret_net"]).prod() / (1 + d["bench_net"]).prod()) ** (4 / len(d)) - 1),
                 **{f"ff_{k}": v for k, v in factor_alpha(d["ret_net"], f).items()})
        out["signals"][s] = m
    bm = metrics(bench["ret_net"])
    bm.update(cost_bps=float(bench["cost"].mean() * 4e4), n_common=float(bench["n"].mean()),
              **{f"ff_{k}": v for k, v in factor_alpha(bench["ret_net"], f).items()})
    out["benchmark_common_after_costs"] = bm
    out["universe_gross"] = metrics(bench["univ_gross"])
    rng = np.random.default_rng(SEED)
    for a in DCF_SIGNALS:
        for b in RATIOS:
            out["bootstrap"][f"{a}_minus_{b}"] = paired_bootstrap(data[a]["ret_net"], data[b]["ret_net"], rng)
    r = pd.DataFrame({s: d["ret_net"].values for s, d in data.items()})
    out["return_correlation"] = r.corr().round(3).to_dict()
    sc = V2 / "split_check.csv"
    out["split_cases"] = int(len(pd.read_csv(sc))) if sc.exists() and sc.stat().st_size > 1 else 0
    (V2 / "s22v2_metrics.json").write_text(json.dumps(out, indent=2))

    print(f"commit {out['commit']}, {out['quarters']} quarters, split cases {out['split_cases']}")
    for s, m in out["signals"].items():
        print(f"{SIGNALS[s]:36s} ann {m['annual']:+.3%} sharpe {m['sharpe']:.2f} worst {m['worst_fall']:.0%} "
              f"calmar {m['calmar']:.2f} IC {m['mean_ic']:+.3f} beat {m['beat_bench']:.0%} "
              f"FF alpha {m['ff_alpha_annual']:+.2%} (t {m['ff_alpha_t']:.2f})")
    print(f"{'Benchmark (common list, after costs)':36s} ann {bm['annual']:+.3%} sharpe {bm['sharpe']:.2f} "
          f"worst {bm['worst_fall']:.0%} calmar {bm['calmar']:.2f} FF alpha {bm['ff_alpha_annual']:+.2%}")
    for k, v in out["bootstrap"].items():
        print(f"{k:16s} diff {v['mean_diff_annual']:+.2%} 95% [{v['ci_low']:+.2%}, {v['ci_high']:+.2%}] "
              f"p(no better) {v['p_no_better']:.2f}")


if __name__ == "__main__":
    main()
