"""
S22 rerun (October 2026) on the current DCF model.

Same question, universe, window, portfolio rule and cost model as the July 2026
run (run.py / run_intersection.py), with these changes:

  * The DCF is the current model (dcf/valuation), with base rates fitted only
    on outcomes known by the end of 2014 and the risk-free rate published on
    or before each rebalance date.
  * Two Monte Carlo signals are added: the P50 margin of safety and the share
    of simulated scenarios above the price (1,000 scenarios per stock).
  * Banks, insurers and brokers (SIC 6000-6799) get no DCF score, as in the
    live tool, so they drop out of the common list for every signal.
  * Returns use Tiingo's adjusted close (splits and dividends) instead of the
    raw close.  A stock split inside a holding quarter made raw-close returns
    wrong; results/v2/split_check.csv lists every case where the two differ.
  * The benchmark is every stock in the common list held equally, rebalanced
    each quarter and charged the same trading costs as the portfolios.

Like the July run, it ranks only stocks that every signal can score.

    venv/bin/python research/valuationcompare_S22/rerun_v2.py [--quarters N] [--workers 6]
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("s22_run_orig", str(HERE / "run.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
GPI, DA = R.GPI, R.DA

OUT = HERE / "results" / "v2"
OUT.mkdir(parents=True, exist_ok=True)
SIGNALS = ["dcf", "dcf_mc", "dcf_pu", "pe", "ps", "pb"]
SPLIT_GAP = 0.20     # raw and adjusted quarterly returns more than 20 points apart

logging.basicConfig(level=logging.WARNING)


def adjusted_return(ticker: str, start: pd.Timestamp, end: pd.Timestamp, delist_date) -> tuple[float, float]:
    """(adjusted return, raw return) over the holding quarter, NaN when unavailable.
    A stock that delists inside the quarter is held to its last trading day."""
    stop = delist_date if pd.notna(delist_date) and delist_date < end else end
    out = []
    for frame, col in ((GPI._adj_slice(ticker, start, end + pd.Timedelta(days=5)), "adjClose"),
                       (GPI._get_price_slice(ticker, start, end + pd.Timedelta(days=5)), "close")):
        entry = frame[frame["date"] >= start]
        exit_ = frame[frame["date"] <= stop]
        if entry.empty or exit_.empty:
            out.append(np.nan)
            continue
        p0, p1 = float(entry.iloc[0][col]), float(exit_.iloc[-1][col])
        out.append(p1 / p0 - 1 if p0 > 0 else np.nan)
    return out[0], out[1]


def _score(args):
    ticker, cik, t, price, shares, beta, sic = args
    return ticker, DA.pit_dcf_signals(ticker, cik, t, price, shares, beta, sic)


def run(n_quarters: int | None, workers: int) -> None:
    print("[load] universe, prices and filings")
    tiingo = GPI.load_tiingo_tickers()
    pre = GPI.load_prefilter()
    ok = pre[pre["passed"]][["ticker", "cik", "sic"]]
    cik_sic = {r["ticker"]: (int(r["cik"]), r["sic"]) for _, r in ok.iterrows()}
    ciks = ok["cik"].dropna().astype(int).tolist()
    GPI.preload_all_prices()
    GPI.preload_all_adj_prices()
    GPI.preload_gp_facts(ciks)
    R.PF.preload_facts(ciks)
    delisted = {r["ticker"]: r["endDate"] for _, r in tiingo.iterrows() if pd.notna(r["endDate"])}

    # Warm the per-process caches before forking, so workers share them.
    from valuation import wacc as W
    from valuation.priors import BASE_RATES_PRE2015, load_base_rates
    W._fred_series(W.RF_SERIES["USD"])
    load_base_rates(BASE_RATES_PRE2015)

    dates = GPI.REBALANCE_DATES[: n_quarters or None]
    records = {s: [] for s in SIGNALS}
    bench, splits, snaps = [], [], []
    prev = {s: set() for s in SIGNALS + ["bench"]}
    pool = Pool(workers) if workers > 1 else None
    for i, t in enumerate(dates):
        t0 = time.time()
        end = GPI.REBALANCE_DATES[i + 1] if i + 1 < len(GPI.REBALANCE_DATES) else GPI.IS_END
        univ = GPI.build_universe(t, tiingo, cik_sic)
        if univ.empty:
            continue
        ratios = R.compute_ratio_signals(univ, t)
        betas = GPI.compute_betas(univ, t)
        jobs = [(r["ticker"], int(r["cik"]), t, r["price"], r["shares"], betas.get(r["ticker"], 1.0), r["sic"])
                for _, r in univ.iterrows()]
        scored = dict(pool.map(_score, jobs, chunksize=8) if pool else map(_score, jobs))
        for col, key in (("score_dcf", "mos_base"), ("score_dcf_mc", "mos_p50"), ("score_dcf_pu", "p_above")):
            ratios[col] = [scored.get(tk)[key] if scored.get(tk) else np.nan for tk in ratios["ticker"]]

        df = univ.merge(ratios.drop(columns=["mcap", "sic"]), on=["ticker", "cik"])
        adj, raw = zip(*(adjusted_return(tk, t, end, delisted.get(tk, pd.NaT)) for tk in df["ticker"]))
        df["raw_return"] = adj          # the cost model reads this column name
        df["close_return"] = raw
        gap = (df["raw_return"] - df["close_return"]).abs()
        for _, r in df[gap > SPLIT_GAP].iterrows():
            splits.append({"rebalance_date": t, "ticker": r["ticker"], "adjusted_return": r["raw_return"],
                           "raw_close_return": r["close_return"]})

        univ_gross = float(df["raw_return"].mean())
        inter = df.dropna(subset=[f"score_{s}" for s in SIGNALS] + ["raw_return"]).copy()
        snaps.append(inter[["ticker", "cik", "sic", "mcap", "raw_return"] + [f"score_{s}" for s in SIGNALS]]
                     .assign(rebalance_date=t))

        # Benchmark: every stock in the common list, equal weight, same costs.
        names = inter["ticker"].tolist()
        turn = len(set(names) - prev["bench"]) / max(len(names), 1) if prev["bench"] else 1.0
        prev["bench"] = set(names)
        b_net, b_gross, b_cost, _ = GPI._leg_return_detail(inter, names, {n: 1 / len(names) for n in names},
                                                           is_short=False, turnover=turn)
        bench.append({"rebalance_date": t, "ret_net": b_net, "ret_gross": b_gross, "cost": b_cost,
                      "turnover": turn, "n": len(names), "univ_gross": univ_gross, "n_universe": len(df)})

        line = []
        for s in SIGNALS:
            col = f"score_{s}"
            port = R.build_long_only_portfolio(inter, col)
            if not port["longs"]:
                continue
            longs = port["longs"]
            turn = len(set(longs) - prev[s]) / len(longs) if prev[s] else 1.0
            prev[s] = set(longs)
            net, gross, cost, _ = GPI._leg_return_detail(inter, longs, port["w"], is_short=False, turnover=turn)
            ic = stats.spearmanr(inter[col], inter["raw_return"])[0] if len(inter) >= 10 else np.nan
            records[s].append({"rebalance_date": t, "ret_net": net, "ret_gross": gross, "cost": cost,
                               "turnover": turn, "ic": ic, "n_long": len(longs), "n_universe": len(inter),
                               "bench_net": b_net, "univ_gross": univ_gross})
            line.append(f"{s}={net:+.1%}")
        print(f"[{i + 1:02d}/{len(dates)}] {t.date()} universe={len(df)} common={len(inter)} "
              f"bench={b_net:+.1%} " + " ".join(line) + f" ({time.time() - t0:.0f}s)", flush=True)
    if pool:
        pool.close()

    for s in SIGNALS:
        pd.DataFrame(records[s]).to_csv(OUT / f"s22v2_{s}.csv", index=False)
    pd.DataFrame(bench).to_csv(OUT / "s22v2_benchmark.csv", index=False)
    pd.DataFrame(splits).to_csv(OUT / "split_check.csv", index=False)
    pd.concat(snaps, ignore_index=True).to_csv(OUT / "s22v2_signal_snapshots.csv.gz", index=False)
    print(f"[saved] {OUT}  ({len(splits)} stock-quarters where raw and adjusted returns differ by >20 points)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quarters", type=int, default=None, help="only the first N quarters (for a quick test)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run(a.quarters, a.workers)
