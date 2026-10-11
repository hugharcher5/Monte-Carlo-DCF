"""
Value 100 stocks with the live Monte Carlo DCF and compare each with analyst
price targets and FMP's free DCF.  Writes dcf/data/analyst_benchmark.csv, which
the "Analyst estimates" tab of the app reads.

    venv/bin/python scripts/benchmark_vs_analysts.py            # full run
    venv/bin/python scripts/benchmark_vs_analysts.py --no-fmp   # keep FMP values from the last CSV

Each stock goes through the same pipeline as the app (valuation.pipeline), with
one difference: FMP's free plan allows 250 calls a day and covers only some
symbols, so financial statements come from Yahoo and SEC EDGAR only, and the
run spends at most one FMP call per ticker, on FMP's own DCF.

The verdicts used by the weekly price tracking are frozen separately in
dcf/data/tracking_verdicts.csv, so rerunning this script never changes them.
"""
import argparse
import contextlib
import io
import os
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "dcf")]
OUT = REPO / "dcf" / "data" / "analyst_benchmark.csv"
N_SIMS = 3_000

TICKERS = """
AAPL MSFT NVDA GOOGL AMZN META TSLA AVGO ORCL ADBE CRM AMD INTC CSCO IBM QCOM TXN
KO PEP PG WMT COST MCD NKE SBUX HD LOW TGT DIS NFLX
JNJ PFE MRK ABBV LLY UNH AMGN TMO ABT
CAT DE BA GE HON UPS LMT MMM UNP
XOM CVX COP SLB OXY
LIN NEE DUK SO DOW NEM FCX
T VZ TMUS
JPM BAC GS V MA BRK-B
ETSY ROKU PLTR SNOW UBER ABNB DOCU ZM CROX DECK POOL
ASML TSM NVO SAP TM SONY BABA SHOP UL BP
RIVN LCID
ADP INTU NOW PANW CMG ORLY AZO ISRG
""".split()


def value_one(args) -> dict:
    ticker, use_fmp = args
    from valuation.pipeline import value_ticker
    from valuation.gates import NotApplicable
    from valuation.analysts import analyst_targets, fmp_dcf

    row = {"ticker": ticker}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            r, _ = value_ticker(ticker, n_sims=N_SIMS, use_fmp=False)
        row.update(price=r.current_price_usd)
        if r.dcf_applicable:
            g = r.implied.get("revenue_growth")
            row.update(status="valued", p10=r.p10, p50=r.p50, p90=r.p90,
                       pct_above_price=r.pct_undervalued, model_growth=r.assumptions.revenue_growth,
                       implied_growth=g, flags=len(r.flags),
                       note="unproven company" if r.unproven else "")
        else:
            row.update(status="not applicable", note="terminal-year free cash flow is negative")
    except NotApplicable:
        row.update(status="not applicable", note="bank, insurer or broker: FCFF DCF does not apply")
    except Exception as e:      # keep going: one bad ticker must not stop a 100-stock run
        row.update(status="failed", note=str(e).split(". ")[0][:160])

    for _ in range(3):   # Yahoo occasionally returns an empty frame under load
        try:
            targets, summary = analyst_targets(ticker)
        except Exception:
            targets, summary = pd.DataFrame(), {}
        if not targets.empty:
            break
    if not targets.empty:
        row.update(analyst_mean=targets["Target"].mean(), analyst_median=targets["Target"].median(),
                   analyst_sd=targets["Target"].std(ddof=1), analyst_n=len(targets))
    elif summary.get("mean"):
        # No firm-by-firm targets in the last year: fall back to Yahoo's consensus
        row.update(analyst_mean=summary["mean"], analyst_median=summary.get("median"))
    row.setdefault("price", summary.get("current"))
    if use_fmp:
        try:
            row["fmp_dcf"] = fmp_dcf(ticker)
        except Exception:
            row["fmp_dcf"] = None
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fmp", action="store_true", help="reuse FMP DCF values from the existing CSV")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")

    rows = []
    with Pool(4) as pool:
        for row in pool.imap_unordered(value_one, [(t, not args.no_fmp) for t in TICKERS]):
            rows.append(row)
            print(f"{row['ticker']:6s} {row.get('status')}", flush=True)

    df = pd.DataFrame(rows).set_index("ticker").reindex(TICKERS)
    if args.no_fmp and OUT.exists():
        df["fmp_dcf"] = pd.read_csv(OUT).set_index("ticker")["fmp_dcf"].reindex(df.index)
    df["run_date"] = date.today().isoformat()
    cols = ["price", "p10", "p50", "p90", "pct_above_price", "model_growth", "implied_growth",
            "analyst_mean", "analyst_median", "analyst_sd", "analyst_n", "fmp_dcf", "flags",
            "status", "note", "run_date"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = df.reindex(columns=cols)
    df[["model_growth", "implied_growth"]] = df[["model_growth", "implied_growth"]].round(4)
    money = ["price", "p10", "p50", "p90", "pct_above_price", "analyst_mean", "analyst_median",
             "analyst_sd", "fmp_dcf"]
    df[money] = df[money].round(2)
    df.rename_axis("ticker").to_csv(OUT)
    print(f"wrote {OUT} ({(df['status'] == 'valued').sum()} valued)")


if __name__ == "__main__":
    main()
