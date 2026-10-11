"""
Record this week's closing price for every stock in the 100-stock analyst
comparison, plus SPY as the market benchmark.  Appends to
dcf/data/price_history.csv (date, ticker, close); the "Analyst estimates" tab
reads it to show how each stock has moved since the comparison was run.

The tickers and starting prices come from dcf/data/tracking_verdicts.csv,
which froze each stock's verdict on the start date, so rerunning the model
later never changes what is being tracked.

Run weekly by .github/workflows/track_prices.yml.  The first run also writes
the comparison's own prices as the starting row, so returns are measured from
the date the DCF and analyst verdicts were recorded.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

REPO = Path(__file__).resolve().parent.parent
VERDICTS = REPO / "dcf" / "data" / "tracking_verdicts.csv"
BENCHMARK = REPO / "dcf" / "data" / "analyst_benchmark.csv"
HISTORY = REPO / "dcf" / "data" / "price_history.csv"
MARKET = "SPY"


def _closes(tickers: list[str], **kw) -> pd.DataFrame:
    px = yf.download(tickers, progress=False, auto_adjust=False, **kw)["Close"]
    return px.dropna(how="all")


def main() -> None:
    frozen = pd.read_csv(VERDICTS)
    # Every stock in the original comparison is priced, including the few the model
    # could not value, so the record stays complete.
    bench = pd.read_csv(BENCHMARK)
    tickers = sorted(set(frozen["ticker"]) | set(bench["ticker"])) + [MARKET]

    if HISTORY.exists():
        hist = pd.read_csv(HISTORY)
    else:
        run_date = str(frozen["start_date"].iloc[0])
        start = frozen[["ticker", "start_price"]].rename(columns={"start_price": "close"})
        start.insert(0, "date", run_date)
        # SPY's starting price: its last close on or before the comparison date
        end = (pd.Timestamp(run_date) + pd.Timedelta(days=1)).date().isoformat()
        start_ = (pd.Timestamp(run_date) - pd.Timedelta(days=10)).date().isoformat()
        spy = _closes([MARKET], start=start_, end=end)[MARKET].dropna()
        start.loc[len(start)] = [run_date, MARKET, round(float(spy.iloc[-1]), 2)]
        hist = start

    px = _closes(tickers, period="7d")
    last_day = px.index[-1].date().isoformat()
    if last_day <= str(hist["date"].max()):
        print(f"{last_day} is already covered (latest recorded {hist['date'].max()})")
        hist.to_csv(HISTORY, index=False)
        return
    latest = px.iloc[-1].dropna()
    new = pd.DataFrame({"date": last_day, "ticker": latest.index, "close": latest.values.round(2)})
    missing = sorted(set(tickers) - set(latest.index))
    pd.concat([hist, new], ignore_index=True).to_csv(HISTORY, index=False)
    print(f"recorded {len(new)} closes for {last_day}" + (f"; missing {missing}" if missing else ""))


if __name__ == "__main__":
    main()
