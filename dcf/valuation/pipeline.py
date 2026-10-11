"""One entry point that fetches, reconciles and values a ticker.

The Streamlit app, the 100-stock benchmark script and the CLI all call
value_ticker(), so they run the same steps.  The only switch is use_fmp: FMP's
free plan allows about 250 calls a day and rejects many symbols, so the
benchmark turns it off.
"""
from __future__ import annotations

import logging

from .data import fetch_raw
from .gates import check_not_financial
from .montecarlo import run_valuation
from .reconcile import reconcile
from .result import ValuationResult
from .sources import fetch_edgar, fetch_fmp, yahoo_source_from_raw

log = logging.getLogger(__name__)


def value_ticker(ticker: str, n_sims: int = 10_000, use_fmp: bool = True
                 ) -> tuple[ValuationResult, dict[str, str]]:
    """Returns (result, source_status).  source_status maps each source to "ok",
    "skipped" or the reason it failed, so a dropped source is visible in the UI.
    Raises gates.NotApplicable for banks, insurers and brokers."""
    ticker = ticker.upper().strip()
    raw = fetch_raw(ticker)
    # Check before spending FMP calls on a company the model will not value.
    check_not_financial(ticker, raw.sector, raw.industry)

    sources, status = [yahoo_source_from_raw(raw)], {"Yahoo": "ok"}
    fetchers = [("FMP", fetch_fmp)] if use_fmp else []
    fetchers.append(("EDGAR", fetch_edgar))
    if not use_fmp:
        status["FMP"] = "skipped"
    for name, fetcher in fetchers:
        try:
            sources.append(fetcher(ticker))
            status[name] = "ok"
        except RuntimeError as e:      # both fetchers wrap every failure in RuntimeError
            status[name] = str(e)
            log.info("%s source %s failed: %s", ticker, name, e)

    if len(sources) >= 2:
        recon = reconcile(sources)
        recon.source_status = status
        result = run_valuation(ticker, n_sims, sigma_cross=recon.sigma_cross,
                               reconcile_result=recon, raw=raw)
    else:
        result = run_valuation(ticker, n_sims, raw=raw)
    return result, status
