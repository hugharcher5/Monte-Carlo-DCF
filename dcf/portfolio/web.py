"""Streamlit entry point (Streamlit Cloud runs this file).

    venv/bin/streamlit run dcf/portfolio/web.py

The page sections live in dcf/portfolio/app/; the model is in dcf/valuation/.
"""
import logging
import sys
import traceback
from pathlib import Path

_DCF_ROOT = Path(__file__).resolve().parent.parent      # .../Archer-Capital/dcf
_REPO_ROOT = _DCF_ROOT.parent
# Streamlit runs this file as a script, so the packages beside it are not importable
# until their folders are on the path.
for p in (_DCF_ROOT, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

from portfolio.app import analysts, summary  # noqa: E402
from valuation.gates import NotApplicable  # noqa: E402
from valuation.pipeline import value_ticker  # noqa: E402

log = logging.getLogger("web")
N_SIMS = 10_000

st.set_page_config(page_title="Monte Carlo DCF", layout="wide")
st.title("Monte Carlo DCF")
tab_valuation, tab_research = st.tabs(["Valuation", "Research"])


@st.cache_data(show_spinner=False, ttl=3600)
def cached_valuation(ticker: str):
    result, status = value_ticker(ticker, n_sims=N_SIMS)
    result.source_status = status
    return result, status


def _show_error(ticker: str, exc: Exception) -> None:
    st.error(f"**Valuation failed for {ticker}**: {exc}")
    # File and line only, so failures on the hosted app can be diagnosed without server logs.
    frames = traceback.extract_tb(exc.__traceback__)[-4:]
    with st.expander("Technical details"):
        st.code("\n".join(f"{Path(f.filename).name}:{f.lineno} in {f.name}: {f.line}" for f in frames)
                + f"\n{type(exc).__name__}: {exc}")
    log.exception("valuation of %s failed", ticker)


with tab_valuation:
    ticker_input = st.text_input("Ticker symbol", value="MSFT", key="val_ticker")
    if st.button("Run valuation", key="val_run"):
        ticker = ticker_input.upper().strip()
        st.session_state.pop("val_result", None)
        st.session_state.pop("val_source_status", None)
        if not ticker:
            st.warning("Enter a ticker symbol first.")
        else:
            try:
                with st.spinner(f"Valuing {ticker}: fetching data and running {N_SIMS:,} scenarios…"):
                    result, status = cached_valuation(ticker)
                st.session_state["val_result"] = result
                st.session_state["val_source_status"] = status
            except NotApplicable as exc:
                st.info(f"**Not applicable.** {exc}")
            except Exception as exc:   # show any failure in the page instead of a blank app
                _show_error(ticker, exc)

    if "val_result" in st.session_state:
        status = st.session_state.get("val_source_status", {})
        failed = {k: v for k, v in status.items() if v not in ("ok", "skipped")}
        if failed:
            st.warning(f"Only {len(status) - len(failed)} of {len(status)} data sources were available, "
                       "so fewer sources were compared. "
                       + "  ".join(f"**{k}**: {v}" for k, v in failed.items()))
        result = st.session_state["val_result"]
        sub_dcf, sub_analysts = st.tabs(["Our DCF", "Analyst estimates"])
        with sub_dcf:
            summary.render(result)
        with sub_analysts:
            with st.spinner("Fetching analyst price targets…"):
                benchmarks = analysts.cached_benchmarks(result.ticker)
            analysts.render_stock(result, benchmarks)
            st.divider()
            analysts.render_study()


with tab_research:
    from strategies.valuation_compare.page import render_valuation_compare_page

    render_valuation_compare_page()
