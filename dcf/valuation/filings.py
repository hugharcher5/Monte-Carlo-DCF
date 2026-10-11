"""
Build the model's inputs from official SEC filings first, Yahoo Finance second.

fetch_raw() reads everything from Yahoo Finance.  apply_filing_data() then
replaces each field with the SEC EDGAR (10-K / 10-Q) figure wherever the filing
data is present, current and consistent, and records where every input came
from and why, so the UI can show it.

Fields fall back to Yahoo when:
  * the company does not file with the SEC under US GAAP (foreign 20-F filers),
  * the company does not tag that line in its XBRL, or stopped tagging it,
  * the tagged figure is inconsistent with the reported total (a sign that the
    company used an unusual tag or split the line across several tags).
Market data (price, market cap, beta, exchange rate) is never in filings.
"""

from __future__ import annotations
import math

import pandas as pd

YEARS: int = 5                 # annual statements used (→ 4 years of growth)
DATE_TOLERANCE_DAYS: int = 20  # fiscal-year ends from two sources count as the same year

# Largest gap (vs Yahoo, latest year) before a filing figure is treated as a
# tagging problem rather than a genuine number.  Revenue definitions agree
# closely across providers, so its tolerance is tight.
_MAX_GAP = {"revenue": 0.15}
_DEFAULT_MAX_GAP = 0.50

EDGAR = "SEC EDGAR"
YAHOO = "Yahoo Finance"

# Display labels for the provenance table, in display order.
FIELD_LABELS: dict[str, str] = {
    "revenue":          "Revenue",
    "ebit":             "Operating income (EBIT)",
    "pretax_income":    "Pre-tax income",
    "tax_provision":    "Income tax expense",
    "interest_expense": "Interest expense",
    "da":               "Depreciation & amortisation",
    "capex":            "Capital expenditure",
    "sbc":              "Stock-based compensation",
    "nwc":              "Working capital (current assets & liabilities)",
    "diluted_shares":   "Diluted share count",
    "total_debt":       "Total debt",
    "cash":             "Cash & equivalents",
    "price":            "Share price",
    "market_cap":       "Market capitalisation",
    "beta":             "Beta",
    "fx_rate":          "Exchange rate",
    "industry":         "Industry classification",
}

_NOT_IN_FILINGS = {
    "price":      "Market data. Filings do not contain a live share price.",
    "beta":       "Market data. Beta is estimated from share-price history, which filings do not contain.",
    "fx_rate":    "Market data. Live exchange rates are not in filings.",
    "industry":   "Yahoo's industry label is used to pick the industry margin for unproven companies.",
}


def _align(series: pd.Series, ref: pd.DatetimeIndex) -> pd.Series:
    """Map a series onto reference fiscal-year-end dates (nearest within tolerance)."""
    if series.empty or len(ref) == 0:
        return pd.Series(dtype=float)
    out = {}
    for d in ref:
        gaps = abs(series.index - d)
        i = int(gaps.argmin())
        if gaps[i].days <= DATE_TOLERANCE_DAYS:
            out[d] = float(series.iloc[i])
    return pd.Series(out, dtype=float).sort_index()


def _value_at(series: pd.Series, d) -> float:
    s = _align(series, pd.DatetimeIndex([d]))
    return float(s.iloc[0]) if not s.empty else float("nan")


def _gap(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)) or abs(b) < 1e-9:
        return float("nan")
    return abs(a - b) / abs(b)


MAX_SCALE_GAP: float = 10.0     # a count >10x away from Yahoo's is a scaling error
TSM_MAX_GAP: float = 0.25       # TSM count must sit within 25% of the next-best count
STALE_DAYS: int = 400           # option data older than this is ignored


def _plausible(count: float, yahoo: float) -> bool:
    if not (math.isfinite(count) and count > 0):
        return False
    return not (yahoo > 0 and not 1.0 / MAX_SCALE_GAP < count / yahoo < MAX_SCALE_GAP)


def tsm_diluted_shares(basic: float, price: float, options: float, strike: float,
                       rsus: float) -> tuple[float, float]:
    """Treasury-stock method: basic shares plus in-the-money options net of the
    shares their exercise cash could buy back, plus unvested RSUs.
    Returns (diluted shares, dilution)."""
    dil = 0.0
    if math.isfinite(options) and options > 0 and math.isfinite(strike) and price > 0:
        dil += options * max(0.0, 1.0 - strike / price)
    if math.isfinite(rsus) and rsus > 0:
        dil += rsus
    return basic + dil, dil


def _apply_share_count(raw, pref, edgar, edgar_reason, mark, pref_detail) -> bool:
    """Pick the share count, in order of preference:

    1. Treasury-stock-method diluted shares from the filings: cover-page basic
       shares plus in-the-money options (net of buy-backs at today's price)
       plus unvested RSUs.
    2. The latest 10-Q/10-K diluted weighted-average count (also TSM-based,
       but averaged over the quarter at the quarter's average price).
    3. Yahoo's implied shares outstanding (basic, all share classes).

    Any filing count more than 10x away from Yahoo's is treated as mis-scaled
    (MCD tags its weighted-average count in millions: 712.3).
    Returns True when the count came from the filings.
    """
    yahoo = raw.diluted_shares
    price_local = raw.current_price_usd / raw.fx_rate if raw.fx_rate > 0 else raw.current_price_usd
    notes = []
    wavg = pref.diluted_shares if pref is not None else float("nan")
    wavg_ok = _plausible(wavg, yahoo)
    if pref is not None and math.isfinite(wavg) and wavg > 0 and not wavg_ok:
        notes.append(f"The filing's weighted-average count ({wavg:,.0f}) is more than 10x away "
                     f"from Yahoo's ({yahoo:,.0f}), so it looks mis-scaled and is not used.")

    if edgar is not None and _plausible(edgar.basic_shares, yahoo):
        today = pd.Timestamp(getattr(raw, "valuation_date", None) or pd.Timestamp.today())
        opt_date = edgar.options_date
        fresh = opt_date is not None and (today - pd.Timestamp(opt_date)).days <= STALE_DAYS
        options = edgar.options_outstanding if fresh else float("nan")
        rsus = edgar.rsus_unvested if fresh else float("nan")
        count, dil = tsm_diluted_shares(edgar.basic_shares, price_local, options,
                                        edgar.options_strike, rsus)
        check = wavg if wavg_ok else yahoo
        if not check > 0 or abs(count / check - 1.0) <= TSM_MAX_GAP:
            raw.diluted_shares = float(count)
            raw.shares_method = "treasury-stock method"
            when = pd.Timestamp(edgar.basic_shares_date).date() if edgar.basic_shares_date is not None else "latest"
            detail = (f"Treasury-stock method: {edgar.basic_shares:,.0f} basic shares (cover page, {when}) "
                      f"+ {dil:,.0f} from options and RSUs at today's price")
            if not fresh:
                detail += " (no current option or RSU data in the filings, so no dilution added)"
            mark("diluted_shares", EDGAR, detail + ("" if not notes else ". " + " ".join(notes)))
            return True
        notes.append(f"The treasury-stock count ({count:,.0f}) is more than {TSM_MAX_GAP:.0%} away "
                     "from the reported diluted count, so it is not used.")

    if wavg_ok:
        raw.diluted_shares = float(wavg)
        raw.shares_method = "diluted weighted average"
        src = EDGAR if pref.source_name == "EDGAR" else pref.source_name
        mark("diluted_shares", src, pref_detail(pref, "shares") + ", diluted weighted-average shares"
             + ("" if not notes else ". " + " ".join(notes)))
        return pref.source_name == "EDGAR"

    raw.shares_method = "Yahoo implied shares outstanding (basic, all share classes)"
    mark("diluted_shares", YAHOO, " ".join(notes) or edgar_reason or
         "Share count not found in the filings. Yahoo's implied shares outstanding is a basic "
         "count, so option and RSU dilution is not included.")
    return False


def apply_filing_data(raw, reconcile_result) -> dict:
    """
    Overwrite fields on `raw` (a RawData from fetch_raw) with SEC filing data
    where usable.  Returns provenance: field → {"source": ..., "detail": ...}.
    """
    sources = list(getattr(reconcile_result, "sources", []) or [])
    edgar = next((s for s in sources if s.source_name == "EDGAR"), None)
    prov: dict[str, dict] = {}

    def mark(field, source, detail):
        prov[field] = {"source": source, "detail": detail}

    if edgar is not None and (raw.currency or "USD").upper() != "USD":
        edgar_reason = (f"The company reports in {raw.currency}; its SEC filings are not in "
                        "US-GAAP dollar form, so Yahoo's figures are used throughout.")
        edgar = None
    elif edgar is None:
        status = getattr(reconcile_result, "source_status", {}) if reconcile_result else {}
        why = status.get("EDGAR") if isinstance(status, dict) else None
        if why and "IFRS" in why:
            edgar_reason = ("The company files under IFRS (Form 20-F), whose tags differ from "
                            "US GAAP, so Yahoo's figures are used throughout.")
        else:
            edgar_reason = ("SEC EDGAR data was unavailable for this ticker (not an SEC "
                            "filer, or the request failed), so Yahoo's figures are used.")
    else:
        edgar_reason = ""

    # ── Income statement and cash-flow lines ─────────────────────────────────
    yahoo_series = {
        "revenue": raw.revenue, "ebit": raw.ebit, "pretax_income": raw.pretax_income,
        "tax_provision": raw.tax_provision, "interest_expense": raw.interest_expense,
        "da": raw.da, "capex": raw.capex, "sbc": raw.sbc,
    }
    edgar_series = {} if edgar is None else {
        "revenue": edgar.revenue, "ebit": edgar.ebit, "pretax_income": edgar.pretax_income,
        "tax_provision": edgar.tax_provision, "interest_expense": edgar.interest_expense,
        "da": edgar.dep_amort, "capex": edgar.capex, "sbc": edgar.sbc,
    }
    tags = getattr(edgar, "tags_used", {}) if edgar is not None else {}
    tag_key = {"da": "dep_amort"}

    # Revenue sets the fiscal years every other line is aligned to.
    use_edgar_rev = False
    if edgar is not None:
        e_rev, y_rev = edgar.revenue, raw.revenue
        if len(e_rev) < 2:
            rev_reason = "Revenue is not tagged in a usable way in the company's XBRL filings."
        elif not y_rev.empty and e_rev.index[-1] < y_rev.index[-1] - pd.Timedelta(days=DATE_TOLERANCE_DAYS):
            rev_reason = ("The latest 10-K in the XBRL data is older than the latest year "
                          "Yahoo reports.")
        else:
            g = _gap(float(e_rev.iloc[-1]), _value_at(y_rev, e_rev.index[-1]))
            if math.isfinite(g) and g > _MAX_GAP["revenue"]:
                rev_reason = (f"The tagged revenue differs from the reported total by {g:.0%}, "
                              "which usually means the company tags revenue by segment or "
                              "product line rather than as one total.")
            else:
                use_edgar_rev = True
                rev_reason = ""

    if use_edgar_rev:
        ref = edgar.revenue.index[-YEARS:]
        raw.revenue = edgar.revenue.iloc[-YEARS:].astype(float)
        mark("revenue", EDGAR, f"10-K, XBRL tag {tags.get('revenue', '')}")
        for f in ("ebit", "pretax_income", "tax_provision", "interest_expense", "da", "capex", "sbc"):
            e = _align(edgar_series[f], ref)
            y = _align(yahoo_series[f], ref)
            latest = ref[-1]
            reason = ""
            if e.empty or latest not in e.index:
                if f == "da":
                    reason = ("The company does not report D&A as a single tagged line; it "
                              "splits it across several tags, and adding those up misses "
                              "untagged items. Yahoo's cash-flow-statement total is used.")
                elif f == "ebit":
                    reason = ("Operating income is not tagged in the filings (common for banks "
                              "and insurers, which have no operating-income line).")
                elif not edgar_series[f].empty:
                    reason = ("The company stopped tagging this line in its recent filings, so "
                              "the filing data is out of date.")
                else:
                    reason = "This line is not tagged in the company's XBRL filings."
            else:
                g = _gap(float(e[latest]), float(y.get(latest, float("nan"))))
                small = (abs(float(y.get(latest, 0.0))) < 0.01 * abs(float(raw.revenue.iloc[-1])))
                if math.isfinite(g) and g > _MAX_GAP.get(f, _DEFAULT_MAX_GAP) and not small:
                    reason = (f"The tagged figure differs from the reported line by {g:.0%}, a "
                              "sign the company uses an unusual tag or splits the line, so "
                              "Yahoo's reported figure is used.")
            if reason:
                setattr(raw, f, y)
                mark(f, YAHOO, reason)
            else:
                setattr(raw, f, e)
                mark(f, EDGAR, f"10-K, XBRL tag {tags.get(tag_key.get(f, f), '')}")
    else:
        reason = edgar_reason or (rev_reason + " To keep every line on the same fiscal "
                                  "years, the income and cash-flow statements all come from Yahoo.")
        for f in yahoo_series:
            mark(f, YAHOO, reason)

    # ── Working capital: fiscal-year-end balance sheet ───────────────────────
    nwc_items = None
    if use_edgar_rev:
        vals = (edgar.current_assets, edgar.current_liabilities, edgar.cash_fy)
        if all(math.isfinite(v) for v in vals):
            nwc_items = vals
    if nwc_items:
        raw.current_assets, raw.current_liabilities, raw.cash_nwc = nwc_items
        raw.current_debt = edgar.current_debt if math.isfinite(edgar.current_debt) else 0.0
        mark("nwc", EDGAR, f"10-K balance sheet at {raw.revenue.index[-1].date()}")
    else:
        raw.cash_nwc = raw.cash
        mark("nwc", YAHOO, edgar_reason or ("Current assets or liabilities are not tagged "
                                            "(banks and insurers do not split them), so "
                                            "Yahoo's balance sheet is used."))

    # ── Bridge items: latest balance sheet (EDGAR > FMP > Yahoo) ─────────────
    pref = getattr(reconcile_result, "preferred", None)
    if pref is not None and (pref.currency or "").upper() != (raw.currency or "").upper():
        pref = None

    def _pref_detail(p, what):
        if p.source_name == "EDGAR":
            d = p.shares_date if what == "shares" else p.balance_date
            when = f" at {pd.Timestamp(d).date()}" if d is not None else ""
            return f"latest 10-Q/10-K{when}"
        return f"{p.source_name} (SEC data unavailable for this field)"

    shares_from_filing = _apply_share_count(raw, pref, edgar, edgar_reason, mark, _pref_detail)

    if (pref is not None and getattr(pref, "debt_ex_leases", True)
            and math.isfinite(pref.total_debt) and pref.total_debt >= 0):
        raw.total_debt = float(pref.total_debt)   # already standardised by the source fetcher
        mark("total_debt", EDGAR if pref.source_name == "EDGAR" else pref.source_name,
             _pref_detail(pref, "debt") + "; bonds, loans, commercial paper and finance leases "
             "(operating leases excluded)")
    else:
        raw.total_debt = max(0.0, raw.total_debt - getattr(raw, "op_lease_liab", 0.0))
        mark("total_debt", YAHOO, (edgar_reason or "Debt not found in the filings.") +
             " Operating lease liabilities are removed from Yahoo's total.")

    if pref is not None and math.isfinite(pref.cash) and pref.cash >= 0:
        raw.cash = float(pref.cash)
        mark("cash", EDGAR if pref.source_name == "EDGAR" else pref.source_name,
             _pref_detail(pref, "cash"))
    else:
        mark("cash", YAHOO, edgar_reason or "Cash is not tagged in the filings (common for banks).")

    # ── Market data: always Yahoo ────────────────────────────────────────────
    if shares_from_filing and raw.current_price_usd > 0:
        # Keep market cap consistent with the share count the model divides by.
        raw.market_cap_usd = raw.current_price_usd * raw.diluted_shares
        raw.market_cap_local = (raw.market_cap_usd / raw.fx_rate) if raw.fx_rate > 0 else raw.market_cap_usd
        mark("market_cap", YAHOO, "Yahoo share price × filing share count.")
    else:
        if not raw.market_cap_usd > 0 and raw.current_price_usd > 0 and raw.diluted_shares > 0:
            raw.market_cap_usd = raw.current_price_usd * raw.diluted_shares
            raw.market_cap_local = (raw.market_cap_usd / raw.fx_rate) if raw.fx_rate > 0 else raw.market_cap_usd
        mark("market_cap", YAHOO, "Market data. Filings do not contain a live market value.")
    for f, why in _NOT_IN_FILINGS.items():
        if f == "fx_rate" and (raw.currency or "USD").upper() == "USD":
            continue
        mark(f, YAHOO, why)

    if not raw.current_price_usd > 0:
        raise ValueError(f"No share price available for {raw.ticker} from Yahoo (quote or price history).")
    if not raw.diluted_shares > 0:
        raise ValueError("No share count available from any data source (Yahoo, FMP or SEC EDGAR).")

    return {k: prov[k] for k in FIELD_LABELS if k in prov}
