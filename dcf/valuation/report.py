"""Terminal tables for the CLI scripts.  The model code itself never prints."""
from __future__ import annotations


def _sec(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def print_drivers(r) -> None:
    d, fi = r.drivers, r.forecast_inputs
    _sec(f"HISTORICAL DRIVERS — {r.ticker}  ({d.years_used} years of growth)")
    if d.hist_df is not None:
        cols = ["revenue_growth", "ebit_margin", "tax_rate", "da_pct", "capex_pct", "sbc_pct"]
        df = d.hist_df[[c for c in cols if c in d.hist_df]].copy()
        df.index = [str(i.year) for i in df.index]
        print(df.T.map(lambda x: f"{x:.1%}").to_string())
    print(f"\n  Revenue growth: trailing {d.revenue_growth:.2%} (year-to-year sd {d.std_revenue_growth:.2%})")
    print(f"  EBIT margin   : trailing {d.ebit_margin:.2%} (best {d.best_ebit_margin:.2%})")
    if fi is not None:
        print(f"  Forecast growth (shrunk) {fi.revenue_growth:.2%} ± {fi.revenue_growth_sd:.2%}")
        print(f"  Forecast margin (shrunk) {fi.ebit_margin:.2%} -> {fi.target_margin:.2%} ± {fi.margin_sd:.2%}")
    for f in d.flags:
        print(f"  ! {f.strip()}")


def print_wacc(r) -> None:
    w = r.wacc_result
    _sec(f"WACC — {r.ticker}")
    print(f"  Risk-free rate : {w.rf:.3%} ({w.rf_source})")
    print(f"  Beta raw / adj : {w.beta_raw:.3f} / {w.beta_adj:.3f}")
    print(f"  ERP            : {w.erp:.2%}")
    print(f"  Cost of equity : {w.cost_of_equity:.3%}")
    print(f"  Rating, spread : {w.implied_rating}, {w.credit_spread:.3%}")
    print(f"  Cost of debt   : {w.cost_of_debt_pretax:.3%} pre-tax, {w.cost_of_debt_aftertax:.3%} after tax")
    print(f"  Weights E / D  : {w.equity_weight:.1%} / {w.debt_weight:.1%}")
    print(f"  WACC           : {w.wacc:.3%}  (sd {w.std_wacc:.3%}{', fallback' if w.wacc_sigma_fallback else ''})")
