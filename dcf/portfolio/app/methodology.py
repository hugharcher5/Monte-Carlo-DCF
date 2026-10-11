"""The "How the model works" expander.  Every claim here should match the code
in dcf/valuation; the module named in each section is where to check it."""
from __future__ import annotations

import streamlit as st

from valuation.dcf import G_CEILING_A, G_CEILING_B, HIGH_GROWTH_THRESHOLD
from valuation.montecarlo import MC_RANDOM_SEED, PROMOTE_THRESHOLD
from valuation.priors import TERM_G_HALF, TERM_G_MODE, load_base_rates
from valuation.wacc import ERP


def render() -> None:
    br = load_base_rates()
    g, m, c = br["growth"], br["margin"], br["corr"]
    with st.expander("How the model works", expanded=False):
        st.markdown(f"""
### Overview

I built this to value the stocks in my own portfolio and to see what a share price
assumes. If a stock trades at 30 times earnings, what growth and margins would make
that price fair?

The model is a standard free-cash-flow-to-the-firm (FCFF) discounted cash flow, the
approach taught in Aswath Damodaran's valuation books, with a Monte Carlo layer on
top. The DCF gives one value from one set of inputs. The Monte Carlo runs the same
DCF 10,000 times with the uncertain inputs drawn from ranges, which shows how much
the answer depends on those inputs. The choices on top of the textbook method, such
as how the ranges are set and how the data sources are compared, are mine. The code
was written with Claude Code.

The spread of the 10,000 results is the spread of the model's answers given the
input ranges. It is not a probability that the stock is worth a given amount: the
ranges are estimates, and the model leaves out anything that is not in them.

---

### 1. Data

*Code: valuation/data.py, sources.py, filings.py, reconcile.py*

Up to three sources are used: SEC EDGAR (the company's own 10-K and 10-Q filings),
Financial Modeling Prep (FMP) and Yahoo Finance.

- **Filings come first.** Revenue, operating income, pre-tax income, tax, interest,
  D&A, CapEx and stock-based compensation come from the 10-K where the filing tags
  them cleanly. Cash, debt and the share count come from the latest 10-Q or 10-K.
- **Yahoo Finance** supplies market data that filings never contain (price, market
  value, beta, exchange rates), and any line where the filing figure is missing,
  stale or far from the reported total. The "Where the data comes from" table says,
  for every input, which source was used and why.
- **FMP** is a third source. Its cash, debt and share count are used when the filings
  have none, and its figures are compared with the others. FMP's free plan refuses
  many symbols, so some valuations compare only two sources. The app says when that
  happens.

**Debt** means bonds, loans, commercial paper and finance leases. Operating leases
are left out because their cost is already inside operating income, so counting the
lease liability as debt as well would count it twice. FMP's total debt includes all
leases, so its lease line is removed before it is used.

**Share count.** Where the filings allow, the model uses the treasury-stock method:
the basic shares on the latest filing's cover page, plus in-the-money options net of
the shares their exercise cash would buy back at today's price, plus unvested
restricted stock units. If that is not possible it uses the latest diluted
weighted-average count, and then Yahoo's share count (which is a basic count, so
it misses dilution). A filing count more than 10 times away from Yahoo's is treated
as a scaling error (McDonald's tags one count in millions) and is not used.

**When the sources disagree.** For revenue growth, operating margin and CapEx as a
share of revenue, the model takes the latest-year figure from each source and
measures how far apart they are. The measure is 1.4826 times the median absolute
deviation from the median, which equals the standard deviation for normally spread
numbers but cannot be dragged far by one odd source. This spread is always added to
that input's uncertainty, in quadrature:
""")
        st.latex(r"\sigma = \sqrt{\sigma_{\text{model}}^2 + \sigma_{\text{sources}}^2}")
        st.markdown(f"""
If every source agrees, the spread is zero and nothing changes. Debt is treated as a
definition question, not a data question, because the sources mostly disagree about
whether leases count. So net-debt disagreement comes from cash alone. Share counts
and net debt become sampled inputs only when their spread is more than
{PROMOTE_THRESHOLD:.0%} of the value; otherwise they are fixed.

---

### 2. History

*Code: valuation/drivers.py*

The model reads up to five years of annual accounts, which gives up to four years
of revenue growth. For each year it works out revenue growth, operating (EBIT)
margin, tax rate, D&A, CapEx and stock-based compensation as shares of revenue.

- **Tax rate** is tax expense divided by pre-tax income, the rate the company
  actually pays.
- **Stock-based compensation** stays as an expense inside operating income. It is not
  added back, so margins here are lower than "adjusted" margins in broker reports.
- **Working capital** is taken from the latest fiscal-year balance sheet.
- **Forecast length** is 10 years if trailing growth is above
  {HIGH_GROWTH_THRESHOLD:.0%}, otherwise 5 years, so fast growers have time to slow down.

---

### 3. From history to a forecast

*Code: valuation/priors.py, scripts/estimate_base_rates.py*

Four years of growth is a poor guide to the next five. Fast growers usually slow
down and weak years are often followed by recovery. So the model does not just
carry the company's average forward. It uses a regression fitted on
{g['n_obs']:,} company-years from about 1,700 US companies' SEC filings, which
predicts the average of the next three years from the trailing average and the
company's size:
""")
        st.latex(r"\text{forecast} = a + b \times \text{trailing average} + c \times \text{size}")
        st.markdown(f"""
- For **revenue growth** the slope b is {g['b']:.2f}. Each extra point of past growth
  adds only about {g['b']:.2f} of a point to the forecast, and bigger companies are
  forecast to grow more slowly.
- For **operating margin** the slope is {m['b']:.2f}, so margins carry forward much
  more than growth does.

The same regression gives the **uncertainty**. The leftover errors are larger for
small companies and for companies whose own history is short or erratic, and that
error size becomes the standard deviation the Monte Carlo samples from. This is
uncertainty about the company's long-run level, which is what a draw held for the
whole forecast needs. It is not the year-to-year swing in four annual figures, which
the earlier version of this model used and which made the ranges far too wide.

The data behind the fit is mostly small and mid-sized companies, so for the very
largest companies the fit is an extrapolation (size is capped at the range seen in
the data).

---

### 4. The DCF

*Code: valuation/dcf.py*

**Revenue** starts at the forecast growth rate and falls in a straight line to the
terminal growth rate by the last forecast year.

**Size limit on growth.** Each year's growth is capped at
max(terminal growth, {G_CEILING_A:.0f} / R^{G_CEILING_B:.3f}), where R is revenue in
USD billions at the start of that year. The two anchor points behind it ($250bn of
revenue can grow about 25% a year, $1tn about 6%) are my judgement, not fitted to
data. The cap is about 65% at $100bn of revenue, so it only matters for very large
companies still growing fast.

**Operating margin** moves in a straight line from the forecast starting margin to a
mature margin by the last year, and the terminal value keeps that margin. For a
company with a profitable history, the mature margin starts as the midpoint of its
average and best year, then is pulled toward the base rate by the margin regression.
A company with no profitable history is "unproven": its own margins say nothing about
maturity, so the mature margin is its industry's average operating margin
(Damodaran, January 2026 data), or about 13% if the industry is unknown, and all its
input ranges are 1.5 times wider.

**CapEx.** Net reinvestment (CapEx minus D&A, as a share of revenue) falls in a
straight line from its historical level toward the level the terminal growth rate
needs. With the company's return on capital held fixed, reinvestment needed is
proportional to growth, so net reinvestment is scaled by terminal growth divided by
current growth. It is never scaled up, and a company whose CapEx is below D&A keeps
its CapEx share unchanged.

**Free cash flow each year:**
""")
        st.latex(r"\text{FCFF}_t = \text{EBIT}_t (1-t) + \text{D\&A}_t - \text{CapEx}_t - \Delta\text{NWC}_t")
        st.markdown("**Terminal value** (the final year's cash flow growing at a constant rate):")
        st.latex(r"TV = \frac{\text{FCFF}_N (1+g)}{\text{WACC} - g}")
        st.markdown(f"""
**Terminal growth** is {TERM_G_MODE:.1%}, about 2% inflation plus a little real growth,
or the reporting currency's 10-year bond yield if that is lower. It does not depend on
the company. In the Monte Carlo it ranges from {TERM_G_HALF * 100:.0f} percentage point
below to {TERM_G_HALF * 100:.0f} point above that ({TERM_G_MODE - TERM_G_HALF:.1%} to
{TERM_G_MODE + TERM_G_HALF:.1%} for a US company), and it is always at least 1 point
below WACC.

**Timing.** Cash flows are discounted to today, not to the last fiscal year end. If
the last annual accounts closed eight months ago, only the remaining four months of
the first forecast year are counted, and each year's cash arrives in the middle of
that year (the mid-year convention). The forecast still starts from the last full
fiscal year's revenue; it does not use the latest quarters.

**Equity value.** Enterprise value minus net debt (debt minus cash) gives equity
value, which is divided by the share count.
""")
        st.latex(r"V/\text{share} = \frac{EV - \text{net debt}}{\text{diluted shares}}")
        st.markdown(f"""
**Currency.** A company that reports in another currency is valued in that currency,
discounted at a rate built on that currency's own 10-year government bond yield, and
converted to US dollars once at today's exchange rate. Net debt is converted at the
same rate. Where no local yield is available (for example the Taiwan dollar), the US
yield is used and a warning is shown.

---

### 5. Discount rate (WACC)

*Code: valuation/wacc.py*
""")
        st.latex(r"\text{WACC} = w_E K_E + w_D K_D (1-t), \quad K_E = R_f + \beta_{\text{adj}} \times \text{ERP}")
        st.markdown(f"""
- **Risk-free rate:** the 10-year government bond yield of the reporting currency
  from FRED (DGS10 for US dollars, OECD series for others). In backtests it is the
  yield published on or before the valuation date.
- **Beta:** Yahoo's beta pulled toward 1 (0.67 x raw + 0.33), because extreme betas
  tend to drift back toward the market average.
- **Equity risk premium:** a fixed {ERP:.1%}, close to Damodaran's implied US premium at
  the start of 2026. It is not estimated here, and no extra country risk premium is
  added for companies outside the US.
- **Cost of debt:** interest cover (EBIT / interest) maps to a credit rating and spread
  using Damodaran's synthetic rating table (January 2024 version). After tax.
- **Uncertainty:** WACC is rebuilt for each past year using that year's bond yield
  and interest cover, and the standard deviation of those values is the WACC range.
  With fewer than three usable years it is 1.5 points.

---

### 6. The Monte Carlo

*Code: valuation/montecarlo.py*

Each of the 10,000 scenarios draws five inputs once and holds them for the whole
forecast: starting revenue growth, the margin level (one draw shifts the starting and
the mature margin together), terminal growth, WACC and CapEx as a share of revenue.

**PERT distributions.** Each input is drawn from a PERT distribution (a rescaled Beta)
centred on the forecast. Its end points are set so its standard deviation equals the
estimated uncertainty: for a symmetric PERT that means mode plus or minus
sqrt(7) x standard deviation. Every draw is then kept inside plausibility limits (for
example, revenue growth between -30% and +100%). Unlike a normal distribution, a
PERT cannot produce a wild value outside its end points.

**Correlations.** The inputs are tied together with a Gaussian copula, so each keeps
its own PERT shape. The correlations are estimated from how the regression errors in
section 3 move together across companies:

| Pair | Correlation |
|---|---|
| Revenue growth and margin | {c['growth_margin']:+.2f} |
| Revenue growth and CapEx | {c['growth_capex']:+.2f} |
| Margin and CapEx | {c['margin_capex']:+.2f} |

Terminal growth and WACC are market and economy-wide inputs with nothing
company-level to estimate a link from, so they are drawn independently. The
correlations matter less than you might expect: on example companies, switching them
off narrows the P10 to P90 range by 5% to 11% and moves the median by under 2%
(dcf/tests/test_montecarlo.py checks this).

Share count and net debt, when the sources disagree about them (section 1), are drawn
separately from narrow ranges.

All 10,000 scenarios are valued in one vectorised calculation. The random seed is
fixed ({MC_RANDOM_SEED}), so the same inputs always give the same result.

---

### 7. Checks on the output

*Code: valuation/gates.py, reverse_dcf.py*

- **Banks, insurers and brokers are refused.** Debt and customer money are part of
  how they operate, so free cash flow to the firm and net debt have no clean meaning
  for them.
- **Not applicable.** If the final forecast year's free cash flow is zero or negative,
  or the base-case value is below zero, the headline value is replaced by an
  explanation. A DCF cannot value a business that does not produce cash.
- **Warnings** appear when the median value is more than 5 times above or below the
  price, when P10 is negative, when the accounts are more than a year old, or when a
  foreign company had to use the US bond yield.
- **Reverse DCF.** The model solves for the starting revenue growth (and separately
  the mature margin and WACC) that would make the base case equal today's price, with
  everything else unchanged. Comparing that with the company's history is often more
  useful than the value itself.

---

### 8. Known weaknesses

- The forecast is built from each company's own history, pulled toward averages.
  A company about to change (new products, a turnaround, the end of a heavy
  investment cycle) will be misjudged.
- The equity risk premium, the rating table and the size limit on growth are fixed
  assumptions, not estimated.
- Stock-based compensation counts as an expense, so high-SBC companies look less
  profitable than on an "adjusted" basis.
- For large, highly valued companies the model's median is usually well below the
  price. The "Analyst estimates" tab measures how often the price falls inside the
  model's P10 to P90 range across 100 stocks.
- Pre-profit companies cannot be valued sensibly with a DCF. Their value depends on
  whether they reach profitability and how much new money they need on the way.
""")
