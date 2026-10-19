# Backtest review: multi-agent GPT-4o strategy (reported Sharpe about 2.2)

Reviewed 2026-10-19. Input: `returns.csv` (daily strategy and benchmark returns, 2023-01-03 to 2024-12-31).
Every number below comes from `review_analysis.py` in this folder; its full printout is in
`review_analysis_output.txt`.

## Verdict

**Not credible as it stands. Do not present the 2.2 Sharpe to the investment committee as evidence of alpha.**

The first problem is the returns file. It has rows for 19 days when the NYSE was closed, every one of them with a non-zero return, and the benchmark does not move with the market on well-known days. Until someone explains that, nothing computed from the file can be traced to market data. Even if the file is fixed, the backtest design has four known upward biases, and any one of them could produce a Sharpe this high on its own: a universe of today's survivors, fills at the same close that generated the signal, zero costs, and choosing the best of about 40 prompts on the same two years it is reported on. The statistics are also weak. Once the holiday rows are removed, the strategy's outperformance of its own benchmark is no longer statistically distinguishable from zero, and that is before any adjustment for the 40-variant search.

What can honestly go to the committee next week: "prototype built; preliminary in-sample result with known biases; this is the validation plan, and this is when results will be ready." Leave the Sharpe figure off.

## Assumptions

I could not ask questions, so I assumed the following. Each one is a point the team should confirm.

- Both columns are simple daily returns. The strategy column is gross: no commissions, spread, impact or borrow costs.
- "Current S&P 500 constituents" means index membership on the date the backtest was built (2025 or 2026), applied to every day of 2023 and 2024. The benchmark is an equal-weight portfolio of those same names.
- The model is a GPT-4o snapshot with a knowledge cutoff of 1 Oct 2023 (OpenAI's stated cutoff for the gpt-4o snapshots). If a later snapshot or `chatgpt-4o-latest` was used, its cutoff is later and the overlap problem in P7 gets worse.
- Sharpe ratios are annualised with √252 and a risk-free rate of zero, which matches how the 2.2 was reported, except where I state otherwise.
- I received no positions, trades, prompts, decision logs or pipeline code. Turnover and costs are therefore scenarios, and the leak checks are recommendations rather than results.

## Headline numbers from the file as delivered

| | Strategy | Benchmark | Active (strategy − benchmark) |
|---|---|---|---|
| Rows | 521 | 521 | 521 |
| Annualised mean return | 35.1% | 11.1% | 24.0% |
| Annualised volatility | 15.9% | 16.1% | 15.4% (tracking error) |
| Sharpe ratio (rf = 0) / information ratio | **2.21** | 0.69 | 1.56 |
| Sharpe ratio with rf ≈ 5% (2023–24 T-bills) | 1.90 | 0.38 | – |
| Total return / CAGR | +101% / 41.9% | +22.5% / 10.7% | – |
| Max drawdown | −12.0% | −11.5% | – |
| CVaR 95% / 99% (daily) | −1.91% / −2.62% | −2.06% / −2.53% | – |
| Beta to benchmark (HAC t) | 0.53 (13.9) | – | – |
| Alpha, annualised (HAC t, 5 lags) | 29.2% (3.15) | – | – |

The 2.2 reproduces from the file. These are the figures the team would present. The rest of this review explains why they shouldn't.

## Problems, in order of severity

### P1. The returns file does not look like it came from market data (blocker)

- **It has returns on days the exchange was closed.** The file has 521 rows, one for every weekday from 2023-01-03 to 2024-12-31. The NYSE traded on 502 days in that window (250 in 2023, 252 in 2024). Each of the 19 surplus rows falls on an NYSE full-day closure (Good Friday, July 4, Thanksgiving, Christmas, 2024-01-01 and others), and each has a non-zero return in both columns. The benchmark's largest absolute move in the whole file, −3.29%, is dated 2023-07-04, when the market was shut. A price-based backtest cannot generate a return on a day with no prices.
- **Those phantom rows drive a large share of the result.** They make up 3.6% of the rows but 27% of the summed active return. Summed over the holiday rows, the benchmark loses 9.5% while the strategy gains 3.9%. Dropping them moves the benchmark Sharpe from 0.69 to 1.04 and the information ratio from 1.56 to 1.22. The strategy Sharpe stays at 2.21.
- **The benchmark doesn't track known market days.** The S&P 500 fell about 1.8% on 2023-03-09 (SVB), about 3.0% on 2024-08-05 (VIX spike) and about 2.9% on 2024-12-18 (FOMC). Equal-weight indices fell sharply on the same days. The file's benchmark on those days is +0.12%, **+1.78%** and −0.47%.
- **The benchmark's yearly returns are implausible.** The file shows +6.2% for 2023 and +15.4% for 2024. The published S&P 500 Equal Weight total return was roughly +14% in 2023 and +13% in 2024 (verify against your data vendor). An equal-weight portfolio of today's survivors ought to *beat* the published index, because survivorship flatters it. Trailing it by about 8 points in 2023 is the wrong direction.

Taken together, this looks like a placeholder or simulated series, or a broken data join, for example a plain business-day calendar with returns attached by position or generated synthetically. It is not what a backtest engine produces from exchange data. Fix: regenerate the series from the engine on an exchange calendar, reconcile the benchmark against an independent source to within a few basis points a day, and supply the positions and trades file behind it.

### P2. Survivorship bias: a universe built from today's constituents

The universe uses information from the future. Stocks enter the S&P 500 largely because they have already gone up. During 2024, for example, Super Micro, Vistra, CrowdStrike and Palantir were added after large rallies, while names such as American Airlines and Etsy were removed (check the index change history for the full list). Stocks that fell out or were delisted during the window are missing entirely. Holding a survivor list fixed over 2023–24 builds the outcome into the universe. The benchmark carries the same bias, so it partly cancels in the active return. It does not cancel in the absolute Sharpe, and it rewards any tilt toward the names that were later added, which an LLM that favours momentum and well-known stories is likely to have. Fix: use point-in-time constituents for each date, including later delistings (CRSP, Compustat or a similar point-in-time source), and benchmark against the point-in-time equal-weight index.

### P3. Filling at the close that generated the signal (same-bar look-ahead)

A TradingAgents-style pipeline gives its analysts the day's OHLCV plus news and social posts stamped with that date. Daily-dated news includes after-close earnings releases and guidance. If those inputs drive a fill at the same day's close, the backtest books the overnight reaction to news that appeared after the fill. Even with perfect data hygiene the fill can't happen in practice: NYSE closing-auction orders must be entered by 15:50 ET, before the close is known, and four analysts plus a debate, a trader and a risk manager take time to run. The skill's evaluation protocol lists "executing at the signal-generating close instead of the next tradable price" among the mundane choices that make unusable strategies look profitable.

The return series alone doesn't show this leak. The Treynor–Mazuy timing coefficient is 4.18 (HAC t = 1.45, not significant), and up/down capture is 0.72/0.44. But a returns-only test cannot rule it out. Fix: have every input carry an availability timestamp, set a decision cutoff such as data stamped at or before 15:30 ET or the prior close, fill at the next open or next-day VWAP, and report both fill rules.

### P4. No transaction costs

The file has no positions, so I can't measure turnover. The table below shows Sharpe after costs for a range of daily two-sided turnover levels (fraction of NAV traded per day) and per-side costs.

| Daily turnover | 2 bps | 5 bps | 10 bps | 20 bps |
|---|---|---|---|---|
| 10% | 2.18 | 2.13 | 2.05 | 1.90 |
| 25% | 2.13 | 2.02 | 1.82 | 1.42 |
| 50% | 2.05 | 1.82 | 1.42 | 0.62 |
| 100% | 1.90 | 1.42 | 0.62 | −0.96 |

The strategy can absorb about 9.6 bps of NAV per day in costs before its Sharpe falls to the benchmark's, and about 13.9 bps per day before it reaches zero. An LLM issuing daily BUY/SELL/HOLD calls across 500 names can easily trade 50–100% of NAV a day, and in that range realistic large-cap costs (spread plus impact plus fees) consume most of the edge. Report the actual turnover and the Sharpe at 0×, 1× and 2× a stated cost model.

Inference cost is real money too. Under assumed inputs (500 tickers, 12 LLM calls per decision, 6k input and 600 output tokens per call, GPT-4o list price of $2.50 and $10 per million tokens), one full two-year run costs about $63k, 40 variants about $2.5M, and live running about $32k a year. If the team spent far less than that, then fewer tickers or fewer calls were actually run, and the committee needs to know how that subset was chosen, because the choice is another selection effect.

### P5. Best of about 40 prompts, chosen on the same data, with no holdout

Every prompt variant is a separate trial. Picking the best on 2023–24 and then reporting 2023–24 results is in-sample selection. Nothing in this result is out-of-sample. GPT-4o's sampling noise makes it worse: rerunning the same prompt produces a different Sharpe, so the winning variant was partly a lucky draw.

The Deflated Sharpe Ratio (Bailey & López de Prado) measures the probability that the true Sharpe beats the best result 40 skill-less variants would be expected to produce. Computing it needs the spread of Sharpe ratios across the 40 variants, which I don't have, so here is a range. Two years of daily data give a Sharpe sampling error of about 0.70 by themselves, so a standard deviation of 0.5–0.75 across variants is plausible.

| Information ratio vs benchmark (stock-picking skill) | sd(SR) across variants 0.25 | 0.50 | 0.75 |
|---|---|---|---|
| As delivered (IR 1.56), N_eff = 40 | 0.93 | 0.75 | 0.45 |
| As delivered, N_eff = 10 (highly correlated prompts) | 0.95 | 0.87 | 0.71 |
| Trading days only (IR 1.22), N_eff = 40 | 0.83 | 0.57 | 0.28 |
| Trading days only, N_eff = 10 | 0.88 | 0.73 | 0.52 |

The usual bar is a DSR of at least 0.95. The excess over the benchmark clears it only in the most generous cell. The raw Sharpe against zero deflates less (DSR 0.79–1.00 in the same range), but beating cash in a bull market says nothing about skill. Fix: send all 40 variants' daily returns so the DSR and the probability of backtest overfitting (CSCV) can be computed properly, then freeze the chosen prompt and test it on data the search never saw.

### P6. The statistical evidence is thin even before the fixes

- Moving-block bootstrap (10-day blocks, 10,000 draws) 95% CI of the strategy's Sharpe: **[0.85, 3.54]**. Two years of data can't separate a 1 from a 3.
- Outperformance vs the benchmark, as delivered: the mean active return CI is [+2.9%, +44.4%] a year and the Sharpe-difference CI is [0.20, 2.81]. That clears zero, but only barely.
- **With the 19 holiday rows removed:** the mean active return CI is [−2.7%, +38.7%] a year, the Sharpe-difference CI is [−0.15, 2.49], and the bootstrap P(no outperformance) is 4%. The paired Wilcoxon on the full file gives p = 0.063. **The edge over the benchmark is not statistically established**, and that is before the multiple-testing adjustment in P5.
- A beta of 0.53 means part of the absolute return is plain market exposure in a strong two-year rally. Charging a 5% cash rate cuts the Sharpe to 1.90.
- The result is unstable over time. Quarterly information ratios run from −1.40 (2023Q2) to +3.29 (2023Q1), and 2024H1 alone has a Sharpe of 3.9. Two of the eight quarters underperform the benchmark.

### P7. The backtest overlaps the LLM's training data

Assuming a 1 Oct 2023 cutoff, Jan–Sep 2023 (194 of the 521 rows) lies inside GPT-4o's training data. The model may "know" which companies won in 2023, and that knowledge colours its analysis of the same names in 2024 as well. In this file the pre-cutoff stretch is not the better one (information ratio 1.40 before vs 1.65 after), which is mildly reassuring. It doesn't clear the concern, though, because the file itself is suspect (P1) and the prompt search used both stretches. News, fundamentals and social tools also need an `as_of ≤ t` filter, and fundamentals should be as first reported, not restated. Fix: evaluate only on data after the model's cutoff. 2025 onward is clean for GPT-4o, and it is also a holdout the 40-variant search never touched. Alternatively, run a "blindfolded" version with anonymised tickers and dates as an ablation.

### P8. No ablations or attribution (MR-7)

There is no evidence that the multi-agent machinery adds anything. Before the committee sees this, compare it with: a single GPT-4o agent given matched compute; the pipeline with the debate removed and with news or social inputs removed; and simple rules over the same universe, such as 12-1 momentum, a mega-cap or AI-theme tilt, and low volatility. With a beta of 0.53 and 15% tracking error in a market led by a few mega-caps, a factor and sector regression (Fama-French 5 + momentum + sectors) may explain much of the 29% alpha.

### P9. Deployment risk: the LLM risk manager has the final say

This doesn't affect the backtest numbers, but the committee will be approving a design. In a published red-team of a TradingAgents-style pipeline (Na et al. 2026, GPT-4.1), a single jailbreak prompt reaching the Risk Manager flipped 95.5% of decisions toward a SELL target and 98.9% toward a BUY target. Poisoning 80% of news items reached only about 31%. The cause is a terminal LLM that is both the safety check and the final decision-maker. Fix: put a deterministic, non-LLM gate after the risk manager (position, notional and turnover caps, a symbol allowlist, drawdown kill switch, cooldowns), and keep position and limit state read-only to the LLM. `agentic-trading-research/scripts/execution_gate.py` is a reference implementation. Red-team in both directions before any capital goes in.

## Reporting checklist status (MR-1..MR-7)

| Item | Status |
|---|---|
| MR-1 Data and universe | **Fail**: survivor universe; calendar and benchmark integrity problems (P1, P2) |
| MR-2 Temporal split / sealed holdout | **Fail**: selection and evaluation on the same window, no holdout (P5) |
| MR-3 Action semantics | Not stated: weights, sizing, long/short and constraints unknown |
| MR-4 Execution and costs | **Fail**: same-close fills, zero costs (P3, P4) |
| MR-5 Leakage audit | Not done: no availability table, no lint or oracle run, pre-cutoff LLM overlap (P3, P7) |
| MR-6 Artifacts | Not supplied: positions, decision logs, prompts, model snapshot ID, temperature, seeds |
| MR-7 Multi-agent ablations | Not done (P8) |

## What to do before presenting

Blocking. Without these there is no result to present:

1. **Explain and regenerate `returns.csv`.** Run it from the engine on the NYSE calendar, reconcile the benchmark daily against a published equal-weight series, and supply positions, trades and turnover.
2. **Point-in-time universe** including delisted and removed names, with a point-in-time equal-weight benchmark.
3. **Realistic execution.** Set an information cutoff before the decision and fill at the next open or next-day VWAP. Report both.
4. **Costs.** State a commission, half-spread and impact model, report turnover, and show results at 0×, 1× and 2× costs.
5. **Out-of-sample test.** Freeze the chosen prompt (no more tuning) and run it once on 2025-01-01 onward. That period is after GPT-4o's cutoff and was never seen by the search. Pre-register the pass criteria now, for example: active-return block-bootstrap CI excludes zero, positive after 2× costs, DSR ≥ 0.95 given the full search count.
6. **Disclose the search.** Report all ~40 variants' returns and compute the DSR and PBO. Rerun the chosen prompt at least 5 times to measure LLM run-to-run variance.

Before any capital is committed:

7. Factor and sector attribution. Ablations: single agent with matched compute, no debate, no text inputs, simple baselines.
8. Leak audit of the pipeline code. Write an availability rule for every input, then run `leak_check.py lint` and `leak_check.py oracle` with many epochs and several perturbation types. As-of filters on every news and fundamentals call.
9. Deterministic execution gate after the LLM risk manager, plus a two-direction red-team (P9).
10. Inference-cost budget and break-even gross P&L at the intended capital.

If steps 1–5 are done and the strategy still shows a positive, cost-adjusted, statistically significant edge on 2025+ data against a point-in-time benchmark, that is worth bringing to the committee. The current number is not.

## Reproducing the numbers

```
cd "<this folder>"
python3 review_analysis.py > review_analysis_output.txt
```

The script only reads the input CSV and imports `robust_stats.py` from the skill's `scripts/` folder (block bootstrap, PSR and DSR, drawdown, CVaR). The other dependencies are pandas, numpy and statsmodels.
