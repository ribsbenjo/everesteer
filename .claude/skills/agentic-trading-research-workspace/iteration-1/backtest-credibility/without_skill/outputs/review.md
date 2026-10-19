# Review: GPT-4o multi-agent backtest (best of ~40 prompt variants)

Reviewed: 2026-10-19
Input: `returns.csv`, 521 daily rows, 2023-01-03 to 2024-12-31, columns `strategy` and `benchmark`
Reproduce every number below with `python3 analysis.py` (same folder; pandas/numpy/scipy, fixed bootstrap seed).

## Verdict

**Not credible as it stands. Do not present the 2.2 Sharpe to the investment committee as evidence of skill.**

There are three reasons, listed from quickest to check to slowest:

1. **The returns file fails basic integrity checks.** It has a row for every weekday, including all 19 NYSE holidays in the window, and every one of those rows has a non-zero return. The benchmark's worst day in the whole file is 4 July 2023, when the market was closed. On days with well-known market moves, the benchmark does not match what happened. Until someone can explain where these numbers came from, nothing calculated from them belongs in an IC deck.
2. **The backtest design has several biases, and each one pushes the Sharpe up.** The best of ~40 prompts was picked on the same two years with no holdout. The LLM's training data overlaps the test window. The universe is today's S&P 500. Trades fill at the close of the same day the signal is generated. Costs are zero. For an LLM strategy, none of these is a small effect.
3. **Even at face value, ignoring every bias, the evidence is weaker than "Sharpe 2.2" sounds.** The 95% confidence interval runs from about 0.9 to 3.5. With a Bonferroni haircut for 40 variants, the Sharpe drops to about 1.3 and the adjusted p-value is 0.07. The deflated Sharpe ratio comes out at 0.84, below the usual 0.95 bar.

What the team can honestly say today is: "an in-sample research prototype with an encouraging but unverified backtest." Section 4 lists what has to happen first. Some of it is quick: provenance, the calendar, the cash rate and the execution-lag table. The rest needs a re-run of the pipeline on clean, point-in-time inputs. The out-of-sample test can run right now on 2025 to 2026 data, which the model never saw and which was never used to pick the prompt.

---

## 1. Data integrity (blocking)

| Check | Result |
|---|---|
| Row count | 521. NYSE had 502 sessions in 2023-2024 (250 + 252). |
| Date index | Identical to pandas `bdate_range("2023-01-03", "2024-12-31")`, i.e. every Monday to Friday |
| NYSE holidays in the file | 19 of 19 present. All have non-zero strategy and benchmark returns. |
| Worst benchmark day in the file | 2023-07-04 (Independence Day, market closed): -3.29% |
| Other large holiday moves | 2024-01-01 -2.35%, 2024-06-19 -2.26%, 2023-01-16 -2.15% |

Spot checks against widely reported S&P 500 moves (the index moves are approximate):

| Date | What the market did | File benchmark | File strategy |
|---|---|---|---|
| 2024-08-05 | S&P 500 about -3.0% (carry-trade unwind) | +1.78% | +0.62% |
| 2024-12-18 | about -2.9% (Fed meeting) | -0.47% | +1.34% |
| 2023-03-09 | about -1.8% (SVB) | +0.12% | +1.29% |
| 2024-11-06 | about +2.5% (post-election) | +1.59% | +1.81% |

Other inconsistencies:

- **Calendar-year benchmark returns don't fit.** The file shows +6.2% for 2023 and +15.4% for 2024. The S&P 500 Equal Weight index returned roughly +14% and +13%. An equal-weight basket of *today's* constituents should beat that index because of survivorship bias. It should not trail it by about 8 points in 2023.
- **The return distribution doesn't look like equities.** Both series have slightly negative excess kurtosis (-0.11 and -0.22), pass a Jarque-Bera normality test (p = 0.36 and 0.41), and show no volatility clustering (Ljung-Box on squared returns, p = 0.50 and 0.61). Real daily equity returns over 2023-2024 are fat-tailed and their volatility clusters.

**Interpretation.** These series look like simulated Gaussian returns placed on a Monday-to-Friday calendar. Another possibility is a pipeline that builds its own business-day calendar and then fills or misaligns prices. Either case is serious:

- If this file is not the real backtest output, the committee deck has no supporting numbers yet.
- If it is the real output, the backtest engine is broken. That also calls into question how signals are lined up with returns, which is exactly where look-ahead bugs live.

Dropping the 19 holiday rows doesn't fix the problem. The strategy Sharpe stays at 2.21, but the benchmark Sharpe rises from 0.69 to 1.04 because the holiday rows were mostly bad days for the benchmark. The apparent gap between the two shrinks.

---

## 2. Methodology problems (these apply even after the data is fixed)

### 2.1 Best of ~40 prompts, chosen on the whole test period (high)

The 2.2 is the maximum of ~40 attempts, all evaluated on the same 521 days with no holdout.

- **The bar for "better than luck" is high.** With this sample length, the expected best Sharpe among 40 independent variants *with no skill at all* is about 1.5.
- **Two effects push in opposite directions.** Variants of one pipeline are correlated, so the effective number of trials is below 40. On the other hand, "about 40" understates the real search. Agent roles, debate rounds, thresholds, the universe, the date range and any code changes made after seeing results all count as trials.

Deflated Sharpe ratio, i.e. the probability that the true Sharpe beats the best-of-N result expected from noise (0.95 is the usual bar):

| Effective trials | Hurdle Sharpe (noise-level dispersion 0.70) | DSR | Hurdle (dispersion 1.0) | DSR |
|---|---|---|---|---|
| 5 | 0.83 | 0.975 | 1.19 | 0.926 |
| 10 | 1.10 | 0.943 | 1.57 | 0.817 |
| 20 | 1.32 | 0.896 | 1.90 | 0.671 |
| 40 | 1.52 | 0.836 | 2.19 | 0.514 |

**Bonferroni haircut.** The single-test p-value is 0.0018. With 40 trials the adjusted p-value is 0.071 and the haircut Sharpe is **1.28**, a 42% haircut. With 20 trials the haircut Sharpe is 1.49. The alpha t-stat (3.15) gives the same picture: its p-value after a ×40 Bonferroni correction is 0.066.

**What we need from the team:** the Sharpe of *every* variant tried. If the median variant is close to the benchmark and only the top one reaches 2.2, the result is mostly selection.

### 2.2 LLM look-ahead: GPT-4o has seen part of the test period (high)

GPT-4o's documented training-data cutoff is October 2023, so January to September 2023 (194 of 521 rows) falls inside the model's training data. The model may already know how earnings came out, guidance changes, deals and which stocks rallied. Telling it "today is 2023-03-01" does not remove that knowledge.

Leakage also comes in through the inputs, and the training cutoff does not limit this kind:

- news or sentiment feeds that return articles by relevance without a strict "published before decision time" filter
- restated fundamentals used instead of as-reported figures
- social data scraped later
- analyst agents that run web searches

What the file shows: performance is *stronger* after the cutoff (Sharpe 2.63 after vs 1.44 before), so the classic memorisation pattern is not visible here. That is only partly reassuring:

- the data itself is unreliable (section 1)
- the post-cutoff window is 15 months, and all of it was used to pick the prompt
- input-side leaks don't depend on the cutoff

Tests to run:

- **Anonymisation test.** Replace tickers and company names with neutral labels, rescale prices, and re-run. If performance collapses, the model was relying on what it knows about named companies.
- **Knowledge probe.** Ask the model about outcomes after the decision date that it should not know.
- **Cutoff split.** Repeat the before/after-cutoff split on the regenerated backtest.

### 2.3 Survivorship bias: today's S&P 500 members (high)

Using today's members means the universe was chosen knowing which companies grew into the index or stayed in it:

- It **leaves out** companies that were removed after falling.
- It **includes** names that joined after big runs in the test window, for example Super Micro Computer (added March 2024) and Palantir (September 2024).

The benchmark carries the same bias, so it partly cancels in the relative comparison, but only partly. An active strategy can pile into the names that hindsight selection guarantees did well. An LLM is also likely to favour heavily covered names it "knows", and those are the same names hindsight put in the index.

**Fix:** use point-in-time index membership, including returns for delisted and removed stocks.

### 2.4 Filling at the close of the signal day (high)

If the analyst agents read that day's close, daily bar, volume or intraday news, the signal doesn't exist until after the close it supposedly trades at. A multi-agent GPT-4o pipeline also makes many sequential calls per ticker, so it cannot cover 500 names between the last piece of information and the closing auction.

**Fix:** trade at the next open or next close, and report the Sharpe at execution lags of 0, 1 and 2 days. A large drop from lag 0 to lag 1 means the edge was look-ahead.

From the returns alone, I found no significant same-day timing signature:

- Henriksson-Merton γ = 0.13, t = 1.0
- correlations with leads and lags of the benchmark are all |ρ| < 0.09
- down-capture of 0.44 vs up-capture of 0.72 is lopsided, but not significant

This doesn't clear the issue. Settling it needs the position file and signal timestamps.

### 2.5 No transaction costs (medium to high)

Turnover isn't reported. TradingAgents-style per-ticker daily BUY/HOLD/SELL calls tend to flip often, so costs matter. Annual Sharpe (rf = 0) under different daily turnover and all-in cost assumptions:

| Daily turnover (share of book) | 2 bps | 5 bps | 10 bps | 20 bps | Break-even cost to fall to Sharpe 1.0 |
|---|---|---|---|---|---|
| 10% | 2.18 | 2.13 | 2.05 | 1.90 | 76 bps |
| 25% | 2.13 | 2.02 | 1.82 | 1.42 | 31 bps |
| 50% | 2.05 | 1.82 | 1.42 | 0.62 | 15 bps |
| 100% | 1.90 | 1.42 | 0.62 | -0.96 | 8 bps |

Other costs to include:

- market impact in the closing auction at the intended size
- borrow costs if the book shorts
- GPT-4o API spend for running the full pipeline on every name every day

### 2.6 How the Sharpe is defined (medium)

The 2.21 matches mean/std × √252 with a zero risk-free rate. T-bills paid about 5% over 2023-2024, so the excess-return Sharpe is **1.90** (benchmark 0.38).

The annualisation is also off. The file has 260.5 rows per year, not 252. Annualising at the file's own frequency gives 2.25. The difference is small, but it is another symptom of the calendar problem.

### 2.7 Short sample covering one market regime (medium)

The test covers two years of a strong market led by a narrow group of large-cap tech stocks.

- **Sharpe 95% CI:** [0.83, 3.60] analytic, [0.89, 3.54] stationary block bootstrap
- **Alpha 95% CI:** 9% to 49% per year (bootstrap)
- **Stability:** rolling 6-month Sharpe ranged from 0.29 to 4.88. The first half of 2024 alone ran at 3.9.

There is no factor attribution yet. A book with beta 0.53 that leaned into momentum or mega-cap tech during 2023-2024 would look like alpha against an equal-weight benchmark. Regress the returns on market, size, value, momentum, quality and low-volatility factors.

### 2.8 Reproducibility of LLM outputs (medium)

GPT-4o is not deterministic even at temperature 0, and OpenAI updates and retires snapshots. Part of the best prompt's lead may be sampling noise.

- Re-run the chosen variant at least 5 times and report the spread.
- Pin the model snapshot.
- Log every prompt and response.

### 2.9 An LLM makes the final risk call (governance)

This doesn't affect whether the backtest is credible, but the IC will ask about it. Limits that an LLM "enforces" are not hard limits. Position, gross, net and drawdown limits need to sit in deterministic code outside the model.

---

## 3. Face-value numbers from returns.csv

These numbers take the file as given. Read them together with section 1.

| Metric | Strategy | Benchmark |
|---|---|---|
| Total return (2 yrs) | +101.2% | +22.5% |
| CAGR | 42.0% | 10.7% |
| Annualised mean / volatility | 35.1% / 15.9% | 11.1% / 16.1% |
| Sharpe, rf = 0 | 2.21 | 0.69 |
| Sharpe, rf = 5% | 1.90 | 0.38 |
| Sortino, rf = 0 | 3.44 | 1.00 |
| Max drawdown | -12.0% | -11.5% |
| Daily hit rate | 55.5% | 52.2% |
| 2023 / 2024 return | +23.7% / +62.7% | +6.2% / +15.4% |
| 2023 / 2024 Sharpe | 1.47 / 2.88 | 0.44 / 0.94 |

| Benchmark-relative | Value |
|---|---|
| Beta (Newey-West se) | 0.53 (0.04) |
| Correlation / R² | 0.54 / 0.29 |
| Alpha | 29.2% per year, NW t = 3.15 (bootstrap 95% CI 9% to 49%) |
| Residual volatility / appraisal ratio | 13.4% / 2.18 |
| Active return / tracking error / IR | 24.0% / 15.4% / 1.56 |
| Up-capture / down-capture | 0.72 / 0.44 |
| Months positive / months beating benchmark | 18 of 24 / 18 of 24 |

| Period | Days | Strategy Sharpe | Benchmark Sharpe |
|---|---|---|---|
| Inside GPT-4o training window (to 2023-09-30) | 194 | 1.44 | 0.06 |
| After the cutoff (2023-10-01 onward) | 327 | 2.63 | 1.07 |
| 2023 H1 / H2 | 129 / 130 | 1.67 / 1.25 | 0.93 / 0.02 |
| 2024 H1 / H2 | 130 / 132 | 3.91 / 1.88 | 1.14 / 0.72 |

An alpha of 29% per year at a 2.2 appraisal ratio would be exceptional by any institutional standard. A result that strong should be met with suspicion until someone else has independently reproduced it.

---

## 4. What we need to do before presenting

Must do before the IC (each item has a "done when"):

1. **Trace provenance and regenerate the returns.** Done when the returns come from the backtest engine on real prices with an exchange calendar (502 sessions) and the benchmark reconciles to the S&P 500 Equal Weight index on daily moves.
2. **Point-in-time universe.** Done when the universe uses historical S&P 500 membership, including delisted and removed names and their returns.
3. **Realistic execution.** Done when signals use only data timestamped before decision time, trades fill at the next open or later, and a lag 0/1/2 table is reported.
4. **Point-in-time agent inputs.** Done when news is filtered by publication time, fundamentals are as-reported, web search has a date cap, and the anonymised-ticker re-run is reported next to the main result.
5. **Costs and turnover.** Done when daily turnover is reported, a spread + impact + commission model is applied (plus borrow if short), and the net Sharpe is the headline number.
6. **Honest statistics.** Done when the deck shows the excess-of-cash Sharpe, a confidence interval, the Sharpe distribution of all ~40 variants, and a deflated Sharpe computed from the real dispersion across variants.
7. **True out-of-sample test.** Freeze the chosen prompt and the GPT-4o snapshot, then run it unchanged on January 2025 to September 2026. That window is after the model's cutoff and was never used for selection. Done when this result, and not the 2023-2024 one, is the headline.

Should do (the IC will ask):

8. Run the chosen variant at least 5 times and report the dispersion caused by LLM non-determinism.
9. Compare against baselines: the median of the 40 variants, a single-agent version, a version without the debate, and a beta-matched mix of benchmark and cash. Also add a factor regression (market, size, value, momentum, quality, low-volatility).
10. Move risk limits into hard-coded rules outside the LLM risk manager, and start paper trading with a pre-registered evaluation plan.

## 5. If the IC date can't move

Present this as a research update, not a performance claim. Describe the architecture. Say plainly that the 2023-2024 backtest is in-sample, selected from ~40 variants, and not yet cleaned for the issues above. Then show the validation plan from section 4 with dates. Don't put "Sharpe 2.2" on a slide. Once it has been shown to an IC it becomes the number everyone remembers.

---

## Assumptions

- `returns.csv` holds daily simple returns. The strategy column is gross of all costs, and both columns are aligned to the dates shown.
- The quoted 2.2 is mean/std × √252 with rf = 0, because that reproduces 2.21 exactly.
- The risk-free rate is about 5% per year (2023-2024 T-bill yields).
- GPT-4o's training cutoff is October 2023, as OpenAI documents for the GPT-4o snapshots. If a later snapshot or a different model was used, the overlap shifts.
- All ~40 variants were evaluated on the same January 2023 to December 2024 window. Their results were not available to me, so the deflated-Sharpe figures use assumed dispersions (0.5, 0.70 = pure noise, 1.0) and a range of effective trial counts.
- Turnover, positions and signal timestamps were not provided. The cost table is therefore a grid, and the look-ahead timing tests use returns only.
- The market moves quoted for the spot checks are approximate and from public reporting. The holiday-row finding does not depend on them: it only needs the NYSE holiday calendar.
