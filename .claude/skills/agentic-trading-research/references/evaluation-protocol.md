# Evaluating and reporting agentic trading results

Sources: the survey's MR-1..MR-7, §13 and Appendix B (Xia et al. 2026); statistical practice from
SAE (Borjigin et al. 2026), Fonseca 2026 and Na et al. 2026. Helpers are in
`scripts/robust_stats.py`.

Contents: 1. Why most results don't transfer · 2. MR-1..MR-7 · 3. Applicability by study type ·
4. Splits, sealed tests and search budgets · 5. Costs and execution realism · 6. Statistics that respect
dependence · 7. Multiple testing and overfitting · 8. Reproducibility tiers · 9. Reporting template ·
10. Anti-patterns

---

## 1. Why most results don't transfer

Of the survey's 80 closed-loop systems:
- 62.5% report a time-consistent split;
- 58.8% report explicit transaction costs;
- 60% report survivorship handling;
- 58.8% report execution semantics;
- 56% have no runnable artifacts, and none has a full reproduction package.

Reported split protocols: walk-forward 5, rolling 8, other 37, unknown 30. Execution choices that
look mundane can turn an apparently profitable strategy into an unusable one: executing at the
signal-generating close instead of the next tradable price, building the universe from future
constituents, or omitting costs. Further reading: "The Alpha Illusion" (reported alpha from LLM
agents isn't deployment evidence) and backtest-to-live gaps measured directly in crypto (Yu et al.
2026b). Negative results are rarely published, so the literature itself has survivorship bias.

## 2. MR-1..MR-7

- **MR-1 Data and universe.** Universe construction criteria, how survivorship is handled, and
  synchronized availability timestamps across every modality.
- **MR-2 Temporal splitting.** Strictly chronological train, validation and test sets; walk-forward
  or rolling windows; explicit **embargo** periods; a **frozen holdout** for iterative alpha search.
- **MR-3 Action semantics.** Discrete orders or continuous target weights; decision timestamps;
  order-mapping rules; constraints (leverage, gross and net exposure).
- **MR-4 Execution and costs.** Execution timing; fill-price assumption (next open, VWAP, …);
  rebalancing interval; frictions (fees, bid–ask spread, market impact).
- **MR-5 Leakage controls.** Causal integrity across feature normalization, text-sentiment
  alignment and label construction.
- **MR-6 Artifacts and verification.** Code, configurations, random seeds and structured decision
  logs. For adaptive systems, also update frequency and rollback criteria.
- **MR-7 Multi-agent evaluation.** Role topology, communication protocol, and ablations that isolate
  the marginal value of interaction.

## 3. Applicability by study type

| Item | End-to-end trading | Execution algorithm | Component (perception or risk) |
|---|---|---|---|
| MR-1 Data and universe | Essential | Task-dependent | Essential |
| MR-2 Split and leakage | Essential | Essential | Essential |
| MR-3 Action semantics | Essential | Essential | – |
| MR-4 Execution and costs | Essential | Essential | – |
| MR-5 Leakage audit | Essential | Recommended | Recommended |
| MR-6 Artifacts | Recommended | Recommended | Recommended |
| MR-7 Multi-agent evaluation | Task-dependent | – | – |

## 4. Splits, sealed tests and search budgets

- Use chronological splits only. Shuffled K-fold on time series leaks the future into training.
- Put an embargo around each test window so labels that overlap the boundary don't leak.
- **Walk-forward:** choose parameters on a validation segment, freeze them, evaluate once on the
  next disjoint test segment, then roll forward. SAE's protocol selects on validation and evaluates
  the frozen choice once on test, with several seeds.
- **Sealed test set:** any loop that feeds results back (alpha mining, reflection, prompt tuning,
  hyperparameter search, or "submit, look at the score, tweak") must never see the final test data.
- **Report the search budget:** the number of hypotheses, factors, nodes or variants tried. Without
  it, a best result can't be interpreted.
- **Reflection budget:** cap the number of critique–revise iterations, and never use out-of-sample
  feedback to decide when to stop.
- **Meta-learning tasks** are split by time or by disjoint asset classes, never by random episodes.
- **Regimes:** report results by sub-period or regime as well as pooled. Valid features can fail
  across regime boundaries.

## 5. Costs and execution realism

- Include fees, bid–ask spread, slippage and market impact. Report **sensitivity** to the cost
  assumptions, for example at 0×, 1× and 2× the base cost.
- **Alpha-to-Trade contract:** does the signal frequency fit the execution latency, does the edge
  survive turnover, and is the universe liquid enough for the capital?
- Separate decision timing from fill timing, and never fill at the signal close.
- For perpetuals, include funding cash flows and tiered maintenance margin with liquidation (SAE).
- **Inference costs:** annual API cost = D × N × C_req (trading days × decisions per day × cost per
  request). Report the model version, pricing snapshot date, token budgets, tool charges and the
  **break-even gross P&L**. Na et al. made about 10 LLM calls per decision, so a 305-decision run
  takes about 3,000 calls.

## 6. Statistics that respect dependence

Per-step returns are autocorrelated, and naive i.i.d. tests overstate significance.

| Question | Method | Used by |
|---|---|---|
| CI of mean return | (moving) **block bootstrap** | SAE (block bootstrap), Na et al. (moving block, 10-day blocks within each ticker, 10,000 draws) |
| Is strategy A better than B on the same dates? | **paired** test on aligned returns: block bootstrap of the difference, or Wilcoxon signed-rank | SAE (Wilcoxon p = 0.0113) |
| Compare two rates (attack success, hit rate) | two-proportion z-test, cross-checked with Fisher's exact test for small cells | SAE (p = 1.76e−8), Na et al. |
| Zero events in a test arm | rule of three: 95% upper bound ≈ 3/n | Fonseca (0/33 gives 9.1%) |
| Small corpus clustered by mechanism or asset | cluster bootstrap and cluster-respecting permutation test; Benjamini–Hochberg across comparisons | Fonseca |
| Correlation over only a few points (4 architectures) | treat as directional; no meaningful CI | Na et al. |
| Power | report post-hoc power honestly, even when below 80% | Fonseca (76.7%) |

How much this matters: across 200 simulated AR(1) series (φ = 0.6, n = 400, true mean 0), a naive
i.i.d. 95% CI of the mean covered the truth only **69%** of the time. A moving-block bootstrap
(block 20, `robust_stats.block_bootstrap_ci`) covered it **94%** of the time.

Other good practice:
- pre-register the analysis plan and report deviations (Fonseca);
- fix seeds and record commit hashes;
- report a permutation-limited p-value as "p < 10⁻⁴" with the number of discordant pairs, not as a
  spuriously precise number.

**Risk metrics:** MDD = min over t of (E_t − max_{s≤t} E_s) / max_{s≤t} E_s. CVaR_α is the mean of
returns at or below the (1−α) quantile, used at 0.95 and 0.99. Also report liquidation count, tail
loss, turnover and exposure.

**Attack and robustness metrics:** ASR on attackable days in both directions; AttackSuccess and
FalseBlock; DG rate and loss; signed EV and ΔCR. A flip rate is not a dollar loss.

## 7. Multiple testing and overfitting

- Code-generated, evolutionary and MCTS factor search, and repeated strategy tweaking, all make the
  probability of a false positive approach 1 as the search grows.
- **Classical controls** named by the survey: White's Reality Check, Hansen's SPA and the
  **Deflated Sharpe Ratio** (Bailey & López de Prado 2014), plus the probability of backtest
  overfitting (Bailey et al.).
- `robust_stats.py dsr` implements the probabilistic and deflated Sharpe ratio. The formula comes
  from Bailey & López de Prado (2014), which the survey cites; it isn't reproduced in the PDFs.
  Inputs are the observed per-period Sharpe, the number of observations, skewness, kurtosis, the
  number of trials N and the variance of the Sharpe ratios across trials.
- Regularization (complexity penalties, AST-originality limits, ensembles) and reflection budgets
  are useful heuristics, but they **don't bound** the false discovery rate. Report both the search
  constraints and a classical statistic.
- Separate model selection from final evaluation, and be explicit about multiple-testing risk
  (Arian et al. 2024).

## 8. Reproducibility tiers (survey Table B.24)

| Level | Definition | Share of the 80 |
|---|---|---|
| R0 | no runnable artifacts (no repository, a broken link, or essential files missing) | 56.2% |
| R1 | public code, but key pieces missing (dependencies, data access, evaluation scripts) | 41.2% |
| R2 | runnable with gaps: complete code, accessible or documented data, evaluation scripts, but no pinned environment or end-to-end pipeline | 2.5% |
| R3 | full package: pinned environment, data snapshots or access instructions, complete documentation, end-to-end evaluation | 0% |

Aim for R3: pin the environment, snapshot the data with checksums, provide one command per
table, fix seeds and keep decision logs.

## 9. Reporting template

```
Universe & data   : instruments, selection rule as of each date, delistings handled, sources + availability rules
Split             : train/val/test dates, walk-forward step, embargo, sealed holdout (never touched by search)
Search budget     : #variants/factors/prompts tried, selection criterion, deflated Sharpe / SPA p-value
Actions           : orders vs target weights, decision time, fill rule (next open/VWAP), constraints
Costs             : fees, spread, slippage, impact, funding; sensitivity at 0x / 1x / 2x
Leakage audit     : availability per input, lint results, two-run oracle (epochs, perturbations), LLM cutoff / anonymization
Results           : return, vol, Sharpe, MDD, CVaR95/99, turnover; per-regime breakdown
Uncertainty       : block-bootstrap CI of mean and of the difference vs a baseline; paired test
Ablations         : text on/off, roles removed, single-agent with matched compute, gate variants
Robustness        : attack suite, ASR (both directions), AttackSuccess/FalseBlock, $ impact
Costs of inference: model/version, pricing date, calls per decision, break-even gross P&L
Artifacts         : repo, commit, env lockfile, data checksums, seeds, exact commands, decision logs
```

## 10. Anti-patterns (survey §13.1.6 and elsewhere)

- Temporal leakage: shuffled splits, scalers or features fitted on the full sample, centered
  indicators, hindsight in text.
- Unrealistic costs: none at all, or no sensitivity analysis.
- An unchecked universe: survivorship, selection bias, undocumented inclusion rules.
- Backtest overfitting: selection and evaluation on the same data; an undisclosed search budget.
- Missing deployment safeguards: no immutable action logs linking each action to its inputs,
  evidence and model version; no verifiable retrieval, citation checks or cross-source consistency
  tests.
- Implicit execution: fills at the close with no latency, rejections or partial fills.
- Treating LLM explanations as evidence. They are summaries until tied to Order_IDs and snapshot
  hashes that can be replayed.
- Claiming multi-agent gains without role ablations and a compute-matched single-agent baseline.
- Reading ASR as dollar harm, or a flat equity curve as safety, when the tails haven't been checked.
- Claiming adaptation without frozen execution and cost semantics, rollback, and evaluation on
  sealed data.
