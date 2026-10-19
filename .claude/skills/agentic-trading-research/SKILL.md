---
name: agentic-trading-research
description: Research-backed playbook for LLM-agent ("agentic") trading and forecasting systems, distilled from five 2026 papers — system architecture (perception, memory, reasoning, execution, risk), look-ahead bias as temporal non-interference, execution-layer safety (Survivability-Aware Execution, Delegation Gap), adversarial attacks on multi-agent trading pipelines, evaluation/reporting standards (MR-1..MR-7, block bootstrap, multiple testing), and the Agentic ETF market thesis. Use this skill whenever the user designs, builds, backtests, audits, red-teams, evaluates or writes about a trading agent, trading bot, backtest, alpha/factor search, forecasting-competition submission or multi-agent finance pipeline — including checking features for look-ahead leakage, adding risk gates or kill switches, choosing an agent topology, judging whether a backtest result is real, or citing these papers — even if the word "agentic" never appears.
---

# Agentic trading research playbook

This skill condenses five papers (≈140 pages) into decisions you can act on. The body
gives the core lessons and task checklists; the `references/` files hold the detail,
formulas and numbers. Read only the reference file that matches the task.

| Need | Read |
|---|---|
| Which paper says what, key numbers, how much to trust each | `references/sources.md` |
| Architecture of a trading agent (perception, memory, reasoning, execution, alpha, portfolio, risk, learning, multi-agent, self-evolution) | `references/architecture.md` |
| Look-ahead bias, leakage channels, point-in-time data, leak testing | `references/lookahead-leakage.md` |
| Execution safety: SAE contract, budgets, cooldowns, Delegation Gap, attack suite | `references/execution-safety.md` |
| Prompt injection / data poisoning / jailbreaks in multi-agent pipelines, topology robustness | `references/adversarial-multiagent.md` |
| How to evaluate and report results (MR-1..MR-7, statistics, reproducibility tiers, costs) | `references/evaluation-protocol.md` |
| Market size, Agentic ETF, infrastructure stack, regulation, incidents, crowding | `references/market-and-regulation.md` |

Bundled tools (pandas + numpy + stdlib only, so they run where scipy is missing):

- `scripts/leak_check.py`: a static linter for common leak patterns in pandas code, plus a
  two-run perturbation oracle and a truncation test for any `build(df) -> panel` function.
- `scripts/execution_gate.py`: a reference Survivability-Aware Execution gate
  (ALLOW/LIMIT/BLOCK, budget projection, cooldown, rate limit, allowlists, audit log).
- `scripts/robust_stats.py`: max drawdown, CVaR, moving-block bootstrap CIs (single series and
  paired differences), two-proportion z-test, rule-of-three bound, attack success rate,
  probabilistic and deflated Sharpe ratio.

Run any script with `--help`. Import them by adding the `scripts/` folder to `sys.path`.

## The core lessons

These ten points recur across the papers. Use them to frame advice.

1. **Agentic trading means delegating judgment, not just execution.** An agent decides
   what and whether to trade and treats execution algorithms and venues as tools.
   Rule-based algos automate how an order is worked, and robo-advice only rebalances
   static allocations. Neither of those counts as agentic.
2. **Most reported "alpha" is not deployment evidence.** Of the 80 closed-loop systems in
   the survey, only 62.5% report a time-consistent split, 58.8% explicit costs, 60%
   survivorship handling and 58.8% execution timing. 56% ship no runnable code and none
   ships a full reproduction package. Treat a headline Sharpe as a claim until the split,
   costs, universe and fill timing are stated.
3. **Look-ahead bias is temporal non-interference.** Every datum has a *reference time*
   (what instant it describes) and an *availability time* (when it became knowable). A
   decision at t may depend only on data with availability ≤ t. Leakage hides in the gap
   between the two: publication lags, restatements, revised macro prints, post-close
   aggregation and survivorship universes.
4. **Leak detectors certify nothing by their silence.** Freedom from look-ahead is
   undecidable in general (Π⁰₁-complete). A two-run or tiling test can expose a leak but
   can never prove there isn't one. On an adversarial corpus a differential detector missed
   54.5% of planted leaks and a window-tiling detector missed 100%. A static
   availability-tracking check missed 0%. Design pipelines so availability is a property of
   the schema, never computed from data values, and then check it structurally.
5. **LLMs leak through their weights.** A model can know the future of a backtest period
   from pretraining even when the code is clean. Evaluate on data after the model's
   training cutoff, restrict every tool and retrieval call to `as_of ≤ t`, or anonymize
   tickers and dates ("blindfolded" evaluation).
6. **Execution is the attack surface.** Once an agent holds trading privileges, the main
   failure mode shifts from wrong answers to losses caused by execution. Treat every
   upstream intent (LLM output, third-party skill, retrieved text) as untrusted. Enforce
   budgets at the last mile, where they can't be bypassed: projection onto leverage,
   notional, rate and slippage caps, plus cooldowns, staging, allowlists and audit logs.
   In a Binance replay this cut max drawdown 93% and tail loss 97.5%.
7. **Never let the safety role also be the final decision node.** A single jailbreak
   prompt against the terminal Risk Manager flipped 95–99% of decisions, while 80% data
   poisoning only reached about 31%. Put deterministic, non-LLM gates after any LLM risk
   judge, and keep the authoritative position and limit state read-only to the LLM.
8. **Topology changes robustness, but no architecture is robust by itself.** Majority
   voting discards a compromised minority (APS = 0), while a single averaging decision
   agent preserves it fully (APS = 1). Voting gives thresholded robustness, though: once
   attackers can form a quorum, attack success jumps. Multi-agent designs reduce attack
   success relative to a single agent (29–48% vs 70% on GPT-4.1) but never remove it.
9. **Search is overfitting unless budgeted.** Alpha mining, MCTS, evolutionary search,
   reflection loops and repeated "submit and check" iterations are all multiple testing.
   Report the search budget, keep a sealed test set that never feeds back into the loop,
   cap reflection iterations, and deflate Sharpe ratios or use White's Reality Check or
   Hansen's SPA.
10. **Statistics must respect dependence.** Returns are autocorrelated, so use block or
    moving-block bootstrap, paired tests on aligned returns, and cluster-aware resampling.
    Naive i.i.d. tests overstate significance. Also, a decision-flip rate is not financial
    harm. Measure dollar impact separately.

## Task playbooks

### A. Building or auditing a feature pipeline / backtest for leakage

Work through these in order. Each is cheap and catches a distinct leak family.

1. **Write down both clocks for every input.** For each column, record the reference time
   and the availability rule, for example "close of bar t, available at t" or "quarterly
   EPS for Q, available at filing date + ingest lag". If the availability of anything
   can't be stated, that input is the prime suspect.
2. **Lint the code** with `python scripts/leak_check.py lint <file.py>`. It flags negative
   shifts, centered windows, backfill and two-sided interpolation, full-sample
   fit/normalization/ranking over time, `merge_asof` looking forward, and resample calls
   that need a closed/label check. Each hit is a question to answer, not a proof of a leak.
3. **Make every transform causal:**
   - fit scalers, normalizers, regime clusters and quantile bins on past data only
     (expanding or rolling windows, or per training fold);
   - when joining sources, stamp each row with the *latest* availability among its inputs;
   - resample so an output bar aggregates only source bars at or before its own close;
   - use as-first-reported (vintage) values for restated fundamentals and revised macro data;
   - build the universe from constituents known at t, including names that later delisted.
4. **Run the dynamic oracle** with
   `python scripts/leak_check.py oracle --module <file.py> --func <fn> --data <csv>`. It
   perturbs every row after a cutoff t several ways (noise, sign flip, shuffle, NaN, large
   shocks, ±1e9 extremes) and checks whether outputs at or before t change. If the project
   already has a two-run leak test, check how many epochs and how many perturbation kinds it
   uses. A single noise perturbation over a few epochs misses threshold and gap-dependent
   leaks (see the demonstration table in `references/lookahead-leakage.md`). It also checks truncation:
   whether `build(df[:t]).loc[t]` equals `build(df).loc[t]`. Use many epochs, because a leak
   can sit between sampled epochs or behind a threshold that one kind of perturbation
   doesn't flip.
5. **Know what the oracle can't see.** It trusts the timestamps in your input. A column
   stamped by reference time instead of availability (fundamentals at period end, restated
   or revised values, a universe of survivors) already contains the future, and perturbing
   "later rows" never touches it. This is the practical form of the paper's *re-stamping*
   channel. The oracle also misses leaks that occur only on dates it didn't sample, and
   leakage through LLM weights or a retrieval corpus. For those, rely on step 1 (declared
   availability), post-cutoff evaluation and as-of retrieval.
6. **Shift by one period when unsure.** If the rules don't guarantee that bar t is known at
   submission time, shift every feature by one. The cost is one period of staleness. The
   benefit is that the whole same-bar leakage family is removed.

Details, the leak catalogue and the formal rules are in `references/lookahead-leakage.md`.

### B. Designing an agentic trading system

Start from these contracts. They come from the survey and the ETF paper and prevent the
most common structural failures.

- **Two-layer working memory.** Layer A is a deterministic state store: positions, open
  orders, cash and risk limits. It is updated only by the broker/OMS and is read-only to
  the LLM. Layer B is the generative context: observations, reasoning and reflections.
  When they disagree, Layer A wins. The papers call this the hierarchy of truth.
- **Reasoning contract:** `f(observation_t, retrieved_memory, read-only portfolio) →
  (candidate actions, confidence ∈ [0,1], validity check)`. Check constraints right after
  reasoning, not only at execution.
- **Action contract:** typed orders (side, size, limit price, time-in-force) plus an
  execution algorithm (for example TWAP over 5 minutes), with explicit failure states
  (rejected, partial fill, timeout) fed back to memory. Never assume a fill at the
  signal-generating close.
- **Target-position orders.** Declaring "hold 10 AAPL" and letting the platform compute
  the delta makes execution idempotent and safe to retry.
- **Isolation and reconciliation.** Each strategy gets its own capital and position book.
  Reconcile continuously against the broker, and freeze a strategy when the book and the
  broker disagree.
- **Memory hygiene.** Use an *outcome embargo*: an episode's outcome can't be retrieved
  before its outcome time. Use time-decayed retrieval relevance and record provenance
  (document ID plus version timestamp) for every retrieved fact. Consolidated summaries
  keep a hash pointer to the raw logs.
- **Latency tiers.** LLMs work on a scale of seconds, so they are reflective or strategic,
  never reactive. Keep sub-millisecond stops and circuit breakers deterministic. Use a
  *cascaded controller* that routes by regime: reactive stops in a flash crash, strategic
  reallocation when markets are calm.
- **Risk outside the generative loop.** Apply checks pre-trade (whitelists, size and
  leverage caps, liquidity), in-trade (VaR, concentration, drawdown stops, volatility
  scaling, kill switch) and post-trade (reconciliation, slippage, attribution, audit log).
  Keep a deterministic gating layer, and let an asynchronous LLM explanation layer
  attribute causes after the fact.
- **Cost check before building.** Annual API cost = trading days × decisions per day ×
  cost per request. State the break-even gross P&L.

The full module-by-module design space is in `references/architecture.md`.

### C. Adding execution safety to an agent that can trade

1. Define the *Intended Policy Spec* S = (allowed tools/venues/symbols/accounts, risk
   budgets, market-state tightening rules, account-state rules). It drives both
   enforcement and measurement, so one spec defines both "allowed" and "out of scope".
2. Put a gate between strategy and executor that returns ALLOW, LIMIT or BLOCK and enforces
   budget = B₀ · g(market) · h(account) · q(trust), with projection (clamping) and
   cooldown, rate limit, slippage cap, staging and an audit record for every decision.
   `scripts/execution_gate.py` implements this with the paper's defaults: leverage caps of
   3×/2×/1× for calm/volatile/extreme regimes, cooldown 60 s/120 s, staging slices 4/5.
3. Red-team it with the five attack families: parameter escalation, cooldown bypass/order
   flooding, tool/venue misuse, state-violation stress, and narrative-induced flip stress.
4. Report AttackSuccess, FalseBlock, Delegation-Gap rate and loss, MDD, CVaR₀.₉₉ and gate
   latency. Tune under explicit feasibility constraints so the gate doesn't win by
   blocking everything (the paper used FalseBlock ≤ 0.20 and AttackSuccess ≤ 0.80).
   More checks are not always better: the budget-only gate had the best drawdown, and the
   full gate had the best attack resistance.

See `references/execution-safety.md`.

### D. Hardening a multi-agent pipeline against manipulation

- Map each role to its attack surface. Analysts face data poisoning and indirect prompt
  injection through news and social feeds. Researchers face persuasion (FOMO, appeals to
  authority). Traders face objective hijacking. Risk managers face jailbreaks ("executive
  override", "risk protocols suspended").
- Treat all ingested text as data: strip or flag instruction-like blocks, check source
  reputation and provenance, and cross-check numeric sentiment against the text.
- Prefer aggregation that discards minorities (voting) over aggregation that preserves them
  (a single averaging decision agent), and add peer revision. Remember the quorum effect:
  voting fails once attackers plus naturally agreeing benign agents form a majority.
- Separate safety enforcement from final decision authority, and add a deterministic check
  after the last LLM.
- Measure attack success only on attackable days (where the clean decision differs from the
  attacker's target), in both directions (BUY- and SELL-targeted), and measure dollar
  impact separately. The system's directional prior changes which attacks work.

See `references/adversarial-multiagent.md`.

### E. Evaluating and reporting a result

Use MR-1..MR-7 as the checklist: data and universe, temporal split with embargo and sealed
holdout, action semantics, execution and costs, leakage audit, artifacts (code, seeds,
logs), and multi-agent ablations. Then:

- compare against a simple baseline on aligned periods and bootstrap the *difference*
  with blocks (`robust_stats.py paired-ci`);
- report the search budget, meaning how many variants were tried, and deflate the best
  Sharpe ratio accordingly (`robust_stats.py dsr`);
- report sensitivity to cost assumptions, and split results by regime;
- report ablations: text signal on vs off, each role removed, a single agent with matched
  compute;
- for a small adversarial or test corpus, use cluster-aware resampling and give the
  rule-of-three bound when an arm has zero events.

See `references/evaluation-protocol.md`.

## Forecasting competitions and short-horizon submissions

The papers concern trading, but the same rules govern any scored-on-realised-outcomes
forecasting task:

- **Availability is the contract.** Shift features so the row for t uses only data
  knowable at submission time. When in doubt, shift by one.
- **Every submitted version is a test.** Picking the best of many backtested variants
  inflates the expected score. Keep the number of variants you've tried, prefer changes
  with an economic rationale, and require the block-bootstrap CI of the gain over the
  baseline to clear zero before switching.
- **A passing leak test is necessary, not sufficient.** Combine the dynamic test with the
  static lint and a written availability rule per input.
- **Regimes break signals.** Valid features can fail across regime boundaries, so check
  stability over sub-periods, not just the pooled score.
- **Trust LLM knowledge only after its cutoff.** If an LLM produces or ranks features, its
  pretraining may contain the outcome for historical periods.

## Caveats when citing these papers

- The Agentic ETF paper (Scalar Field) and the SAE paper (True Trading) come from vendors
  promoting their own stacks. Their market sizing relies on assumed penetration rates (5%,
  20%, 50% of a $4.2T 2030 active-ETF base), and SAE was evaluated on one three-month
  window of two perpetual contracts.
- The poisoning study backtests daily closes without costs on five large caps over one
  quarter, using GPT-4.1 and Qwen3. Its absolute attack success rates don't transfer
  across backbones, but the orderings do.
- Fonseca's real-data validation used proprietary data. The public artifact reproduces the
  qualitative claims, not the exact numbers. The soundness guarantee applies only to
  pipelines expressed in, or faithfully modelled by, the calculus.
- The survey's protocol-coverage numbers are descriptive counts over a mixed-depth coding
  of its corpus, not a trend.
