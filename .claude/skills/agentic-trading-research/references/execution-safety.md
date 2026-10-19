# Execution-layer safety: Survivability-Aware Execution (SAE)

Source: Borjigin, Stadnyk, Bilski, Hovorov, Pidturkina 2026, arXiv:2603.10092 ("Execution Is the New
Attack Surface"). Supporting points from the survey's risk chapter and the Agentic ETF
reconciliation layer. Reference implementation: `scripts/execution_gate.py`.

Contents: 1. Why execution is the attack surface · 2. Delegation Gap · 3. Objective and selection
protocol · 4. Execution contract · 5. Layers beyond a classic OMS · 6. Budgets, trust and projection ·
7. Modules and algorithms · 8. Default hyperparameters · 9. Threat model and mitigations ·
10. Attack suite and metrics · 11. Results and lessons · 12. Implementation and reproducibility

---

## 1. Why execution is the attack surface

- In OpenClaw-style stacks, language is a control surface. LLM intents pass through tool
  interception and a policy gateway to a local executor that holds privileges.
- Skill marketplaces (skills.sh, ClawHub, `npx skills add …`) turn capability acquisition into a
  **supply chain**. Installing a skill is close to importing code plus instructions that steer tool
  calls and parameters. In early 2026 there were reports of malware distributed through marketplace
  skills, including crypto-themed skills that harvested credentials. Moltbook, a social site for AI
  agents, leaked private agent messages, owner emails and credentials through a basic flaw.
- So the dominant failure mode moves from wrong answers to **losses caused by execution**, and
  "assume compromise" becomes a reasonable baseline. Treat upstream intent as **untrusted** and
  enforce survivability where actions become side effects.
- Crypto perpetuals amplify mistakes. Funding is deducted from the wallet and then from position
  margin, which moves the liquidation price. Maintenance margin is tiered by notional. Choices that
  look minor in an intent string (leverage, order rate, slippage tolerance, timing under stress)
  produce nonlinear tail outcomes.
- OpenClaw's own guidance: strict allowlists for high-risk tools, sandboxing whenever untrusted
  input is in the loop, and secrets kept out of prompts and the agent-readable filesystem.

## 2. Delegation Gap (DG)

A_intended ⊊ A_actual: the deployed system can execute more than the operator meant to authorize,
and the gap grows under prompt injection, compromised inputs or supply-chain events.

DG = E[ ℓ(a_t) · 1{a_t ∉ A_intended(S_t)} ], the expected loss from executable but out-of-scope
actions.

**Intended Policy Spec** S_t = (T_t, R_t, M_t, U_t):
- **T_t, allowed actions and tools:** {open, close, modify, cancel} plus the permitted venues,
  symbols and accounts.
- **R_t, risk budgets:** maximum leverage, notional, order rate, slippage, holding time and number
  of concurrent positions.
- **M_t, market-state constraints:** tighten in volatile, illiquid or extreme-funding regimes.
- **U_t, account constraints:** margin ratio, drawdown and P&L state, cooldown timers,
  concentration.

**Out-of-scope rule classes** (deterministic and reproducible):
- *cap violation*: exceeds leverage, notional, rate, slippage or holding caps;
- *tool/venue violation*: uses an unauthorized tool, market, venue, symbol or account;
- *state violation*: takes risk in a state where only reduce-only is allowed (extreme volatility,
  low margin).

**Estimators:** DG_rate = (1/N) Σ 1{OOS}. DG_loss = Σ ℓ·1{OOS} / (Σ ℓ + ε).

The same spec drives enforcement and measurement, so "allowed" means the same thing in the gate
and in the metrics.

## 3. Objective and selection protocol

min over π of CVaR_α(R_π), subject to E[R_π] ≥ μ₀ and P(liquidation) ≤ ε.

SAE is **not an alpha clamp**. It narrows the feasible policy set. There are also operational
feasibility constraints: AttackSuccess ≤ α, FalseBlock ≤ β, Latency ≤ τ. The paper used β = 0.20
and α = 0.80, so a gate can't look good just by blocking everything.

**Selection score:** J = w₁·MDD + w₂·|CVaR₀.₉₉| + w₃·DG_loss + w₄·Latency.

**Search:** a black-box, best-so-far constrained random search over strategy parameters φ and gate
parameters θ, run in batches with early stopping (patience). Select on a validation segment, freeze,
then evaluate **once** on a disjoint test segment (walk-forward). That ordering is what prevents
cherry-picking.

## 4. Execution contract

- **ExecutionRequest:** symbol, venue, timestamp_ms, intent, side, requested notional, requested
  leverage, order type, max_slippage_bps, strategy_id.
- **ExecutionContext:** account state (equity, drawdown, positions, recent trades, margin ratio);
  market state (realized volatility, funding, liquidity proxy, regime label); a first-class trust
  state z_t; and an optional narrative or anomaly signal.
- **ExecutionDecision:** ALLOW, LIMIT or BLOCK, plus the enforced leverage, notional, order-rate and
  slippage caps, cooldown and staging plan, plus audit fields (the rule that fired, budgets, state
  snapshot, timing).

Worked example from the paper. The request is BTCUSDT LONG at 5× leverage, 50% notional and an
80 bp slippage tolerance. The decision is **LIMIT** at 1× leverage, 20% notional, 30 bp, with a
120 s cooldown, for the reason "extreme regime + low skill provenance". The intent to trade
survives, inside survivability bounds.

Minimal action schema: `(type ∈ {open, close, modify, cancel}, symbol, side, notional, leverage,
limit_px, meta)`, where `meta` carries tool provenance and trust flags.

## 5. Layers beyond a classic OMS

- **Layer 0, static OMS baseline:** fixed leverage and size limits, stop-loss and reduce-only
  triggers, rate limits. It assumes the intent is trusted.
- **Layer 1, trust-conditioned budgeting:** budgets tighten when provenance is weak, capability
  risk is high or an injection alert fires.
- **Layer 2, supply-chain and scope enforcement:** the Intended Policy Spec plus non-bypassable
  allowlists over tools, venues, symbols and accounts.
- **Layer 3, attack-aware evaluation:** AttackSuccess, FalseBlock, latency, DG rate and DG loss are
  reported alongside P&L.

NoSAE, meaning direct wiring from strategy to executor, is a realistic baseline because many
deployed bots work that way: the strategy's output *is* the action.

## 6. Budgets, trust and projection

**Trust state** z_t = (p_prov ∈ [0,1], provenance of the active skill or toolchain; r_cap,
capability risk, meaning privileges and side-effect surface; I_inj ∈ {0,1}, injection alert).

**Budget vector** B_t = B₀ · g(σ_t, λ_t, f_t) · h(m_t, DD_t) · q(z_t):
- g tightens in volatile, illiquid or extreme-funding markets;
- h tightens for constrained accounts (low margin, high drawdown);
- q tightens on low trust.

Worked walkthrough: B₀,L = 3×, g_L = 1/3 (extreme regime), q_L = 0.5 (provenance 0.3 plus an
injection alert) gives L_cap = 0.5×, floored to 1× in practice. For notional, g_N = 0.5 and
q_N = 0.4 give N_cap = 0.20 of the default.

**Projection:** a_eff = argmin over a ∈ F(B_t) of D(a, a_req). When D and F are convex this is a
convex projection. Clamping is a special case: L_eff = min(L_req, L_cap). Projection blocks
parameter-escalation attacks *without needing to understand the prompt*.

**Robust bound.** If the per-step loss ℓ is L-Lipschitz in exposure and ‖e_t‖ ≤ B_t, then
ℓ(e_t) − ℓ(0) ≤ L·‖e_t‖ ≤ L·B_t. Under projection,
ℓ(e_eff) − ℓ(e_req) ≤ L·dist(e_req, F(B_t)). The bound is deliberately weak, since it assumes
nothing about returns, but it links budgets directly to worst-case one-step loss amplification.
Liquidation mechanics are left to the replay.

## 7. Modules and algorithms

1. **Trader-State Service:** a calibrated probability p_t of risk escalation. Features come from the
   recent trade window, equity versus peak, margin ratio, pacing, and leverage after a loss. The
   base model is logistic regression or GBM, calibrated by isotonic regression or temperature
   scaling.
2. **Market Regime Detector:** {calm, volatile, extreme} from realized volatility, funding extremes
   and a liquidity proxy. It uses deterministic proxies that can be recomputed in replay, not GARCH
   fits in the hot path. ARCH/GARCH motivate the volatility-clustering view.
   - r_t = log p_t − log p_{t−1}; σ_t = √(Σ r² over W_σ); λ_t = EMA(volume) / (σ_t + ε);
     DD_t = 1 − E_t / E_max.
3. **Policy Engine:** YAML rules mapping (regime, risk, trust) to a decision and budgets. Compute
   B_t = B₀·g·h·q, then the *first* rule whose predicate matches sets the decision and adjusts the
   budgets.
4. **Enforcement Runtime:**
   ```
   if decision == BLOCK: reject (or NoOp / reduce-only) and write an audit record
   else:
       a_proj = project(a_req, budgets)
       if a_proj violates cooldown or rate caps: a_eff = NoOp and write an audit record
       else: a_eff = stage(a_proj)          # time-sliced execution when staging is on
       enforce slippage cap and venue/tool allowlists; write an audit record
   return a_eff, audit
   ```

## 8. Default hyperparameters (Table 1)

| Component | Parameter | Default |
|---|---|---|
| Trader state | trade window N | 200 |
| Trader state | calibration | isotonic |
| Trader state | thresholds (τ_LIMIT, τ_BLOCK) | (0.50, 0.70) |
| Regime | realized-vol window W_σ (bars) | 60 |
| Regime | thresholds (τ_σ1, τ_σ2) | (1.0, 2.0) |
| Funding | extreme-funding threshold τ_f | 0.01 |
| Liquidity | illiquidity threshold τ_λ | 0.0 (depends on the proxy) |
| Enforcement | leverage cap baseline / volatile / extreme | 3.0× / 2.0× / 1.0× |
| Enforcement | cooldown volatile / extreme | 60 s / 120 s |
| Enforcement | rate-limit window W_rate | 60 s |
| Enforcement | staging slices volatile / extreme | 4 / 5 |

The paper doesn't define the units of the regime thresholds (1.0, 2.0). `execution_gate.py`
interprets them as a z-score of current realized volatility against its own trailing history
(calm < 1, volatile 1–2, extreme ≥ 2), and also marks the regime extreme when |funding| ≥ 0.01.
Recalibrate both for your market.

## 9. Threat model and mitigations

**Pipeline:** Strategy (LLM or not) → SAE middleware → local executor → exchange API. SAE runs
inside the executor boundary and intercepts every outbound call.

**Assets protected:** capital and margin safety; execution integrity; availability (no order floods
or churn); audit integrity.

**Adversaries:**
- prompt injection, direct or indirect;
- a skill or plugin supply-chain attacker;
- narrative or instruction contagion through agent social channels;
- an execution-stress adversary, who needs to steal nothing and only has to induce leverage,
  flooding or slippage during stress.

**Out of scope:** custody, key management, 2FA and exchange-side controls. SAE doesn't promise to
recognize malicious content.

**Attack success conditions:** an out-of-scope action executes; constraints are evaded (caps,
cooldown or rate limit exceeded); or tails get statistically worse compared with a compliant
baseline.

| Mitigation | Covers |
|---|---|
| M1 Intended spec, hard OOS rules, allowlists, state-dependent reduce-only | injection, supply chain, contagion |
| M2 Trust-conditioned tightening (provenance, capability risk, injection flag) | supply chain, contagion |
| M3 Projection into the budget region | injection, supply chain, stress |
| M4 Cooldown and order-rate limits (flooding, churn, rapid flips) | injection, contagion, stress |
| M5 Slippage bounds and staged execution ("unbounded slippage tolerance is permission to trade at any price") | stress |
| M6 Audit record per decision (rule, budgets, effective action, state snapshot incl. trust); replaying a stream and config regenerates the same decisions | all |

## 10. Attack suite and metrics

Attack generators, all seeded, interleaved with normal intended actions and labelled by the OOS
rules:
1. **Parameter escalation:** leverage, notional, slippage or rate beyond the caps, or beyond the
   tightened caps.
2. **Cooldown bypass and order flooding.**
3. **Tool or venue misuse:** unauthorized tools, venues, symbols or cross-account actions.
4. **State-violation stress:** taking risk during extreme volatility or at low margin.
5. **Narrative-induced flip stress:** rapid long/short flipping that is in scope by action type but
   out of scope by rate or state.

Metrics:
- **AttackSuccess (AS):** the share of attack attempts whose effective action is out of scope or
  breaks a cap or cooldown;
- **FalseBlock (FB):** the share of legitimate in-scope actions that get blocked (or, under a strict
  definition, overly constrained); it measures opportunity cost;
- DG rate and DG loss;
- MDD, CVaR₀.₉₅ and CVaR₀.₉₉ of per-step returns *including funding*;
- liquidation count;
- per-step decision latency.

## 11. Results and lessons

Binance USD-M BTCUSDT and ETHUSDT, 15-minute bars, 2025-09-01 → 2025-12-01. The full table is in
`sources.md`.
- Max drawdown 0.4643 → 0.0319 with the Full gate (−93.1%). Budget alone reached 0.0325.
- |CVaR₀.₉₉| 4.025e−3 → about 1.02e−4 (−97.5%). DG loss 0.647 → 0.019 (−97%).
- AttackSuccess 1.00 → 0.885 (static OMS) → 0.76 (Budget) → 0.728 (Full), with zero FalseBlock.
  The static OMS had FalseBlock 0.011.
- Latency: the Full gate costs about 7.6× NoSAE but still only about 0.01 ms per step.
- **More checks are not always better.** Budget-only had the best risk profile. Full had the best
  AttackSuccess and DG loss but slightly worse drawdown and higher latency. Report ablations and
  pick a point on the enforcement–throughput frontier.
- Remaining attack success comes from **timing- and state-sensitive attempts near regime
  boundaries**, and from requests that are formally in scope but harmful in context. With the gate
  calibrated to zero FalseBlock, ambiguous cases are LIMITed rather than BLOCKed. Stricter settings
  trade FalseBlock for lower AttackSuccess.
- SAE leaves nominal equity curves in calm regimes almost unchanged. It removes the tails.
- **Statistics:** because per-step returns are autocorrelated, use a block bootstrap of the mean,
  a paired Wilcoxon on aligned returns and a two-proportion z-test for AttackSuccess. Naive i.i.d.
  tests overstate significance.

## 12. Implementation and reproducibility

- **Services:** MarketData (Binance public REST `/fapi/v1/klines`, `/fapi/v1/fundingRate`,
  `/fapi/v1/exchangeInfo`, no API key needed, cached to disk); AgentCore (intended actions); SAE
  Gate; Executor/Sim (fills, fees, slippage, margin, liquidation); Logger (equity, actions, safety
  events, DG labels).
- **Alignment:** klines form the master index. Funding events align to the *next* kline boundary.
  Per-step state = (OHLCV, funding, spread/slip proxy, account, trust).
- **Liquidation** happens when margin_balance ≤ maintenance_margin = notional · mmr − maint_amount,
  with tiers. If the official tiers are missing, fall back to conservative placeholders so
  survivability isn't overstated.
- **Tuning command shape:** `auto_optimize.py --mode binance --batch_trials 20 --max_batches 15
  --patience 5 --falseblock_max 0.20 --attacksucc_max 0.80 --resume`. Seeds are fixed at the entry
  point.
- **Reproducibility checklist:** symbols, interval and window; commit hash; the exported best-params
  YAML; cached data with checksums; the exact command line; the skill version and exported policy
  spec.
- **Limitations and next steps:** regime-diverse windows; verified margin tiers and impact models;
  SBOM, signatures and attestations for skill provenance; adaptive multi-step adversaries; online
  trust calibration under drift; SAE-like contracts for other privileged executors (payments,
  cloud ops); a conformance suite.

Related controls from the other papers:
- the survey's deterministic **gating layer** with an asynchronous LLM **explanation layer**, and
  pre-trade, in-trade and post-trade risk stages;
- the Agentic ETF paper's **continuous reconciliation**: freeze a strategy when its holdings fall
  below the reconciled floor, use share claims for strategies sharing an account, and use
  target-position orders so retries are idempotent;
- Na et al.: don't let an LLM risk manager be the last step before execution.
