# Architecture of an agentic trading system

Source: mainly the survey (Xia et al. 2026, arXiv:2605.19337), with infrastructure from the Agentic
ETF paper (Singh 2026). Tags such as R0–R3 are the survey's reproducibility levels.

Contents: 1. Definition and loop · 2. Perception · 3. Memory · 4. Reasoning · 5. Action and execution ·
6. Alpha discovery · 7. Portfolio management · 8. Risk management · 9. Learning paradigms ·
10. Multi-agent coordination · 11. Adaptation and self-evolution · 12. Infrastructure stack (six layers) ·
13. Open challenges

---

## 1. Definition and loop

**Agentic trading** (Singh): an autonomous software agent, typically an LLM with tools, perceives
market state, reasons over heterogeneous data, forms an investment decision and executes it across
one or more venues on a recurring schedule, *without a human approving each trade*. The defining
property is that judgment is delegated, not only execution.
- Rule-based algorithmic trading (VWAP, HFT) automates *how* a given order is worked. *What* to
  trade, and whether to trade at all, comes from outside.
- Robo-advice keeps static allocations across passive funds and rebalances mechanically. It doesn't
  interpret news or earnings.
- An agent sits above both and treats execution algorithms and venues as tools.

Three properties an agent needs before it can manage capital:
1. **Isolation.** It trades its own capital and its own position book, so its behavior can be
   measured and its risk contained.
2. **Scheduled autonomy.** It runs on a cadence and acts within that cadence, without waiting for
   a prompt.
3. **Accounting truth.** Every action reduces to a verifiable change in positions and cash,
   reconciled against the venue of record, so performance is a NAV series.

**Survey loop:** Perception → Memory → Reasoning (candidate actions) → Forward planning or
Reflection → Action selection → Execution → Market feedback.

**Expert-system framing** (survey §2.1):
- Perception is knowledge acquisition.
- Memory, RAG and tool stores form the knowledge base. Its provenance and update rules must be
  inspectable.
- Reasoning is the inference engine.
- Execution is the external action interface. Target weights, orders, fills, latency and frictions
  must be explicit there.
- Risk, compliance and portfolio limits are rule constraints on the path from inference to action.
- Logs, immutable artifacts and human oversight form the accountability layer.

## 2. Perception

The perception layer produces an observation O_t from market state S_t *at decision time t*. It
must output representations the rest of the agent can act on (sentiment scores, regime labels,
visual summaries), not just predictions. A standalone predictor such as FinBERT, or an LSTM, is
background material until it is wired into an action loop.

| Modality | Inputs | Mechanisms | Protocol risk |
|---|---|---|---|
| Text | news, earnings, filings, social | FinBERT, FinGPT, BloombergGPT, FinLlama, Instruct-FinGPT, RAG-FinGPT | publication lag, embargo violations, post-publication revisions |
| Time series | OHLCV, limit order book, options-implied, fundamentals | attention encoders, regime switching (Hamilton 1989 HMM, Bai–Perron breaks) | look-ahead, data revisions, non-stationarity |
| Multimodal | charts, text, price, audio | FinVis-GPT, FinAgent (cross-attention, high latency), VISTA (zero-shot VLM) | cross-modal asynchrony, inference latency |

- **Text alignment.** A news timestamp is usually the publication time, not when the item reached
  the pipeline. Model the ingestion lag or enforce an embargo. Use the as-first-reported version of
  documents that get revised later.
- **Cross-modal asynchrony.** Pairing "the latest news" with the current price misrepresents the
  context when the news is stale. Vision or multimodal inference latency means the perceived state
  is already old by the time the agent acts. Report latency assumptions and alignment rules.
- **Fusion options:** early fusion (concatenate raw features), late fusion (merge high-level
  representations), or cross-attention.
- Most implemented systems still use OHLCV. Limit-order-book data and rich fundamentals are rare.

## 3. Memory

| Category | Capacity | Volatility | Notes |
|---|---|---|---|
| Working memory | limited (context) | volatile | eviction, summary compression, hierarchical (MemGPT-style); FinMem's working layer |
| Associative vector store | large (disk) | persistent | similarity search, temporal indexing |
| Structured knowledge base | source-dependent | persistent | knowledge graphs, sourced records, explicit relations (FinSage: versioned, doc-date embargo, drift check) |
| Parametric (related, implicit) | in the weights | static | opaque; "outdated priors" persist after a regime ends |

- **Layer A, the deterministic state store (audit truth):** positions, open orders, balances,
  risk limits. It is read-only to the LLM, updated only by EMS/OMS feedback and exposed through
  tools. Treating it as memory that can be forgotten or hallucinated is a critical vulnerability.
- **Layer B, the generative context:** observations, chain of thought, reflections. It is
  transient. Too much history dilutes attention ("Lost in the Middle"), while pruning too hard
  can drop reasoning that still matters for a live position.
- **Episode tuple** (S_t, A_t, O_{t+k}, τ), where τ is when the outcome becomes observable.
- **The "Oracle Fallacy"** is retrieving a past episode whose narrative contains hindsight, such as
  "failed due to news released tomorrow". The fix is an **outcome embargo**: the outcome field
  stays hidden from retrieval until t_now ≥ τ.
- **Time-aware retrieval:** add a decay factor e^{−λ(t_now − t_k)} to relevance. Handle drift, since
  similar-looking episodes from another regime can mislead.
- **Curated vs uncurated knowledge bases.** Curated sources (rulebooks, SEC filings) have high
  integrity. Uncurated sources (Reddit, forums) are open to noise injection and poisoning.
- **Source tracking.** Every retrieved fact carries a document ID and a version timestamp.
- **Hierarchy of truth.** Internal Layer A state overrides external noisy signals when they
  conflict. Aggressive summarization can distort inputs.
- **Trade-offs:** latency vs recall depth, and context length vs distraction. Prefer less context
  with higher relevance.

## 4. Reasoning

| Paradigm | Time scale | Mechanism | Examples | Evaluation gap |
|---|---|---|---|---|
| Reactive | µs–ms | rules, tiny NN (<100K params), precompiled tables; **not LLMs** | reactive parts of FinAgent | execution latency ignored |
| Reflective | seconds | CoT, self-critique, tools (ReAct is reflective) | FinCoT, FinCon, CryptoTrade, TradingGroup, Reflexion | inference cost not reported |
| Strategic | minutes–days | ToT, MCTS, hierarchical planning | Alpha Jungle, RAPTOR | search budget, overfitting |

- **Reasoning I/O contract:** f(O_t, M_assoc, P_portfolio [read-only]) → (A_cand, α ∈ [0,1],
  V_check), where V_check is a validity check such as "holds < max_leverage". Enforce constraints
  during or right after reasoning to cut down hallucinated actions.
- **Five reflective stages:** (1) analyze the situation, (2) retrieve memories, (3) consider several
  hypotheses, (4) assess each candidate's benefits and risks, (5) select one and justify it. Use low
  temperature for determinism, and checkpoints plus consistency checks on long chains. Cost grows
  linearly with the number of steps. Reflexion assumes prompt feedback, but trading outcomes arrive
  late.
- **Strategic planning.** Model-based methods learn a transition model and model-free methods learn
  value functions. Decompose hierarchically: a strategic agent sets targets such as "reduce tech
  exposure 20%" and tactical agents trade toward them. Plan over sectors or factors rather than
  single names to keep the search tractable.
- **Search-based overfitting.** MCTS or ToT over noisy history is multiple testing. Report the
  search budget (nodes visited) and freeze the plan before walk-forward out-of-sample testing.
- **Cascaded controller.** A master router classifies the regime ("Flash Crash" vs "Stable Trend")
  and dispatches: reactive stops in volatile conditions, strategic reallocation in calm ones.
  Use fast System 1 for routine decisions and escalate hard cases to System 2. Distillation and
  compute budgets that adapt to importance and volatility help here.

## 5. Action and execution

Execution is the boundary between a forecasting model and a real trading agent.
- **Action I/O contract:** f(D_t) → (A_order: side, size, limit price, time-in-force; A_algo, for
  example "submit now" or "TWAP over 5 min"). Failure modes are Rejection (risk check), Partial
  Fill (liquidity) and Timeout (latency), and each is fed back to memory. The contract prevents
  **implicit execution**, the assumption that the agent trades at the close with no delay or
  constraints.
- **Action definition risk.** Emitting only target weights silently merges intent with execution
  and understates the execution gap. State the order type (limit or market), queue assumptions and
  fill model.
- **Execution policies:** TWAP, VWAP and POV schedules, implementation shortfall (Perold 1988),
  optimal execution (Almgren–Chriss 2001, Bertsimas–Lo 1998), RL execution, and LOB fill models
  (Foucault 1999, Parlour 1998). Market impact follows Kyle 1985 and Glosten–Milgrom 1985.
- **Microstructure as feasibility.** (1) Condition on O_{t−δ}, because LOB features computed at t
  may not exist yet due to processing latency. (2) If T_compute > T_decay the action is invalid
  whatever its alpha (Budish et al. 2015).
- **Pre-trade validation and kill switches** keep theoretical alpha from turning into realized loss.
- **Log-to-narrative verification.** Post-trade LLM narratives cite the Order_ID and a cryptographic
  Snapshot_Hash, so an auditor can replay and check claims such as "spread was high". Without that
  link, explanations are summaries, not audit artifacts.
- Examples: FinPos (position-aware), StockSim (dual-mode order-level simulator), AutoQuant
  (auditable execution for crypto perpetuals), MEV and cross-chain arbitrage agents (where the
  execution is the strategy).

## 6. Alpha discovery

Classical baseline: Fama–French factors, the information coefficient (IC: correlation of factor
scores with later returns), and multiple-testing control (White's Reality Check, Hansen's SPA).
Grinold & Kahn.

| Paradigm | How it works | Examples | What to check |
|---|---|---|---|
| Code-based | LLM turns a hypothesis into factor code, runs it in a sandbox, backtests, refines | CogAlpha, AlphaAgent (originality, hypothesis alignment, complexity control vs decay), FactorMAD (debate) | search budget, sealed test set |
| Retrieval-based | adapt factors from a library or corpus | RAG-Fintech, PRISM | point-in-time library governance |
| Evolutionary | mutation, crossover, selection, MCTS, DRL over primitive operations | Alpha Jungle, Alpha², EFS | primitive-ops audit, total search depth, White RC / SPA / deflated Sharpe |

- **Alpha-to-Trade contract.** Does the signal's update frequency fit the execution latency? Does
  the gain survive turnover costs and impact? Is the universe liquid enough for the intended
  capital? Skipping this conflates IC with realized alpha.
- **Adaptive overfitting.** The generate → validate → refine loop "mines" the validation set.
  Report the number of hypotheses and measure on a sealed test set never shown to the loop.
  Sandboxing is a security control and doesn't enforce split integrity.
- **Library governance.** A library of historically successful factors conditions on the future
  (survivorship). Embeddings trained on descriptions such as "a successful high-volatility factor"
  leak. The corpus available at t should contain only documents and metadata known at t.
- **Multiple-testing scalability.** As the search space grows, the chance of a false positive
  approaches 1. Reflection budgets and AST-originality limits shrink the search but don't bound the
  false discovery rate. Report classical statistics as well.
- Valid LLM signals can fail across regime boundaries (Yang 2026), so alpha is regime-conditional.

## 7. Portfolio management

| Level | Traditional | LLM add-on | Key challenges |
|---|---|---|---|
| Allocation | mean-variance (Markowitz), Black–Litterman, risk parity, HRP | views from news and macro, regime detection, sentiment | covariance error, regime misclassification |
| Sizing | Kelly, risk parity, convex VaR/CVaR | qualitative risk flags, constraint reasoning | estimation-error sensitivity, oversizing |
| Rebalancing | thresholds, calendar, cost-aware | event timing, cost estimation, tax harvesting | costs, impact, tax complexity |

- LLMs work well as **view generators** for Black–Litterman (Lee et al. 2025) or as a sentiment
  input to an optimizer (Kirtac & Germano 2025; HARLF).
- **Point-in-time textual alignment.** Macro inputs use the unrevised value at its release
  timestamp. Generate regime tags online from a t−k lookback, because full-sample clustering leaks
  future volatility states.
- HRP reduces covariance-noise sensitivity by allocating across clusters first. Use Ledoit–Wolf
  shrinkage. The practical core is constraint-aware optimization (exposure, leverage, turnover)
  with explicit costs.
- **Sizing.** Kelly is fragile, so use **fractional Kelly**, shrunk estimates and volatility
  targeting. Position-sizing checklist: (1) the risk metric minimized (vol, CVaR, MaxDD), (2) the
  constraints (long-only, leverage < 1.5, sector < 20%), (3) the scaling logic (fractional Kelly,
  vol target). Treat LLM risk flags as timestamped features and verify them with a text-on/off
  ablation.
- **Rebalancing.** No-trade bands reduce churn. Separate *decision* timing from *fill* timing.
  Calendar events (earnings, FOMC) are known in advance. Shock events must be detected with
  real-time proxies observable at t (VIX spike, spread widening), never with future-annotated
  labels. Tax logic must match the account type and jurisdiction.

## 8. Risk management

LLMs hallucinate, are overconfident and can be manipulated, so guardrails must be **independent and
deterministically governed**. The engineering problem is a non-bypassable override that intercepts
errant LLM actions before they reach the market, without unacceptable delay.

- **Pre-trade:** instrument whitelists, maximum order size, leverage ceilings, liquidity checks, VaR
  limits (for example 99% one-day VaR of $1M), CVaR, concentration (≤ 10% per name), compliance.
  LLM policy-to-code translation adds non-determinism, so validate it against deterministic test
  sets. Report the VaR configuration (confidence, holding period, window) and the scenario source.
  - Portfolio variance V = wᵀΣw. Marginal ∂V/∂wᵢ = 2(Σw)ᵢ. Risk contribution RCᵢ = wᵢ(Σw)ᵢ.
    ∂σ_p/∂wᵢ = (Σw)ᵢ/σ_p.
  - Stress scenarios: equities −30%, rates +200 bp, volatility ×3 (Basel 2009). Rule engines encode
    Reg T margin and Reg NMS order protection.
- **In-trade:** greeks, duration and beta monitoring; drawdown stops (for example halt at −5% from
  peak); volatility scaling; trailing stops; anomaly detection (price spikes, volume surges, book
  imbalance; VPIN; the Flash Crash); circuit breakers; manual review before restart.
  - Keep a **deterministic gating layer** (sub-ms circuit breakers) separate from an
    **asynchronous explanation layer** (LLM cause attribution). Disclose latency budgets and
    trigger false-positive rates.
  - Write exposure rules as formulas, for example w_i(t) = min(w_i^max, risk_budget / risk_per_unit_i).
- **Post-trade:** attribution (alpha, beta, timing, selection), historical stress tests (2008),
  regime analysis, and causal splits (market vs idiosyncratic vs model error, using Granger, Shapley
  and structural breaks). Counterfactuals are labelled *hypothetical inference*, not facts. Every
  narrative claim must be grounded in immutable, replayable logs.
- **Safety layers outside the generative loop:** pre-trade exposure checks, hard position limits,
  concentration caps, drawdown triggers, and a kill switch that halts order flow when conditions or
  model behavior leave the approved envelope. Papers often mention these but rarely specify them
  reproducibly.

## 9. Learning paradigms

| Paradigm | What updates | Data | Strength | Watch out for |
|---|---|---|---|---|
| In-context learning | context | prompts, retrieved episodes (FinAgent) | immediate | context capacity, no persistence |
| SFT | weights | demonstrations (behavior cloning, multi-task, LoRA) | persistent specialization | label quality, distribution shift |
| RL | policy | interactive trajectories (FinRL-DeepSeek, Alpha², MacroHFT, EarnHFT) | sequential decisions | simulator fidelity, reward hacking |

- **Reward design is a first-order choice.** Separate immediate execution feedback (fill quality,
  slippage, rejections) from delayed portfolio feedback (P&L, drawdown, turnover, risk-adjusted
  return). State which signal is optimized, over what horizon, and how reward hacking is prevented.
- **RL reporting:** chronological splits and how hyperparameters were chosen; the cost model; and
  simulator fidelity (fill and latency, state/action/reward definitions, constraints, and whether
  the instrument and microstructure match). The main risks are simulator overfitting and policies
  that exploit backtester artifacts.

## 10. Multi-agent coordination

The open question is whether coordination beats a single agent once compute, information access
and protocol are held equal.

- **Patterns.** *Role-based*: TradingAgents (analysts, bull/bear researchers, traders, risk team,
  manager); FinCon (manager–analyst with Conceptual Verbal Reinforcement, which updates beliefs
  across episodes); TradingGroup (roles plus reflection plus data synthesis). *Hierarchical*:
  portfolio manager → sector specialists → executors; FinMem's layered memory. *Market ecology*:
  agent-based simulations (FinEvo, Hashimoto et al.).
- **Communication:** broadcast, peer-to-peer, shared memory.
- **Consensus:** voting is fast; expert weights by past accuracy; debate is slower but finds blind
  spots; dynamic schemes adapt. On latency cost, voting < expert-weighted < debate.
- **Task allocation:** by capability, by workload, or hybrid; centralized or decentralized; dynamic
  reallocation when an agent fails.
- **Conflict resolution:** negotiation (shrink the size so both profit and risk goals hold),
  arbitration by a supervisor, and authority hierarchies (the **risk manager vetoes sizing**, the
  analyst controls selection).
- **Hierarchies:** decompose goals top-down (for example "15% return, 10% vol" → 60/30/10 equities,
  bonds, cash → 50 tech names). Use constraint-based and dynamic decomposition, with clear I/O
  contracts between layers to prevent semantic drift. Feedback flows upward: bottom-up reports
  ("not enough liquidity for this size"), exception escalation with predefined triggers, and
  performance reports. Temporal abstraction: strategic agents run daily to monthly, tactical ones
  intraday, and a temporary spike shouldn't trigger a full reallocation.
- **Reporting for multi-agent claims:** cross-play (swap agents in and out to test team
  overfitting), communication constraints (bandwidth, tool-call budgets), crowding and impact
  metrics, and role ablations against a single agent with matched compute.
- **Market ecology.** Market mechanisms (continuous double auction, call auction, dealer market)
  shape which strategies survive. Reflexivity runs both ways: trend followers create positive
  feedback and bubbles, mean-reverters create negative feedback. Emergent outcomes include bubbles,
  crashes, flash crashes and tacit collusion. Treat agent-based simulation as hypothesis generation,
  calibrate it against real microstructure, and audit it for timestamp leakage and consensus
  smoothing.
- **Crowding.** Agents built on similar foundation models, corpora and CoT/RAG patterns produce
  correlated signals. The consequences: faster alpha decay, self-reinforcing trends, correlated
  exits (stop-losses firing together) and flash-crash risk. There is not yet any evidence at scale.

## 11. Adaptation and self-evolution

- *Self-evolution* is an agent-managed change that persists across decision episodes. *Structural*
  self-evolution changes code, update procedures, modules or topology. Reflection inside one
  episode doesn't count.
- **Governance for any adaptive claim:** state what can change (prompts, memory summaries, retrieval
  indices, weights), when updates trigger and over what history window. Keep immutable logs, frozen
  execution and cost semantics, and a rollback path. Evaluate the updated agent on sealed data.
- **Memory consolidation:** score memories by recency, outcome or retrieval frequency, then replay,
  compress, abstract or forget. Each summary keeps a hash or timestamp pointer to the raw logs.
  Periodically check summaries for factual errors and for future information.
- **Meta-learning:** MAML inner/outer loops; finance prototypes are mostly multi-agent evolutionary
  loops. Split tasks **by time** (for example 2010–2015 as source, 2016 as target) or by
  non-overlapping asset classes. Shuffling episodes across time produces fake "adaptation" that is
  really memorization of future regimes.
- **Five-stage self-reflection protocol:**
  1. Critique: produce actionable checks tied to the proposal (data timestamps, assumptions, costs,
     liquidity and slippage, risk constraints).
  2. Refine: run a small, bounded number of critique–revise rounds with early stopping, and log
     them.
  3. Check consistency: sample several reasoning paths and treat disagreement as a reason to defer
     or abstain. Consensus is not correctness, so validate the thresholds.
  4. Review after the trade: classify failures as prediction, execution or risk, and store short
     lessons keyed by instrument and regime.
  5. Verify before the trade with deterministic hard-constraint checks.
- **Search bias.** Unbounded reflection is a search ("testing into success"). Enforce a reflection
  budget, and never let out-of-sample feedback decide when the loop stops.

## 12. Infrastructure stack (Agentic ETF paper)

Six layers, from bottom to top. Each has strong incumbents, but nothing composes them into one
surface an agent can operate. The value is in the integration.

| Layer | Incumbents | Gap |
|---|---|---|
| Data | Bloomberg, ICE, exchange feeds, alt data | agent-ready: programmatic, normalized across asset classes, cheap enough to query every cycle |
| Reasoning/model | OpenAI, Anthropic, Google, xAI | a raw model has no position memory, no execution rights and no accountability |
| Orchestration/compute | schedulers, sandboxes, agent frameworks | trading-grade runtime with deterministic scheduling and cost accounting (a runaway loop costs money directly) |
| Execution/venue | Alpaca, IBKR, Polymarket, Jupiter/DEXs | one uniform execution surface across equities, prediction markets and tokenized assets |
| Risk/reconciliation/settlement | largely absent | authoritative book, continuous reconciliation, drift detection, **halt when book ≠ broker**, share-claim arbitration across agents sharing an account; this layer turns an agent into a fund |
| Distribution/wrapper | fund administrators, APs, exchanges | linking a software-produced NAV and holdings to a listed fund |

Scalar Field (`scalarlib`) primitives, given as a reference design:
- A venue registry with `account()`, `positions()`, `balances()`, `history()` and `trade()`, in
  compatibility groups so strategies can migrate between venues.
- Strategies with their own cash, an isolated book and NAV.
- `execute("AAPL", 10)` takes a target position and is idempotent.
- Continuous reconciliation, with automatic freeze below a reconciled floor and share claims that
  prevent double counting.
- More than 12 normalized datasets (OHLCV, options and greeks, insider, congressional and
  institutional filings, prediction-market data).
- A `query_llm()` router priced by tokens × model multiplier, and per-second metered compute.

### Classification insights (survey Appendix C, 19 systems)

- The same architecture can serve different capabilities. FinAgent and AgentGuard are both
  reactive, yet one does alpha and the other risk gating. AgentGuard's insight is that risk works
  best as a **gating mechanism inside the action architecture, not post hoc**.
- Separate the adaptation levels, since each needs different evaluation: prompt and memory updates
  (FinAgent, RAG-Fintech, FinRS, FinCon), weight-based RL (Alpha², FinRL-DeepSeek), and structural
  search (AlphaAgent, FinEvo).
- AlphaAgent's multi-agent debate is a *validation mechanism* for alpha, not a coordination
  capability. Retrieval is memory architecture, not just a way to augment prompts. Portfolio
  management is strategic reasoning integrated with execution.
- Don't classify by agent count. Name the coordination pattern: role-based, hierarchical or
  market ecology.
- Pure perception models (FinBERT, FinLlama) and generation without execution (FinGPT) are
  background: the Action Output criterion excludes pure prediction.

## 13. Open challenges (survey §13)

- **Hallucination propagation.** A fabricated earnings report can cause (1) a wrong entry, (2) a
  stop-loss cascade if other agents react to the "signal", and (3) larger losses through
  confidence-based sizing. Source binding (FinMCP, HydraRAG) shifts the risk rather than removing
  it, because source selection can also be wrong. Fact checking runs on a scale of seconds. For
  liability, document source binding, validation checks and human review.
- **Latency vs accuracy:** cascaded designs, distillation, compute allocated by decision
  importance.
- **Explainability:** MiFID II record keeping; GDPR "meaningful information"; neuro-symbolic
  methods; attention visualization; counterfactual explanations ("the trade would not have
  occurred if the FOMC minutes had been less hawkish"). Production needs decision logs, data
  lineage and controlled model and prompt changes.
- **Economics:** annual API cost = D × N × C_req. Report the model version, pricing date, token
  budget, tool charges and break-even gross P&L.
- **Human in the loop** at escalation points (large notional changes, regime breaks, tool
  failures). More autonomy calls for more logging and explicit rules for when human approval is
  required.
- **Negative results are under-reported.** The absence of failure reports is a reporting
  limitation, not evidence that the approach works.
- **Future directions:** neuro-symbolic integration (the LLM extracts structure, symbolic code does
  exact calculation and constraint satisfaction, plus program synthesis with formal verification);
  large market "wind tunnels" for crowding and systemic risk; governance of who approved, monitored
  or overrode each action; verifiable execution contracts with cryptographic audit.
