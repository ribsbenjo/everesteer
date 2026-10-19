# Sources: the five papers, what each contributes, and how far to trust it

Contents: 1. Survey (Xia et al.) · 2. Look-ahead-freedom (Fonseca) · 3. SAE (Borjigin et al.) ·
4. Poisoning Agentic Alpha (Na et al.) · 5. Agentic ETF (Singh) · 6. Further reading named in the papers

---

## 1. Agentic Trading: When LLM Agents Meet Financial Markets
Xia, You, Wang, Liu, Qi, Wu, Zhang (Shenzhen University). arXiv:2605.19337v2 [cs.AI], 4 Oct 2026, 59 pp.

**What it is.** A systematic survey of 185 studies screened through 2026-09-30 (246 candidates;
initial multi-database window to 2026-03-09, plus an arXiv-only extension 2026-03-10 → 2026-09-30
that retrieved 1,539 records, of which 154 were assessed). A *primary empirical subset* of
n = 80 studies must emit tradable actions, be evaluated closed-loop, and use an eligible
evaluation type (backtest / simulation / live / benchmark). The other 105 are background.

**Organizing lens: Architecture–Capability–Adaptation (A-C-A).**
- Architecture: perception (text, time series, multimodal), memory (working, associative vector,
  structured KB), reasoning (reactive, reflective, strategic), execution.
- Capability: alpha discovery, portfolio management, risk management.
- Adaptation: in-context learning, SFT, RL, multi-agent coordination, self-evolution.

**Key numbers (n = 80).** Time-consistent split 50/80 (62.5%); explicit transaction costs 47/80
(58.8%); universe/survivorship controls 48/80 (60.0%); execution semantics 47/80 (58.8%);
R1–R3 artifact proxy 35/80. The initial-window cohort (n = 19) was far worse: splits 10.5%,
costs 5.3%, survivorship 5.3%. Reproducibility: R0 (no runnable artifacts) 45 (56.2%),
R1 (code, not runnable) 33 (41.2%), R2 (runnable with gaps) 2 (2.5%), R3 (full package) 0.
Evaluation types: backtest 50, benchmark 14, live 9, simulation 7. Split protocol: walk-forward 5,
rolling 8, other 37, unknown 30. Peer-reviewed 19/80; 79/80 published 2024–2026.

**Contributions to reuse.** I/O contracts for reasoning and action; Layer A / Layer B memory;
outcome embargo; Alpha-to-Trade contract; position-sizing checklist; risk lifecycle; five-stage
reflection protocol; MR-1..MR-7 reporting guidelines; anti-pattern list.

**Trust.** Descriptive mapping. The coding depth differs between cohorts (audited vs single pass), so
the percentages are counts, not trends. TradingAgents is coded R2, and FinCon and TradingGroup R0.

---

## 2. Look-Ahead-Freedom as Temporal Non-Interference
Xavier Fonseca (Breda University of Applied Sciences). arXiv:2607.04958v1 [cs.CR], ACM TOSEM, Jul 2026, 30 pp.

**What it is.** A formal treatment that identifies look-ahead bias with *non-interference* from
information-flow security, using a time-indexed lattice: data available after epoch t ("high") must
not influence the decision at t ("low").

**Results.**
- A pipeline calculus that separates a datum's *availability* stamp from its *reference time*.
- Undecidability (Theorem 3): if availability may be computed from data values, look-ahead-freedom is
  Π⁰₁-complete. The proof reduces the halting problem to it. Leakage is recursively enumerable (a
  leak has a finite witness) but freedom is not. That is why detectors can show leaks and can never
  certify their absence.
- On the value-independent fragment (windows, scans, joins, resampling, normalization, as-of/vintage
  reads, agentic retrieval), a type-and-effect system is sound (Theorem 1) and decidable in linear
  time (Theorem 2). The key rule is that re-stamping changes a datum's advertised timestamp but not
  its effect, so a future value can't be relabelled as past.
- Precision ceiling (Prop. 2): the checker rejects clean pipelines whose future dependence cancels
  inside an opaque value operation (x − x) or is ignored by it. It is the most permissive sound
  discipline that treats operations as opaque.

**Evaluation.** Checker cost scales linearly in node count (log–log slope 1.023, 95% CI
[1.021, 1.026], over 10–10⁵ nodes). An early quadratic implementation measured a slope of 1.39 and
was found by profiling. Cost becomes super-linear only for deep alternating min/max availability
terms (0.11 s at 128 literals). In the oracle validation on five proprietary archetypes
(fundamentals with a reference-time leak and a restatement-vintage leak, weekly releases, intraday
bars aggregated at the session close, a survivorship-filtered universe), every accepted pipeline had
no oracle witness, every leaky variant was rejected, and the oracle *missed* the weekly-release
leak. On an adversarial corpus of 33 leaks and 20 clean pipelines: checker false negatives 0/33
(rule-of-three 95% upper bound 9.1%), two-run detector 18/33 (54.5%), tiling detector 33/33
(100%). Checker false positives 15/20, all in the algebraic-cancellation and ignored-argument cases.

**Trust.** The formal results are rigorous. The headline real-data numbers come from proprietary
data, and the public artifact reproduces the qualitative claims only. The guarantee applies to
pipelines modelled in the calculus, not to arbitrary Python.

---

## 3. Execution Is the New Attack Surface: Survivability-Aware Agentic Crypto Trading
Borjigin, Stadnyk, Bilski (True Trading), Hovorov, Pidturkina (Inc4.net). arXiv:2603.10092v1 [cs.CR], Mar 2026, 26 pp.

**What it is.** Survivability-Aware Execution (SAE) is middleware between any strategy engine
(LLM or not) and the exchange executor. It is designed for "OpenClaw-style" stacks, where LLM
intents pass through tool interception, a policy gateway and a local executor, and for
skill-marketplace supply chains (skills.sh, ClawHub).

**Concepts.** The Delegation Gap is the expected loss from actions that are executable but outside
the intended scope, measured against a logged Intended Policy Spec. The execution contract is
ExecutionRequest / ExecutionContext / ExecutionDecision (ALLOW, LIMIT or BLOCK). Budgets are
B = B₀ · g(market) · h(account) · q(trust), with projection-based enforcement and a Lipschitz
one-step loss bound.

**Results** (Binance USD-M BTCUSDT and ETHUSDT perpetuals, 15-minute bars, 2025-09-01 → 2025-12-01,
with funding):

| Variant | MDD | CVaR₀.₉₅ | CVaR₀.₉₉ | Liq. | AttackSuccess | FalseBlock | Latency ms | DG rate | DG loss |
|---|---|---|---|---|---|---|---|---|---|
| NoSAE | 0.4643 | −1.363e−3 | −4.025e−3 | 0 | 1.000 | 0.000 | 0.00135 | 0.0589 | 0.647 |
| StaticOMS | 0.1184 | −2.34e−4 | −5.62e−4 | 0 | 0.885 | 0.011 | 0.00194 | 0.0516 | 0.119 |
| Budget | 0.0325 | −5.10e−5 | −1.01e−4 | 0 | 0.760 | 0.000 | 0.00251 | 0.0422 | 0.021 |
| Budget+Cooldown | 0.0363 | −5.90e−5 | −1.03e−4 | 0 | 0.759 | 0.000 | 0.00321 | 0.0433 | 0.026 |
| Full | 0.0319 | −4.80e−5 | −1.02e−4 | 0 | 0.728 | 0.000 | 0.01029 | 0.0470 | 0.019 |

Significance: the block-bootstrap 95% CIs of the mean per-step return don't overlap (NoSAE
[−9.85, −5.01]e−5 vs Full [−2.27, −1.71]e−5). Paired Wilcoxon p = 0.0113. Two-proportion test on
AttackSuccess p = 1.76e−8.

**Trust.** Vendor paper with its own installable skill (`sae-policy-guard` on skills.sh). The
evaluation covers one window and two symbols, uses a simplified simulator, and the attack generator
and policy spec are the authors' own. Residual attack success (0.728) comes from timing- and
state-sensitive attempts near regime boundaries.

---

## 4. Poisoning Agentic Alpha: Adversarial Vulnerabilities Across Roles and Architectures in Multi-Agent Trading Systems
Na, Ni, Szpruch, Wang, Mehta (BlackRock), Nagrecha, Lopez-Lira, Choi (LinqAlpha), Y. Lee, J.-H. Lee.
arXiv:2608.24069v1 [cs.AI], Aug 2026, 16 pp. Code: github.com/cwna97/multi_agent_trading_attack.

**What it is.** The first systematic finance-specific study of where adversarial signals enter a
TradingAgents-style pipeline (Analyst → Researcher debate → Trader → Risk Manager) and how far they
survive across four communication topologies. The threat model is black-box: the attacker can alter
only source data and prompt-level content.

**Results** (GPT-4.1; AAPL, MSFT, NVDA, TSLA, BTC-USD; 2026 Q1; 61 days; 305 asset-days;
post-cutoff data only). Attack success rate, SELL-targeted / BUY-targeted: data poisoning
21.8 / 19.1, indirect injection 29.8 / 24.0, persuasive adversary 41.9 / 53.3, objective hijacking
18.0 / 13.8, jailbreaking the Risk Manager 95.5 / 98.9. APS orders the architectures from
Decentralized vote (0) through Hybrid (0.33) and Linear (0.5) to Centralized (1.0). Its correlation
with ASR is r = 0.85 for GPT-4.1 and 0.88 for Qwen3-235B. Against a single-agent baseline, pooled
BUY-targeted ASR was 70.2% vs 29.1–48.1% for the multi-agent designs (GPT-4.1) and 87.9% vs
60.1–82.2% (Qwen).

**Trust.** Academic plus industry authors, with code released. Limitations: one backbone on the role
axis, large caps only, a long/flat daily-close backtest without costs, temperature 0.

---

## 5. The Agentic ETF: How Agentic Trading Becomes an Asset Class
Amandeep Singh (Scalar Field Research). arXiv:2610.06856v1 [q-fin.TR], Jun 2026, 10 pp.

**What it is.** A conceptual and market-sizing paper. The ETF wrapper, the rise of active ETFs and
the automation of trading decisions converge into a fund whose investment process is run
end-to-end by LLM agents. It defines agentic trading, gives three properties needed to manage
capital (isolation, scheduled autonomy, accounting truth), decomposes a six-layer infrastructure
stack, and presents Scalar Field (`scalarlib`) as a reference implementation.

**Sizing.** AUM_agentic = E · a · π, with E ≈ $27T global ETF assets in 2030 and a ≈ 16% active
share, giving a $4.2T base. π = 5%, 20% and 50% give $0.21T, $0.84T and $2.10T.

**Trust.** A vendor white paper. The penetration rates are assumptions and no realized returns are
presented. The claim is about structure and feasibility, not alpha.

---

## 6. Further reading named in the papers

Leakage and validity: Sarkar & Vafa 2025 (look-ahead bias in pretrained LMs); Li et al. 2025
"Profit Mirage"; Ye et al. 2026 "The Alpha Illusion" (2605.16895); Benhenda 2026 Look-Ahead-Bench
(2601.13770); Zhang et al. 2026 one-switch benchmark (2605.23959); Zhang, Chen & Stadie 2026
Shapley claim-level contamination (2602.17234); Zhang & Stadie 2026 temporal leakage in LLM
backtesting (2608.02985); Yao & Zheng 2026 execution assumptions and reproducibility (2606.08285);
Jeon & Lee 2026 "Can blindfolded LLMs still trade?" (2603.17692); Yu et al. 2026 backtest-to-live
gap in crypto (2609.34510); Duan et al. 2026 "Can agentic trading systems pay for their own
intelligence?" (2607.10286); Wang & Saxena 2026 SoK on trading-agent failures (2609.19705).

Overfitting and statistics: Bailey, Borwein, López de Prado & Zhu 2014/2017 (backtest overfitting,
probability of backtest overfitting); Bailey & López de Prado 2014 (Deflated Sharpe Ratio); White
2000 (Reality Check); Hansen 2005 (SPA); Brown et al. 1992 (survivorship); López de Prado 2018
("10 reasons ML funds fail"); Arian et al. 2024 (out-of-sample testing methods).

Systems: TradingAgents (2412.20138), FinCon (2407.06567), FinMem (2311.13743), FinAgent
(2402.18485), TradingGroup (2508.17565), AlphaAgent (KDD'25), Alpha Jungle MCTS (2505.11122),
Alpha² (2406.16505), FinRL-DeepSeek (2502.07393), StockSim (2507.09255), FinPos, FinRS, RAPTOR.
Benchmarks: StockBench, AI-Trader, Agent Market Arena, LiveTradeBench, ContestTrade, TraderBench,
MarS, CLQT, LiveOption, OpenPM (auditable point-in-time portfolio evaluation, 2608.09988),
PredictionMarketBench, BacktestBench, FinToolBench.

Security: Greshake et al. 2023 (indirect prompt injection); PoisonedRAG; NetSafe (topological
safety); AutoRedTrader; TradeTrap; FinVault; Rizvani et al. 2026 (adversarial headlines).
