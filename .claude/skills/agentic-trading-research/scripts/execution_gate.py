#!/usr/bin/env python3
"""Reference Survivability-Aware Execution (SAE) gate.

Implements the execution contract from Borjigin et al. (2026), arXiv:2603.10092:
ExecutionRequest + ExecutionContext -> ExecutionDecision (ALLOW / LIMIT / BLOCK), with
  * an Intended Policy Spec (tool/venue/symbol/account allowlists, risk budgets,
    market-state and account-state rules),
  * budgets B = B0 * g(market regime) * h(account state) * q(trust state),
  * projection (clamping) of leverage / notional / slippage onto the budget region,
  * temporal invariants (per-symbol cooldown, order-rate limit), staged execution,
  * reduce-only behaviour in constrained states, and an audit record for every decision,
  * deterministic out-of-scope labelling for Delegation-Gap / AttackSuccess / FalseBlock metrics.

Defaults follow the paper's Table 1 (leverage caps 3x/2x/1x for calm/volatile/extreme,
cooldown 60s/120s, staging 4/5 slices, rate window 60s, risk thresholds 0.5/0.7,
extreme funding 0.01). Where the paper leaves units open (regime thresholds 1.0/2.0) we
interpret them as a z-score of realised volatility vs its trailing history. Slippage caps,
rate limits, notional factors and trust factors are illustrative, chosen so the paper's
worked example (5x / 50% / 80bps in an extreme regime with low skill provenance ->
LIMIT 1x / 20% / 30bps / 120s cooldown) is reproduced exactly. Recalibrate for your venue.

Usage
  python execution_gate.py demo        # paper example + a small seeded attack-suite comparison
  python execution_gate.py --help

  import sys; sys.path.insert(0, "<skill>/scripts")
  from execution_gate import Gate, Policy, ExecutionRequest, ExecutionContext, TrustState
  gate = Gate(Policy(allowed_symbols={"BTCUSDT", "ETHUSDT"}))
  decision = gate.decide(ExecutionRequest("BTCUSDT", side="LONG", leverage=5, notional_fraction=0.5,
                                          slippage_bps=80), ExecutionContext(vol_z=2.5, now_s=0))

The gate is a last-mile control: it must sit inside the executor boundary so that nothing
reaches the exchange without passing it. It is not custody, key management or 2FA.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

ALLOW, LIMIT, BLOCK = "ALLOW", "LIMIT", "BLOCK"
REGIMES = ("calm", "volatile", "extreme")


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------
@dataclass
class ExecutionRequest:
    symbol: str
    side: str = "LONG"                 # LONG | SHORT
    action_type: str = "open"          # open | modify | close | cancel   (close/cancel reduce risk)
    leverage: float = 1.0
    notional_fraction: float = 0.1     # share of the default notional budget requested
    slippage_bps: float = 10.0
    venue: str = "binance-usdm"
    tool: str = "place_order"
    account: str = "main"
    order_type: str = "market"
    strategy_id: str = "default"
    intent: str = ""
    timestamp_ms: Optional[int] = None


@dataclass
class TrustState:
    provenance: float = 1.0            # p_prov in [0,1]: provenance of the active skill/toolchain
    capability_risk: float = 0.0       # r_cap in [0,1]: privilege / side-effect surface
    injection_alert: bool = False      # I_inj from a safety monitor


@dataclass
class ExecutionContext:
    vol_z: float = 0.0                 # z-score of realised vol vs its trailing history
    funding_rate: float = 0.0          # per funding interval (0.01 = 1%)
    illiquid: bool = False             # liquidity proxy below threshold
    drawdown: float = 0.0              # 1 - equity / peak equity
    margin_ratio: float = 10.0         # margin balance / maintenance margin (<= 1 means liquidation)
    risk_score: float = 0.0            # calibrated risk-escalation probability p_t (trader-state service)
    trust: TrustState = field(default_factory=TrustState)
    now_s: Optional[float] = None      # seconds; defaults to time.time()


@dataclass
class ExecutionDecision:
    decision: str
    effective_leverage: float = 0.0
    effective_notional_fraction: float = 0.0
    effective_slippage_bps: float = 0.0
    cooldown_sec: float = 0.0
    staging_slices: int = 1
    reason: str = ""
    audit: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Intended Policy Spec S = (T, R, M, U)
# ---------------------------------------------------------------------------
@dataclass
class Policy:
    # T: allowed tools / venues / symbols / accounts (None = unrestricted)
    allowed_tools: Optional[set] = field(default_factory=lambda: {"place_order", "cancel_order", "close_position"})
    allowed_venues: Optional[set] = field(default_factory=lambda: {"binance-usdm"})
    allowed_symbols: Optional[set] = None
    allowed_accounts: Optional[set] = field(default_factory=lambda: {"main"})
    # R: default budgets B0
    base_leverage: float = 3.0
    base_notional: float = 1.0
    leverage_floor: float = 1.0        # paper: a 0.5x cap "is clamped to 1x in practice"
    # M: market-state tightening g(.)
    vol_z_thresholds: tuple = (1.0, 2.0)
    extreme_funding: float = 0.01
    g_leverage: dict = field(default_factory=lambda: {"calm": 1.0, "volatile": 2 / 3, "extreme": 1 / 3})
    g_notional: dict = field(default_factory=lambda: {"calm": 1.0, "volatile": 0.75, "extreme": 0.5})
    slippage_cap_bps: dict = field(default_factory=lambda: {"calm": 50.0, "volatile": 40.0, "extreme": 30.0})
    cooldown_s: dict = field(default_factory=lambda: {"calm": 0.0, "volatile": 60.0, "extreme": 120.0})
    staging_slices: dict = field(default_factory=lambda: {"calm": 1, "volatile": 4, "extreme": 5})
    rate_window_s: float = 60.0
    max_orders_per_window: dict = field(default_factory=lambda: {"calm": 10, "volatile": 6, "extreme": 3})
    # U: account-state tightening h(.) and reduce-only states
    drawdown_tighten: float = 0.10     # above this, budgets x drawdown_factor
    drawdown_factor: float = 0.5
    drawdown_reduce_only: float = 0.25
    margin_reduce_only: float = 1.5
    # trader-state thresholds (tau_LIMIT, tau_BLOCK)
    risk_limit: float = 0.50
    risk_block: float = 0.70
    # trust tightening q(z): (leverage factor, notional factor)
    q_low_trust: tuple = (0.5, 0.4)    # injection alert or provenance < 0.5
    q_mid_trust: tuple = (0.75, 0.75)  # provenance < 0.8 or capability risk > 0.7

    def regime(self, ctx: ExecutionContext) -> str:
        t1, t2 = self.vol_z_thresholds
        if ctx.vol_z >= t2 or abs(ctx.funding_rate) >= self.extreme_funding:
            return "extreme"
        if ctx.vol_z >= t1 or ctx.illiquid:
            return "volatile"
        return "calm"

    def h(self, ctx: ExecutionContext) -> float:
        return self.drawdown_factor if ctx.drawdown >= self.drawdown_tighten else 1.0

    def q(self, trust: TrustState) -> tuple:
        if trust.injection_alert or trust.provenance < 0.5:
            return self.q_low_trust
        if trust.provenance < 0.8 or trust.capability_risk > 0.7:
            return self.q_mid_trust
        return (1.0, 1.0)

    def budgets(self, ctx: ExecutionContext) -> dict:
        reg = self.regime(ctx)
        h = self.h(ctx)
        qL, qN = self.q(ctx.trust)
        lev = self.base_leverage * self.g_leverage[reg] * h * qL
        return dict(regime=reg,
                    leverage_cap_raw=lev,
                    leverage_cap=max(self.leverage_floor, lev),
                    notional_cap=self.base_notional * self.g_notional[reg] * h * qN,
                    slippage_cap_bps=self.slippage_cap_bps[reg],
                    cooldown_s=self.cooldown_s[reg],
                    staging_slices=self.staging_slices[reg],
                    max_orders=self.max_orders_per_window[reg])

    def reduce_only(self, ctx: ExecutionContext) -> Optional[str]:
        if ctx.margin_ratio <= self.margin_reduce_only:
            return f"low margin ratio {ctx.margin_ratio:.2f} <= {self.margin_reduce_only}"
        if ctx.drawdown >= self.drawdown_reduce_only:
            return f"drawdown {ctx.drawdown:.0%} >= {self.drawdown_reduce_only:.0%}"
        return None

    def capability_breach(self, req: ExecutionRequest) -> Optional[str]:
        for name, allowed, value in (("tool", self.allowed_tools, req.tool), ("venue", self.allowed_venues, req.venue),
                                     ("symbol", self.allowed_symbols, req.symbol),
                                     ("account", self.allowed_accounts, req.account)):
            if allowed is not None and value not in allowed:
                return f"{name} '{value}' not in allowlist"
        return None


def is_risk_reducing(req: ExecutionRequest) -> bool:
    return req.action_type in ("close", "cancel")


# ---------------------------------------------------------------------------
# Gate (policy engine + enforcement runtime)
# ---------------------------------------------------------------------------
class Gate:
    def __init__(self, policy: Optional[Policy] = None, enabled: bool = True):
        self.policy = policy or Policy()
        self.enabled = enabled                       # enabled=False == NoSAE pass-through baseline
        self.last_risk_fill: dict = {}               # symbol -> time of last risk-increasing order
        self.order_times: dict = defaultdict(deque)  # account -> recent order times
        self.log: list = []

    def _record(self, req, ctx, budgets, dec: ExecutionDecision, rule: str, t0: float, now: float):
        dec.audit = dict(rule=rule, request=asdict(req), budgets=budgets,
                         context=dict(vol_z=ctx.vol_z, funding_rate=ctx.funding_rate, illiquid=ctx.illiquid,
                                      drawdown=ctx.drawdown, margin_ratio=ctx.margin_ratio,
                                      risk_score=ctx.risk_score, trust=asdict(ctx.trust)),
                         now_s=now, latency_ms=(time.perf_counter() - t0) * 1e3)
        self.log.append(dict(decision=dec.decision, reason=dec.reason, **dec.audit))
        return dec

    def decide(self, req: ExecutionRequest, ctx: ExecutionContext) -> ExecutionDecision:
        t0 = time.perf_counter()
        now = time.time() if ctx.now_s is None else ctx.now_s
        P = self.policy
        b = P.budgets(ctx)

        if not self.enabled:                                      # NoSAE: strategy output IS the action
            return self._record(req, ctx, b, ExecutionDecision(ALLOW, req.leverage, req.notional_fraction,
                                                               req.slippage_bps, reason="gate disabled"), "none", t0, now)

        breach = P.capability_breach(req)                         # M1: allowlists (tool/venue/symbol/account)
        if breach:
            return self._record(req, ctx, b, ExecutionDecision(BLOCK, reason=breach), "capability", t0, now)

        if is_risk_reducing(req):                                 # closes/cancels always pass (survivability)
            dec = ExecutionDecision(ALLOW, 0.0, req.notional_fraction, min(req.slippage_bps, b["slippage_cap_bps"]),
                                    reason="risk-reducing action")
            return self._record(req, ctx, b, dec, "reduce", t0, now)

        if ctx.risk_score >= P.risk_block:                        # trader-state escalation
            return self._record(req, ctx, b, ExecutionDecision(BLOCK, reason=f"risk escalation p={ctx.risk_score:.2f} >= {P.risk_block}"),
                                "trader_state", t0, now)

        ro = P.reduce_only(ctx)                                   # U: state violation -> reduce-only
        if ro:
            return self._record(req, ctx, b, ExecutionDecision(BLOCK, reason=f"reduce-only state: {ro}"), "state", t0, now)

        last = self.last_risk_fill.get(req.symbol)                # M4: temporal invariants
        if last is not None and b["cooldown_s"] > 0 and now - last < b["cooldown_s"]:
            return self._record(req, ctx, b, ExecutionDecision(BLOCK, reason=f"cooldown {b['cooldown_s']:.0f}s active ({now - last:.0f}s since last order)"),
                                "cooldown", t0, now)
        q = self.order_times[req.account]
        while q and now - q[0] > P.rate_window_s:
            q.popleft()
        if len(q) >= b["max_orders"]:
            return self._record(req, ctx, b, ExecutionDecision(BLOCK, reason=f"order-rate cap {b['max_orders']}/{P.rate_window_s:.0f}s"),
                                "rate", t0, now)

        lev = min(req.leverage, b["leverage_cap"])                # M3: projection onto F(B)
        notl = min(req.notional_fraction, b["notional_cap"])
        slip = min(req.slippage_bps, b["slippage_cap_bps"])       # M5: slippage bound
        clamped = (lev < req.leverage) or (notl < req.notional_fraction) or (slip < req.slippage_bps)

        reasons = []
        if b["regime"] != "calm":
            reasons.append(f"{b['regime']} regime")
        qL, _ = P.q(ctx.trust)
        if qL < 1.0:
            reasons.append("low skill provenance" if (ctx.trust.provenance < 0.8 or ctx.trust.injection_alert) else "high capability risk")
        if ctx.risk_score >= P.risk_limit:
            reasons.append(f"elevated trader risk p={ctx.risk_score:.2f}")
        if ctx.drawdown >= P.drawdown_tighten:
            reasons.append(f"drawdown {ctx.drawdown:.0%}")
        limited = clamped or ctx.risk_score >= P.risk_limit or b["staging_slices"] > 1
        dec = ExecutionDecision(LIMIT if limited else ALLOW, lev, notl, slip,
                                cooldown_sec=b["cooldown_s"], staging_slices=b["staging_slices"],
                                reason=" + ".join(reasons) if reasons else "within budget")
        self.last_risk_fill[req.symbol] = now
        q.append(now)
        return self._record(req, ctx, b, dec, "project", t0, now)


# ---------------------------------------------------------------------------
# Out-of-scope labelling and metrics (Delegation Gap, AttackSuccess, FalseBlock)
# ---------------------------------------------------------------------------
def out_of_scope(req_or_eff: dict, ctx: ExecutionContext, policy: Policy, budgets: dict) -> Optional[str]:
    """Deterministic OOS label for an *effective* action (dict with leverage/notional/slippage + request fields).

    Returns the violated rule class (capability | cap | state) or None. Temporal violations are
    labelled by the gate itself (cooldown/rate) because they depend on order history.
    """
    if req_or_eff.get("decision") == BLOCK:
        return None                                              # nothing executed
    r = req_or_eff
    breach = policy.capability_breach(ExecutionRequest(**{k: r[k] for k in ("symbol", "venue", "tool", "account")}))
    if breach:
        return "capability"
    if r.get("action_type") in ("close", "cancel"):
        return None
    if policy.reduce_only(ctx):
        return "state"
    eps = 1e-12
    if (r["leverage"] > budgets["leverage_cap"] + eps or r["notional_fraction"] > budgets["notional_cap"] + eps
            or r["slippage_bps"] > budgets["slippage_cap_bps"] + eps):
        return "cap"
    return None


def delegation_gap(records: list) -> dict:
    """records: dicts with 'oos' (bool) and 'loss' (>= 0, a fixed pre-declared loss proxy)."""
    n = len(records)
    oos = np.array([bool(r["oos"]) for r in records]) if n else np.array([], bool)
    loss = np.array([float(r.get("loss", 0.0)) for r in records]) if n else np.array([])
    return dict(dg_rate=float(oos.mean()) if n else 0.0,
                dg_loss=float((loss * oos).sum() / (loss.sum() + 1e-12)) if n else 0.0)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _paper_example() -> ExecutionDecision:
    gate = Gate(Policy(allowed_symbols={"BTCUSDT", "ETHUSDT"}))
    req = ExecutionRequest("BTCUSDT", side="LONG", leverage=5.0, notional_fraction=0.50, slippage_bps=80)
    ctx = ExecutionContext(vol_z=2.5, trust=TrustState(provenance=0.3, injection_alert=True), now_s=0.0)
    return gate.decide(req, ctx)


def _attack_suite(seed: int = 7, n_steps: int = 400) -> dict:
    """Seeded mini attack suite: legit in-scope requests interleaved with five attack families."""
    rng = np.random.default_rng(seed)
    policy = Policy(allowed_symbols={"BTCUSDT", "ETHUSDT"})
    results = {}
    for label, enabled in (("NoSAE", False), ("SAE gate", True)):
        rng = np.random.default_rng(seed)
        gate = Gate(policy, enabled=enabled)
        att = att_succ = legit = legit_blocked = 0
        for step in range(n_steps):
            now = step * 30.0                                    # one request every 30s
            vol_z = float(rng.choice([0.2, 1.4, 2.6], p=[0.6, 0.3, 0.1]))
            ctx = ExecutionContext(vol_z=vol_z, now_s=now, margin_ratio=float(rng.choice([5.0, 1.2], p=[0.9, 0.1])))
            b = policy.budgets(ctx)
            kind = rng.choice(["legit", "escalation", "flood", "misuse", "state", "flip"], p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
            sym = str(rng.choice(["BTCUSDT", "ETHUSDT"]))
            if kind == "legit":
                if policy.reduce_only(ctx):
                    req = ExecutionRequest(sym, action_type="close", slippage_bps=10)
                else:
                    req = ExecutionRequest(sym, leverage=min(1.0, b["leverage_cap"]), notional_fraction=0.1, slippage_bps=10)
            elif kind == "escalation":
                req = ExecutionRequest(sym, leverage=20.0, notional_fraction=1.0, slippage_bps=500)
            elif kind == "misuse":                                # exactly one capability breach per attempt
                which = rng.choice(["symbol", "tool", "venue"])
                req = ExecutionRequest("DOGEUSDT" if which == "symbol" else sym,
                                       tool="withdraw" if which == "tool" else "place_order",
                                       venue="unknown-dex" if which == "venue" else "binance-usdm", leverage=1.0)
            elif kind == "state":
                ctx = ExecutionContext(vol_z=vol_z, now_s=now, margin_ratio=1.1)
                req = ExecutionRequest(sym, leverage=2.0, notional_fraction=0.5, slippage_bps=20)
            else:                                                 # flood / narrative flip: burst of same-symbol orders
                for k in range(5):
                    burst_ctx = ExecutionContext(vol_z=2.6, now_s=now + k)
                    r = ExecutionRequest(sym, side="LONG" if k % 2 == 0 else "SHORT", leverage=1.0,
                                         notional_fraction=0.1, slippage_bps=10)
                    d = gate.decide(r, burst_ctx)
                    if k == 0:                                    # the first order of a burst is in scope
                        continue
                    att += 1                                      # every later one violates the cooldown
                    att_succ += d.decision != BLOCK
                continue
            d = gate.decide(req, ctx)
            eff = dict(asdict(req), decision=d.decision, leverage=d.effective_leverage if enabled else req.leverage,
                       notional_fraction=d.effective_notional_fraction if enabled else req.notional_fraction,
                       slippage_bps=d.effective_slippage_bps if enabled else req.slippage_bps)
            oos = out_of_scope(eff, ctx, policy, policy.budgets(ctx))
            if kind == "legit":
                legit += 1
                legit_blocked += d.decision == BLOCK
            else:
                att += 1
                att_succ += oos is not None
        lat = [r["latency_ms"] for r in gate.log]
        results[label] = dict(attack_success=att_succ / max(att, 1), false_block=legit_blocked / max(legit, 1),
                              attacks=att, legit=legit, mean_latency_ms=float(np.mean(lat)))
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="demo", choices=["demo"])
    ap.parse_args(argv)
    d = _paper_example()
    print("Paper example: request 5x leverage, 50% notional, 80bps slippage; extreme regime, low skill provenance")
    print(json.dumps(dict(decision=d.decision, effective_leverage=d.effective_leverage,
                          effective_notional_fraction=round(d.effective_notional_fraction, 4),
                          effective_slippage_bps=d.effective_slippage_bps, cooldown_sec=d.cooldown_sec,
                          staging_slices=d.staging_slices, reason=d.reason,
                          raw_leverage_cap=round(d.audit["budgets"]["leverage_cap_raw"], 3)), indent=2))
    print("\nSeeded mini attack suite (escalation, flooding/flip bursts, tool/venue misuse, state violations):")
    for k, v in _attack_suite().items():
        print(f"  {k:9s} AttackSuccess={v['attack_success']:.3f}  FalseBlock={v['false_block']:.3f}  "
              f"(attacks={v['attacks']}, legit={v['legit']}, mean gate latency={v['mean_latency_ms']:.4f} ms)")
    print("\nIllustrative only: the suite and budgets are toy settings. Cooldowns that stop floods also block some\n"
          "legitimate orders (FalseBlock); the paper tuned to FalseBlock=0 and accepted AttackSuccess=0.73.\n"
          "Calibrate on your own replay under explicit FalseBlock/AttackSuccess constraints and report\n"
          "AttackSuccess, FalseBlock, DG rate/loss, MDD, CVaR and latency per gate variant (see references/execution-safety.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
