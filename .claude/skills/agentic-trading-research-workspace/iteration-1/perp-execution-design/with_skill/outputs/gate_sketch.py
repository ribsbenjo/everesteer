#!/usr/bin/env python3
"""gate_sketch.py - core deterministic pre-trade gate for an LLM agent on Binance USD-M perps.

Sketch companion to risk_execution_design.md. It is the "last mile" check that sits inside the
executor process, after every LLM / skill / LLM-risk-judge step, and before any exchange call.

Flow:  raw LLM tool call --parse_tool_call--> OrderIntent --Gate.check--> Decision (ALLOW/LIMIT/BLOCK)
                                                   ^ market state, account state (Layer A), trust state
                                                     are supplied by the executor/orchestrator, never by the LLM.

Design points (see the doc for the reasons and sources):
  * Target-position orders: the LLM states a signed target notional in USDT per symbol. It never
    sets leverage, order type, reduceOnly flags or account settings, so those cannot be escalated.
  * Budgets B = B0 * g(regime) * h(account) * q(trust), enforced by projection (clamping), after
    the SAE design of Borjigin et al. 2026 (arXiv:2603.10092). Leverage caps 3x/2x/1x follow that paper.
  * Reduce-only states (low margin, daily loss, stale data, kill switch) still let risk-reducing orders through.
  * Every decision gets an audit record chained by SHA-256.

Run:  python3 gate_sketch.py          # scenario checks + 20k-case adversarial fuzz, exits non-zero on failure
All numbers are starting defaults to calibrate on replay, not tuned values.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

ALLOW, LIMIT, BLOCK = "ALLOW", "LIMIT", "BLOCK"


# --------------------------------------------------------------------------- policy (Intended Policy Spec)
@dataclass(frozen=True)
class Policy:
    symbols: tuple = ("BTCUSDT", "ETHUSDT")
    tools: tuple = ("set_target_position", "cancel_all")
    symbol_weight: dict = field(default_factory=lambda: {"BTCUSDT": 1.0, "ETHUSDT": 0.75})
    # R: base budgets
    base_gross_leverage: float = 3.0          # gross |notional| / equity after the trade
    leverage_floor: float = 1.0               # SAE: a computed 0.5x cap is floored to 1x
    base_order_frac: float = 0.5              # max |delta notional| per order, as a fraction of equity
    min_order_usdt: float = 100.0             # below this a clamped order is dropped (exchange min notional)
    implausible_mult: float = 20.0            # |target| > 20x equity => unit error / garbage => reject
    # M: market-state tightening g(.)
    g_lev: dict = field(default_factory=lambda: {"calm": 1.0, "volatile": 2 / 3, "extreme": 1 / 3})
    g_order: dict = field(default_factory=lambda: {"calm": 1.0, "volatile": 0.75, "extreme": 0.5})
    slip_cap_bps: dict = field(default_factory=lambda: {"calm": 10.0, "volatile": 20.0, "extreme": 30.0})
    close_slip_cap_bps: float = 75.0          # risk-reducing orders get a wider (still bounded) band
    cooldown_s: dict = field(default_factory=lambda: {"calm": 30.0, "volatile": 60.0, "extreme": 120.0})
    staging: dict = field(default_factory=lambda: {"calm": 1, "volatile": 4, "extreme": 5})
    rate_window_s: float = 60.0
    max_orders: dict = field(default_factory=lambda: {"calm": 10, "volatile": 6, "extreme": 3})
    max_flips_per_hour: int = 2
    min_liq_distance: dict = field(default_factory=lambda: {"calm": 0.20, "volatile": 0.30, "extreme": 0.40})
    mmr: float = 0.01                         # conservative placeholder maintenance-margin rate (real tiers are lower)
    vol_z: tuple = (1.0, 2.0)                 # realised-vol z-score thresholds volatile / extreme
    funding_abs: tuple = (0.0005, 0.0015)     # |funding per 8h| thresholds volatile / extreme
    move_5m_extreme: float = 0.02             # |5-minute return| >= 2% => extreme
    basis_extreme: float = 0.005              # |mark/index - 1| >= 0.5% => extreme
    max_data_age_s: float = 2.0
    # U: account-state rules (Binance margin ratio = maintenance margin / margin balance; 1.0 = liquidation)
    mr_tighten: float = 0.15
    mr_reduce_only: float = 0.30
    dd_tighten: float = 0.10                  # drawdown from equity peak
    dd_halt: float = 0.20
    daily_loss_reduce_only: float = 0.03
    h_factor: float = 0.5
    # q: trust tightening (leverage factor, order-size factor)
    q_low: tuple = (0.5, 0.4)                 # injection alert or provenance < 0.5
    q_mid: tuple = (0.75, 0.75)               # provenance < 0.8


# --------------------------------------------------------------------------- contract
@dataclass
class OrderIntent:
    tool: str
    symbol: str = ""
    target_notional_usdt: float = 0.0         # signed: + long, - short, 0 flat
    max_slippage_bps: float = 10.0
    client_id: str = ""


@dataclass
class MarketState:                            # computed by the executor from exchange data
    mark: float
    vol_z: float = 0.0
    funding_8h: float = 0.0
    ret_5m: float = 0.0
    basis: float = 0.0                        # mark / index - 1
    data_age_s: float = 0.0


@dataclass
class AccountState:                           # Layer A: from the exchange + reconciler, read-only to the LLM
    equity: float
    positions: dict                           # symbol -> signed notional USDT
    margin_ratio: float = 0.02
    drawdown: float = 0.0
    daily_pnl_frac: float = 0.0
    halted: bool = False                      # kill switch / reconciliation mismatch


@dataclass
class TrustState:                             # built by the orchestrator from what entered the LLM context
    provenance: float = 1.0                   # min provenance of skills whose output was in context
    injection_alert: bool = False


@dataclass
class Decision:
    decision: str
    reason: str
    target_notional_usdt: float = 0.0
    delta_usdt: float = 0.0
    slippage_bps: float = 0.0
    limit_price: float = 0.0
    staging_slices: int = 1
    budgets: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- step 1: strict schema parse
_FIELDS = {"tool": str, "symbol": str, "target_notional_usdt": float, "max_slippage_bps": float, "client_id": str}


def parse_tool_call(raw) -> tuple[Optional[OrderIntent], str]:
    """Reject anything that is not exactly the expected shape. No coercion of strings to numbers."""
    if not isinstance(raw, dict):
        return None, "tool call is not an object"
    extra = set(raw) - set(_FIELDS)
    if extra:
        return None, f"unknown fields {sorted(extra)}"
    if not isinstance(raw.get("tool"), str):
        return None, "missing tool"
    out = {}
    for k, typ in _FIELDS.items():
        if k not in raw:
            continue
        v = raw[k]
        if typ is float:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                return None, f"{k} must be a finite number"
            v = float(v)
        elif not isinstance(v, str) or len(v) > 64:
            return None, f"{k} must be a short string"
        out[k] = v
    if out.get("tool") == "set_target_position":
        if "target_notional_usdt" not in out or "client_id" not in out:
            return None, "set_target_position needs target_notional_usdt and client_id"
        if not 0.0 <= out.get("max_slippage_bps", 10.0) <= 1000.0:
            return None, "max_slippage_bps out of [0, 1000]"
    return OrderIntent(**out), "ok"


# --------------------------------------------------------------------------- step 2: deterministic gate
class Gate:
    def __init__(self, policy: Policy = Policy()):
        self.P = policy
        self.last_increase = {}                       # symbol -> time of last risk-increasing order
        self.orders = deque()                         # times of sent orders (account-wide)
        self.flips = defaultdict(deque)               # symbol -> times of direction flips
        self.seen_ids = set()                         # replay / duplicate protection
        self.audit = []
        self._prev_hash = "0" * 64

    # regime, budgets ---------------------------------------------------------
    def regime(self, m: MarketState) -> str:
        P = self.P
        if (m.vol_z >= P.vol_z[1] or abs(m.funding_8h) >= P.funding_abs[1]
                or abs(m.ret_5m) >= P.move_5m_extreme or abs(m.basis) >= P.basis_extreme):
            return "extreme"
        if m.vol_z >= P.vol_z[0] or abs(m.funding_8h) >= P.funding_abs[0]:
            return "volatile"
        return "calm"

    def budgets(self, m: MarketState, a: AccountState, t: TrustState) -> dict:
        P, reg = self.P, self.regime(m)
        h = P.h_factor if (a.drawdown >= P.dd_tighten or a.margin_ratio >= P.mr_tighten) else 1.0
        qL, qO = P.q_low if (t.injection_alert or t.provenance < 0.5) else P.q_mid if t.provenance < 0.8 else (1.0, 1.0)
        lev = max(P.leverage_floor, P.base_gross_leverage * P.g_lev[reg] * h * qL)
        return dict(regime=reg, h=h, q=(qL, qO), gross_lev_cap=lev,
                    order_cap_usdt=a.equity * P.base_order_frac * P.g_order[reg] * h * qO,
                    slip_cap_bps=P.slip_cap_bps[reg], cooldown_s=P.cooldown_s[reg],
                    staging=P.staging[reg], max_orders=P.max_orders[reg], min_liq=P.min_liq_distance[reg])

    def reduce_only_reason(self, m: MarketState, a: AccountState) -> Optional[str]:
        P = self.P
        if a.halted or a.drawdown >= P.dd_halt:
            return "kill switch / drawdown halt"
        if a.margin_ratio >= P.mr_reduce_only:
            return f"margin ratio {a.margin_ratio:.2f} >= {P.mr_reduce_only}"
        if a.daily_pnl_frac <= -P.daily_loss_reduce_only:
            return f"daily loss {a.daily_pnl_frac:.1%}"
        if m.data_age_s > P.max_data_age_s:
            return f"stale market data ({m.data_age_s:.1f}s)"
        return None

    # main check ---------------------------------------------------------------
    def check(self, raw, m: MarketState, a: AccountState, t: TrustState, now: float) -> Decision:
        intent, why = parse_tool_call(raw)
        if intent is None:
            return self._log(raw, Decision(BLOCK, f"schema: {why}"), now)
        P = self.P
        if intent.tool not in P.tools:
            return self._log(raw, Decision(BLOCK, f"tool '{intent.tool}' not allowlisted"), now)
        if intent.tool == "cancel_all":                                   # always allowed: reduces risk
            return self._log(raw, Decision(ALLOW, "cancel_all"), now)
        if intent.symbol not in P.symbols:
            return self._log(raw, Decision(BLOCK, f"symbol '{intent.symbol}' not allowlisted"), now)
        if intent.client_id in self.seen_ids:
            return self._log(raw, Decision(BLOCK, "duplicate client_id (replay)"), now)
        if abs(intent.target_notional_usdt) > P.implausible_mult * a.equity:
            return self._log(raw, Decision(BLOCK, "implausible size (unit error or garbage)"), now)

        b = self.budgets(m, a, t)
        cur = a.positions.get(intent.symbol, 0.0)
        tgt = intent.target_notional_usdt
        flip = cur * tgt < 0
        increasing = flip or abs(tgt) > abs(cur) + 1e-9
        reasons = [b["regime"]] if b["regime"] != "calm" else []

        if not increasing:                                                # pure reduction: let it through
            return self._send(raw, intent, cur, tgt, b, m, now, ALLOW, "risk-reducing", close=True)

        ro = self.reduce_only_reason(m, a)
        if ro:                                                            # project to the reducing part only
            if cur != 0.0:
                return self._send(raw, intent, cur, 0.0, b, m, now, LIMIT, f"reduce-only ({ro}): flatten only", close=True)
            return self._log(raw, Decision(BLOCK, f"reduce-only state: {ro}", budgets=b), now)

        last = self.last_increase.get(intent.symbol)
        if last is not None and now - last < b["cooldown_s"]:
            return self._log(raw, Decision(BLOCK, f"cooldown {b['cooldown_s']:.0f}s", budgets=b), now)
        while self.orders and now - self.orders[0] > P.rate_window_s:
            self.orders.popleft()
        if len(self.orders) >= b["max_orders"]:
            return self._log(raw, Decision(BLOCK, f"order rate {b['max_orders']}/{P.rate_window_s:.0f}s", budgets=b), now)
        fq = self.flips[intent.symbol]
        while fq and now - fq[0] > 3600:
            fq.popleft()
        if flip and len(fq) >= P.max_flips_per_hour:                     # too many reversals: flatten only
            return self._send(raw, intent, cur, 0.0, b, m, now, LIMIT, "flip limit: flatten only", close=True)

        # projection onto the budget region ---------------------------------
        sign = 1.0 if tgt > 0 else -1.0
        sym_cap = b["gross_lev_cap"] * a.equity * P.symbol_weight[intent.symbol]
        other = sum(abs(v) for s, v in a.positions.items() if s != intent.symbol)
        gross_room = max(0.0, b["gross_lev_cap"] * a.equity - other)
        new_abs = min(abs(tgt), sym_cap, gross_room)
        base = cur if not flip else 0.0                                    # a flip first closes, then opens
        new_abs = min(new_abs, abs(base) + b["order_cap_usdt"])           # per-order delta cap (opening leg)
        # liquidation distance (worst case: all positions move against us together)
        gross_post = other + new_abs
        if gross_post > 0:
            liq_dist = (a.equity - P.mmr * gross_post) / gross_post
            if liq_dist < b["min_liq"]:
                max_gross = a.equity / (b["min_liq"] + P.mmr)
                new_abs = max(0.0, min(new_abs, max_gross - other))
        new_tgt = sign * new_abs
        if abs(new_tgt) - abs(base) < P.min_order_usdt and not flip:
            return self._log(raw, Decision(BLOCK, "projected order below minimum (no room in budget)", budgets=b), now)
        clamped = abs(new_tgt - tgt) > 1e-6 or intent.max_slippage_bps > b["slip_cap_bps"]
        if clamped:
            reasons.append("projected to budget")
        if b["q"][0] < 1.0:
            reasons.append("low-trust context")
        dec = LIMIT if (clamped or b["staging"] > 1 or b["q"][0] < 1.0) else ALLOW
        self.last_increase[intent.symbol] = now
        if flip:
            fq.append(now)
        return self._send(raw, intent, cur, new_tgt, b, m, now, dec, " + ".join(reasons) or "within budget")

    # helpers ------------------------------------------------------------------
    def _send(self, raw, intent, cur, tgt, b, m, now, dec, reason, close=False) -> Decision:
        cap = self.P.close_slip_cap_bps if close else b["slip_cap_bps"]
        slip = min(intent.max_slippage_bps, cap)
        delta = tgt - cur
        px = m.mark * (1 + slip / 1e4) if delta > 0 else m.mark * (1 - slip / 1e4)  # IOC limit band
        self.seen_ids.add(intent.client_id)
        self.orders.append(now)
        return self._log(raw, Decision(dec, reason, tgt, delta, slip, px, 1 if close else b["staging"], b), now)

    def _log(self, raw, d: Decision, now: float) -> Decision:
        rec = dict(t=now, raw=repr(raw)[:500], decision=d.decision, reason=d.reason,
                   target=d.target_notional_usdt, delta=d.delta_usdt, slip=d.slippage_bps,
                   budgets={k: v for k, v in d.budgets.items()}, prev=self._prev_hash)
        self._prev_hash = hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()
        rec["hash"] = self._prev_hash
        self.audit.append(rec)
        return d


# --------------------------------------------------------------------------- invariants used by tests
def invariant_violation(d: Decision, intent_raw, m, a, t, gate: Gate) -> Optional[str]:
    """Hard rules that must hold for EVERY executed decision, whatever the LLM sent."""
    if d.decision == BLOCK or not isinstance(intent_raw, dict) or intent_raw.get("tool") != "set_target_position":
        return None
    P, b = gate.P, gate.budgets(m, a, t)
    sym = intent_raw.get("symbol")
    if sym not in P.symbols:
        return "executed a non-allowlisted symbol"
    cur = a.positions.get(sym, 0.0)
    new = d.target_notional_usdt
    increasing = (cur * new < 0) or abs(new) > abs(cur) + 1e-6
    if increasing and gate.reduce_only_reason(m, a):
        return "increased risk in a reduce-only state"
    if not increasing:
        return None
    gross = sum(abs(v) for s, v in a.positions.items() if s != sym) + abs(new)
    if gross > b["gross_lev_cap"] * a.equity * (1 + 1e-9) + 1e-6:
        return "gross leverage above cap"
    if d.slippage_bps > b["slip_cap_bps"] + 1e-9:
        return "slippage above cap"
    base = 0.0 if cur * new < 0 else cur
    if abs(new) - abs(base) > b["order_cap_usdt"] + 1e-6:
        return "order size above cap"
    return None


# --------------------------------------------------------------------------- self-test
def _scenarios() -> list:
    P = Policy()
    calm = MarketState(mark=60000.0)
    acct = lambda **kw: AccountState(**{"equity": 10_000.0, "positions": {"BTCUSDT": 0.0, "ETHUSDT": 0.0}, **kw})
    tc = lambda tgt, cid, sym="BTCUSDT", slip=10.0: dict(tool="set_target_position", symbol=sym,
                                                          target_notional_usdt=tgt, max_slippage_bps=slip, client_id=cid)
    cases = [
        ("normal order", tc(5000, "a1"), calm, acct(), TrustState(), ALLOW, None),
        ("escalation 15x / 500bps", tc(150_000, "a2", slip=500), calm, acct(), TrustState(), LIMIT, None),
        ("string number", dict(tc(5000, "a3"), target_notional_usdt="5000"), calm, acct(), TrustState(), BLOCK, "schema"),
        ("NaN size", tc(float("nan"), "a4"), calm, acct(), TrustState(), BLOCK, "schema"),
        ("extra field leverage", dict(tc(5000, "a5"), leverage=125), calm, acct(), TrustState(), BLOCK, "schema"),
        ("withdraw tool", dict(tool="withdraw", symbol="USDT", client_id="a6"), calm, acct(), TrustState(), BLOCK, "tool"),
        ("unlisted symbol", tc(5000, "a7", sym="PEPEUSDT"), calm, acct(), TrustState(), BLOCK, "symbol"),
        ("unit error 1e12", tc(1e12, "a8"), calm, acct(), TrustState(), BLOCK, "implausible"),
        ("low margin open", tc(5000, "a9"), calm, acct(margin_ratio=0.4), TrustState(), BLOCK, "reduce-only"),
        ("low margin close", tc(0.0, "a10"), calm, acct(margin_ratio=0.4, positions={"BTCUSDT": 8000.0}), TrustState(), ALLOW, None),
        ("low margin flip -> flatten", tc(-8000, "a11"), calm, acct(margin_ratio=0.4, positions={"BTCUSDT": 8000.0}), TrustState(), LIMIT, "flatten"),
        ("stale data", tc(5000, "a12"), MarketState(mark=60000.0, data_age_s=9), acct(), TrustState(), BLOCK, "stale"),
        ("kill switch", tc(5000, "a13"), calm, acct(halted=True), TrustState(), BLOCK, "kill"),
        ("extreme regime 5m crash", tc(30000, "a14"), MarketState(mark=60000.0, ret_5m=-0.035), acct(), TrustState(), LIMIT, None),
        ("poisoned skill context", tc(30000, "a15"), calm, acct(), TrustState(provenance=0.3, injection_alert=True), LIMIT, "low-trust"),
    ]
    return cases


def main() -> int:
    fails = 0
    print("Scenario checks")
    for name, raw, m, a, t, want, frag in _scenarios():
        g = Gate()
        d = g.check(raw, m, a, t, now=0.0)
        ok = d.decision == want and (frag is None or frag in d.reason) and invariant_violation(d, raw, m, a, t, g) is None
        fails += not ok
        print(f"  [{'ok' if ok else 'FAIL'}] {name:28s} -> {d.decision:5s} target={d.target_notional_usdt:>10.1f} "
              f"slip={d.slippage_bps:>4.0f}bps  {d.reason}")

    # temporal: flood + flip
    g, m, t = Gate(), MarketState(mark=3000.0, vol_z=1.5), TrustState()
    pos = {"BTCUSDT": 0.0, "ETHUSDT": 0.0}
    sent = 0
    for k in range(10):                                          # 10 alternating long/short calls in 10 s
        a = AccountState(equity=10_000.0, positions=dict(pos))
        d = g.check(dict(tool="set_target_position", symbol="ETHUSDT", target_notional_usdt=4000.0 * (-1) ** k,
                         max_slippage_bps=10.0, client_id=f"f{k}"), m, a, t, now=float(k))
        if d.decision != BLOCK:
            sent += 1
            pos["ETHUSDT"] = d.target_notional_usdt
    ok = sent <= 1
    fails += not ok
    print(f"  [{'ok' if ok else 'FAIL'}] {'flood/flip burst (10 in 10s)':28s} -> {sent} executed (volatile cooldown 60s)")

    # fuzz: adversarial / garbage LLM output, invariant must never break
    rng = np.random.default_rng(0)
    g = Gate()
    junk = [None, "BUY ALL", [], {"tool": 1}, {"tool": "set_target_position", "symbol": "BTCUSDT"}]
    n, viol, executed = 20_000, 0, 0
    for i in range(n):
        m = MarketState(mark=60000.0, vol_z=float(rng.choice([0.0, 1.5, 2.5])), ret_5m=float(rng.normal(0, 0.01)),
                        funding_8h=float(rng.choice([0.0001, 0.001, 0.003])), data_age_s=float(rng.choice([0.2, 5.0], p=[0.9, 0.1])))
        a = AccountState(equity=float(rng.uniform(1e3, 1e5)),
                         positions={"BTCUSDT": float(rng.normal(0, 5000)), "ETHUSDT": float(rng.normal(0, 3000))},
                         margin_ratio=float(rng.choice([0.02, 0.2, 0.5])), drawdown=float(rng.uniform(0, 0.25)),
                         daily_pnl_frac=float(rng.normal(0, 0.02)), halted=bool(rng.random() < 0.05))
        t = TrustState(provenance=float(rng.uniform(0, 1)), injection_alert=bool(rng.random() < 0.2))
        if rng.random() < 0.1:
            raw = junk[int(rng.integers(len(junk)))]
        else:
            raw = dict(tool=str(rng.choice(["set_target_position", "set_target_position", "withdraw", "transfer"])),
                       symbol=str(rng.choice(["BTCUSDT", "ETHUSDT", "SOLUSDT"])),
                       target_notional_usdt=float(rng.choice([-1, 1]) * 10 ** rng.uniform(0, 7)),
                       max_slippage_bps=float(rng.uniform(0, 1000)), client_id=f"z{i}")
        d = g.check(raw, m, a, t, now=float(i * 200))           # spaced out so cooldowns do not hide cap checks
        executed += d.decision != BLOCK
        v = invariant_violation(d, raw, m, a, t, g)
        if v:
            viol += 1
            if viol <= 3:
                print("    violation:", v, raw)
    ok = viol == 0
    fails += not ok
    print(f"  [{'ok' if ok else 'FAIL'}] {'fuzz 20k adversarial calls':28s} -> {viol} invariant violations, "
          f"{executed} executed, 95% upper bound on violation rate {3 / n:.2e} (rule of three)")

    # audit chain integrity
    prev = "0" * 64
    for rec in g.audit:
        body = {k: v for k, v in rec.items() if k != "hash"}
        assert body["prev"] == prev
        prev = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
        assert prev == rec["hash"]
    print(f"  [ok] audit hash chain verified over {len(g.audit)} records")
    print("PASS" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
