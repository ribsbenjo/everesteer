"""gate_sketch.py - deterministic pre-trade gate for an LLM trading agent.

Sketch of the core check described in risk_execution_design.md.
The LLM never sends orders. It emits an *intent* (JSON). This gate is the
only path from intent to exchange order, and it fails closed: any parse
problem, missing data, stale data or limit breach means REJECT.

Run:  python3 gate_sketch.py   (runs the adversarial self-test suite)
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Config: loaded from a read-only file the LLM has no tool to modify.
# --------------------------------------------------------------------------
SYMBOLS = {
    # tick/step/minNotional mirror Binance exchangeInfo filters (refresh daily)
    "BTCUSDT": dict(tick=0.1, step=0.001, min_notional=100.0,
                    max_notional_x_equity=1.5, max_spread_bps=3.0),
    "ETHUSDT": dict(tick=0.01, step=0.001, min_notional=20.0,
                    max_notional_x_equity=1.0, max_spread_bps=5.0),
}
LIMITS = dict(
    risk_per_trade=0.005,        # 0.5% of equity lost if the stop is hit
    max_leverage=3.0,            # effective leverage on gross notional
    max_gross_x_equity=2.0,      # BTC + ETH gross notional / equity
    min_stop_pct=0.003, max_stop_pct=0.05,
    min_liq_to_stop_ratio=2.5,   # liquidation must be >= 2.5x further than stop
    max_price_dev_pct=0.005,     # limit price vs mark price band
    max_mark_age_s=2.0, max_account_age_s=5.0, max_intent_age_s=30.0,
    max_depth_share=0.10,        # order notional <= 10% of top-10 book depth
    daily_loss_halt=0.02,        # -2% equity today -> no new risk
    drawdown_kill=0.06,          # -6% from high-water mark -> flatten + human
    max_entries_per_day=20, min_entry_gap_s=60.0,
    max_abs_funding=0.0005,      # 0.05% / 8h, blocks entries on the paying side
)
REGIMES = {  # what volatility / margin state does to new risk
    "normal":   dict(size_mult=1.0, lev_cap=3.0, allow_entry=True),
    "elevated": dict(size_mult=0.5, lev_cap=2.0, allow_entry=True),
    "extreme":  dict(size_mult=0.0, lev_cap=0.0, allow_entry=False),
}
ALLOWED_KEYS = {"symbol", "side", "action", "order_type", "limit_price",
                "stop_price", "risk_fraction", "intent_id", "created_ts",
                "rationale"}
REQUIRED_KEYS = ALLOWED_KEYS - {"limit_price", "rationale", "risk_fraction"}


@dataclass
class MarketState:
    mark: float
    bid: float
    ask: float
    depth_usd_side: float      # top-10 levels on the side we take
    funding_rate: float        # current predicted 8h funding
    regime: str                # normal | elevated | extreme (from vol module)
    ts: float


@dataclass
class AccountState:
    equity: float
    hwm_equity: float
    day_start_equity: float
    margin_ratio: float        # Binance: maint margin / margin balance
    positions: dict            # symbol -> signed qty
    entries_today: int
    last_entry_ts: dict        # symbol -> ts
    seen_intent_ids: set
    ts: float


@dataclass
class Decision:
    approved: bool
    reasons: list = field(default_factory=list)
    order: dict | None = None


def _num(x) -> float:
    """Accept only real JSON numbers. Strings, bools, NaN, inf -> reject."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"not a number: {x!r}")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("non-finite number")
    return x


def _floor_step(x: float, step: float) -> float:
    return math.floor(x / step + 1e-9) * step


def margin_regime(margin_ratio: float) -> str:
    if margin_ratio >= 0.70:
        return "flatten"
    if margin_ratio >= 0.50:
        return "reduce"
    if margin_ratio >= 0.30:
        return "no_new_risk"
    return "ok"


def gate(raw: str, mkt: dict, acct: AccountState, now: float) -> Decision:
    """Fail-closed wrapper: any unexpected exception is a rejection."""
    try:
        return _gate(raw, mkt, acct, now)
    except Exception as e:  # noqa: BLE001 - deliberate: never approve on error
        return Decision(False, [f"gate error (fail closed): {type(e).__name__}"])


def _gate(raw: str, mkt: dict, acct: AccountState, now: float) -> Decision:
    rej = lambda *r: Decision(False, list(r))

    # ---- 1. Parse + schema (no repair, no coercion) ----------------------
    try:
        intent = json.loads(raw)
    except Exception:
        return rej("unparseable JSON")
    if not isinstance(intent, dict):
        return rej("intent is not an object")
    extra = set(intent) - ALLOWED_KEYS
    if extra:
        return rej(f"unknown fields {sorted(extra)}")   # e.g. 'leverage', 'override'
    missing = REQUIRED_KEYS - set(intent)
    if missing:
        return rej(f"missing fields {sorted(missing)}")
    sym = intent["symbol"]
    if not isinstance(sym, str) or sym not in SYMBOLS:  # exact match, no normalising
        return rej(f"symbol not allowlisted: {sym!r}")
    if intent["side"] not in ("BUY", "SELL"):
        return rej("bad side")
    if intent["action"] not in ("OPEN", "REDUCE", "CLOSE"):
        return rej("bad action")
    if intent["order_type"] not in ("LIMIT", "MARKET"):
        return rej("bad order_type")
    if not isinstance(intent["intent_id"], str) or not 8 <= len(intent["intent_id"]) <= 64:
        return rej("bad intent_id")
    try:
        stop = _num(intent["stop_price"])
        created = _num(intent["created_ts"])
        risk_frac = _num(intent.get("risk_fraction", LIMITS["risk_per_trade"]))
        limit_px = _num(intent["limit_price"]) if intent["order_type"] == "LIMIT" else None
    except (ValueError, KeyError) as e:
        return rej(f"numeric field invalid: {e}")
    if stop <= 0 or (limit_px is not None and limit_px <= 0):
        return rej("prices must be positive")      # found by the fuzzer below
    cfg, side = SYMBOLS[sym], intent["side"]

    # ---- 2. Freshness + idempotency --------------------------------------
    if intent["intent_id"] in acct.seen_intent_ids:
        return rej("duplicate intent_id (replay)")
    if not (0 <= now - created <= LIMITS["max_intent_age_s"]):
        return rej("intent stale or from the future")
    m = mkt.get(sym)
    if m is None or now - m.ts > LIMITS["max_mark_age_s"]:
        return rej("market data stale/missing")
    if now - acct.ts > LIMITS["max_account_age_s"]:
        return rej("account state stale")
    if not all(math.isfinite(v) and v > 0 for v in (m.mark, m.bid, m.ask, acct.equity)):
        return rej("bad market/account numbers")

    # ---- 3. Reduce / close path: always allowed if it truly reduces ------
    pos = acct.positions.get(sym, 0.0)
    if intent["action"] in ("REDUCE", "CLOSE"):
        reduces = (pos > 0 and side == "SELL") or (pos < 0 and side == "BUY")
        if not reduces:
            return rej("REDUCE/CLOSE does not reduce the position")
        return Decision(True, ["reduce-only"], dict(
            symbol=sym, side=side, type="MARKET", quantity=abs(pos),
            reduceOnly=True, newClientOrderId=intent["intent_id"]))

    # ---- 4. Account-level circuit breakers (OPEN only) -------------------
    reasons = []
    if acct.equity <= acct.hwm_equity * (1 - LIMITS["drawdown_kill"]):
        reasons.append("KILL: drawdown limit hit")
    if acct.equity <= acct.day_start_equity * (1 - LIMITS["daily_loss_halt"]):
        reasons.append("daily loss halt")
    mr = margin_regime(acct.margin_ratio)
    if mr != "ok":
        reasons.append(f"margin regime {mr}")
    if acct.entries_today >= LIMITS["max_entries_per_day"]:
        reasons.append("daily entry cap")
    if now - acct.last_entry_ts.get(sym, -1e18) < LIMITS["min_entry_gap_s"]:
        reasons.append("entry cooldown")
    reg = REGIMES.get(m.regime, REGIMES["extreme"])        # unknown regime = extreme
    if not reg["allow_entry"]:
        reasons.append(f"volatility regime {m.regime}: no entries")
    if (side == "BUY" and pos < 0) or (side == "SELL" and pos > 0):
        reasons.append("OPEN against existing position (flip) - close first")
    if reasons:
        return Decision(False, reasons)

    # ---- 5. Market-quality checks ----------------------------------------
    spread_bps = (m.ask - m.bid) / m.mark * 1e4
    if spread_bps > cfg["max_spread_bps"]:
        reasons.append(f"spread {spread_bps:.1f}bps too wide")
    pays = (side == "BUY" and m.funding_rate > 0) or (side == "SELL" and m.funding_rate < 0)
    if pays and abs(m.funding_rate) > LIMITS["max_abs_funding"]:
        reasons.append("funding too expensive on this side")
    if limit_px is not None and abs(limit_px / m.mark - 1) > LIMITS["max_price_dev_pct"]:
        reasons.append("limit price outside mark band")
    if reasons:
        return Decision(False, reasons)
    entry =limit_px if limit_px is not None else (m.ask if side == "BUY" else m.bid)

    # ---- 6. Stop sanity: correct side, sensible distance -----------------
    stop_pct = (entry - stop) / entry if side == "BUY" else (stop - entry) / entry
    if stop_pct <= 0:
        reasons.append("stop on wrong side of entry")
    elif not LIMITS["min_stop_pct"] <= stop_pct <= LIMITS["max_stop_pct"]:
        reasons.append(f"stop distance {stop_pct:.3%} out of range")
    if reasons:
        return Decision(False, reasons)

    # ---- 7. Sizing: gate computes size, LLM only proposes risk fraction --
    risk_frac = min(max(risk_frac, 0.0), LIMITS["risk_per_trade"]) * reg["size_mult"]
    notional = acct.equity * risk_frac / stop_pct
    gross_other = sum(abs(q) * mkt[s].mark for s, q in acct.positions.items()
                      if s != sym and s in mkt)
    cur_sym = abs(pos) * m.mark
    lev_cap = min(LIMITS["max_leverage"], reg["lev_cap"])
    caps = [
        cfg["max_notional_x_equity"] * acct.equity - cur_sym,
        LIMITS["max_gross_x_equity"] * acct.equity - gross_other - cur_sym,
        lev_cap * acct.equity - gross_other - cur_sym,
        LIMITS["max_depth_share"] * m.depth_usd_side,
    ]
    notional = min([notional] + caps)
    qty = _floor_step(max(notional, 0.0) / entry, cfg["step"])
    if qty * entry < cfg["min_notional"]:
        return rej("size after caps below exchange minNotional")

    # ---- 8. Liquidation distance (isolated, approximate) -----------------
    lev_eff = min(lev_cap, max(1.0, (cur_sym + qty * entry) / acct.equity))
    liq_pct = 1.0 / lev_eff - 0.005          # ~ minus maint. margin rate
    if liq_pct < LIMITS["min_liq_to_stop_ratio"] * stop_pct:
        return rej("liquidation too close relative to stop")

    order = dict(
        symbol=sym, side=side, quantity=round(qty, 6),
        type="LIMIT" if limit_px is not None else "LIMIT_IOC_PROTECTED",
        price=round(_floor_step(entry, cfg["tick"]), 8), timeInForce="GTC" if limit_px else "IOC",
        newClientOrderId=intent["intent_id"],
        attached_stop=dict(type="STOP_MARKET", stopPrice=round(_floor_step(stop, cfg["tick"]), 8),
                           closePosition=True, workingType="MARK_PRICE"),
    )
    return Decision(True, [f"approved notional={qty*entry:.0f} regime={m.regime}"], order)


# --------------------------------------------------------------------------
# Adversarial self-test: every bad intent must be rejected, the good one pass.
# --------------------------------------------------------------------------
def _fixtures(now):
    mkt = {
        "BTCUSDT": MarketState(60000, 59999.9, 60000.1, 5e6, 0.0001, "normal", now - 0.5),
        "ETHUSDT": MarketState(3000, 2999.99, 3000.01, 2e6, 0.0001, "normal", now - 0.5),
    }
    acct = AccountState(equity=10_000, hwm_equity=10_200, day_start_equity=10_050,
                        margin_ratio=0.05, positions={"BTCUSDT": 0.0, "ETHUSDT": 0.0},
                        entries_today=2, last_entry_ts={}, seen_intent_ids={"dup-00000001"},
                        ts=now - 1)
    return mkt, acct


def _intent(now, **kw):
    base = dict(symbol="BTCUSDT", side="BUY", action="OPEN", order_type="LIMIT",
                limit_price=60000.0, stop_price=59100.0, risk_fraction=0.005,
                intent_id="abc-12345678", created_ts=now - 3, rationale="momentum")
    base.update(kw)
    return json.dumps({k: v for k, v in base.items() if v is not ...})


def self_test():
    now = time.time()
    mkt, acct = _fixtures(now)
    cases = [
        ("good intent", _intent(now), True),
        ("not JSON", "BUY 10 BTC NOW", False),
        ("extra field leverage", _intent(now, leverage=100), False),
        ("override field", _intent(now, override_limits=True), False),
        ("unlisted symbol", _intent(now, symbol="PEPEUSDT"), False),
        ("lookalike symbol", _intent(now, symbol="BTCUSDT​"), False),
        ("lowercase side", _intent(now, side="buy"), False),
        ("missing stop", _intent(now, stop_price=...), False),
        ("stop as string", _intent(now, stop_price="59100"), False),
        ("NaN stop", _intent(now, stop_price=float("nan")), False),
        ("stop wrong side", _intent(now, stop_price=60500.0), False),
        ("stop too tight", _intent(now, stop_price=59990.0), False),
        ("stop too wide", _intent(now, stop_price=40000.0), False),
        ("fat-finger limit", _intent(now, limit_price=66000.0, stop_price=65000.0), False),
        ("stale intent", _intent(now, created_ts=now - 600), False),
        ("future intent", _intent(now, created_ts=now + 600), False),
        ("replayed id", _intent(now, intent_id="dup-00000001"), False),
        ("fake REDUCE (no position)", _intent(now, action="REDUCE"), False),
    ]
    # huge risk_fraction must be clamped, not rejected: check size is capped
    big = gate(_intent(now, risk_fraction=5.0), mkt, acct, now)
    ok = big.approved and big.order["quantity"] * 60000 <= 1.5 * acct.equity + 1
    print(f"{'PASS' if ok else 'FAIL'}  risk_fraction=5.0 clamped -> {big.reasons}")
    failures = 0 if ok else 1

    for name, raw, expect in cases:
        d = gate(raw, mkt, acct, now)
        good = d.approved == expect
        failures += not good
        print(f"{'PASS' if good else 'FAIL'}  {name:28s} -> {d.approved} {d.reasons}")

    # state-dependent cases
    scenarios = [
        ("extreme vol regime", dict(regime="extreme"), {}),
        ("stale mark price", dict(ts=now - 10), {}),
        ("wide spread", dict(bid=59950.0, ask=60050.0), {}),
        ("margin ratio 55%", {}, dict(margin_ratio=0.55)),
        ("daily loss -2.5%", {}, dict(equity=9_790)),
        ("drawdown -7%", {}, dict(equity=9_480, day_start_equity=9_500)),
    ]
    for name, mk, ak in scenarios:
        m2, a2 = _fixtures(now)
        for k, v in mk.items():
            setattr(m2["BTCUSDT"], k, v)
        for k, v in ak.items():
            setattr(a2, k, v)
        d = gate(_intent(now), m2, a2, now)
        good = not d.approved
        failures += not good
        print(f"{'PASS' if good else 'FAIL'}  {name:28s} -> {d.approved} {d.reasons}")

    # elevated regime halves size vs normal
    m3, _ = _fixtures(now)
    m3["BTCUSDT"].regime = "elevated"
    n = gate(_intent(now), mkt, acct, now).order["quantity"]
    e = gate(_intent(now, intent_id="abc-99999999"), m3, acct, now).order["quantity"]
    good = e <= n * 0.5 + 0.001
    failures += not good
    print(f"{'PASS' if good else 'FAIL'}  elevated regime size {e} vs normal {n}")

    # fuzz: random garbage must never be approved with an unbounded size
    import random
    rng = random.Random(0)
    for _ in range(3000):
        raw = _intent(now, side=rng.choice(["BUY", "SELL", "HOLD", 1]),
                      limit_price=rng.choice([60000.0, -1, 0, 1e308, 59990.0]),
                      stop_price=rng.uniform(-1e5, 1e5),
                      risk_fraction=rng.choice([0.001, 1, -3, 1e9]),
                      intent_id=f"fz-{rng.getrandbits(40):012x}")
        d = gate(raw, mkt, acct, now)
        if any("gate error" in r for r in d.reasons):  # crash = bug, even if safe
            failures += 1
            print("FAIL fuzz crash", d.reasons, raw)
        if d.approved and d.order.get("quantity", 0) * 60000 > 1.5 * acct.equity + 1:
            failures += 1
            print("FAIL fuzz breach", raw)
    print(f"\n{failures} failure(s)")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if self_test() else 0)
