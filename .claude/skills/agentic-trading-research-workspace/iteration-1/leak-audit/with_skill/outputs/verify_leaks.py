"""Leak audit harness for features.py vs features_fixed.py.

Run from this folder:  python3 verify_leaks.py
Cells (# %%) can also be run one at a time in VS Code / Jupyter.

Checks
  1. static lint (bundled leak_check.py) on both files
  2. bundled two-run oracle + truncation test, prices perturbed, fundamentals held fixed
  3. two-input oracle: prices after t and fundamentals filed on/after t are perturbed
     (7 kinds), outputs on rows <= t (timing contract) or <= t+1 (shift-by-one contract)
     must not change; witnesses counted per feature
  4. fix ladder: the original with one fix added at a time, oracle after each step
  5. structural checks: shift identity, schema, determinism, live-row == backtest-row
  6. leak impact: daily rank IC against the scored return close t -> close t+1
"""

# %% 1. setup
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SKILL_SCRIPTS = HERE.parents[4] / "agentic-trading-research" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))
sys.path.insert(0, str(HERE))

import leak_check as lc          # noqa: E402  (bundled with the skill)
import robust_stats as rs        # noqa: E402
import features as orig          # noqa: E402  (teammate's file, unchanged copy)
import features_fixed as fixed   # noqa: E402

PRICES = pd.read_csv(HERE / "prices.csv", index_col=0, parse_dates=True).sort_index()
FUND = pd.read_csv(HERE / "fundamentals.csv", parse_dates=["period_end", "filing_date"])
FEATS = fixed.FEATURE_COLUMNS
RESULTS: dict = {}


def wide(long: pd.DataFrame) -> pd.DataFrame:
    """date x (feature, ticker) panel of a long feature frame."""
    w = long.set_index(["date", "ticker"])[FEATS].astype(float).unstack("ticker")
    return w.sort_index()


# %% 2. static lint
for name in ["features.py", "features_fixed.py"]:
    hits = lc.lint_files([str(HERE / name)])
    RESULTS[f"lint_{name}"] = [(h["line"], h["severity"], h["rule"]) for h in hits]
    print(f"lint {name}: {len(hits)} hits, {sum(h['severity'] == 'high' for h in hits)} high")
    for h in hits:
        print(f"   line {h['line']:>3} [{h['severity']}] {h['rule']}: {h['code']}")


# %% 3. fix ladder: original logic with each fix switchable (must reproduce both files exactly)
ALL_FIXES = ["universe", "fill", "mom_z", "rev_pct", "vol", "high_vol", "weekly", "eps"]


def build_variant(prices, fundamentals, fixes=(), lag=0):
    fixes = set(fixes)
    px = prices.sort_index()
    if "universe" not in fixes:
        px = px.dropna(axis=1)                              # survivorship filter
    px = px.ffill(limit=fixed.FFILL_LIMIT) if "fill" in fixes else px.bfill()
    ret = px.pct_change(fill_method=None)

    mom = px.pct_change(20, fill_method=None)
    if "mom_z" in fixes:
        e = mom.expanding(min_periods=fixed.ZSCORE_MIN_PERIODS)
        mom_z = (mom - e.mean()) / e.std()
    else:
        mom_z = (mom - mom.mean()) / mom.std()

    rev = -ret.rolling(5).sum()
    rev_pct = (rev.expanding(min_periods=fixed.RANK_MIN_PERIODS).rank(pct=True)
               if "rev_pct" in fixes else rev.rank(pct=True))

    vol = ret.rolling(21).std() if "vol" in fixes else ret.rolling(21, center=True).std()

    mkt_vol = ret.mean(axis=1).rolling(20).std()
    if "high_vol" in fixes:
        thr = mkt_vol.expanding(min_periods=fixed.REGIME_MIN_PERIODS).quantile(0.8)
        high_vol = (mkt_vol > thr).astype(float).where(mkt_vol.notna() & thr.notna())
    else:
        high_vol = (mkt_vol > mkt_vol.quantile(0.8)).astype(float)

    weekly = px.resample("W-FRI").last().pct_change(4, fill_method=None)
    weekly_mom = weekly.reindex(px.index, method="ffill" if "weekly" in fixes else "bfill")

    cs_move = ret.rank(axis=1, pct=True) - 0.5

    f = fundamentals.copy()
    f["key"] = f["filing_date"] + fixed.FILING_LAG if "eps" in fixes else f["period_end"]
    f = f.sort_values(["key", "period_end"])
    grid = px.stack(future_stack=True).rename("px").reset_index()
    grid.columns = ["date", "ticker", "px"]
    grid = pd.merge_asof(grid.sort_values("date"), f[["ticker", "key", "eps"]], left_on="date",
                         right_on="key", by="ticker", direction="backward")
    eps = grid.pivot(index="date", columns="ticker", values="eps").reindex(index=px.index, columns=px.columns)

    panels = dict(mom_z=mom_z, rev_pct=rev_pct, vol=vol, weekly_mom=weekly_mom, cs_move=cs_move,
                  eps=eps, ey=eps / px)
    panels = {k: v.shift(lag) for k, v in panels.items()}
    long = pd.DataFrame({k: v.stack(future_stack=True) for k, v in panels.items()})
    long.index.names = ["date", "ticker"]
    long = long.reset_index()
    long["high_vol"] = long["date"].map(high_vol.shift(lag))
    return long[["date", "ticker"] + FEATS].sort_values(["date", "ticker"]).reset_index(drop=True)


def same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return lc._first_diff(wide(a), wide(b), 1e-9, 1e-12) is None


assert same(build_variant(PRICES, FUND), orig.build_features(PRICES, FUND)), "rung 0 != features.py"
assert same(build_variant(PRICES, FUND, ALL_FIXES, lag=1), fixed.build_features(PRICES, FUND)), \
    "last rung != features_fixed.py"
assert same(build_variant(PRICES, FUND, ALL_FIXES, lag=0), fixed._assemble(PRICES, FUND, lag=0))
print("ladder reproduces features.py (no fixes) and features_fixed.py (all fixes, lag 1) exactly")


# %% 4. epochs: grid + random + every gap day + filing days (where coverage misses hide)
def pick_epochs(n_grid=30, n_rand=15, n_filing=20, seed=0):
    rng = np.random.default_rng(seed)
    idx = PRICES.index
    lo, hi = 30, len(idx) - 3                                    # keep a t+1 row for the strict check
    pos = set(np.linspace(lo, hi, n_grid).astype(int)) | set(rng.integers(lo, hi, n_rand))
    listed = PRICES.notna().cumsum().gt(0) & PRICES[::-1].notna().cumsum()[::-1].gt(0)
    gap_days = PRICES.isna() & listed                            # isolated missing prints
    pos |= {idx.get_loc(d) for d in idx[gap_days.any(axis=1)]}
    fdays = FUND.loc[FUND["filing_date"].isin(idx), "filing_date"].drop_duplicates()
    fdays = fdays.sample(min(n_filing, len(fdays)), random_state=seed)
    pos |= {idx.get_loc(d) for d in fdays}
    return [idx[p] for p in sorted(pos) if lo <= p <= hi]


EPOCHS = pick_epochs()
EPOCHS_LADDER = EPOCHS[::2]
KINDS = list(lc.DEFAULT_KINDS) + ["delist"]
print(f"{len(EPOCHS)} epochs (ladder: {len(EPOCHS_LADDER)}), kinds: {KINDS}")


# %% 5. two-input oracle
def perturb_prices(px, t, kind, rng):
    after = np.asarray(px.index > t)
    if kind == "delist":                                         # one random ticker stops printing after t
        out = px.copy()
        out.loc[after, rng.choice(px.columns)] = np.nan
        return out
    return lc._perturb(px, after, kind, rng)


def perturb_fund(fund, t, kind, rng):
    mask = np.asarray(fund["filing_date"] >= t)                  # filed on t may land after the close
    return lc._perturb(fund, mask, kind, rng)


def oracle(build, epochs, kinds=KINDS, source="prices", contract_lag=0, seed=0):
    """Count, per feature, the (epoch, kind) runs whose outputs on rows <= t + contract_lag changed."""
    rng = np.random.default_rng(seed)
    base = wide(build(PRICES, FUND))
    idx = PRICES.index
    count = {c: 0 for c in FEATS}
    first = {}
    errors, runs = 0, 0
    for t in epochs:
        edge = idx[idx.get_loc(t) + contract_lag]
        for kind in kinds:
            if source == "fundamentals" and kind == "delist":
                continue
            runs += 1
            try:
                if source == "prices":
                    out = build(perturb_prices(PRICES, t, kind, rng), FUND)
                else:
                    out = build(PRICES, perturb_fund(FUND, t, kind, rng))
                pert = wide(out)
            except Exception:
                errors += 1
                continue
            for c in FEATS:
                a = base[c].loc[:edge] if c in base else pd.DataFrame()
                b = pert[c].loc[:edge] if c in pert else pd.DataFrame()
                d = lc._first_diff(a, b, 1e-9, 1e-12)
                if d is not None:
                    count[c] += 1
                    first.setdefault(c, (str(t.date()), kind, str(d[0].date()), d[1]))
    return dict(runs=runs, errors=errors, count=count, first=first)


def show(label, rep):
    leaky = {c: n for c, n in rep["count"].items() if n}
    status = "PASS" if not leaky else "FAIL"
    print(f"  {label:<46} {status}  runs={rep['runs']} errors={rep['errors']} witnesses={leaky or '-'}")
    for c, w in rep["first"].items():
        print(f"      {c}: first witness epoch {w[0]} kind={w[1]} changed row {w[2]} ticker {w[3]}")


# %% 6. main oracle runs: original vs fixed, both sources, both contracts
ship = fixed.build_features
lag0 = lambda p, f: fixed._assemble(p, f, lag=0)   # noqa: E731  causal transforms, no extra shift

RUNS = {
    "original | prices | rows<=t": (orig.build_features, "prices", 0),
    "original | fundamentals | rows<=t": (orig.build_features, "fundamentals", 0),
    "fixed lag0 | prices | rows<=t": (lag0, "prices", 0),
    "fixed lag0 | fundamentals | rows<=t": (lag0, "fundamentals", 0),
    "fixed lag0 | prices | rows<=t+1 (power check)": (lag0, "prices", 1),
    "shipped lag1 | prices | rows<=t+1": (ship, "prices", 1),
    "shipped lag1 | fundamentals | rows<=t+1": (ship, "fundamentals", 1),
}
print("two-input oracle")
RESULTS["oracle"] = {}
for label, (fn, src, cl) in RUNS.items():
    rep = oracle(fn, EPOCHS, source=src, contract_lag=cl)
    RESULTS["oracle"][label] = rep
    show(label, rep)


# %% 7. bundled oracle + truncation test (prices perturbed, fundamentals fixed), as the skill ships them
def as_panel(build):
    return lambda px: wide(build(px, FUND))


print("bundled leak_check (prices only)")
RESULTS["bundled"] = {}
for label, fn in [("original", orig.build_features), ("shipped", ship)]:
    o = lc.two_run_oracle(as_panel(fn), PRICES, n_epochs=40, seed=1)
    tr = lc.truncation_test(as_panel(fn), PRICES, n_epochs=40, seed=1)
    RESULTS["bundled"][label] = dict(oracle_ok=o["ok"], oracle_leaks=len(o["leaks"]), oracle_errors=len(o["errors"]),
                                     trunc_ok=tr["ok"], trunc_leaks=len(tr["leaks"]), trunc_errors=len(tr["errors"]))
    print(f"  {label:<9} oracle {'PASS' if o['ok'] else 'FAIL'} ({len(o['leaks'])} witnesses, "
          f"{len(o['errors'])} crashes over {o['epochs']} epochs x 6 kinds) | truncation "
          f"{'PASS' if tr['ok'] else 'FAIL'} ({len(tr['leaks'])} mismatches)")


# %% 8. truncation with both inputs cut at t = what the live submission at t would compute
def truncation_both(build, epochs):
    full = wide(build(PRICES, FUND))
    bad = []
    for t in epochs:
        live = wide(build(PRICES.loc[:t], FUND[FUND["filing_date"] < t]))   # only filings known by close t
        d = lc._first_diff(full.loc[:t], live.loc[:t], 1e-9, 1e-12)
        if d is not None:
            bad.append((str(t.date()), str(d[0].date()), d[1]))
    return bad


print("truncation, both inputs cut at t (live row == backtest row)")
RESULTS["trunc_both"] = {}
for label, fn in [("original", orig.build_features), ("shipped", ship)]:
    bad = truncation_both(fn, EPOCHS)
    RESULTS["trunc_both"][label] = len(bad)
    print(f"  {label:<9} {'PASS' if not bad else 'FAIL'}: {len(bad)}/{len(EPOCHS)} epochs differ"
          + (f", first {bad[0]}" if bad else ""))


# %% 9. fix ladder: add one fix at a time, re-run the oracle (rows <= t contract)
print("fix ladder (cumulative; rows<=t contract; prices + fundamentals perturbed separately)")
RESULTS["ladder"] = []
steps = [("0 original", [], 0)] + [(f"{i + 1} + {fx}", ALL_FIXES[: i + 1], 0) for i, fx in enumerate(ALL_FIXES)]
steps.append((f"{len(ALL_FIXES) + 1} + shift(1)  [shipped, rows<=t+1]", ALL_FIXES, 1))
for label, fx, lag in steps:
    fn = lambda p, f, fx=fx, lag=lag: build_variant(p, f, fx, lag)   # noqa: E731
    rp = oracle(fn, EPOCHS_LADDER, source="prices", contract_lag=lag)
    rf = oracle(fn, EPOCHS_LADDER, source="fundamentals", contract_lag=lag)
    leaky = sorted({c for c in FEATS if rp["count"][c] or rf["count"][c]})
    RESULTS["ladder"].append(dict(step=label, leaky=leaky, prices=rp["count"], fundamentals=rf["count"],
                                  errors=rp["errors"] + rf["errors"]))
    print(f"  {label:<42} leaking features: {', '.join(leaky) if leaky else 'none'}"
          + (f"  (crashes: {rp['errors'] + rf['errors']})" if rp["errors"] + rf["errors"] else ""))


# %% 10. structural checks
a0, a1 = lag0(PRICES, FUND), ship(PRICES, FUND)
w0, w1 = wide(a0), wide(a1)
shift_ok = lc._first_diff(w0.shift(1), w1, 1e-12, 1e-14) is None
schema_ok = list(a1.columns) == list(orig.build_features(PRICES, FUND).columns)
det_ok = same(ship(PRICES, FUND), ship(PRICES.copy(), FUND.copy()))
inputs_untouched = PRICES.equals(pd.read_csv(HERE / "prices.csv", index_col=0, parse_dates=True).sort_index())
RESULTS["structural"] = dict(shift_identity=shift_ok, same_columns=schema_ok, deterministic=det_ok,
                             inputs_not_mutated=inputs_untouched,
                             rows_original=len(orig.build_features(PRICES, FUND)), rows_fixed=len(a1),
                             tickers_original=int(orig.build_features(PRICES, FUND)["ticker"].nunique()),
                             tickers_fixed=int(a1["ticker"].nunique()))
print("structural:", RESULTS["structural"])


# %% 11. leak impact: daily cross-sectional rank IC against the scored return
# (the label is the only place a negative shift is allowed: it is the target, never a feature)
fwd = (PRICES.shift(-1) / PRICES - 1).stack().rename("fwd").rename_axis(["date", "ticker"])  # close t -> t+1, scored
same_day = (PRICES / PRICES.shift(1) - 1).stack().rename("same").rename_axis(["date", "ticker"])  # close t-1 -> t


def daily_ic(long, target):
    df = long.set_index(["date", "ticker"]).join(target, how="inner")
    out = {}
    for c in FEATS:
        if c == "high_vol":                                           # constant across tickers on a date
            continue
        x = df[[c, target.name]].dropna()
        r = x.groupby(level="date").rank()
        ic = r.groupby(level="date").apply(lambda g: g[c].corr(g[target.name]) if len(g) > 5 else np.nan).dropna()
        lo, hi = rs.block_bootstrap_ci(ic.to_numpy(), block=10, n_boot=2000)
        out[c] = dict(mean=round(float(ic.mean()), 4), ci=(round(lo, 4), round(hi, 4)), days=int(ic.size))
    return out


print("mean daily rank IC vs scored return (close t -> t+1), 95% block-bootstrap CI (10-day blocks)")
RESULTS["ic"] = {"original": daily_ic(orig.build_features(PRICES, FUND), fwd),
                 "fixed lag0": daily_ic(a0, fwd),
                 "shipped lag1": daily_ic(a1, fwd)}
for c in [c for c in FEATS if c != "high_vol"]:
    row = "  ".join(f"{k}: {v[c]['mean']:+.3f} [{v[c]['ci'][0]:+.3f},{v[c]['ci'][1]:+.3f}]"
                    for k, v in RESULTS["ic"].items())
    print(f"  {c:<10} {row}")

print("label-alignment hazard: cs_move IC against the SAME-day return (close t-1 -> t)")
RESULTS["ic_same_day_cs_move"] = {
    "original": daily_ic(orig.build_features(PRICES, FUND), same_day)["cs_move"],
    "shipped lag1": daily_ic(a1, same_day)["cs_move"]}
print("  ", RESULTS["ic_same_day_cs_move"])


# %% 12. save
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


(HERE / "verification_results.json").write_text(json.dumps(_jsonable(RESULTS), indent=1))
print("saved verification_results.json")
