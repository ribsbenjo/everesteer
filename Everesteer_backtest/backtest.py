"""
backtest.py: walk-forward harness that replays our forecasting file (main.py)
as if it were submitting in the live Everesteer rounds.

Fill in every line marked `# FILL:` once the organizers announce the rules
(18:00-18:20). Search the file for "FILL" to find them all.

Usage
    python backtest.py --main main.py --label "baseline momentum" --rounds 60 --seed 1
    python backtest.py --main main.py --label "added reversal 0.5" --stress
    python backtest.py --main main.py --inprocess          # fast iteration, no fresh process
    python backtest.py --replay 2026-10-16 --replay-file submission.csv   # re-score one real round
    (During the contest round.py drives this file and asks for our points after each round.)

main.py contract (treat as a black box; this file never modifies it)
    forecast(prices: pd.DataFrame) -> pd.Series
        prices: wide panel, DatetimeIndex x asset ID, only data available at submission.
        returns one value per asset ID for the latest row (this round).
    signal(prices: pd.DataFrame) -> pd.DataFrame        (optional, enables the leak test)
        full date x asset signal panel; signal(prices).iloc[-1] should equal forecast(prices).

Assumptions until the rules are known (change them in CONFIG, not in the code):
    - Data is a wide price panel (date x asset), one row per bar.
    - Target is the forward return from entry (t + execution_delay bars) over `horizon` bars.
    - Metric is cross-sectional Spearman rank correlation per round (placeholder).
    - Rounds are spaced `round_step` bars apart so realised outcomes do not overlap.
    - An invalid / late / crashed submission scores `missed_round_score`.
Only pandas and numpy are used (scipy is not installed here).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import stats

BACKTEST_VERSION = "0.1"  # bump on deliberate scoring/validation/frictions changes
# The logged version also carries a hash of this file, so every tester edit is tracked automatically.
TESTER_VERSION = f"{BACKTEST_VERSION}-{hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:6]}"

# ---------------------------------------------------------------------------
# CONFIG: everything the organizers will tell us at 18:00
# ---------------------------------------------------------------------------
CONFIG = {
    # --- data ---
    "data_path": "data/prices.csv",       # FILL: path to the provided dataset
    "date_col": "date",                   # FILL: timestamp column name
    "frequency": "D",                     # FILL: D / H / 15min ... (informational)
    "data_lag": 0,                        # FILL: bars between a timestamp and when we can see it (0 = visible at t)

    # --- target ---
    "target": "return",                   # FILL: return | direction | rank | probability
    "horizon": 1,                         # FILL: forecast horizon in bars
    "execution_delay": 0,                 # FILL: bars between submission and entry (P&L metrics: usually 1)
    "round_step": None,                   # FILL: bars between rounds (None -> = horizon, non-overlapping)

    # --- scoring ---
    "metric": "spearman",                 # FILL: spearman | pearson | hit_rate | log_loss | pnl | custom
    "higher_is_better": True,             # FILL: False for log_loss / error metrics
    "missed_round_score": 0.0,            # FILL: score for invalid / late / crashed rounds
    "allow_last_good_fallback": False,    # FILL: True only if the real rules re-use our last valid submission

    # --- submission template ---
    "id_col": "asset_id",                 # FILL: exact column name from the template
    "pred_col": "prediction",             # FILL: exact column name from the template
    "pred_range": (-1.0, 1.0),            # FILL: allowed value range, None = unbounded
    "require_exact_ids": True,            # FILL: must the submission list exactly the template IDs?
    "universe_rule": "has_history",       # FILL: has_history | has_price_at_t | all_columns | template_file
    "template_path": None,                # FILL: path to template if universe_rule == template_file

    # --- deadline ---
    "time_limit_s": 60,                   # FILL: per-round compute budget we allow ourselves

    # --- frictions (only used by the pnl metric) ---
    "cost_bps": 5.0,                      # FILL: transaction cost per unit turnover, in bps
    "max_weight": 0.10,                   # FILL: per-asset absolute weight cap
    "allow_short": True,                  # FILL

    # --- leak test ---
    "leak_epochs": 20,                    # cut-off dates t tested: half round dates, half spread over the history
    "leak_kinds": ("noise", "shock", "extreme", "sign", "shuffle", "nan"),  # ways to rewrite the data after t
    "leak_gap_frac": 0.05,                # also test a copy with this share of random NaN gaps (0 = off)
    "leak_budget_s": 60,                  # stop starting new epochs after this many seconds

    # --- reporting ---
    "holdout_frac": 0.3,                  # later share of rounds kept as holdout
    "bootstrap_n": 2000,
    "log_path": "scorekeeper_log.csv",    # only the tester PC writes this file
}


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Return a wide price panel: DatetimeIndex (sorted, unique) x asset ID columns.

    FILL: adapt to the real format. If the data is long (date, asset, price),
    pivot it here: df.pivot(index=date_col, columns=asset_col, values=price_col).
    """
    df = pd.read_csv(path)
    df[CONFIG["date_col"]] = pd.to_datetime(df[CONFIG["date_col"]])
    df = df.set_index(CONFIG["date_col"]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.columns = df.columns.astype(str)
    return df.astype(float)


def visible_data(prices: pd.DataFrame, t) -> pd.DataFrame:
    """What we are allowed to see when submitting at t (respects data_lag)."""
    pos = prices.index.get_loc(t)
    last = pos - CONFIG["data_lag"]
    return prices.iloc[: last + 1]


def realised_outcome(prices: pd.DataFrame, t) -> pd.Series:
    """Per-asset outcome we are scored against for a submission at t.

    FILL: match the organizers' target definition exactly (returns vs log returns,
    open vs close, direction, residualised returns, ...).
    """
    pos = prices.index.get_loc(t)
    entry = pos + CONFIG["execution_delay"]
    exit_ = entry + CONFIG["horizon"]
    if exit_ >= len(prices):
        return pd.Series(dtype=float)
    ret = prices.iloc[exit_] / prices.iloc[entry] - 1.0
    if CONFIG["target"] == "direction":
        return np.sign(ret)
    if CONFIG["target"] == "rank":
        return ret.rank(pct=True) - 0.5
    if CONFIG["target"] == "probability":
        return (ret > 0).astype(float).where(ret.notna())
    return ret


def universe_at(prices: pd.DataFrame, t) -> list[str]:
    """Asset IDs the real template would expect at t."""
    rule = CONFIG["universe_rule"]
    if rule == "all_columns":
        return list(prices.columns)
    if rule == "template_file":
        tpl = pd.read_csv(CONFIG["template_path"])
        return tpl[CONFIG["id_col"]].astype(str).tolist()
    if rule == "has_price_at_t":
        return list(prices.columns[prices.loc[t].notna()])
    return list(prices.columns[prices.loc[:t].notna().any()])  # has_history


def round_dates(prices: pd.DataFrame, n_rounds: int) -> list:
    """Last n_rounds submission dates that still have a full realised outcome."""
    step = CONFIG["round_step"] or CONFIG["horizon"]
    last_ok = len(prices) - 1 - CONFIG["execution_delay"] - CONFIG["horizon"]
    positions = list(range(last_ok, -1, -step))[:n_rounds][::-1]
    min_history = 60 + CONFIG["data_lag"]  # FILL: longest lookback our signals need
    positions = [p for p in positions if p >= min_history]
    return [prices.index[p] for p in positions]


# ---------------------------------------------------------------------------
# 2. Running main.py
# ---------------------------------------------------------------------------
def load_module(path: str):
    spec = importlib.util.spec_from_file_location("team_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_inprocess(mod, data: pd.DataFrame) -> pd.Series:
    return mod.forecast(data.copy())


def run_subprocess(main_path: str, data: pd.DataFrame, workdir: str) -> pd.Series:
    """Fresh Python process per round: no state leaks between rounds."""
    inp = os.path.join(workdir, "in.pkl")
    out = os.path.join(workdir, "out.pkl")
    data.to_pickle(inp)
    if os.path.exists(out):
        os.remove(out)
    subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--_worker", main_path, inp, out],
        check=True, timeout=CONFIG["time_limit_s"], capture_output=True, text=True,
    )
    return pd.read_pickle(out)


def worker(main_path: str, inp: str, out: str) -> None:
    mod = load_module(main_path)
    pred = mod.forecast(pd.read_pickle(inp))
    with open(out, "wb") as f:
        pickle.dump(pred, f)


# ---------------------------------------------------------------------------
# 3. Submission format + validation (never silently fixes anything)
# ---------------------------------------------------------------------------
def to_submission(pred) -> pd.DataFrame:
    """Turn main.py's output into the template layout.

    FILL: mirror the real template (extra columns, ordering, round id ...).
    """
    if isinstance(pred, pd.DataFrame):
        return pred
    s = pd.Series(pred)
    return pd.DataFrame({CONFIG["id_col"]: s.index.astype(str), CONFIG["pred_col"]: s.values})


def validate_submission(sub: pd.DataFrame, expected_ids: list[str]) -> list[str]:
    errors = []
    id_col, pred_col = CONFIG["id_col"], CONFIG["pred_col"]
    missing_cols = {id_col, pred_col} - set(sub.columns)
    if missing_cols:
        return [f"missing columns {sorted(missing_cols)}"]
    ids = sub[id_col].astype(str)
    if ids.duplicated().any():
        errors.append(f"{ids.duplicated().sum()} duplicate asset IDs")
    exp = set(expected_ids)
    got = set(ids)
    if exp - got:
        errors.append(f"{len(exp - got)} expected IDs missing, e.g. {sorted(exp - got)[:3]}")
    if CONFIG["require_exact_ids"] and got - exp:
        errors.append(f"{len(got - exp)} unexpected IDs, e.g. {sorted(got - exp)[:3]}")
    vals = pd.to_numeric(sub[pred_col], errors="coerce")
    if vals.isna().any():
        errors.append(f"{vals.isna().sum()} NaN / non-numeric predictions")
    if np.isinf(vals).any():
        errors.append("inf in predictions")
    rng = CONFIG["pred_range"]
    if rng is not None and ((vals < rng[0]) | (vals > rng[1])).any():
        errors.append(f"predictions outside {rng}")
    if CONFIG["metric"] == "log_loss" and ((vals <= 0) | (vals >= 1)).any():
        errors.append("log_loss needs probabilities strictly in (0, 1)")
    return errors


# ---------------------------------------------------------------------------
# 4. Metrics (pluggable). FILL: replace with the organizers' exact definition.
# ---------------------------------------------------------------------------
def _align(pred: pd.Series, outcome: pd.Series):
    df = pd.concat([pred.rename("p"), outcome.rename("y")], axis=1, join="inner").dropna()
    return df["p"], df["y"]


def m_pearson(pred, outcome, state):
    p, y = _align(pred, outcome)
    return float(np.corrcoef(p, y)[0, 1]) if len(p) > 2 and p.std() > 0 and y.std() > 0 else 0.0


def m_spearman(pred, outcome, state):
    p, y = _align(pred, outcome)
    return m_pearson(p.rank(), y.rank(), state)


def m_hit_rate(pred, outcome, state):
    p, y = _align(pred, outcome)
    nz = (p != 0) & (y != 0)
    return float((np.sign(p[nz]) == np.sign(y[nz])).mean()) if nz.any() else 0.5


def m_log_loss(pred, outcome, state):
    p, y = _align(pred, outcome)  # y must be 0/1
    p = p.clip(1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def weights_from_pred(pred: pd.Series) -> pd.Series:
    w = pred.rank(pct=True) - 0.5 if CONFIG["allow_short"] else pred.clip(lower=0)
    w = w.fillna(0.0)
    if w.abs().sum() > 0:
        w = w / w.abs().sum()
    cap = CONFIG["max_weight"]
    if cap:
        w = w.clip(-cap, cap)
    return w


def m_pnl(pred, outcome, state):
    """Long/short return for one round, net of turnover cost. outcome must be a return."""
    w = weights_from_pred(pred)
    prev = state.get("prev_w", pd.Series(dtype=float))
    turnover = w.sub(prev, fill_value=0.0).abs().sum()
    state["prev_w"] = w
    gross = (w * outcome.reindex(w.index).fillna(0.0)).sum()
    return float(gross - turnover * CONFIG["cost_bps"] / 1e4)


def m_custom(pred, outcome, state):
    raise NotImplementedError("FILL: paste the organizers' scoring function here")


METRICS = {
    "spearman": m_spearman, "pearson": m_pearson, "hit_rate": m_hit_rate,
    "log_loss": m_log_loss, "pnl": m_pnl, "custom": m_custom,
}


def score(pred: pd.Series, outcome: pd.Series, state: dict) -> float:
    return METRICS[CONFIG["metric"]](pred, outcome, state)


def better(a: float, b: float) -> bool:
    return a > b if CONFIG["higher_is_better"] else a < b


# ---------------------------------------------------------------------------
# 5. Reference submissions
# ---------------------------------------------------------------------------
def ref_momentum(data: pd.DataFrame) -> pd.Series:
    """Plain 20-bar momentum, ranked and centred (data is already truncated at t)."""
    mom = data.pct_change(20, fill_method=None).iloc[-1]
    return (mom.rank(pct=True) - 0.5).fillna(0.0)


def ref_random(data: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    return pd.Series(rng.uniform(-0.5, 0.5, data.shape[1]), index=data.columns)


def adapt_to_metric(pred: pd.Series) -> pd.Series:
    """Map a centred rank forecast onto the metric's expected range (reference strategies only)."""
    if CONFIG["metric"] == "log_loss":
        return 0.5 + 0.8 * pred.clip(-0.5, 0.5)
    return pred


# ---------------------------------------------------------------------------
# 6. Stress (messy data)
# ---------------------------------------------------------------------------
def stress(data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Randomly blank out some assets near the cutoff and drop a few rows."""
    d = data.copy()
    n_assets = max(1, int(0.05 * d.shape[1]))
    cols = rng.choice(d.columns, size=n_assets, replace=False)
    d.loc[d.index[-5:], cols] = np.nan
    drop = rng.choice(len(d) - 1, size=max(1, len(d) // 100), replace=False)
    keep = np.setdiff1d(np.arange(len(d)), drop)
    return d.iloc[keep]


# ---------------------------------------------------------------------------
# 7. Leak test
# ---------------------------------------------------------------------------
def perturb(data: pd.DataFrame, after: np.ndarray, kind: str, rng: np.random.Generator) -> pd.DataFrame:
    """Copy of data with every row where `after` is True rewritten; earlier rows untouched."""
    out = data.copy()
    block = out.loc[after].to_numpy(dtype=float, copy=True)
    if kind == "noise":        # x exp(N(0, 0.2)): small moves, never crosses a far threshold
        block = block * np.exp(rng.normal(0.0, 0.2, size=block.shape))
    elif kind == "shock":      # level jump: x10 plus an offset
        block = block * 10.0 + np.nanstd(block) * 10.0
    elif kind == "extreme":    # +/-1e9: crosses almost any threshold a future value could pass
        block = np.where(rng.random(block.shape) < 0.5, 1e9, -1e9)
    elif kind == "sign":       # negative prices: flips anything return-like
        block = -block
    elif kind == "shuffle":    # same values, different order in time
        block = block[rng.permutation(len(block))]
    elif kind == "nan":        # the future disappears
        block[:] = np.nan
    else:
        raise ValueError(f"unknown leak_kinds entry: {kind!r}")
    out.loc[after] = block
    return out


def first_diff(a: pd.DataFrame, b: pd.DataFrame, t):
    """(date, asset) of the first cell on rows <= t where a and b differ (NaN == NaN), or None."""
    a, b = a.loc[:t].align(b.loc[:t], join="outer")
    same = np.isclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=1e-9, atol=1e-10, equal_nan=True)
    if same.all():
        return None
    r, c = np.argwhere(~same)[0]
    return a.index[r], a.columns[c]


def leak_epochs(prices: pd.DataFrame, dates: list, n: int, rng: np.random.Generator) -> list:
    """Half of the n cut-offs from the round dates, half spread over the history (one per slice)."""
    k = min(len(dates), n // 2)
    picked = {dates[i] for i in rng.choice(len(dates), size=k, replace=False)}
    lo, hi = 60 + CONFIG["data_lag"], len(prices) - 2  # skip the warm-up; keep a later row to perturb
    if hi > lo:
        edges = np.linspace(lo, hi, n - k + 1).astype(int)
        picked |= {prices.index[rng.integers(a, max(a + 1, b))] for a, b in zip(edges[:-1], edges[1:])}
    picked = sorted(picked)
    return [picked[i] for i in rng.permutation(len(picked))]  # random order: a budget stop still spreads out


def leak_test(mod, prices: pd.DataFrame, dates: list, rng: np.random.Generator, n: int | None = None) -> tuple[bool, str]:
    """Rewrite the data after t in several ways; signal rows <= t must not change.

    For each cut-off t (CONFIG leak_epochs):
      - each kind in CONFIG leak_kinds rewrites the rows after t; signal() must be unchanged on rows <= t,
      - signal(prices[:t]) must equal signal(prices) on rows <= t (truncation),
      - forecast(data visible at t) must equal signal(prices).loc[t].
    With leak_gap_frac > 0 the first two checks also run on a copy with random NaN gaps, where
    backfill / interpolation leaks show up. A crash on a rewritten input is reported, not counted as
    a leak. A pass only means no witness was found. Needs main.signal(); without it only determinism.
    """
    rng = np.random.default_rng(rng.integers(2**63))  # own stream: the bootstrap CI does not depend on leak settings
    n = n or CONFIG["leak_epochs"]
    if not hasattr(mod, "signal"):
        for i in rng.choice(len(dates), size=min(n, len(dates)), replace=False):
            data = visible_data(prices, dates[i])
            a, b = mod.forecast(data.copy()), mod.forecast(data.copy())
            if not a.equals(b):
                return False, f"non-deterministic forecast at {dates[i].date()}"
        return True, "determinism only (add signal() to main.py for a real leak test)"

    started, errors = time.time(), []
    kinds = CONFIG["leak_kinds"]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")  # log of negative prices etc. would flood the report
        try:
            clean = pd.DataFrame(mod.signal(prices.copy()))
            again = pd.DataFrame(mod.signal(prices.copy()))
        except Exception as e:  # noqa: BLE001
            return False, f"could not run: signal() raised {type(e).__name__}: {e}"
        if first_diff(clean, again, prices.index[-1]) is not None:
            return False, "signal() gives different output on identical input (fix seeds / state first)"
        panels = [("", prices, clean)]
        gap = CONFIG["leak_gap_frac"]
        if gap:
            gapped = prices.mask(rng.random(prices.shape) < gap)
            try:
                panels.append((f", {gap:.0%} gaps", gapped, pd.DataFrame(mod.signal(gapped.copy()))))
            except Exception as e:  # noqa: BLE001
                errors.append(f"gapped data: {type(e).__name__}: {e}")

        epochs = leak_epochs(prices, dates, n, rng)
        done = 0
        for t in epochs:
            if done and time.time() - started > CONFIG["leak_budget_s"]:  # always test at least one epoch
                break
            done += 1
            after = np.asarray(prices.index > t)
            for tag, data, base in panels:
                for kind in kinds:
                    try:
                        out = pd.DataFrame(mod.signal(perturb(data, after, kind, rng)))
                    except Exception as e:  # noqa: BLE001 - a crash on odd input is not a leak
                        errors.append(f"{kind}{tag} at {t.date()}: {type(e).__name__}: {e}")
                        continue
                    d = first_diff(base, out, t)
                    if d is not None:
                        return False, (f"LEAK: signal at {d[0].date()} [{d[1]}] changes when data after "
                                       f"{t.date()} changes ({kind}{tag})")
                try:
                    part = pd.DataFrame(mod.signal(data.loc[:t].copy()))
                except Exception as e:  # noqa: BLE001
                    errors.append(f"truncation{tag} at {t.date()}: {type(e).__name__}: {e}")
                    continue
                d = first_diff(base, part, t)
                if d is not None:
                    return False, (f"LEAK: signal at {d[0].date()} [{d[1]}] changes when the data is cut "
                                   f"after {t.date()} (truncation{tag})")
            # forecast(prices[:t]) must match signal(full).loc[t]
            try:
                f = mod.forecast(visible_data(prices, t).copy())
            except Exception as e:  # noqa: BLE001
                errors.append(f"forecast at {t.date()}: {type(e).__name__}: {e}")
                continue
            last_visible = visible_data(prices, t).index[-1]
            ref = clean.loc[last_visible].reindex(f.index)
            if not np.allclose(f.fillna(0).values, ref.fillna(0).values, atol=1e-8):
                return False, f"forecast() != signal().loc[t] at {t.date()} (window / truncation mismatch)"

    msg = f"passed on {done} epochs x {len(kinds)} kinds + truncation"
    if len(panels) > 1:
        msg += f" (clean + {gap:.0%} gaps)"
    if done < len(epochs):
        msg += f"; {CONFIG['leak_budget_s']}s budget hit after {done}/{len(epochs)} epochs"
    if errors:
        msg += f"; {len(errors)} run(s) raised, not counted as leaks, e.g. {errors[0][:120]}"
    return True, msg


# ---------------------------------------------------------------------------
# 8. Statistics
# ---------------------------------------------------------------------------
def summarise(s: pd.Series) -> dict:
    s = s.dropna()
    n = len(s)
    return {
        "n": n, "mean": s.mean(), "std": s.std(ddof=1),
        "stderr": s.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
        "worst": s.min() if CONFIG["higher_is_better"] else s.max(),
    }


def bootstrap_ci(diff: pd.Series, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    d = diff.dropna().values
    if len(d) < 2:
        return (np.nan, np.nan)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


# ---------------------------------------------------------------------------
# 9. Main loop
# ---------------------------------------------------------------------------
def run_backtest(args, log: bool = True, verbose: bool = True) -> dict:
    """Run the walk-forward backtest. Returns {'res': per-round scores, 'row': log row}."""
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    prices = load_data(args.data or CONFIG["data_path"])
    dates = round_dates(prices, args.rounds)
    if not dates:
        sys.exit("No round dates: not enough history for the requested horizon / lookback.")
    mod = load_module(args.main)

    # Build every round's inputs first (sequential, so --stress is reproducible with --seed)
    rounds = []
    for t in dates:
        data = visible_data(prices, t)
        if args.stress:
            data = stress(data, rng)
        rounds.append((t, data))

    # Team forecasts: one fresh process per round, several at once (results keep round order)
    def run_one(item):
        t, data = item
        started = time.time()
        try:
            if args.inprocess:
                pred = run_inprocess(mod, data)
                if time.time() - started > CONFIG["time_limit_s"]:
                    return None, "timeout"
            else:
                with tempfile.TemporaryDirectory() as workdir:
                    pred = run_subprocess(args.main, data, workdir)
            sub = to_submission(pred)
            errs = validate_submission(sub, universe_at(prices, t))
            return sub, ("invalid: " + "; ".join(errs)) if errs else "ok"
        except subprocess.TimeoutExpired:
            return None, "timeout"
        except subprocess.CalledProcessError as e:
            return None, "crash: " + (e.stderr.strip().splitlines() or ["?"])[-1]
        except Exception as e:  # noqa: BLE001 - any failure is a missed round
            return None, f"crash: {type(e).__name__}: {e}"

    if args.inprocess:
        team = [run_one(r) for r in rounds]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            team = list(pool.map(run_one, rounds))

    # Score in round order (the pnl metric carries positions from round to round)
    rows = []
    states = {"team": {}, "momentum": {}, "random": {}}
    last_good = None
    n_invalid = 0
    for (t, data), (sub, status) in zip(rounds, team):
        outcome = realised_outcome(prices, t)
        row = {"date": t, "status": status}
        if status == "ok":
            p = sub.set_index(sub[CONFIG["id_col"]].astype(str))[CONFIG["pred_col"]].astype(float)
            last_good = p
            row["team"] = score(p, outcome, states["team"])
        else:
            n_invalid += 1
            if CONFIG["allow_last_good_fallback"] and last_good is not None:
                row["team"] = score(last_good, outcome, states["team"])
            else:
                row["team"] = CONFIG["missed_round_score"]
            if verbose:
                print(f"  ! {t.date()}: {status}")

        # references see the same (possibly stressed) data
        row["momentum"] = score(adapt_to_metric(ref_momentum(data)), outcome, states["momentum"])
        row["random"] = score(adapt_to_metric(ref_random(data, rng)), outcome, states["random"])
        rows.append(row)

    res = pd.DataFrame(rows).set_index("date")
    leak_ok, leak_msg = leak_test(mod, prices, dates, rng)
    return report(res, args, leak_ok, leak_msg, n_invalid, time.time() - t0, rng, log, verbose)


def report(res, args, leak_ok, leak_msg, n_invalid, runtime, rng, log=True, verbose=True) -> dict:
    diff = res["team"] - res["momentum"]
    if not CONFIG["higher_is_better"]:
        diff = -diff
    lines = []
    for name, sl in [("all", res), ("last 60", res.tail(60)), ("last 20", res.tail(20))]:
        for strat in ["team", "momentum", "random"]:
            s = summarise(sl[strat])
            lines.append({"window": name, "strategy": strat, **s})
    table = pd.DataFrame(lines)

    n_hold = max(1, int(round(len(res) * CONFIG["holdout_frac"])))
    tune, hold = res["team"].iloc[:-n_hold], res["team"].iloc[-n_hold:]
    beat = float(np.mean([better(a, b) for a, b in zip(res["team"], res["momentum"])]))
    lo, hi = bootstrap_ci(diff, rng, CONFIG["bootstrap_n"])

    if verbose:
        pd.set_option("display.width", 140)
        print()
        print(stats.bold(f"=== {args.label}  |  {Path(args.main).name}  |  metric={CONFIG['metric']}  "
                         f"horizon={CONFIG['horizon']}  rounds={len(res)}{'  STRESS' if args.stress else ''} ==="))
        print(table.to_string(index=False, float_format=lambda x: f"{x: .4f}"))
        print(f"\nBeats momentum in {beat:.0%} of rounds | paired diff vs momentum "
              f"{diff.mean(): .4f}  95% CI [{lo: .4f}, {hi: .4f}]")
        print(f"Tuning period mean {tune.mean(): .4f} (n={len(tune)}) | holdout mean {hold.mean(): .4f} (n={n_hold})")
        if len(tune) > 1 and hold.mean() < tune.mean() - 2 * tune.std(ddof=1) / np.sqrt(n_hold):
            print(stats.warn("  WARNING: holdout much worse than tuning period: possible overfit"))
        print(stats.backtest_breakdown(res, CONFIG["metric"], CONFIG["higher_is_better"]))
        print("Leak test:", stats.ok("PASS") if leak_ok else stats.bad("FAIL"), f"({leak_msg})")
        print("Validation:", stats.ok("PASS") if n_invalid == 0 else stats.bad(f"FAIL ({n_invalid} invalid/missed rounds)"))
        print(f"Runtime {runtime:.1f}s")
        if not leak_ok:
            print(stats.bad("\n*** LEAK TEST FAILED: do not trust these scores ***"))

    log_row = {
        "row_type": "backtest",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": datetime.now().strftime("%H%M%S"),
        "label": args.label,
        "main_hash": hashlib.sha1(Path(args.main).read_bytes()).hexdigest()[:10],
        "backtest_version": TESTER_VERSION,
        "metric": CONFIG["metric"],
        "n_rounds": len(res),
        "stress": bool(args.stress),
        "bt_mean": res["team"].mean(),
        "bt_std": res["team"].std(ddof=1),
        "baseline_score": res["momentum"].mean(),
        "hit_rate_vs_baseline": beat,
        "holdout_mean": hold.mean(),
        "leak_test": "pass" if leak_ok else "fail",
        "validation": "pass" if n_invalid == 0 else f"fail({n_invalid})",
        "runtime_s": round(runtime, 1),
    }
    if log:
        log_row["snapshot"] = snapshot(log_row["run_id"], args.label, args.main)
        append_log(log_row)
        if verbose:
            print(f"Logged run {log_row['run_id']}, snapshot in {log_row['snapshot']}")
    if args.save_rounds:
        res.to_csv(args.save_rounds)
    return {"res": res, "row": log_row}


def snapshot(run_id: str, label: str, main_path: str) -> str:
    """Copy main.py and this tester next to the run, so any logged version can be restored or re-checked."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-")[:40] or "run"
    d = Path("runs") / f"{run_id}_{slug}"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(main_path, d / "main.py")
    shutil.copy2(__file__, d / "backtest.py")
    return d.as_posix()


def append_log(row: dict) -> None:
    path = Path(CONFIG["log_path"])
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path, dtype={"run_id": str, "backtest_version": str, "main_hash": str})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# 10. Replay: re-score a real submitted file on the exact real round
# ---------------------------------------------------------------------------
def replay(args) -> None:
    prices = load_data(args.data or CONFIG["data_path"])
    t = prices.index[prices.index.get_indexer([pd.Timestamp(args.replay)], method="pad")[0]]
    sub = pd.read_csv(args.replay_file)
    errs = validate_submission(sub, universe_at(prices, t))
    if errs:
        print("Validation errors:", *errs, sep="\n  ")
    outcome = realised_outcome(prices, t)
    if outcome.dropna().empty:
        sys.exit(f"No realised outcome yet for {t.date()} (need horizon + delay bars after it).")
    p = sub.set_index(sub[CONFIG["id_col"]].astype(str))[CONFIG["pred_col"]].astype(float)
    print(f"Replay score for {t.date()}: {score(p, outcome, {}):.6f}")
    print("Enter it as the replay score when round.py asks for this round's points.")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", default="main.py")
    ap.add_argument("--label", default="unlabelled")
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data", default=None, help="override CONFIG['data_path']")
    ap.add_argument("--stress", action="store_true", help="randomly drop assets / rows each round")
    ap.add_argument("--inprocess", action="store_true", help="skip the fresh process per round (faster)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="fresh processes run in parallel")
    ap.add_argument("--save-rounds", default=None, help="write per-round scores to this CSV")
    ap.add_argument("--replay", default=None, help="date of a real round to re-score")
    ap.add_argument("--replay-file", default=None, help="the submission file we actually sent")
    return ap.parse_args(argv)


def make_args(**overrides) -> argparse.Namespace:
    """CLI defaults with overrides, for calling run_backtest() from other scripts."""
    a = parse_args([])
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--_worker":
        worker(*sys.argv[2:])
        sys.exit(0)
    a = parse_args()
    if a.replay:
        replay(a)
    else:
        run_backtest(a)
