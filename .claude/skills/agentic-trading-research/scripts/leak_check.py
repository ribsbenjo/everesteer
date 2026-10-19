#!/usr/bin/env python3
"""Look-ahead leak checks for time-indexed pandas pipelines.

Three tools, following Fonseca (2026) "Look-Ahead-Freedom as Temporal Non-Interference":

  lint     Static scan of Python source for code patterns that often break causality
           (negative shifts, centered windows, backfill, full-sample fits, forward joins...).
           Hits are review prompts, not verdicts.
  oracle   Two-run non-interference test: for many epochs t, perturb every input row after t
           in several ways and check that outputs at rows <= t do not change.
  trunc    Truncation test: build(df[:t]) must equal build(df) on rows <= t.

A pass certifies nothing: leak detection is semi-decidable, freedom is not (Theorem 3).
The dynamic tests trust the input's timestamps; data stamped by reference time instead of
availability (period-end fundamentals, restated/revised values, survivor universes) leaks
in a way no row perturbation can reveal. Declare each input's availability as well.

Usage
  python leak_check.py lint main.py features.py
  python leak_check.py oracle --module main.py --func signal --data data/prices.csv
  python leak_check.py oracle --module main.py --func signal --data prices.csv --epochs 40 \
         --kinds noise,shock,sign,shuffle,nan --trunc

Python use
  import sys; sys.path.insert(0, "<skill>/scripts"); import leak_check as lc
  report = lc.two_run_oracle(my_build_fn, prices_df, n_epochs=30)
  report = lc.truncation_test(my_build_fn, prices_df, n_epochs=30)
  hits = lc.lint_source(open("main.py").read())

Only numpy, pandas and the standard library are required.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Static lint
# ---------------------------------------------------------------------------
# (rule id, severity, regex, explanation). "high" = usually a leak; "review" = often fine,
# but only if the fit/statistic/resample is restricted to data available at t.
LINT_RULES: list[tuple[str, str, str, str]] = [
    ("negative-shift", "high", r"\.shift\(\s*(periods\s*=\s*)?-\s*\d",
     "shift(-k) pulls row t+k into row t: future data in the present."),
    ("negative-diff", "high", r"\.(pct_change|diff)\(\s*(periods\s*=\s*)?-\s*\d",
     "negative periods compare with a future row."),
    ("centered-window", "high", r"center\s*=\s*True",
     "centered rolling window reaches past t (classic centered moving-average leak)."),
    ("backfill", "high", r"\.(bfill|backfill)\(|method\s*=\s*['\"](bfill|backfill)['\"]|limit_direction\s*=\s*['\"](backward|both)['\"]",
     "backfill copies a later value into earlier rows (a re-stamped future value)."),
    ("interpolate", "review", r"\.interpolate\(",
     "interpolation between t-1 and t+1 uses the future unless strictly forward-only."),
    ("merge-asof-forward", "high", r"direction\s*=\s*['\"](forward|nearest)['\"]",
     "merge_asof forward/nearest attaches values available after the row's time."),
    ("dropna-columns", "high", r"dropna\([^)]*axis\s*=\s*(1|['\"]columns['\"])",
     "dropping assets with any NaN over the full sample is survivorship (uses future listing info)."),
    ("shuffled-split", "high", r"train_test_split\((?![^)]*shuffle\s*=\s*False)",
     "train_test_split shuffles by default: future rows land in training."),
    ("kfold", "high", r"\b(KFold|StratifiedKFold|ShuffleSplit|cross_val_score)\(",
     "non-temporal cross-validation mixes past and future; use walk-forward / TimeSeriesSplit + embargo."),
    ("fit", "review", r"\.(fit|fit_transform)\(",
     "a fitted scaler/model/regime clusterer must see only data available at t (per fold / expanding)."),
    ("time-axis-rank", "review", r"\.rank\((?![^)]*axis\s*=\s*(1|['\"]columns['\"]))",
     "rank() defaults to axis=0 (over time) on a date x asset panel: uses future rows. Use axis=1 for cross-section."),
    ("full-sample-bins", "review", r"\b(pd\.)?q?cut\(",
     "quantile/bin edges computed on the full sample leak the future distribution."),
    ("resample", "review", r"\.resample\(",
     "check label/closed: an output bar may only aggregate source rows at or before its own label time."),
]
_GLOBAL_STAT = re.compile(r"\.(mean|std|median|min|max|quantile|var)\((?![^)]*axis\s*=\s*(1|['\"]columns['\"]))")
_WINDOWED = re.compile(r"rolling\(|expanding\(|ewm\(|groupby\(|resample\(|axis\s*=\s*(1|['\"]columns['\"])")


def _strip_comment(line: str) -> str:
    """Drop a trailing # comment (naive about # inside strings, which is fine for a linter)."""
    return line.split("#", 1)[0]


def lint_source(text: str, filename: str = "<string>") -> list[dict]:
    """Return a list of {file, line, rule, severity, code, why} hits."""
    hits = []
    for no, raw in enumerate(text.splitlines(), 1):
        code = _strip_comment(raw)
        if not code.strip():
            continue
        for rule, sev, pattern, why in LINT_RULES:
            if re.search(pattern, code):
                hits.append(dict(file=filename, line=no, rule=rule, severity=sev, code=raw.strip(), why=why))
        if _GLOBAL_STAT.search(code) and not _WINDOWED.search(code):
            hits.append(dict(file=filename, line=no, rule="global-statistic", severity="review", code=raw.strip(),
                             why="a full-sample statistic (no rolling/expanding/axis=1) used as a feature or "
                                 "normaliser includes future rows."))
    return hits


def lint_files(paths: Iterable[str]) -> list[dict]:
    out = []
    for p in paths:
        out.extend(lint_source(Path(p).read_text(encoding="utf-8"), str(p)))
    return out


# ---------------------------------------------------------------------------
# 2. Dynamic tests
# ---------------------------------------------------------------------------
Builder = Callable[[pd.DataFrame], "pd.DataFrame | pd.Series"]
DEFAULT_KINDS = ("noise", "shock", "extreme", "sign", "shuffle", "nan")


def _as_frame(out) -> pd.DataFrame:
    if isinstance(out, pd.Series):
        return out.to_frame()
    if isinstance(out, pd.DataFrame):
        return out
    raise TypeError("build function must return a pandas DataFrame or Series indexed by time")


def _perturb(df: pd.DataFrame, after: np.ndarray, kind: str, rng: np.random.Generator) -> pd.DataFrame:
    """Return a copy of df with rows where `after` is True perturbed; rows <= t untouched."""
    out = df.copy()
    num = out.select_dtypes(include="number").columns
    if not after.any() or len(num) == 0:
        return out
    block = out.loc[after, num].to_numpy(dtype=float, copy=True)
    if kind == "noise":        # multiplicative noise, keeps sign (safe for prices)
        block = block * np.exp(rng.normal(0.0, 0.2, size=block.shape))
    elif kind == "shock":      # large level change (x10 plus an offset)
        block = block * 10.0 + np.nanstd(block) * 10.0
    elif kind == "extreme":    # +/-1e9: crosses almost any threshold a future value might pass through
        block = np.where(rng.random(block.shape) < 0.5, 1e9, -1e9)
    elif kind == "sign":       # flips direction of anything return-like
        block = -block
    elif kind == "shuffle":    # same values, different order in time
        block = block[rng.permutation(block.shape[0])]
    elif kind == "nan":        # future disappears
        block[:] = np.nan
    else:
        raise ValueError(f"unknown perturbation kind: {kind}")
    out.loc[after, num] = block
    return out


def _pick_epochs(index: pd.Index, n_epochs: int, min_frac: float, rng: np.random.Generator) -> list:
    n = len(index)
    lo = max(1, int(n * min_frac))
    hi = n - 2                           # need at least one future row to perturb
    if hi <= lo:
        return []
    grid = np.linspace(lo, hi, num=max(2, n_epochs // 2)).astype(int)
    rand = rng.integers(lo, hi + 1, size=max(1, n_epochs - len(grid)))
    pos = sorted(set(grid.tolist()) | set(rand.tolist()))
    return [index[p] for p in pos]


def _first_diff(a: pd.DataFrame, b: pd.DataFrame, rtol: float, atol: float):
    """First (row, column) where a and b differ (NaN == NaN), or None."""
    a, b = a.align(b, join="outer")
    av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    same = np.isclose(av, bv, rtol=rtol, atol=atol, equal_nan=True)
    if same.all():
        return None
    r, c = np.argwhere(~same)[0]
    return a.index[r], a.columns[c], av[r, c], bv[r, c]


def check_determinism(build: Builder, df: pd.DataFrame, rtol=1e-9, atol=1e-12) -> bool:
    a, b = _as_frame(build(df.copy())), _as_frame(build(df.copy()))
    return _first_diff(a, b, rtol, atol) is None


def two_run_oracle(build: Builder, df: pd.DataFrame, n_epochs: int = 30, epochs: list | None = None,
                   kinds: Iterable[str] = DEFAULT_KINDS, min_frac: float = 0.1, seed: int = 0,
                   rtol: float = 1e-9, atol: float = 1e-12, stop_at_first: bool = False) -> dict:
    """Perturb every row with index > t; outputs at index <= t must be unchanged.

    `df` must be indexed by availability time (sorted). Returns
    {"ok": bool, "epochs": n, "kinds": [...], "leaks": [witness dicts], "errors": [...], "note": str}.
    """
    rng = np.random.default_rng(seed)
    df = df.sort_index()
    if not check_determinism(build, df, rtol, atol):
        return dict(ok=False, epochs=0, kinds=list(kinds), leaks=[], errors=[],
                    note="build() is not deterministic on identical input: fix seeds/state first.")
    base = _as_frame(build(df.copy()))
    ep = epochs if epochs is not None else _pick_epochs(df.index, n_epochs, min_frac, rng)
    leaks, errors = [], []
    for t in ep:
        after = np.asarray(df.index > t)
        for kind in kinds:
            try:
                pert = _as_frame(build(_perturb(df, after, kind, rng)))
            except Exception as exc:          # a crash on weird input is not a leak
                errors.append(dict(epoch=t, kind=kind, error=f"{type(exc).__name__}: {exc}"))
                continue
            d = _first_diff(base.loc[:t], pert.loc[:t], rtol, atol)
            if d is not None:
                row, col, v0, v1 = d
                leaks.append(dict(epoch=t, kind=kind, row=row, column=col, clean=v0, perturbed=v1))
                if stop_at_first:
                    return dict(ok=False, epochs=len(ep), kinds=list(kinds), leaks=leaks, errors=errors,
                                note="stopped at first witness")
    note = ("no witness found; this does NOT prove freedom (unsampled epochs, threshold effects, "
            "mis-stamped inputs and LLM/retrieval leakage are invisible to this test)")
    return dict(ok=not leaks, epochs=len(ep), kinds=list(kinds), leaks=leaks, errors=errors,
                note=note if not leaks else "leak witnessed: outputs at or before t depend on data after t")


def truncation_test(build: Builder, df: pd.DataFrame, n_epochs: int = 30, epochs: list | None = None,
                    min_frac: float = 0.1, seed: int = 0, rtol: float = 1e-9, atol: float = 1e-12) -> dict:
    """build(df[:t]) must reproduce build(df) on all rows <= t (catches full-sample fits/normalisers)."""
    rng = np.random.default_rng(seed)
    df = df.sort_index()
    base = _as_frame(build(df.copy()))
    ep = epochs if epochs is not None else _pick_epochs(df.index, n_epochs, min_frac, rng)
    leaks, errors = [], []
    for t in ep:
        try:
            part = _as_frame(build(df.loc[:t].copy()))
        except Exception as exc:
            errors.append(dict(epoch=t, error=f"{type(exc).__name__}: {exc}"))
            continue
        d = _first_diff(base.loc[:t], part.loc[:t], rtol, atol)
        if d is not None:
            row, col, v0, v1 = d
            leaks.append(dict(epoch=t, row=row, column=col, full=v0, truncated=v1))
    return dict(ok=not leaks, epochs=len(ep), leaks=leaks, errors=errors)


# ---------------------------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------------------------
def _load_func(module_path: str, func: str) -> Builder:
    path = Path(module_path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem + "_leakcheck", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func)


def _load_data(path: str, date_col: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    col = date_col or df.columns[0]
    df[col] = pd.to_datetime(df[col])
    df = df.set_index(col).sort_index()
    return df[~df.index.duplicated(keep="last")]


def _fmt_t(t) -> str:
    return str(t.date()) if hasattr(t, "date") else str(t)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_l = sub.add_parser("lint", help="static scan of python files")
    p_l.add_argument("files", nargs="+")
    p_o = sub.add_parser("oracle", help="two-run perturbation test (+ optional truncation test)")
    p_o.add_argument("--module", required=True, help="python file defining the build function")
    p_o.add_argument("--func", default="signal", help="function name: f(df) -> DataFrame/Series indexed by time")
    p_o.add_argument("--data", required=True, help="CSV; first column (or --date-col) is the time index")
    p_o.add_argument("--date-col", default=None)
    p_o.add_argument("--epochs", type=int, default=30)
    p_o.add_argument("--kinds", default=",".join(DEFAULT_KINDS))
    p_o.add_argument("--seed", type=int, default=0)
    p_o.add_argument("--trunc", action="store_true", help="also run the truncation test")
    args = ap.parse_args(argv)

    if args.cmd == "lint":
        hits = lint_files(args.files)
        for h in hits:
            print(f"{h['file']}:{h['line']}: [{h['severity']}] {h['rule']}: {h['code']}\n    -> {h['why']}")
        n_high = sum(h["severity"] == "high" for h in hits)
        print(f"\n{len(hits)} hit(s), {n_high} high. Review each; 'review' hits are fine only if restricted to data <= t.")
        return 1 if n_high else 0

    build = _load_func(args.module, args.func)
    df = _load_data(args.data, args.date_col)
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    rep = two_run_oracle(build, df, n_epochs=args.epochs, kinds=kinds, seed=args.seed)
    print(f"two-run oracle: {'PASS' if rep['ok'] else 'FAIL'} over {rep['epochs']} epochs x {len(kinds)} perturbations")
    for w in rep["leaks"][:10]:
        print(f"  LEAK: output at {_fmt_t(w['row'])} [{w['column']}] changed ({w['clean']!r} -> {w['perturbed']!r}) "
              f"when data after {_fmt_t(w['epoch'])} was perturbed ({w['kind']})")
    if len(rep["leaks"]) > 10:
        print(f"  ... {len(rep['leaks']) - 10} more")
    for e in rep["errors"][:3]:
        print(f"  (build failed under '{e['kind']}' at {_fmt_t(e['epoch'])}: {e['error']}; not counted as a leak)")
    print(f"  note: {rep['note']}")
    ok = rep["ok"]
    if args.trunc:
        tr = truncation_test(build, df, n_epochs=args.epochs, seed=args.seed)
        print(f"truncation test: {'PASS' if tr['ok'] else 'FAIL'} over {tr['epochs']} epochs")
        for w in tr["leaks"][:10]:
            print(f"  MISMATCH at {_fmt_t(w['row'])} [{w['column']}]: full={w['full']!r} truncated@{_fmt_t(w['epoch'])}={w['truncated']!r}")
        ok = ok and tr["ok"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
