"""Programmatic grader: does a features_fixed.py stay unchanged on rows <= t when data after t changes?

Perturbs prices rows with date > t (noise, extreme, nan, shuffle) and fundamentals rows with
filing_date > t (eps -> random), then compares build_features output rows with date <= t.
Also a truncation test: prices[:t] and fundamentals filed <= t must reproduce rows <= t.
"""
import importlib.util, sys
import numpy as np, pandas as pd

def load(path):
    spec = importlib.util.spec_from_file_location("fixed_mod", path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m

def norm(out, t):
    out = out.copy(); out["date"] = pd.to_datetime(out["date"])
    out = out[out["date"] <= t].set_index(["date", "ticker"]).sort_index()
    return out.select_dtypes("number")

def same(a, b):
    if not a.index.equals(b.index): return False, "row set differs"
    a, b = a.align(b, join="outer", axis=1)
    ok = np.isclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-9, atol=1e-12, equal_nan=True)
    return bool(ok.all()), ("" if ok.all() else f"{(~ok).sum()} values differ, first col {a.columns[np.argwhere(~ok)[0][1]]}")

def main(fixed, inputs):
    m = load(fixed)
    P = pd.read_csv(f"{inputs}/prices.csv", index_col=0, parse_dates=True)
    F = pd.read_csv(f"{inputs}/fundamentals.csv", parse_dates=["period_end", "filing_date"])
    base = m.build_features(P.copy(), F.copy())
    rng = np.random.default_rng(0)
    idx = P.index; epochs = [idx[i] for i in np.linspace(120, 560, 15).astype(int)]
    fails = []
    for t in epochs:
        b = norm(base, t)
        for kind in ["noise", "extreme", "nan", "shuffle"]:
            p = P.copy(); after = p.index > t; blk = p.loc[after].to_numpy(float)
            if kind == "noise": blk = blk * np.exp(rng.normal(0, 0.2, blk.shape))
            elif kind == "extreme": blk = np.where(rng.random(blk.shape) < .5, 1e6, 1e-3)
            elif kind == "nan": blk[:] = np.nan
            else: blk = blk[rng.permutation(len(blk))]
            p.loc[after] = blk
            f = F.copy(); fa = f["filing_date"] > t; f.loc[fa, "eps"] = rng.normal(0, 50, fa.sum())
            try: o = norm(m.build_features(p, f), t)
            except Exception as e: fails.append(f"{t.date()} {kind}: error {e}"); continue
            ok, why = same(b, o)
            if not ok: fails.append(f"{t.date()} {kind}: {why}")
        try:
            o = norm(m.build_features(P.loc[:t].copy(), F[F["filing_date"] <= t].copy()), t)
            ok, why = same(b, o)
            if not ok: fails.append(f"{t.date()} truncation: {why}")
        except Exception as e: fails.append(f"{t.date()} truncation: error {e}")
    print("PASS" if not fails else f"FAIL ({len(fails)} witnesses)")
    for x in fails[:8]: print("  ", x)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
