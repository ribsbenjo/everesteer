"""Look-ahead checks for features.py (original) and features_fixed.py.

Run from this folder:  python3 leak_check.py [--step N]

1. Truncation test: for every cutoff date T, rebuild the features from only the data available at the
   close of T (prices up to T, filings dated strictly before T) and compare row T with the full-sample run.
   Any difference means row T used information from after the close of T.
2. Perturbation test: scramble everything after T (prices, missing prints, future EPS) and check that
   every row dated <= T is unchanged, including which rows exist.
3. Component check for prices.bfill(), which features.py hides behind dropna(axis=1).
4. Smell test: mean daily rank IC of each feature against the scored target (close t -> close t+1).
"""
import argparse
import warnings

import numpy as np
import pandas as pd

import features as original
import features_fixed as fixed

warnings.simplefilter("ignore", FutureWarning)
FEATS = ["mom_z", "rev_pct", "vol", "weekly_mom", "cs_move", "high_vol", "eps", "ey"]


def load():
    prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
    fund = pd.read_csv("fundamentals.csv", parse_dates=["period_end", "filing_date"])
    return prices, fund


def known_at(prices, fund, T):
    """The information set at the close of T. A filing dated T may land after the close, so it is excluded."""
    return prices.loc[:T], fund[fund["filing_date"] < T]


def diff_rows(a, b):
    """Compare two feature frames on (date, ticker). Return mismatch counts per feature and row-set differences."""
    m = a.merge(b, on=["date", "ticker"], how="outer", suffixes=("_a", "_b"), indicator=True)
    both = m[m["_merge"] == "both"]
    out = {"only_a": int((m["_merge"] == "left_only").sum()), "only_b": int((m["_merge"] == "right_only").sum())}
    for c in FEATS:
        x, y = both[c + "_a"].astype(float).to_numpy(), both[c + "_b"].astype(float).to_numpy()
        same = (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=1e-9, atol=1e-12)
        out[c] = int((~same).sum())
    return out


def truncation_test(build, prices, fund, step):
    full = build(prices, fund)
    totals = {k: 0 for k in ["only_a", "only_b"] + FEATS}
    dates_hit = {k: 0 for k in totals}
    for T in prices.index[1::step]:
        p_T, f_T = known_at(prices, fund, T)
        live = build(p_T, f_T)
        d = diff_rows(full[full["date"] == T], live[live["date"] == T])
        for k, v in d.items():
            totals[k] += v
            dates_hit[k] += v > 0
    n_cut = len(prices.index[1::step])
    return pd.DataFrame({"mismatched cells": totals, "cutoffs affected": dates_hit}).assign(cutoffs=n_cut)


def perturb_after(prices, fund, T, rng):
    """Replace everything not known at the close of T with noise, and knock out 5% of future prints."""
    p = prices.copy()
    fut = p.index > T
    shocks = np.exp(rng.normal(0, 0.05, size=(fut.sum(), p.shape[1])).cumsum(axis=0))
    vals = p.loc[fut].to_numpy() * shocks
    vals[rng.random(vals.shape) < 0.05] = np.nan
    p.loc[fut] = vals
    f = fund.copy()
    late = f["filing_date"] >= T
    f.loc[late, "eps"] = rng.normal(0, 5, late.sum())
    return p, f


def perturbation_test(build, prices, fund, cutoffs, seed=0):
    rng = np.random.default_rng(seed)
    base = build(prices, fund)
    rows = []
    for T in cutoffs:
        p2, f2 = perturb_after(prices, fund, T, rng)
        pert = build(p2, f2)
        d = diff_rows(base[base["date"] <= T], pert[pert["date"] <= T])
        rows.append(pd.Series(d, name=T.date()))
    return pd.DataFrame(rows)


def bfill_component_check(prices):
    """At each isolated missing print, does bfill() put tomorrow's close on today's row?"""
    filled = prices.bfill()
    gaps = prices.isna() & prices.ffill().notna() & prices.bfill().notna()
    nxt = prices.shift(-1)
    hits = [(d, t) for d, t in zip(*np.where(gaps.to_numpy()))]
    same_as_tomorrow = sum(np.isclose(filled.iat[i, j], nxt.iat[i, j]) for i, j in hits)
    return len(hits), same_as_tomorrow


def mean_ic(feat_df, target):
    """Mean daily Spearman IC of each feature against the target, with a t-stat."""
    df = feat_df.merge(target, on=["date", "ticker"], how="inner")
    df["abs_fwd"] = df["fwd_ret"].abs()
    res = {}
    for c in FEATS:
        if c == "high_vol":
            continue  # one value per date, so it has no cross-sectional IC
        for tgt in ["fwd_ret", "abs_fwd"]:
            ic = df.dropna(subset=[c, tgt]).groupby("date").apply(
                lambda g: g[c].corr(g[tgt], method="spearman") if g[c].nunique() > 1 else np.nan,
                include_groups=False).dropna()
            res[(c, tgt)] = (ic.mean(), ic.mean() / ic.std() * np.sqrt(len(ic)))
    out = pd.Series(res).apply(pd.Series)
    out.columns = ["mean_ic", "t_stat"]
    return out.unstack()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=1, help="test every Nth date as a cutoff")
    args = ap.parse_args()
    prices, fund = load()
    pd.set_option("display.width", 200)

    print("== 1. Truncation test (row T, full sample vs data known at close of T) ==")
    print("only_a = row in full-sample output but not computable live; only_b = live row missing from full output")
    for name, mod in [("features.py", original), ("features_fixed.py", fixed)]:
        print(f"\n-- {name}")
        print(truncation_test(mod.build_features, prices, fund, args.step).to_string())

    print("\n== 1b. Truncation test on the 24 tickers with a complete history ==")
    print("dropna(axis=1) is a no-op here, so the universe cannot change and each feature's own leak shows")
    complete = prices.columns[prices.notna().all()]
    p24, f24 = prices[complete], fund[fund["ticker"].isin(complete)]
    for name, mod in [("features.py", original), ("features_fixed.py", fixed)]:
        print(f"\n-- {name}")
        print(truncation_test(mod.build_features, p24, f24, args.step).to_string())

    print("\n== 2. Perturbation test (scramble after T, rows <= T must not change) ==")
    cutoffs = prices.index[[100, 250, 400, 500, 590]]
    for name, mod in [("features.py", original), ("features_fixed.py", fixed)]:
        print(f"\n-- {name}")
        print(perturbation_test(mod.build_features, prices, fund, cutoffs).to_string())

    print("\n== 3. bfill component check (masked by dropna in features.py) ==")
    n, k = bfill_component_check(prices)
    print(f"internal missing prints: {n}; filled with the NEXT day's close: {k}")
    t07_live = prices.loc[:"2023-06-30"].dropna(axis=1).columns
    t07_full = prices.dropna(axis=1).columns
    print(f"dropna(axis=1) universe: {len(t07_full)} tickers on the full sample, "
          f"{len(t07_live)} if run on data up to 2023-06-30; dropped from full sample: "
          f"{sorted(set(prices.columns) - set(t07_full))}")

    print("\n== 4. Mean daily rank IC vs scored target (close t -> close t+1) ==")
    target = fixed.forward_return(prices)
    for name, mod in [("features.py", original), ("features_fixed.py", fixed)]:
        print(f"\n-- {name}")
        print(mean_ic(mod.build_features(prices, fund), target).round(4).to_string())


if __name__ == "__main__":
    main()
