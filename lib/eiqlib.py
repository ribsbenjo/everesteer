"""Fast offline scoring + data helpers matching everestapi.scoring exactly."""
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import rankdata

B = 0.6366197723675814

def load():
    tr = pd.read_parquet('/home/climber/work/futures_train.parquet')
    bm = pd.read_parquet('/home/climber/work/futures_train_benchmark_models.parquet')
    tr['v1_sherpa'] = bm.loc[tr.index, 'v1_sherpa'].values
    tr = tr.sort_values('exped', kind='stable')
    tr['exped_i'] = tr.exped.str.slice(6).astype(int)
    return tr

def segments(exped_i):
    """exped_i must be sorted. returns list of (start, end) slices."""
    e = np.asarray(exped_i)
    idx = np.flatnonzero(np.diff(e)) + 1
    starts = np.r_[0, idx]; ends = np.r_[idx, len(e)]
    return list(zip(starts, ends))

def gauss_rank(x):
    r = (rankdata(x, method='average') - 0.5) / len(x)
    return stats.norm.ppf(np.clip(r, 0.001, 0.999))

def core_avg(F):
    """F: 2D array of core features with -1 missing. Row mean skipping missing (NaN if all missing)."""
    F = np.asarray(F, dtype=float)
    F = np.where(F == -1, np.nan, F)
    return np.nanmean(F, axis=1)

def score_per_exped(pred, target, bench, core, segs, zero_rule=True):
    """Return DataFrame with per-exped fit, unq, inov. pred/target/bench/core aligned 1D arrays.
    bench or core may be None -> NaN columns."""
    pred = np.asarray(pred, float); target = np.asarray(target, float)
    out = np.full((len(segs), 3), np.nan)
    for k, (s, e) in enumerate(segs):
        p = pred[s:e]; t = target[s:e]
        ok = ~np.isnan(p) & ~np.isnan(t)
        if bench is not None: ok &= ~np.isnan(bench[s:e])
        if core is not None: ok &= ~np.isnan(core[s:e])
        if ok.sum() < 5: continue
        p = p[ok]; t = t[ok]
        if p.std() == 0: out[k] = (0, 0, 0); continue
        z = gauss_rank(p); tc = t - t.mean(); n = len(t)
        zc = z - z.mean()
        out[k, 0] = zc @ tc / n
        for j, ref in ((1, bench), (2, core)):
            if ref is None: continue
            m = gauss_rank(ref[s:e][ok])
            mc = m - m.mean()
            if zero_rule and (mc @ tc) / n < 0: out[k, j] = 0.0; continue
            mdm = m @ m
            if mdm == 0: out[k, j] = 0.0; continue
            ortho = z - m * (m @ z) / mdm
            out[k, j] = tc @ ortho / n
    return pd.DataFrame(out, columns=['fit', 'unq', 'inov'])

def round_score(S, b=B):
    return b * np.arctan(3 * S / b)

def summarize(df, bench_df=None):
    """df: per-exped fit/unq/inov of candidate; bench_df: per-exped fit/inov of the benchmark (same expeds).
    S = fit + 3 unq + inov - (bfit + binov)."""
    m = df.mean()
    S = m.fit + 3 * m.unq + m.inov
    if bench_df is not None:
        bm = bench_df.mean(); S -= (bm.fit + bm.inov)
    return dict(fit=m.fit, unq=m.unq, inov=m.inov, S=S, score=round_score(S))
