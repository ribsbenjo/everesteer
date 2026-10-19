import numpy as np, pandas as pd, sys
sys.path.insert(0, '/home/climber/work/lib')
from eiqlib import *

GAP = 40
TEST_LEN = 300

def make_folds(expeds):
    """expeds: sorted unique exped ints. 3 folds: 300-exped test windows ending at 60/80/100% of train."""
    n = len(expeds); folds = []
    for frac in (0.6, 0.8, 1.0):
        end_i = int(round(frac * n)) - 1
        test = expeds[end_i - TEST_LEN + 1: end_i + 1]
        train_end = test[0] - GAP  # exclusive upper bound on training expeds
        folds.append(dict(name=f'f{int(frac*100)}', test_lo=int(test[0]), test_hi=int(test[-1]), train_hi=int(train_end)))
    return folds

def zscore_per_exped(F, segs):
    """F float32 with NaN for missing. per-exped standardise each column, NaN->0."""
    Z = np.empty_like(F)
    for s, e in segs:
        f = F[s:e]
        with np.errstate(invalid='ignore'):
            m = np.nanmean(f, 0); sd = np.nanstd(f, 0)
        z = (f - m) / np.where(sd > 0, sd, np.nan)
        Z[s:e] = np.where(np.isnan(z), 0, z)
    return Z

def orthogonalize(pred, ref, segs, alpha=1.0):
    """Per exped: gauss-rank pred and ref, remove alpha * projection of pred on ref. Return residual."""
    out = np.empty_like(pred, dtype=float)
    for s, e in segs:
        p = gauss_rank(pred[s:e]); m = gauss_rank(ref[s:e])
        mc = m - m.mean(); pc = p - p.mean()
        beta = (mc @ pc) / (mc @ mc) if (mc @ mc) > 0 else 0.0
        out[s:e] = pc - alpha * beta * mc
    return out

def to_unit(pred, segs):
    out = np.empty_like(pred, dtype=float)
    for s, e in segs:
        r = rankdata(pred[s:e], method='average')
        out[s:e] = (r - 0.5) / (e - s)
    return out

def eval_fold(pred, tr, fold, bench, core, segs_all, ei, zero_rule=True):
    """Evaluate predictions (aligned with tr) on the fold's test expeds. Returns per-exped frame + summary."""
    msk = (ei >= fold['test_lo']) & (ei <= fold['test_hi'])
    idx = np.flatnonzero(msk); s0, e0 = idx[0], idx[-1] + 1
    segs = [(s - s0, e - s0) for s, e in segs_all if s >= s0 and e <= e0]
    y = tr.target_everest.values[s0:e0].astype(float)
    d = score_per_exped(pred[s0:e0], y, bench[s0:e0], core[s0:e0], segs, zero_rule)
    bd = score_per_exped(bench[s0:e0], y, None, core[s0:e0], segs, zero_rule)
    m = d.mean(); bm = bd.mean()
    S_raw = m.fit + 3 * m.unq + m.inov
    S_rel = S_raw - (bm.fit + bm.inov)
    return dict(fit=m.fit, unq=m.unq, inov=m.inov, S_raw=S_raw, S_rel=S_rel, bench_fit=bm.fit, bench_inov=bm.inov,
                sharpe=m.fit / d.fit.std() if d.fit.std() > 0 else np.nan)

def table(results):
    """results: {candidate: {fold: dict}} -> DataFrame"""
    rows = []
    for c, fr in results.items():
        for f, r in fr.items():
            rows.append(dict(cand=c, fold=f, **{k: r[k] for k in ('fit', 'unq', 'inov', 'S_raw', 'S_rel')}))
    df = pd.DataFrame(rows)
    piv = df.pivot(index='cand', columns='fold', values=['fit', 'unq', 'inov', 'S_raw'])
    piv[('S_raw', 'mean')] = df.groupby('cand').S_raw.mean()
    piv[('S_raw', 'min')] = df.groupby('cand').S_raw.min()
    piv[('S_rel', 'mean')] = df.groupby('cand').S_rel.mean()
    return piv.sort_values(('S_raw', 'mean'), ascending=False)
