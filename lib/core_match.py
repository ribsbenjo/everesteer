import sys, json; sys.path.insert(0,'/home/climber/work/lib')
from eiqlib import *
import numpy as np, pandas as pd, time
t0=time.time()
tr = load(); feats=[c for c in tr.columns if c.startswith('feature_')]
ei = tr.exped_i.values; segs = segments(ei); y = tr.target_everest.values.astype(float)
F = tr[feats].values.astype(np.float32)
res = json.load(open('lib/core_probe_results.json'))
done = [f for f in feats if f in res and res[f]['status']=='completed' and res[f].get('fit') and res[f].get('inov')]
print(len(done), 'features with platform numbers')
pf = pd.DataFrame({f: dict(fit=res[f]['fit']['mean'], inov=res[f]['inov']['mean'], tfit=res[f]['train_fit']['mean']) for f in done}).T
pf['deficit'] = pf.fit - pf.inov; pf['ratio'] = pf.inov / pf.fit
# subsample expeds for speed
sub = segs[::8]
# precompute per-exped gauss ranks for every feature, with two missing conventions
def gr_feature(x):  # x 1D with -1 missing -> rank with -1 as lowest (fillna-like) 
    return gauss_rank(x)
G = np.empty((len(sub), ), dtype=object); T = []; 
GF = []  # list of (n, 178) gauss-ranked feature matrices per sub exped (missing=-1 kept as lowest bin, i.e. fillna(0) would differ slightly)
for s, e in sub:
    X = F[s:e].astype(float); n = e - s
    g = np.column_stack([gauss_rank(X[:, j]) for j in range(X.shape[1])])
    GF.append(g); T.append(y[s:e] - y[s:e].mean())
# per-feature sign = sign of overall corr (platform ridge on one feature has coefficient sign = train corr sign)
sign = np.sign(np.array([np.nanmean([gf[:, j] @ t / len(t) for gf, t in zip(GF, T)]) for j in range(len(feats))]))
# our FIT_f estimate
fit_est = np.array([np.mean([sign[j] * (gf[:, j] - gf[:, j].mean()) @ t / len(t) for gf, t in zip(GF, T)]) for j in range(len(feats))])
cmp = pd.DataFrame(dict(platform_fit=pf.fit, est_fit=pd.Series(fit_est, index=feats).reindex(pf.index)))
print('replication check: corr(platform FIT, offline FIT) =', cmp.corr().iloc[0,1].round(3), ' mean abs diff', (cmp.platform_fit-cmp.est_fit).abs().mean().round(4))
def inov_vec(S):
    """INOV for every feature given core set S (indices), with zero rule, on sub expeds."""
    acc = np.zeros(len(feats)); cnt = 0
    for (s, e), gf, t in zip(sub, GF, T):
        X = F[s:e][:, S].astype(float); X[X == -1] = np.nan
        with np.errstate(invalid='ignore'): ca = np.nanmean(X, 1)
        ok = ~np.isnan(ca)
        if ok.sum() < 5: continue
        m = gauss_rank(ca[ok]); mc = m - m.mean(); tt = t[ok]; g = gf[ok] * sign[None, :]
        n = ok.sum()
        if (mc @ tt) / n < 0: cnt += 1; continue  # inov 0 for all features this exped
        proj = m[:, None] * (m @ g)[None, :] / (m @ m)
        acc += (tt @ (g - proj)) / n; cnt += 1
    return acc / cnt
target = pf.deficit.reindex(feats).values  # platform fit - inov
fit_sub = fit_est
def loss(S):
    iv = inov_vec(S); d_est = fit_sub - iv
    ok = ~np.isnan(target)
    return np.nanmean((d_est[ok] - target[ok])**2), iv
# candidate pool: top 40 by platform deficit
pool = list(pf.deficit.sort_values(ascending=False).index[:40]); pool_idx = [feats.index(f) for f in pool]
print('top deficit features:', [(f, round(pf.deficit[f],4), round(pf.ratio[f],2)) for f in pool[:15]])
S = []
for step in range(10):
    best = None
    for j in pool_idx:
        if j in S: continue
        l, _ = loss(S + [j])
        if best is None or l < best[0]: best = (l, j)
    S.append(best[1]); print(f'step {step}: add {feats[best[1]]} loss {best[0]:.3e}  t={time.time()-t0:.0f}s', flush=True)
# swap refinement
improved = True
while improved:
    improved = False; base, _ = loss(S)
    for i in range(len(S)):
        for j in pool_idx:
            if j in S: continue
            S2 = S[:i] + [j] + S[i+1:]; l, _ = loss(S2)
            if l < base * 0.98: S = S2; base = l; improved = True; print('swap ->', feats[j], 'loss', f'{l:.3e}', flush=True); break
        if improved: break
l, iv = loss(S)
print('FINAL core guess:', [feats[j] for j in S], 'loss', f'{l:.3e}')
print('fit of deficit: corr(est, platform) =', np.corrcoef((fit_sub - iv)[~np.isnan(target)], target[~np.isnan(target)])[0,1].round(3))
json.dump([feats[j] for j in S], open('lib/core_features_guess.json', 'w'))
