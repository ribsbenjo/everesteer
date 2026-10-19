import sys; sys.path.insert(0,'/home/climber/work/lib')
from eiqlib import *; from folds import *
import time, lightgbm as lgb, pickle
t0=time.time()
tr = load(); feats=[c for c in tr.columns if c.startswith('feature_')]; tgts=[c for c in tr.columns if c.startswith('target_')]
ei = tr.exped_i.values; u = np.unique(ei); segs = segments(ei); folds = make_folds(u)
F = tr[feats].values.astype(np.float32); F[F==-1]=np.nan
Z = np.load('lib/Z.npy'); y = tr.target_everest.values.astype(float); sh = tr.v1_sherpa.values.astype(float)
mimic = np.load('lib/mimic_folds.npy'); ex_corr = np.load('lib/ex_corr.npy')
mc = np.nanmean(ex_corr, 0); core_guess = np.argsort(-mc)[:10]; core = core_avg(F[:, core_guess])

# ---- H1: can instruments be linked across consecutive expeds via feature fingerprints?
# count exact-duplicate feature rows between exped k and k+1 (would indicate slow features / linkable ids)
dups=[]
for k in range(len(segs)-1000, len(segs)-1, 50):
    a = F[segs[k][0]:segs[k][1]]; b = F[segs[k+1][0]:segs[k+1][1]]
    an = np.nan_to_num(a, nan=-1); bn = np.nan_to_num(b, nan=-1)
    # for each row in a, min hamming distance to rows in b
    d = (an[:,None,:] != bn[None,:,:]).sum(2)
    dups.append(np.median(d.min(1)) / a.shape[1])
print('H1 median min-hamming-fraction between consecutive expeds (0 = identical rows exist):', np.round(np.mean(dups),3))

# ---- H2: leakage check. LGBM trained on even expeds, tested on odd (interleaved) vs out-of-time f100, vs in-sample
p = dict(objective='regression', learning_rate=0.03, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=2, seed=1)
even = (ei % 2 == 0); m = lgb.train(p, lgb.Dataset(F[even], y[even]), 150)
pr = m.predict(F)
d_in = score_per_exped(pr, y, None, None, [s for s,e_ in zip(segs, u) if e_%2==0] and [sg for sg,e_ in zip(segs,u) if e_%2==0])
d_odd = score_per_exped(pr, y, None, None, [sg for sg,e_ in zip(segs,u) if e_%2==1])
print(f'H2: lgbm fit in-sample(even)={d_in.fit.mean():.4f} interleaved(odd)={d_odd.fit.mean():.4f}  (out-of-time folds from exp1: -0.0049/+0.0050/-0.0385)', time.time()-t0)

# ---- Rolling FIT of a long-history ridge (fit on expeds < 50%) over the whole train period
def ridge_fit(mask, lam=1e4):
    X = Z[mask]; t = y[mask]-y[mask].mean(); return np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@t)
w50 = ridge_fit(ei < u[len(u)//2]); pr50 = Z @ w50
d50 = score_per_exped(pr50, y, None, None, segs); d50['exped']=u
roll = d50.fit.rolling(100).mean()
print('ridge(first half) FIT by 300-exped block (first 11 in-sample):'); print(d50.groupby(np.arange(len(u))//300).fit.mean().round(4).to_string())

# ---- aux target per-exped corr matrices
excorr_t = {}
for tname in tgts:
    t_all = tr[tname].values.astype(float)
    M = np.full((len(segs), len(feats)), np.nan, dtype=np.float32)
    for k,(s,e) in enumerate(segs):
        t = t_all[s:e]; ok = ~np.isnan(t)
        if ok.sum() < 10: continue
        f = F[s:e][ok]; tc = t[ok]-t[ok].mean()
        with np.errstate(invalid='ignore', divide='ignore'):
            fm = np.nanmean(f, 0); fc = f - fm; msk = ~np.isnan(fc); fc0 = np.where(msk, fc, 0)
            M[k] = (fc0*tc[:,None]).sum(0)/np.sqrt((fc0**2).sum(0)*((tc[:,None]**2)*msk).sum(0))
    excorr_t[tname] = M
pickle.dump(excorr_t, open('lib/excorr_targets.pkl','wb'))
# target correlation with everest (per exped mean)
tc_ = {}
for tname in tgts:
    t_all = tr[tname].values.astype(float); cs=[]
    for s,e in segs[-2000:]:
        a=t_all[s:e]; b=y[s:e]; ok=~np.isnan(a)
        if ok.sum()>10 and a[ok].std()>0: cs.append(np.corrcoef(a[ok],b[ok])[0,1])
    tc_[tname]=np.mean(cs)
tc_ = pd.Series(tc_).sort_values(); print('target corr with everest (last 2000 expeds):'); print(tc_.round(3).to_string())
print('aux corr done', time.time()-t0)
