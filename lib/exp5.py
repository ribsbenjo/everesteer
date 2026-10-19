import sys; sys.path.insert(0,'/home/climber/work/lib')
from eiqlib import *; from folds import *
import time, pickle
tr = load(); feats=[c for c in tr.columns if c.startswith('feature_')]
ei = tr.exped_i.values; u = np.unique(ei); segs = segments(ei); folds = make_folds(u)
F = tr[feats].values.astype(np.float32); F[F==-1]=np.nan
Z = np.load('lib/Z.npy'); y = tr.target_everest.values.astype(float); sh = tr.v1_sherpa.values.astype(float)
mimic = np.load('lib/mimic_folds.npy'); ex_corr = np.load('lib/ex_corr.npy'); exT = pickle.load(open('lib/excorr_targets.pkl','rb'))
mc = np.nanmean(ex_corr, 0); core = core_avg(F[:, np.argsort(-mc)[:10]])
def signed(C, fo, W):
    hi = np.searchsorted(u, fo['train_hi']); m = np.nanmean(C[hi-W:hi], 0); return Z @ np.where(np.isnan(m), 0, np.sign(m)).astype(np.float32)
def ridge(fo, W, lam):
    trn = (ei >= fo['train_hi']-W) & (ei < fo['train_hi']); X = Z[trn]; t = y[trn]-y[trn].mean()
    return Z @ np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@t)
def zs(p):  # per-exped standardise for blending
    out = np.empty_like(p)
    for s,e in segs: q=p[s:e]; out[s:e]=(q-q.mean())/(q.std()+1e-12)
    return out
def run(name, fn, alpha=0.5):
    r = {fo['name']: eval_fold(orthogonalize(fn(fo), mimic, segs, alpha), tr, fo, sh, core, segs, ei) for fo in folds}
    S=[r[f]['S_raw'] for f in r]; print(f'{name:36s} S ' + ' '.join(f'{s:+.4f}' for s in S) + f' | mean {np.mean(S):+.4f} min {np.min(S):+.4f} | fit ' + ' '.join(f"{r[f]['fit']:+.4f}" for f in r) + ' | unq ' + ' '.join(f"{r[f]['unq']:+.4f}" for f in r), flush=True)
m3 = lambda fo: ridge(fo, 50, 3e5)
for aux in ('Tinzar', 'Tizal', 'Tasserirt', 'Yagour', 'Azgour', 'Moussa'):
    C = exT['target_'+aux]
    run(f'{aux}_W50 alone', lambda fo, C=C: signed(C, fo, 50))
    run(f'blend m3 + {aux}_W50 (1:1)', lambda fo, C=C: zs(m3(fo)) + zs(signed(C, fo, 50)))
    run(f'blend m1 + {aux}_W50 (1:1)', lambda fo, C=C: zs(signed(ex_corr, fo, 100)) + zs(signed(C, fo, 50)))
run('blend m1 + Azgour50 + Tinzar50 + Moussa50', lambda fo: zs(signed(ex_corr, fo, 100)) + zs(signed(exT['target_Azgour'], fo, 50)) + zs(signed(exT['target_Tinzar'], fo, 50)) + zs(signed(exT['target_Moussa'], fo, 50)))
