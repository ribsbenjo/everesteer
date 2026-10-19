import sys; sys.path.insert(0,'/home/climber/work/lib')
from eiqlib import *; from folds import *
import time, pickle
t0=time.time()
tr = load(); feats=[c for c in tr.columns if c.startswith('feature_')]; tgts=[c for c in tr.columns if c.startswith('target_')]
ei = tr.exped_i.values; u = np.unique(ei); segs = segments(ei); folds = make_folds(u)
F = tr[feats].values.astype(np.float32); F[F==-1]=np.nan
Z = np.load('lib/Z.npy'); y = tr.target_everest.values.astype(float); sh = tr.v1_sherpa.values.astype(float)
mimic = np.load('lib/mimic_folds.npy'); ex_corr = np.load('lib/ex_corr.npy'); exT = pickle.load(open('lib/excorr_targets.pkl','rb'))
mc = np.nanmean(ex_corr, 0); core_guess = np.argsort(-mc)[:10]; core = core_avg(F[:, core_guess])

def signed_w(C_all, fold, W):
    hi = np.searchsorted(u, fold['train_hi']); lo = max(0, hi-W) if W else 0
    m = np.nanmean(C_all[lo:hi], 0); return np.where(np.isnan(m), 0, np.sign(m)).astype(np.float32)
# precompute Gram per fold for ridge-all
gram = {}
for fo in folds:
    trn = ei < fo['train_hi']; X = Z[trn]; gram[fo['name']] = (X.T @ X, trn)
def ridge_all(fold, t, lam=1e4):
    G, trn = gram[fold['name']]; tt = t[trn]; ok = ~np.isnan(tt)
    X = Z[trn][ok]; w = np.linalg.solve(G*(ok.mean()) + lam*np.eye(G.shape[0]), X.T @ (tt[ok]-tt[ok].mean())); return Z @ w
def evalp(pred, fo, alpha=1.0):
    pp = orthogonalize(pred, mimic, segs, alpha) if alpha else pred
    return eval_fold(pp, tr, fo, sh, core, segs, ei)
results = {}
def run(name, fn):
    results[name] = {fo['name']: evalp(fn(fo), fo) for fo in folds}
    r = results[name]; S = [r[f]['S_raw'] for f in r]
    print(f'{name:34s} S: ' + ' '.join(f'{s:+.4f}' for s in S) + f' | mean {np.mean(S):+.4f} min {np.min(S):+.4f} | fit ' + ' '.join(f"{r[f]['fit']:+.4f}" for f in r) + ' | unq ' + ' '.join(f"{r[f]['unq']:+.4f}" for f in r), flush=True)
for tname in tgts:
    C = exT[tname]; t = tr[tname].values.astype(float)
    for W in (50, 100, 200):
        run(f'{tname[7:]}|signed_W{W}', lambda fo, C=C, W=W: Z @ signed_w(C, fo, W))
    run(f'{tname[7:]}|ridge_all', lambda fo, t=t: ridge_all(fo, t))
pickle.dump(results, open('lib/results_exp3.pkl','wb'))
print('done', time.time()-t0)
