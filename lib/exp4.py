import sys; sys.path.insert(0,'/home/climber/work/lib')
from eiqlib import *; from folds import *
import time, lightgbm as lgb
t0=time.time()
tr = load(); feats=[c for c in tr.columns if c.startswith('feature_')]
ei = tr.exped_i.values; u = np.unique(ei); segs = segments(ei); folds = make_folds(u)
F = tr[feats].values.astype(np.float32); F[F==-1]=np.nan
Z = np.load('lib/Z.npy'); y = tr.target_everest.values.astype(float); sh = tr.v1_sherpa.values.astype(float)
mimic = np.load('lib/mimic_folds.npy'); ex_corr = np.load('lib/ex_corr.npy')
mc = np.nanmean(ex_corr, 0); core = core_avg(F[:, np.argsort(-mc)[:10]])
def ridge(fo, W, lam):
    trn = (ei >= fo['train_hi']-W) & (ei < fo['train_hi']); X = Z[trn]; t = y[trn]-y[trn].mean()
    return Z @ np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@t)
def lgbm(fo, W, rounds, leaves):
    trn = (ei >= fo['train_hi']-W) & (ei < fo['train_hi'])
    p = dict(objective='regression', learning_rate=0.05, num_leaves=leaves, min_data_in_leaf=300, feature_fraction=0.3, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=2, seed=1)
    m = lgb.train(p, lgb.Dataset(F[trn], y[trn]), rounds); return m.predict(F)
def run(name, fn):
    r = {fo['name']: eval_fold(orthogonalize(fn(fo), mimic, segs, 0.5), tr, fo, sh, core, segs, ei) for fo in folds}
    S=[r[f]['S_raw'] for f in r]; print(f'{name:22s} S ' + ' '.join(f'{s:+.4f}' for s in S) + f' | mean {np.mean(S):+.4f} min {np.min(S):+.4f} | fit ' + ' '.join(f"{r[f]['fit']:+.4f}" for f in r) + f' | {time.time()-t0:.0f}s', flush=True)
for W in (50, 100, 150):
    for lam in (3e3, 3e4, 3e5): run(f'ridge_W{W}_lam{lam:.0e}', lambda fo, W=W, lam=lam: ridge(fo, W, lam))
for W in (50, 100, 200):
    run(f'lgbm_W{W}_r100_l7', lambda fo, W=W: lgbm(fo, W, 100, 7))
run('lgbm_W100_r300_l15', lambda fo: lgbm(fo, 100, 300, 15))
