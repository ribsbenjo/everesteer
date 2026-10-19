import sys; sys.path.insert(0,'/home/climber/work/lib')
from eiqlib import *; from folds import *
import time, lightgbm as lgb, pickle
t0=time.time()
tr = load(); feats=[c for c in tr.columns if c.startswith('feature_')]
ei = tr.exped_i.values; u = np.unique(ei); segs = segments(ei)
folds = make_folds(u)
F = tr[feats].values.astype(np.float32); F[F==-1]=np.nan
Z = zscore_per_exped(F, segs); np.save('lib/Z.npy', Z)
y = tr.target_everest.values.astype(float)
sh = tr.v1_sherpa.values.astype(float)
mimic = np.load('lib/mimic_folds.npy')
ex_corr = np.load('lib/ex_corr.npy')  # (n_exped, n_feat), NaN where undefined
exp_of_row = np.searchsorted(u, ei)
# provisional core guess: top-10 by mean corr over full train (positive orientation)
mc = np.nanmean(ex_corr, 0); core_guess = np.argsort(-mc)[:10]
print('provisional core:', [feats[i] for i in core_guess])
core = core_avg(F[:, core_guess])
print('prep', time.time()-t0)

def signed_pred(fold, W, decay=None, mode='sign'):
    hi = np.searchsorted(u, fold['train_hi'])  # exped index exclusive
    lo = max(0, hi - W) if W else 0
    C = ex_corr[lo:hi]
    if decay:
        w = 0.5 ** ((hi - 1 - np.arange(lo, hi)) / decay); w = w[:, None]
        m = np.nansum(C * w, 0) / np.nansum(~np.isnan(C) * w, 0)
    else:
        m = np.nanmean(C, 0)
    sd = np.nanstd(C, 0); n = (~np.isnan(C)).sum(0)
    m = np.where(np.isnan(m), 0, m)
    if mode == 'sign': wts = np.sign(m)
    elif mode == 'corr': wts = m
    elif mode == 'tstat': wts = np.where(sd > 0, m / (sd / np.sqrt(np.maximum(n, 1))), 0)
    elif mode == 'sign_stable':  # sign weighted by fraction of expeds agreeing with the sign
        agree = np.nanmean(np.sign(C) == np.sign(m)[None, :], 0); wts = np.sign(m) * np.where(agree > 0.5, agree - 0.5, 0)
    return Z @ wts.astype(np.float32)

def lgb_pred(fold, W, target=y, rounds=150, leaves=15, lr=0.03, extra=None):
    hi = fold['train_hi']; lo = hi - W if W else 0
    trn = (ei >= lo) & (ei < hi) & ~np.isnan(target)
    p = dict(objective='regression', learning_rate=lr, num_leaves=leaves, min_data_in_leaf=500, feature_fraction=0.5,
             bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=2, seed=1)
    if extra: p.update(extra)
    m = lgb.train(p, lgb.Dataset(F[trn], target[trn]), num_boost_round=rounds)
    tst = (ei >= fold['test_lo']) & (ei <= fold['test_hi'])
    out = np.zeros(len(tr)); out[tst] = m.predict(F[tst]); return out

def ridge_pred(fold, W, lam=1e4):
    hi = fold['train_hi']; lo = hi - W if W else 0
    trn = (ei >= lo) & (ei < hi)
    X = Z[trn]; t = y[trn] - y[trn].mean()
    w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ t)
    return Z @ w

def post(pred, alpha_b=1.0, alpha_c=0.0):
    """orthogonalise against mimic (alpha_b) then core guess avg (alpha_c); return gauss residual."""
    p = pred.astype(float)
    if alpha_b: p = orthogonalize(p, mimic_or_full, segs, alpha_b)
    if alpha_c: p = orthogonalize(p, np.where(np.isnan(core), 0, core), segs, alpha_c)
    return p

results = {}
def run(name, fn, alphas=((1.0, 0.0),)):
    for ab, ac in alphas:
        key = f'{name}|ob{ab}|oc{ac}'
        results[key] = {}
        for fo in folds:
            global mimic_or_full; mimic_or_full = mimic
            pred = fn(fo)
            pp = post(pred, ab, ac)
            results[key][fo['name']] = eval_fold(pp, tr, fo, sh, core, segs, ei)
        r = results[key]; print(f'{key:40s}', ' '.join(f"{f}: S={r[f]['S_raw']:+.4f} fit={r[f]['fit']:+.4f} unq={r[f]['unq']:+.4f} inov={r[f]['inov']:+.4f}" for f in r), f'| t={time.time()-t0:.0f}s', flush=True)

# baseline sanity: sherpa itself, and random
run('sherpa', lambda fo: sh, alphas=((0.0, 0.0),))
# A. signed sums, window sweep, alpha sweep at W=100
run('signed_W100', lambda fo: signed_pred(fo, 100), alphas=((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 1.0)))
for W in (50, 200, 300, 500, 1000, 0):
    run(f'signed_W{W}', lambda fo, W=W: signed_pred(fo, W))
for d in (50, 100, 200):
    run(f'signed_decay{d}', lambda fo, d=d: signed_pred(fo, 1000, decay=d))
for W in (100, 300):
    run(f'corrw_W{W}', lambda fo, W=W: signed_pred(fo, W, mode='corr'))
    run(f'tstat_W{W}', lambda fo, W=W: signed_pred(fo, W, mode='tstat'))
    run(f'stable_W{W}', lambda fo, W=W: signed_pred(fo, W, mode='sign_stable'))
# B. recent lgbm / ridge
for W in (300, 600, 1200, 0):
    run(f'lgb_W{W}', lambda fo, W=W: lgb_pred(fo, W))
    run(f'ridge_W{W}', lambda fo, W=W: ridge_pred(fo, W))
pickle.dump(results, open('lib/results_exp1.pkl', 'wb'))
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(table(results).round(4))
