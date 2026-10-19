import sys; sys.path.insert(0,'/home/climber/work/lib')
import numpy as np, pandas as pd, cloudpickle, lightgbm as lgb, json
from eiqlib import *
tr = load(); feats=[c for c in tr.columns if c.startswith('feature_')]
ei = tr.exped_i.values; u = np.unique(ei)
ex_corr = np.load('lib/ex_corr.npy')
W = int(sys.argv[1]) if len(sys.argv)>1 else 100
ALPHA = float(sys.argv[2]) if len(sys.argv)>2 else 0.5
NAME = sys.argv[3] if len(sys.argv)>3 else 'model1'
m = np.nanmean(ex_corr[-W:], 0); signs = np.where(np.isnan(m), 0, np.sign(m)).astype(float)
print('signs: +', (signs>0).sum(), '-', (signs<0).sum(), '0', (signs==0).sum())
mimic_str = open('lib/sherpa_mimic_full.txt').read()
sign_map = dict(zip(feats, signs.tolist()))

def make_predict(sign_map, mimic_str, alpha):
    import numpy as np, pandas as pd
    from scipy.stats import rankdata, norm
    feats = list(sign_map.keys()); w = np.array([sign_map[f] for f in feats], dtype=float)
    _booster = {}
    def gauss(x):
        r = (rankdata(x, method='average') - 0.5) / len(x); return norm.ppf(np.clip(r, 0.001, 0.999))
    def one_section(X):
        """X: 2D float array (n, nfeat) with -1/NaN missing. returns prediction in [0,1]."""
        X = X.astype(float); X[X == -1] = np.nan
        with np.errstate(invalid='ignore'):
            mu = np.nanmean(X, 0); sd = np.nanstd(X, 0)
        Zs = (X - mu) / np.where(sd > 0, sd, np.nan); Zs = np.where(np.isnan(Zs), 0.0, Zs)
        raw = Zs @ w
        if alpha > 0 and len(raw) >= 5:
            import lightgbm as lgb
            if 'b' not in _booster: _booster['b'] = lgb.Booster(model_str=mimic_str)
            Xm = X.astype(np.float32)  # NaN for missing, as in training
            bench = _booster['b'].predict(Xm)
            p = gauss(raw); m_ = gauss(bench); mc = m_ - m_.mean(); pc = p - p.mean()
            beta = (mc @ pc) / (mc @ mc) if (mc @ mc) > 0 else 0.0
            raw = pc - alpha * beta * mc
        r = rankdata(raw, method='average'); return (r - 0.5) / len(r)
    def predict(live_features, live_benchmark_models=None):
        df = live_features
        X = df.reindex(columns=feats).to_numpy(dtype=float)
        out = np.empty(len(df))
        if 'exped' in df.columns:
            codes, _ = pd.factorize(df['exped'].astype(str))
            for c in np.unique(codes):
                idx = np.flatnonzero(codes == c); out[idx] = one_section(X[idx])
        else:
            out[:] = one_section(X)
        out = np.clip(out, 0.0, 1.0)
        return pd.DataFrame({'prediction': out}, index=df.index)
    return predict

predict = make_predict(sign_map, mimic_str, ALPHA)
with open(f'models/{NAME}.pkl', 'wb') as f: cloudpickle.dump(predict, f)
json.dump(dict(W=W, alpha=ALPHA, signs=sign_map), open(f'models/{NAME}.json','w'))
print('saved', NAME)
