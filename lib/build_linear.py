"""Build a linear predict callable from a weight vector over per-exped z-scored features.
usage: build_linear.py NAME ALPHA WEIGHTS_NPY [EXTRA_NPY EXTRA_SCALE]  (weights over the 178 features in column order)"""
import sys; sys.path.insert(0,'/home/climber/work/lib')
import numpy as np, pandas as pd, cloudpickle, json
NAME, ALPHA, WPATH = sys.argv[1], float(sys.argv[2]), sys.argv[3]
feats = [c for c in pd.read_parquet('/home/climber/work/futures_validation.parquet').columns if c.startswith('feature_')]
w = np.load(WPATH).astype(float)
mimic_str = open('/home/climber/work/lib/sherpa_mimic_full.txt').read()
weight_map = dict(zip(feats, w.tolist()))
def make_predict(weight_map, mimic_str, alpha):
    import numpy as np, pandas as pd
    from scipy.stats import rankdata, norm
    feats = list(weight_map.keys()); w = np.array([weight_map[f] for f in feats], dtype=float); _b = {}
    def gauss(x):
        r = (rankdata(x, method='average') - 0.5) / len(x); return norm.ppf(np.clip(r, 0.001, 0.999))
    def one_section(X):
        X = X.astype(float); X[X == -1] = np.nan
        with np.errstate(invalid='ignore'):
            mu = np.nanmean(X, 0); sd = np.nanstd(X, 0)
        Zs = (X - mu) / np.where(sd > 0, sd, np.nan); Zs = np.where(np.isnan(Zs), 0.0, Zs)
        raw = Zs @ w
        if alpha > 0 and len(raw) >= 5:
            import lightgbm as lgb
            if 'b' not in _b: _b['b'] = lgb.Booster(model_str=mimic_str)
            bench = _b['b'].predict(X.astype(np.float32))
            p = gauss(raw); m_ = gauss(bench); mc = m_ - m_.mean(); pc = p - p.mean()
            beta = (mc @ pc) / (mc @ mc) if (mc @ mc) > 0 else 0.0
            raw = pc - alpha * beta * mc
        r = rankdata(raw, method='average'); return (r - 0.5) / len(r)
    def predict(live_features, live_benchmark_models=None):
        df = live_features; X = df.reindex(columns=feats).to_numpy(dtype=float); out = np.empty(len(df))
        if 'exped' in df.columns:
            codes, _ = pd.factorize(df['exped'].astype(str))
            for c in np.unique(codes):
                idx = np.flatnonzero(codes == c); out[idx] = one_section(X[idx])
        else: out[:] = one_section(X)
        return pd.DataFrame({'prediction': np.clip(out, 0.0, 1.0)}, index=df.index)
    return predict
predict = make_predict(weight_map, mimic_str, ALPHA)
cloudpickle.dump(predict, open(f'/home/climber/work/models/{NAME}.pkl', 'wb'))
json.dump(dict(alpha=ALPHA, weights=weight_map), open(f'/home/climber/work/models/{NAME}.json', 'w'))
# fresh-load test + predictions
import subprocess
code = f"""
import cloudpickle, pandas as pd, numpy as np
p = cloudpickle.load(open('/home/climber/work/models/{NAME}.pkl','rb'))
va = pd.read_parquet('/home/climber/work/futures_validation.parquet'); feats=[c for c in va.columns if c.startswith('feature_')]
out = p(va[['exped']+feats]); assert out.index.equals(va.index) and out.prediction.between(0,1).all() and not out.isna().any().any()
out.to_parquet('/home/climber/work/models/{NAME}_preds.parquet')
m1 = pd.read_parquet('/home/climber/work/models/model1_preds.parquet')
print('{NAME} ok; corr with model1 =', np.corrcoef(m1.prediction, out.prediction)[0,1].round(3))
"""
print(subprocess.run([sys.executable, '-c', code], capture_output=True, text=True).stdout)
