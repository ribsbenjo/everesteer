"""Linear predict callable with benchmark-mimic neutralisation (alpha_b) and core-average neutralisation (alpha_c).
usage: build_linear2.py NAME ALPHA_B WEIGHTS_NPY CORE_JSON ALPHA_C"""
import sys, json, subprocess; sys.path.insert(0,'/home/climber/work/lib')
import numpy as np, pandas as pd, cloudpickle
NAME, AB, WPATH, CJ, AC = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], float(sys.argv[5])
feats = [c for c in pd.read_parquet('/home/climber/work/futures_validation.parquet').columns if c.startswith('feature_')]
w = np.load(WPATH).astype(float); core_list = json.load(open(CJ))
mimic_str = open('/home/climber/work/lib/sherpa_mimic_full.txt').read()
weight_map = dict(zip(feats, w.tolist()))
def make_predict(weight_map, mimic_str, alpha_b, core_list, alpha_c):
    import numpy as np, pandas as pd
    from scipy.stats import rankdata, norm
    feats = list(weight_map.keys()); w = np.array([weight_map[f] for f in feats], dtype=float); _b = {}
    core_idx = [feats.index(c) for c in core_list]
    def gauss(x):
        r = (rankdata(x, method='average') - 0.5) / len(x); return norm.ppf(np.clip(r, 0.001, 0.999))
    def neut(raw, ref, alpha):
        ok = ~np.isnan(ref)
        if ok.sum() < 5: return raw
        p = gauss(raw); pc = p - p.mean(); m_ = np.zeros_like(p); m_[ok] = gauss(ref[ok]); m_[ok] -= m_[ok].mean()
        beta = (m_ @ pc) / (m_ @ m_) if (m_ @ m_) > 0 else 0.0
        return pc - alpha * beta * m_
    def one_section(X):
        X = X.astype(float); X[X == -1] = np.nan
        with np.errstate(invalid='ignore'):
            mu = np.nanmean(X, 0); sd = np.nanstd(X, 0)
        Zs = (X - mu) / np.where(sd > 0, sd, np.nan); Zs = np.where(np.isnan(Zs), 0.0, Zs)
        raw = Zs @ w
        if len(raw) >= 5:
            if alpha_b > 0:
                import lightgbm as lgb
                if 'b' not in _b: _b['b'] = lgb.Booster(model_str=mimic_str)
                raw = neut(raw, _b['b'].predict(X.astype(np.float32)), alpha_b)
            if alpha_c > 0:
                with np.errstate(invalid='ignore'): ca = np.nanmean(X[:, core_idx], 1)
                raw = neut(raw, ca, alpha_c)
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
predict = make_predict(weight_map, mimic_str, AB, core_list, AC)
cloudpickle.dump(predict, open(f'/home/climber/work/models/{NAME}.pkl', 'wb'))
json.dump(dict(alpha_b=AB, alpha_c=AC, core=core_list, weights=weight_map), open(f'/home/climber/work/models/{NAME}.json', 'w'))
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
