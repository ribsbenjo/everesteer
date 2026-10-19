"""Download the open round's live split and write predictions for the given models.
usage: round_predict.py ROUND_TAG model1 model6 ...   (does NOT submit)"""
import sys, os, subprocess, cloudpickle, pandas as pd, numpy as np
tag = sys.argv[1]; names = sys.argv[2:]
out_dir = f'/home/climber/work/rounds/{tag}'; os.makedirs(out_dir, exist_ok=True)
live_path = f'{out_dir}/live.parquet'
if not os.path.exists(live_path):
    from everestapi import EverestAPI
    api = EverestAPI(); print(api.download_dataset(universe='futures', split='live', output_path=live_path))
live = pd.read_parquet(live_path); feats = [c for c in live.columns if c.startswith('feature_')]
print('live rows', len(live), 'expeds', live.exped.nunique(), 'index unique', live.index.is_unique, 'cols', len(live.columns))
preds = {}
for n in names:
    p = cloudpickle.load(open(f'/home/climber/work/models/{n}.pkl', 'rb'))
    out = p(live[['exped'] + feats])
    assert out.index.equals(live.index) and out.prediction.between(0, 1).all() and not out.isna().any().any()
    out.to_parquet(f'{out_dir}/{n}_preds.parquet'); preds[n] = out.prediction.values
    print(n, 'written', out.shape)
if len(preds) > 1:
    print('pairwise corr:'); print(pd.DataFrame(preds).corr().round(3))
