"""Validate a round's live split against train and the prediction files. usage: validate_live.py TAG model6 model1"""
import sys, pandas as pd, numpy as np
tag = sys.argv[1]; names = sys.argv[2:]
live = pd.read_parquet(f'/home/climber/work/rounds/{tag}/live.parquet')
trn = pd.read_parquet('/home/climber/work/futures_train.parquet').head(5000)
tf = [c for c in trn.columns if c.startswith('feature_')]; lf = [c for c in live.columns if c.startswith('feature_')]
print('live rows', len(live), 'expeds', live.exped.nunique(), 'index name', live.index.name, 'index unique', live.index.is_unique)
print('feature names identical & same order:', tf == lf, '| n', len(tf), len(lf), '| missing in live:', sorted(set(tf)-set(lf)), '| extra in live:', sorted(set(lf)-set(tf)))
dt_mismatch = [c for c in tf if c in live.columns and str(live[c].dtype) != str(trn[c].dtype)]
print('dtype mismatches:', dt_mismatch[:10] or 'none', '| live feature dtypes:', sorted(set(str(live[c].dtype) for c in lf)))
vals = np.unique(live[lf].values); print('feature value set in live:', vals.tolist(), '| within {-1,0..9}:', set(vals.tolist()) <= set(range(-1, 10)))
print('frac -1 live:', round(float((live[lf].values == -1).mean()), 4), '(train 0.111)')
tg = [c for c in live.columns if c.startswith('target_')]; print('target cols', len(tg), 'all blank:', live[tg].isna().all().all())
print('exped labels sample:', live.exped.unique()[:3].tolist(), '| overlap with train ids:', len(set(live.index) & set(pd.read_parquet('/home/climber/work/futures_train.parquet', columns=['exped']).index)))
for n in names:
    p = pd.read_parquet(f'/home/climber/work/rounds/{tag}/{n}_preds.parquet')
    print(f'{n}: shape {p.shape} cols {list(p.columns)} min {p.iloc[:,0].min():.4f} max {p.iloc[:,0].max():.4f} nan {int(p.isna().sum().sum())} rows==live {len(p)==len(live)} index==live {p.index.equals(live.index)}')
