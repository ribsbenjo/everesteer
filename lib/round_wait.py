"""Wait for a round to open, download its live split, write predictions for the named models. No submission."""
import sys, os, time, json, cloudpickle, pandas as pd, numpy as np
from everestapi import EverestAPI
tag = sys.argv[1]; names = sys.argv[2:]
api = EverestAPI()
while True:
    st = api.get_status(); cad = st.get('cadence', {})
    ph = cad.get('phase'); ow = cad.get('open_window'); lda = cad.get('live_data_available')
    print(time.strftime('%H:%M:%S'), 'phase', ph, 'open_window', ow, 'live', lda, 'next in', cad.get('seconds_until_next_phase'), flush=True)
    if ow == tag and lda: break
    time.sleep(20)
out_dir = f'/home/climber/work/rounds/{tag}'; os.makedirs(out_dir, exist_ok=True)
live_path = f'{out_dir}/live.parquet'
for attempt in range(20):
    try: print(api.download_dataset(universe='futures', split='live', output_path=live_path)); break
    except Exception as e: print('download err', repr(e)[:200], flush=True); time.sleep(15)
live = pd.read_parquet(live_path); feats = [c for c in live.columns if c.startswith('feature_')]
print('live rows', len(live), 'expeds', live.exped.nunique(), 'first expeds', live.exped.unique()[:3].tolist(), 'index unique', live.index.is_unique, 'cols', len(live.columns), flush=True)
preds = {}
for n in names:
    p = cloudpickle.load(open(f'/home/climber/work/models/{n}.pkl', 'rb'))
    out = p(live[['exped'] + feats])
    assert out.index.equals(live.index) and out.prediction.between(0, 1).all() and not out.isna().any().any()
    out.to_parquet(f'{out_dir}/{n}_preds.parquet'); preds[n] = out.prediction.values; print(n, 'written', out.shape, flush=True)
print('corr between models:', np.corrcoef(preds[names[0]], preds[names[1]])[0,1].round(3) if len(names) > 1 else '')
json.dump(dict(open_window=ow, phase_ends_at=cad.get('phase_ends_at')), open(f'{out_dir}/window.json', 'w'))
print('READY', flush=True)
