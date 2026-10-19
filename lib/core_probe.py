import json, time, os, sys
import pandas as pd
from everestapi import EverestAPI
api = EverestAPI()
feats = [c for c in pd.read_parquet('/home/climber/work/futures_validation.parquet').columns if c.startswith('feature_')]
OUT = '/home/climber/work/lib/core_probe_results.json'
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
jobs = {}  # feat -> job id
MAXC = 24
def launch(f):
    body = {"model": "ridge", "features": [f], "universe": "futures", "gpu": "CPU", "max_hours": 0.5,
            "cv": {"folds": 2}, "idempotency_key": f"core-single-{f}", "client_label": f"core-single-{f}"}
    for attempt in range(5):
        try:
            r = api._request("POST", "/api/v1/compute/train", json=body); return r["id"]
        except Exception as e:
            print('launch err', f, repr(e)[:200], flush=True); time.sleep(30)
    return None
todo = [f for f in feats if f not in res]
print('todo', len(todo), flush=True)
while todo or jobs:
    while todo and len(jobs) < MAXC:
        f = todo.pop(0); jid = launch(f)
        if jid: jobs[f] = jid; print('launched', f, jid, flush=True)
        else: todo.append(f); break
    time.sleep(20)
    for f, jid in list(jobs.items()):
        try: st = api.get_job_status(jid)
        except Exception as e: print('status err', repr(e)[:100], flush=True); continue
        s = st.get('status')
        if s in ('completed', 'failed', 'timeout', 'cancelled'):
            m = (st.get('metrics') or {}).get('cv') or {}
            res[f] = dict(status=s, cost=st.get('cost_cents'), fit=m.get('fit'), inov=m.get('inov'), unq=m.get('unq_estimate'), train_fit=m.get('train_fit'))
            json.dump(res, open(OUT, 'w'))
            print('done', f, s, (m.get('fit') or {}).get('mean'), (m.get('inov') or {}).get('mean'), flush=True)
            del jobs[f]
            if s != 'completed': todo.append(f)
print('ALL DONE', flush=True)
