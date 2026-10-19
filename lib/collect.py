import json
from everestapi import EverestAPI
api = EverestAPI()
OUT='/home/climber/work/lib/core_probe_results.json'; res=json.load(open(OUT))
r = api.list_compute_jobs(limit=200); jobs = r['jobs'] if isinstance(r, dict) and 'jobs' in r else r
if isinstance(jobs, dict): jobs = next(v for v in jobs.values() if isinstance(v, list))
from collections import Counter; print(len(jobs), Counter(j['status'] for j in jobs))
for j in jobs:
    if j['status'] != 'completed': continue
    st = api.get_job_status(j['id']); lab = (st.get('config') or {}).get('client_label') or ''
    if not lab.startswith('core-single-'): continue
    f = lab[len('core-single-'):]
    if f in res: continue
    m=(st.get('metrics') or {}).get('cv') or {}
    res[f]=dict(status='completed', cost=st.get('cost_cents'), fit=m.get('fit'), inov=m.get('inov'), unq=m.get('unq_estimate'), train_fit=m.get('train_fit'))
json.dump(res, open(OUT,'w')); print('results now', len(res), 'total cost cents', sum((r.get('cost') or 0) for r in res.values()))
