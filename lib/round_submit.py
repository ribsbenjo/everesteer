"""Submit prepared round predictions for one model via the event lane. usage: round_submit.py TAG MODEL_NAME SLOT LABEL"""
import sys, json
from everestapi import EverestAPI
tag, name, slot, label = sys.argv[1:5]
api = EverestAPI()
r = api.submit_event_predictions(slot, f'/home/climber/work/rounds/{tag}/{name}_preds.parquet', model_pkl=f'/home/climber/work/models/{name}.pkl', model_pkl_python_version='3.11', client_label=label, wait=False)
keep = {k: v for k, v in r.items() if k in ('upload_id','status','error','round','window','scoring_window','n_predictions','coverage_pct','accepted','message','detail','model_id','model_name','pkl_sha256','size_bytes')}
print(json.dumps(keep, indent=1, default=str))
