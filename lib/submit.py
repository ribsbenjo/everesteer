import sys, json, time
from everestapi import EverestAPI
api = EverestAPI()
model, preds, pkl, label = sys.argv[1:5]
r = api.submit_validation_diagnostics(model, preds, model_pkl=pkl, model_pkl_python_version='3.11', client_label=label, wait=True)
keep = {k: v for k, v in r.items() if k not in ('predictions','model_pkl','fields','url','command','commands','upload')}
print(json.dumps(keep, indent=1, default=str)[:4000])
