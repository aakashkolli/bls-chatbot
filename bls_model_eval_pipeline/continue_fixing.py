#!/usr/bin/env python3
import json,sys,subprocess
from pathlib import Path

RESULTS_DIR=Path('bls_model_eval_pipeline/results')
SA=RESULTS_DIR/'single_agent_results.json'
DA=RESULTS_DIR/'dual_agent_results.json'

EXPECTED=54

def load(p):
    if not p.exists():
        return {}
    return json.loads(p.read_text())

sdata=load(SA)
data=load(DA)

def analyze(d):
    out={}
    for cfg,vals in d.items():
        total=len(vals)
        errors=[k for k,v in vals.items() if v.get('error') or (v.get('answer') or '').strip().startswith('MODEL_UNAVAILABLE') or 'No response captured' in (v.get('answer') or '')]
        out[cfg]={'total':total,'errors':len(errors)}
    return out

print('Single-agent status:')
sa_status=analyze(sdata)
for k,v in sa_status.items():
    print(f" - {k}: {v['total']} answers, {v['errors']} errors")

print('\nDual-agent status:')
da_status=analyze(data)
for k,v in da_status.items():
    print(f" - {k}: {v['total']} answers, {v['errors']} errors")

# Remove bad entries (keep minimal removal behavior)
changed=False
for container,fp in [(sdata,SA),(data,DA)]:
    for cfg,vals in list(container.items()):
        bad=[k for k,v in vals.items() if v.get('error') or (v.get('answer') or '').strip().startswith('MODEL_UNAVAILABLE') or 'No response captured' in (v.get('answer') or '')]
        if bad:
            for k in bad:
                del container[cfg][k]
            print(f'Removed {len(bad)} bad entries from {cfg}')
            changed=True
if changed:
    SA.write_text(json.dumps(sdata,indent=2,ensure_ascii=False))
    DA.write_text(json.dumps(data,indent=2,ensure_ascii=False))
    print('Wrote cleaned checkpoints')

# Launch resumable runs for any config with <EXPECTED answers
# Build list of config ids to run
to_run=[]
for cfg,vals in sdata.items():
    if len(vals)<EXPECTED:
        to_run.append(('single',cfg,EXPECTED-len(vals)))
for cfg,vals in data.items():
    if len(vals)<EXPECTED:
        to_run.append(('dual',cfg,EXPECTED-len(vals)))

if not to_run:
    print('All configs have full answers — nothing to run')
    sys.exit(0)

print('\nScheduling runs for:')
for mode,cfg,missing in to_run:
    print(f' - {cfg} ({mode}) -> {missing} missing')
    # start run_tests.py for that config in background
    cmd=['/Users/kolli/bls-chatbot/venv/bin/python','bls_model_eval_pipeline/run_tests.py','--mode',mode,'--config-id',cfg,'--delay','1.0']
    subprocess.Popen(cmd)

print('Launched background runners. Monitor terminals for progress.')
