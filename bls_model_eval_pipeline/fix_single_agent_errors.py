#!/usr/bin/env python3
import json,sys,subprocess
from pathlib import Path
cp=Path('bls_model_eval_pipeline/results/single_agent_results.json')
if not cp.exists():
    print('checkpoint missing')
    sys.exit(1)
with open(cp) as f:
    data=json.load(f)
affected=[]
for cfg,vals in list(data.items()):
    keys_to_remove=[]
    for k,v in list(vals.items()):
        ans=(v.get('answer') or '')
        err=v.get('error')
        if err or 'MODEL_UNAVAILABLE' in ans or 'No response captured' in ans or ans.strip().startswith('MODEL_UNAVAILABLE'):
            keys_to_remove.append(k)
    if keys_to_remove:
        affected.append((cfg,keys_to_remove))
        for k in keys_to_remove:
            del data[cfg][k]
        print(f'Removed {len(keys_to_remove)} failing entries from {cfg}')
# write back
with open(cp,'w') as f:
    json.dump(data,f,indent=2,ensure_ascii=False)
print('Saved cleaned checkpoint')
if not affected:
    print('No failures found; nothing to re-run')
    sys.exit(0)
# Re-run affected configs
for cfg,keys in affected:
    print('\nRe-running architecture',cfg,'for',len(keys),'questions')
    cmd=['/Users/kolli/bls-chatbot/venv/bin/python','bls_model_eval_pipeline/run_tests.py','--mode','single','--config-id',cfg,'--delay','1.0']
    p=subprocess.Popen(cmd)
    p.communicate()
    if p.returncode!=0:
        print('ERROR: run_tests returned',p.returncode)
# regenerate docs
print('\nRegenerating single-agent markdown')
subprocess.run(['/Users/kolli/bls-chatbot/venv/bin/python','bls_model_eval_pipeline/generate_docs.py'])
# quick verification
with open(cp) as f:
    data2=json.load(f)
for cfg in data2:
    print(cfg,'now has',len(data2[cfg]),'answers')
print('Done')
