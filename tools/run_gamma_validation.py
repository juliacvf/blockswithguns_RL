"""Watch isolated training and validate immutable snapshots with verified control reuse."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,indent=2),encoding='utf-8')

def main(folder):
    old=folder.parent/'training_20261111_round2'
    manifest=read(folder/'manifest.json')
    prior=read(old/'manifest.json')
    current=manifest['input_hashes'];previous=prior['input_hashes']
    assert current.keys()==previous.keys()
    changes={p:dict(previous=previous[p],current=h) for p,h in current.items() if previous[p]!=h}
    assert all(Path(p)==Path('MeuAlgo/train.py') for p in changes)
    assert all(sha(ROOT/p)==h for p,h in current.items())
    assert manifest['seed']==prior['seed'] and manifest['actual_train_map_seeds']==prior['actual_train_map_seeds']
    assert all(manifest[k]==prior[k] for k in ['epsilon_start','alpha_start','groups','episodes_before','additional_episodes'])
    assert sha(folder/'baseline.npz')==sha(old/'baseline.npz')
    provenance=dict(training_source_changes=changes,source_manifest=str(old/'manifest.json'),source_manifest_sha256=sha(old/'manifest.json'),reused=[])
    shutil.copyfile(old/'checkpoint_120.npz',folder/'control_gamma099_120.npz')
    for source_label,target_label in [('baseline','baseline'),('checkpoint_120','control_gamma099_120')]:
        src=old/f'validation_{source_label}.json'
        data=read(src)
        assert data['complete'] and len(data['matches'])==150
        assert data['manifest']['official_inputs']==previous
        assert data['manifest']['weight_sha256']==sha(folder/f'{target_label}.npz')
        data['manifest']['official_inputs']=current
        data['manifest']['checkpoint']=str(folder/f'{target_label}.npz')
        for row in data['matches']:row['checkpoint']=target_label
        save(folder/f'validation_{target_label}.json',data)
        provenance['reused'].append(dict(source=str(src),sha256=sha(src),target=f'validation_{target_label}.json',reason='Identical frozen inference inputs, initial weights and validation maps; training source does not run during evaluation'))
    save(folder/'baseline_reuse.json',provenance)
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    for count in [40,80,120]:
        label=f'checkpoint_{count:03d}'
        while not (folder/f'{label}.npz').exists():
            if (folder/'progress.json').exists():
                p=read(folder/'progress.json')
                print(json.dumps(dict(waiting_for=label,training_matches=p['training_matches'],training_results=p['results'])),flush=True)
            time.sleep(30)
        with (folder/f'validation_{label}.log').open('w') as log:
            subprocess.run([sys.executable,'-B',str(ROOT/'tools/evaluate_trained_generalista.py'),str(folder),'validation',label,'--workers','3','--stop-if-dominated'],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        result=read(folder/f'validation_{label}.json')
        print(json.dumps(dict(evaluated=label,complete=result['complete'],summary=result['summary'],rejection=result.get('rejection'))),flush=True)
    print('ALL_VALIDATIONS_FINISHED',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    main(parser.parse_args().folder)
