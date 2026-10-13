#!/usr/bin/env python3
"""Wait for complete validated MDC outputs, then make a scoped result commit.

Refuse unrelated working-tree changes. Optional push fails explicitly on missing
authentication; neither analysis completion nor a local commit implies deployment.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

APP=Path(__file__).resolve().parents[1]
ROOT=APP.parents[1]/'out/mdc_muscle_control_l4/20261013_label_balance_ct_ts'
PUBLIC=APP/'data/muscle_control_l4/mdc_label_validation'
FILES=['data/data_manifest.json','data/muscle_control_l4/module_set_manifest.json',
       *[f'data/muscle_control_l4/mdc_label_validation/{n}' for n in ['manifest.json','results.parquet','label_replicates.parquet','balanced_replicates.parquet']]]


def run(cmd):
    subprocess.run(cmd,cwd=APP,check=True,env={**os.environ,'GIT_TERMINAL_PROMPT':'0',
        'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait',action='store_true');p.add_argument('--detach',action='store_true')
    p.add_argument('--push',action='store_true');p.add_argument('--expected-head')
    args=p.parse_args()
    if args.detach:
        with (ROOT/'publication.log').open('ab',buffering=0) as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*[a for a in sys.argv[1:] if a!='--detach']],cwd=APP,
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'publisher_pid':proc.pid}));return
    record={'status':'waiting'}
    def save():
        record['updated_utc']=datetime.now(timezone.utc).isoformat()
        tmp=ROOT/'publication_status.partial';tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(ROOT/'publication_status.json')
    save()
    try:
        while True:
            source=json.loads((ROOT/'run_manifest.json').read_text())
            public=json.loads((PUBLIC/'manifest.json').read_text())
            if source['status']=='failed':raise RuntimeError(source.get('error','Analysis failed'))
            if source['status']==public['status']=='complete':break
            if not args.wait:print('Analysis incomplete; nothing published.');return
            time.sleep(30)
        # The analysis writes its public manifest last; verify immutable file hashes below.
        staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=APP,text=True).splitlines()
        if staged:raise RuntimeError('Existing staged changes require review')
        dirty=subprocess.check_output(['git','diff','--name-only'],cwd=APP,text=True).splitlines()
        if set(dirty)-set(FILES):raise RuntimeError('Unrelated tracked changes require review')
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=APP,text=True).strip()
        if args.expected_head and head!=args.expected_head:raise RuntimeError('App HEAD changed; review before result commit')
        if subprocess.check_output(['git','branch','--show-current'],cwd=APP,text=True).strip()!='master':raise RuntimeError('Unexpected app branch')
        record['status']='validating';save()
        run([sys.executable,'scripts/validate_muscle_mdc_label_results.py'])
        run([sys.executable,'scripts/register_muscle_mdc_label_validation.py'])
        run([sys.executable,'-m','pytest','-q','tests/test_mdc_label_validation.py',
            'tests/test_streamlit_app.py::test_streamlit_muscle_mdc_label_validation_keeps_existing_catalog',
            'tests/test_streamlit_app.py::test_streamlit_muscle_mdc_normalization_sensitivity_routes_every_view'])
        run(['git','add','-u','--',*FILES[:2]])
        run(['git','add','-f','--',*FILES[2:]])
        staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=APP,text=True).splitlines()
        if set(staged)-set(FILES):raise RuntimeError('Unexpected staged files')
        if staged:run(['git','commit','-m','Publish complete MDC label, balanced and CT-TS validation'])
        record['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=APP,text=True).strip()
        if args.push:run(['git','push','origin','master'])
        record['status']='published' if args.push else 'committed';save();print(json.dumps(record,indent=2))
    except Exception as e:
        record.update(status='needs_review',error=str(e));save();raise


if __name__=='__main__':main()
