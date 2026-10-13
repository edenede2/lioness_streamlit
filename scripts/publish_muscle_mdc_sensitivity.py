#!/usr/bin/env python3
"""Publish complete no-q90 MDC inference after the background analysis passes QA.

Only the explicitly listed public files are staged. Private outputs and other
working-tree changes never enter this commit. The watcher stops on failed QA,
failed analysis, unexpected staged work, or a rejected GitHub push.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

APP=Path(__file__).resolve().parents[1]
UPSTREAM=APP.parents[1]/'out/mdc_muscle_control_l4/20261013_no_q90_mdc'
PUBLIC=APP/'data/muscle_control_l4/mdc_sensitivity'
FILES=['data/data_manifest.json','data/muscle_control_l4/module_set_manifest.json',
       'data/muscle_control_l4/mdc_sensitivity/summary.parquet',
       'data/muscle_control_l4/mdc_sensitivity/resolved.parquet',
       'data/muscle_control_l4/mdc_sensitivity/manifest.json','drive_file_index.json']


def run(command):
    subprocess.run(command,cwd=APP,check=True,env={**os.environ,'GIT_TERMINAL_PROMPT':'0',
                   'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})


def status(record):
    record['updated_utc']=datetime.now(timezone.utc).isoformat()
    path=UPSTREAM/'publication_status.json'
    temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(record,indent=2)+'\n')
    os.replace(temporary,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait',action='store_true')
    p.add_argument('--push',action='store_true')
    p.add_argument('--sync-drive',action='store_true')
    p.add_argument('--detach',action='store_true')
    args=p.parse_args()
    if args.detach:
        command=[sys.executable,str(Path(__file__).resolve()),*[arg for arg in sys.argv[1:] if arg!='--detach']]
        with (UPSTREAM/'publication_background.log').open('ab',buffering=0) as log:
            process=subprocess.Popen(command,cwd=APP,stdin=subprocess.DEVNULL,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'publisher_pid':process.pid,'status':'watcher_started'}))
        return
    record={'status':'waiting_for_complete_inference'}
    rollback=UPSTREAM.parent/'public_rollback'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')/'pre_inference_public'
    rollback.mkdir(parents=True,exist_ok=True)
    for relative in FILES:
        path=APP/relative
        if path.exists():
            destination=rollback/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
    record['rollback']=str(rollback)
    status(record)
    try:
        while True:
            manifest=json.loads((UPSTREAM/'run_manifest.json').read_text())
            if manifest['status']=='complete':break
            if manifest['status']=='failed':raise RuntimeError(manifest.get('error','Analysis failed'))
            if not args.wait:
                print('Inference is incomplete; no publication was attempted.')
                return
            record['completed_sample_permutations']=manifest.get('completed_sample_permutations',0)
            record['completed_gene_permutations']=manifest.get('completed_gene_permutations',0)
            status(record)
            time.sleep(30)
        if manifest['completed_sample_permutations']!=200 or manifest['completed_gene_permutations']!=200:
            raise ValueError('The required 200+200 nulls are incomplete')
        record['status']='validating';status(record)
        run([sys.executable,'scripts/register_muscle_mdc_sensitivity.py'])
        run([sys.executable,'-m','pytest','-q','tests/test_muscle_mdc_sensitivity.py',
             'tests/test_streamlit_app.py::test_streamlit_muscle_mdc_normalization_sensitivity_routes_every_view'])
        if args.sync_drive:
            record['status']='syncing_drive';status(record)
            run([sys.executable,'scripts/upload_prediction_data.py',
                 '--oauth-client','client_secret_688842367267-8o25rcsq6jvrb1go8d0gsr1o304v1knq.apps.googleusercontent.com.json',
                 '--oauth-token','google-drive-oauth-token.json',
                 '--inventory-credentials','lionessexplorer-05e37b3ea2a6.json',
                 '--root-folder-id','1dTj5SkLxuDIvII5LayLViqOzjKQt-dos',
                 '--local-prediction','data/muscle_control_l4','--remote-directory','muscle_control_l4',
                 '--manifest-name','module_set_manifest.json'])
            run([sys.executable,'scripts/build_drive_index.py','--credentials','lionessexplorer-05e37b3ea2a6.json',
                 '--root-folder-id','1dTj5SkLxuDIvII5LayLViqOzjKQt-dos','--local-data','data',
                 '--output','drive_file_index.json','--skip-local-md5','--validation-prefix','muscle_control_l4'])
        staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=APP,text=True).splitlines()
        if staged:raise RuntimeError('Existing staged user changes require review before the automated result commit')
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=APP,text=True).strip()
        if branch!='master':raise RuntimeError('Public app result publication requires the existing master branch')
        record['status']='committing_public_results';status(record)
        run(['git','add','-u','--',FILES[0],FILES[1],FILES[5]])
        run(['git','add','-f','--',*FILES[2:5]])
        staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=APP,text=True).splitlines()
        if set(staged)-set(FILES):raise RuntimeError('Unexpected files entered the result commit')
        if staged:run(['git','commit','-m','Publish complete no-q90 muscle MDC inference'])
        record['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=APP,text=True).strip()
        if args.push:run(['git','push','origin','master'])
        record['status']='published' if args.push else 'committed';status(record)
        print(json.dumps(record,indent=2))
    except Exception as error:
        record.update(status='failed',error=str(error));status(record)
        raise


if __name__=='__main__':main()
