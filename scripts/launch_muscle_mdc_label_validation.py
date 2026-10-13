#!/usr/bin/env python3
"""Detach the checked MDC validation run; leave prediction processes untouched."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone


def main():
    app=Path(__file__).resolve().parents[1]
    root=app.parents[1]/'out/mdc_muscle_control_l4/20261013_label_balance_ct_ts'
    record_path=root/'background_process.json'
    if record_path.exists():
        previous=json.loads(record_path.read_text());cmdline=Path(f"/proc/{previous['pid']}/cmdline")
        if cmdline.exists() and 'run_muscle_mdc_label_validation.py' in cmdline.read_text():
            print(json.dumps({'status':'already_running','pid':previous['pid']}));return
    command=[sys.executable,str(app/'scripts/run_muscle_mdc_label_validation.py'),'--stage','all','--workers','8']
    env={**os.environ,**{k:'1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']}}
    with (root/'background.log').open('ab',buffering=0) as log:
        process=subprocess.Popen(command,cwd=app,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,preexec_fn=lambda:os.nice(5))
    record={'status':'started','pid':process.pid,'workers':8,'started_utc':datetime.now(timezone.utc).isoformat(),'command':command}
    record_path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))


if __name__=='__main__':main()
