#!/usr/bin/env python3
"""Start the resumable MDC runner independently of an interactive SSH session."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    if not 1<=args.workers<=12:
        raise ValueError('The measured memory/CPU budget supports one to twelve workers')
    app=Path(__file__).resolve().parents[1]
    output=app.parents[1]/'out/mdc_muscle_control_l4/20261013_no_q90_mdc'
    output.mkdir(parents=True,exist_ok=True)
    pid_path=output/'background_process.json'
    if pid_path.exists():
        previous=json.loads(pid_path.read_text())
        cmdline=Path(f"/proc/{previous['pid']}/cmdline")
        if cmdline.exists() and 'run_muscle_mdc_sensitivity.py' in cmdline.read_text():
            print(json.dumps({'status':'already_running','pid':previous['pid']}))
            return
    command=[sys.executable,str(app/'scripts/run_muscle_mdc_sensitivity.py'),
             '--stage','nulls','--workers',str(args.workers)]
    environment=os.environ.copy()
    environment.update({name:'1' for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})
    log_path=output/'background.log'
    with log_path.open('ab',buffering=0) as log:
        process=subprocess.Popen(command,cwd=app,env=environment,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(5))
    record={'status':'started','pid':process.pid,'workers':args.workers,
            'started_utc':datetime.now(timezone.utc).isoformat(),'log':str(log_path),'command':command}
    temporary=pid_path.with_suffix('.partial')
    temporary.write_text(json.dumps(record,indent=2)+'\n')
    os.replace(temporary,pid_path)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
