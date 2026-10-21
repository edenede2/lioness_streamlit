#!/usr/bin/env python3
"""Serialized brain completion queue; never uploads, commits or pushes.

Run only after any manually started analysis stage exits. At most two analysis
workers are active. Checkpoints make every stage resumable after interruption.
Motor analyses deliberately wait for the separately reviewed brain milestone.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
APP=Path(__file__).resolve().parents[1];sys.path[:0]=[str(APP),str(APP/"scripts")]
from run_partition_validation import RUN
from run_camel_validation import atomic_json


def stages():
    result=[]
    for method in ("control_anchored","standard"):
        result.append((method,"first_release_associations","build_partition_validation_statistics.py",["--rules","first_release","--workers","2"]))
    for method in ("control_anchored","standard"):
        for stage in ("pilot","run"):
            result.append((method,"mdc_"+stage,"run_partition_validation_mdc.py",["--stage",stage,"--workers","2"]))
        result.append((method,"first_review","build_partition_validation_review.py",[]))
        result.append((method,"first_release_qa","validate_partition_validation.py",["--release"]))
        result.append((method,"first_aggregate_install","build_partition_validation_review.py",["--install-aggregate-review"]))
    for method in ("control_anchored","standard"):
        result.extend([
            (method,"remaining_associations","build_partition_validation_statistics.py",["--rules","all_masks","--workers","2"]),
            (method,"final_review","build_partition_validation_review.py",[]),
            (method,"final_qa","validate_partition_validation.py",["--release"]),
            (method,"final_aggregate_install","build_partition_validation_review.py",["--install-aggregate-review"]),
        ])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=RUN)
    p.add_argument("--detached",action="store_true",help="Launch a persistent queue with a private log")
    args=p.parse_args(); root=args.root; root.mkdir(parents=True,exist_ok=True)
    if args.detached:
        log=root/"pipeline.log"
        with log.open("a") as output:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--root",str(root)],
                stdout=output,stderr=subprocess.STDOUT,start_new_session=True,cwd=APP)
        print(json.dumps(dict(pid=child.pid,log=str(log),uploads=False,commits=False,pushes=False)))
        return
    with (root/"pipeline.lock").open("a") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("Another validation queue is already running")
        status_path=root/"pipeline_status.json"
        for method,name,script,extra in stages():
            started=time.monotonic()
            status=dict(status="running",module_set="full_cohort",method=method,stage=name,
                started_utc=datetime.now(timezone.utc).isoformat(),motor_started=False,
                uploads=False,commits=False,pushes=False)
            atomic_json(status,status_path)
            command=[sys.executable,str(APP/"scripts"/script),"--root",str(root),"--module-set","full_cohort","--method",method,*extra]
            print(json.dumps(dict(stage=name,method=method,status="running")),flush=True)
            result=subprocess.run(command,cwd=APP,check=False)
            if result.returncode:
                atomic_json({**status,"status":"failed","returncode":result.returncode},status_path)
                raise RuntimeError(f"Validation stage failed: {method}/{name}; see private pipeline.log")
            print(json.dumps(dict(stage=name,method=method,status="complete",elapsed_seconds=time.monotonic()-started)),flush=True)
        atomic_json(dict(status="brain_analysis_complete_publication_pending",motor_started=False,
            next_step="Review and commit validated brain summaries; approve/upload donor bundle; then proceed to motor validation",
            completed_utc=datetime.now(timezone.utc).isoformat(),uploads=False,commits=False,pushes=False),status_path)


if __name__=="__main__":main()
