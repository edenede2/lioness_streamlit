#!/usr/bin/env python3
"""Publish explicitly approved brain milestones only after independent QA.

Separate from the analysis queue: no analysis, commits, pushes or sharing
changes. A first-release and a final snapshot are eligible per method. OAuth
and reader credentials are used only by the existing scoped uploader.
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

APP=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(APP),str(APP/"scripts")]
from app_helpers.validation_datasets import dataset
from run_partition_validation import RUN
from run_camel_validation import atomic_json,sha
from publish_partition_validation import approval_gate

METHODS=("control_anchored","standard")


def candidate(root,method,completed):
    path=Path(root)/"full_cohort"/method
    report_path=path/"public_bundle/release_manifest.json"
    manifest_path=path/"public_bundle/manifest.json"
    if not report_path.exists() or not manifest_path.exists(): return None
    report=json.loads(report_path.read_text()); manifest=json.loads(manifest_path.read_text())
    if report.get("status")!="passed" or report.get("mdc_status")!="complete": return None
    rules=set(manifest.get("completed_rules",[])); expected=set(manifest["rules"])
    first={r for r in expected if r=="all" or "__hedges_g__fixed_0.40__" in r}
    if set(report.get("completed_rules",[]))!=rules or not first<=rules: return None
    # Intermediate masks are intentionally not published one at a time, to
    # minimize Drive writes and avoid exposing unaudited incremental families.
    if rules not in (first,expected): return None
    digest=sha(report_path)
    if completed.get(method,{}).get("audit_sha256")==digest: return None
    approval_gate(path,"full_cohort",method)
    return dict(method=method,stage="final" if rules==expected else "first_release",audit_sha256=digest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=RUN)
    parser.add_argument("--watch",action="store_true",help="Wait for the already-running analysis queue")
    parser.add_argument("--timeout-hours",type=float,default=48.)
    args=parser.parse_args()
    lock=(args.root/"brain_publication.lock").open("a")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError("A brain publication waiter is already active")
    # Fail before waiting if either method lacks its explicit scoped approval.
    for method in METHODS:
        config=dataset("full_cohort",method)
        approval=json.loads((args.root/"full_cohort"/method/"publication_approval.json").read_text())
        if not approval.get("approved") or approval.get("module_set")!="full_cohort" or method not in approval.get("methods",[]) or approval.get("score_rows")!=config.expected_rows():
            raise ValueError("Both brain methods require explicit approval before publication is queued")
    state_path=args.root/"brain_publication_status.json"
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(completed={})
    started=time.monotonic()
    while True:
        for method in METHODS:
            job=candidate(args.root,method,state["completed"])
            if job is None: continue
            state.update(status="uploading",current=job,pushes=False,commits=False)
            atomic_json(state,state_path);print(json.dumps({"status":"uploading",**job}),flush=True)
            result=subprocess.run([sys.executable,str(APP/"scripts/publish_partition_validation.py"),
                "--root",str(args.root),"--module-set","full_cohort","--method",method,"--workers","2"],cwd=APP,check=False)
            if result.returncode:
                state.update(status="failed",returncode=result.returncode);atomic_json(state,state_path)
                raise RuntimeError("Approved publication stopped; no automatic Git operation was attempted")
            state["completed"][method]={**job,"installed_utc":datetime.now(timezone.utc).isoformat()}
            state.update(status="waiting_for_next_validated_milestone",current=None)
            atomic_json(state,state_path)
        if all(state["completed"].get(method,{}).get("stage")=="final" for method in METHODS):
            state.update(status="complete_ready_for_local_commit",current=None)
            atomic_json(state,state_path);print(json.dumps({"status":state["status"],"git_pushed":False}),flush=True)
            return
        pipeline_path=args.root/"pipeline_status.json"
        if pipeline_path.exists() and json.loads(pipeline_path.read_text()).get("status")=="failed":
            state.update(status="stopped_analysis_failed",current=None);atomic_json(state,state_path)
            raise RuntimeError("Analysis queue failed; publication will not bypass its gates")
        if not args.watch: return
        if time.monotonic()-started>=args.timeout_hours*3600:
            state.update(status="wait_timeout",current=None);atomic_json(state,state_path)
            raise TimeoutError("No full release within the bounded publication wait; restart after reviewing analysis status")
        state.update(status="waiting_for_release_qa",current=None,pushes=False,commits=False)
        atomic_json(state,state_path)
        time.sleep(30)


if __name__=="__main__":main()
