#!/usr/bin/env python3
"""Publish exactly one approved, audited partition/method validation bundle.

This command is never called by the analysis queue. It requires a separately
recorded explicit approval for these donor-level files before any API request.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
APP=Path(__file__).resolve().parents[1];sys.path[:0]=[str(APP),str(APP/"scripts")]
from app_helpers.validation_datasets import dataset
from run_partition_validation import RUN
import publish_camel_validation as publisher


def approval_gate(root,key,method):
    config=dataset(key,method)
    approval=json.loads((root/"publication_approval.json").read_text())
    report=json.loads((root/"public_bundle/release_manifest.json").read_text())
    manifest=json.loads((root/"public_bundle/manifest.json").read_text())
    if not approval.get("approved") or approval.get("module_set")!=key or method not in approval.get("methods",[]):
        raise ValueError("Explicit approval for this partition and method is required")
    if approval.get("score_rows")!=config.expected_rows() or report.get("score_rows")!=config.expected_rows():
        raise ValueError("Approved donor-score grain differs from the audited bundle")
    if approval.get("score_files")!=config.modules or approval.get("transformation_files")!=config.modules:
        raise ValueError("Approved module-file counts differ from this partition")
    if approval.get("transformation_rows")!=report.get("transform_rows") or approval.get("validation_samples")!=report.get("samples"):
        raise ValueError("Approved transformation or sample grain differs from the audited bundle")
    if report.get("status")!="passed" or report.get("modules")!=config.modules:
        raise ValueError("Release audit has not passed")
    if manifest.get("module_set")!=key or manifest.get("network_method")!=method:
        raise ValueError("Publication would mix module definitions or methods")
    if set(report["completed_rules"])!=set(manifest["completed_rules"]):
        raise ValueError("Repeat the release audit after the latest analysis milestone")
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=RUN)
    p.add_argument("--module-set",choices=("full_cohort","muscle_control_l4","muscle_all_donors_l4"),required=True)
    p.add_argument("--method",choices=("standard","control_anchored"),default="control_anchored")
    p.add_argument("--workers",type=int,choices=(1,2),default=2)
    args=p.parse_args(); root=args.root/args.module_set/args.method
    config=approval_gate(root,args.module_set,args.method)
    publisher.PREFIX=config.relative_root(args.method).as_posix()
    # The isolated CLI process binds a fixed upload prefix; no app globals or
    # original analysis directories are changed. Existing uploader preserves
    # immutable snapshots, MD5 reuse, reader readback and index-race safeguards.
    sys.argv=[publisher.__file__,"--root",str(root),"--workers",str(args.workers)]
    publisher.main()


if __name__=="__main__":main()
