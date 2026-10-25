#!/usr/bin/env python3
"""Audit and stage the already-uploaded full-cohort validation MDC release.

No network recomputation, uploads, sharing changes, Git operations, or donor
publication. Optional backend checks use the app's existing read-only identity.
"""
from __future__ import annotations
import os
for _thread_option in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_thread_option] = "1"
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP),str(APP/"scripts")]
import numpy as np
import pandas as pd
from app_helpers.validation_datasets import dataset
from scripts.run_camel_validation import atomic_json,sha
from scripts.run_partition_validation import RUN
from scripts.validate_partition_validation import validate_mdc


def audit_local(root,method):
    root = Path(root);bundle = root/"public_bundle"
    manifest = json.loads((bundle/"manifest.json").read_text())
    release = json.loads((bundle/"release_manifest.json").read_text())
    audit = json.loads((root/"audit.json").read_text())
    config = dataset("full_cohort",method)
    if (manifest.get("module_set") != "full_cohort" or manifest.get("network_method") != method
            or manifest.get("mdc_status") != "complete" or release.get("status") != "passed"):
        raise ValueError("MDC release identity/completion failed")
    membership = json.loads((root/"discovery_membership_audit.json").read_text())
    if not membership.get("passed") or membership.get("validation_overlap") != 0 or audit["excluded_complete_donors"] != 464:
        raise ValueError("Discovery cohort exclusion gate failed")
    modules = set(manifest["completed_module_ids"])
    if len(modules) != 154 or 1918 not in modules or len(manifest["rules"]) != 31:
        raise ValueError("Unexpected full-cohort module/mask coverage")
    counts = {r["component"]:r for r in audit["component_counts"]}
    frame = pd.read_parquet(bundle/"mdc/resolved.parquet")
    validate_mdc(frame,modules,config.components,manifest["rules"],counts)
    if frame.loc[frame.component.eq("CT_AC__PCGBA23"),"inferential"].any():
        raise ValueError("Sparse AC–PCG must remain descriptive")
    for relative,expected in audit["source_hashes"].items():
        if sha(APP.parents[1]/relative) != expected:
            raise ValueError("Frozen MDC source hash changed")
    inventory = {k:v for k,v in release["files"].items() if k.startswith("mdc/")}
    if len(inventory) != 925:
        raise ValueError("Incomplete MDC summary/adjacency inventory")
    total = 0
    for relative,entry in inventory.items():
        path = bundle/relative
        if sha(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError("MDC file differs from completed release receipt")
        total += path.stat().st_size
    parts = list((root/"mdc_parts").glob("*/M*.json"))
    if len(parts) != 924:
        raise ValueError("Incomplete MDC numerical QA coverage")
    errors = [json.loads(p.read_text())["direct_adjacency_error"] for p in parts]
    if max(errors) > 1e-10:
        raise ValueError("Original direct adjacency QA failed")
    return dict(status="passed",module_set="full_cohort",network_method=method,
        modules=154,components=6,edge_sets=31,mdc_rows=len(frame),adjacency_files=924,
        total_mdc_bytes=total,validation_donors=793,excluded_complete_donors=464,
        component_counts=list(counts.values()),sample_permutations=200,gene_permutations=200,seed=42,
        max_existing_direct_adjacency_error=max(errors),mdc_sha256=sha(bundle/"mdc/resolved.parquet"),
        privacy="Group/module aggregates only; no new donor fields or scores",
        fdr="Separate BH across valid modules for every component, frozen mask, direction, and null; conservative maximum across nulls",
        sparse_component="AC–PCG: 3 AD/6 Controls; no inferential p-values or FDR")


def sampled_direct_check(root,method):
    import run_partition_validation as P
    import run_partition_validation_mdc as R
    import run_camel_validation_mdc as M
    state = P.load_inputs("full_cohort",method);R.bind(state)
    saved = pd.read_parquet(Path(root)/"public_bundle/mdc/resolved.parquet")
    records = []
    # M188 exercises duplicate-node conventions; M1918 is a standard sentinel.
    for module in (188,1918):
        assignment,_,rows,columns,masks,_ = P.module_source(module,state)
        tissues = assignment.public_tissue.to_numpy()
        for component in ("TS_DLPFC","CT_AC__DLPFC","CT_AC__PCGBA23"):
            wanted = state["config"].components[component]
            nodes = np.flatnonzero(np.isin(tissues,wanted));local = {int(v):i for i,v in enumerate(nodes)}
            keep = ((tissues[rows] == wanted[0]) & (tissues[columns] == wanted[0])) if len(wanted)==1 else (
                np.isin(tissues[rows],wanted)&np.isin(tissues[columns],wanted)&(tissues[rows] != tissues[columns]))
            lr = np.array([local[int(i)] for i in rows[keep]],int)
            lc = np.array([local[int(i)] for i in columns[keep]],int)
            ad,control,lookup,_ = M.component_background(component)
            indices = np.array([lookup[assignment.iloc[i].public_tissue+"_"+assignment.iloc[i]["Gene Symbol"]] for i in nodes])
            power = 3 if len(wanted)==1 else 2
            weights = []
            for matrix in (ad,control):
                if not len(lr):
                    weights.append(np.array([]))
                else:
                    r = np.nan_to_num(np.corrcoef(matrix[:,indices],rowvar=False)[lr,lc])
                    weights.append(np.sign(r)*np.abs(np.clip(r,-1,1))**power)
            selected = saved.loc[saved.module.eq(module)&saved.component.eq(component)].set_index("differential_edge_rule")
            error = 0.
            for rule,mask in masks.items():
                retained = mask[keep]
                if not retained.any():
                    if np.isfinite(selected.loc[rule,"mdc"]):
                        raise ValueError("Empty mask has finite MDC")
                    continue
                numerator = np.mean(np.abs(weights[0][retained]));denominator = np.mean(np.abs(weights[1][retained]))
                direct = numerator/denominator if denominator > 1e-10 else np.nan
                np.testing.assert_allclose([selected.loc[rule,"mean_abs_ad"],selected.loc[rule,"mean_abs_control"],selected.loc[rule,"mdc"]],
                    [numerator,denominator,direct],rtol=1e-10,atol=1e-12,equal_nan=True)
                if np.isfinite(direct):error = max(error,abs(direct-selected.loc[rule,"mdc"]))
            records.append(dict(module=module,component=component,masks_checked=len(masks),max_ratio_error=error))
        M._COMPONENT_CACHE.clear()
    return records


def verify_backend(method,reader_credentials):
    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2 import service_account
    index = json.loads((APP/"drive_file_index.json").read_text())["files"]
    prefix = f"validation/full_cohort/{method}"
    session = AuthorizedSession(service_account.Credentials.from_service_account_file(
        str(reader_credentials),scopes=["https://www.googleapis.com/auth/drive.readonly"]))
    verified = []
    for relative in ("manifest.json","mdc/resolved.parquet","mdc/adjacency/TS_DLPFC/M1918.npz"):
        key = prefix+"/"+relative;entry = index[key]
        url = "https://www.googleapis.com/drive/v3/files/"+entry["id"]
        response = session.get(url,params={"fields":"id,mimeType,size,md5Checksum"},timeout=30)
        response.raise_for_status();metadata = response.json()
        if metadata.get("md5Checksum") != entry["md5"] or int(metadata.get("size",-1)) != entry["size"]:
            raise ValueError("Indexed MDC backend metadata changed")
        response = session.get(url,params={"alt":"media"},timeout=90)
        response.raise_for_status()
        if len(response.content) != entry["size"] or hashlib.md5(response.content).hexdigest() != entry["md5"]:
            raise ValueError("App reader failed MDC content verification")
        if relative == "manifest.json":
            manifest = response.json()
            if not manifest.get("backend_verified") or manifest.get("mdc_status") != "complete" or manifest.get("network_method") != method:
                raise ValueError("Backend manifest cannot enable this MDC cohort")
        verified.append(dict(path=key,bytes=entry["size"],md5=entry["md5"]))
    return dict(status="passed",api_file_requests=6,read_only_identity=True,files=verified,uploads=0,sharing_changes=False)


def install(root,method,report):
    root = Path(root);target = APP/f"data/validation/full_cohort/{method}"
    rollback = root/"rollback"/f"mdc_release_{time.time_ns()}"
    for relative in ("mdc/resolved.parquet","review/mdc_comparison.parquet"):
        source = root/"public_bundle"/relative;destination = target/relative
        if destination.is_file():
            saved = rollback/relative;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(destination,saved)
        destination.parent.mkdir(parents=True,exist_ok=True)
        temp = destination.with_suffix(".partial");shutil.copy2(source,temp);temp.replace(destination)
    atomic_json(report,target/"mdc/release_qa.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=RUN)
    parser.add_argument("--method",choices=("standard","control_anchored"),required=True)
    parser.add_argument("--verify-backend",action="store_true")
    parser.add_argument("--reader-credentials",type=Path,default=APP/"lionessexplorer-05e37b3ea2a6.json")
    parser.add_argument("--install",action="store_true")
    args = parser.parse_args();root = args.root/"full_cohort"/args.method
    report = audit_local(root,args.method)
    report["sampled_independent_reproduction"] = sampled_direct_check(root,args.method)
    report["backend_verification"] = verify_backend(args.method,args.reader_credentials) if args.verify_backend else dict(status="not_checked")
    report["checked_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(report,root/"mdc_publication_qa.json")
    if args.install:
        if report["backend_verification"]["status"] != "passed":
            raise ValueError("Verify reader access before installing the release")
        install(root,args.method,report)
    print(json.dumps(report))
