#!/usr/bin/env python3
"""Validate/stage an incremental CAMEL release; never upload or push.

The default touches only this run's private/staged output. --install-hot-cache
copies aggregate tables and a backend-gated manifest to the app, with rollback.
Donor-level score and transform files stay outside GitHub, on the Drive path.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
APP=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(APP))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from app_helpers.correlations import benjamini_hochberg
from app_helpers.data import load_resolved_statistics
from run_camel_validation import RUN, EDGE_ROOT, REPO, atomic_json, atomic_parquet, sha, update_public_manifest


def validate(root):
    root=Path(root); bundle=root/"public_bundle"
    approval=json.loads((root/"publication_approval.json").read_text())
    if not approval.get("approved") or approval.get("score_rows")!=9415878:
        raise ValueError("Explicit scoped publication approval is required")
    if not json.loads((root/"existing_score_reproduction.json").read_text()).get("passed"):
        raise ValueError("Original R score reproduction gate failed")
    audit=json.loads((root/"audit.json").read_text())
    for relative,expected in audit["source_hashes"].items():
        if sha(REPO/relative)!=expected: raise ValueError("A frozen source changed during the analysis")
    manifest=json.loads((bundle/"manifest.json").read_text())
    if manifest.get("complete_modules")!=186 or "all" not in manifest.get("completed_rules",[]):
        raise ValueError("Complete score catalog and all-edge statistics are required")
    metadata=pd.read_parquet(bundle/"sample_metadata.parquet")
    allowed=set(metadata.sample_id)
    if len(allowed)!=793 or {"donor","projid"}&set(metadata): raise ValueError("Metadata privacy/grain failed")
    score_rows=0
    for module in manifest["completed_module_ids"]:
        path=bundle/f"modules/M{module}.parquet"; f=pd.read_parquet(path)
        qa=json.loads((root/f"checkpoints/M{module}.json").read_text())
        if not qa.get("passed") or sha(path)!=qa["score_sha256"]: raise ValueError("Scoring checkpoint/hash gate failed")
        if sha(EDGE_ROOT/f"M{module}__edge_statistics.parquet")!=qa["source_edge_sha256"]:
            raise ValueError("A discovery edge table changed during validation")
        if not set(f.sample_id)<=allowed or f.duplicated(["sample_id","component","differential_edge_rule"]).any():
            raise ValueError("Score identity/grain failed")
        np.testing.assert_allclose(f.signed_weight_sum,f.positive_weight_sum-f.negative_weight_magnitude,equal_nan=True)
        np.testing.assert_allclose(f.absolute_weight_sum,f.positive_weight_sum+f.negative_weight_magnitude,equal_nan=True)
        if not (f.n_positive_edges+f.n_negative_edges+f.n_zero_edges==f.n_retained_edges).all():
            raise ValueError("Sign-count identity failed")
        for rule in manifest["rules"]:
            if not rule.endswith("__either"): continue
            base=rule.rsplit("__",1)[0]
            def select(direction):
                return f.loc[f.differential_edge_rule.eq(base+"__"+direction)].set_index(["sample_id","component"]).sort_index()
            a,c,e=select("ad_higher"),select("control_higher"),select("either")
            for field in ("n_retained_edges","n_positive_edges","n_negative_edges","positive_weight_sum","negative_weight_magnitude"):
                np.testing.assert_allclose(a[field]+c[field],e[field],rtol=1e-10,atol=1e-9,equal_nan=True)
        score_rows+=len(f)
        maps=bundle/f"rint/M{module}.parquet"
        if sha(maps)!=qa["rint_sha256"]: raise ValueError("Frozen transformation hash changed")
    if score_rows!=9415878: raise ValueError("Unexpected total validation score rows")
    # Independently reproduce the full-family BH values, never groups × modules.
    for rule in manifest["completed_rules"]:
        for norm in ("standard_pruned","retained_edge"):
            f=pd.read_parquet(bundle/"statistics"/rule/f"{norm}.parquet")
            keys=["metric_family","component","phenotype","diagnosis_group"]
            if f.module.nunique()!=186 or f.duplicated(["module",*keys]).any(): raise ValueError("Statistics module family is incomplete")
            for p in ("p_raw","p_asinh","p_rint","p_spearman_raw","p_spearman_rint"):
                expected=f.groupby(keys,dropna=False)[p].transform(benjamini_hochberg)
                np.testing.assert_allclose(expected,f[p.replace("p_","q_",1)+"_across_modules"],atol=1e-12,equal_nan=True)
    mdc_path=bundle/"mdc/resolved.parquet"
    if mdc_path.exists():
        m=pd.read_parquet(mdc_path)
        small=m.loc[m.component.eq("CT_AC__PCGBA23")]
        if not small.empty and (small.inferential.any() or small.directional_fdr.notna().any()):
            raise ValueError("Sparse AC–PCG inference must remain unavailable")
    entries={}
    for p in sorted(bundle.rglob("*")):
        if not p.is_file() or p.suffix in (".lock",".partial") or p.name=="release_manifest.json": continue
        if p.stat().st_size>=95*1024**2: raise ValueError("Public shard exceeds 95 MiB")
        record=dict(bytes=p.stat().st_size,sha256=sha(p))
        if p.suffix==".parquet":
            schema=pq.read_schema(p)
            if {"donor","projid","donor_private","projid_private"}&set(schema.names): raise ValueError("Private identifier in release")
            record["rows"]=pq.ParquetFile(p).metadata.num_rows
        entries[str(p.relative_to(bundle))]=record
    report=dict(status="passed",score_rows=score_rows,samples=793,modules=186,completed_rules=manifest["completed_rules"],
        mdc_status=manifest.get("mdc_status"),files=entries,backend_verified=False,
        publication_approval="Explicit 2026-10-21 user approval; scores, frozen maps and existing pseudonymous metadata",
        predictions_unchanged=True,validation="Within-ROSMAP non-discovery donors, conditional on existing expression preprocessing")
    atomic_json(report,bundle/"release_manifest.json")
    return report


def concordance(root,rule,normalization):
    bundle=Path(root)/"public_bundle"
    validation=pd.read_parquet(bundle/"statistics"/rule/f"{normalization}.parquet")
    original=load_resolved_statistics("control_anchored",module_set="control_derived",
        differential_edge_rule=rule,score_normalization=normalization)
    keys=["module","metric_family","component","phenotype","diagnosis_group"]
    if original.duplicated(keys).any(): raise ValueError("Original comparison catalog has duplicate tests")
    columns=[*keys,"n","r_raw","p_raw","r_asinh","p_asinh","r_rint","p_rint","rho","p_spearman"]
    original=original[columns].copy()
    family=["metric_family","component","phenotype","diagnosis_group"]
    for prefix,p in (("raw","p_raw"),("asinh","p_asinh"),("rint","p_rint"),("spearman","p_spearman")):
        original["q_"+prefix]=original.groupby(family,dropna=False)[p].transform(benjamini_hochberg)
    original=original.rename(columns={c:"original_"+c for c in original if c not in keys})
    # Keep every validation row. Missing original strata (pooled/unclassified)
    # are explicitly missing, never borrowed from another group or cohort.
    v_columns=[*keys,"n","unavailable_reason","r_raw","p_raw","r_asinh","p_asinh","r_rint","p_rint","rho_raw","p_spearman_raw",
               "rho_asinh","p_spearman_asinh","rho_rint","p_spearman_rint",
               "q_raw_across_modules","q_asinh_across_modules","q_rint_across_modules","q_spearman_raw_across_modules",
               "q_spearman_asinh_across_modules","q_spearman_rint_across_modules"]
    v=validation[v_columns].rename(columns={c:"validation_"+c for c in v_columns if c not in keys})
    result=v.merge(original,on=keys,how="left",validate="one_to_one")
    result["original_comparison_available"]=result.original_n.notna()
    result["unavailable_reason"]=np.where(result.original_n.notna(),"","Original diagnosis-stratified table has no matching pooled/unclassified row")
    result["same_sign_spearman"]=np.where(result.original_rho.notna()&result.validation_rho_raw.notna(),
        np.sign(result.original_rho)==np.sign(result.validation_rho_raw),pd.NA)
    atomic_parquet(result,bundle/"concordance"/rule/f"{normalization}.parquet")


def install_hot_cache(root):
    source=Path(root)/"public_bundle"; target=APP/"data/control_derived/camel_validation"
    rollback=Path(root)/"rollback"/str(time.time_ns())
    # Aggregate/default summaries only. Donor rows and frozen maps stay on Drive.
    paths=["manifest.json","cohort_counts.tsv","release_manifest.json","mdc/resolved.parquet","sample_metadata.parquet","scales.parquet"]
    paths.extend(str(p.relative_to(source)) for p in (source/"concordance").glob("all/*.parquet"))
    paths.extend(str(p.relative_to(source)) for p in (source/"statistics").glob("all/*.parquet"))
    for relative in paths:
        p=source/relative
        if not p.exists(): continue
        destination=target/relative
        if destination.exists():
            backup=rollback/relative; backup.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(destination,backup)
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix(destination.suffix+".partial"); shutil.copy2(p,temporary); temporary.replace(destination)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=RUN)
    p.add_argument("--install-hot-cache",action="store_true")
    p.add_argument("--concordance",action="store_true")
    args=p.parse_args()
    if args.concordance:
        m=json.loads((args.root/"public_bundle/manifest.json").read_text())
        for rule in m.get("completed_rules",[]):
            for norm in ("standard_pruned","retained_edge"): concordance(args.root,rule,norm)
    # A small scalar-only cache avoids downloading RINT curves for asinh plots.
    scale_columns=["module","component","metric_family","differential_edge_rule","score_normalization","asinh_scale"]
    scales=pd.concat([pd.read_parquet(p,columns=scale_columns) for p in sorted((args.root/"public_bundle/rint").glob('M*.parquet'))],ignore_index=True)
    atomic_parquet(scales,args.root/"public_bundle/scales.parquet")
    report=validate(args.root)
    if args.install_hot_cache: install_hot_cache(args.root)
    print(json.dumps({k:report[k] for k in ("status","score_rows","samples","modules","completed_rules","mdc_status")}))


if __name__=="__main__": main()
