#!/usr/bin/env python3
"""Independent release gates for frozen-reference validation; no upload."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
APP=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(APP),str(APP/"scripts")]
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import pearsonr,spearmanr
from app_helpers.camel import feature_values
from app_helpers.validation_datasets import dataset
from run_partition_validation import RUN
from run_camel_validation import atomic_json,atomic_parquet,sha


def independent_bh(values):
    p=np.asarray(values,float); q=np.full(len(p),np.nan); good=np.flatnonzero(np.isfinite(p))
    order=good[np.argsort(p[good],kind="stable")]
    if len(order): q[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1])
    return q


def validate_mdc(frame,modules,components,rules,cohort_counts):
    keys=["component","differential_edge_rule"]
    if frame.duplicated(["module",*keys]).any(): raise ValueError("Duplicate MDC row")
    if len(frame)!=len(modules)*len(components)*len(rules): raise ValueError("Incomplete MDC rows")
    if set(frame.module)!=set(modules) or set(frame.component)!=set(components) or set(frame.differential_edge_rule)!=set(rules):
        raise ValueError("MDC context mismatch")
    for component,part in frame.groupby("component",observed=True):
        counts=cohort_counts[component]
        if not part.n_ad.eq(counts["AD"]).all() or not part.n_control.eq(counts["Control"]).all():
            raise ValueError("MDC group counts mismatch")
    np.testing.assert_allclose(frame.mdc,np.divide(frame.mean_abs_ad,frame.mean_abs_control,
        out=np.full(len(frame),np.nan),where=frame.mean_abs_control.to_numpy()>1e-10),equal_nan=True)
    for direction in ("loss","gain"):
        for null in ("sample","gene"):
            p=f"p_{direction}_{null}";q=f"q_{direction}_{null}"
            expected=frame.groupby(keys,dropna=False)[p].transform(lambda v:independent_bh(v.to_numpy()))
            np.testing.assert_allclose(frame[q],expected,atol=1e-12,equal_nan=True)
            np.testing.assert_array_equal(frame[q+"_family_n"],frame.groupby(keys,dropna=False)[p].transform("count"))
        np.testing.assert_allclose(frame["q_"+direction],np.maximum(frame[f"q_{direction}_sample"],frame[f"q_{direction}_gene"]),equal_nan=True)
    np.testing.assert_allclose(frame.directional_fdr,np.where(frame.mdc.ge(1),frame.q_loss,frame.q_gain),equal_nan=True)
    unavailable=(~frame.inferential)|frame.n_retained_edges.eq(0)
    p_columns=[f"p_{d}_{n}" for d in ("loss","gain") for n in ("sample","gene")]
    if frame.loc[unavailable,p_columns].notna().any().any(): raise ValueError("Unavailable MDC must not have inferential p-values")
    for null in ("sample","gene"):
        draws=frame[f"{null}_null_valid_draws"]
        if not draws.between(0,200).all(): raise ValueError("MDC null draw counts outside prespecified range")
        expected=np.divide(1.,1+draws,out=np.full(len(frame),np.nan),where=draws.to_numpy()>0)
        np.testing.assert_allclose(frame[f"{null}_p_resolution"],expected,equal_nan=True)
    return len(frame)


def validate(root,release=False):
    root=Path(root); bundle=root/"public_bundle"
    m=json.loads((bundle/"manifest.json").read_text()); config=dataset(m["module_set"],m["network_method"])
    audit=json.loads((root/"audit.json").read_text())
    if m["status"]!="complete" or m["complete_modules"]!=config.modules: raise ValueError("Incomplete module catalog")
    membership=json.loads((root/"discovery_membership_audit.json").read_text())
    if not membership["passed"] or membership["validation_overlap"]: raise ValueError("Discovery donor overlap")
    for relative,expected in audit["source_hashes"].items():
        if sha(APP.parents[1]/relative)!=expected: raise ValueError("A frozen source changed")
    if sha(APP.parents[1]/membership["source"])!=membership["source_sha256"]: raise ValueError("Discovery source changed")
    metadata=pd.read_parquet(bundle/"sample_metadata.parquet"); allowed=set(metadata.sample_id)
    if config.module_set=="full_cohort":
        previous=pd.read_parquet(APP/"data/control_derived/camel_validation/sample_metadata.parquet")
        pd.testing.assert_frame_equal(metadata.sort_values("sample_id").reset_index(drop=True),
                                      previous.sort_values("sample_id").reset_index(drop=True))
    original=pd.read_parquet(root/"private/original_metadata.parquet")
    if len(allowed)!=config.validation_donors or allowed&set(original.sample_id): raise ValueError("Public sample grain/overlap failed")
    source_ids=set(m["completed_module_ids"]); total=0; map_count=0; formula_max=0.; reproduction_max=0.; scales=[]
    for module in sorted(source_ids):
        path=bundle/f"modules/M{module}.parquet"; maps=bundle/f"rint/M{module}.parquet"
        qa=json.loads((root/f"checkpoints/M{module}.json").read_text())
        if not qa["passed"] or sha(path)!=qa["score_sha256"] or sha(maps)!=qa["rint_sha256"]: raise ValueError("Module QA/hash failed")
        formula_max=max(formula_max,max(q["max_abs_edge_error"] for q in qa["direct_checks"]))
        reproduction_max=max(reproduction_max,qa["reproduction"]["max_relative_error"])
        frame=pd.read_parquet(path); total+=len(frame)
        if {"donor","projid"}&set(frame) or not set(frame.sample_id)<=allowed: raise ValueError("Private/unapproved identifier")
        if frame.duplicated(["sample_id","component","differential_edge_rule"]).any(): raise ValueError("Duplicate score row")
        if set(frame.component)!=set(config.components) or set(frame.differential_edge_rule)!=set(m["rules"]): raise ValueError("Incomplete component/mask set")
        np.testing.assert_allclose(frame.signed_weight_sum,frame.positive_weight_sum-frame.negative_weight_magnitude,equal_nan=True)
        np.testing.assert_allclose(frame.absolute_weight_sum,frame.positive_weight_sum+frame.negative_weight_magnitude,equal_nan=True)
        if not (frame.n_positive_edges+frame.n_negative_edges+frame.n_zero_edges==frame.n_retained_edges).all(): raise ValueError("Sign counts failed")
        if not (frame.n_retained_edges+frame.n_pruned_edges==frame.n_possible_edges).all(): raise ValueError("Possible/pruned counts failed")
        for rule in m["rules"]:
            part=frame.loc[frame.differential_edge_rule.eq(rule)]
            if part.groupby("component",observed=True).size().to_dict()!=config.expected_counts: raise ValueError("Component donor counts failed")
            if not rule.endswith("__either"): continue
            base=rule.rsplit("__",1)[0]
            d={label:frame.loc[frame.differential_edge_rule.eq(base+"__"+label)].set_index(["sample_id","component"]).sort_index() for label in ["ad_higher","control_higher","either"]}
            for field in ["n_retained_edges","n_positive_edges","n_negative_edges","positive_weight_sum","negative_weight_magnitude"]:
                np.testing.assert_allclose(d["ad_higher"][field]+d["control_higher"][field],d["either"][field],atol=1e-9,rtol=1e-10,equal_nan=True)
        anchors=pd.read_parquet(maps,columns=["module","component","metric_family","differential_edge_rule","score_normalization","asinh_scale"])
        scales.append(anchors); map_count+=len(anchors)
    if total!=config.expected_rows(): raise ValueError("Unexpected total score rows")
    atomic_parquet(pd.concat(scales,ignore_index=True),bundle/"scales.parquet")
    checked=0
    for rule in m.get("completed_rules",[]):
        for norm in ("standard_pruned","retained_edge"):
            for directory in ("statistics","original_statistics"):
                frame=pd.read_parquet(bundle/directory/rule/f"{norm}.parquet")
                if set(frame.module)!=source_ids: raise ValueError("Incomplete BH family")
                keys=["metric_family","component","phenotype","diagnosis_group"]
                if frame.duplicated(["module",*keys]).any(): raise ValueError("Duplicate test row")
                for p in ("p_raw","p_asinh","p_rint","p_spearman_raw","p_spearman_asinh","p_spearman_rint"):
                    q=p.replace("p_","q_",1)+"_across_modules"
                    expected=frame.groupby(keys,dropna=False)[p].transform(lambda v:independent_bh(v.to_numpy()))
                    np.testing.assert_allclose(expected,frame[q],atol=1e-12,equal_nan=True)
                    counts=frame.groupby(keys,dropna=False)[p].transform("count")
                    np.testing.assert_array_equal(counts,frame[q+"_n"])
                checked+=len(frame)
    mdc_rows=0
    if m.get("mdc_status")=="complete":
        counts={r["component"]:r for r in audit["component_counts"]}
        mdc_rows=validate_mdc(pd.read_parquet(bundle/"mdc/resolved.parquet"),source_ids,config.components,m["rules"],counts)
        parts=list((root/"mdc_parts").glob("*/M*.json"))
        if len(parts)!=config.modules*len(config.components): raise ValueError("MDC QA checkpoint coverage incomplete")
        if any(json.loads(p.read_text())["direct_adjacency_error"]>1e-10 for p in parts):
            raise ValueError("MDC direct adjacency gate failed")
    if release:
        first={r for r in m["rules"] if r=="all" or "__hedges_g__fixed_0.40__" in r}
        if not first<=set(m["completed_rules"]) or m["mdc_status"]!="complete":
            raise ValueError("First-release associations and validated MDC must finish before release")
    inventory={}
    for p in bundle.rglob("*"):
        if not p.is_file() or p.suffix in (".lock",".partial") or p.name=="release_manifest.json": continue
        if p.stat().st_size>=95*1024**2: raise ValueError("Public shard exceeds 95 MiB")
        entry=dict(bytes=p.stat().st_size,sha256=sha(p))
        if p.suffix==".parquet":
            if {"donor","projid","donor_private","projid_private"}&set(pq.read_schema(p).names): raise ValueError("Private identifier in public schema")
            entry["rows"]=pq.ParquetFile(p).metadata.num_rows
        inventory[str(p.relative_to(bundle))]=entry
    report=dict(status="passed" if release else "scores_passed",module_set=config.module_set,network_method=m["network_method"],
        score_rows=total,transform_rows=map_count,samples=len(allowed),modules=config.modules,
        completed_rules=m["completed_rules"],mdc_status=m["mdc_status"],statistics_rows_verified=checked,
        mdc_rows_verified=mdc_rows,
        max_direct_edge_error=formula_max,max_historical_relative_error=reproduction_max,
        files=inventory,backend_verified=False,publication_approved=False)
    approval_path=root/"publication_approval.json"
    if approval_path.exists():
        approval=json.loads(approval_path.read_text())
        report["publication_approved"]=bool(approval.get("approved")
            and approval.get("module_set")==config.module_set
            and m["network_method"] in approval.get("methods",[])
            and approval.get("score_rows")==total and approval.get("transformation_rows")==map_count
            and approval.get("validation_samples")==len(allowed))
    atomic_json(report,bundle/"release_manifest.json")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=RUN)
    p.add_argument("--module-set",choices=("full_cohort","muscle_control_l4","muscle_all_donors_l4"),required=True)
    p.add_argument("--method",choices=("standard","control_anchored"),default="control_anchored")
    p.add_argument("--release",action="store_true")
    a=p.parse_args();report=validate(a.root/a.module_set/a.method,a.release)
    print(json.dumps({k:v for k,v in report.items() if k!="files"}))


if __name__=="__main__":main()
