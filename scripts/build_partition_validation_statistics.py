#!/usr/bin/env python3
"""Saved-score robustness and exact-context, module-only BH for validation.

Original associations use the *saved original score columns*, not rounded
recomputations or averages of diagnosis coefficients. No API writes or uploads.
"""
from __future__ import annotations
import os
for _name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_name]="1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
APP=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(APP),str(APP/"scripts")]
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from app_helpers import camel_validation as V, data as D
from app_helpers.correlations import benjamini_hochberg
from app_helpers.validation_datasets import dataset
from app_helpers.validation_review import MATCH_KEYS
from app_helpers.validation_statistics import robust_rows
from run_camel_validation import atomic_json, atomic_parquet, sha
from run_partition_validation import RUN

FAMILY=["metric_family","component","phenotype","diagnosis_group"]
P_COLUMNS=["p_raw","p_asinh","p_rint","p_winsor","p_spearman_raw","p_spearman_asinh","p_spearman_rint"]
VERSION=2


def write_statistics(frame,path):
    """Small predicate-readable row groups, not one 50-MiB all-outcome group."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".partial")
    schema=pa.Schema.from_pandas(frame,preserve_index=False)
    with pq.ParquetWriter(temporary,schema,compression="zstd") as writer:
        for _,part in frame.groupby(["metric_family","phenotype"],sort=True,observed=True):
            writer.write_table(pa.Table.from_pandas(part,schema=schema,preserve_index=False))
    temporary.replace(path)


def correct_statistics(frame, modules):
    if set(frame.module.astype(int))!=set(map(int,modules)) or frame.duplicated(["module",*FAMILY]).any():
        raise ValueError("BH requires the complete, unique module family")
    for field in ("network_method","differential_edge_rule","score_normalization","cohort_scope"):
        if field in frame and frame[field].nunique(dropna=False)!=1:
            raise ValueError("BH input mixes fixed analysis identities")
    result=frame.copy(); grouped=result.groupby(FAMILY,dropna=False,sort=False)
    for p in P_COLUMNS:
        q=p.replace("p_","q_",1)+"_across_modules"
        result[q]=grouped[p].transform(benjamini_hochberg)
        result[q+"_n"]=grouped[p].transform("count").astype(int)
    result["q_spearman_across_modules"]=result.q_spearman_raw_across_modules
    return result


def original_source(config,method,rule,norm):
    path=D.differential_estimator_path("resolved_plot_data.parquet",config.module_set,"lioness",method,
        "all",rule,"global",.05,norm)
    files=[path] if path.is_file() else sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Saved original scores must be local; no automatic Drive download: {path}")
    return path,files


def evaluate(scores,metadata,config,method,rule,norm,scope,convention):
    outcomes=[o for o in D.NUMERIC_OUTCOMES if o in metadata]
    scores=scores.drop(columns=[c for c in metadata if c!="sample_id" and c in scores])
    data=scores.merge(metadata,on="sample_id",how="left",validate="many_to_one")
    if data.diagnosis_group.isna().any(): raise ValueError("Score-to-metadata donor join failed")
    rows=[]
    for (family,component),block in data.groupby(["metric_family","component"],sort=False,observed=True):
        subsets=[("All donors",block),*list(block.groupby("diagnosis_group",sort=False,observed=True))]
        for diagnosis,group in subsets:
            base=dict(module=int(block.module.iloc[0]),metric_family=family,component=component,
                component_label=config.labels[component],diagnosis_group=diagnosis,lioness_method=method,
                estimator="lioness",network_method=method,differential_edge_rule=rule,
                score_normalization=norm,cohort_scope=scope,node_convention=convention,
                cross_filter_node_convention_warning=bool(block.get("cross_filter_node_convention_warning",pd.Series([False])).any()))
            for field in ("n_features","n_source_assignment_rows","n_possible_edges","n_retained_edges"):
                if field in block: base[field]=int(block[field].iloc[0])
            rows.extend(robust_rows(base,group,outcomes))
    return pd.DataFrame(rows)


def module_statistics(module,root,key,method,rule,norm,context):
    root=Path(root); config=dataset(key,method)
    part=root/f"statistics_parts/{rule}/{norm}/M{module}.parquet"
    original_part=root/f"original_statistics_parts/{rule}/{norm}/M{module}.parquet"
    stamp=part.with_suffix(".json")
    sources=[root/f"public_bundle/modules/M{module}.parquet",root/f"public_bundle/rint/M{module}.parquet",
        root/"public_bundle/sample_metadata.parquet",root/"private/original_metadata.parquet"]
    fingerprint=hashlib.sha256((str(VERSION)+context+"".join(sha(p) for p in sources)).encode()).hexdigest()
    if stamp.exists() and part.exists() and original_part.exists() and json.loads(stamp.read_text()).get("fingerprint")==fingerprint:
        return str(part),str(original_part)
    started=time.monotonic(); os.environ["PARTITION_VALIDATION_DATA_ROOT"]=str(root/"public_bundle")
    source=V.for_dataset(key,method)
    scores=source.scores(module,None,rule,norm,embedded=False)
    validation=evaluate(scores,source.metadata(),config,method,rule,norm,"incomplete_region_validation",
        "legacy_unfiltered_unique_tissue_gene" if rule=="all" else "legacy_effect_positional_assignment_rows")
    path,_=original_source(config,method,rule,norm)
    columns=["sample_id","module","metric_family","component","metric_raw","metric_asinh","metric_rint"]
    saved=pd.read_parquet(path,filters=[("module","=",module),("lioness_method","=",method)],columns=columns)
    # Public historical scores retain the exact transformed values but not the
    # asinh fitting scale. Do not invent one or refit those stored values.
    saved["metric_scale"]=np.nan
    if saved.duplicated(["sample_id","module","metric_family","component"]).any():
        raise ValueError("Saved original scores have duplicate donor/module/feature/component rows")
    saved["component"]=saved.component.astype(str).str.replace("MFBA9BA46","DLPFC",regex=False)
    if set(saved.component)-set(config.components): raise ValueError("Unexpected saved original component")
    saved["cross_filter_node_convention_warning"]=module in source.manifest().get("cross_filter_node_convention_modules",[])
    original=evaluate(saved,pd.read_parquet(root/"private/original_metadata.parquet"),config,method,rule,norm,
        "original_comparison","saved_original_score_columns")
    atomic_parquet(validation,part); atomic_parquet(original,original_part)
    atomic_json(dict(fingerprint=fingerprint,module=module,validation_rows=len(validation),original_rows=len(original),
        original_score_source=str(path.relative_to(APP)),elapsed_seconds=time.monotonic()-started),stamp)
    return str(part),str(original_part)


def comparison(original,validation):
    originals=[*MATCH_KEYS,"n","r_raw","p_raw","r_asinh","p_asinh","r_rint","p_rint","rho","p_spearman"]
    o=original[originals].rename(columns={c:"original_"+c for c in originals if c not in MATCH_KEYS})
    for scale in ("raw","asinh","rint","spearman"):
        suffix="spearman_raw" if scale=="spearman" else scale
        o["original_q_"+scale]=original["q_"+suffix+"_across_modules"].to_numpy()
        o["original_family_n_"+scale]=original["q_"+suffix+"_across_modules_n"].to_numpy()
    cols=[*MATCH_KEYS,*[c for c in validation if c not in MATCH_KEYS and (
        c in ["n","unavailable_reason"] or c.startswith(("r_","rho_","p_","q_")))]]
    v=validation[cols].rename(columns={c:"validation_"+c for c in cols if c not in MATCH_KEYS})
    result=v.merge(o,on=MATCH_KEYS,how="left",validate="one_to_one")
    for scale in ("raw","asinh","rint","spearman_raw","spearman_asinh","spearman_rint"):
        result["validation_family_n_"+scale]=result["validation_q_"+scale+"_across_modules_n"]
    result["original_comparison_available"]=result.original_n.notna()
    result["unavailable_reason"]=np.where(result.original_n.notna(),"","Original cohort has no matching diagnosis level")
    result["same_sign_spearman"]=np.where(result.original_rho.notna()&result.validation_rho_raw.notna(),np.sign(result.original_rho)==np.sign(result.validation_rho_raw),pd.NA)
    result["cross_filter_node_convention_warning"]=validation.cross_filter_node_convention_warning.to_numpy()
    result["node_convention"]=validation.node_convention.to_numpy()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=RUN)
    p.add_argument("--module-set",choices=("full_cohort","muscle_control_l4","muscle_all_donors_l4"),required=True)
    p.add_argument("--method",choices=("standard","control_anchored"),default="control_anchored")
    p.add_argument("--rules",nargs="+",default=["first_release"])
    p.add_argument("--workers",type=int,choices=(1,2),default=2)
    args=p.parse_args(); config=dataset(args.module_set,args.method)
    root=args.root/args.module_set/args.method
    os.environ["PARTITION_VALIDATION_DATA_ROOT"]=str(root/"public_bundle")
    manifest=V.manifest(args.module_set,args.method)
    if manifest.get("status")!="complete" or not (root/"pilot_passed.json").exists(): raise ValueError("Complete scoring and its pilot first")
    modules=sorted(manifest["completed_module_ids"])
    rules=manifest["rules"] if args.rules==["all_masks"] else (
        [r for r in manifest["rules"] if r=="all" or "__hedges_g__fixed_0.40__" in r] if args.rules==["first_release"] else args.rules)
    for rule in rules:
        if rule not in manifest["rules"]: raise ValueError("Unknown frozen mask")
        for norm in ("standard_pruned","retained_edge"):
            _,files=original_source(config,args.method,rule,norm)
            context=hashlib.sha256(json.dumps({str(p.relative_to(APP)):sha(p) for p in files},sort_keys=True).encode()).hexdigest()
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                jobs=[pool.submit(module_statistics,m,root,args.module_set,args.method,rule,norm,context) for m in modules]
                parts=[]
                for future in as_completed(jobs):
                    parts.append(future.result())
                    if len(parts)%20==0 or len(parts)==len(modules): print(json.dumps(dict(rule=rule,norm=norm,modules=len(parts),expected=len(modules))),flush=True)
            validation=correct_statistics(pd.concat([pd.read_parquet(p[0]) for p in parts],ignore_index=True),modules)
            original=correct_statistics(pd.concat([pd.read_parquet(p[1]) for p in parts],ignore_index=True),modules)
            for directory,frame in [("statistics",validation),("original_statistics",original),("concordance",comparison(original,validation))]:
                destination=root/"public_bundle"/directory/rule/f"{norm}.parquet"
                write_statistics(frame,destination)
                if destination.stat().st_size>=95*1024**2: raise ValueError("Shard exceeds 95 MiB; do not publish")
        # A rule is exposed only after both normalizations have full BH families.
        path=root/"public_bundle/manifest.json"; latest=json.loads(path.read_text())
        latest["completed_rules"]=list(dict.fromkeys([*latest.get("completed_rules",[]),rule]))
        latest["statistics_status"]="complete" if set(latest["completed_rules"])==set(latest["rules"]) else "running"
        latest["statistics_updated_utc"]=datetime.now(timezone.utc).isoformat()
        atomic_json(latest,path)


if __name__=="__main__": main()
