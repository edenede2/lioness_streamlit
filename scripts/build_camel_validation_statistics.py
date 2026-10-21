#!/usr/bin/env python3
"""Offline, resumable robustness tables; module-only BH, no network recomputation."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
APP=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(APP))
from app_helpers import camel_validation as V
from app_helpers.camel import FEATURES
from app_helpers.data import NUMERIC_OUTCOMES
from app_helpers.correlations import benjamini_hochberg
from app_helpers.score_robustness import (_fast_corr,_fast_spearman,_leave_one_out,
    _winsorize_asinh,_robust_skew,_fast_slope)
from run_camel_validation import RUN, atomic_json, atomic_parquet, sha, update_public_manifest


def robust_row(base, frame, outcome):
    y=pd.to_numeric(frame[outcome],errors="coerce").to_numpy(float)
    raw=frame.metric_raw.to_numpy(float); finite=np.isfinite(raw)&np.isfinite(y)
    y=y[finite]
    row={**base,"phenotype":outcome,"n":len(y),"eligible":False,"unavailable_reason":"",
         "scale":float(frame.metric_scale.iloc[0]),"analysis_subset":"all_donors"}
    row["metric_zero_variance"]=np.unique(raw[finite]).size<2
    if len(y)<10: row["unavailable_reason"]="n < 10"
    elif row["metric_zero_variance"]: row["unavailable_reason"]="constant network score"
    elif np.unique(y).size<2: row["unavailable_reason"]="constant outcome"
    else: row["eligible"]=True
    for scale in ("raw","asinh","rint"):
        x=frame[f"metric_{scale}"].to_numpy(float)[finite]
        valid=row["eligible"] and np.isfinite(x).sum()>=10
        _,r,p=_fast_corr(x,y) if valid else (0,np.nan,np.nan)
        _,rho,rp=_fast_spearman(x,y) if valid else (0,np.nan,np.nan)
        row.update({f"r_{scale}":r,f"p_{scale}":p,f"rho_{scale}":rho,
                    f"p_spearman_{scale}":rp,f"skew_{scale}":_robust_skew(x)})
        if scale in ("asinh","rint"):
            loo=_leave_one_out(x,y) if valid else (np.nan,)*3
            row.update(dict(zip([f"loo_{scale}_max_delta",f"loo_{scale}_min_abs",f"loo_{scale}_sign_flip"],loo)))
    _,row["r_winsor"],row["p_winsor"]=_fast_corr(_winsorize_asinh(frame.metric_asinh.to_numpy(float)[finite]),y) if row["eligible"] else (0,np.nan,np.nan)
    row["rho"]=row["rho_raw"]; row["p_spearman"]=row["p_spearman_raw"]
    line=_fast_slope(frame.metric_rint.to_numpy(float)[finite],y) if row["eligible"] else (np.nan,)*4
    row.update(dict(zip(["intercept_rint","slope_rint","x_min_rint","x_max_rint"],line)))
    return row


def module_statistics(module, rule, normalization, output):
    part=Path(output)/f"statistics_parts/{rule}/{normalization}/M{module}.parquet"
    qa=part.with_suffix(".json")
    sources=[V.root()/f"modules/M{module}.parquet",V.root()/f"rint/M{module}.parquet",V.root()/"sample_metadata.parquet"]
    fingerprint="|".join(sha(p) for p in sources)
    if part.exists() and qa.exists() and json.loads(qa.read_text()).get("source_fingerprint")==fingerprint:
        return str(part)
    reused=None; recalculate=None
    if normalization=="retained_edge":
        previous=Path(output)/f"statistics_parts/{rule}/standard_pruned/M{module}.parquet"
        previous_qa=previous.with_suffix(".json")
        if previous.exists() and previous_qa.exists() and json.loads(previous_qa.read_text()).get("source_fingerprint")==fingerprint:
            # Frozen masks have the same retained-edge count for every donor.
            # Positive constant rescaling preserves correlations, RINT, skew and
            # LOO diagnostics. Reuse only when the *actual* frozen asinh scales
            # also prove this invariance; otherwise recalculate that block.
            reused=pd.read_parquet(previous)
            maps=pd.read_parquet(V.root()/f"rint/M{module}.parquet",filters=[("differential_edge_rule","=",rule)],
                columns=["component","metric_family","score_normalization","asinh_scale"])
            counts=V.summaries(module,rule).groupby("component")[["n_features","n_retained_edges","n_possible_edges"]].first()
            recalculate=set()
            for (component,family),block in maps.groupby(["component","metric_family"]):
                scales=block.set_index("score_normalization").asinh_scale
                c=counts.loc[component]; retained=int(c.n_retained_edges)
                mask=reused.component.eq(component)&reused.metric_family.eq(family)
                if retained==0:
                    reused.loc[mask,"n"]=0
                    reused.loc[mask,"unavailable_reason"]="no retained edges" if c.n_possible_edges else "structurally unavailable component"
                else:
                    factor=1. if family in ("positive_density","negative_density") else (
                        c.n_features/(2*retained) if family=="connectivity" else 1/retained)
                    if not np.isclose(scales.retained_edge,scales.standard_pruned*factor,rtol=1e-12,atol=0):
                        recalculate.add((family,component))
                reused.loc[mask,"scale"]=scales.retained_edge
            reused["score_normalization"]="retained_edge"
            reused["normalization_statistics_reused"]=True
            if not recalculate:
                atomic_parquet(reused,part)
                atomic_json(dict(source_fingerprint=fingerprint,rows=len(reused),constant_scaling_reuse=True),qa)
                return str(part)
    data=V.scores(module,None,rule,normalization)
    outcomes=[o for o in NUMERIC_OUTCOMES if o in data]
    rows=[]
    for (family,component),block in data.groupby(["metric_family","component"],sort=False):
        if recalculate is not None and (family,component) not in recalculate:
            continue
        subsets=[("All donors",block),*list(block.groupby("diagnosis_group",sort=False))]
        for diagnosis,group in subsets:
            base=dict(module=module,metric_family=family,component=component,
                component_label=block.component_label.iloc[0],diagnosis_group=diagnosis,
                lioness_method="control_anchored",estimator="lioness",network_method="control_anchored",
                differential_edge_rule=rule,score_normalization=normalization,cohort_scope="incomplete_region_validation")
            for outcome in outcomes: rows.append(robust_row(base,group,outcome))
    result=pd.DataFrame(rows)
    result["normalization_statistics_reused"]=False
    if reused is not None:
        keep=[(r.metric_family,r.component) not in recalculate for r in reused.itertuples()]
        result=pd.concat([reused.loc[keep],result],ignore_index=True)
    atomic_parquet(result,part)
    atomic_json(dict(source_fingerprint=fingerprint,rows=len(rows)),qa)
    return str(part)


def finish(parts, destination):
    frame=pd.concat([pd.read_parquet(p) for p in parts],ignore_index=True)
    if frame.module.nunique()!=186: raise ValueError("Do not publish partial-module BH families")
    keys=["metric_family","component","phenotype","diagnosis_group"]
    if frame.duplicated(["module",*keys]).any(): raise ValueError("Duplicated module tests")
    g=frame.groupby(keys,dropna=False,sort=False)
    for p in ["p_raw","p_asinh","p_rint","p_winsor","p_spearman_raw","p_spearman_asinh","p_spearman_rint"]:
        q=p.replace("p_","q_",1)+"_across_modules"
        frame[q]=g[p].transform(benjamini_hochberg)
        frame[q+"_n"]=g[p].transform("count").astype(int)
    frame["q_spearman_across_modules"]=frame.q_spearman_raw_across_modules
    atomic_parquet(frame.sort_values(["module",*keys]),destination)
    if destination.stat().st_size>=95*1024**2:
        raise ValueError("Statistics shard exceeds the public file-size gate")
    return len(frame)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=RUN)
    p.add_argument("--rules",nargs="+",default=["all"])
    p.add_argument("--workers",type=int,choices=(1,2),default=2)
    args=p.parse_args(); os.environ["CAMEL_VALIDATION_DATA_ROOT"]=str(args.root/"public_bundle")
    manifest=V.manifest()
    if manifest.get("status")!="complete": raise ValueError("Finish scoring and all module QA first")
    modules=sorted(manifest["completed_module_ids"])
    if len(modules)!=186: raise ValueError("Expected exactly 186 modules")
    rules=manifest["rules"] if args.rules==["all_masks"] else args.rules
    for rule in rules:
        if rule not in manifest["rules"]: raise ValueError("Unknown frozen effect mask")
        for normalization in ("standard_pruned","retained_edge"):
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                jobs=[pool.submit(module_statistics,int(m),rule,normalization,args.root) for m in modules]
                parts=[f.result() for f in as_completed(jobs)]
            path=args.root/"public_bundle/statistics"/rule/f"{normalization}.parquet"
            n=finish(parts,path)
            print(json.dumps(dict(rule=rule,normalization=normalization,rows=n,status="complete")),flush=True)
        # Atomic status marks only fully reconciled masks selectable.
        update_public_manifest(args.root,completed_rule=rule)


if __name__=="__main__": main()
