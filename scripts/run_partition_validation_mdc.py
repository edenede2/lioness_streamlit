#!/usr/bin/env python3
"""Resolved brain MDC with frozen masks and partition-specific checkpoints.

Reuse the independently validated CAMEL MDC kernel in an isolated analysis
process. No original file is changed. Motor M5 is deliberately not routed to
this signedAlt engine.
"""
from __future__ import annotations
import os
for _name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_name]="1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import resource
import sys
APP=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(APP),str(APP/"scripts")]
import numpy as np
import pandas as pd
from app_helpers.correlations import benjamini_hochberg
from app_helpers.validation_datasets import dataset
import run_partition_validation as P
import run_camel_validation_mdc as M
from run_camel_validation import atomic_json, atomic_parquet, sha


def bind(state):
    if state["config"].motor:
        raise ValueError("Motor MDC requires the no-q90 M5/unpowered engine, not brain signedAlt")
    M.COMPONENT_TISSUES=state["config"].components
    M.COMPONENT_LABELS=state["config"].labels
    M.load_inputs=lambda:state
    M.module_source=lambda module:P.module_source(module,state)
    M._COMPONENT_CACHE.clear()


def run_module(module,component,root):
    context="".join(sha(p) for p in (Path(root)/"audit.json",Path(__file__),Path(P.__file__),Path(M.__file__)))
    path=M.mdc_job(module,component,root,200,source_context=context)
    # The shared kernel counts positional edges. Unfiltered legacy unique-node
    # analyses have a different possible-edge universe in duplicate modules.
    frame=pd.read_parquet(path)
    frame.loc[frame.differential_edge_rule.eq("all"),"n_possible_edges"]=frame.loc[
        frame.differential_edge_rule.eq("all"),"n_retained_edges"].to_numpy()
    affected=module in P._STATE["assignment"].loc[
        P._STATE["assignment"].duplicated(["Tissue","Gene Symbol"],keep=False),"module"].values
    frame["cross_filter_node_convention_warning"]=affected
    frame["node_convention"]=np.where(frame.differential_edge_rule.eq("all"),
        "legacy_unfiltered_unique_tissue_gene","legacy_effect_positional_assignment_rows")
    frame["module_set"]=P._STATE["config"].module_set
    frame["network_method"]=P._STATE["method"]
    a=P._STATE["assignment"].loc[P._STATE["assignment"].module.eq(module)]
    nodes=a.loc[a.public_tissue.isin(P._STATE["config"].components[component])]
    frame["n_features"]=np.where(frame.differential_edge_rule.eq("all"),a.feature_id.nunique(),len(a))
    frame["n_source_assignment_rows"]=len(a)
    frame["n_component_nodes"]=np.where(frame.differential_edge_rule.eq("all"),nodes.feature_id.nunique(),len(nodes))
    empty_all=frame.differential_edge_rule.eq("all")&frame.n_possible_edges.eq(0)
    frame.loc[empty_all,"unavailable_reason"]="structurally unavailable component"
    atomic_parquet(frame,Path(path))
    adjacency=Path(root)/f"public_bundle/mdc/adjacency/{component}/M{module}.npz"
    with np.load(adjacency,allow_pickle=False) as archive:
        arrays={name:archive[name] for name in archive.files if name!="canonical_nodes"}
    M.atomic_npz(adjacency,**arrays,canonical_nodes=~nodes.duplicated(["Tissue","Gene Symbol"]).to_numpy())
    qa_path=Path(path).with_suffix(".json")
    qa=json.loads(qa_path.read_text())
    qa["peak_rss_mib"]=max(qa.get("peak_rss_mib",0),resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    qa["node_convention_policy"]="source_specific_legacy"
    atomic_json(qa,qa_path)
    return path


def reconcile(root,state):
    root=Path(root); modules=set(map(int,state["assignment"].module.unique()))
    parts=sorted((root/"mdc_parts").glob("*/M*.parquet"))
    if not parts: return
    frame=pd.concat([pd.read_parquet(p) for p in parts],ignore_index=True)
    complete=[c for c,g in frame.groupby("component",observed=True) if set(g.module)==modules]
    frame=frame.loc[frame.component.isin(complete)].copy()
    if frame.empty: return
    keys=["component","differential_edge_rule"]
    if frame.duplicated(["module",*keys]).any(): raise ValueError("Duplicate MDC module/context")
    grouped=frame.groupby(keys,sort=False,dropna=False)
    for direction in ("loss","gain"):
        for null in ("sample","gene"):
            p=f"p_{direction}_{null}"; q=f"q_{direction}_{null}"
            frame[q]=grouped[p].transform(benjamini_hochberg)
            frame[q+"_family_n"]=grouped[p].transform("count").astype(int)
    frame["q_loss"]=np.maximum(frame.q_loss_sample,frame.q_loss_gene)
    frame["q_gain"]=np.maximum(frame.q_gain_sample,frame.q_gain_gene)
    frame["directional_fdr"]=np.where(frame.mdc.ge(1),frame.q_loss,frame.q_gain)
    sparse=frame.component.eq("CT_AC__PCGBA23")
    if frame.loc[sparse,"inferential"].any() or frame.loc[sparse,"directional_fdr"].notna().any():
        raise ValueError("Sparse AC–PCG MDC must remain descriptive only")
    atomic_parquet(frame,root/"public_bundle/mdc/resolved.parquet")
    path=root/"public_bundle/manifest.json"; manifest=json.loads(path.read_text())
    manifest.update(mdc_components_complete=complete,mdc_status="complete" if len(complete)==6 else "running",
        mdc_sample_permutations=200,mdc_gene_permutations=200,mdc_seed=42,
        mdc_nulls={"sample":"Independent gene-wise donor shuffles within AD and Control",
                  "gene":"Tissue-count-preserving random gene sets; frozen edge masks"})
    atomic_json(manifest,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=P.RUN)
    p.add_argument("--module-set",choices=("full_cohort",),required=True)
    p.add_argument("--method",choices=("standard","control_anchored"),default="control_anchored")
    p.add_argument("--stage",choices=("pilot","run"),required=True)
    p.add_argument("--workers",type=int,choices=(1,2),default=2)
    args=p.parse_args(); root=args.root/args.module_set/args.method
    manifest=json.loads((root/"public_bundle/manifest.json").read_text())
    if manifest.get("status")!="complete" or not (root/"pilot_passed.json").exists():
        raise ValueError("Scoring and its pilot must complete first")
    if args.stage=="run" and not (root/"mdc_pilot_passed.json").exists():
        raise ValueError("MDC pilot must pass before expansion")
    state=P.load_inputs(args.module_set,args.method); bind(state)
    modules=[188,589] if args.stage=="pilot" else sorted(map(int,state["assignment"].module.unique()))
    components=["TS_DLPFC"] if args.stage=="pilot" else list(state["config"].components)
    for component in components:
        M.component_background(component)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs={pool.submit(run_module,m,component,root):m for m in modules}
            for future in as_completed(jobs):
                path=future.result()
                print(json.dumps(dict(component=component,module=jobs[future],status="complete",sha256=sha(path))),flush=True)
        M._COMPONENT_CACHE.clear(); reconcile(root,state)
    if args.stage=="pilot":
        checks=[json.loads((root/f"mdc_parts/{components[0]}/M{m}.json").read_text()) for m in modules]
        if max(q["peak_rss_mib"] for q in checks)*args.workers>32768:
            raise ValueError("MDC pilot exceeds the 32-GiB combined worker-memory gate")
        atomic_json(dict(passed=True,modules=modules,component=components[0],sample_permutations=200,
            gene_permutations=200,node_convention_policy="source_specific_legacy",workers=args.workers,
            results=checks),root/"mdc_pilot_passed.json")


if __name__=="__main__": main()
