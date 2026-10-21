#!/usr/bin/env python3
"""Frozen-reference non-discovery validation, isolated by partition and method.

No upload, Git mutation, prediction, mask learning or module discovery. Pilot
and full runs checkpoint each module; existing CAMEL outputs are read-only.
"""
from __future__ import annotations

import os
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

APP = Path(__file__).resolve().parents[1]
REPO = APP.parents[1]
sys.path[:0] = [str(APP), str(REPO), str(APP/"scripts")]
import numpy as np
import pandas as pd

from app_helpers.camel import FEATURES, feature_values
from app_helpers.validation_datasets import dataset
from app_helpers.validation_scoring import score_component
from models.differential_edges import EffectMaskSpec, effect_mask_specs, effect_rule_key, effect_size_mask
from run_camel_validation import atomic_json, atomic_parquet, canon, sha

RUN = REPO / "out/partition_validation/20261021_frozen_references_legacy_conventions"
_STATE = None
VERSION = 2


def assignments_and_edges(key, method):
    if key == "full_cohort":
        from scripts.python.run_bonobo_app_rosmap import configs, load_assignment_table
        a = load_assignment_table(next(c for c in configs(REPO) if c.key == key))
        root = REPO / "out/lioness_app_expansion/20260822_ad_control_differential_edges_all_networks/full_cohort/checkpoints" / f"lioness__{method}"
        source = next(c.assignments for c in configs(REPO) if c.key == key)
        return a, root, source
    from run_muscle_lioness import read_assignments, DEFAULT_ASSIGNMENTS
    source = (DEFAULT_ASSIGNMENTS if key == "muscle_control_l4" else REPO /
        "results/se2_m5_muscle_all_genes/all_donors/20261012_all_donors_fitted_betas/modules_filtered_min30/M5_all_genes_table_3.csv")
    a = read_assignments(source, 30)
    # Match module_matrix's tissue-first positional order exactly.
    a = pd.concat([a.loc[a.Tissue.eq(t)] for t in ("VH", "SMA", "QUAD")], ignore_index=True)
    a["module"] = a["Cluster ID"].astype(int)
    a["feature_id"] = a.Tissue + "_" + a["Gene Symbol"]
    root = (REPO/"out/lioness_app_expansion/20261012_muscle_effect_size_edge_filtering" if key == "muscle_control_l4"
            else REPO/"out/muscle_all_donors_l4/20261013_nonprediction/effect_edges")
    return a, root/"checkpoints", source


def load_inputs(key, method):
    global _STATE
    if _STATE is not None:
        if (_STATE["config"].module_set, _STATE["method"]) != (key, method):
            raise ValueError("Use separate processes for each immutable validation identity")
        return _STATE
    config = dataset(key, method)
    if config.motor:
        raise ValueError("Motor input audit must pass before enabling motor scoring")
    import run_camel_validation as legacy
    from app_helpers.data import load_sample_metadata
    old = legacy.load_inputs()
    a, edge_root, assignment_path = assignments_and_edges(key, method)
    a["public_tissue"] = a.Tissue.replace({"MFBA9BA46": "DLPFC"})
    original_root = REPO / "out/lioness_reference_comparison_rosmap/20260817_standard_control_anchored_allmodules_5phenotypes_6features" / method / "data"
    original = pd.read_parquet(original_root/"phenotypes.parquet")
    original["donor"] = original.donor.map(canon)
    if set(original.donor) != set(old["original"].donor) or original.donor.duplicated().any():
        raise ValueError("Original method cohort differs from the frozen common cohort")
    if a.module.nunique() != config.modules or len(a) != config.assignments:
        raise ValueError("Assignment audit failed")
    original_meta = load_sample_metadata(key).copy()
    if set(original_meta.sample_id) != {old["mapping"][d] for d in original.donor}:
        raise ValueError("Original donor-to-public-ID mapping does not reproduce public metadata")
    sources = [assignment_path, original_root/"phenotypes.parquet", *legacy.RAW_PATHS.values()]
    _STATE = dict(config=config, method=method, assignment=a, edge_root=edge_root,
        assignment_path=assignment_path, original=original, original_meta=original_meta,
        original_root=original_root, matrix=old["matrix"], complete=old["complete"],
        incomplete=old["incomplete"], mapping=old["mapping"], metadata=old["metadata"].copy(),
        donors=old["donors"], controls=old["controls"], sources=sources,
        beta_ts={t:3. for t in legacy.RAW_PATHS},
        beta_ct={c:2. for c in config.components if c.startswith("CT_")})
    return _STATE


def audit(root, state):
    root = Path(root); c = state["config"]
    root.mkdir(parents=True, exist_ok=True)
    counts=[]
    meta=state["metadata"].set_index("sample_id")
    for component, donors in state["donors"].items():
        subset=meta.loc[[state["mapping"][d] for d in donors]]
        counts.append(dict(component=component, component_label=c.labels[component], n=len(donors),
            **{g:int(subset.diagnosis_group.eq(g).sum()) for g in ("Control","MCI","AD","Unclassified")}))
    if {r["component"]:r["n"] for r in counts} != c.expected_counts:
        raise ValueError("Component cohort audit failed")
    reference = state["original"].donor.tolist() if state["method"] == "standard" else state["controls"]
    record=dict(status="passed", module_set=c.module_set, network_method=state["method"],
        modules=c.modules, assignment_rows=len(state["assignment"]),
        repeated_assignment_rows=int(state["assignment"].duplicated(["Tissue","Gene Symbol"]).sum()),
        validation_donors=len(state["incomplete"]), excluded_complete_donors=len(state["complete"]),
        original_donors=len(state["original"]), reference_donors=len(reference), reference_controls=c.controls,
        reference_donor_hash=hashlib.sha256("\n".join(sorted(reference)).encode()).hexdigest(),
        score_framework=c.framework(state["method"]), component_counts=counts,
        source_hashes={str(p.relative_to(REPO)):sha(p) for p in state["sources"]},
        reference_policy=f"Frozen original {len(reference)} {'donors' if state['method']=='standard' else 'Controls'}; every validation donor add-one",
        beta_ts=state["beta_ts"], beta_ct=state["beta_ct"], predictions_unchanged=True)
    record["node_convention_policy"]="User-approved source-specific legacy conventions: unfiltered unique tissue-gene nodes; effect-filtered positional assignment rows"
    record["node_convention_approval"]="Explicit user selection, 2026-10-21"
    previous=root/"audit.json"
    if previous.exists() and json.loads(previous.read_text()) != record:
        raise ValueError("Frozen audit changed; do not mix checkpoints")
    atomic_json(record, previous)
    atomic_parquet(state["metadata"], root/"public_bundle/sample_metadata.parquet")
    atomic_parquet(state["original_meta"], root/"private/original_metadata.parquet")
    pd.DataFrame(counts).to_csv(root/"public_bundle/cohort_counts.tsv", sep="\t", index=False)
    return record


def audit_discovery_membership(root,state):
    """Check actual discovery donors, not just the later scoring cohort."""
    if state["config"].module_set!="full_cohort":
        raise ValueError("Register and verify the dataset's actual discovery artifact first")
    source=REPO/"data/proccessed/se2_filtered/se2_rosmap_full_signed_alt/se2_4_rosmap_full_signed_alt_global2_global2_ME_commonDonors_cross_tissue.tsv"
    ids=pd.read_csv(source,sep="\t",usecols=[0]).iloc[:,0].map(canon)
    if ids.duplicated().any() or set(ids)!=state["complete"] or set(ids)&state["incomplete"]:
        raise ValueError("Validation donors overlap actual full-cohort module discovery")
    report=dict(passed=True,discovery_donors=len(ids),validation_overlap=0,
        source=str(source.relative_to(REPO)),source_sha256=sha(source),excluded_all_discovery_donors=True)
    atomic_json(report,Path(root)/"discovery_membership_audit.json")
    return report


def matrix_for(a, donor_ids, state):
    """Positional selection preserves duplicate nodes exactly once per row."""
    result=np.empty((len(donor_ids),len(a)), dtype=float)
    tissues=a.public_tissue.to_numpy()
    for tissue in dict.fromkeys(tissues):
        columns=np.flatnonzero(tissues==tissue)
        result[:,columns]=state["matrix"][tissue].loc[donor_ids,a.iloc[columns]["Gene Symbol"].tolist()].to_numpy(float)
    return result


def module_source(module, state):
    a=state["assignment"].loc[state["assignment"].module.eq(module)].copy().reset_index(drop=True)
    original=state["original"].donor.tolist()
    full=matrix_for(a,original,state)
    # Historical row-mean, then original-cohort column-mean fallback, unchanged.
    frame=pd.DataFrame(full)
    full=frame.apply(lambda r:r.fillna(r.mean()),axis=1).fillna(frame.mean()).fillna(0.).to_numpy(float)
    rows, columns=(np.triu_indices(len(a),1) if state["config"].motor else np.tril_indices(len(a),-1))
    source=state["edge_root"]/f"M{module}__edge_statistics.parquet"
    edge=pd.read_parquet(source)
    if not np.array_equal(edge.row_index,rows) or not np.array_equal(edge.column_index,columns) or not np.array_equal(edge.edge_index,np.arange(len(rows))):
        raise ValueError("Frozen edge table does not match positional module assignments")
    canonical=~a.duplicated(["Tissue","Gene Symbol"]).to_numpy()
    # R's named gene-to-module map selects unique expression columns. Preserve
    # the Python effect-table multiplicity separately, without changing masks.
    masks={"all":canonical[rows]&canonical[columns]}
    specs=[EffectMaskSpec("hedges_g","fixed",.4)] if state["config"].motor else effect_mask_specs()
    for spec in specs:
        values=edge["discovery_"+spec.statistic].to_numpy(float)
        for direction in ("ad_higher","control_higher","either"):
            masks[effect_rule_key(spec,direction)]=effect_size_mask(values,edge.edge_index.to_numpy(),spec,direction=direction)
        if not np.array_equal(masks[effect_rule_key(spec,"ad_higher")]|masks[effect_rule_key(spec,"control_higher")],masks[effect_rule_key(spec,"either")]):
            raise ValueError("Frozen directional-mask union failed")
    return a,full,rows,columns,masks,source


def reproduce_original(module, summary, state):
    old=next(state["original_root"].glob("*tissue_resolved_component_data.parquet"))
    original=state["original"].groupby("diagnosis_group",sort=False).head(3)
    donors=original.donor.tolist(); ids=[state["mapping"][d] for d in donors]
    saved=pd.read_parquet(old,filters=[("module","=",module),("donor","in",donors)],
        columns=["donor","component","metric_family","metric_raw"])
    lookup=summary.loc[summary.differential_edge_rule.eq("all")&summary.sample_id.isin(ids)].copy()
    errors=[]
    for family in FEATURES:
        lookup[family]=feature_values(lookup,family,"standard_pruned")
    lookup=lookup.set_index(["sample_id","component"])
    for row in saved.itertuples(index=False):
        component=row.component.replace("MFBA9BA46","DLPFC")
        value=lookup.loc[(state["mapping"][canon(row.donor)],component),row.metric_family]
        if pd.isna(value) and pd.isna(row.metric_raw):
            continue
        if not np.isfinite(value) or not np.isfinite(row.metric_raw):
            raise ValueError("Historical structural availability differs")
        errors.append(abs(value-row.metric_raw)/max(1.,abs(row.metric_raw)))
    maximum=float(max(errors,default=0.))
    if maximum>1e-8 or not errors:
        raise ValueError(f"M{module}: historical score reproduction failed ({maximum})")
    return dict(module=int(module),tested_values=len(errors),donors=len(donors),max_relative_error=maximum)


def module_job(module, root, edge_batch=2048):
    state=_STATE
    if state is None: raise ValueError("Load immutable inputs before forking")
    root=Path(root); started=time.monotonic()
    a,full,ri,ci,masks,source=module_source(module,state)
    source_hash=sha(source)
    audit_hash=sha(root/"audit.json")
    qa_path=root/f"checkpoints/M{module}.json"
    output=root/f"public_bundle/modules/M{module}.parquet"
    maps_path=root/f"public_bundle/rint/M{module}.parquet"
    original_path=root/f"private/original/M{module}.parquet"
    if qa_path.exists() and all(p.exists() for p in (output,maps_path,original_path)):
        old=json.loads(qa_path.read_text())
        if old.get("passed") and old.get("source_edge_sha256")==source_hash and old.get("audit_sha256")==audit_hash and old.get("version")==VERSION:
            return old
    original=state["original"].donor.tolist()
    reference=original if state["method"]=="standard" else state["controls"]
    ref_indices=np.array([original.index(d) for d in reference])
    positions={d:i for i,d in enumerate(reference)}
    roles=np.array([positions.get(d,-1) for d in original])
    tissue=a.public_tissue.to_numpy(); orig_parts=[]; parts=[]; maps=[]; checks=[]
    for component,tissues in state["config"].components.items():
        nodes=np.flatnonzero(np.isin(tissue,tissues)); local={j:i for i,j in enumerate(nodes)}
        keep=(tissue[ri]==tissues[0])&(tissue[ci]==tissues[0]) if len(tissues)==1 else (
            np.isin(tissue[ri],tissues)&np.isin(tissue[ci],tissues)&(tissue[ri]!=tissue[ci]))
        lr=np.array([local[j] for j in ri[keep]],int); lc=np.array([local[j] for j in ci[keep]],int)
        new_ids=state["donors"][component]
        new=matrix_for(a.iloc[nodes],new_ids,state)
        ref=full[ref_indices][:,nodes]
        new=np.where(np.isfinite(new),new,ref.mean(axis=0))
        power=state["beta_ts"][tissues[0]] if len(tissues)==1 else state["beta_ct"][component]
        old,new,anchor,qa=score_component(reference=ref, original=full[:,nodes], validation=new,
            reference_positions=roles, rows=lr, columns=lc, masks={r:m[keep] for r,m in masks.items()},
            power=power,module=module,component=component,
            n_features={r:int(a.feature_id.nunique()) if r=="all" else len(a) for r in masks},
            possible_edges={r:int(masks["all"][keep].sum()) if r=="all" else int(keep.sum()) for r in masks},
            node_conventions={r:"legacy_unfiltered_unique_tissue_gene" if r=="all" else "legacy_effect_positional_assignment_rows" for r in masks},
            source_n_features=len(a),
            original_ids=[state["mapping"][d] for d in original], validation_ids=[state["mapping"][d] for d in new_ids], edge_batch=edge_batch)
        orig_parts.append(old); parts.append(new); maps.append(anchor)
        checks.extend([dict(component=component,**q) for q in qa])
    result=pd.concat(parts,ignore_index=True); original_summary=pd.concat(orig_parts,ignore_index=True)
    if len(result)!=sum(state["config"].counts)*state["config"].rule_count or result.duplicated(["sample_id","component","differential_edge_rule"]).any():
        raise ValueError("Donor/component/mask grain failed")
    reproduction=reproduce_original(module,original_summary,state)
    atomic_parquet(result,output); atomic_parquet(pd.concat(maps,ignore_index=True),maps_path)
    atomic_parquet(original_summary,original_path)
    qa=dict(module=int(module),passed=True,version=VERSION,rows=len(result),source_edge_sha256=source_hash,
        audit_sha256=audit_hash,score_sha256=sha(output),rint_sha256=sha(maps_path),
        direct_checks=checks,reproduction=reproduction,all_validation_donors_add_one=True,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        elapsed_seconds=time.monotonic()-started)
    atomic_json(qa,qa_path)
    return qa


def update_manifest(root, state):
    c=state["config"]; root=Path(root)
    complete=[]
    for module in sorted(state["assignment"].module.unique()):
        p=root/f"checkpoints/M{module}.json"
        if p.exists() and json.loads(p.read_text()).get("passed"): complete.append(int(module))
    rules=["all",*[effect_rule_key(s,d) for s in ([EffectMaskSpec("hedges_g","fixed",.4)] if c.motor else effect_mask_specs()) for d in ("ad_higher","control_higher","either")]]
    path=root/"public_bundle/manifest.json"
    previous=json.loads(path.read_text()) if path.exists() else {}
    manifest={**previous, "schema_version":2,"module_set":c.module_set,"network_method":state["method"],
        "status":"complete" if len(complete)==c.modules else "running","modules":c.modules,
        "complete_modules":len(complete),"completed_module_ids":complete,"rules":rules,
        "completed_rules":previous.get("completed_rules",[]),"components":list(c.components),
        "component_labels":c.labels,"expected_component_donors":c.expected_counts,
        "validation_donors":c.validation_donors,"original_donors":c.original_donors,
        "excluded_complete_donors":c.excluded_complete,"reference_controls":c.controls,
        "reference_donors":c.original_donors if state["method"]=="standard" else c.controls,
        "score_framework":c.framework(state["method"]),"cohort_scope":"incomplete_region_validation",
        "pooled_scores_available":False,"backend_verified":False,"publication_approved":False,
        "reference_policy":json.loads((root/"audit.json").read_text())["reference_policy"],
        "mdc_status":previous.get("mdc_status","unavailable"),
        "node_convention_policy":"source_specific_legacy",
        "cross_filter_node_convention_modules":sorted(state["assignment"].loc[
            state["assignment"].duplicated(["Tissue","Gene Symbol"],keep=False),"module"].astype(int).unique().tolist()),
        "rint_policy":"Original-cohort fitted average ranks; frozen interpolation and boundary clipping",
        "generated_utc":datetime.now(timezone.utc).isoformat()}
    atomic_json(manifest,path)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module-set",choices=("full_cohort","muscle_control_l4","muscle_all_donors_l4"),required=True)
    p.add_argument("--method",choices=("standard","control_anchored"),default="control_anchored")
    p.add_argument("--stage",choices=("audit","pilot","score"),required=True)
    p.add_argument("--root",type=Path,default=RUN)
    p.add_argument("--workers",type=int,choices=(1,2),default=2)
    p.add_argument("--modules",type=int,nargs="+")
    p.add_argument("--edge-batch",type=int,default=2048)
    args=p.parse_args(); state=load_inputs(args.module_set,args.method)
    root=args.root/args.module_set/args.method; report=audit(root,state)
    audit_discovery_membership(root,state)
    print(json.dumps(dict(stage="audit",module_set=args.module_set,method=args.method,status=report["status"])),flush=True)
    if args.stage=="audit": return
    if args.stage=="score" and not (root/"pilot_passed.json").exists():
        raise ValueError("Pilot gate must pass before expansion")
    sizes=state["assignment"].groupby("module").size().sort_values()
    repeated=state["assignment"].loc[state["assignment"].duplicated(["Tissue","Gene Symbol"],keep=False),"module"].tolist()
    pilot=list(dict.fromkeys([int(sizes.index[0]),int(sizes.index[-1]),1918 if not state["config"].motor else int(sizes.index[1]),*repeated,*map(int,sizes.index)]))[:4]
    modules=args.modules or (pilot if args.stage=="pilot" else list(map(int,sizes.index)))
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(module_job,m,root,args.edge_batch):m for m in modules}
        for future in as_completed(jobs):
            q=future.result(); results.append(q); update_manifest(root,state)
            print(json.dumps({k:q[k] for k in ("module","passed","rows","elapsed_seconds","peak_rss_mib")}),flush=True)
    if args.stage=="pilot":
        if set(modules)!=set(pilot) or max(q["peak_rss_mib"] for q in results)*args.workers>32768:
            raise ValueError("Pilot coverage or 32-GiB combined memory gate failed")
        atomic_json(dict(passed=True,modules=modules,results=results,workers=args.workers),root/"pilot_passed.json")
    print(json.dumps(dict(stage=args.stage,status="complete",modules=len(results))),flush=True)


if __name__=="__main__": main()
