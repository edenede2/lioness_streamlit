#!/usr/bin/env python3
"""Resumable frozen-164-Control CAMEL scoring in incomplete-region donors.

Private analysis only: never uploads, pushes, changes original analyses, or fits
any validation-dependent reference, edge mask, imputer or RINT transformation.
Run the pilot before --stage score. The public bundle is staged outside the app.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
APP = Path(__file__).resolve().parents[1]
REPO = APP.parents[1]
sys.path[:0] = [str(APP), str(REPO)]
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from app_helpers.camel import (COMPONENT_TISSUES, COMPONENT_LABELS, FEATURES,
    FrozenRINT, correlation_edges, frozen_edge_weights, feature_values, robust_asinh_scale)
from models.differential_edges import effect_mask_specs, effect_size_mask, effect_rule_key
from scripts.python.run_bonobo_app_rosmap import configs, load_assignment_table

RUN = REPO / "out/camel_validation/20261021_frozen164_incomplete_regions"
ORIGINAL = REPO / "out/lioness_reference_comparison_rosmap/20260818_control_anchored_control_derived_l4_5phenotypes_6features/control_anchored/data/phenotypes.parquet"
EDGE_ROOT = REPO / "out/lioness_app_expansion/20260822_ad_control_differential_edges_all_networks/control_derived/checkpoints/lioness__control_anchored"
MAPPING = REPO / "out/lioness_regional_modules_rosmap/20261002_independent_region_app/private_regional_public_id_mapping.json"
RAW_PATHS = {"AC": REPO/"data/raw/ROSMAP_brain/ROSMAP_fixed_AC.csv",
             "DLPFC": REPO/"data/raw/ROSMAP_brain/ROSMAP_fixed_DLPFC.csv",
             "PCGBA23": REPO/"data/raw/ROSMAP_brain/ROSMAP_fixed_PCGBA23.csv"}
EXPECTED = {"TS_AC": 266, "TS_DLPFC": 752, "TS_PCGBA23": 195,
            "CT_AC__DLPFC": 230, "CT_AC__PCGBA23": 14, "CT_DLPFC__PCGBA23": 176}
_INPUTS = None


def canon(value):
    text = str(value).strip().replace("ROSMAP-", "")
    if text.endswith(".0"): text = text[:-2]
    if not text.isdigit(): raise ValueError("Invalid private source identifier")
    return "ROSMAP-" + text.zfill(8)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(2**20), b""): h.update(chunk)
    return h.hexdigest()


def atomic_json(value, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+"\n")
    temporary.replace(path)


def atomic_parquet(frame, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".partial")
    frame.to_parquet(temporary,index=False,compression="zstd")
    temporary.replace(path)


def update_public_manifest(root, updates=None, completed_rule=None):
    """Serialize concurrent score/statistics/MDC status updates, not whole jobs."""
    import fcntl
    path=Path(root)/"public_bundle/manifest.json"
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        current=json.loads(path.read_text())
        current.update(updates or {})
        if completed_rule is not None:
            current["completed_rules"]=list(dict.fromkeys([*current.get("completed_rules",[]),completed_rule]))
            current["statistics_status"]="complete" if len(current["completed_rules"])==31 else "running"
        atomic_json(current,path)
        return current


def load_inputs():
    global _INPUTS
    if _INPUTS is not None: return _INPUTS
    assignment=load_assignment_table(next(c for c in configs(REPO) if c.key=="control_derived"))
    assignment["public_tissue"]=assignment.Tissue.replace({"MFBA9BA46":"DLPFC"})
    original=pd.read_parquet(ORIGINAL)
    original["donor"]=original.donor.map(canon)
    if original.donor.duplicated().any() or len(original)!=450:
        raise ValueError("Original cohort is not the unique frozen 450 donors")
    matrix={}
    for tissue,path in RAW_PATHS.items():
        x=pd.read_csv(path,index_col=0)
        x.index=x.index.map(canon)
        if x.index.duplicated().any() or x.columns.duplicated().any():
            raise ValueError("Expression duplicate identifiers require explicit reconciliation")
        matrix[tissue]=x
    complete=set.intersection(*(set(x.index) for x in matrix.values()))
    incomplete=set.union(*(set(x.index) for x in matrix.values()))-complete
    if len(complete)!=464 or len(incomplete)!=793 or not set(original.donor)<=complete:
        raise ValueError("Complete/incomplete cohort does not reproduce the source audit")
    mapping={canon(k):v for k,v in json.loads(MAPPING.read_text()).items()}
    if len(set(mapping.values()))!=len(mapping): raise ValueError("Pseudonym collision")
    if not (incomplete|set(original.donor))<=mapping.keys():
        raise ValueError("Existing private mapping does not cover all required samples")
    # Metadata is already public. No donor linkage is reconstructed from values.
    from app_helpers.study import with_study_metadata
    frames=[pd.read_parquet(APP/f"data/regional_modules/metadata/maximum_tissue/{t}.parquet") for t in ("AC","DLPFC","PCG")]
    metadata=pd.concat(frames,ignore_index=True)
    if (metadata.groupby("sample_id").nunique(dropna=True)>1).any().any():
        raise ValueError("Conflicting existing public metadata across tissue files")
    metadata=with_study_metadata(metadata.groupby("sample_id",as_index=False).first(),"brain")
    selected=[mapping[d] for d in sorted(incomplete)]
    metadata=metadata.set_index("sample_id").loc[selected].reset_index()
    metadata["ad_control_split"]="Non-discovery validation"
    if len(metadata)!=793 or {"donor","projid"}&set(metadata): raise ValueError("Metadata privacy/grain failure")
    donors={c:sorted(set.intersection(*(set(matrix[t].index) for t in tissues))-complete)
            for c,tissues in COMPONENT_TISSUES.items()}
    if {c:len(d) for c,d in donors.items()}!=EXPECTED: raise ValueError("Component cohorts changed")
    controls=original.loc[original.diagnosis_group.eq("Control"),"donor"].tolist()
    source_eigengene=REPO/"data/proccessed/se2_signed_alt_control_filtered/se2_rosmap_control_signed-alt/se2_4_rosmap_full_Control_Brain_ME_commonDonors_cross_tissue.tsv"
    detection=set(map(canon,pd.read_csv(source_eigengene,nrows=0).columns[1:]))
    if len(controls)!=164 or detection!=set(controls) or incomplete&detection:
        raise ValueError("Saved module-discovery donor header does not match the frozen reference")
    for t in RAW_PATHS:
        genes=set(assignment.loc[assignment.public_tissue.eq(t),"Gene Symbol"])
        if not genes<=set(matrix[t].columns): raise ValueError("Module expression header mapping incomplete")
    _INPUTS=dict(assignment=assignment,original=original,matrix=matrix,complete=complete,
        incomplete=incomplete,mapping=mapping,metadata=metadata,donors=donors,controls=controls)
    return _INPUTS


def audit(root):
    x=load_inputs(); root=Path(root)
    atomic_parquet(x["metadata"],root/"public_bundle/sample_metadata.parquet")
    rows=[]
    for component,donors in x["donors"].items():
        public=x["metadata"].set_index("sample_id").loc[[x["mapping"][d] for d in donors]]
        rows.append(dict(component=component,component_label=COMPONENT_LABELS[component],n=len(donors),
                         **{g:int(public.diagnosis_group.eq(g).sum()) for g in ("Control","MCI","AD","Unclassified")}))
    pd.DataFrame(rows).to_csv(root/"public_bundle/cohort_counts.tsv",sep="\t",index=False)
    source_paths=[ORIGINAL,MAPPING,*RAW_PATHS.values(),
        next(c.assignments for c in configs(REPO) if c.key=="control_derived")]
    record=dict(status="passed",complete_donors=464,excluded_complete_donors=464,validation_donors=793,
        reference_controls=164,modules=186,assignment_rows=49392,
        component_counts=rows,source_hashes={str(p.relative_to(REPO)):sha(p) for p in source_paths if p!=MAPPING},
        private_mapping_sha256=sha(MAPPING),scope="non-discovery donors within ROSMAP; not external validation",
        score_framework="CAMEL",reference_policy="Frozen original 164 Controls; all new donors add-one",
        no_absent_region_imputation=True,source_discovery_donor_check="164 saved eigengene header IDs equal reference IDs",
        predictions_unchanged=True)
    previous=root/"audit.json"
    if previous.exists():
        old=json.loads(previous.read_text())
        if old.get("source_hashes")!=record["source_hashes"] or old.get("private_mapping_sha256")!=record["private_mapping_sha256"]:
            raise ValueError("Frozen source inputs changed; use a new run root, not mixed checkpoints")
    atomic_json(record,previous)
    return record


def module_source(module):
    x=load_inputs(); a=x["assignment"].loc[x["assignment"].module.eq(module)].copy()
    features=a.feature_id.tolist()
    original=x["original"].donor.tolist()
    pieces=[]
    for tissue in RAW_PATHS:
        sub=a.loc[a.public_tissue.eq(tissue)]
        frame=x["matrix"][tissue].loc[original,sub["Gene Symbol"]].copy()
        frame.columns=sub.feature_id
        pieces.append(frame)
    full=pd.concat(pieces,axis=1)[features]
    # Reproduce legacy reference preprocessing, using original donors only.
    full=full.apply(lambda r:r.fillna(r.mean()),axis=1).fillna(full.mean(axis=0)).fillna(0.)
    rows,columns=np.tril_indices(len(a),-1)
    source=EDGE_ROOT/f"M{module}__edge_statistics.parquet"
    edge=pd.read_parquet(source,columns=["edge_index","row_index","column_index","discovery_hedges_g","discovery_mean_difference"])
    if not np.array_equal(edge.row_index,rows) or not np.array_equal(edge.column_index,columns) or not np.array_equal(edge.edge_index,np.arange(len(rows))):
        raise ValueError("Discovery edge ordering does not exactly match module assignments")
    masks={"all":np.ones(len(rows),bool)}
    for spec in effect_mask_specs():
        values=edge["discovery_"+spec.statistic].to_numpy(float)
        for direction in ("ad_higher","control_higher","either"):
            masks[effect_rule_key(spec,direction)]=effect_size_mask(values,edge.edge_index.to_numpy(),spec,direction=direction)
        ad=masks[effect_rule_key(spec,"ad_higher")]; control=masks[effect_rule_key(spec,"control_higher")]
        if not np.array_equal(ad|control,masks[effect_rule_key(spec,"either")]):
            raise ValueError("Directional mask union identity failed")
    return a,full,rows,columns,masks,source


def module_job(module, root, edge_batch=2048):
    started=time.monotonic(); root=Path(root)
    destination=root/f"public_bundle/modules/M{module}.parquet"
    anchors_path=root/f"public_bundle/rint/M{module}.parquet"
    qa_path=root/f"checkpoints/M{module}.json"
    a,full,ri,ci,masks,source=module_source(module)
    source_hash=sha(source)
    if destination.exists() and anchors_path.exists() and qa_path.exists():
        qa=json.loads(qa_path.read_text())
        if qa.get("passed") and qa.get("calculation_version")==2 and qa.get("source_edge_sha256")==source_hash:
            return qa
    x=load_inputs(); original=x["original"].donor.tolist()
    control_ids=x["controls"]; control_pos={d:i for i,d in enumerate(control_ids)}
    roles=np.array([control_pos.get(d,-1) for d in original])
    rules=list(masks); tissue=a.public_tissue.to_numpy(); out=[]; anchors=[]; qa_rows=[]
    for component,tissues in COMPONENT_TISSUES.items():
        nodes=np.flatnonzero(np.isin(tissue,tissues)); local={j:i for i,j in enumerate(nodes)}
        if component.startswith("TS_"):
            keep=np.flatnonzero((tissue[ri]==tissues[0])&(tissue[ci]==tissues[0]))
        else:
            keep=np.flatnonzero(np.isin(tissue[ri],tissues)&np.isin(tissue[ci],tissues)&(tissue[ri]!=tissue[ci]))
        new_ids=x["donors"][component]
        valid=pd.DataFrame(index=new_ids)
        for t in tissues:
            sub=a.loc[a.public_tissue.eq(t)]
            part=x["matrix"][t].loc[new_ids,sub["Gene Symbol"]].copy(); part.columns=sub.feature_id
            valid=pd.concat([valid,part],axis=1)
        gene_ids=a.iloc[nodes].feature_id.tolist()
        ref=full.loc[control_ids,gene_ids].to_numpy(float)
        # Sparse gene-cell missingness only: frozen Control means, never missing regions.
        new=valid[gene_ids].to_numpy(float)
        if new.size: new=np.where(np.isfinite(new),new,ref.mean(axis=0))
        targets=np.vstack([full[gene_ids].to_numpy(float),new])
        positions=np.r_[roles,np.full(len(new_ids),-1)]
        lr=np.array([local[j] for j in ri[keep]],dtype=int)
        lc=np.array([local[j] for j in ci[keep]],dtype=int)
        acc=np.zeros((len(targets),len(rules),5))
        selected_masks=np.vstack([masks[r][keep] for r in rules])
        n_retained=selected_masks.sum(axis=1)
        if len(keep):
            mean,variance,cross=correlation_edges(ref,lr,lc)
            for start in range(0,len(keep),edge_batch):
                sl=slice(start,start+edge_batch); power=np.full(len(lr[sl]),3 if component.startswith("TS_") else 2)
                w=frozen_edge_weights(ref,targets,lr[sl],lc[sl],power,reference_positions=positions,
                    reference_statistics=(mean,variance,cross[sl]))
                mask=selected_masks[:,sl].astype(float).T
                acc[:,:,0]+=np.maximum(w,0)@mask
                acc[:,:,1]+=np.maximum(-w,0)@mask
                acc[:,:,2]+=(w>0).astype(float)@mask
                acc[:,:,3]+=(w<0).astype(float)@mask
                acc[:,:,4]+=(w*w)@mask
            # Direct covariance recalculation and role-neutral target invariance.
            sample_indices=[0,roles.tolist().index(-1),len(original)]
            for i in sample_indices:
                target=targets[i]; position=positions[i]
                changed=np.delete(ref,position,axis=0) if position>=0 else np.vstack([ref,target])
                r=np.corrcoef(ref,rowvar=False)[lr,lc]
                q=np.corrcoef(changed,rowvar=False)[lr,lc]
                p=3 if component.startswith("TS_") else 2
                ar=np.sign(r)*np.abs(r)**p; aq=np.sign(q)*np.abs(q)**p
                direct=len(ref)*ar-(len(ref)-1)*aq if position>=0 else (len(ref)+1)*aq-len(ref)*ar
                got=frozen_edge_weights(ref,target[None,:],lr,lc,np.full(len(lr),p),reference_positions=np.array([position]))[0]
                err=float(np.nanmax(np.abs(got-direct)))
                if err>1e-8: raise ValueError("Direct frozen-reference formula validation failed")
                if i==len(original):
                    together=frozen_edge_weights(ref,targets[-2:],lr[:5],lc[:5],np.full(min(5,len(lr)),p))
                    alone=frozen_edge_weights(ref,targets[-1:],lr[:5],lc[:5],np.full(min(5,len(lr)),p))
                    if not np.array_equal(together[-1],alone[0]): raise ValueError("Other-target invariance failed")
                qa_rows.append(dict(component=component,role="loo" if position>=0 else "add_one",max_abs_edge_error=err))
        ids=original+new_ids
        for k,rule in enumerate(rules):
            p,m,pc,mc,ss=acc[:,k,:].T
            retained=int(n_retained[k]); count=len(keep)
            frame=pd.DataFrame(dict(sample_id=[x["mapping"][d] for d in ids],module=int(module),
                component=component,differential_edge_rule=rule,n_features=len(a),n_possible_edges=count,
                n_retained_edges=retained,n_positive_edges=np.rint(pc).astype(int),
                n_negative_edges=np.rint(mc).astype(int),n_zero_edges=retained-np.rint(pc+mc).astype(int),
                n_pruned_edges=count-retained,positive_weight_sum=p,negative_weight_magnitude=m,
                signed_weight_sum=p-m,absolute_weight_sum=p+m,weight_squared_sum=ss))
            if count==0:
                for name in ("positive_weight_sum","negative_weight_magnitude","signed_weight_sum","absolute_weight_sum","weight_squared_sum"):
                    frame[name]=np.nan
            if (frame.n_positive_edges+frame.n_negative_edges+frame.n_zero_edges!=retained).any():
                raise ValueError("Sign count identity failed")
            for normalization in ("standard_pruned","retained_edge"):
                for family in FEATURES:
                    values=feature_values(frame,family,normalization)
                    fitted=FrozenRINT.fit(values[:len(original)])
                    anchors.append(dict(module=int(module),component=component,differential_edge_rule=rule,
                        score_normalization=normalization,metric_family=family,x=fitted.x.tolist(),z=fitted.z.tolist(),
                        asinh_scale=robust_asinh_scale(values[:len(original)]),
                        training_n=int(np.isfinite(values[:len(original)]).sum())))
            out.append(frame.iloc[len(original):].copy())
    result=pd.concat(out,ignore_index=True)
    if len(result)!=sum(EXPECTED.values())*len(rules) or result.duplicated(["sample_id","component","differential_edge_rule"]).any():
        raise ValueError("Module donor/component/mask grain failed")
    atomic_parquet(result,destination); atomic_parquet(pd.DataFrame(anchors),anchors_path)
    qa=dict(module=int(module),passed=True,calculation_version=2,rows=len(result),edge_masks=len(rules),source_edge_sha256=source_hash,
        score_sha256=sha(destination),rint_sha256=sha(anchors_path),direct_checks=qa_rows,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        elapsed_seconds=time.monotonic()-started,all_validation_donors_add_one=True)
    atomic_json(qa,qa_path)
    return qa


def update_manifest(root, required_modules):
    root=Path(root); completed={}
    for m in required_modules:
        p=root/f"checkpoints/M{m}.json"
        if p.exists():
            q=json.loads(p.read_text())
            if q.get("passed") and q.get("calculation_version")==2: completed[str(m)]=q
    rules=["all",*[effect_rule_key(s,d) for s in effect_mask_specs() for d in ("ad_higher","control_higher","either")]]
    manifest=dict(schema_version=1,status="complete" if len(completed)==186 else "running",
        score_framework="CAMEL",module_set="control_derived",cohort_scope="incomplete_region_validation",
        modules=186,complete_modules=len(completed),components=list(COMPONENT_TISSUES),expected_component_donors=EXPECTED,
        completed_module_ids=list(map(int,completed)),rules=rules,reference_controls=164,
        completed_rules=[],
        reference_policy="Frozen original Controls; every validation donor add-one",
        rint_policy="Original-450 fitted average ranks; monotone interpolation; clipped boundary scores",
        pooled_scores_available=False,network_method="control_anchored",mdc_status="unavailable",
        publication_approved=False,public_donor_identifiers="Existing pseudonyms only",
        generated_utc=datetime.now(timezone.utc).isoformat())
    old=root/"public_bundle/manifest.json"
    if old.exists():
        previous=json.loads(old.read_text())
        for k in ("mdc_status","statistics_status","publication_approved","completed_rules"): manifest[k]=previous.get(k,manifest.get(k,"unavailable"))
    atomic_json(manifest,old)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("audit","pilot","score"),required=True)
    p.add_argument("--root",type=Path,default=RUN)
    p.add_argument("--modules",type=int,nargs="+")
    p.add_argument("--workers",type=int,choices=(1,2),default=2)
    p.add_argument("--edge-batch",type=int,default=2048)
    args=p.parse_args(); args.root.mkdir(parents=True,exist_ok=True)
    report=audit(args.root)
    print(json.dumps({"stage":"audit","status":report["status"],"validation_donors":793}),flush=True)
    if args.stage=="audit": return
    if args.stage=="score" and not (args.root/"pilot_passed.json").exists():
        raise ValueError("Run and pass the pilot before full scoring")
    load_inputs()  # Linux fork shares read-only expression without per-worker copies.
    modules=args.modules or ([436,356,121,771] if args.stage=="pilot" else sorted(load_inputs()["assignment"].module.unique()))
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(module_job,int(m),args.root,args.edge_batch):int(m) for m in modules}
        for f in as_completed(jobs):
            q=f.result(); results.append(q)
            print(json.dumps({k:q[k] for k in ("module","passed","rows","elapsed_seconds","peak_rss_mib")}),flush=True)
            update_manifest(args.root,sorted(load_inputs()["assignment"].module.unique()))
    if args.stage=="pilot":
        if len(results)!=4 or max(q["peak_rss_mib"] for q in results)*args.workers>32*1024:
            raise ValueError("Pilot module coverage or 32-GiB memory gate failed")
        atomic_json(dict(passed=True,modules=modules,workers=args.workers,results=results),args.root/"pilot_passed.json")
    print(json.dumps({"stage":args.stage,"status":"complete","modules":len(results)}),flush=True)


if __name__=="__main__": main()
