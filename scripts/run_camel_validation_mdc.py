#!/usr/bin/env python3
"""Validation-only, resolved MDC with fixed discovery masks and dual nulls.

Each component is calculated on its own non-discovery AD/Control cohort. No
pooled CT/TS is invented. Checkpoints every ten permutations are deterministic.
No uploads, Git writes, or changes to existing MDC outputs are performed.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
APP=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(APP))
import numpy as np
import pandas as pd
from app_helpers.camel import COMPONENT_TISSUES, COMPONENT_LABELS
from app_helpers.camel_mdc import standardized, adjacency_edges, mdc_from_edges, empirical
from app_helpers.correlations import benjamini_hochberg
from app_helpers.gene_symbols import gene_symbol
from run_camel_validation import RUN, load_inputs, module_source, atomic_json, atomic_parquet, sha, update_public_manifest

_COMPONENT_CACHE={}


def component_background(component):
    if component in _COMPONENT_CACHE: return _COMPONENT_CACHE[component]
    x=load_inputs(); ids=x["donors"][component]
    meta=x["metadata"].set_index("sample_id").loc[[x["mapping"][d] for d in ids]]
    matrices=[]; names=[]; pools={}; offset=0
    for t in COMPONENT_TISSUES[component]:
        frame=x["matrix"][t].loc[ids]
        matrices.append(frame.to_numpy(float)); names.extend(t+"_"+frame.columns.astype(str))
        pools[t]=np.arange(offset,offset+len(frame.columns)); offset+=len(frame.columns)
    values=np.concatenate(matrices,axis=1)
    ad=standardized(values[meta.diagnosis_group.to_numpy()=="AD"])
    control=standardized(values[meta.diagnosis_group.to_numpy()=="Control"])
    result=(ad,control,{name:i for i,name in enumerate(names)},pools)
    _COMPONENT_CACHE[component]=result
    return result


def atomic_npz(path, **values):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".partial")
    with tmp.open("wb") as f: np.savez_compressed(f,**values)
    tmp.replace(path)


def mdc_job(module,component,root,permutations,source_context=""):
    started=time.monotonic(); root=Path(root)
    checkpoint=root/f"mdc_checkpoints/{component}/M{module}.npz"
    output=root/f"mdc_parts/{component}/M{module}.parquet"
    qa=output.with_suffix(".json")
    a,_,ri,ci,masks,source=module_source(module)
    signature=hashlib.sha256((sha(source)+component+str(permutations)+"v1"+source_context).encode()).hexdigest()
    if output.exists() and qa.exists() and json.loads(qa.read_text()).get("fingerprint")==signature:
        return str(output)
    tissues=a.public_tissue.to_numpy(); wanted=COMPONENT_TISSUES[component]
    nodes=np.flatnonzero(np.isin(tissues,wanted)); local={j:i for i,j in enumerate(nodes)}
    keep=(tissues[ri]==wanted[0])&(tissues[ci]==wanted[0]) if len(wanted)==1 else np.isin(tissues[ri],wanted)&np.isin(tissues[ci],wanted)&(tissues[ri]!=tissues[ci])
    edges=np.flatnonzero(keep)
    lr=np.array([local[j] for j in ri[keep]],int); lc=np.array([local[j] for j in ci[keep]],int)
    rule_names=list(masks); selected=np.vstack([masks[k][keep] for k in rule_names]).astype(float)
    ad,control,columns,pools=component_background(component)
    indices=np.array([columns[a.iloc[j].public_tissue+"_"+a.iloc[j]["Gene Symbol"]] for j in nodes],dtype=int)
    az,cz=ad[:,indices],control[:,indices]; p=3 if len(wanted)==1 else 2
    ae=adjacency_edges(az,lr,lc,p) if len(edges) else np.array([])
    ce=adjacency_edges(cz,lr,lc,p) if len(edges) else np.array([])
    observed,numerator,denominator=mdc_from_edges(ae,ce,selected)
    # Group correlation and edge-power agreement against a direct calculation.
    direct_error=0.
    if len(edges):
        for z,edge in ((az,ae),(cz,ce)):
            direct=np.corrcoef(z,rowvar=False)[lr,lc]
            direct=np.sign(direct)*np.abs(direct)**p
            direct_error=max(direct_error,float(np.nanmax(np.abs(edge-direct))))
        if direct_error>1e-10: raise ValueError("MDC direct adjacency agreement failed")
    inferential=len(az)>=5 and len(cz)>=5 and len(edges)>0
    sample=np.full((len(rule_names),permutations),np.nan)
    gene=np.full_like(sample,np.nan); done=0
    if checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as state:
            if str(state["fingerprint"])!=signature: raise ValueError("MDC checkpoint source/config changed")
            sample=state["sample"]; gene=state["gene"]; done=int(state["done"])
    if inferential:
        component_seed=int(hashlib.sha256(component.encode()).hexdigest()[:8],16)
        for permutation in range(done,permutations):
            rng=np.random.default_rng(np.random.SeedSequence([42,module,component_seed,permutation]))
            pa=np.take_along_axis(az,np.argsort(rng.random(az.shape),axis=0),axis=0)
            pc=np.take_along_axis(cz,np.argsort(rng.random(cz.shape),axis=0),axis=0)
            sample[:,permutation]=mdc_from_edges(adjacency_edges(pa,lr,lc,p),adjacency_edges(pc,lr,lc,p),selected)[0]
            sampled=np.empty(len(nodes),int)
            for t in wanted:
                positions=np.flatnonzero(tissues[nodes]==t)
                sampled[positions]=rng.choice(pools[t],len(positions),replace=False)
            gene[:,permutation]=mdc_from_edges(adjacency_edges(ad[:,sampled],lr,lc,p),
                adjacency_edges(control[:,sampled],lr,lc,p),selected)[0]
            if (permutation+1)%10==0 or permutation+1==permutations:
                atomic_npz(checkpoint,sample=sample,gene=gene,done=permutation+1,fingerprint=signature)
    pl_s=empirical(observed,sample,True); pl_g=empirical(observed,gene,True)
    pg_s=empirical(observed,sample,False); pg_g=empirical(observed,gene,False)
    rows=[]
    for k,rule in enumerate(rule_names):
        available=np.isfinite(observed[k]); reason=""
        if len(edges)==0: reason="structurally unavailable component"
        elif selected[k].sum()==0: reason="no retained discovery edges"
        elif not available: reason="undefined/near-zero Control denominator"
        elif not inferential: reason="descriptive only: fewer than five AD or five Controls"
        rows.append(dict(module=module,component=component,component_label=COMPONENT_LABELS[component],
            differential_edge_rule=rule,n_ad=len(az),n_control=len(cz),n_possible_edges=len(edges),
            n_retained_edges=int(selected[k].sum()),mdc=observed[k],log2_mdc=np.log2(observed[k]) if observed[k]>0 else np.nan,
            mean_abs_ad=numerator[k],mean_abs_control=denominator[k],inferential=bool(inferential and available),
            unavailable_reason=reason,p_loss_sample=pl_s[k],p_loss_gene=pl_g[k],
            p_gain_sample=pg_s[k],p_gain_gene=pg_g[k],sample_permutations=permutations if inferential else 0,
            gene_permutations=permutations if inferential else 0,
            sample_null_valid_draws=int(np.isfinite(sample[k]).sum()),gene_null_valid_draws=int(np.isfinite(gene[k]).sum()),
            sample_p_resolution=1/(1+np.isfinite(sample[k]).sum()) if np.isfinite(sample[k]).any() else np.nan,
            gene_p_resolution=1/(1+np.isfinite(gene[k]).sum()) if np.isfinite(gene[k]).any() else np.nan,
            seed=42,cohort_scope="incomplete_region_validation"))
    atomic_parquet(pd.DataFrame(rows),output)
    # Group-level adjacencies, not donor expression/edge networks. Bit masks
    # permit the selected view to render without storing 31 duplicate matrices.
    atomic_npz(root/f"public_bundle/mdc/adjacency/{component}/M{module}.npz",
        ad=ae.astype(np.float32),control=ce.astype(np.float32),row=lr,column=lc,
        mask_bits=np.packbits(selected.astype(bool),axis=1),rules=np.array(rule_names),
        gene_labels=np.array([str(gene_symbol(a.iloc[j]["Gene Symbol"])) for j in nodes]),
        tissue_labels=np.array([str(tissues[j]).replace("PCGBA23","PCG") for j in nodes]),
        n_nodes=len(nodes),n_edges=len(edges))
    atomic_json(dict(fingerprint=signature,elapsed_seconds=time.monotonic()-started,
                     direct_adjacency_error=direct_error,component=component,module=module,
                     permutations=permutations if inferential else 0),qa)
    return str(output)


def reconcile(root,permutations):
    parts=list((root/"mdc_parts").glob("*/M*.parquet"))
    if not parts: return
    frame=pd.concat([pd.read_parquet(p) for p in parts],ignore_index=True)
    keys=["component","differential_edge_rule"]
    if frame.duplicated(["module",*keys]).any(): raise ValueError("Duplicate MDC grain")
    complete_components=[c for c,b in frame.groupby("component") if b.module.nunique()==186]
    frame=frame.loc[frame.component.isin(complete_components)].copy()
    if frame.empty: return
    grouped=frame.groupby(keys,dropna=False,sort=False)
    for direction in ("loss","gain"):
        for null in ("sample","gene"):
            p=f"p_{direction}_{null}"; q=f"q_{direction}_{null}"
            frame[q]=grouped[p].transform(benjamini_hochberg)
            frame[q+"_family_n"]=grouped[p].transform("count").astype(int)
    frame["q_loss"]=np.maximum(frame.q_loss_sample,frame.q_loss_gene)
    frame["q_gain"]=np.maximum(frame.q_gain_sample,frame.q_gain_gene)
    frame["directional_fdr"]=np.where(frame.mdc.ge(1),frame.q_loss,frame.q_gain)
    atomic_parquet(frame,root/"public_bundle/mdc/resolved.parquet")
    manifest_path=root/"public_bundle/manifest.json"
    update_public_manifest(root,dict(mdc_components_complete=complete_components,
        mdc_status="complete" if len(complete_components)==6 and permutations==200 else "running"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=RUN)
    p.add_argument("--stage",choices=("pilot","run"),required=True)
    p.add_argument("--workers",type=int,choices=(1,2),default=2)
    p.add_argument("--permutations",type=int,default=200)
    p.add_argument("--components",nargs="+",choices=list(COMPONENT_TISSUES),default=list(COMPONENT_TISSUES))
    args=p.parse_args()
    if args.permutations!=200: raise ValueError("Public validation catalog requires the prespecified 200 permutations per null")
    audit=json.loads((args.root/"audit.json").read_text())
    if audit.get("status")!="passed" or not (args.root/"pilot_passed.json").exists(): raise ValueError("Scoring audit/pilot gates have not passed")
    if args.stage=="run" and not (args.root/"mdc_pilot_passed.json").exists(): raise ValueError("Run MDC pilot before expansion")
    load_inputs()
    modules=[436,771] if args.stage=="pilot" else sorted(load_inputs()["assignment"].module.unique())
    components=["TS_DLPFC"] if args.stage=="pilot" else args.components
    for component in components:
        component_background(component)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs={pool.submit(mdc_job,int(module),component,args.root,args.permutations):int(module) for module in modules}
            for f in as_completed(jobs):
                f.result(); print(json.dumps(dict(component=component,module=jobs[f],status="complete")),flush=True)
        _COMPONENT_CACHE.clear()
        reconcile(args.root,args.permutations)
    if args.stage=="pilot": atomic_json(dict(passed=True,modules=modules,component=components[0],permutations=200),args.root/"mdc_pilot_passed.json")


if __name__=="__main__": main()
