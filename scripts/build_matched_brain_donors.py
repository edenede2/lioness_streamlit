#!/usr/bin/env python3
"""Package the exact donor-score bundle approved on 2026-11-03.

No expression, PCA loadings, private pair linkage or identity mapping is copied.
The original scientific files are read-only. No network or statistical fit runs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd
import pyarrow.parquet as pq

APP=Path(__file__).resolve().parents[1]
METADATA_COLUMNS=["sample_id","diagnosis_group","cogn_global","cogng_demog_slope","cogng_path_slope",
    "motor10_demog_slope","sqrt_parksc_demog_slope","sex_code","age_at_death","education_years",
    "cogdx","braak_stage","cerad_score","parkinsonism","apoe_genotype","adnc","ad_control_split",
    "parkinsonism_label","clusters","gpath","amyloid","tangles","study","analysis_role","legacy_ad_control_split"]
FORBIDDEN={"donor","projid","pair_id","validation_pair_id","anchor_pair_id","ad","control","anchor_ad","anchor_control"}
EXPECTED={"scores":5614804,"eigengenes":621121,"transformations":106704}
BALANCED="Balanced incomplete-region validation (AD–Control)"
PROFILE="Discovery-demography-matched incomplete-region validation (AD–Control)"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024**2),b""):h.update(b)
    return h.hexdigest()


def public_frame(frame,*,metadata=False):
    if FORBIDDEN&set(frame):raise ValueError("Unapproved private columns in donor bundle")
    if metadata and set(frame)!=set(METADATA_COLUMNS):raise ValueError("Metadata approval schema changed")
    for c in frame.select_dtypes(include=["object","string"]).columns:
        if frame[c].astype(str).str.contains(r"ENSG\d+|-----BEGIN .*PRIVATE KEY",regex=True).any():
            raise ValueError("Internal identifier or secret in public output")


def build(root,output):
    output.mkdir(parents=True,exist_ok=True)
    meta=pd.read_parquet(root/"private_metadata.parquet",columns=METADATA_COLUMNS)
    if len(meta)!=1243 or meta.sample_id.nunique()!=1243:raise ValueError("Approved pseudonym population changed")
    public_frame(meta,metadata=True);samples=set(meta.sample_id)
    records=[]
    def record(path,kind,**extra):
        size=path.stat().st_size
        if size>=95*1024**2:raise ValueError("Shard exceeds public file-size limit")
        records.append(dict(path=path.relative_to(output).as_posix(),kind=kind,
            rows=pq.ParquetFile(path).metadata.num_rows,bytes=size,sha256=sha(path),**extra))
    meta.to_parquet(output/"sample_metadata.parquet",index=False,compression="zstd")
    record(output/"sample_metadata.parquet","metadata",source_sha256=sha(root/"private_metadata.parquet"))
    modules=None;cohorts=set()
    for kind,expected in EXPECTED.items():
        paths=sorted((root/kind).glob("M*.parquet"));ids={int(p.stem[1:]) for p in paths}
        if len(ids)!=247 or (modules is not None and ids!=modules):raise ValueError("Module membership differs across score types")
        modules=ids;total=0
        for p in paths:
            f=pd.read_parquet(p);public_frame(f)
            if f.module.nunique()!=1 or int(f.module.iloc[0])!=int(p.stem[1:]):raise ValueError("Module namespace mismatch")
            if "sample_id" in f and not set(f.sample_id)<=samples:raise ValueError("Unapproved sample added")
            if "cohort_scope" in f:cohorts.update(f.cohort_scope.unique())
            if kind=="transformations" and set(f.fit_cohort)!={"Discovery"}:raise ValueError("Transformation reference changed")
            target=output/kind/p.name;target.parent.mkdir(exist_ok=True)
            shutil.copyfile(p,target)
            record(target,kind,module=int(p.stem[1:]),source_sha256=sha(p))
            total+=len(f)
        if total!=expected:raise ValueError(f"Unexpected {kind} row count: {total}")
        print(json.dumps(dict(kind=kind,modules=len(ids),rows=total)),flush=True)
    # Publish membership only, never the partner or discovery-anchor linkage.
    balanced=pd.read_parquet(root/"balanced_validation/private_membership.parquet",columns=["sample_id","component"])
    balanced["cohort_scope"]=BALANCED
    pairs=pd.read_parquet(root/"demographic_sensitivity_v1/private_discovery_matched_pairs.parquet",columns=["component","ad","control"])
    profile=pairs.melt(id_vars="component",value_vars=["ad","control"],value_name="sample_id")[["sample_id","component"]]
    profile["cohort_scope"]=PROFILE
    memberships=pd.concat([balanced,profile],ignore_index=True)
    public_frame(memberships)
    if memberships.duplicated(["sample_id","component","cohort_scope"]).any():raise ValueError("Duplicate cohort membership")
    incomplete=set(meta.loc[meta.analysis_role.eq("Incomplete-region validation"),"sample_id"])
    if not set(memberships.sample_id)<=incomplete:raise ValueError("Complete/discovery donor in incomplete matching")
    memberships.to_parquet(output/"cohort_membership.parquet",index=False,compression="zstd")
    record(output/"cohort_membership.parquet","membership")
    manifest=dict(schema_version=1,status="complete",module_set="matched_control_brain_l4",module_count=247,
        modules=sorted(modules),sample_count=1243,expected_rows=EXPECTED,files=records,
        cohort_roles=sorted(cohorts),matched_cohorts=[BALANCED,PROFILE],
        approval="Explicit user approval of this exact pseudonymized bundle on 2026-11-03",
        publication_target="App GitHub bundle; no automatic push or Drive upload",
        excluded=["donor","projid","private pair/anchor IDs","raw expression","PCA loadings","edge matrices"],
        frozen_reference_controls=80,transformation_fit_scope="Discovery only; copied unchanged",
        metadata_columns=METADATA_COLUMNS,privacy="Existing public pseudonyms; downloadable derived scores and cohort-role labels")
    temporary=output/"manifest.partial";temporary.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    temporary.replace(output/"manifest.json")
    print(json.dumps(dict(status="complete",files=len(records),MiB=sum(r["bytes"] for r in records)/1024**2)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-root",type=Path,required=True)
    p.add_argument("--output",type=Path,default=APP/"data/matched_brain_donors")
    a=p.parse_args();build(a.analysis_root,a.output)
