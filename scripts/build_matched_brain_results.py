#!/usr/bin/env python3
"""Package complete module-level results; never copy donor scores or metadata."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

APP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(APP))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FORBIDDEN={"donor","projid","sample_id","pair_id","validation_pair_id","ad","control","anchor_ad","anchor_control"}
FAMILY=["test_type","metric_family","score_normalization","differential_edge_rule","scale","phenotype","correlation_method","module"]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def assert_public(frame):
    if FORBIDDEN&set(frame): raise ValueError("Donor-level fields cannot enter this release")
    for c in frame.select_dtypes(include=["object","string"]).columns:
        if frame[c].astype(str).str.contains(r"ENSG\d+",regex=True).any():
            raise ValueError(f"Internal gene identifiers in {c}")


def build(analysis,baseline,output):
    output.mkdir(parents=True,exist_ok=True)
    source=analysis/"demographic_sensitivity_v1"
    result=json.loads((source/"results_manifest.json").read_text())
    if result["status"]!="complete": raise ValueError("Finish the complete module families first")
    records=[]
    def record(path,kind,**extra):
        rows=pq.ParquetFile(path).metadata.num_rows
        if path.stat().st_size>=95*1024**2: raise ValueError("GitHub file-size limit exceeded")
        entry=dict(path=path.relative_to(output).as_posix(),kind=kind,rows=rows,
            bytes=path.stat().st_size,sha256=sha(path),**extra)
        records.append(entry);return entry
    for item in result["camel_statistics_files"]:
        original=analysis/item["path"]
        if sha(original)!=item["sha256"]: raise ValueError("Finalized source hash changed")
        frame=pd.read_parquet(original)
        assert_public(frame)
        cohort=str(frame.cohort_scope.iloc[0]);component=str(frame.component.iloc[0])
        if not frame.cohort_scope.eq(cohort).all() or not frame.component.eq(component).all():
            raise ValueError("Cohort/component shard is not isolated")
        if len(frame)!=item["rows"]: raise ValueError("Source row count changed")
        tag=hashlib.sha256((cohort+"|"+component).encode()).hexdigest()[:16]
        path=output/"camel"/f"{tag}.parquet";path.parent.mkdir(exist_ok=True)
        # Preserve the original size-bounded, dictionary-encoded row groups.
        # Native predicates return only the selected module tests to Pandas;
        # no complete million-row DataFrame is built by the app.
        shutil.copyfile(original,path)
        choices={c:sorted(map(str,frame[c].dropna().unique())) for c in FAMILY if c!="module"}
        entry=record(path,"camel",cohort_scope=cohort,component=component,choices=choices,source_sha256=item["sha256"])
        # Read back representative nonmissing and unavailable rows by exact
        # statistical identity; FDR and coefficients are copied, not refitted.
        for position in sorted({0,len(frame)//2,len(frame)-1}):
            row=frame.iloc[position]
            filters=[(c,"=",row[c]) for c in FAMILY]
            got=pd.read_parquet(path,filters=filters)
            if len(got)!=1: raise ValueError("Non-unique public test identity")
            for c in ["effect","r","p_permutation","q_permutation","p_nominal","q_nominal","n"]:
                np.testing.assert_allclose(got[c].iloc[0],row[c],equal_nan=True,rtol=0,atol=0)
        print(json.dumps(dict(context=tag,rows=entry["rows"],MiB=round(entry["bytes"]/1024**2,2))),flush=True)
    for name,path in [("mdc_labels.parquet",source/"mdc_label_statistics.parquet"),
        ("mdc_baseline.parquet",baseline/"mdc_summary.parquet"),
        ("module_details.parquet",baseline/"module_details.parquet")]:
        frame=pd.read_parquet(path);assert_public(frame)
        if name=="module_details.parquet":
            frame=frame.rename(columns={"Cluster ID":"module","Cluster Size":"module_size","Cluster Type":"cluster_type"})
            frame["n_tissues"]=frame[["AC","DLPFC","PCG"]].gt(0).sum(axis=1)
        frame.to_parquet(output/name,index=False,compression="zstd");record(output/name,name.removesuffix(".parquet"))
    for p in baseline.glob("kegg_*.parquet"):
        frame=pd.read_parquet(p);assert_public(frame)
        shutil.copyfile(p,output/p.name);record(output/p.name,"kegg")
    audits={}
    for name,p in [("discovery_design",analysis/"cohort_audit.json"),
        ("balanced_validation",analysis/"balanced_validation/cohort_manifest.json"),
        ("discovery_demography_validation",source/"cohort_manifest.json"),
        ("pilot",source/"pilot_report.json"),
        ("export_recovery",source/"finalization_recovery.json")]:
        audits[name]=json.loads(p.read_text())
    # These input reports contain aggregate balance measurements and hashes,
    # not private membership. Fail closed if their schema ever expands.
    def check_json(value):
        if isinstance(value,dict):
            if FORBIDDEN&set(value): raise ValueError("Private membership in an audit report")
            for v in value.values(): check_json(v)
        elif isinstance(value,list):
            for v in value: check_json(v)
    check_json(audits)
    (output/"audits.json").write_text(json.dumps(audits,indent=2,sort_keys=True)+"\n")
    manifest=dict(schema_version=1,status="complete",module_set="matched_control_brain_l4",module_count=247,
        assignment_count=49330,reference_controls=80,discovery_AD=80,heldout_AD=34,heldout_Control=34,
        source_analysis_commit="6b51e45274c99ad3e704f22319bdf9c996d94518",
        camel_rows=sum(x["rows"] for x in records if x["kind"]=="camel"),
        files=records,audit_sha256=sha(output/"audits.json"),
        BH=result["BH"],full_module_catalog=True,rankings_are_display_only=True,
        donor_level_data_published=False,
        scope=["All demographic-adjusted CAMEL/eigengene association tests","All CAMEL diagnosis-label tests",
            "Baseline and new MDC tests","Four edge sets","KEGG and regional enrichment","Module composition and entropy","Cohort balance and technical QA"],
        not_in_this_aggregate_release=["Donor scatter/distribution plots","Earlier unadjusted-association and dependent CT–TS catalogs"],
        important_limits=["Within-ROSMAP non-discovery-donor evidence, not external validation",
            "Matched-pair swaps are approximate for age and education; adjustment assumes a suitable nuisance model",
            "Discovery modules and effect masks stay frozen; no new discovery inferential p-values",
            "Incomplete-region cohorts have resolved components only; no imputed tissue or pooled CT/TS"])
    if manifest["camel_rows"]!=30794712: raise ValueError("Incomplete CAMEL catalog")
    temporary=output/"manifest.partial";temporary.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    temporary.replace(output/"manifest.json")
    print(json.dumps(dict(status="complete",rows=manifest["camel_rows"],MiB=sum(x["bytes"] for x in records)/1024**2)))
    return manifest


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-root",type=Path,required=True)
    p.add_argument("--baseline-root",type=Path,required=True)
    p.add_argument("--output",type=Path,default=APP/"data/matched_brain_results")
    a=p.parse_args();build(a.analysis_root,a.baseline_root,a.output)
