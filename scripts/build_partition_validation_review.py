#!/usr/bin/env python3
"""Package aggregate-only validation evidence; no upload or Git operation.

Default installation exposes locally installed, complete-context review shards
only. Donor scores, curves and metadata require separate approved publication.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
APP=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(APP),str(APP/"scripts")]
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from app_helpers.data import load_mdc_resolved
from app_helpers.validation_datasets import dataset
from run_camel_validation import atomic_json, atomic_parquet, sha
from run_partition_validation import RUN


def build(root):
    root=Path(root); bundle=root/"public_bundle"; review=bundle/"review"
    manifest=json.loads((bundle/"manifest.json").read_text()); audit=json.loads((root/"audit.json").read_text())
    approval_path=root/"publication_approval.json"
    approval=json.loads(approval_path.read_text()) if approval_path.exists() else {}
    approved=bool(approval.get("approved") and approval.get("module_set")==manifest["module_set"]
                  and manifest["network_method"] in approval.get("methods",[]))
    published=bool(manifest.get("backend_verified"))
    config=dataset(manifest["module_set"],manifest["network_method"])
    checks=[json.loads((root/f"checkpoints/M{m}.json").read_text()) for m in manifest["completed_module_ids"]]
    formula=pd.DataFrame([dict(module=q["module"],**r) for q in checks for r in q["direct_checks"]])
    reproduction=pd.DataFrame([q["reproduction"] for q in checks])
    formula["passed"]=formula.max_abs_edge_error.le(1e-8); formula["tolerance"]=1e-8
    reproduction["passed"]=reproduction.max_relative_error.le(1e-8)
    atomic_parquet(formula,review/"formula_checks.parquet")
    atomic_parquet(reproduction,review/"score_reproduction.parquet")
    atomic_parquet(pd.DataFrame(audit["component_counts"]),review/"cohorts.parquet")
    completion=pd.DataFrame([dict(rule=rule,scores="complete" if manifest["complete_modules"]==config.modules else "partial",
        associations="complete" if rule in manifest["completed_rules"] else "not yet complete",
        mdc=manifest.get("mdc_status","unavailable")) for rule in manifest["rules"]])
    atomic_parquet(completion,review/"completion.parquet")
    technical=pd.DataFrame([
        dict(check="Cohort exclusion and frozen inputs",status=audit["status"],detail=f"Exclude {config.excluded_complete} complete-expression donors; {config.validation_donors} component-overlapping validation donors",evidence="audit.json"),
        dict(check="Direct LIONESS formulas",status="passed" if formula.passed.all() else "failed",detail=f"Maximum sampled edge error {formula.max_abs_edge_error.max():.3g}",evidence="formula_checks.parquet"),
        dict(check="Original saved score reproduction",status="passed" if reproduction.passed.all() else "failed",detail=f"{int(reproduction.tested_values.sum())} sampled values; maximum relative error {reproduction.max_relative_error.max():.3g}",evidence="score_reproduction.parquet"),
        dict(check="Held-out donor scoring role",status="passed" if all(q["all_validation_donors_add_one"] for q in checks) else "failed",detail=audit["reference_policy"],evidence="Scoring checkpoints"),
        dict(check="Legacy assignment conventions",status="passed",detail="User-approved unique-node unfiltered and positional-node effect-filtered conventions; 16 affected brain modules flagged",evidence="Node counts and convention fields"),
        dict(check="Publication",status="backend verified" if published else "approved; backend pending" if approved else "pending approval",
             detail="Donor bundle backend readback passed" if published else
                 "Scoped donor-bundle approval recorded; checksum verification and independent backend readback are still required" if approved else
                 "No new donor-level files uploaded; backend readback and scoped approval remain required",evidence="Publication manifest"),
        dict(check="Prediction",status="not applicable",detail="No prediction models or prediction null experiments are run by this extension",evidence="Scope")])
    atomic_parquet(technical,review/"technical_checks.parquet")
    entries=[]
    for rule in manifest["completed_rules"]:
        for norm in ("standard_pruned","retained_edge"):
            path=bundle/"concordance"/rule/f"{norm}.parquet"
            frame=pd.read_parquet(path)
            for (feature,outcome),part in frame.groupby(["metric_family","phenotype"],sort=True,observed=True):
                relative=Path("associations")/rule/norm/feature/f"{outcome}.parquet"
                atomic_parquet(part,review/relative)
                entries.append(dict(rule=rule,normalization=norm,feature=feature,outcome=outcome,
                    path=str(relative),rows=len(part),sha256=sha(review/relative),original_pooled_included=True))
    path=bundle/"mdc/resolved.parquet"
    if path.exists():
        mdc=pd.read_parquet(path).rename(columns={c:"validation_"+c for c in ("mdc","log2_mdc","directional_fdr")})
        original=load_mdc_resolved(config.module_set)[["module","component","mdc","log2_mdc","directional_fdr"]]
        if original.duplicated(["module","component"]).any(): raise ValueError("Ambiguous original MDC")
        original=original.rename(columns={c:"original_"+c for c in original if c not in ("module","component")})
        mdc=mdc.merge(original,on=["module","component"],how="left",validate="many_to_one")
        mdc.loc[mdc.differential_edge_rule.ne("all"),[c for c in mdc if c.startswith("original_")]]=np.nan
        mdc["original_comparison_available"]=mdc.original_mdc.notna()
        atomic_parquet(mdc,review/"mdc_comparison.parquet")
    result=dict(schema_version=2,module_set=config.module_set,network_method=manifest["network_method"],
        run_id=root.parent.parent.name,generated_utc=datetime.now(timezone.utc).isoformat(),
        validation_donors=config.validation_donors,excluded_complete_donors=config.excluded_complete,
        modules=config.modules,assignment_rows=config.assignments,complete_modules=manifest["complete_modules"],
        reference_controls=config.controls,reference_donors=manifest["reference_donors"],
        completed_rules=manifest["completed_rules"],mdc_status=manifest["mdc_status"],
        association_shards=entries,original_pooled_included=True,source_hashes=audit["source_hashes"],
        cross_filter_node_convention_modules=manifest.get("cross_filter_node_convention_modules",[]),
        limitations=["Non-discovery donors within ROSMAP, not external-cohort validation.",
            "Expression preprocessing, references, module definitions and discovery edge masks remain frozen.",
            "Original reference donors use leave-one-out scoring; validation donors use add-one scoring. Formula QA alone does not rule out reference-role artifacts.",
            "No missing region is imputed; component cohorts overlap and no pooled three-tissue score is available.",
            "Original and validation FDRs are separate module-only BH families, calculated before display filters.",
            "Pooled associations are calculated directly on pooled donors, not averages of subgroup correlations.",
            "Frozen validation RINT clipping may introduce ties.",
            "Effect-filtered original discovery associations are selection-biased; replication is descriptive.",
            "Sixteen full-cohort brain modules use different node conventions across unfiltered and filtered analyses.",
            "Missing original filtered-MDC counterparts are unavailable; no all-edge result is substituted."])
    atomic_json(result,review/"manifest.json")
    return result


def install(root,review_manifest):
    root=Path(root); bundle=root/"public_bundle"; config=dataset(review_manifest["module_set"],review_manifest["network_method"])
    target=APP/"data"/config.relative_root(review_manifest["network_method"])
    rollback=root/"rollback"/str(time.time_ns())
    # Compact common contexts only. Uninstalled shards are omitted from the
    # deployed registry until indexed/verified in the backend, never queried.
    entries=[e for e in review_manifest["association_shards"] if
        (e["rule"]=="all" or "__hedges_g__fixed_0.40__" in e["rule"]) and e["feature"]=="connectivity" and e["outcome"] in
        ("cogn_global","cogng_demog_slope","cogng_path_slope","gpath","amyloid","tangles")]
    files=[(bundle/"review"/e["path"],target/"review"/e["path"]) for e in entries]
    files += [(p,target/"review"/p.name) for p in (bundle/"review").glob("*.parquet")]
    installed={**review_manifest,"association_shards":entries,
        "unpublished_association_shards":len(review_manifest["association_shards"])-len(entries)}
    for source,destination in files:
        if {"sample_id","donor","projid"}&set(pq.read_schema(source).names): raise ValueError("Only aggregate evidence can be installed without donor-bundle approval")
        if source.stat().st_size>=95*1024**2: raise ValueError("Public shard exceeds 95 MiB")
        if destination.exists():
            saved=rollback/destination.relative_to(target); saved.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(destination,saved)
        destination.parent.mkdir(parents=True,exist_ok=True)
        temp=destination.with_suffix(destination.suffix+".partial"); shutil.copy2(source,temp); temp.replace(destination)
    # Registry last. No donor manifest, metadata, scores or transformations are
    # installed here. The Associations cohort control therefore remains hidden.
    manifest_path=target/"review/manifest.json"
    if manifest_path.exists():
        previous=rollback/"review/manifest.json"; previous.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(manifest_path,previous)
    atomic_json(installed,manifest_path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=RUN)
    p.add_argument("--module-set",choices=("full_cohort","muscle_control_l4","muscle_all_donors_l4"),required=True)
    p.add_argument("--method",choices=("standard","control_anchored"),default="control_anchored")
    p.add_argument("--install-aggregate-review",action="store_true")
    args=p.parse_args(); root=args.root/args.module_set/args.method
    result=build(root)
    if args.install_aggregate_review: install(root,result)
    print(json.dumps(dict(association_shards=len(result["association_shards"]),mdc_status=result["mdc_status"],installed=args.install_aggregate_review)))


if __name__=="__main__": main()
