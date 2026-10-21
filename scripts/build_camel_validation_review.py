#!/usr/bin/env python3
"""Build compact, privacy-safe validation evidence and replication shards.

Only saved scores/statistics/QA are read. No network, mask, or permutation is
recomputed. --install copies only these aggregate review files, with rollback.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import numpy as np
import pandas as pd

from app_helpers.correlations import benjamini_hochberg
from app_helpers.data import load_mdc_resolved
from app_helpers.validation_review import MATCH_KEYS
from run_camel_validation import RUN, atomic_json, atomic_parquet, sha


def build(root: Path) -> dict:
    bundle = root / "public_bundle"
    review = bundle / "review"
    audit = json.loads((root / "audit.json").read_text())
    manifest = json.loads((bundle / "manifest.json").read_text())
    reproduction = json.loads((root / "existing_score_reproduction.json").read_text())
    release_path = bundle / "release_manifest.json"
    release = json.loads(release_path.read_text()) if release_path.exists() else {}
    qas = [json.loads((root / f"checkpoints/M{m}.json").read_text()) for m in manifest["completed_module_ids"]]
    formula = pd.DataFrame([dict(module=q["module"], **r) for q in qas for r in q["direct_checks"]])
    if not formula.empty:
        formula["tolerance"] = 1e-8
        formula["passed"] = formula.max_abs_edge_error.le(formula.tolerance)
    atomic_parquet(formula, review / "formula_checks.parquet")
    r = pd.DataFrame(reproduction["checks"])
    r["relative_tolerance"] = 1e-8
    r["passed"] = r.max_relative_error.le(r.relative_tolerance)
    atomic_parquet(r, review / "score_reproduction.parquet")
    atomic_parquet(pd.DataFrame(audit["component_counts"]), review / "cohorts.parquet")
    completed = set(manifest.get("completed_rules", []))
    completion = pd.DataFrame([
        dict(rule=rule, scores="complete" if manifest["complete_modules"] == audit["modules"] else "partial",
             associations="complete" if rule in completed else "not yet complete",
             mdc=manifest.get("mdc_status", "unavailable")) for rule in manifest["rules"]
    ])
    atomic_parquet(completion, review / "completion.parquet")
    maximum = float(formula.max_abs_edge_error.max()) if len(formula) else np.nan
    def check(name, status, detail, evidence):
        return dict(check=name, status=status, detail=detail, evidence=evidence)
    checks = [
        check("Cohort and fixed module audit", audit["status"],
              f'{audit["validation_donors"]} validation donors; exclude all {audit["excluded_complete_donors"]} complete-expression donors; {audit["modules"]} modules / {audit["assignment_rows"]:,} assignments.', "audit.json"),
        check("Direct leave-one-out / add-one formula", "passed" if len(formula) and formula.passed.all() else "unavailable",
              f'{len(formula):,} sampled formula checks across {formula.module.nunique() if len(formula) else 0} modules; maximum absolute edge error {maximum:.3g}; tolerance 1e-8.', "formula_checks.parquet"),
        check("Reproduce original saved R scores", "passed" if reproduction.get("passed") and r.passed.all() else "failed",
              f'{len(r)} sampled modules, {int(r.tested_values.sum()):,} values; maximum relative error {r.max_relative_error.max():.3g}. This is sampled technical reproduction, not donor-cohort replication.', "score_reproduction.parquet"),
        check("Uniform scoring role", "passed" if qas and all(q.get("all_validation_donors_add_one") for q in qas) else "unavailable",
              "Every validation donor, including Controls, uses add-one scoring against the frozen original 164 Controls. Diagnosis is not a scoring input.", "Scoring checkpoints and frozen-reference implementation"),
        check("Other-target batch invariance", "passed" if qas and all(q.get("passed") for q in qas) else "unavailable",
              "Sampled singleton versus batched target scoring is asserted before scoring checkpoints are saved. No new target enters the reference.", "Scoring checkpoints; bounded edge samples"),
        check("Edge sums, masks, source hashes and privacy", release.get("status", "not yet audited"),
              "Release audit checks signed/absolute sums, sign counts, directional-mask unions, score/transform hashes, source hashes and public identifier schema.", "release_manifest.json"),
        check("Module-only association BH", release.get("status", "not yet audited"),
              "BH is independently reconciled across valid modules for each fixed feature, component, outcome and diagnosis; masks/normalizations remain separate.", "release_manifest.json"),
        check("Sparse AC–PCG MDC inference", "passed" if manifest.get("mdc_status") == "complete" else "not yet complete",
              "AC–PCG has 3 AD / 6 Controls: descriptive MDC only, with inferential p-values/FDR unavailable.", "Saved MDC and release audit"),
        check("Prediction validation", "not applicable",
              "This donor-validation extension does not fit prediction models. Prediction null tests and full-experiment completion are not claimed here.", "Scope restriction"),
    ]
    atomic_parquet(pd.DataFrame(checks), review / "technical_checks.parquet")
    entries = []
    # Copy already corrected full-family statistics into small outcome/feature
    # shards, so opening this page never downloads a 59-MiB all-outcome table.
    for rule in manifest.get("completed_rules", []):
        for norm in ("standard_pruned", "retained_edge"):
            source = bundle / "concordance" / rule / f"{norm}.parquet"
            if not source.exists():
                continue
            f = pd.read_parquet(source)
            if f.duplicated(MATCH_KEYS).any():
                raise ValueError("Duplicate replication comparison rows")
            family = ["metric_family", "component", "phenotype", "diagnosis_group"]
            for prefix, p in (("raw", "p_raw"), ("asinh", "p_asinh"), ("rint", "p_rint"), ("spearman", "p_spearman")):
                f["original_family_n_" + prefix] = f.groupby(family, dropna=False)["original_" + p].transform("count")
            for prefix in ("raw", "asinh", "rint", "spearman_raw", "spearman_asinh", "spearman_rint"):
                f["validation_family_n_" + prefix] = f.groupby(family, dropna=False)["validation_p_" + prefix].transform("count")
            for (feature, outcome), group in f.groupby(["metric_family", "phenotype"], sort=True):
                relative = Path("associations") / rule / norm / feature / f"{outcome}.parquet"
                atomic_parquet(group, review / relative)
                entries.append(dict(rule=rule, normalization=norm, feature=feature, outcome=outcome,
                                    path=str(relative), rows=len(group), sha256=sha(review / relative)))
    # All-edge original MDC is available. Effect-filtered original MDC has not
    # been generated in this catalog; leave it explicitly unmatched.
    original = load_mdc_resolved("control_derived")
    original = original[["module", "component", "mdc", "log2_mdc", "directional_fdr"]]
    if original.duplicated(["module", "component"]).any():
        raise ValueError("Original MDC has duplicate module/component rows")
    original = original.rename(columns={c: "original_" + c for c in original if c not in {"module", "component"}})
    path = bundle / "mdc/resolved.parquet"
    if path.exists():
        mdc = pd.read_parquet(path)
        mdc = mdc.rename(columns={c: "validation_" + c for c in ("mdc", "log2_mdc", "directional_fdr")})
        mdc = mdc.merge(original, on=["module", "component"], how="left", validate="many_to_one")
        mask = ~mdc.differential_edge_rule.eq("all")
        mdc.loc[mask, [c for c in mdc if c.startswith("original_")]] = np.nan
        mdc["original_comparison_available"] = mdc.original_mdc.notna()
        atomic_parquet(mdc, review / "mdc_comparison.parquet")
    generated = dict(
        schema_version=1, run_id=root.name, generated_utc=datetime.now(timezone.utc).isoformat(),
        scope="Control-derived brain CAMEL; non-discovery donors within ROSMAP",
        validation_donors=audit["validation_donors"], excluded_complete_donors=audit["excluded_complete_donors"],
        modules=audit["modules"], assignment_rows=audit["assignment_rows"], reference_controls=audit["reference_controls"],
        complete_modules=manifest["complete_modules"], completed_rules=manifest.get("completed_rules", []),
        mdc_status=manifest.get("mdc_status"), source_hashes=audit["source_hashes"],
        association_shards=entries, score_reproduction_max_relative_error=float(r.max_relative_error.max()),
        direct_formula_max_absolute_error=maximum,
        limitations=["Non-discovery donors within ROSMAP, not external-cohort validation.",
                     "Expression preprocessing and module definitions remain fixed.",
                     "Component cohorts overlap; do not sum their donor counts.",
                     "No pooled three-region CT/TS score exists for this cohort.",
                     "Biological replication is not established by formula/score reproduction.",
                     "Original pooled/unclassified association comparisons are unavailable in the saved stratified catalog.",
                     "RINT uses original-cohort interpolation/clipping and can introduce ties.",
                     "Effect masks are frozen discovery masks; never reselected in validation donors."])
    atomic_json(generated, review / "manifest.json")
    for p in review.rglob("*.parquet"):
        import pyarrow.parquet as pq
        if p.stat().st_size >= 95 * 1024**2:
            raise ValueError("Review shard exceeds public size limit")
        if {"donor", "projid", "sample_id"}.intersection(pq.read_schema(p).names):
            raise ValueError("Review page must contain aggregate evidence only")
    return generated


def install(root: Path):
    source = root / "public_bundle/review"
    target = APP / "data/control_derived/camel_validation/review"
    rollback = root / "rollback" / str(time.time_ns()) / "review"
    # Readers see the new registry only after every referenced file is present.
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.name != "manifest.json")
    files.append(source / "manifest.json")
    for p in files:
        relative = p.relative_to(source)
        destination = target / relative
        if destination.exists():
            saved = rollback / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, saved)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        shutil.copy2(p, temporary)
        temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUN)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    generated = build(args.root)
    if args.install:
        install(args.root)
    print(json.dumps({"review_shards": len(generated["association_shards"]), "installed": args.install,
                      "complete_modules": generated["complete_modules"], "mdc_status": generated["mdc_status"]}))


if __name__ == "__main__":
    main()
