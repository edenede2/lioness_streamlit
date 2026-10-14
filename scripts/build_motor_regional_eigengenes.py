#!/usr/bin/env python3
"""Append validated independent motor-tissue eigengenes without replacing brain data.

Uses only the 357 already-public motor-cohort pseudonyms. The private mapping
never enters the bundle. Fits descriptive PCA1, not prediction-fold eigengenes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

APP = Path(__file__).resolve().parents[1]
REPO = APP.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(APP))

from scripts.python.build_regional_module_app_data import (  # noqa: E402
    build_kegg, load_kegg_terms, parquet_entry, rint, sha256,
)
from src.rosmap_multitissue.deep_features import ModuleEigengeneTransformer  # noqa: E402
from app_helpers.regional_modules import MOTOR_SOURCE, MOTOR_TISSUE_ORDER  # noqa: E402
from scripts.build_muscle_public_data import gene_symbol_mapping, replace_overlap_genes  # noqa: E402

SOURCE_LABEL = "Independent motor tissues: matched complete-expression donors"
COHORT = "common_357"


def module_assignments(run_root: Path) -> tuple[pd.DataFrame, dict[str, int], dict[str, str]]:
    manifest = json.loads((run_root / "manifest.json").read_text())
    if manifest["status"] != "complete" or any(manifest["regions"][t]["status"] != "complete" for t in MOTOR_TISSUE_ORDER):
        raise ValueError("All three tissue partitions must be complete")
    pieces, counts, hashes = [], {}, {}
    for tissue in MOTOR_TISSUE_ORDER:
        path = run_root / tissue / f"se2_{tissue.lower()}_signed_alt_table_3.csv"
        details_path = run_root / tissue / f"se2_{tissue.lower()}_signed_alt_details_3.csv"
        frame = pd.read_csv(path).rename(columns={"Cluster ID": "module", "Gene ID": "gene_id"})
        frame["gene_id"] = frame["gene_id"].astype(str).str.split(".").str[0]
        frame["tissue"] = tissue
        if frame["gene_id"].duplicated().any():
            raise ValueError(f"{tissue}: duplicate gene assignments")
        sizes = frame.groupby("module").size()
        details = pd.read_csv(details_path).set_index("Cluster ID")["Cluster Size"]
        if not sizes.sort_index().equals(details.sort_index()) or sizes.min() < 30:
            raise ValueError(f"{tissue}: assignment/detail counts or size cutoff disagree")
        pieces.append(frame[["tissue", "module", "gene_id"]])
        counts[tissue] = len(sizes)
        hashes[tissue] = sha256(path)
    return pd.concat(pieces, ignore_index=True), counts, hashes


def public_cohort(app_root: Path, private_map: Path, partition_manifest: dict) -> tuple[pd.Index, dict[str, str]]:
    metadata = pd.read_parquet(app_root / "data/muscle_control_l4/sample_metadata.parquet")
    mapping = pd.read_csv(private_map, sep="\t", dtype=str)
    if mapping["donor"].duplicated().any() or mapping["sample_id"].duplicated().any():
        raise ValueError("Private pseudonym mapping is not one-to-one")
    mapping = mapping.loc[mapping["sample_id"].isin(metadata["sample_id"])]
    if len(metadata) != 357 or set(mapping["sample_id"]) != set(metadata["sample_id"]):
        raise ValueError("The current 357-donor public motor cohort did not map exactly")
    if not set(mapping["donor"]).issubset(set(map(str, partition_manifest["cohort"]["donor_ids"]))):
        raise ValueError("Public motor donors are not all in the partition discovery cohort")
    return pd.Index(sorted(mapping["donor"]), name="donor"), dict(zip(mapping["donor"], mapping["sample_id"]))


def build_scores(run_root: Path, assignments: pd.DataFrame, donors: pd.Index, sample_map: dict[str, str]):
    manifest = json.loads((run_root / "manifest.json").read_text())
    scores, audits, universes, input_hashes = {}, [], {}, {}
    with threadpool_limits(limits=2):
        for tissue in MOTOR_TISSUE_ORDER:
            path = Path(manifest["inputs"][tissue]["path"])
            frame = pd.read_csv(path, index_col=0)
            frame.index = frame.index.astype(str)
            frame.columns = frame.columns.astype(str).str.split(".").str[0]
            if frame.index.duplicated().any() or frame.columns.duplicated().any():
                raise ValueError(f"{tissue}: duplicate expression donors or genes")
            expression = frame.loc[donors].copy()
            membership = assignments.loc[assignments["tissue"].eq(tissue)]
            if not set(membership["gene_id"]).issubset(expression.columns):
                raise ValueError(f"{tissue}: expression mapping is incomplete")
            universes[tissue] = set(expression.columns)
            input_hashes[tissue] = sha256(path)
            transformer = ModuleEigengeneTransformer(membership, seed=42, tissue_order=(tissue,))
            transformer.fit({tissue: expression}, donors)
            if not transformer.audit_["status"].eq("retained").all():
                raise ValueError(f"{tissue}: unavailable PCA1 module")
            transformed = transformer.transform({tissue: expression}, donors)[tissue]
            rows = []
            for module in sorted(membership["module"].unique()):
                raw = transformed[f"EIG_{tissue}_M{module}"]
                rows.append(pd.DataFrame({
                    "sample_id": donors.map(sample_map), "partition_source": MOTOR_SOURCE,
                    "cohort_scope": COHORT, "score_variant": "eigengene", "network_method": "not_applicable",
                    "tissue": tissue, "module": int(module), "module_key": f"{MOTOR_SOURCE}::{tissue}::M{module}",
                    "metric_raw": raw.to_numpy(), "metric_asinh": np.arcsinh(raw.to_numpy()), "metric_rint": rint(raw),
                }))
            scores[tissue] = pd.concat(rows, ignore_index=True).sort_values(["module", "sample_id"])
            # Independent reconstruction of sampled small/large modules.
            selected = membership.groupby("module").size().sort_values().iloc[[0, -1]].index
            for module in selected:
                fitted = transformer.models_[(tissue, int(module))]
                values = expression[fitted["genes"]].to_numpy()
                standardized = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(values))
                pca = PCA(n_components=1, svd_solver="randomized", random_state=42 + int(module))
                direct = pca.fit_transform(standardized)[:, 0]
                if np.corrcoef(direct, standardized.mean(axis=1))[0, 1] < 0:
                    direct *= -1
                stored = transformed[f"EIG_{tissue}_M{module}"].to_numpy()
                if not np.allclose(direct, stored, atol=1e-8, rtol=1e-8):
                    raise AssertionError("Sampled PCA1 direct reconstruction failed")
            audit = transformer.audit_.copy()
            audit["partition_source"] = MOTOR_SOURCE
            audits.append(audit)
            print(json.dumps({"tissue": tissue, "status": "PCA validated", "modules": len(rows), "rows": len(scores[tissue])}), flush=True)
    return scores, pd.concat(audits), universes, input_hashes


def validate_scores(scores: dict, sample_ids: set[str], counts: dict) -> None:
    for tissue, frame in scores.items():
        if len(frame) != len(sample_ids) * counts[tissue] or frame.duplicated(["sample_id", "module"]).any():
            raise AssertionError("Wrong donor/module grain")
        if set(frame["sample_id"]) != sample_ids or not frame.groupby("module").size().eq(357).all():
            raise AssertionError("Wrong pseudonym or module coverage")
        if {"donor", "projid"}.intersection(frame.columns):
            raise AssertionError("Private identifiers entered the public bundle")
        if not np.isfinite(frame[["metric_raw", "metric_asinh", "metric_rint"]]).all().all():
            raise AssertionError("Nonfinite eigengenes")
        if not np.allclose(frame["metric_asinh"], np.arcsinh(frame["metric_raw"])):
            raise AssertionError("asinh identity failed")


def build(args):
    run_root, app_root, output = args.partition_root.resolve(), args.app_root.resolve(), args.output_root.resolve()
    private = json.loads((run_root / "manifest.json").read_text())
    assignments, counts, hashes = module_assignments(run_root)
    donors, mapping = public_cohort(app_root, args.private_map, private)
    scores, audit, universes, expression_hashes = build_scores(run_root, assignments, donors, mapping)
    validate_scores(scores, set(mapping.values()), counts)
    terms, annotations, kegg_inputs = load_kegg_terms(REPO, universes, min_term_size=10)
    kegg = build_kegg(REPO, MOTOR_SOURCE, assignments, terms, annotations, universes, min_overlap=3)
    symbols = gene_symbol_mapping()
    kegg["overlap_genes"] = kegg["overlap_genes"].map(lambda value: replace_overlap_genes(value, symbols))
    if kegg["overlap_genes"].str.contains("ENSG", na=False).any():
        raise AssertionError("Gene identifiers are not public gene symbols")
    details = assignments.groupby(["tissue", "module"]).size().rename("n_genes").reset_index().merge(
        audit[["tissue", "module", "mapped_genes", "retained_genes", "pc1_variance_explained", "status"]],
        on=["tissue", "module"], validate="one_to_one")
    details["partition_source"] = MOTOR_SOURCE
    details["partition_source_label"] = SOURCE_LABEL
    details["module_key"] = [f"{MOTOR_SOURCE}::{t}::M{m}" for t, m in zip(details.tissue, details.module)]
    details["module_label"] = [f"{t} regional M{m}" for t, m in zip(details.tissue, details.module)]
    details["partition_level_zero_indexed"] = 3
    details["partition_level_description"] = "fourth zero-indexed hierarchy level"
    details["assignment_file_sha256"] = details["tissue"].map(hashes)
    existing = app_root / "data/regional_modules"
    old_details = pd.read_parquet(existing / "module_details.parquet")
    old_kegg = pd.read_parquet(existing / "kegg_enrichment.parquet")
    brain_details = old_details.loc[old_details.partition_source.ne(MOTOR_SOURCE)]
    brain_kegg = old_kegg.loc[old_kegg.partition_source.ne(MOTOR_SOURCE)]
    merged_details = pd.concat([brain_details, details], ignore_index=True)[old_details.columns]
    merged_kegg = pd.concat([brain_kegg, kegg], ignore_index=True)[old_kegg.columns]
    pd.testing.assert_frame_equal(brain_details.reset_index(drop=True), merged_details.loc[merged_details.partition_source.ne(MOTOR_SOURCE)].reset_index(drop=True))
    pd.testing.assert_frame_equal(brain_kegg.reset_index(drop=True), merged_kegg.loc[merged_kegg.partition_source.ne(MOTOR_SOURCE)].reset_index(drop=True))
    manifest = json.loads((existing / "manifest.json").read_text())
    now = datetime.now(timezone.utc).isoformat()
    manifest["sources"] = list(dict.fromkeys([*manifest["sources"], MOTOR_SOURCE]))
    manifest["source_labels"][MOTOR_SOURCE] = SOURCE_LABEL
    manifest["module_counts"][MOTOR_SOURCE] = counts
    manifest["cohort_scopes"][COHORT] = {"label": "Common 357 motor-tissue donors", "donors": 357}
    manifest["completed_combinations"] = [row for row in manifest["completed_combinations"] if row["partition_source"] != MOTOR_SOURCE]
    manifest["source_definitions"] = {**manifest.get("source_definitions", {}), MOTOR_SOURCE: {
        "tissues": list(MOTOR_TISSUE_ORDER), "dataset": "motor", "status": "complete", "supported_scores": ["eigengene"],
        "discovery_donors": private["cohort"]["donor_count"], "analysis_donors": 357,
        "cohort_note": "Seven discovery-cohort donors are outside the existing app motor cohort; no new donor metadata or identifiers were exported.",
        "partition_level_zero_indexed": 3, "partition_manifest_sha256": sha256(run_root / "manifest.json"),
        "assignment_sha256": hashes, "expression_sha256": expression_hashes,
        "kegg_input_sha256": {path.name: sha256(path) for path in kegg_inputs},
        "orientation": "PCA1 aligned to mean standardized member-gene expression; no outcomes used",
        "transforms": ["raw", "asinh", "rint"], "prediction_scores_reused": False,
    }}
    stage = output / "public_bundle/regional_modules"
    stage.mkdir(parents=True, exist_ok=True)
    manifest["score_files"][MOTOR_SOURCE] = {}
    for tissue, frame in scores.items():
        path = stage / MOTOR_SOURCE / COHORT / "eigengene" / f"{tissue}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False, compression="zstd", row_group_size=357)
        manifest["score_files"][MOTOR_SOURCE][tissue] = parquet_entry(path, stage)
        manifest["completed_combinations"].append(dict(partition_source=MOTOR_SOURCE, cohort_scope=COHORT,
            score_variant="eigengene", network_method="not_applicable", tissue=tissue, status="complete",
            modules=counts[tissue], donors=357, rows=len(frame)))
    merged_details.to_parquet(stage / "module_details.parquet", index=False, compression="zstd")
    merged_kegg.to_parquet(stage / "kegg_enrichment.parquet", index=False, compression="zstd")
    manifest["module_details"] = parquet_entry(stage / "module_details.parquet", stage)
    manifest["kegg_enrichment"] = parquet_entry(stage / "kegg_enrichment.parquet", stage)
    manifest["updated_utc"] = now
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    audit.to_parquet(output / "eigengene_audit.parquet", index=False)
    qa = dict(status="pass", modules=counts, rows=sum(len(f) for f in scores.values()), donors=357,
              genes=int(len(assignments)), brain_rows_unchanged=True, tested_direct_pca_modules=6,
              maximum_public_file_bytes=max(p.stat().st_size for p in stage.rglob("*.parquet")))
    (output / "validation.json").write_text(json.dumps(qa, indent=2) + "\n")
    if qa["maximum_public_file_bytes"] >= 95 * 1024**2:
        raise AssertionError("Public shard exceeds 95 MiB")
    if args.publish:
        backup = output / "rollback_regional_metadata"
        backup.mkdir(exist_ok=True)
        for name in ("manifest.json", "module_details.parquet", "kegg_enrichment.parquet"):
            if not (backup / name).exists():
                shutil.copy2(existing / name, backup / name)
        app_manifest_path = app_root / "data/data_manifest.json"
        if not (backup / "data_manifest.json").exists():
            shutil.copy2(app_manifest_path, backup / "data_manifest.json")
        for path in sorted(stage.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                target = existing / path.relative_to(stage)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".partial")
                shutil.copy2(path, temporary)
                temporary.replace(target)
        temporary = existing / "manifest.json.partial"
        shutil.copy2(stage / "manifest.json", temporary)
        temporary.replace(existing / "manifest.json")
        app_manifest = json.loads(app_manifest_path.read_text())
        app_manifest["regional_modules"]["completed_combinations"] = manifest["completed_combinations"]
        app_manifest["regional_modules"]["sources"] = manifest["sources"]
        temporary = app_manifest_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(app_manifest, separators=(",", ":"), sort_keys=True) + "\n")
        temporary.replace(app_manifest_path)
    print(json.dumps(qa), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-root", type=Path, default=REPO / "results/se2_rosmap_single_motor_tissues_signed_alt_complete_case/20261014_161640")
    parser.add_argument("--app-root", type=Path, default=APP)
    parser.add_argument("--private-map", type=Path, default=REPO / "out/muscle_control_l4/private_public_sample_mapping.tsv")
    parser.add_argument("--output-root", type=Path, default=REPO / "out/regional_motor_eigengenes/20261014_table3_common357")
    parser.add_argument("--publish", action="store_true", help="Append validated files to the local app; do not push Git or upload Drive")
    build(parser.parse_args())
