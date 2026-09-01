#!/usr/bin/env python3
"""Build the capability-limited all-donor CorShrink L4 Streamlit bundle.

This builder is intentionally separate from ``build_public_data.py``: the new
module definition has LIONESS/KEGG/MDC data but no BONOBO, differential-edge,
edge-summary, or prediction catalogs.  Existing public module sets are never
rewritten.
"""

from __future__ import annotations

import argparse
import json
import secrets
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import build_public_data as base  # noqa: E402
import build_cluster_association_statistics as cluster_builder  # noqa: E402


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_SET = "all_donor_corrshrink_l4"
METHODS = ("standard", "control_anchored")
EXPECTED_MODULES = 138


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root", type=Path,
        default=REPO_ROOT / "out/lioness_reference_comparison_rosmap/20260901_all_donor_corrshrink_l4_core",
    )
    parser.add_argument(
        "--prepared-root", type=Path,
        default=REPO_ROOT / "out/all_donor_corrshrink_l4_rosmap/20260901_prepared_inputs",
    )
    parser.add_argument(
        "--kegg-root", type=Path,
        default=REPO_ROOT / "out/kegg_enrichment_brain_all_donor_corrshrink_l4_20260901",
    )
    parser.add_argument(
        "--mdc", type=Path,
        default=REPO_ROOT / "out/mdc_all_donor_corrshrink_l4_rosmap/20260901_ad_reference_control_target_200perms/AD_vs_Control_all_donor_corrshrink_l4_MDC_only.tsv",
    )
    parser.add_argument(
        "--resolved-mdc", type=Path,
        default=REPO_ROOT / "out/mdc_resolved_rosmap/20260901_all_donor_corrshrink_l4_ad_reference_control_target_200perms/all_donor_corrshrink_l4/mdc_resolved_ad_vs_control.tsv",
    )
    parser.add_argument(
        "--statistics-root", type=Path,
        default=REPO_ROOT / "out/lioness_app_expansion/20260901_all_donor_corrshrink_l4_all12_statistics",
    )
    parser.add_argument(
        "--expanded-root", type=Path,
        default=REPO_ROOT / "out/lioness_reference_comparison_rosmap/20260901_all_donor_corrshrink_l4_core/expanded_components",
    )
    parser.add_argument(
        "--expanded-statistics-root", type=Path,
        default=REPO_ROOT / "out/lioness_app_expansion/20260901_all_donor_corrshrink_l4_expanded_all12_statistics",
    )
    parser.add_argument("--data-root", type=Path, default=APP_ROOT / "data")
    parser.add_argument(
        "--private-map", type=Path,
        default=REPO_ROOT / "out/all_donor_corrshrink_l4_rosmap/private_public_sample_mapping.tsv",
    )
    parser.add_argument("--publish", action="store_true")
    return parser.parse_args()


def robust_source(root: Path, method: str, suffix: str) -> Path:
    return base.source_file(root, method, suffix)


def recover_existing_map(data_root: Path) -> pd.DataFrame:
    public = pd.read_parquet(data_root / "sample_metadata.parquet")
    private_path = (
        REPO_ROOT
        / "out/lioness_reference_comparison_rosmap"
        / "20260817_standard_control_anchored_allmodules_5phenotypes_6features"
        / "standard/data/phenotypes.parquet"
    )
    private = pd.read_parquet(private_path).rename(columns=base.METADATA_RENAME)
    keys = [
        "diagnosis_group", *base.PHENOTYPES, "age_at_death", "education_years",
        "cogdx", "braak_stage", "cerad_score", "parkinsonism", "adnc",
    ]
    joined = public[["sample_id", *keys]].merge(
        private[["donor", *keys]], on=keys, how="left", validate="one_to_one",
    )
    if joined["donor"].isna().any() or joined["donor"].nunique() != 450:
        raise ValueError("Could not recover the existing 450-donor pseudonym map")
    return joined[["donor", "sample_id"]].assign(
        donor=lambda frame: frame["donor"].astype(str),
        sample_id=lambda frame: frame["sample_id"].astype(str),
    )


def expanded_source(root: Path, method: str) -> Path:
    return root / method / "data/expanded_resolved_component_data.parquet"


def expanded_donor_rows(root: Path) -> pd.DataFrame:
    path = expanded_source(root, "standard")
    parquet = pq.ParquetFile(path)
    first_module = int(
        pq.read_table(path, columns=["module"]).column("module")[0].as_py()
    )
    columns = [
        "donor", "projid", "diagnosis_group", *base.PHENOTYPES,
        *base.METADATA_RENAME, "apoe_genotype", "adnc",
    ]
    rows = pq.read_table(path, filters=[("module", "=", first_module)], columns=columns).to_pandas()
    conflicts = rows.groupby("donor", observed=True)["projid"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("Expanded component rows contain conflicting donor/projid pairs")
    return rows.drop_duplicates("donor").copy()


def build_private_map(
    data_root: Path, expanded_root: Path, private_map_path: Path,
) -> tuple[dict[str, str], pd.DataFrame]:
    existing = recover_existing_map(data_root)
    expanded = expanded_donor_rows(expanded_root)
    donors = set(expanded["donor"].astype(str))
    if private_map_path.exists():
        mapping = pd.read_csv(private_map_path, sep="\t", dtype=str)
    else:
        mapping = existing.copy()
        known = set(mapping["donor"])
        used = set(mapping["sample_id"])
        new_rows = []
        for donor in sorted(donors - known):
            sample_id = f"S-{secrets.token_hex(6)}"
            while sample_id in used:
                sample_id = f"S-{secrets.token_hex(6)}"
            used.add(sample_id)
            new_rows.append({"donor": donor, "sample_id": sample_id})
        mapping = pd.concat([mapping, pd.DataFrame(new_rows)], ignore_index=True)
        private_map_path.parent.mkdir(parents=True, exist_ok=True)
        mapping.to_csv(private_map_path, sep="\t", index=False)
    if mapping["donor"].duplicated().any() or mapping["sample_id"].duplicated().any():
        raise ValueError("Private pseudonym map is not one-to-one")
    recovered = mapping.loc[mapping["donor"].isin(set(existing["donor"]))]
    check = recovered.merge(existing, on="donor", suffixes=("", "_expected"))
    if len(check) != 450 or not check["sample_id"].eq(check["sample_id_expected"]).all():
        raise ValueError("Existing public sample IDs would change")
    missing = donors.difference(mapping["donor"])
    if missing:
        raise ValueError(f"Expanded donors lack pseudonyms: {len(missing)}")
    return dict(zip(mapping["donor"], mapping["sample_id"], strict=True)), expanded


def write_expanded_metadata(
    data_root: Path, output: Path, sample_map: dict[str, str], expanded: pd.DataFrame,
) -> int:
    existing = pd.read_parquet(data_root / "sample_metadata.parquet")
    expanded = expanded.rename(columns=base.METADATA_RENAME).copy()
    expanded["donor"] = expanded["donor"].astype(str)
    expanded["projid"] = expanded["projid"].astype(str)
    clusters = pd.read_csv(
        REPO_ROOT / "data/external/rosmapPhenos/rosmap_combined_phenotypes.csv",
        usecols=["projid", "clusters"], low_memory=False,
    )
    clusters = clusters.loc[clusters["clusters"].notna()].copy()
    clusters["projid"] = clusters["projid"].astype(str)
    if clusters.groupby("projid", observed=True)["clusters"].nunique().gt(1).any():
        raise ValueError("Conflicting ROSMAP cluster assignments")
    expanded = expanded.merge(
        clusters.drop_duplicates("projid"), on="projid", how="left", validate="one_to_one",
    )
    expanded.insert(0, "sample_id", expanded["donor"].map(sample_map))
    expanded = expanded.drop(columns=["donor", "projid"])
    expanded["sex_code"] = expanded["sex_code"].map({0: "Code 0", 1: "Code 1"}).astype("string")
    expanded["apoe_genotype"] = expanded["apoe_genotype"].map(base.format_apoe_genotype).astype("string")
    expanded["parkinsonism_label"] = expanded["parkinsonism"].map({0.0: "No", 1.0: "Yes"}).astype("string")
    expanded["clusters"] = pd.to_numeric(expanded["clusters"], errors="coerce").astype("Int64")
    expanded["ad_control_split"] = pd.NA
    existing_index = existing.set_index("sample_id")
    new_only = expanded.loc[~expanded["sample_id"].isin(existing_index.index)].copy()
    columns = list(existing.columns)
    for column in columns:
        if column not in new_only:
            new_only[column] = pd.NA
    combined = pd.concat([existing, new_only[columns]], ignore_index=True)
    if combined["sample_id"].duplicated().any():
        raise ValueError("Expanded public metadata contains duplicate sample IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False, compression="zstd")
    return len(combined)


def add_entropy(path: Path) -> None:
    details = pd.read_csv(path, sep="\t")
    proportions = details[["proportion_ac", "proportion_dlpfc", "proportion_pcg"]].to_numpy(float)
    logp = np.zeros_like(proportions)
    np.log2(proportions, out=logp, where=proportions > 0)
    details["tissue_entropy"] = -(proportions * logp).sum(axis=1)
    details["tissue_entropy_normalized"] = details["tissue_entropy"] / np.log2(3.0)
    details.to_csv(path, sep="\t", index=False, float_format="%.10g")


def copy_resolved_mdc(source: Path, destination: Path) -> int:
    frame = pd.read_csv(source, sep="\t")
    frame = frame.loc[~frame["component"].isin(["total", "TS", "CT"])].copy()
    frame = base.normalize_public_tissue_labels(frame)
    labels = {
        "TS_AC": "TS: AC", "TS_DLPFC": "TS: DLPFC", "TS_PCGBA23": "TS: PCG",
        "CT_AC__DLPFC": "CT: AC - DLPFC", "CT_AC__PCGBA23": "CT: AC - PCG",
        "CT_DLPFC__PCGBA23": "CT: DLPFC - PCG",
    }
    frame["component_label"] = frame["component"].map(labels)
    if frame["component_label"].isna().any():
        raise ValueError("Unknown resolved MDC component")
    frame.to_csv(destination, sep="\t", index=False)
    return len(frame)


def bh(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.notna()
    if not finite.any():
        return result
    selected = numeric.loc[finite]
    order = np.argsort(selected.to_numpy(), kind="stable")
    ranked = selected.to_numpy()[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    values_out = np.empty(len(ranked), dtype=float)
    values_out[order] = np.clip(adjusted, 0.0, 1.0)
    result.loc[selected.index] = values_out
    return result


def association_long(root: Path) -> pd.DataFrame:
    frames = []
    aggregate = pd.read_parquet(root / "aggregate_statistics.parquet")
    for component in ("CT", "TS"):
        for method, effect, pvalue, scale in (
            ("Spearman", f"rho_{component}", f"p_spearman_{component}", "raw"),
            ("Pearson", f"r_rint_{component}", f"p_rint_{component}", "rint"),
        ):
            part = aggregate[[
                "module", "lioness_method", "metric_family", "phenotype",
                "diagnosis_group", effect, pvalue,
            ]].copy()
            part["component"] = component
            part["correlation_method"] = method
            part["scale"] = scale
            part = part.rename(columns={effect: "effect", pvalue: "p_value"})
            frames.append(part)
    resolved = pd.read_parquet(root / "resolved_statistics.parquet")
    for method, effect, pvalue, scale in (
        ("Spearman", "rho", "p_spearman", "raw"),
        ("Pearson", "r_rint", "p_rint", "rint"),
    ):
        part = resolved[[
            "module", "lioness_method", "metric_family", "component", "phenotype",
            "diagnosis_group", effect, pvalue,
        ]].copy().rename(columns={effect: "effect", pvalue: "p_value"})
        part["correlation_method"] = method
        part["scale"] = scale
        frames.append(part)
    result = pd.concat(frames, ignore_index=True)
    family = [
        "lioness_method", "metric_family", "component", "phenotype",
        "diagnosis_group", "correlation_method", "scale",
    ]
    result["module_set_fdr"] = result.groupby(
        family, observed=True, dropna=False
    )["p_value"].transform(bh)
    return result


def write_association_concordance(
    new_root: Path, data_root: Path, mapping: pd.DataFrame, output: Path,
) -> int:
    pair = mapping.loc[mapping["mapping"].eq("maximum_weight_one_to_one"), [
        "new_module", "matched_module", "jaccard",
    ]].dropna(subset=["matched_module"]).copy()
    pair["matched_module"] = pair["matched_module"].astype(int)
    new = association_long(new_root).rename(columns={
        "module": "new_module", "effect": "new_effect", "p_value": "new_p",
        "module_set_fdr": "new_fdr",
    })
    old = association_long(data_root).rename(columns={
        "module": "matched_module", "effect": "matched_effect", "p_value": "matched_p",
        "module_set_fdr": "matched_fdr",
    })
    keys = [
        "lioness_method", "metric_family", "component", "phenotype",
        "diagnosis_group", "correlation_method", "scale",
    ]
    result = pair.merge(new, on="new_module", how="inner").merge(
        old, on=["matched_module", *keys], how="inner", validate="many_to_one",
    )
    result.to_parquet(output, index=False, compression="zstd")
    return len(result)


def mdc_long(root: Path) -> pd.DataFrame:
    summary = pd.read_csv(root / "mdc_ad_vs_control_summary.tsv", sep="\t")
    aggregate = []
    for component in ("total", "ts", "ct"):
        part = summary[[
            "module", f"mdc_{component}", f"log2_mdc_{component}",
            f"directional_fdr_{component}",
        ]].copy()
        part.columns = ["module", "mdc", "log2_mdc", "directional_fdr"]
        part["component"] = component.upper() if component != "total" else "total"
        aggregate.append(part)
    resolved = pd.read_csv(root / "mdc_resolved_ad_vs_control.tsv", sep="\t")[[
        "module", "component", "mdc", "log2_mdc", "directional_fdr",
    ]]
    return pd.concat([*aggregate, resolved], ignore_index=True)


def write_mdc_concordance(
    new_root: Path, data_root: Path, mapping: pd.DataFrame, output: Path,
) -> int:
    pair = mapping.loc[mapping["mapping"].eq("maximum_weight_one_to_one"), [
        "new_module", "matched_module", "jaccard",
    ]].dropna(subset=["matched_module"]).copy()
    pair["matched_module"] = pair["matched_module"].astype(int)
    new = mdc_long(new_root).rename(columns={
        "module": "new_module", "mdc": "new_mdc", "log2_mdc": "new_log2_mdc",
        "directional_fdr": "new_directional_fdr",
    })
    old = mdc_long(data_root).rename(columns={
        "module": "matched_module", "mdc": "matched_mdc",
        "log2_mdc": "matched_log2_mdc", "directional_fdr": "matched_directional_fdr",
    })
    result = pair.merge(new, on="new_module").merge(
        old, on=["matched_module", "component"], how="inner", validate="many_to_one",
    )
    result.to_parquet(output, index=False, compression="zstd")
    return len(result)


def write_kegg_concordance(
    new_root: Path, data_root: Path, mapping: pd.DataFrame, output: Path,
) -> int:
    pair = mapping.loc[mapping["mapping"].eq("maximum_weight_one_to_one"), [
        "new_module", "matched_module", "jaccard",
    ]].dropna(subset=["matched_module"]).copy()
    pair["matched_module"] = pair["matched_module"].astype(int)
    new = pd.read_csv(new_root / "module_kegg_annotations.tsv", sep="\t").add_prefix("new_")
    old = pd.read_csv(data_root / "module_kegg_annotations.tsv", sep="\t").add_prefix("matched_")
    result = pair.merge(new, on="new_module", how="left").merge(
        old, on="matched_module", how="left", validate="many_to_one",
    )
    result.to_csv(output, sep="\t", index=False)
    return len(result)


def build_comparison(
    prepared_root: Path, output: Path, new_details: pd.DataFrame, data_root: Path,
) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((prepared_root / "prepared_manifest.json").read_text())
    (output / "summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    mapping = pd.read_csv(prepared_root / "partition_module_mapping.tsv", sep="\t")
    old_details = pd.read_csv(data_root / "module_details.tsv", sep="\t")
    mapping = mapping.merge(
        new_details.add_prefix("new_").rename(columns={"new_module": "new_module"}),
        on="new_module", how="left", validate="many_to_one",
    ).merge(
        old_details.add_prefix("matched_").rename(columns={"matched_module": "matched_module"}),
        on="matched_module", how="left", validate="many_to_one",
    )
    mapping.to_csv(output / "module_mapping.tsv", sep="\t", index=False)
    overlap = pd.read_parquet(prepared_root / "partition_overlap_all_pairs.parquet")
    overlap.to_parquet(output / "all_pair_overlap.parquet", index=False, compression="zstd")

    assignments_new = pd.read_csv(prepared_root / "all_donor_corrshrink_m1_l4_assignments.csv")
    assignments_old = pd.read_csv(
        REPO_ROOT / "data/proccessed/se2_filtered/se2_rosmap_full_signed_alt/se2_table_filtered_4.csv"
    ).drop_duplicates(["Tissue", "Gene ID"])
    tissue_rows = []
    for tissue, public in (("AC", "AC"), ("MFBA9BA46", "DLPFC"), ("PCGBA23", "PCG")):
        new_genes = set(assignments_new.loc[assignments_new["Tissue"].eq(tissue), "Gene ID"])
        old_genes = set(assignments_old.loc[assignments_old["Tissue"].eq(tissue), "Gene ID"])
        tissue_rows.append({
            "tissue": public, "new_genes": len(new_genes), "matched_genes": len(old_genes),
            "shared_genes": len(new_genes & old_genes),
            "jaccard": len(new_genes & old_genes) / len(new_genes | old_genes),
        })
    pd.DataFrame(tissue_rows).to_csv(output / "tissue_gene_overlap.tsv", sep="\t", index=False)
    association_rows = write_association_concordance(
        output.parent, data_root, mapping, output / "association_concordance.parquet"
    )
    mdc_rows = write_mdc_concordance(
        output.parent, data_root, mapping, output / "mdc_concordance.parquet"
    )
    kegg_rows = write_kegg_concordance(
        output.parent, data_root, mapping, output / "kegg_concordance.tsv"
    )
    return {
        "mapping_rows": len(mapping), "overlap_rows": len(overlap),
        "association_rows": association_rows, "mdc_rows": mdc_rows,
        "kegg_rows": kegg_rows,
    }


def build_parent_cluster_statistics(
    data_root: Path, new_root: Path, output: Path,
) -> int:
    metadata = pd.read_parquet(data_root / "sample_metadata.parquet")[[
        "sample_id", "diagnosis_group", "clusters",
    ]]
    existing = pd.read_parquet(data_root / "cluster_association_statistics.parquet")
    existing = existing.loc[~existing["module_set"].eq(MODULE_SET)].copy()
    new_frames = [
        cluster_builder._statistics_for_file(
            new_root / f"{resolution}_plot_data.parquet", metadata,
            module_set=MODULE_SET, estimator="lioness", edge_rule="all",
            resolution=resolution,
        )
        for resolution in ("aggregate", "resolved")
    ]
    result = pd.concat([existing, *new_frames], ignore_index=True)
    result["clusters_are_nominal"] = True
    result["numeric_correlation_used"] = False
    result["effect_definition"] = "max(0,(Kruskal H-k+1)/(n-k))"
    result.to_parquet(output, index=False, compression="zstd", row_group_size=50_000)
    if output.stat().st_size >= 95 * 1024 * 1024:
        raise ValueError("Cluster association hot cache exceeds 95 MiB")
    return len(result)


def publish_staging(staging: Path, destination: Path) -> Path | None:
    rollback = None
    if destination.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rollback = REPO_ROOT / "out/all_donor_corrshrink_l4_rosmap/rollback" / stamp
        rollback.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(destination), str(rollback))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging), str(destination))
    return rollback


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    audit_path = (
        REPO_ROOT / "out/corshrink_m1_all_donors_audit"
        / "20260901_extended_fit_5000/audit_manifest.json"
    )
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "complete" or audit.get("decision") != "accept_existing_partition":
        raise RuntimeError("The CorShrink audit has not accepted the published partition")
    staging = (
        REPO_ROOT / "out/all_donor_corrshrink_l4_rosmap/public_bundle_staging"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    staging.mkdir(parents=True, exist_ok=False)
    assignments = args.prepared_root / "all_donor_corrshrink_m1_l4_assignments.csv"
    details_source = args.prepared_root / "all_donor_corrshrink_m1_l4_details.csv"
    sample_map, expanded_donors = build_private_map(
        data_root, args.expanded_root.resolve(), args.private_map.resolve()
    )

    aggregate_sources = [
        robust_source(args.analysis_root, method, "_transformed_component_data.parquet")
        for method in METHODS
    ]
    resolved_sources = [
        robust_source(args.analysis_root, method, "_tissue_resolved_component_data.parquet")
        for method in METHODS
    ]
    aggregate_rows = base.write_sanitized_plot_data(
        aggregate_sources, staging / "aggregate_plot_data.parquet", sample_map, False,
    )
    resolved_rows = base.write_sanitized_plot_data(
        resolved_sources, staging / "resolved_plot_data.parquet", sample_map, True,
    )
    aggregate_stat_rows = base.write_combined_statistics(
        [args.statistics_root / MODULE_SET / method / "aggregate_statistics.parquet" for method in METHODS],
        staging / "aggregate_statistics.parquet",
    )
    resolved_stat_rows = base.write_combined_statistics(
        [args.statistics_root / MODULE_SET / method / "resolved_statistics.parquet" for method in METHODS],
        staging / "resolved_statistics.parquet", resolved=True,
    )

    expanded_output = staging / "expanded"
    expanded_output.mkdir()
    expanded_rows_by_method = {}
    for method in METHODS:
        method_output = expanded_output / method
        method_output.mkdir()
        expanded_rows_by_method[method] = base.write_sanitized_plot_data(
            [expanded_source(args.expanded_root, method)],
            method_output / "resolved_plot_data.parquet", sample_map, True,
        )
    expanded_rows = sum(expanded_rows_by_method.values())
    expanded_stat_rows = base.write_combined_statistics(
        [args.expanded_statistics_root / MODULE_SET / method / "resolved_statistics.parquet" for method in METHODS],
        expanded_output / "resolved_statistics.parquet", resolved=True,
    )
    expanded_metadata_rows = write_expanded_metadata(
        data_root, expanded_output / "sample_metadata.parquet", sample_map, expanded_donors,
    )
    expanded_metadata_public = pd.read_parquet(
        expanded_output / "sample_metadata.parquet"
    )

    kegg, kegg_audit = base.write_public_kegg(
        args.kegg_root / "method4_tissue_expanded_kegg_annotated.tsv",
        args.kegg_root / "method2_meta_kegg_annotated.tsv", staging,
    )
    annotation_source = base.table_file(
        args.analysis_root, "standard", "_module_kegg_annotations.tsv"
    )
    annotations = pd.read_csv(annotation_source, sep="\t")
    annotations.to_csv(staging / "module_kegg_annotations.tsv", sep="\t", index=False)
    modules = set(pd.to_numeric(annotations["module"], errors="raise").astype(int))
    if len(modules) != EXPECTED_MODULES:
        raise ValueError(f"Expected {EXPECTED_MODULES} annotation modules; found {len(modules)}")
    base.write_module_details(details_source, staging / "module_details.tsv", modules)
    add_entropy(staging / "module_details.tsv")
    assignment_rows = base.validate_assignment_details(
        assignments, staging / "module_details.tsv", modules,
    )
    mdc_rows = base.write_mdc_summary(args.mdc, staging / "mdc_ad_vs_control_summary.tsv", modules)
    resolved_mdc_rows = copy_resolved_mdc(args.resolved_mdc, staging / "mdc_resolved_ad_vs_control.tsv")
    comparison = build_comparison(
        args.prepared_root, staging / "partition_comparison",
        pd.read_csv(staging / "module_details.tsv", sep="\t"), data_root,
    )
    parent_cluster_path = staging / "_parent_cluster_association_statistics.parquet"
    parent_cluster_rows = build_parent_cluster_statistics(
        data_root, staging, parent_cluster_path,
    )

    expected = {
        "aggregate_plot_rows": EXPECTED_MODULES * 450 * 6 * len(METHODS),
        "resolved_plot_rows": EXPECTED_MODULES * 450 * 6 * 6 * len(METHODS),
        "aggregate_stat_rows": EXPECTED_MODULES * 6 * 12 * 3 * len(METHODS),
        "resolved_stat_rows": EXPECTED_MODULES * 6 * 6 * 12 * 3 * len(METHODS),
    }
    observed = {
        "aggregate_plot_rows": aggregate_rows, "resolved_plot_rows": resolved_rows,
        "aggregate_stat_rows": aggregate_stat_rows, "resolved_stat_rows": resolved_stat_rows,
    }
    if observed != expected:
        raise ValueError(f"Primary public row counts differ: {observed} != {expected}")
    if assignment_rows != 43754 or mdc_rows != EXPECTED_MODULES or resolved_mdc_rows != EXPECTED_MODULES * 6:
        raise ValueError("Assignment or MDC public row counts differ from expectations")
    for path in staging.rglob("*"):
        if path.is_file() and path.stat().st_size >= 95 * 1024 * 1024:
            raise ValueError(f"Public file exceeds 95 MiB: {path}")
    forbidden = {"donor", "projid"}
    for path in staging.rglob("*.parquet"):
        if forbidden.intersection(pq.ParquetFile(path).schema_arrow.names):
            raise ValueError(f"Private identifier in {path}")

    module_manifest = {
        "status": "complete", "label": "All-donor CorShrink M1 L4 modules",
        "modules": EXPECTED_MODULES, "assignments": assignment_rows,
        "methods": list(METHODS), "estimators": ["lioness"],
        "cohorts": {
            "complete_450": {"donors": 450, "Control": 164, "MCI": 119, "AD": 167},
            "maximum_component": {
                "metadata_donors": expanded_metadata_rows,
                "diagnosis_counts": {
                    str(key): int(value) for key, value in
                    expanded_metadata_public["diagnosis_group"].value_counts(
                        dropna=False
                    ).items()
                },
                "cluster_counts": {
                    str(int(key)): int(value) for key, value in
                    expanded_metadata_public["clusters"].value_counts().sort_index().items()
                },
                "component_counts": {"AC": 730, "DLPFC": 1216, "PCG": 659,
                                     "AC_DLPFC": 694, "AC_PCG": 478, "DLPFC_PCG": 640},
            },
        },
        "capabilities": {
            "lioness": True, "associations": True, "kegg": True, "mdc": True,
            "module_details": True, "partition_comparison": True,
            "expanded_components": True, "prediction": False, "bonobo": False,
            "differential_edges": False, "edge_summaries": False,
        },
        "row_counts": {**observed, "expanded_plot_rows": expanded_rows,
                       "expanded_stat_rows": expanded_stat_rows},
        "kegg": kegg_audit, "comparison": comparison,
        "provenance": {
            "prepared_manifest_sha256": base.sha256(
                args.prepared_root / "prepared_manifest.json"
            ),
            "prepared_manifest": json.loads(
                (args.prepared_root / "prepared_manifest.json").read_text()
            ),
            "corshrink_audit": audit,
            "biological_level": 4,
            "zero_indexed_source_level": 3,
        },
        "files": base.dataset_file_manifest(staging),
    }
    module_manifest["recursive_files"] = {
        path.relative_to(staging).as_posix(): base.deploy_file_manifest_entry(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file() and not path.name.startswith("_parent_")
    }
    manifest_path = data_root / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("module_sets", {})[MODULE_SET] = module_manifest
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    (staging / "module_set_manifest.json").write_text(
        json.dumps(module_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (staging / "parent_manifest_patch.json").write_text(
        json.dumps({MODULE_SET: module_manifest}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rollback = None
    if args.publish:
        destination = data_root / MODULE_SET
        rollback = publish_staging(staging, destination)
        staged_cluster = destination / parent_cluster_path.name
        current_cluster = data_root / "cluster_association_statistics.parquet"
        cluster_rollback = (
            REPO_ROOT / "out/all_donor_corrshrink_l4_rosmap/rollback"
            / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            / current_cluster.name
        )
        cluster_rollback.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current_cluster, cluster_rollback)
        staged_cluster.replace(current_cluster)
        manifest.setdefault("files", {})[current_cluster.name] = (
            base.deploy_file_manifest_entry(current_cluster)
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({
        "status": "published" if args.publish else "staged", "staging": str(staging),
        "rollback": str(rollback) if rollback else None, "rows": observed,
        "expanded_plot_rows": expanded_rows, "expanded_stat_rows": expanded_stat_rows,
        "kegg_rows": len(kegg), "parent_cluster_rows": parent_cluster_rows,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
