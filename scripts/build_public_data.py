#!/usr/bin/env python3
"""Build the deidentified, GitHub-sized data bundle used by the Streamlit app.

The source Parquet files contain ROSMAP donor and projid fields. This script never
writes either identifier to the deploy bundle. It creates a random salted HMAC
label that is stable only within one generated bundle and discards the salt.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


RUN_NAME = "20260817_standard_control_anchored_allmodules_5phenotypes_6features"
CONTROL_DERIVED_RUN_NAME = (
    "20260818_control_anchored_control_derived_l4_5phenotypes_6features"
)
METHODS = ("standard", "control_anchored")
PHENOTYPES = (
    "cogn_global",
    "cogng_demog_slope",
    "cogng_path_slope",
    "motor10_demog_slope",
    "sqrt_parksc_demog_slope",
)
AGGREGATE_VALUE_COLUMNS = (
    "CT_raw",
    "TS_raw",
    "CT_asinh",
    "TS_asinh",
    "CT_rint",
    "TS_rint",
)
RESOLVED_VALUE_COLUMNS = ("metric_raw", "metric_asinh", "metric_rint")

METADATA_RENAME = {
    "msex.x": "sex_code",
    "age_death.x": "age_at_death",
    "educ.x": "education_years",
    "cogdx.y": "cogdx",
    "braaksc": "braak_stage",
    "ceradsc": "cerad_score",
    "parkinsonism_yn_lv": "parkinsonism",
}

MDC_OUTPUT_NAME = "mdc_ad_vs_control_summary.tsv"
MODULE_DETAILS_OUTPUT_NAME = "module_details.tsv"
MDC_SOURCE_METADATA = {
    "comparison": "AD reference / Control target",
    "reference_group": "AD",
    "target_group": "Control",
    "reference_assembled_donors": 517,
    "target_assembled_donors": 408,
    "reference_complete_three_tissue_donors": 167,
    "target_complete_three_tissue_donors": 164,
    "mci_included": False,
    "sample_permutations": 200,
    "gene_permutations": 200,
    "beta_ts": 3,
    "beta_ct": 2,
    "adjacency": "signedAlt",
    "fdr_definition": (
        "Directional FDR is the maximum of the Benjamini-Hochberg-adjusted "
        "sample-permutation and gene-permutation p-values for the observed direction."
    ),
}


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    label: str
    analysis_root: Path
    methods: tuple[str, ...]
    module_count: int
    sentinel_module: int
    kegg: Path
    kegg_per_tissue: Path
    mdc: Path
    module_details: Path
    assignments: Path
    output: Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=repo_root / "out" / "lioness_reference_comparison_rosmap" / RUN_NAME,
        help="Completed standard/control-referenced analysis directory.",
    )
    parser.add_argument(
        "--control-derived-analysis-root",
        type=Path,
        default=repo_root
        / "out"
        / "lioness_reference_comparison_rosmap"
        / CONTROL_DERIVED_RUN_NAME,
        help="Completed Control-derived, Control-referenced app-data analysis directory.",
    )
    parser.add_argument(
        "--kegg",
        type=Path,
        default=repo_root
        / "out"
        / "kegg_enrichments_per_se2_level4"
        / "method4_tissue_expanded_kegg_annotated.tsv",
        help="Full annotated tissue-expanded KEGG table.",
    )
    parser.add_argument(
        "--control-derived-kegg",
        type=Path,
        default=repo_root
        / "out"
        / "kegg_enrichment_brain_control_derived_se2_l4_method4_20260818"
        / "method4_tissue_expanded_kegg_annotated.tsv",
        help="Control-derived tissue-expanded KEGG table.",
    )
    parser.add_argument(
        "--kegg-per-tissue",
        type=Path,
        default=repo_root
        / "out"
        / "kegg_enrichments_per_se2_level4"
        / "method2_meta_kegg_annotated.tsv",
        help="Full-cohort per-tissue KEGG p-value and FDR table.",
    )
    parser.add_argument(
        "--control-derived-kegg-per-tissue",
        type=Path,
        default=repo_root
        / "out"
        / "kegg_enrichment_brain_control_derived_se2_l4_method4_20260818"
        / "method2_meta_kegg_annotated.tsv",
        help="Control-derived per-tissue KEGG p-value and FDR table.",
    )
    parser.add_argument(
        "--mdc",
        type=Path,
        default=repo_root
        / "MDC_Preservation_signedAlt_AD_Control_reverse2"
        / "AD_vs_Control"
        / "AD_vs_Control_MDC_Preservation.tsv",
        help="Latest completed AD-reference versus Control-target MDC table.",
    )
    parser.add_argument(
        "--control-derived-mdc",
        type=Path,
        default=repo_root
        / "out"
        / "mdc_control_derived_l4_rosmap"
        / "20260818_ad_reference_control_target_200perms"
        / "AD_vs_Control_control_derived_l4_MDC_only.tsv",
        help="Control-derived AD-reference versus Control-target MDC table.",
    )
    parser.add_argument(
        "--module-details",
        type=Path,
        default=repo_root
        / "data"
        / "proccessed"
        / "se2_filtered"
        / "se2_rosmap_full_signed_alt"
        / "se2_details_filtered_4.csv",
        help="Level-4 SE2 module details table.",
    )
    parser.add_argument(
        "--full-assignments",
        type=Path,
        default=repo_root
        / "data"
        / "proccessed"
        / "se2_filtered"
        / "se2_rosmap_full_signed_alt"
        / "se2_table_filtered_4.csv",
        help="Full-cohort level-4 tissue-gene assignments.",
    )
    parser.add_argument(
        "--control-derived-module-details",
        type=Path,
        default=repo_root
        / "data"
        / "proccessed"
        / "se2_signed_alt_control_filtered"
        / "se2_rosmap_control_signed-alt"
        / "speakeasy_clusters_details_level_4_filtered.csv",
        help="Control-derived level-4 module details.",
    )
    parser.add_argument(
        "--control-derived-assignments",
        type=Path,
        default=repo_root
        / "data"
        / "proccessed"
        / "se2_signed_alt_control_filtered"
        / "se2_rosmap_control_signed-alt"
        / "speakeasy_clusters_table_level_4_filtered.csv",
        help="Control-derived level-4 tissue-gene assignments.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Streamlit deploy data directory.",
    )
    parser.add_argument(
        "--bonobo-app-data-root",
        type=Path,
        default=repo_root
        / "out/bonobo_app_rosmap/20260821_bonobo_all450_full_and_control_derived/app_data",
    )
    parser.add_argument(
        "--bonobo-network-root",
        type=Path,
        default=repo_root
        / "out/bonobo_app_rosmap/20260821_bonobo_all450_full_and_control_derived",
    )
    parser.add_argument(
        "--lioness-expansion-root",
        type=Path,
        default=repo_root
        / "out/lioness_app_expansion/20260821_lioness_edge_summaries_entropy",
    )
    parser.add_argument(
        "--lioness-all12-stats-root",
        type=Path,
        default=repo_root
        / "out/lioness_app_expansion/20260821_lioness_all12_robust_statistics",
    )
    parser.add_argument(
        "--resolved-mdc-root",
        type=Path,
        default=repo_root
        / "out/mdc_resolved_rosmap/20260821_ad_reference_control_target_200perms",
    )
    parser.add_argument(
        "--differential-analysis-root",
        type=Path,
        default=repo_root
        / "out/lioness_app_expansion/20260822_ad_control_differential_edges_all_networks",
        help="Completed upstream AD-Control differential-edge analysis.",
    )
    parser.add_argument(
        "--differential-app-data-root",
        type=Path,
        default=repo_root
        / "out/lioness_app_expansion/20260822_ad_control_differential_edges_all_networks/app_data",
        help="Transformed differential-edge app data.",
    )
    parser.add_argument(
        "--mdc-only",
        action="store_true",
        help="Refresh only the module-level MDC summary and existing data manifest.",
    )
    parser.add_argument(
        "--module-details-only",
        action="store_true",
        help="Refresh only the module details table and existing data manifest.",
    )
    parser.add_argument(
        "--statistics-only",
        action="store_true",
        help="Refresh only aggregate and tissue-resolved statistics and the data manifest.",
    )
    parser.add_argument(
        "--kegg-only",
        action="store_true",
        help=(
            "Refresh both public KEGG tables and manifest entries without changing "
            "donor plot data or pseudonymous sample labels."
        ),
    )
    return parser.parse_args()


def source_file(
    data_dir: Path, method: str, suffix: str, exclude_text: str | None = None
) -> Path:
    matches = sorted((data_dir / method / "data").glob(f"*{suffix}"))
    if exclude_text:
        matches = [path for path in matches if exclude_text not in path.name]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {method} file ending in {suffix!r}; found {matches}"
        )
    return matches[0]


def table_file(data_dir: Path, method: str, suffix: str) -> Path:
    matches = sorted((data_dir / method / "tables").glob(f"*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {method} table ending in {suffix!r}; found {matches}"
        )
    return matches[0]


def create_sample_map(aggregate_source: Path) -> dict[str, str]:
    donor_table = pq.read_table(aggregate_source, columns=["donor"])
    donors = sorted({str(value) for value in donor_table.column("donor").to_pylist()})
    if len(donors) != 450:
        raise ValueError(f"Expected 450 donors, found {len(donors)}")
    salt = secrets.token_bytes(32)
    return {
        donor: "S-" + hmac.new(salt, donor.encode("utf-8"), hashlib.sha256).hexdigest()[:12]
        for donor in donors
    }


def format_apoe_genotype(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    code = str(int(float(value)))
    if len(code) != 2:
        return f"Code {code}"
    return f"ε{code[0]}/ε{code[1]}"


def write_sample_metadata(
    phenotype_source: Path,
    output: Path,
    sample_map: dict[str, str],
    split_source: Path | None = None,
) -> int:
    source_columns = [
        "donor",
        "diagnosis_group",
        *PHENOTYPES,
        *METADATA_RENAME,
        "apoe_genotype",
        "adnc",
    ]
    frame = pd.read_parquet(phenotype_source, columns=source_columns)
    frame["donor"] = frame["donor"].astype(str)
    if split_source is not None:
        split = pd.read_csv(split_source, sep="\t", dtype={"donor": str})
        frame = frame.merge(
            split[["donor", "ad_control_split"]],
            on="donor",
            how="left",
            validate="one_to_one",
        )
        if frame["ad_control_split"].isna().any():
            raise ValueError("Public AD-Control split mapping failed")
    frame.insert(0, "sample_id", frame.pop("donor").map(sample_map))
    if frame["sample_id"].isna().any() or frame["sample_id"].nunique() != 450:
        raise ValueError("Sample metadata did not map cleanly to all 450 pseudonyms")
    frame = frame.rename(columns=METADATA_RENAME)
    frame["sex_code"] = frame["sex_code"].map({0: "Code 0", 1: "Code 1"}).astype("string")
    frame["apoe_genotype"] = frame["apoe_genotype"].map(format_apoe_genotype).astype("string")
    frame["parkinsonism_label"] = frame["parkinsonism"].map(
        {0.0: "No", 1.0: "Yes"}
    ).astype("string")
    frame["age_at_death"] = pd.to_numeric(frame["age_at_death"], errors="coerce")
    numeric_columns = [
        *PHENOTYPES,
        "age_at_death",
        "education_years",
        "cogdx",
        "braak_stage",
        "cerad_score",
        "adnc",
        "parkinsonism",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    frame["sample_id"] = frame["sample_id"].astype("string")
    frame["diagnosis_group"] = frame["diagnosis_group"].astype("string")
    frame.to_parquet(
        output, index=False, compression="zstd", row_group_size=50_000
    )
    return len(frame)


def normalize_batch(
    batch: pa.RecordBatch,
    sample_map: dict[str, str],
    value_columns: tuple[str, ...],
    resolved: bool,
) -> pd.DataFrame:
    frame = batch.to_pandas()
    frame.insert(0, "sample_id", frame.pop("donor").astype(str).map(sample_map))
    if frame["sample_id"].isna().any():
        raise ValueError("A source donor was absent from the pseudonym map")
    frame["sample_id"] = frame["sample_id"].astype("string")
    frame["module"] = pd.to_numeric(frame["module"], errors="raise").astype("int32")
    string_columns = ["metric_family", "diagnosis_group", "lioness_method"]
    if resolved:
        string_columns += ["component", "component_class", "component_label"]
    for column in string_columns:
        frame[column] = frame[column].astype("string")
    if resolved:
        frame = normalize_resolved_component_labels(frame)
    for column in (*value_columns, *PHENOTYPES):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame


def normalize_resolved_component_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the public DLPFC name to resolved component keys and labels."""
    result = frame.copy()
    if "component" in result:
        result["component"] = result["component"].astype("string").str.replace(
            "MFBA9BA46", "DLPFC", regex=False
        )
    if "component_label" in result:
        result["component_label"] = (
            result["component_label"]
            .astype("string")
            .str.replace("DLPFC (MFBA9/BA46)", "DLPFC", regex=False)
            .str.replace("MFBA9/BA46", "DLPFC", regex=False)
        )
    return result


def write_sanitized_plot_data(
    sources: list[Path],
    output: Path,
    sample_map: dict[str, str],
    resolved: bool,
) -> int:
    value_columns = RESOLVED_VALUE_COLUMNS if resolved else AGGREGATE_VALUE_COLUMNS
    source_columns = ["donor", "module", "metric_family"]
    if resolved:
        source_columns += ["component", "component_class", "component_label"]
    source_columns += [*value_columns, "diagnosis_group", *PHENOTYPES, "lioness_method"]

    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for source in sources:
            parquet = pq.ParquetFile(source)
            for batch in parquet.iter_batches(batch_size=150_000, columns=source_columns):
                frame = normalize_batch(batch, sample_map, value_columns, resolved)
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        output,
                        table.schema,
                        compression="zstd",
                        compression_level=9,
                        use_dictionary=True,
                    )
                writer.write_table(table, row_group_size=50_000)
                rows += len(frame)
    finally:
        if writer is not None:
            writer.close()
    return rows


def write_combined_statistics(
    sources: list[Path], output: Path, resolved: bool = False
) -> int:
    frames = [pd.read_parquet(path) for path in sources]
    frame = pd.concat(frames, ignore_index=True)
    frame["module"] = pd.to_numeric(frame["module"], errors="raise").astype("int32")
    if resolved:
        frame = normalize_resolved_component_labels(frame)
    frame.to_parquet(
        output, index=False, compression="zstd", row_group_size=50_000
    )
    return len(frame)


def normalize_public_tissue_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace every internal DLPFC/PCG token in string columns."""

    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = (
            result[column]
            .astype("string")
            .str.replace("MFBA9BA46", "DLPFC", regex=False)
            .str.replace("MFBA9/BA46", "DLPFC", regex=False)
            .str.replace("PCGBA23", "PCGBA23", regex=False)
        )
    return result


def write_sanitized_edge_data(
    source: Path,
    output: Path,
    sample_map: dict[str, str],
    edge_rule: str | None = None,
) -> int:
    filters = [("edge_rule", "=", edge_rule)] if edge_rule is not None else None
    frame = pq.read_table(source, filters=filters).to_pandas()
    if "donor" not in frame:
        raise ValueError(f"Edge-summary source has no donor column: {source}")
    frame.insert(0, "sample_id", frame.pop("donor").astype(str).map(sample_map))
    if frame["sample_id"].isna().any():
        raise ValueError(f"Edge-summary donor mapping failed: {source}")
    frame = frame.drop(columns=["projid"], errors="ignore")
    frame = normalize_public_tissue_labels(frame)
    frame = frame.sort_values(
        ["module", "sample_id", "scope"], kind="stable"
    ).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(
        output,
        index=False,
        compression="zstd",
        row_group_size=450 * 9,
    )
    return len(frame)


def write_expanded_module_bundle(
    config: DatasetConfig,
    sample_map: dict[str, str],
    *,
    bonobo_app_data_root: Path,
    bonobo_network_root: Path,
    lioness_expansion_root: Path,
    lioness_all12_stats_root: Path,
    resolved_mdc_root: Path,
) -> dict[str, object]:
    """Add BONOBO, all-12 stats, entropy, resolved MDC, and edge summaries."""

    # Replace the legacy five-outcome statistics with all-12-outcome tables.
    aggregate_stats = []
    resolved_stats = []
    for method in config.methods:
        source_dir = lioness_all12_stats_root / config.key / method
        aggregate_stats.append(source_dir / "aggregate_statistics.parquet")
        resolved_stats.append(source_dir / "resolved_statistics.parquet")
    aggregate_stat_rows = write_combined_statistics(
        aggregate_stats, config.output / "aggregate_statistics.parquet"
    )
    resolved_stat_rows = write_combined_statistics(
        resolved_stats, config.output / "resolved_statistics.parquet", resolved=True
    )

    bonobo_rows: dict[str, dict[str, int]] = {}
    for edge_rule in ("all", "native_p05", "bh_fdr05"):
        source_dir = bonobo_app_data_root / config.key / "bonobo" / edge_rule
        output_dir = config.output / "bonobo" / edge_rule
        output_dir.mkdir(parents=True, exist_ok=True)
        aggregate_rows = write_sanitized_plot_data(
            [source_dir / "aggregate_plot_data.parquet"],
            output_dir / "aggregate_plot_data.parquet",
            sample_map,
            resolved=False,
        )
        resolved_rows = write_sanitized_plot_data(
            [source_dir / "resolved_plot_data.parquet"],
            output_dir / "resolved_plot_data.parquet",
            sample_map,
            resolved=True,
        )
        bonobo_aggregate_stats = write_combined_statistics(
            [source_dir / "aggregate_statistics.parquet"],
            output_dir / "aggregate_statistics.parquet",
        )
        bonobo_resolved_stats = write_combined_statistics(
            [source_dir / "resolved_statistics.parquet"],
            output_dir / "resolved_statistics.parquet",
            resolved=True,
        )
        bonobo_rows[edge_rule] = {
            "aggregate_plot_rows": aggregate_rows,
            "resolved_plot_rows": resolved_rows,
            "aggregate_stat_rows": bonobo_aggregate_stats,
            "resolved_stat_rows": bonobo_resolved_stats,
        }

    edge_output = config.output / "edge_summaries"
    edge_output.mkdir(parents=True, exist_ok=True)
    lioness_edge_rows = {}
    for method in config.methods:
        source = (
            lioness_expansion_root
            / config.key
            / f"lioness_edge_summaries__{method}.parquet"
        )
        lioness_edge_rows[method] = write_sanitized_edge_data(
            source,
            edge_output / f"lioness__{method}.parquet",
            sample_map,
        )
    bonobo_edge_source = (
        bonobo_network_root / config.key / "bonobo_edge_summaries.parquet"
    )
    bonobo_edge_rows = {
        edge_rule: write_sanitized_edge_data(
            bonobo_edge_source,
            edge_output / f"bonobo__{edge_rule}.parquet",
            sample_map,
            edge_rule=edge_rule,
        )
        for edge_rule in ("all", "native_p05", "bh_fdr05")
    }

    entropy = pd.read_csv(
        lioness_expansion_root / config.key / "module_tissue_entropy.tsv", sep="\t"
    )[["module", "tissue_entropy", "tissue_entropy_normalized"]]
    details_path = config.output / MODULE_DETAILS_OUTPUT_NAME
    details = pd.read_csv(details_path, sep="\t")
    details = details.merge(entropy, on="module", how="left", validate="one_to_one")
    if details[["tissue_entropy", "tissue_entropy_normalized"]].isna().any().any():
        raise ValueError(f"{config.key}: entropy merge failed")
    details.to_csv(details_path, sep="\t", index=False, float_format="%.10g")

    resolved_mdc = pd.read_csv(
        resolved_mdc_root / config.key / "mdc_resolved_ad_vs_control.tsv", sep="\t"
    )
    resolved_mdc = resolved_mdc.loc[
        ~resolved_mdc["component"].isin(["total", "TS", "CT"])
    ].copy()
    resolved_mdc = normalize_public_tissue_labels(resolved_mdc)
    component_labels = {
        "TS_AC": "TS: AC",
        "TS_DLPFC": "TS: DLPFC",
        "TS_PCGBA23": "TS: PCG",
        "CT_AC__DLPFC": "CT: AC - DLPFC",
        "CT_AC__PCGBA23": "CT: AC - PCG",
        "CT_DLPFC__PCGBA23": "CT: DLPFC - PCG",
    }
    resolved_mdc["component_label"] = resolved_mdc["component"].map(component_labels)
    if resolved_mdc["component_label"].isna().any():
        raise ValueError(f"{config.key}: unknown resolved MDC component")
    resolved_mdc.to_csv(
        config.output / "mdc_resolved_ad_vs_control.tsv",
        sep="\t", index=False,
    )
    expected_bonobo = {
        "aggregate_plot_rows": config.module_count * 450 * 3,
        "resolved_plot_rows": config.module_count * 450 * 3 * 6,
        "aggregate_stat_rows": config.module_count * 3 * 12 * 3,
        "resolved_stat_rows": config.module_count * 3 * 6 * 12 * 3,
    }
    for edge_rule, rows in bonobo_rows.items():
        if rows != expected_bonobo:
            raise ValueError(
                f"{config.key}/{edge_rule}: BONOBO row mismatch {rows} != {expected_bonobo}"
            )
    return {
        "outcomes": 12,
        "lioness_aggregate_stat_rows": aggregate_stat_rows,
        "lioness_resolved_stat_rows": resolved_stat_rows,
        "bonobo": bonobo_rows,
        "edge_summaries": {
            "lioness": lioness_edge_rows,
            "bonobo": bonobo_edge_rows,
        },
        "resolved_mdc_rows": len(resolved_mdc),
        "entropy_rows": len(entropy),
    }


def _write_public_edge_dataset(
    sources: list[Path],
    output_directory: Path,
    *,
    rows_per_file: int = 100_000,
) -> int:
    """Write identifier-free edge tables as bounded Parquet dataset parts."""

    output_directory.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    part = 0
    for source in sources:
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(batch_size=rows_per_file):
            frame = normalize_public_tissue_labels(batch.to_pandas())
            forbidden = {"donor", "projid"}.intersection(frame.columns)
            if forbidden:
                raise ValueError(f"Public edge dataset contains identifiers: {forbidden}")
            destination = output_directory / f"part-{part:05d}.parquet"
            frame.to_parquet(
                destination,
                index=False,
                compression="zstd",
                row_group_size=min(rows_per_file, len(frame)),
            )
            if destination.stat().st_size >= 95 * 1024 * 1024:
                raise ValueError(f"Differential edge part exceeds 95 MiB: {destination}")
            total_rows += len(frame)
            part += 1
    return total_rows


def write_differential_module_bundle(
    config: DatasetConfig,
    sample_map: dict[str, str],
    *,
    analysis_root: Path,
    app_data_root: Path,
) -> dict[str, object]:
    """Package filtered scores, edge summaries, and scalable volcano data."""

    output_root = config.output / "differential"
    method_catalog = {
        "lioness": config.methods,
        "bonobo": ("bonobo",),
    }
    variant_rows: dict[str, dict[str, int]] = {}
    candidate_sources: list[Path] = []
    bin_sources: list[Path] = []
    edge_summary_rows: dict[str, int] = {}
    for estimator, methods in method_catalog.items():
        edge_rules = ("all",) if estimator == "lioness" else (
            "all",
            "native_p05",
            "bh_fdr05",
        )
        for method in methods:
            source_method = analysis_root / config.key / estimator / method
            candidate_sources.append(source_method / "volcano_candidates.parquet")
            bin_sources.append(source_method / "volcano_bins.parquet")
            edge_source = source_method / "filtered_edge_summaries.parquet"
            for edge_rule in edge_rules:
                edge_output = (
                    output_root
                    / "edge_summaries"
                    / f"{estimator}__{method}__{edge_rule}.parquet"
                )
                edge_summary_rows[f"{estimator}/{method}/{edge_rule}"] = (
                    write_sanitized_edge_data(
                        edge_source, edge_output, sample_map, edge_rule=edge_rule
                    )
                )
                for fdr_scope in ("global", "per_module"):
                    for fdr_threshold in (0.05, 0.10):
                        threshold_key = f"fdr_{fdr_threshold:.2f}"
                        for normalization in ("standard_pruned", "retained_edge"):
                            source = (
                                app_data_root
                                / config.key
                                / estimator
                                / method
                                / "ad_control_discovery_fdr05"
                                / fdr_scope
                                / threshold_key
                                / edge_rule
                                / normalization
                            )
                            output = (
                                output_root
                                / estimator
                                / method
                                / "ad_control_discovery_fdr05"
                                / fdr_scope
                                / threshold_key
                                / edge_rule
                                / normalization
                            )
                            output.mkdir(parents=True, exist_ok=True)
                            rows = {
                                "aggregate_plot_rows": write_sanitized_plot_data(
                                    [source / "aggregate_plot_data.parquet"],
                                    output / "aggregate_plot_data.parquet",
                                    sample_map,
                                    resolved=False,
                                ),
                                "resolved_plot_rows": write_sanitized_plot_data(
                                    [source / "resolved_plot_data.parquet"],
                                    output / "resolved_plot_data.parquet",
                                    sample_map,
                                    resolved=True,
                                ),
                                "aggregate_stat_rows": write_combined_statistics(
                                    [source / "aggregate_statistics.parquet"],
                                    output / "aggregate_statistics.parquet",
                                ),
                                "resolved_stat_rows": write_combined_statistics(
                                    [source / "resolved_statistics.parquet"],
                                    output / "resolved_statistics.parquet",
                                    resolved=True,
                                ),
                            }
                            variant_rows[
                                f"{estimator}/{method}/{fdr_scope}/{threshold_key}/"
                                f"{edge_rule}/{normalization}"
                            ] = rows

    missing = [path for path in [*candidate_sources, *bin_sources] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Differential volcano sources are missing: {missing}")
    candidate_rows = _write_public_edge_dataset(
        candidate_sources, output_root / "volcano_candidates.parquet"
    )
    bin_rows = _write_public_edge_dataset(
        bin_sources, output_root / "volcano_bins.parquet", rows_per_file=20_000
    )
    return {
        "split": {
            "discovery": {"AD": 117, "Control": 114},
            "validation": {"AD": 50, "Control": 50},
            "mci_external": 119,
            "stratification": "diagnosis and sex; 70/30; seed 42",
        },
        "test": "two-sided Welch t-test on raw signedAlt donor-edge weights",
        "effect": "Hedges g and AD-minus-Control mean difference",
        "fdr": {
            "global": (
                "BH across all tested edges in each module-definition, estimator, "
                "network-method, and analysis-set family"
            ),
            "per_module": (
                "BH within each module across all undirected edges, separately by "
                "module definition, estimator, network method, and analysis set"
            ),
        },
        "feature_masks": {
            "global": "discovery global BH FDR below the selected 0.05/0.10 cutoff",
            "per_module": (
                "discovery per-module BH FDR below the selected 0.05/0.10 cutoff"
            ),
        },
        "feature_mask_thresholds": [0.05, 0.10],
        "variants": variant_rows,
        "edge_summary_rows": edge_summary_rows,
        "volcano_candidate_rows": candidate_rows,
        "volcano_bin_rows": bin_rows,
    }


def validate_resolved_mdc_run(
    root: Path, expected_modules: dict[str, int]
) -> dict[str, object]:
    """Require the completed 200/200 resolved-MDC run before public packaging."""
    manifest_path = root / "run_manifest.tsv"
    completed_path = root / "COMPLETED.txt"
    if not manifest_path.exists() or not completed_path.exists():
        raise FileNotFoundError(f"Resolved MDC run is not complete under {root}")
    manifest_frame = pd.read_csv(manifest_path, sep="\t", dtype=str)
    if set(manifest_frame.columns) != {"key", "value"}:
        raise ValueError("Resolved MDC manifest must contain key and value columns")
    values = dict(manifest_frame[["key", "value"]].itertuples(index=False, name=None))
    required = {
        "sample_permutations": "200",
        "gene_permutations": "200",
        "seed": "42",
        "beta_TS": "3",
        "beta_CT": "2",
        "rows": str(sum(expected_modules.values()) * 9),
    }
    mismatches = {
        key: {"observed": values.get(key), "expected": expected}
        for key, expected in required.items()
        if values.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Resolved MDC run-manifest mismatch: {mismatches}")
    expected_components = {
        "total", "TS", "CT", "TS_AC", "TS_MFBA9BA46", "TS_PCGBA23",
        "CT_AC__MFBA9BA46", "CT_AC__PCGBA23", "CT_MFBA9BA46__PCGBA23",
    }
    for module_set, module_count in expected_modules.items():
        source = root / module_set / "mdc_resolved_ad_vs_control.tsv"
        frame = pd.read_csv(source, sep="\t")
        if len(frame) != module_count * 9:
            raise ValueError(
                f"{module_set}: expected {module_count * 9} MDC rows, found {len(frame)}"
            )
        if frame["module"].nunique() != module_count:
            raise ValueError(f"{module_set}: resolved MDC module count mismatch")
        if set(frame["component"].astype(str)) != expected_components:
            raise ValueError(f"{module_set}: resolved MDC component set mismatch")
        unavailable = frame["n_edges"].eq(0)
        if not frame["mdc"].isna().eq(unavailable).all():
            raise ValueError(
                f"{module_set}: MDC availability does not match structural edges"
            )
        inferential_columns = [
            "p_loss_sample", "p_loss_gene", "q_loss_sample", "q_loss_gene",
            "p_gain_sample", "p_gain_gene", "q_gain_sample", "q_gain_gene",
            "directional_p_sample", "directional_p_gene", "directional_fdr",
        ]
        if frame.loc[unavailable, inferential_columns].notna().any().any():
            raise ValueError(
                f"{module_set}: structurally unavailable MDC rows have inferential values"
            )
        if frame.loc[~unavailable, "directional_fdr"].isna().any():
            raise ValueError(f"{module_set}: available MDC rows lack directional FDR")
    return {
        "sample_permutations": 200,
        "gene_permutations": 200,
        "seed": 42,
        "beta_ts": 3,
        "beta_ct": 2,
        "gene_null": values.get("gene_null"),
        "directional_fdr": values.get("directional_fdr"),
        "source_manifest_sha256": sha256(manifest_path),
    }


def validate_bonobo_run(root: Path) -> dict[str, object]:
    """Validate the complete BONOBO cohort, assignments, and direct equivalence audit."""
    manifest_path = root / "run_manifest.json"
    validation_path = root / "optimized_direct_validation.tsv"
    if not manifest_path.exists() or not validation_path.exists():
        raise FileNotFoundError(f"BONOBO validation outputs are missing under {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("donors", -1)) != 450:
        raise ValueError("BONOBO manifest does not contain exactly 450 donors")
    if manifest.get("diagnosis_counts") != {"AD": 167, "Control": 164, "MCI": 119}:
        raise ValueError("BONOBO diagnosis counts do not match the app cohort")
    expected = {
        "full_cohort": {"modules": 154, "assignment_rows": 46_898},
        "control_derived": {"modules": 186, "assignment_rows": 49_392},
    }
    for module_set, values in expected.items():
        observed = manifest.get("module_sets", {}).get(module_set, {})
        for key, expected_value in values.items():
            if int(observed.get(key, -1)) != expected_value:
                raise ValueError(
                    f"BONOBO {module_set} {key}={observed.get(key)!r}, "
                    f"expected {expected_value}"
                )
    validation = pd.read_csv(validation_path, sep="\t")
    if validation.empty or not validation["passed"].fillna(False).astype(bool).all():
        raise ValueError("Optimized BONOBO direct-equivalence validation did not pass")
    error_columns = [
        "max_abs_correlation_error", "max_abs_p_error", "abs_delta_error"
    ]
    max_error = float(validation[error_columns].max().max())
    if not np.isfinite(max_error) or max_error >= 1e-9:
        raise ValueError(f"Optimized BONOBO maximum direct error is {max_error}")
    return {
        "donors": 450,
        "diagnosis_counts": manifest["diagnosis_counts"],
        "module_sets": expected,
        "direct_validation_cases": len(validation),
        "maximum_direct_error": max_error,
        "source_manifest_sha256": sha256(manifest_path),
        "direct_validation_sha256": sha256(validation_path),
    }


def write_module_details(
    source: Path,
    output: Path,
    expected_modules: set[int],
) -> int:
    """Create a validated, public-facing level-4 module composition table."""
    frame = pd.read_csv(source)
    required = {
        "Cluster ID",
        "Cluster Size",
        "Cluster Type",
        "Cluster Tissues",
        "AC",
        "PCGBA23",
        "MFBA9BA46",
        "Dominant Tissue",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Module details source is missing columns: {sorted(missing)}")

    module = pd.to_numeric(frame["Cluster ID"], errors="raise").astype("int32")
    if module.duplicated().any():
        raise ValueError("Module details source contains duplicate module rows")
    observed_modules = set(module.astype(int))
    if observed_modules != expected_modules:
        raise ValueError(
            "Module details do not match the app modules: "
            f"missing={sorted(expected_modules - observed_modules)}, "
            f"extra={sorted(observed_modules - expected_modules)}"
        )

    module_size = pd.to_numeric(frame["Cluster Size"], errors="raise").astype("int32")
    if module_size.le(0).any():
        raise ValueError("Module details contain a non-positive module size")

    source_count_columns = {
        "n_genes_ac": "AC",
        "n_genes_dlpfc": "MFBA9BA46",
        "n_genes_pcg": "PCGBA23",
    }
    counts = pd.DataFrame(index=frame.index)
    for public_column, source_column in source_count_columns.items():
        values = pd.to_numeric(frame[source_column], errors="coerce").fillna(0)
        if values.lt(0).any() or not np.allclose(values, values.round()):
            raise ValueError(f"{source_column} contains an invalid gene count")
        counts[public_column] = values.astype("int32")
    if not counts.sum(axis=1).eq(module_size).all():
        raise ValueError("Per-tissue gene counts do not sum to Cluster Size")

    tissue_order = [
        ("AC", "n_genes_ac"),
        ("DLPFC", "n_genes_dlpfc"),
        ("PCG", "n_genes_pcg"),
    ]
    tissue_labels = []
    for row in counts.itertuples(index=False):
        row_counts = dict(zip(counts.columns, row))
        tissue_labels.append(
            ", ".join(label for label, column in tissue_order if row_counts[column] > 0)
        )

    source_tissues = (
        frame["Cluster Tissues"]
        .astype("string")
        .str.replace("MFBA9BA46", "DLPFC", regex=False)
        .str.replace("PCGBA23", "PCG", regex=False)
        .str.replace(",", ", ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )
    if not source_tissues.eq(pd.Series(tissue_labels, index=frame.index, dtype="string")).all():
        raise ValueError("Cluster Tissues disagrees with the per-tissue gene counts")

    dominant_tissue = (
        frame["Dominant Tissue"]
        .astype("string")
        .replace({"MFBA9BA46": "DLPFC", "PCGBA23": "PCG"})
    )
    result = pd.DataFrame(
        {
            "module": module,
            "module_size": module_size,
            "cluster_type": frame["Cluster Type"].astype("string"),
            "tissues": pd.Series(tissue_labels, dtype="string"),
            "n_tissues": counts.gt(0).sum(axis=1).astype("int8"),
            "dominant_tissue": dominant_tissue,
            **{column: counts[column] for column in counts.columns},
        }
    )
    result["proportion_ac"] = result["n_genes_ac"] / result["module_size"]
    result["proportion_dlpfc"] = result["n_genes_dlpfc"] / result["module_size"]
    result["proportion_pcg"] = result["n_genes_pcg"] / result["module_size"]
    proportions = result[["proportion_ac", "proportion_dlpfc", "proportion_pcg"]]
    if not np.allclose(proportions.sum(axis=1), 1.0):
        raise ValueError("Per-tissue proportions do not sum to one")

    result = result.sort_values("module").reset_index(drop=True)
    result.to_csv(output, sep="\t", index=False, float_format="%.10g")
    return len(result)


def write_mdc_summary(
    source: Path,
    output: Path,
    expected_modules: set[int],
) -> int:
    """Create the compact module-level MDC table used by the public app."""
    frame = pd.read_csv(source, sep="\t")
    required = {
        "module",
        "size",
        "MDC",
        "MDC_TS",
        "MDC_CT",
        "q_loss_max",
        "q_gain_max",
        "q_loss_max_TS",
        "q_gain_max_TS",
        "q_loss_max_CT",
        "q_gain_max_CT",
        "n_TS_edges",
        "n_CT_edges",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"MDC source is missing columns: {sorted(missing)}")

    frame["module"] = pd.to_numeric(frame["module"], errors="raise").astype("int32")
    if frame["module"].duplicated().any():
        raise ValueError("MDC source contains duplicate module rows")
    observed_modules = set(frame["module"].astype(int))
    if observed_modules != expected_modules:
        raise ValueError(
            "MDC modules do not match the app modules: "
            f"missing={sorted(expected_modules - observed_modules)}, "
            f"extra={sorted(observed_modules - expected_modules)}"
        )

    result = pd.DataFrame(
        {
            "module": frame["module"],
            "module_size_mapped": pd.to_numeric(frame["size"], errors="raise").astype("int32"),
            "n_ts_edges": pd.to_numeric(frame["n_TS_edges"], errors="coerce").astype("Int64"),
            "n_ct_edges": pd.to_numeric(frame["n_CT_edges"], errors="coerce").astype("Int64"),
        }
    )
    scope_columns = {
        "total": ("MDC", "q_loss_max", "q_gain_max"),
        "ts": ("MDC_TS", "q_loss_max_TS", "q_gain_max_TS"),
        "ct": ("MDC_CT", "q_loss_max_CT", "q_gain_max_CT"),
    }
    for scope, (mdc_column, loss_column, gain_column) in scope_columns.items():
        mdc = pd.to_numeric(frame[mdc_column], errors="coerce").astype("float64")
        if mdc.dropna().le(0).any():
            raise ValueError(f"{mdc_column} contains a non-positive ratio")
        loss_q = pd.to_numeric(frame[loss_column], errors="coerce").astype("float64")
        gain_q = pd.to_numeric(frame[gain_column], errors="coerce").astype("float64")
        directional_fdr = loss_q.where(mdc.ge(1), gain_q).where(mdc.notna())
        direction = pd.Series("Not available", index=frame.index, dtype="string")
        direction = direction.mask(mdc.gt(1), "Higher in AD")
        direction = direction.mask(mdc.lt(1), "Higher in Control")
        direction = direction.mask(mdc.eq(1), "Equal")

        result[f"mdc_{scope}"] = mdc
        result[f"log2_mdc_{scope}"] = np.log2(mdc)
        result[f"direction_{scope}"] = direction
        result[f"directional_fdr_{scope}"] = directional_fdr
        result[f"significant_{scope}_fdr05"] = directional_fdr.lt(0.05)
        result[f"significant_{scope}_fdr10"] = directional_fdr.lt(0.10)

    result["ts_minus_ct_log2_mdc"] = (
        result["log2_mdc_ts"] - result["log2_mdc_ct"]
    )
    result["any_significant_fdr05"] = result[
        [
            "significant_total_fdr05",
            "significant_ts_fdr05",
            "significant_ct_fdr05",
        ]
    ].any(axis=1)
    result["any_significant_fdr10"] = result[
        [
            "significant_total_fdr10",
            "significant_ts_fdr10",
            "significant_ct_fdr10",
        ]
    ].any(axis=1)
    result = result.sort_values("module").reset_index(drop=True)
    result.to_csv(output, sep="\t", index=False)
    return len(result)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deploy_file_manifest_entry(path: Path) -> dict[str, object]:
    """Return integrity metadata and, for Parquet files, an explicit schema."""
    entry: dict[str, object] = {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if path.suffix.lower() == ".parquet":
        parquet = pq.ParquetFile(path)
        entry["rows"] = parquet.metadata.num_rows
        entry["schema"] = {
            field.name: str(field.type) for field in parquet.schema_arrow
        }
    return entry


def prepare_public_kegg(
    expanded_source: Path,
    per_tissue_source: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Join whole expanded-tissue KEGG results to per-region ORA statistics."""
    expanded = pd.read_csv(expanded_source, sep="\t")
    per_tissue = pd.read_csv(per_tissue_source, sep="\t")
    keys = ["cluster_id", "term"]
    expanded_required = {
        *keys,
        "p",
        "fdr",
        "significant",
        "overlap_AC",
        "overlap_MFBA9BA46",
        "overlap_PCGBA23",
    }
    per_tissue_required = {
        *keys,
        "p_AC",
        "fdr_AC",
        "significant_AC",
        "overlap_AC",
        "p_MFBA9BA46",
        "fdr_MFBA9BA46",
        "significant_MFBA9BA46",
        "overlap_MFBA9BA46",
        "p_PCGBA23",
        "fdr_PCGBA23",
        "significant_PCGBA23",
        "overlap_PCGBA23",
    }
    for label, frame, required in [
        ("expanded", expanded, expanded_required),
        ("per-tissue", per_tissue, per_tissue_required),
    ]:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} KEGG source is missing columns: {sorted(missing)}")
        frame["cluster_id"] = pd.to_numeric(
            frame["cluster_id"], errors="raise"
        ).astype("int32")
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} KEGG source has duplicate cluster/pathway keys")

    region_source_columns = keys.copy()
    for tissue in ("AC", "MFBA9BA46", "PCGBA23"):
        region_source_columns.extend(
            [
                f"p_{tissue}",
                f"fdr_{tissue}",
                f"significant_{tissue}",
                f"overlap_{tissue}",
            ]
        )
    region = per_tissue[region_source_columns].rename(
        columns={
            "overlap_AC": "per_tissue_overlap_AC",
            "overlap_MFBA9BA46": "per_tissue_overlap_MFBA9BA46",
            "overlap_PCGBA23": "per_tissue_overlap_PCGBA23",
        }
    )
    public = expanded.merge(
        region,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = public.loc[public["_merge"].ne("both"), keys]
    if not unmatched.empty:
        raise ValueError(
            "Expanded KEGG rows are missing per-tissue results: "
            + unmatched.head(10).to_dict(orient="records").__repr__()
        )
    public = public.drop(columns="_merge")

    for tissue in ("AC", "MFBA9BA46", "PCGBA23"):
        region_overlap = f"per_tissue_overlap_{tissue}"
        if not pd.to_numeric(public[f"overlap_{tissue}"], errors="raise").eq(
            pd.to_numeric(public[region_overlap], errors="raise")
        ).all():
            raise ValueError(
                f"Expanded and per-tissue KEGG overlap counts disagree for {tissue}"
            )
        public = public.drop(columns=region_overlap)
        for prefix in ("p", "fdr"):
            values = pd.to_numeric(public[f"{prefix}_{tissue}"], errors="raise")
            if not values.between(0.0, 1.0, inclusive="both").all():
                raise ValueError(f"{prefix}_{tissue} contains values outside [0, 1]")
        expected_significant = public[f"fdr_{tissue}"].le(0.05)
        if not public[f"significant_{tissue}"].astype(bool).eq(
            expected_significant
        ).all():
            raise ValueError(f"significant_{tissue} does not match FDR <= 0.05")

    public = public.rename(
        columns={
            "overlap_MFBA9BA46": "overlap_DLPFC",
            "p_MFBA9BA46": "p_DLPFC",
            "fdr_MFBA9BA46": "fdr_DLPFC",
            "significant_MFBA9BA46": "significant_DLPFC",
        }
    )
    if "overlap_genes" in public:
        public["overlap_genes"] = (
            public["overlap_genes"]
            .astype("string")
            .str.replace("(MFBA9BA46)", "(DLPFC)", regex=False)
        )

    priority_columns = [
        "cluster_id",
        "cluster_size_expanded",
        "pathway_id",
        "pathway_num",
        "pathway_name",
        "category_level1",
        "category_level2",
        "term",
        "p",
        "fdr",
        "significant",
        "p_AC",
        "fdr_AC",
        "significant_AC",
        "p_DLPFC",
        "fdr_DLPFC",
        "significant_DLPFC",
        "p_PCGBA23",
        "fdr_PCGBA23",
        "significant_PCGBA23",
    ]
    priority_columns = [column for column in priority_columns if column in public]
    public = public[
        priority_columns
        + [column for column in public.columns if column not in priority_columns]
    ]

    pathways_per_module = per_tissue.groupby("cluster_id", observed=True).size()
    audit = {
        "source_sha256": sha256(expanded_source),
        "per_region_source_sha256": sha256(per_tissue_source),
        "per_region_source_output": "/".join(per_tissue_source.parts[-3:]),
        "join_keys": keys,
        "join_coverage": 1.0,
        "per_region_input_modules": int(per_tissue["cluster_id"].nunique()),
        "per_region_pathways_per_module_min": int(pathways_per_module.min()),
        "per_region_pathways_per_module_max": int(pathways_per_module.max()),
        "whole_expanded_fdr_definition": (
            "Benjamini-Hochberg correction within each module across pathways with "
            "at least the configured minimum overlap in the expanded tissue-gene universe."
        ),
        "per_region_fdr_definition": (
            "For each module and region separately, Benjamini-Hochberg correction "
            "across the stable KEGG pathway term space; pathways below the minimum "
            "overlap receive p=1 before correction."
        ),
        "region_statistics": {
            "AC": ["p_AC", "fdr_AC", "significant_AC"],
            "DLPFC": ["p_DLPFC", "fdr_DLPFC", "significant_DLPFC"],
            "PCG": ["p_PCGBA23", "fdr_PCGBA23", "significant_PCGBA23"],
        },
    }
    return public, audit


def write_public_kegg(
    expanded_source: Path,
    per_tissue_source: Path,
    output: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    public, audit = prepare_public_kegg(expanded_source, per_tissue_source)
    output.mkdir(parents=True, exist_ok=True)
    public.to_parquet(
        output / "kegg_tissue_expanded_full.parquet",
        index=False,
        compression="zstd",
    )
    public.to_csv(output / "kegg_tissue_expanded_full.tsv", sep="\t", index=False)
    return public, audit


def mdc_manifest(source: Path) -> dict[str, object]:
    metadata = dict(MDC_SOURCE_METADATA)
    metadata.update(
        {
            "source_output": "/".join(source.parts[-3:]),
            "source_last_modified_utc": datetime.fromtimestamp(
                source.stat().st_mtime, timezone.utc
            ).isoformat(),
            "source_sha256": sha256(source),
            "cohort_note": (
                "The MDC assembled cohort is the tissue union and contains every complete-tissue "
                "AD/Control donor used by LIONESS plus donors with partial tissue availability."
            ),
        }
    )
    return metadata


def module_details_manifest(source: Path) -> dict[str, object]:
    return {
        "source_output": "/".join(source.parts[-4:]),
        "source_last_modified_utc": datetime.fromtimestamp(
            source.stat().st_mtime, timezone.utc
        ).isoformat(),
        "source_sha256": sha256(source),
        "definition": (
            "Level-4 module size and tissue composition using the public labels AC, "
            "DLPFC, and PCG. Tissue proportions use module size as the denominator."
        ),
    }


def validate_assignment_details(
    assignment_source: Path,
    details_output: Path,
    expected_modules: set[int],
) -> int:
    """Verify exact tissue-gene counts against the public module-details table."""
    assignments = pd.read_csv(assignment_source)
    required = {"Cluster ID", "Tissue", "Gene ID"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"Assignment source is missing columns: {sorted(missing)}")
    assignments["Cluster ID"] = pd.to_numeric(
        assignments["Cluster ID"], errors="raise"
    ).astype("int32")
    # Preserve source row multiplicity. The legacy full-cohort assignment file
    # intentionally contains repeated tissue-gene rows and its module-details
    # counts include those rows; the Control-derived source has no repeats.
    observed_modules = set(assignments["Cluster ID"].astype(int))
    if observed_modules != expected_modules:
        raise ValueError(
            "Assignment modules do not match analysis modules: "
            f"missing={sorted(expected_modules - observed_modules)}, "
            f"extra={sorted(observed_modules - expected_modules)}"
        )

    details = pd.read_csv(details_output, sep="\t").set_index("module")
    counts = (
        assignments.groupby(["Cluster ID", "Tissue"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    expected_tissues = {
        "AC": "n_genes_ac",
        "MFBA9BA46": "n_genes_dlpfc",
        "PCGBA23": "n_genes_pcg",
    }
    unknown_tissues = set(counts.columns).difference(expected_tissues)
    if unknown_tissues:
        raise ValueError(f"Unexpected tissues in assignments: {sorted(unknown_tissues)}")
    counts = counts.reindex(index=details.index, fill_value=0)
    for tissue, public_column in expected_tissues.items():
        observed = counts[tissue] if tissue in counts else pd.Series(0, index=details.index)
        if not observed.astype(int).eq(details[public_column].astype(int)).all():
            raise ValueError(
                f"Assignment counts for {tissue} disagree with module details"
            )
    if len(assignments) != int(details["module_size"].sum()):
        raise ValueError("Assignment row count does not equal summed module sizes")
    return len(assignments)


def donor_set(source: Path) -> set[str]:
    return {
        str(value)
        for value in pq.read_table(source, columns=["donor"]).column("donor").to_pylist()
    }


def dataset_file_manifest(output: Path) -> dict[str, dict[str, object]]:
    module_set_filenames = {
        "aggregate_plot_data.parquet",
        "resolved_plot_data.parquet",
        "aggregate_statistics.parquet",
        "resolved_statistics.parquet",
        "kegg_tissue_expanded_full.parquet",
        "kegg_tissue_expanded_full.tsv",
        "module_kegg_annotations.tsv",
        MODULE_DETAILS_OUTPUT_NAME,
        MDC_OUTPUT_NAME,
    }
    files = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name in module_set_filenames
    )
    result = {path.name: deploy_file_manifest_entry(path) for path in files}
    too_large = {
        name: values["bytes"]
        for name, values in result.items()
        if int(values["bytes"]) >= 95 * 1024 * 1024
    }
    if too_large:
        raise ValueError(f"Deploy files exceed the 95-MiB release cap: {too_large}")
    return result


def validate_public_module_set(output: Path, expected_sample_ids: set[str]) -> None:
    """Reject identifiers or internal tissue labels in a deploy module bundle."""
    forbidden_columns = {"donor", "projid"}
    internal_labels = ("MFBA9BA46", "MFBA9/BA46")
    parquet_paths = sorted(output.rglob("*.parquet"))
    for path in parquet_paths:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        if forbidden_columns.intersection(names):
            raise ValueError(f"Private identifiers remain in {path}")
        for field in parquet.schema_arrow:
            if not (
                pa.types.is_string(field.type)
                or pa.types.is_large_string(field.type)
                or pa.types.is_dictionary(field.type)
            ):
                continue
            values = pc.unique(
                pq.read_table(path, columns=[field.name])[field.name].cast(pa.string())
            )
            for internal_label in internal_labels:
                found = pc.any(pc.match_substring(values, internal_label)).as_py()
                if bool(found):
                    raise ValueError(
                        f"Internal tissue label {internal_label} remains in {path}"
                    )
        if "sample_id" in names:
            sample_ids = set(
                pc.unique(
                    pq.read_table(path, columns=["sample_id"])["sample_id"]
                ).to_pylist()
            )
            if sample_ids != expected_sample_ids:
                raise ValueError(f"Public sample labels are inconsistent in {path}")
    for path in output.rglob("*.tsv"):
        contents = path.read_text(encoding="utf-8")
        header = contents.splitlines()[0].split("\t") if contents else []
        if forbidden_columns.intersection(header):
            raise ValueError(f"Private identifier columns remain in {path}")
        if any(label in contents for label in internal_labels):
            raise ValueError(f"An internal tissue label remains in {path}")


def build_module_set(
    config: DatasetConfig,
    sample_map: dict[str, str],
) -> dict[str, object]:
    """Build one isolated module-definition bundle with shared sample pseudonyms."""
    config.output.mkdir(parents=True, exist_ok=True)
    analysis_inputs_path = config.analysis_root / "analysis_inputs.tsv"
    analysis_inputs = pd.read_csv(analysis_inputs_path, sep="\t")
    assignment_inputs = analysis_inputs.loc[
        analysis_inputs["input_role"].eq("module_assignments"), "path"
    ]
    if len(assignment_inputs) != 1 or Path(assignment_inputs.iloc[0]).resolve() != config.assignments:
        raise ValueError(
            f"{config.key}: LIONESS analysis input does not match the supplied assignments"
        )
    mdc_inputs_path = config.mdc.parent / "analysis_inputs.tsv"
    if config.key == "control_derived":
        mdc_inputs = pd.read_csv(mdc_inputs_path, sep="\t")
        mdc_assignment_inputs = mdc_inputs.loc[
            mdc_inputs["input_role"].eq("module_assignments"), "path"
        ]
        if (
            len(mdc_assignment_inputs) != 1
            or Path(mdc_assignment_inputs.iloc[0]).resolve() != config.assignments
        ):
            raise ValueError(
                "Control-derived MDC input does not match the supplied assignments"
            )
    aggregate_sources = [
        source_file(config.analysis_root, method, "_transformed_component_data.parquet")
        for method in config.methods
    ]
    resolved_sources = [
        source_file(
            config.analysis_root, method, "_tissue_resolved_component_data.parquet"
        )
        for method in config.methods
    ]
    aggregate_stat_sources = [
        source_file(
            config.analysis_root,
            method,
            "_robust_statistics.parquet",
            exclude_text="tissue_resolved",
        )
        for method in config.methods
    ]
    resolved_stat_sources = [
        source_file(
            config.analysis_root,
            method,
            "_tissue_resolved_robust_statistics.parquet",
        )
        for method in config.methods
    ]

    expected_donors = set(sample_map)
    for source in aggregate_sources:
        observed_donors = donor_set(source)
        if observed_donors != expected_donors:
            raise ValueError(
                f"{config.key} donor set differs from the shared 450-donor mapping"
            )

    aggregate_rows = write_sanitized_plot_data(
        aggregate_sources,
        config.output / "aggregate_plot_data.parquet",
        sample_map,
        resolved=False,
    )
    resolved_rows = write_sanitized_plot_data(
        resolved_sources,
        config.output / "resolved_plot_data.parquet",
        sample_map,
        resolved=True,
    )
    aggregate_stat_rows = write_combined_statistics(
        aggregate_stat_sources,
        config.output / "aggregate_statistics.parquet",
        resolved=False,
    )
    resolved_stat_rows = write_combined_statistics(
        resolved_stat_sources,
        config.output / "resolved_statistics.parquet",
        resolved=True,
    )

    kegg, kegg_audit = write_public_kegg(
        config.kegg,
        config.kegg_per_tissue,
        config.output,
    )

    annotation_source = table_file(
        config.analysis_root, config.methods[0], "_module_kegg_annotations.tsv"
    )
    annotations = pd.read_csv(annotation_source, sep="\t")
    annotations["module"] = pd.to_numeric(
        annotations["module"], errors="raise"
    ).astype("int32")
    expected_modules = set(annotations["module"].astype(int))
    if len(expected_modules) != config.module_count:
        raise ValueError(
            f"{config.key}: expected {config.module_count} modules, found "
            f"{len(expected_modules)}"
        )
    if config.sentinel_module not in expected_modules:
        raise ValueError(
            f"{config.key}: sentinel M{config.sentinel_module} is absent"
        )
    annotations.to_csv(
        config.output / "module_kegg_annotations.tsv", sep="\t", index=False
    )
    reported_kegg_modules = set(kegg["cluster_id"].astype(int))
    if not reported_kegg_modules.issubset(expected_modules):
        raise ValueError(f"{config.key}: KEGG contains modules outside the analysis set")
    kegg_gene_lists = config.kegg.parent / "cluster_gene_lists"
    admitted_kegg_modules = {
        int(path.name.split("_", 2)[1])
        for path in kegg_gene_lists.glob("cluster_*_UNION_ensembl.txt")
    }
    if config.key == "control_derived" and admitted_kegg_modules != expected_modules:
        raise ValueError(
            "Control-derived KEGG did not admit all 186 modules: "
            f"missing={sorted(expected_modules - admitted_kegg_modules)}, "
            f"extra={sorted(admitted_kegg_modules - expected_modules)}"
        )

    mdc_rows = write_mdc_summary(
        config.mdc,
        config.output / MDC_OUTPUT_NAME,
        expected_modules,
    )
    module_details_rows = write_module_details(
        config.module_details,
        config.output / MODULE_DETAILS_OUTPUT_NAME,
        expected_modules,
    )
    assignment_rows = validate_assignment_details(
        config.assignments,
        config.output / MODULE_DETAILS_OUTPUT_NAME,
        expected_modules,
    )
    public_details = pd.read_csv(config.output / MODULE_DETAILS_OUTPUT_NAME, sep="\t")
    ts_labeled_modules = sorted(
        public_details.loc[
            public_details["cluster_type"].astype(str).str.upper().eq("TS"), "module"
        ].astype(int)
    )
    if config.key == "control_derived" and ts_labeled_modules != [355, 356, 867]:
        raise ValueError(
            "Control-derived TS module labels must be M355, M356, and M867; found "
            f"{ts_labeled_modules}"
        )

    method_count = len(config.methods)
    expected_rows = {
        "aggregate_rows": method_count * 450 * config.module_count * 6,
        "resolved_rows": method_count * 450 * config.module_count * 6 * 6,
        "aggregate_stat_rows": method_count * config.module_count * 5 * 6 * 3,
        "resolved_stat_rows": method_count * config.module_count * 5 * 6 * 6 * 3,
        "mdc_rows": config.module_count,
        "module_details_rows": config.module_count,
        "assignment_rows": assignment_rows,
    }
    observed_rows = {
        "aggregate_rows": aggregate_rows,
        "resolved_rows": resolved_rows,
        "aggregate_stat_rows": aggregate_stat_rows,
        "resolved_stat_rows": resolved_stat_rows,
        "mdc_rows": mdc_rows,
        "module_details_rows": module_details_rows,
        "assignment_rows": assignment_rows,
    }
    if observed_rows != expected_rows:
        raise ValueError(
            f"{config.key} row-count validation failed: "
            f"observed={observed_rows}, expected={expected_rows}"
        )

    mdc_public = pd.read_csv(config.output / MDC_OUTPUT_NAME, sep="\t")
    ct_unavailable_modules = sorted(
        mdc_public.loc[
            pd.to_numeric(mdc_public["n_ct_edges"], errors="coerce").fillna(0).eq(0),
            "module",
        ].astype(int)
    )
    if config.key == "control_derived" and ct_unavailable_modules != [356]:
        raise ValueError(
            "Control-derived MDC must have CT unavailable only for M356; found "
            f"{ct_unavailable_modules}"
        )

    formula_validation_path = (
        config.analysis_root / "control_reference_formula_validation.tsv"
    )
    formula_validation: dict[str, object] | None = None
    if formula_validation_path.exists():
        formula_frame = pd.read_csv(formula_validation_path, sep="\t")
        required_formula_groups = {"Control", "MCI", "AD"}
        if set(formula_frame["diagnosis_group"].astype(str)) != required_formula_groups:
            raise ValueError(
                f"{config.key}: formula validation does not contain Control/MCI/AD"
            )
        if not formula_frame["passed"].fillna(False).astype(bool).all():
            raise ValueError(f"{config.key}: a direct formula validation failed")
        formula_validation = {
            "module": int(formula_frame["module"].iloc[0]),
            "groups": sorted(required_formula_groups),
            "max_absolute_error": float(formula_frame["max_absolute_error"].max()),
            "max_relative_error": float(formula_frame["max_relative_error"].max()),
            "tolerance": float(formula_frame["tolerance"].max()),
            "source_sha256": sha256(formula_validation_path),
        }
    elif config.key == "control_derived":
        raise FileNotFoundError(
            "Control-derived analysis is missing control_reference_formula_validation.tsv"
        )

    files = dataset_file_manifest(config.output)
    aggregate_global = config.module_count * 5 * 6 * 3
    aggregate_within = config.module_count * 6 * 3
    return {
        "key": config.key,
        "label": config.label,
        "relative_data_dir": "." if config.key == "full_cohort" else config.key,
        "source_run": config.analysis_root.name,
        "methods": list(config.methods),
        "donors_per_method": 450,
        "diagnosis_counts_per_method": {"Control": 164, "MCI": 119, "AD": 167},
        "modules": config.module_count,
        "sentinel_module": config.sentinel_module,
        "phenotypes": list(PHENOTYPES),
        "feature_families": 6,
        "resolved_components": 6,
        "ct_unavailable_modules": ct_unavailable_modules,
        "ts_labeled_modules": ts_labeled_modules,
        "fdr_test_families": {
            "aggregate_global": aggregate_global,
            "aggregate_within_phenotype": aggregate_within,
            "resolved_global": aggregate_global * 6,
            "resolved_within_phenotype": aggregate_within * 6,
        },
        "rows": observed_rows,
        "kegg": {
            "input_modules": config.module_count,
            "admitted_modules": len(admitted_kegg_modules),
            "modules_with_reported_pathways": len(reported_kegg_modules),
            "modules_without_reported_pathways": sorted(
                expected_modules - reported_kegg_modules
            ),
            "rows": len(kegg),
            "fdr_definition": kegg_audit["whole_expanded_fdr_definition"],
            **kegg_audit,
        },
        "mdc": mdc_manifest(config.mdc),
        "module_details": module_details_manifest(config.module_details),
        "assignments": {
            "rows": assignment_rows,
            "source_sha256": sha256(config.assignments),
        },
        "formula_validation": formula_validation,
        "files": files,
    }


def refresh_existing_mdc_bundle(source: Path, output: Path) -> dict[str, object]:
    """Refresh MDC context without rebuilding donor pseudonyms or large Parquet files."""
    annotations = pd.read_csv(output / "module_kegg_annotations.tsv", sep="\t")
    expected_modules = set(annotations["module"].astype(int))
    summary_path = output / MDC_OUTPUT_NAME
    mdc_rows = write_mdc_summary(source, summary_path, expected_modules)

    manifest_path = output / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("rows", {})["mdc_rows"] = mdc_rows
    manifest["mdc"] = mdc_manifest(source)
    file_entry = deploy_file_manifest_entry(summary_path)
    manifest["files"][summary_path.name] = file_entry
    if "module_sets" in manifest:
        full = manifest["module_sets"]["full_cohort"]
        full["rows"]["mdc_rows"] = mdc_rows
        full["mdc"] = manifest["mdc"]
        full["files"][summary_path.name] = file_entry
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def refresh_existing_module_details_bundle(
    source: Path, output: Path
) -> dict[str, object]:
    """Refresh module composition without rebuilding donor-level deploy data."""
    annotations = pd.read_csv(output / "module_kegg_annotations.tsv", sep="\t")
    expected_modules = set(annotations["module"].astype(int))
    details_path = output / MODULE_DETAILS_OUTPUT_NAME
    details_rows = write_module_details(source, details_path, expected_modules)
    details = pd.read_csv(details_path, sep="\t")
    proportions = details[["n_genes_ac", "n_genes_dlpfc", "n_genes_pcg"]].div(
        details["module_size"], axis=0
    )
    details["tissue_entropy"] = -(
        proportions.where(proportions > 0)
        * np.log2(proportions.where(proportions > 0))
    ).sum(axis=1)
    details["tissue_entropy_normalized"] = details["tissue_entropy"] / np.log2(3)
    details.to_csv(details_path, sep="\t", index=False, float_format="%.10g")

    manifest_path = output / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("rows", {})["module_details_rows"] = details_rows
    manifest["module_details"] = module_details_manifest(source)
    file_entry = deploy_file_manifest_entry(details_path)
    manifest["files"][details_path.name] = file_entry
    if "module_sets" in manifest:
        full = manifest["module_sets"]["full_cohort"]
        full["rows"]["module_details_rows"] = details_rows
        full["module_details"] = manifest["module_details"]
        full["files"][details_path.name] = file_entry
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def refresh_existing_statistics_bundle(
    all12_stats_root: Path, output: Path
) -> dict[str, object]:
    """Refresh all-12-outcome LIONESS statistics without changing donor labels."""
    manifest_path = output / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configurations = {
        "full_cohort": {
            "methods": ("standard", "control_anchored"),
            "modules": 154,
            "directory": output,
        },
        "control_derived": {
            "methods": ("control_anchored",),
            "modules": 186,
            "directory": output / "control_derived",
        },
    }
    for module_set, config in configurations.items():
        aggregate_sources = [
            all12_stats_root / module_set / method / "aggregate_statistics.parquet"
            for method in config["methods"]
        ]
        resolved_sources = [
            all12_stats_root / module_set / method / "resolved_statistics.parquet"
            for method in config["methods"]
        ]
        missing = [str(path) for path in (*aggregate_sources, *resolved_sources) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{module_set} all-12 statistics are incomplete: {', '.join(missing)}"
            )
        aggregate_path = config["directory"] / "aggregate_statistics.parquet"
        resolved_path = config["directory"] / "resolved_statistics.parquet"
        observed = {
            "aggregate_stat_rows": write_combined_statistics(
                aggregate_sources, aggregate_path, resolved=False
            ),
            "resolved_stat_rows": write_combined_statistics(
                resolved_sources, resolved_path, resolved=True
            ),
        }
        expected = {
            "aggregate_stat_rows": len(config["methods"]) * config["modules"] * 6 * 12 * 3,
            "resolved_stat_rows": len(config["methods"]) * config["modules"] * 6 * 6 * 12 * 3,
        }
        if observed != expected:
            raise ValueError(
                f"{module_set} statistics row-count validation failed: "
                f"observed={observed}, expected={expected}"
            )
        module_manifest = manifest["module_sets"][module_set]
        module_manifest["rows"].update(observed)
        for path in (aggregate_path, resolved_path):
            relative = str(path.relative_to(output))
            file_entry = deploy_file_manifest_entry(path)
            module_manifest["files"][path.name] = file_entry
            manifest["files"][relative] = file_entry
        if module_set == "full_cohort":
            manifest.setdefault("rows", {}).update(observed)
    manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def refresh_existing_kegg_bundles(
    output: Path,
    sources: tuple[tuple[str, Path, Path, Path], ...],
) -> dict[str, object]:
    """Refresh both KEGG bundles without changing donor-level public data."""
    manifest_path = output / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for module_set, expanded_source, per_tissue_source, dataset_output in sources:
        public, audit = write_public_kegg(
            expanded_source,
            per_tissue_source,
            dataset_output,
        )
        details = pd.read_csv(dataset_output / MODULE_DETAILS_OUTPUT_NAME, sep="\t")
        expected_modules = set(details["module"].astype(int))
        reported_modules = set(public["cluster_id"].astype(int))
        if not reported_modules.issubset(expected_modules):
            raise ValueError(
                f"{module_set}: KEGG contains modules outside the public module set"
            )

        module_manifest = manifest["module_sets"][module_set]
        kegg_manifest = dict(module_manifest["kegg"])
        kegg_manifest.update(
            {
                "input_modules": len(expected_modules),
                "admitted_modules": int(audit["per_region_input_modules"]),
                "modules_with_reported_pathways": len(reported_modules),
                "modules_without_reported_pathways": sorted(
                    expected_modules - reported_modules
                ),
                "rows": len(public),
                "fdr_definition": audit["whole_expanded_fdr_definition"],
                **audit,
            }
        )
        module_manifest["kegg"] = kegg_manifest

        for filename in (
            "kegg_tissue_expanded_full.parquet",
            "kegg_tissue_expanded_full.tsv",
        ):
            path = dataset_output / filename
            file_entry = deploy_file_manifest_entry(path)
            if path.stat().st_size >= 95 * 1024 * 1024:
                raise ValueError(f"Refreshed KEGG file exceeds 95-MiB cap: {path}")
            module_manifest["files"][filename] = file_entry
            manifest["files"][str(path.relative_to(output))] = file_entry

    manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    refresh_flags = [
        args.mdc_only,
        args.module_details_only,
        args.statistics_only,
        args.kegg_only,
    ]
    if sum(refresh_flags) > 1:
        raise ValueError(
            "Use only one of --mdc-only, --module-details-only, --statistics-only, "
            "or --kegg-only"
        )

    if args.mdc_only:
        manifest = refresh_existing_mdc_bundle(args.mdc.resolve(), output)
        print(json.dumps(manifest["mdc"], indent=2))
        return
    if args.module_details_only:
        manifest = refresh_existing_module_details_bundle(
            args.module_details.resolve(), output
        )
        print(json.dumps(manifest["module_details"], indent=2))
        return
    if args.statistics_only:
        manifest = refresh_existing_statistics_bundle(
            args.lioness_all12_stats_root.resolve(), output
        )
        print(
            json.dumps(
                {
                    "aggregate_stat_rows": manifest["rows"]["aggregate_stat_rows"],
                    "resolved_stat_rows": manifest["rows"]["resolved_stat_rows"],
                },
                indent=2,
            )
        )
        return
    if args.kegg_only:
        manifest = refresh_existing_kegg_bundles(
            output,
            (
                (
                    "full_cohort",
                    args.kegg.resolve(),
                    args.kegg_per_tissue.resolve(),
                    output,
                ),
                (
                    "control_derived",
                    args.control_derived_kegg.resolve(),
                    args.control_derived_kegg_per_tissue.resolve(),
                    output / "control_derived",
                ),
            ),
        )
        print(
            json.dumps(
                {
                    key: manifest["module_sets"][key]["kegg"]
                    for key in ("full_cohort", "control_derived")
                },
                indent=2,
            )
        )
        return

    configs = (
        DatasetConfig(
            key="full_cohort",
            label="Full-cohort L4 modules (154)",
            analysis_root=args.analysis_root.resolve(),
            methods=("standard", "control_anchored"),
            module_count=154,
            sentinel_module=1918,
            kegg=args.kegg.resolve(),
            kegg_per_tissue=args.kegg_per_tissue.resolve(),
            mdc=args.mdc.resolve(),
            module_details=args.module_details.resolve(),
            assignments=args.full_assignments.resolve(),
            output=output,
        ),
        DatasetConfig(
            key="control_derived",
            label="Control-derived L4 modules (186)",
            analysis_root=args.control_derived_analysis_root.resolve(),
            methods=("control_anchored",),
            module_count=186,
            sentinel_module=10,
            kegg=args.control_derived_kegg.resolve(),
            kegg_per_tissue=args.control_derived_kegg_per_tissue.resolve(),
            mdc=args.control_derived_mdc.resolve(),
            module_details=args.control_derived_module_details.resolve(),
            assignments=args.control_derived_assignments.resolve(),
            output=output / "control_derived",
        ),
    )

    resolved_mdc_design = validate_resolved_mdc_run(
        args.resolved_mdc_root.resolve(),
        {config.key: config.module_count for config in configs},
    )
    bonobo_analysis = validate_bonobo_run(args.bonobo_network_root.resolve())

    # Preflight every input before replacing any deploy data.
    for config in configs:
        mandatory = [
            config.kegg,
            config.kegg_per_tissue,
            config.mdc,
            config.module_details,
            config.assignments,
        ]
        if config.key == "control_derived":
            mandatory.append(
                config.analysis_root / "control_reference_formula_validation.tsv"
            )
        mandatory.extend(
            [
                args.bonobo_network_root.resolve()
                / config.key
                / "bonobo_edge_summaries.parquet",
                args.lioness_expansion_root.resolve()
                / config.key
                / "module_tissue_entropy.tsv",
                args.resolved_mdc_root.resolve()
                / config.key
                / "mdc_resolved_ad_vs_control.tsv",
            ]
        )
        differential_methods = {
            "lioness": config.methods,
            "bonobo": ("bonobo",),
        }
        for estimator, methods in differential_methods.items():
            edge_rules = ("all",) if estimator == "lioness" else (
                "all", "native_p05", "bh_fdr05"
            )
            for method in methods:
                differential_source = (
                    args.differential_analysis_root.resolve()
                    / config.key
                    / estimator
                    / method
                )
                mandatory.extend(
                    differential_source / filename
                    for filename in (
                        "filtered_edge_summaries.parquet",
                        "volcano_candidates.parquet",
                        "volcano_bins.parquet",
                    )
                )
                for edge_rule in edge_rules:
                    for fdr_scope in ("global", "per_module"):
                        for fdr_threshold in (0.05, 0.10):
                            threshold_key = f"fdr_{fdr_threshold:.2f}"
                            for normalization in ("standard_pruned", "retained_edge"):
                                app_source = (
                                    args.differential_app_data_root.resolve()
                                    / config.key
                                    / estimator
                                    / method
                                    / "ad_control_discovery_fdr05"
                                    / fdr_scope
                                    / threshold_key
                                    / edge_rule
                                    / normalization
                                )
                                mandatory.extend(
                                    app_source / filename
                                    for filename in (
                                        "aggregate_plot_data.parquet",
                                        "resolved_plot_data.parquet",
                                        "aggregate_statistics.parquet",
                                        "resolved_statistics.parquet",
                                    )
                                )
        for edge_rule in ("all", "native_p05", "bh_fdr05"):
            mandatory.extend(
                args.bonobo_app_data_root.resolve()
                / config.key
                / "bonobo"
                / edge_rule
                / filename
                for filename in (
                    "aggregate_plot_data.parquet",
                    "resolved_plot_data.parquet",
                    "aggregate_statistics.parquet",
                    "resolved_statistics.parquet",
                )
            )
        for method in config.methods:
            mandatory.extend(
                [
                    args.lioness_expansion_root.resolve()
                    / config.key
                    / f"lioness_edge_summaries__{method}.parquet",
                    args.lioness_all12_stats_root.resolve()
                    / config.key
                    / method
                    / "aggregate_statistics.parquet",
                    args.lioness_all12_stats_root.resolve()
                    / config.key
                    / method
                    / "resolved_statistics.parquet",
                ]
            )
        missing = [str(path) for path in mandatory if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{config.key} source bundle is incomplete: {', '.join(missing)}"
            )
        for method in config.methods:
            source_file(
                config.analysis_root, method, "_transformed_component_data.parquet"
            )
            source_file(
                config.analysis_root,
                method,
                "_tissue_resolved_component_data.parquet",
            )
            source_file(
                config.analysis_root,
                method,
                "_robust_statistics.parquet",
                exclude_text="tissue_resolved",
            )
            source_file(
                config.analysis_root,
                method,
                "_tissue_resolved_robust_statistics.parquet",
            )
            table_file(
                config.analysis_root, method, "_module_kegg_annotations.tsv"
            )

    full_aggregate_source = source_file(
        configs[0].analysis_root,
        "standard",
        "_transformed_component_data.parquet",
    )
    sample_map = create_sample_map(full_aggregate_source)
    metadata_rows = write_sample_metadata(
        configs[0].analysis_root / "standard" / "data" / "phenotypes.parquet",
        output / "sample_metadata.parquet",
        sample_map,
        args.differential_analysis_root.resolve()
        / "ad_control_discovery_validation_split.tsv",
    )
    module_set_manifests = {
        config.key: build_module_set(config, sample_map) for config in configs
    }
    expansion_manifests = {}
    for config in configs:
        expansion = write_expanded_module_bundle(
            config,
            sample_map,
            bonobo_app_data_root=args.bonobo_app_data_root.resolve(),
            bonobo_network_root=args.bonobo_network_root.resolve(),
            lioness_expansion_root=args.lioness_expansion_root.resolve(),
            lioness_all12_stats_root=args.lioness_all12_stats_root.resolve(),
            resolved_mdc_root=args.resolved_mdc_root.resolve(),
        )
        expansion_manifests[config.key] = expansion
        module_manifest = module_set_manifests[config.key]
        differential = write_differential_module_bundle(
            config,
            sample_map,
            analysis_root=args.differential_analysis_root.resolve(),
            app_data_root=args.differential_app_data_root.resolve(),
        )
        module_manifest["ad_control_differential_edges"] = differential
        module_manifest["outcomes"] = list(
            (*PHENOTYPES, "age_at_death", "education_years", "cogdx", "braak_stage", "cerad_score", "adnc", "parkinsonism")
        )
        module_manifest["estimators"] = ["lioness", "bonobo"]
        module_manifest["feature_families_by_estimator"] = {
            "lioness": 6,
            "bonobo": 3,
        }
        module_manifest["bonobo"] = expansion["bonobo"]
        module_manifest["bonobo_analysis"] = bonobo_analysis
        module_manifest["edge_summaries"] = expansion["edge_summaries"]
        module_manifest["fdr_test_families"] = {
            "definition": (
                "BH correction is separate by module definition, estimator/network "
                "method, aggregate or resolved component family, correlation test, "
                "and BONOBO edge rule"
            ),
            "lioness": {
                "all12_aggregate_global": config.module_count * 6 * 12 * 3,
                "primary5_aggregate_global": config.module_count * 6 * 5 * 3,
                "aggregate_within_outcome": config.module_count * 6 * 3,
                "all12_resolved_global": config.module_count * 6 * 6 * 12 * 3,
                "primary5_resolved_global": config.module_count * 6 * 6 * 5 * 3,
                "resolved_within_outcome": config.module_count * 6 * 6 * 3,
            },
            "bonobo_per_edge_rule": {
                "all12_aggregate_global": config.module_count * 3 * 12 * 3,
                "primary5_aggregate_global": config.module_count * 3 * 5 * 3,
                "aggregate_within_outcome": config.module_count * 3 * 3,
                "all12_resolved_global": config.module_count * 3 * 6 * 12 * 3,
                "primary5_resolved_global": config.module_count * 3 * 6 * 5 * 3,
                "resolved_within_outcome": config.module_count * 3 * 6 * 3,
            },
        }
        module_manifest["rows"]["aggregate_stat_rows"] = expansion[
            "lioness_aggregate_stat_rows"
        ]
        module_manifest["rows"]["resolved_stat_rows"] = expansion[
            "lioness_resolved_stat_rows"
        ]
        module_manifest["rows"]["resolved_mdc_rows"] = expansion[
            "resolved_mdc_rows"
        ]
        module_manifest["rows"]["entropy_rows"] = expansion["entropy_rows"]
        module_manifest["entropy"] = {
            "raw": "H=-sum(p_t*log2(p_t)) over AC, DLPFC, and PCG",
            "normalized": "H/log2(3)",
        }
        module_manifest["resolved_mdc"] = {
            "components": 6,
            **resolved_mdc_design,
        }
        module_paths = [
            path
            for path in sorted(config.output.rglob("*"))
            if path.is_file()
            and not (
                config.key == "full_cohort"
                and path.relative_to(config.output).parts[0] == "control_derived"
            )
        ]
        module_manifest["files"] = {
            str(path.relative_to(config.output)): deploy_file_manifest_entry(path)
            for path in module_paths
        }

    features = pd.read_csv(
        configs[0].analysis_root / "feature_definitions.tsv", sep="\t"
    )
    features = features.replace("MFBA9BA46", "DLPFC", regex=True)
    features.insert(0, "estimator", "LIONESS")
    features["edge_rules"] = "All edges"
    bonobo_features = pd.DataFrame(
        [
            {
                "estimator": "BONOBO",
                "feature_family": "connectivity",
                "definition": (
                    "Mean signed BONOBO node strength attributable to the selected "
                    "edge block; equivalently 2*sum(signedAlt edge weight)/module gene count"
                ),
            },
            {
                "estimator": "BONOBO",
                "feature_family": "positive_density",
                "definition": (
                    "Mean signedAlt BONOBO edge weight among retained positive edges "
                    "in the selected edge block"
                ),
            },
            {
                "estimator": "BONOBO",
                "feature_family": "negative_density",
                "definition": (
                    "Mean signedAlt BONOBO edge weight among retained negative edges "
                    "in the selected edge block"
                ),
            },
        ]
    )
    bonobo_features["aggregate_components"] = "CT (all tissue pairs); TS (all tissues)"
    bonobo_features["resolved_components"] = (
        "CT_AC__DLPFC,CT_AC__PCGBA23,CT_DLPFC__PCGBA23,"
        "TS_AC,TS_DLPFC,TS_PCGBA23"
    )
    bonobo_features["edge_rules"] = (
        "All edges; native posterior p<0.05; within-donor/module BH FDR<0.05"
    )
    features = pd.concat([features, bonobo_features[features.columns]], ignore_index=True)
    features.to_csv(output / "feature_definitions.tsv", sep="\t", index=False)
    tissues = pd.read_csv(configs[0].analysis_root / "tissue_mapping.tsv", sep="\t")
    tissues["internal_tissue"] = tissues["internal_tissue"].replace(
        {"MFBA9BA46": "DLPFC"}
    )
    tissues["display_name"] = tissues["display_name"].replace(
        {"DLPFC (MFBA9/BA46)": "DLPFC"}
    )
    tissues[["internal_tissue", "display_name"]].to_csv(
        output / "tissue_mapping.tsv", sep="\t", index=False
    )

    expected_sample_ids = set(sample_map.values())
    for config in configs:
        validate_public_module_set(config.output, expected_sample_ids)

    if metadata_rows != 450:
        raise ValueError(f"Expected 450 metadata rows, found {metadata_rows}")
    diagnosis_counts = (
        pd.read_parquet(output / "sample_metadata.parquet")["diagnosis_group"]
        .value_counts()
        .to_dict()
    )
    if diagnosis_counts != {"AD": 167, "Control": 164, "MCI": 119}:
        raise ValueError(f"Unexpected diagnosis counts: {diagnosis_counts}")

    deploy_files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "data_manifest.json"
    )
    too_large = {
        str(path.relative_to(output)): path.stat().st_size
        for path in deploy_files
        if path.stat().st_size >= 95 * 1024 * 1024
    }
    if too_large:
        raise ValueError(f"Deploy files exceed the 95-MiB release cap: {too_large}")

    full_manifest = module_set_manifests["full_cohort"]
    expansion_grains = {
        "bonobo_networks": sum(config.module_count * 450 for config in configs),
        "bonobo_aggregate_plot_rows": sum(
            rule_rows["aggregate_plot_rows"]
            for expansion in expansion_manifests.values()
            for rule_rows in expansion["bonobo"].values()
        ),
        "bonobo_resolved_plot_rows": sum(
            rule_rows["resolved_plot_rows"]
            for expansion in expansion_manifests.values()
            for rule_rows in expansion["bonobo"].values()
        ),
        "bonobo_aggregate_stat_rows": sum(
            rule_rows["aggregate_stat_rows"]
            for expansion in expansion_manifests.values()
            for rule_rows in expansion["bonobo"].values()
        ),
        "bonobo_resolved_stat_rows": sum(
            rule_rows["resolved_stat_rows"]
            for expansion in expansion_manifests.values()
            for rule_rows in expansion["bonobo"].values()
        ),
        "resolved_mdc_rows": sum(
            expansion["resolved_mdc_rows"]
            for expansion in expansion_manifests.values()
        ),
        "edge_summary_rows": sum(
            sum(expansion["edge_summaries"]["lioness"].values())
            + sum(expansion["edge_summaries"]["bonobo"].values())
            for expansion in expansion_manifests.values()
        ),
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": RUN_NAME,
        "methods": full_manifest["methods"],
        "donors_per_method": 450,
        "diagnosis_counts_per_method": {"Control": 164, "MCI": 119, "AD": 167},
        "modules": 154,
        "phenotypes": list(PHENOTYPES),
        "feature_families": 6,
        "feature_families_by_estimator": {"lioness": 6, "bonobo": 3},
        "estimators": ["lioness", "bonobo"],
        "outcomes": list(
            (*PHENOTYPES, "age_at_death", "education_years", "cogdx", "braak_stage", "cerad_score", "adnc", "parkinsonism")
        ),
        "bonobo_significance": {
            "native_p05": "two-sided posterior covariance-edge p < 0.05",
            "bh_fdr05": (
                "BH FDR < 0.05 within each donor-module across undirected edges"
            ),
        },
        "bonobo_analysis": bonobo_analysis,
        "analysis_grains": expansion_grains,
        "edge_summary_identities": [
            "retained=positive+negative+zero",
            "possible=retained+pruned",
            "signed_weight_sum=positive_weight_sum-negative_weight_magnitude",
            "absolute_weight_sum=positive_weight_sum+negative_weight_magnitude",
            "total=TS+CT for every additive edge statistic",
            "TS=sum of AC,DLPFC,PCG blocks",
            "CT=sum of AC-DLPFC,AC-PCG,DLPFC-PCG blocks",
        ],
        "transformations": {
            "raw": "untransformed module/component score",
            "asinh": (
                "asinh(raw/robust_scale), with scale chosen from stable median, "
                "MAD, IQR, or SD fallbacks"
            ),
            "rint": (
                "rank inverse-normal Z-score within module, feature, and component "
                "across all 450 donors, separately by estimator/method and edge rule"
            ),
        },
        "privacy": (
            "donor and projid were removed; sample_id is a random-salted HMAC label "
            "whose salt and source mapping were discarded. Selected deidentified clinical "
            "and neuropathology fields are included for color, hover, and correlation views."
        ),
        "mdc": full_manifest["mdc"],
        "resolved_mdc": {"components": 6, **resolved_mdc_design},
        "module_details": full_manifest["module_details"],
        "rows": {"metadata_rows": metadata_rows, **full_manifest["rows"]},
        "module_sets": module_set_manifests,
        "files": {
            str(path.relative_to(output)): deploy_file_manifest_entry(path)
            for path in deploy_files
        },
    }
    (output / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
