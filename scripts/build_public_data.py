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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


RUN_NAME = "20260817_standard_control_anchored_allmodules_5phenotypes_6features"
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
        "--kegg",
        type=Path,
        default=repo_root
        / "out"
        / "kegg_enrichments_per_se2_level4"
        / "method4_tissue_expanded_kegg_annotated.tsv",
        help="Full annotated tissue-expanded KEGG table.",
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
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Streamlit deploy data directory.",
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
    frame.insert(0, "sample_id", frame.pop("donor").astype(str).map(sample_map))
    if frame["sample_id"].isna().any() or frame["sample_id"].nunique() != 450:
        raise ValueError("Sample metadata did not map cleanly to all 450 pseudonyms")
    frame = frame.rename(columns=METADATA_RENAME)
    frame["sex_code"] = frame["sex_code"].map({0: "Code 0", 1: "Code 1"}).astype("string")
    frame["apoe_genotype"] = frame["apoe_genotype"].map(format_apoe_genotype).astype("string")
    frame["parkinsonism_label"] = frame["parkinsonism"].map(
        {0.0: "No", 1.0: "Yes"}
    ).astype("string")
    frame["age_at_death"] = pd.to_numeric(frame["age_at_death"], errors="coerce").round(1)
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
    frame.to_parquet(output, index=False, compression="zstd")
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
    frame.to_parquet(output, index=False, compression="zstd")
    return len(frame)


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
            "Level-4 module size and tissue composition. MFBA9BA46 is displayed as "
            "DLPFC and PCGBA23 is displayed as PCG. Tissue proportions use module size "
            "as the denominator."
        ),
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
    manifest["files"][summary_path.name] = {
        "bytes": summary_path.stat().st_size,
        "sha256": sha256(summary_path),
    }
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

    manifest_path = output / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("rows", {})["module_details_rows"] = details_rows
    manifest["module_details"] = module_details_manifest(source)
    manifest["files"][details_path.name] = {
        "bytes": details_path.stat().st_size,
        "sha256": sha256(details_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def refresh_existing_statistics_bundle(
    analysis_root: Path, output: Path
) -> dict[str, object]:
    """Refresh statistics and their manifest entries without changing donor labels."""
    aggregate_sources = [
        source_file(
            analysis_root,
            method,
            "_robust_statistics.parquet",
            exclude_text="tissue_resolved",
        )
        for method in METHODS
    ]
    resolved_sources = [
        source_file(analysis_root, method, "_tissue_resolved_robust_statistics.parquet")
        for method in METHODS
    ]
    aggregate_path = output / "aggregate_statistics.parquet"
    resolved_path = output / "resolved_statistics.parquet"
    aggregate_rows = write_combined_statistics(
        aggregate_sources, aggregate_path, resolved=False
    )
    resolved_rows = write_combined_statistics(
        resolved_sources, resolved_path, resolved=True
    )
    expected = {
        "aggregate_stat_rows": 2 * 154 * 5 * 6 * 3,
        "resolved_stat_rows": 2 * 154 * 5 * 6 * 6 * 3,
    }
    observed = {
        "aggregate_stat_rows": aggregate_rows,
        "resolved_stat_rows": resolved_rows,
    }
    if observed != expected:
        raise ValueError(
            f"Statistics row-count validation failed: observed={observed}, expected={expected}"
        )

    manifest_path = output / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("rows", {}).update(observed)
    for path in (aggregate_path, resolved_path):
        manifest["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    analysis_root = args.analysis_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mdc_source = args.mdc.resolve()
    module_details_source = args.module_details.resolve()

    refresh_flags = [args.mdc_only, args.module_details_only, args.statistics_only]
    if sum(refresh_flags) > 1:
        raise ValueError(
            "Use only one of --mdc-only, --module-details-only, or --statistics-only"
        )

    if args.mdc_only:
        manifest = refresh_existing_mdc_bundle(mdc_source, output)
        print(json.dumps(manifest["mdc"], indent=2))
        return
    if args.module_details_only:
        manifest = refresh_existing_module_details_bundle(module_details_source, output)
        print(json.dumps(manifest["module_details"], indent=2))
        return
    if args.statistics_only:
        manifest = refresh_existing_statistics_bundle(analysis_root, output)
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

    aggregate_sources = [
        source_file(analysis_root, method, "_transformed_component_data.parquet")
        for method in METHODS
    ]
    resolved_sources = [
        source_file(analysis_root, method, "_tissue_resolved_component_data.parquet")
        for method in METHODS
    ]
    aggregate_stat_sources = [
        source_file(
            analysis_root,
            method,
            "_robust_statistics.parquet",
            exclude_text="tissue_resolved",
        )
        for method in METHODS
    ]
    resolved_stat_sources = [
        source_file(analysis_root, method, "_tissue_resolved_robust_statistics.parquet")
        for method in METHODS
    ]

    sample_map = create_sample_map(aggregate_sources[0])
    metadata_rows = write_sample_metadata(
        analysis_root / "standard" / "data" / "phenotypes.parquet",
        output / "sample_metadata.parquet",
        sample_map,
    )
    aggregate_rows = write_sanitized_plot_data(
        aggregate_sources,
        output / "aggregate_plot_data.parquet",
        sample_map,
        resolved=False,
    )
    resolved_rows = write_sanitized_plot_data(
        resolved_sources,
        output / "resolved_plot_data.parquet",
        sample_map,
        resolved=True,
    )
    aggregate_stat_rows = write_combined_statistics(
        aggregate_stat_sources,
        output / "aggregate_statistics.parquet",
        resolved=False,
    )
    resolved_stat_rows = write_combined_statistics(
        resolved_stat_sources,
        output / "resolved_statistics.parquet",
        resolved=True,
    )

    kegg = pd.read_csv(args.kegg, sep="\t")
    kegg["cluster_id"] = pd.to_numeric(kegg["cluster_id"], errors="raise").astype("int32")
    kegg = kegg.rename(columns={"overlap_MFBA9BA46": "overlap_DLPFC"})
    if "overlap_genes" in kegg:
        kegg["overlap_genes"] = kegg["overlap_genes"].astype("string").str.replace(
            "(MFBA9BA46)", "(DLPFC)", regex=False
        )
    kegg.to_parquet(output / "kegg_tissue_expanded_full.parquet", index=False, compression="zstd")
    kegg.to_csv(output / "kegg_tissue_expanded_full.tsv", sep="\t", index=False)

    annotation_source = table_file(
        analysis_root, "standard", "_module_kegg_annotations.tsv"
    )
    annotations = pd.read_csv(annotation_source, sep="\t")
    annotations.to_csv(output / "module_kegg_annotations.tsv", sep="\t", index=False)
    mdc_rows = write_mdc_summary(
        mdc_source,
        output / MDC_OUTPUT_NAME,
        set(annotations["module"].astype(int)),
    )
    module_details_rows = write_module_details(
        module_details_source,
        output / MODULE_DETAILS_OUTPUT_NAME,
        set(annotations["module"].astype(int)),
    )

    features = pd.read_csv(analysis_root / "feature_definitions.tsv", sep="\t")
    features = features.replace("MFBA9BA46", "DLPFC", regex=True)
    features.to_csv(output / "feature_definitions.tsv", sep="\t", index=False)
    tissues = pd.read_csv(analysis_root / "tissue_mapping.tsv", sep="\t")
    tissues["internal_tissue"] = tissues["internal_tissue"].replace(
        {"MFBA9BA46": "DLPFC"}
    )
    tissues["display_name"] = tissues["display_name"].replace(
        {"DLPFC (MFBA9/BA46)": "DLPFC"}
    )
    tissues[["internal_tissue", "display_name"]].to_csv(
        output / "tissue_mapping.tsv", sep="\t", index=False
    )

    expected = {
        "aggregate_rows": 2 * 450 * 154 * 6,
        "resolved_rows": 2 * 450 * 154 * 6 * 6,
        "aggregate_stat_rows": 2 * 154 * 5 * 6 * 3,
        "resolved_stat_rows": 2 * 154 * 5 * 6 * 6 * 3,
        "mdc_rows": 154,
        "module_details_rows": 154,
    }
    observed = {
        "metadata_rows": metadata_rows,
        "aggregate_rows": aggregate_rows,
        "resolved_rows": resolved_rows,
        "aggregate_stat_rows": aggregate_stat_rows,
        "resolved_stat_rows": resolved_stat_rows,
        "mdc_rows": mdc_rows,
        "module_details_rows": module_details_rows,
    }
    expected["metadata_rows"] = 450
    if observed != expected:
        raise ValueError(f"Row-count validation failed: observed={observed}, expected={expected}")
    if 1918 not in set(annotations["module"].astype(int)):
        raise ValueError("M1918 is absent from the module annotation table")

    deploy_files = sorted(path for path in output.iterdir() if path.is_file())
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": RUN_NAME,
        "methods": list(METHODS),
        "donors_per_method": 450,
        "diagnosis_counts_per_method": {"Control": 164, "MCI": 119, "AD": 167},
        "modules": 154,
        "phenotypes": list(PHENOTYPES),
        "feature_families": 6,
        "privacy": (
            "donor and projid were removed; sample_id is a random-salted HMAC label "
            "whose salt and source mapping were discarded. Selected deidentified clinical "
            "and neuropathology fields are included for color, hover, and correlation views."
        ),
        "mdc": mdc_manifest(mdc_source),
        "module_details": module_details_manifest(module_details_source),
        "rows": observed,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in deploy_files
            if path.name != "data_manifest.json"
        },
    }
    (output / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
