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
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Streamlit deploy data directory.",
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
        frame["component"] = frame["component"].str.replace(
            "MFBA9BA46", "DLPFC", regex=False
        )
        frame["component_label"] = frame["component_label"].str.replace(
            "MFBA9/BA46", "DLPFC", regex=False
        )
    for column in (*value_columns, *PHENOTYPES):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame


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


def write_combined_statistics(sources: list[Path], output: Path) -> int:
    frames = [pd.read_parquet(path) for path in sources]
    frame = pd.concat(frames, ignore_index=True)
    frame["module"] = pd.to_numeric(frame["module"], errors="raise").astype("int32")
    frame.to_parquet(output, index=False, compression="zstd")
    return len(frame)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    analysis_root = args.analysis_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

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
        aggregate_stat_sources, output / "aggregate_statistics.parquet"
    )
    resolved_stat_rows = write_combined_statistics(
        resolved_stat_sources, output / "resolved_statistics.parquet"
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
    }
    observed = {
        "metadata_rows": metadata_rows,
        "aggregate_rows": aggregate_rows,
        "resolved_rows": resolved_rows,
        "aggregate_stat_rows": aggregate_stat_rows,
        "resolved_stat_rows": resolved_stat_rows,
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
