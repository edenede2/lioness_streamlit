#!/usr/bin/env python3
"""Validate a staged ROSMAP single-sample-network public data bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


MODULE_SETS = {
    "full_cohort": {
        "directory": ".",
        "modules": 154,
        "methods": ("standard", "control_anchored"),
    },
    "control_derived": {
        "directory": "control_derived",
        "modules": 186,
        "methods": ("control_anchored",),
    },
}
RULES = ("all", "native_p05", "bh_fdr05")
DIFFERENTIAL_FDR_SCOPES = ("global", "per_module")
DIFFERENTIAL_FDR_THRESHOLDS = (0.05, 0.10)
DIFFERENTIAL_NORMALIZATIONS = ("standard_pruned", "retained_edge")
OUTCOMES = {
    "cogn_global", "cogng_demog_slope", "cogng_path_slope",
    "motor10_demog_slope", "sqrt_parksc_demog_slope", "age_at_death",
    "education_years", "cogdx", "braak_stage", "cerad_score", "adnc",
    "parkinsonism",
}
FORBIDDEN_COLUMNS = {"donor", "projid"}
INTERNAL_TISSUE_LABELS = ("MFBA9BA46", "MFBA9/BA46")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Staged data directory to validate")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_parquet_private_and_publicly_labeled(path: Path) -> None:
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    if FORBIDDEN_COLUMNS.intersection(names):
        raise ValueError(f"Private identifier column remains in {path}")
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
        for internal in INTERNAL_TISSUE_LABELS:
            if bool(pc.any(pc.match_substring(values, internal)).as_py()):
                raise ValueError(f"Internal tissue label {internal} remains in {path}")


def validate_edge_file(path: Path, expected_rows: int, expected_samples: set[str]) -> None:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(
            f"{path}: expected {expected_rows} edge rows, found {parquet.metadata.num_rows}"
        )
    expected_row_groups = expected_rows // (450 * 9)
    if parquet.metadata.num_row_groups != expected_row_groups:
        raise ValueError(
            f"{path}: expected one predicate-friendly row group per module "
            f"({expected_row_groups}), found {parquet.metadata.num_row_groups}"
        )
    seen_samples: set[str] = set()
    columns = [
        "sample_id", "n_possible_edges", "n_retained_edges", "n_positive_edges",
        "n_negative_edges", "n_zero_edges", "n_pruned_edges", "signed_weight_sum",
        "positive_weight_sum", "negative_weight_magnitude", "absolute_weight_sum",
    ]
    for batch in parquet.iter_batches(batch_size=150_000, columns=columns):
        frame = batch.to_pandas()
        seen_samples.update(frame["sample_id"].dropna().astype(str))
        available = frame["n_possible_edges"].gt(0)
        selected = frame.loc[available]
        retained = pd.to_numeric(selected["n_retained_edges"], errors="raise")
        positive = pd.to_numeric(selected["n_positive_edges"], errors="raise")
        negative = pd.to_numeric(selected["n_negative_edges"], errors="raise")
        zero = pd.to_numeric(selected["n_zero_edges"], errors="raise")
        pruned = pd.to_numeric(selected["n_pruned_edges"], errors="raise")
        possible = pd.to_numeric(selected["n_possible_edges"], errors="raise")
        if not np.array_equal(retained, positive + negative + zero):
            raise ValueError(f"{path}: retained sign-count identity failed")
        if not np.array_equal(possible, retained + pruned):
            raise ValueError(f"{path}: possible/retained/pruned identity failed")
        signed = pd.to_numeric(selected["signed_weight_sum"], errors="coerce")
        positive_weight = pd.to_numeric(selected["positive_weight_sum"], errors="coerce")
        negative_weight = pd.to_numeric(
            selected["negative_weight_magnitude"], errors="coerce"
        )
        absolute = pd.to_numeric(selected["absolute_weight_sum"], errors="coerce")
        if not np.allclose(signed, positive_weight - negative_weight, rtol=1e-9, atol=1e-9):
            raise ValueError(f"{path}: signed weight identity failed")
        if not np.allclose(absolute, positive_weight + negative_weight, rtol=1e-9, atol=1e-9):
            raise ValueError(f"{path}: absolute weight identity failed")
    if seen_samples != expected_samples:
        raise ValueError(f"{path}: anonymous sample IDs are inconsistent")
    scope_children = {
        "total": ["TS", "CT"],
        "TS": ["TS_AC", "TS_DLPFC", "TS_PCGBA23"],
        "CT": [
            "CT_AC__DLPFC", "CT_AC__PCGBA23", "CT_DLPFC__PCGBA23"
        ],
    }
    additive = [
        "n_possible_edges", "n_retained_edges", "n_positive_edges",
        "n_negative_edges", "n_zero_edges", "n_pruned_edges",
        "signed_weight_sum", "positive_weight_sum", "negative_weight_magnitude",
        "absolute_weight_sum",
    ]
    identity_columns = ["sample_id", "module", "edge_rule", "scope", *additive]
    for row_group in range(parquet.metadata.num_row_groups):
        module_frame = parquet.read_row_group(
            row_group, columns=identity_columns
        ).to_pandas()
        if module_frame["module"].nunique() != 1:
            raise ValueError(f"{path}: edge row group spans multiple modules")
        for metric in additive:
            pivot = module_frame.pivot(
                index=["sample_id", "module", "edge_rule"],
                columns="scope",
                values=metric,
            ).apply(pd.to_numeric, errors="coerce").fillna(0.0)
            for parent, children in scope_children.items():
                if not np.allclose(
                    pivot[parent], pivot[children].sum(axis=1),
                    rtol=1e-9, atol=1e-9,
                ):
                    raise ValueError(
                        f"{path}: {metric} scope identity failed for {parent}"
                    )


def validate_differential_module_set(
    directory: Path,
    *,
    key: str,
    modules: int,
    methods: tuple[str, ...],
    expected_samples: set[str],
) -> dict[str, int]:
    """Validate both BH scopes without disturbing the legacy all-edge bundle."""

    root = directory / "differential"
    if not root.exists():
        raise ValueError(f"{key}: differential bundle is missing")
    variants = 0
    aggregate_rows = 0
    resolved_rows = 0
    statistics_rows = 0
    edge_rows = 0
    method_catalog = {"lioness": methods, "bonobo": ("bonobo",)}
    for estimator, network_methods in method_catalog.items():
        edge_rules = ("all",) if estimator == "lioness" else RULES
        feature_count = 6 if estimator == "lioness" else 3
        for method in network_methods:
            for fdr_scope in DIFFERENTIAL_FDR_SCOPES:
                for fdr_threshold in DIFFERENTIAL_FDR_THRESHOLDS:
                    for edge_rule in edge_rules:
                        for normalization in DIFFERENTIAL_NORMALIZATIONS:
                            variant = (
                                root
                                / estimator
                                / method
                                / "ad_control_discovery_fdr05"
                                / fdr_scope
                                / f"fdr_{fdr_threshold:.2f}"
                                / edge_rule
                                / normalization
                            )
                            expected = {
                                "aggregate_plot_data.parquet": modules * 450 * feature_count,
                                "resolved_plot_data.parquet": (
                                    modules * 450 * feature_count * 6
                                ),
                                # all donors contributes 3 diagnosis rows; discovery and
                                # validation contribute 2 each; external MCI contributes 1.
                                "aggregate_statistics.parquet": (
                                    modules * feature_count * len(OUTCOMES) * 8
                                ),
                                "resolved_statistics.parquet": (
                                    modules * feature_count * 6 * len(OUTCOMES) * 8
                                ),
                            }
                            for filename, expected_rows in expected.items():
                                path = variant / filename
                                if not path.exists():
                                    raise ValueError(f"{key}: missing differential file {path}")
                                observed = pq.ParquetFile(path).metadata.num_rows
                                if observed != expected_rows:
                                    raise ValueError(
                                        f"{key}/{path.relative_to(directory)}: expected "
                                        f"{expected_rows} rows, found {observed}"
                                    )
                                if "plot_data" in filename:
                                    variant_columns = {
                                        "differential_edge_rule",
                                        "differential_fdr_scope",
                                        "differential_fdr_threshold",
                                        "score_normalization",
                                        "ad_control_split",
                                    }
                                    schema_names = set(
                                        pq.ParquetFile(path).schema_arrow.names
                                    )
                                    if not variant_columns.issubset(schema_names):
                                        raise ValueError(
                                            f"{key}/{path.relative_to(directory)}: "
                                            "differential provenance fields are incomplete"
                                        )
                                    provenance = pq.read_table(
                                        path,
                                        columns=[
                                            "differential_fdr_scope",
                                            "differential_fdr_threshold",
                                            "score_normalization",
                                        ],
                                    ).to_pandas()
                                    if set(
                                        provenance["differential_fdr_scope"].astype(str)
                                    ) != {fdr_scope}:
                                        raise ValueError(
                                            f"{key}/{path.relative_to(directory)}: "
                                            "incorrect differential FDR scope"
                                        )
                                    if set(
                                        pd.to_numeric(
                                            provenance["differential_fdr_threshold"],
                                            errors="raise",
                                        ).round(2)
                                    ) != {fdr_threshold}:
                                        raise ValueError(
                                            f"{key}/{path.relative_to(directory)}: "
                                            "incorrect differential FDR threshold"
                                        )
                                    if set(
                                        provenance["score_normalization"].astype(str)
                                    ) != {normalization}:
                                        raise ValueError(
                                            f"{key}/{path.relative_to(directory)}: "
                                            "incorrect score normalization"
                                        )
                                    samples = set(
                                        pc.unique(
                                            pq.read_table(path, columns=["sample_id"])[
                                                "sample_id"
                                            ]
                                        )
                                        .cast(pa.string())
                                        .to_pylist()
                                    )
                                    if samples != expected_samples:
                                        raise ValueError(
                                            f"{key}/{path.relative_to(directory)}: "
                                            "anonymous samples mismatch"
                                        )
                            variants += 1
                            aggregate_rows += expected["aggregate_plot_data.parquet"]
                            resolved_rows += expected["resolved_plot_data.parquet"]
                            statistics_rows += (
                                expected["aggregate_statistics.parquet"]
                                + expected["resolved_statistics.parquet"]
                            )
            for edge_rule in edge_rules:
                edge_path = (
                    root
                    / "edge_summaries"
                    / f"{estimator}__{method}__{edge_rule}.parquet"
                )
                expected_edge_rows = modules * 450 * 9 * 4
                if pq.ParquetFile(edge_path).metadata.num_rows != expected_edge_rows:
                    raise ValueError(f"{key}/{edge_path.name}: differential row mismatch")
                selections = pq.read_table(
                    edge_path,
                    columns=["differential_fdr_scope", "differential_fdr_threshold"],
                ).to_pandas()
                if set(selections["differential_fdr_scope"].astype(str)) != set(
                    DIFFERENTIAL_FDR_SCOPES
                ):
                    raise ValueError(f"{key}/{edge_path.name}: FDR scopes are incomplete")
                observed_thresholds = set(
                    pd.to_numeric(
                        selections["differential_fdr_threshold"], errors="raise"
                    ).round(2)
                )
                if observed_thresholds != set(DIFFERENTIAL_FDR_THRESHOLDS):
                    raise ValueError(
                        f"{key}/{edge_path.name}: FDR thresholds are incomplete"
                    )
                edge_rows += expected_edge_rows

    candidates = root / "volcano_candidates.parquet"
    candidate_schema = set(pq.read_schema(candidates).names)
    required_q = {
        "discovery_fdr_global",
        "discovery_fdr_per_module",
        "validation_fdr_global",
        "validation_fdr_per_module",
    }
    if not required_q.issubset(candidate_schema):
        raise ValueError(f"{key}: volcano candidates omit one or more BH columns")
    bins = pq.read_table(
        root / "volcano_bins.parquet", columns=["fdr_scope", "counts", "n_edges"]
    )
    if set(pc.unique(bins["fdr_scope"]).cast(pa.string()).to_pylist()) != set(
        DIFFERENTIAL_FDR_SCOPES
    ):
        raise ValueError(f"{key}: volcano bins do not contain both BH scopes")
    bin_frame = bins.select(["counts", "n_edges"]).to_pandas()
    represented = bin_frame["counts"].map(
        lambda values: int(np.asarray(values).sum())
    )
    if not np.array_equal(
        represented.to_numpy(),
        pd.to_numeric(bin_frame["n_edges"], errors="raise").astype(int).to_numpy(),
    ):
        raise ValueError(f"{key}: volcano bins do not account for every edge")
    return {
        "variants": variants,
        "aggregate_rows": aggregate_rows,
        "resolved_rows": resolved_rows,
        "statistics_rows": statistics_rows,
        "edge_rows": edge_rows,
    }


def validate_module_set(root: Path, key: str, expected_samples: set[str]) -> dict[str, int]:
    config = MODULE_SETS[key]
    directory = root / str(config["directory"])
    modules = int(config["modules"])
    methods = tuple(config["methods"])
    annotations = pd.read_csv(directory / "module_kegg_annotations.tsv", sep="\t")
    details = pd.read_csv(directory / "module_details.tsv", sep="\t")
    if annotations["module"].nunique() != modules or details["module"].nunique() != modules:
        raise ValueError(f"{key}: module catalog mismatch")
    if not details["tissue_entropy_normalized"].between(0, 1).all():
        raise ValueError(f"{key}: normalized tissue entropy outside [0,1]")

    expected_lioness = {
        "aggregate_plot_data.parquet": len(methods) * modules * 450 * 6,
        "resolved_plot_data.parquet": len(methods) * modules * 450 * 6 * 6,
        "aggregate_statistics.parquet": len(methods) * modules * 6 * 12 * 3,
        "resolved_statistics.parquet": len(methods) * modules * 6 * 6 * 12 * 3,
    }
    for filename, rows in expected_lioness.items():
        path = directory / filename
        if pq.ParquetFile(path).metadata.num_rows != rows:
            raise ValueError(f"{key}/{filename}: row count mismatch")
        if "plot_data" in filename:
            samples = set(
                pc.unique(pq.read_table(path, columns=["sample_id"])["sample_id"])
                .cast(pa.string())
                .to_pylist()
            )
            if samples != expected_samples:
                raise ValueError(f"{key}/{filename}: anonymous samples mismatch")
    aggregate_stats = pd.read_parquet(
        directory / "aggregate_statistics.parquet", columns=["phenotype"]
    )
    if set(aggregate_stats["phenotype"].astype(str)) != OUTCOMES:
        raise ValueError(f"{key}: LIONESS statistics do not cover all 12 outcomes")

    for rule in RULES:
        bonobo = directory / "bonobo" / rule
        expected_bonobo = {
            "aggregate_plot_data.parquet": modules * 450 * 3,
            "resolved_plot_data.parquet": modules * 450 * 3 * 6,
            "aggregate_statistics.parquet": modules * 3 * 12 * 3,
            "resolved_statistics.parquet": modules * 3 * 6 * 12 * 3,
        }
        for filename, rows in expected_bonobo.items():
            path = bonobo / filename
            if pq.ParquetFile(path).metadata.num_rows != rows:
                raise ValueError(f"{key}/bonobo/{rule}/{filename}: row count mismatch")
            if "plot_data" in filename:
                samples = set(
                    pc.unique(pq.read_table(path, columns=["sample_id"])["sample_id"])
                    .cast(pa.string())
                    .to_pylist()
                )
                if samples != expected_samples:
                    raise ValueError(
                        f"{key}/bonobo/{rule}/{filename}: anonymous samples mismatch"
                    )
        statistics = pd.read_parquet(
            bonobo / "aggregate_statistics.parquet", columns=["phenotype"]
        )
        if set(statistics["phenotype"].astype(str)) != OUTCOMES:
            raise ValueError(f"{key}/{rule}: BONOBO statistics omit outcomes")

    resolved_mdc = pd.read_csv(directory / "mdc_resolved_ad_vs_control.tsv", sep="\t")
    if len(resolved_mdc) != modules * 6 or resolved_mdc["component"].nunique() != 6:
        raise ValueError(f"{key}: resolved MDC row/component count mismatch")
    if not resolved_mdc["mdc"].isna().eq(resolved_mdc["n_edges"].eq(0)).all():
        raise ValueError(
            f"{key}: resolved MDC availability does not match structural edge presence"
        )
    unavailable_mdc = resolved_mdc["n_edges"].eq(0)
    inference_columns = [
        "p_loss_sample", "p_loss_gene", "q_loss_sample", "q_loss_gene",
        "p_gain_sample", "p_gain_gene", "q_gain_sample", "q_gain_gene",
        "directional_p_sample", "directional_p_gene", "directional_fdr",
    ]
    if resolved_mdc.loc[unavailable_mdc, inference_columns].notna().any().any():
        raise ValueError(f"{key}: unavailable MDC blocks have inferential values")
    for method in methods:
        validate_edge_file(
            directory / "edge_summaries" / f"lioness__{method}.parquet",
            modules * 450 * 9,
            expected_samples,
        )
    for rule in RULES:
        validate_edge_file(
            directory / "edge_summaries" / f"bonobo__{rule}.parquet",
            modules * 450 * 9,
            expected_samples,
        )
    differential = validate_differential_module_set(
        directory,
        key=key,
        modules=modules,
        methods=methods,
        expected_samples=expected_samples,
    )
    return {
        "modules": modules,
        "lioness_aggregate_rows": expected_lioness["aggregate_plot_data.parquet"],
        "bonobo_networks": modules * 450,
        "resolved_mdc_rows": len(resolved_mdc),
        "edge_rows": modules * 450 * 9 * (len(methods) + len(RULES)),
        "differential": differential,
    }


def main() -> None:
    root = parse_args().data.resolve()
    manifest_path = root / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = pd.read_parquet(root / "sample_metadata.parquet")
    if len(metadata) != 450 or metadata["sample_id"].nunique() != 450:
        raise ValueError("Sample metadata must contain 450 unique anonymous donors")
    counts = metadata["diagnosis_group"].value_counts().to_dict()
    if counts != {"AD": 167, "Control": 164, "MCI": 119}:
        raise ValueError(f"Unexpected diagnosis counts: {counts}")
    expected_samples = set(metadata["sample_id"].astype(str))

    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        if path.stat().st_size >= 95 * 1024 * 1024:
            raise ValueError(f"Deploy file is at least 95 MiB: {path}")
        if path.suffix == ".parquet":
            assert_parquet_private_and_publicly_labeled(path)
        elif path.suffix == ".tsv":
            contents = path.read_text(encoding="utf-8")
            header = contents.splitlines()[0].split("\t") if contents else []
            if FORBIDDEN_COLUMNS.intersection(header):
                raise ValueError(f"Private identifier column remains in {path}")
            if any(label in contents for label in INTERNAL_TISSUE_LABELS):
                raise ValueError(f"Internal tissue label remains in {path}")
    observed_manifest_files = {
        str(path.relative_to(root))
        for path in files
        if path.name != "data_manifest.json"
    }
    if observed_manifest_files != set(manifest["files"]):
        raise ValueError(
            "Manifest file catalog mismatch: "
            f"missing={sorted(observed_manifest_files - set(manifest['files']))}, "
            f"stale={sorted(set(manifest['files']) - observed_manifest_files)}"
        )
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.exists() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Manifest size mismatch: {relative}")
        if sha256(path) != expected["sha256"]:
            raise ValueError(f"Manifest hash mismatch: {relative}")

    module_results = {
        key: validate_module_set(root, key, expected_samples) for key in MODULE_SETS
    }
    result = {
        "status": "passed",
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "samples": len(expected_samples),
        "diagnosis_counts": counts,
        "module_sets": module_results,
        "bonobo_networks": sum(values["bonobo_networks"] for values in module_results.values()),
        "resolved_mdc_rows": sum(values["resolved_mdc_rows"] for values in module_results.values()),
        "edge_rows": sum(values["edge_rows"] for values in module_results.values()),
    }
    expected_grains = {
        "bonobo_networks": 153_000,
        "bonobo_aggregate_plot_rows": 1_377_000,
        "bonobo_resolved_plot_rows": 8_262_000,
        "bonobo_aggregate_stat_rows": 110_160,
        "bonobo_resolved_stat_rows": 660_960,
        "resolved_mdc_rows": 2_040,
        "edge_summary_rows": 6_131_700,
    }
    if manifest.get("analysis_grains") != expected_grains:
        raise ValueError(
            f"Analysis-grain manifest mismatch: {manifest.get('analysis_grains')}"
        )
    if result["bonobo_networks"] != expected_grains["bonobo_networks"]:
        raise ValueError("BONOBO donor-module network count mismatch")
    if result["resolved_mdc_rows"] != expected_grains["resolved_mdc_rows"]:
        raise ValueError("Resolved MDC total row count mismatch")
    if result["edge_rows"] != expected_grains["edge_summary_rows"]:
        raise ValueError("Edge-summary total row count mismatch")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
