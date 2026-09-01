#!/usr/bin/env python3
"""Validate the capability-limited all-donor CorShrink public bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


APP_ROOT = Path(__file__).resolve().parents[1]
MODULE_SET = "all_donor_corrshrink_l4"
MODULES = 138
METHODS = ("standard", "control_anchored")
COMPONENT_COUNTS = {
    "TS_AC": 730,
    "TS_DLPFC": 1216,
    "TS_PCGBA23": 659,
    "CT_AC__DLPFC": 694,
    "CT_AC__PCGBA23": 478,
    "CT_DLPFC__PCGBA23": 640,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=APP_ROOT / "data")
    return parser.parse_args()


def assert_public_file(path: Path) -> None:
    if path.stat().st_size >= 95 * 1024 * 1024:
        raise ValueError(f"File exceeds 95 MiB: {path}")
    if path.suffix == ".parquet":
        schema = pq.ParquetFile(path).schema_arrow
        if {"donor", "projid"}.intersection(schema.names):
            raise ValueError(f"Private identifier in {path}")


def validate_primary(root: Path) -> dict[str, int]:
    aggregate = pd.read_parquet(root / "aggregate_plot_data.parquet")
    resolved = pd.read_parquet(root / "resolved_plot_data.parquet")
    expected_aggregate = MODULES * 450 * 6 * len(METHODS)
    expected_resolved = expected_aggregate * 6
    if len(aggregate) != expected_aggregate or len(resolved) != expected_resolved:
        raise ValueError("Primary plot-data row counts differ from expectations")
    for method in METHODS:
        subset = aggregate.loc[aggregate["lioness_method"].eq(method)]
        if subset["sample_id"].nunique() != 450 or subset["module"].nunique() != MODULES:
            raise ValueError(f"{method}: primary data are not rectangular")
        counts = (
            subset[["sample_id", "diagnosis_group"]].drop_duplicates()["diagnosis_group"]
            .value_counts().to_dict()
        )
        if counts != {"AD": 167, "Control": 164, "MCI": 119}:
            raise ValueError(f"{method}: diagnosis counts differ: {counts}")

    # Check CT/TS sums against their six resolved components for additive families.
    components = resolved.pivot_table(
        index=["sample_id", "module", "lioness_method", "metric_family"],
        columns="component", values="metric_raw", aggfunc="first",
    )
    pooled = aggregate.set_index(
        ["sample_id", "module", "lioness_method", "metric_family"]
    )
    additive = {"connectivity", "abs_sum", "positive_abs_sum", "negative_abs_sum"}
    selected = components.index.get_level_values("metric_family").isin(additive)
    components = components.loc[selected]
    pooled = pooled.loc[components.index]
    ts_sum = components[["TS_AC", "TS_DLPFC", "TS_PCGBA23"]].sum(axis=1)
    ct_sum = components[[
        "CT_AC__DLPFC", "CT_AC__PCGBA23", "CT_DLPFC__PCGBA23",
    ]].sum(axis=1)
    for observed, expected, label in (
        (ts_sum, pooled["TS_raw"], "TS"), (ct_sum, pooled["CT_raw"], "CT"),
    ):
        denominator = np.maximum(1.0, np.maximum(np.abs(observed), np.abs(expected)))
        error = np.nanmax(np.abs(observed - expected) / denominator)
        if not np.isfinite(error) or error > 1e-10:
            raise ValueError(f"{label} resolved/pooled identity failed: {error}")
    return {"aggregate_rows": len(aggregate), "resolved_rows": len(resolved)}


def validate_expanded(root: Path) -> dict[str, int]:
    metadata = pd.read_parquet(root / "sample_metadata.parquet")
    if metadata["sample_id"].duplicated().any() or len(metadata) < 1216:
        raise ValueError("Expanded metadata is not a donor-level union")
    total = 0
    for method in METHODS:
        frame = pd.read_parquet(root / method / "resolved_plot_data.parquet")
        total += len(frame)
        if frame["module"].nunique() != MODULES:
            raise ValueError(f"{method}: expanded modules are incomplete")
        observed = (
            frame[["component", "sample_id"]].drop_duplicates()
            .groupby("component", observed=True)["sample_id"].nunique().to_dict()
        )
        if observed != COMPONENT_COUNTS:
            raise ValueError(f"{method}: expanded component counts differ: {observed}")
        expected_rows = MODULES * 6 * sum(COMPONENT_COUNTS.values())
        if len(frame) != expected_rows:
            raise ValueError(f"{method}: expanded row count differs")
    return {"metadata_rows": len(metadata), "plot_rows": total}


def main() -> None:
    data_root = parse_args().data_root.resolve()
    root = data_root / MODULE_SET
    manifest = json.loads((data_root / "data_manifest.json").read_text())
    declared = manifest.get("module_sets", {}).get(MODULE_SET, {})
    if declared.get("status") != "complete" or declared.get("modules") != MODULES:
        raise ValueError("Parent manifest does not declare the completed 138-module set")
    details = pd.read_csv(root / "module_details.tsv", sep="\t")
    annotations = pd.read_csv(root / "module_kegg_annotations.tsv", sep="\t")
    if details["module"].nunique() != MODULES or annotations["module"].nunique() != MODULES:
        raise ValueError("Module details/annotations do not cover all 138 modules")
    proportions = details[["proportion_ac", "proportion_dlpfc", "proportion_pcg"]]
    if not np.allclose(proportions.sum(axis=1), 1.0):
        raise ValueError("Tissue proportions do not sum to one")
    if not details["tissue_entropy_normalized"].between(0, 1).all():
        raise ValueError("Normalized Shannon entropy is outside [0,1]")
    mdc = pd.read_csv(root / "mdc_ad_vs_control_summary.tsv", sep="\t")
    resolved_mdc = pd.read_csv(root / "mdc_resolved_ad_vs_control.tsv", sep="\t")
    if mdc["module"].nunique() != MODULES or len(resolved_mdc) != MODULES * 6:
        raise ValueError("MDC coverage differs from 138 modules × six components")
    for path in root.rglob("*"):
        if path.is_file():
            assert_public_file(path)
    result = {
        "status": "passed", "modules": MODULES,
        "primary": validate_primary(root),
        "expanded": validate_expanded(root / "expanded"),
        "mdc_rows": len(mdc), "resolved_mdc_rows": len(resolved_mdc),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
