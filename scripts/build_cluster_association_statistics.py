#!/usr/bin/env python3
"""Add nominal ROSMAP clusters and build compact all-edge association statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app_helpers.charts import aggregate_to_long, resolved_to_long  # noqa: E402
from app_helpers.correlations import (  # noqa: E402
    add_categorical_across_module_fdr,
    calculate_categorical_associations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=APP_ROOT / "data")
    parser.add_argument(
        "--phenotype-source",
        type=Path,
        default=(
            REPO_ROOT
            / "out/lioness_reference_comparison_rosmap"
            / "20260817_standard_control_anchored_allmodules_5phenotypes_6features"
            / "standard/data/phenotypes.parquet"
        ),
    )
    parser.add_argument(
        "--cluster-source",
        type=Path,
        default=REPO_ROOT / "data/external/rosmapPhenos/rosmap_combined_phenotypes.csv",
    )
    parser.add_argument("--refresh-manifest-only", action="store_true")
    return parser.parse_args()


def _cluster_by_projid(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path, usecols=["projid", "clusters"], low_memory=False)
    rows = rows.loc[rows["clusters"].notna()].copy()
    rows["projid"] = rows["projid"].astype(str)
    conflicts = rows.groupby("projid", observed=True)["clusters"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("Cluster source contains conflicting assignments")
    rows = rows.drop_duplicates("projid")
    if len(rows) != 898:
        raise ValueError(f"Expected 898 unique cluster assignments; found {len(rows)}")
    return rows


def patch_sample_metadata(
    data_root: Path,
    phenotype_source: Path,
    cluster_source: Path,
) -> pd.DataFrame:
    """Recover the private-to-public row join without exporting either identifier."""

    path = data_root / "sample_metadata.parquet"
    public = pd.read_parquet(path)
    private = pd.read_parquet(phenotype_source)
    private["projid"] = private["projid"].astype(str)
    private = private.merge(
        _cluster_by_projid(cluster_source),
        on="projid",
        how="left",
        validate="one_to_one",
    ).rename(
        columns={
            "age_death.x": "age_at_death",
            "educ.x": "education_years",
            "cogdx.y": "cogdx",
            "braaksc": "braak_stage",
            "ceradsc": "cerad_score",
            "parkinsonism_yn_lv": "parkinsonism",
        }
    )
    keys = [
        "diagnosis_group",
        "cogn_global",
        "cogng_demog_slope",
        "cogng_path_slope",
        "motor10_demog_slope",
        "sqrt_parksc_demog_slope",
        "age_at_death",
        "education_years",
        "cogdx",
        "braak_stage",
        "cerad_score",
        "parkinsonism",
        "adnc",
    ]
    matched = public.merge(
        private[["clusters", *keys]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    matched["clusters"] = pd.to_numeric(matched["clusters"], errors="coerce").astype("Int64")
    counts = {
        int(key): int(value)
        for key, value in matched["clusters"].value_counts().sort_index().items()
    }
    if counts != {1: 168, 2: 71, 3: 44, 4: 30} or matched["clusters"].isna().sum() != 137:
        raise ValueError(f"Public cluster join failed: counts={counts}")
    forbidden = {"donor", "projid"}.intersection(matched.columns)
    if forbidden:
        raise ValueError(f"Private identifiers reached public metadata: {forbidden}")
    matched.to_parquet(path, index=False, compression="zstd")
    return matched[["sample_id", "diagnosis_group", "clusters"]]


def _statistics_for_file(
    path: Path,
    metadata: pd.DataFrame,
    *,
    module_set: str,
    estimator: str,
    edge_rule: str,
    resolution: str,
) -> pd.DataFrame:
    source = pd.read_parquet(path)
    source = source.merge(metadata, on=["sample_id", "diagnosis_group"], how="left", validate="many_to_one")
    long = aggregate_to_long(source, "raw") if resolution == "aggregate" else resolved_to_long(source, "raw")
    diagnosis = long.copy()
    pooled = long.copy()
    pooled["diagnosis_group"] = "All donors"
    work = pd.concat([diagnosis, pooled], ignore_index=True)
    result = calculate_categorical_associations(
        work,
        [
            "lioness_method",
            "module",
            "metric_family",
            "component",
            "component_label",
            "diagnosis_group",
        ],
    ).rename(columns={"lioness_method": "network_method"})
    result.insert(0, "module_set", module_set)
    result.insert(1, "estimator", estimator)
    result.insert(3, "edge_rule", edge_rule)
    result.insert(4, "resolution", resolution)
    return add_categorical_across_module_fdr(
        result,
        family_columns=[
            "module_set",
            "estimator",
            "network_method",
            "edge_rule",
            "resolution",
            "metric_family",
            "component",
            "diagnosis_group",
        ],
    )


def build_statistics(data_root: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for module_set, directory in (
        ("full_cohort", data_root),
        ("control_derived", data_root / "control_derived"),
    ):
        for resolution in ("aggregate", "resolved"):
            filename = f"{resolution}_plot_data.parquet"
            frames.append(
                _statistics_for_file(
                    directory / filename,
                    metadata,
                    module_set=module_set,
                    estimator="lioness",
                    edge_rule="all",
                    resolution=resolution,
                )
            )
            for edge_rule in ("all", "native_p05", "bh_fdr05"):
                frames.append(
                    _statistics_for_file(
                        directory / "bonobo" / edge_rule / filename,
                        metadata,
                        module_set=module_set,
                        estimator="bonobo",
                        edge_rule=edge_rule,
                        resolution=resolution,
                    )
                )
    result = pd.concat(frames, ignore_index=True)
    result["clusters_are_nominal"] = True
    result["numeric_correlation_used"] = False
    result["effect_definition"] = "max(0,(Kruskal H-k+1)/(n-k))"
    return result


def refresh_manifest(data_root: Path, output: Path, row_count: int) -> None:
    manifest_path = data_root / "data_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outcomes"] = list(dict.fromkeys([*manifest.get("outcomes", []), "clusters"]))
    manifest["nominal_cluster_associations"] = {
        "source": "rosmap_combined_phenotypes.csv: clusters",
        "eligible_donors": 313,
        "missing_donors": 137,
        "class_counts": {"1": 168, "2": 71, "3": 44, "4": 30},
        "test": "Kruskal-Wallis; categories require n>=5",
        "effect": "epsilon_squared=max(0,(H-k+1)/(n-k))",
        "fdr": "BH across modules only within fixed analysis strata",
        "pearson_or_spearman_used": False,
    }
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest.setdefault("files", {})[output.name] = {
        "rows": int(row_count),
        "bytes": int(output.stat().st_size),
        "sha256": digest,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output = data_root / "cluster_association_statistics.parquet"
    if args.refresh_manifest_only:
        row_count = len(pd.read_parquet(output, columns=["module"]))
        refresh_manifest(data_root, output, row_count)
        print(json.dumps({"manifest_refreshed": True, "rows": row_count}, indent=2))
        return
    metadata = patch_sample_metadata(
        data_root,
        args.phenotype_source.resolve(),
        args.cluster_source.resolve(),
    )
    statistics = build_statistics(data_root, metadata)
    statistics.to_parquet(output, index=False, compression="zstd", row_group_size=50_000)
    if output.stat().st_size >= 95 * 1024 * 1024:
        raise RuntimeError("Cluster statistics hot-cache exceeds GitHub's 95-MiB limit")
    refresh_manifest(data_root, output, len(statistics))
    print(json.dumps({
        "rows": int(len(statistics)),
        "bytes": int(output.stat().st_size),
        "modules": sorted(statistics.groupby("module_set")["module"].nunique().to_dict().items()),
    }, indent=2))


if __name__ == "__main__":
    main()
