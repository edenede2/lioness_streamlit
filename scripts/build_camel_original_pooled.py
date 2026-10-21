#!/usr/bin/env python3
"""Restore original pooled associations using exactly the Associations kernels.

Offline, bounded reads of saved scores only. No donor networks, masks, original
diagnosis-specific results, Drive files, or prediction results are changed.
Install small aggregate sidecars in the repository; the existing review shards
on Drive remain byte-for-byte unchanged. Module checkpoints make this resumable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from app_helpers.correlations import add_across_module_fdr
from app_helpers.data import NUMERIC_OUTCOMES, differential_estimator_path
from app_helpers.streaming_associations import _pooled_module_correlations
from app_helpers.validation_review import MATCH_KEYS
from run_camel_validation import RUN, atomic_json, atomic_parquet, sha

SCALES = ("raw", "asinh", "rint")
FAMILIES = ["metric_family", "component", "outcome", "diagnosis_group"]
VERSION = 1


def write_sidecar(frame: pd.DataFrame, destination: Path):
    """One small row group per feature/outcome for narrow Cloud predicate reads."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial")
    ordered = frame.sort_values(["metric_family", "phenotype", "component", "module"])
    schema = pa.Table.from_pandas(ordered, preserve_index=False).schema
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for _, group in ordered.groupby(["metric_family", "phenotype"], sort=False):
            writer.write_table(pa.Table.from_pandas(group, schema=schema, preserve_index=False))
    temporary.replace(destination)


def module_statistics(scores: pd.DataFrame, metadata: pd.DataFrame, outcomes: tuple[str, ...]) -> pd.DataFrame:
    """Use the identical pairwise-complete pooled kernel, including tie handling."""
    if scores.duplicated(["sample_id", "module", "metric_family", "component"]).any():
        raise ValueError("Duplicate original donor-score rows")
    if metadata.sample_id.duplicated().any():
        raise ValueError("Duplicate original donor metadata")
    # Match the original pooled all-donor context, not a selected diagnosis,
    # discovery split, validation split, or an average of group coefficients.
    data = scores.merge(metadata[["sample_id", "diagnosis_group", *outcomes]],
                        on="sample_id", how="left", validate="many_to_one")
    if data.diagnosis_group.isna().any():
        raise ValueError("Original score-to-metadata join is incomplete")
    data = data.loc[data.diagnosis_group.isin(("Control", "MCI", "AD"))].copy()
    result = None
    keys = ["module", "metric_family", "component", "outcome", "diagnosis_group"]
    for scale in SCALES:
        current = data.drop(columns=[f"metric_{other}" for other in SCALES if other != scale])
        current = current.rename(columns={f"metric_{scale}": "metric_value"})
        f = _pooled_module_correlations(current, outcomes)
        renamed = f[keys + ["n", "pearson_r", "pearson_p", "spearman_rho", "spearman_p"]].rename(columns={
            "n": "original_n", "pearson_r": f"original_r_{scale}",
            "pearson_p": f"original_p_{scale}", "spearman_rho": f"original_rho_{scale}",
            "spearman_p": f"original_p_spearman_{scale}"})
        if result is None:
            result = renamed
        else:
            previous_n = result[keys + ["original_n"]].merge(renamed[keys + ["original_n"]],
                on=keys, validate="one_to_one", suffixes=("_old", "_new"))
            if not previous_n.original_n_old.equals(previous_n.original_n_new):
                raise ValueError("Original transformations changed the eligible donor set")
            result = result.merge(renamed.drop(columns="original_n"), on=keys, validate="one_to_one")
    # Original-cohort transformations are monotone, without validation clipping.
    # The deployed review schema uses the shared original Spearman columns.
    for scale in ("asinh", "rint"):
        np.testing.assert_allclose(result[f"original_rho_{scale}"], result.original_rho_raw, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(result[f"original_p_spearman_{scale}"], result.original_p_spearman_raw, atol=1e-12, equal_nan=True)
    result["original_rho"] = result.original_rho_raw
    result["original_p_spearman"] = result.original_p_spearman_raw
    return result.rename(columns={"outcome": "phenotype"})


def correct_families(frame: pd.DataFrame) -> pd.DataFrame:
    """Call the same full-module BH helper as the original Associations page."""
    result = frame.copy()
    for scale in SCALES:
        f = result.rename(columns={"phenotype": "outcome"}).copy()
        f["pearson_p"] = f[f"original_p_{scale}"]
        f["spearman_p"] = f.original_p_spearman
        corrected = add_across_module_fdr(f, family_columns=FAMILIES)
        result[f"original_q_{scale}"] = corrected.pearson_fdr_across_modules.to_numpy()
        result[f"original_family_n_{scale}"] = corrected.pearson_fdr_module_family_n.to_numpy()
        if scale == "raw":
            result["original_q_spearman"] = corrected.spearman_fdr_across_modules.to_numpy()
            result["original_family_n_spearman"] = corrected.spearman_fdr_module_family_n.to_numpy()
    # Explicit scale-specific fields are not needed in the deployed schema,
    # after the exact monotonic-invariance checks above.
    return result.drop(columns=[f"original_{prefix}_{scale}" for prefix in ("rho", "p_spearman") for scale in SCALES])


def build(root: Path, install=False, selected_rules=None):
    review = root / "public_bundle/review"
    registry_path = review / "manifest.json"
    registry = json.loads(registry_path.read_text())
    public_registry = json.loads((APP / "data/control_derived/camel_validation/review/manifest.json").read_text())
    eligible = set(public_registry["completed_rules"]) & set(registry["completed_rules"])
    rules = [rule for rule in registry["completed_rules"] if rule in eligible]
    if selected_rules:
        if not set(selected_rules) <= set(rules):
            raise ValueError("Only completed, already released comparison masks may be restored")
        rules = list(selected_rules)
    modules = sorted(json.loads((root / "public_bundle/manifest.json").read_text())["completed_module_ids"])
    if len(modules) != registry["modules"]:
        raise ValueError("Original module family is incomplete")
    meta_path = APP / "data/sample_metadata.parquet"
    metadata = pd.read_parquet(meta_path)
    if len(metadata) != 450 or metadata.sample_id.duplicated().any():
        raise ValueError("Original pooled cohort must contain the unchanged 450 donors")
    outcomes = tuple(o for o in NUMERIC_OUTCOMES if o in metadata)
    columns = ["sample_id", "module", "metric_family", "component", "component_label", *[f"metric_{s}" for s in SCALES]]
    entries = []
    for rule in rules:
        for norm in ("standard_pruned", "retained_edge"):
            source = differential_estimator_path("resolved_plot_data.parquet", "control_derived", "lioness",
                "control_anchored", "all", rule, "global", .05, norm)
            files = [source] if source.is_file() else sorted(source.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"Original scores must be present locally: {source}")
            sources = {str(p.relative_to(APP)): sha(p) for p in [meta_path, *files]}
            fingerprint = hashlib.sha256(json.dumps(dict(version=VERSION, sources=sources, modules=modules,
                outcomes=outcomes, scale_policy="saved original score columns", diagnoses=["Control", "MCI", "AD"]), sort_keys=True).encode()).hexdigest()
            part_root = root / "original_pooled_checkpoints" / rule / norm
            parts = []
            needed = []
            for module in modules:
                part = part_root / f"M{module}.parquet"
                stamp = part.with_suffix(".json")
                if part.is_file() and stamp.is_file() and json.loads(stamp.read_text()).get("fingerprint") == fingerprint:
                    parts.append(part)
                else:
                    needed.append(module)
            for offset in range(0, len(needed), 24):
                # A bounded source chunk avoids repeatedly scanning every row
                # group separately for 186 individual module predicates.
                chunk = needed[offset:offset + 24]
                data = pd.read_parquet(source, columns=columns,
                    filters=[("lioness_method", "=", "control_anchored"), ("module", "in", chunk)])
                if set(data.module.unique()) != set(chunk):
                    raise ValueError("An original module is absent from saved scores")
                for module, scores in data.groupby("module", sort=False):
                    stats = module_statistics(scores, metadata, outcomes)
                    part = part_root / f"M{module}.parquet"
                    atomic_parquet(stats, part)
                    atomic_json(dict(fingerprint=fingerprint, rows=len(stats)), part.with_suffix(".json"))
                    parts.append(part)
                print(json.dumps(dict(rule=rule, normalization=norm, checkpoint_modules=len(parts), expected_modules=len(modules))), flush=True)
            frame = correct_families(pd.concat([pd.read_parquet(p) for p in sorted(parts)], ignore_index=True))
            if set(frame.module.unique()) != set(modules) or frame.duplicated(MATCH_KEYS).any():
                raise ValueError("Pooled module family identity/grain failed")
            if {"sample_id", "donor", "projid"} & set(frame):
                raise ValueError("Original pooled sidecars must contain aggregate statistics only")
            relative = Path("original_pooled") / rule / f"{norm}.parquet"
            destination = review / relative
            write_sidecar(frame, destination)
            if destination.stat().st_size >= 95 * 1024**2:
                raise ValueError("Original pooled summary exceeds the public size gate")
            entries.append(dict(rule=rule, normalization=norm, path=str(relative), rows=len(frame),
                sha256=sha(destination), bytes=destination.stat().st_size, source_hashes=sources,
                source_fingerprint=fingerprint, original_donors=450, module_count=len(modules),
                diagnoses=["Control", "MCI", "AD"], analysis_subset="all_donors",
                fdr_policy="Original Associations add_across_module_fdr; separate Pearson/Spearman BH across valid modules per fixed feature, component, outcome, scale and pooled cohort"))
    def update_registry(path):
        current = json.loads(path.read_text())
        replaced = {(e["rule"], e["normalization"]) for e in entries}
        current["original_pooled_shards"] = [e for e in current.get("original_pooled_shards", [])
            if (e["rule"], e["normalization"]) not in replaced] + entries
        current["generated_utc"] = datetime.now(timezone.utc).isoformat()
        current["limitations"] = [note for note in current.get("limitations", [])
            if note != "Original pooled/unclassified association comparisons are unavailable in the saved stratified catalog."]
        note = "Original Unclassified comparisons are unavailable; pooled original comparisons are restored from saved scores with the original module-only FDR rules."
        if note not in current["limitations"]:
            current["limitations"].append(note)
        return current
    atomic_json(update_registry(registry_path), registry_path)
    if install:
        target = APP / "data/control_derived/camel_validation/review"
        rollback = root / "rollback" / str(time.time_ns()) / "original_pooled"
        rollback.mkdir(parents=True, exist_ok=True)
        # Keep the common connectivity/global-cognition comparisons beside the
        # baselines, so M1116 and other modules in these contexts work offline
        # and do not require an additional Drive request after deployment.
        contexts = {(e["rule"], e["normalization"]) for e in entries}
        defaults = [e for e in registry["association_shards"]
                    if (e["rule"], e["normalization"]) in contexts
                    and e["feature"] == "connectivity" and e["outcome"] == "cogn_global"]
        for entry in [*entries, *defaults]:
            path = target / entry["path"]
            if path.is_file():
                saved = rollback / entry["path"]
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, saved)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".partial")
            shutil.copy2(review / entry["path"], temporary)
            if sha(temporary) != entry["sha256"]:
                raise ValueError("Original pooled install hash mismatch")
            temporary.replace(path)
        shutil.copy2(target / "manifest.json", rollback / "manifest.json")
        atomic_json(update_registry(target / "manifest.json"), target / "manifest.json")
    print(json.dumps(dict(original_pooled_shards=len(entries), rows=sum(e["rows"] for e in entries),
                         bytes=sum(e["bytes"] for e in entries), installed=install)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUN)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--rules", nargs="+")
    args = parser.parse_args()
    build(args.root, args.install, args.rules)
