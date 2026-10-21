"""Small, public-metadata-only donor catalog with explicit coverage and provenance.

Brain and motor sample IDs belong to different public namespaces. They must
never be linked, concatenated as unique people, or reconciled via phenotypes.
Regional snapshots are also not silently filled with original-450 values.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from app_helpers.data import ASSOCIATION_OUTCOME_LABELS
from app_helpers.drive_data import DATA_DIR, data_path_available, ensure_data_path, load_drive_index
from app_helpers.study import with_study_metadata

BRAIN_TISSUES = ("AC", "DLPFC", "PCG")
MOTOR_TISSUES = ("VH", "SMA", "QUAD")
DATASET_LABELS = {"brain": "Brain regions: AC / DLPFC / PCG", "motor": "Motor dataset: VH / SMA / QUAD"}
MISSING = "Missing / not recorded"
ALLOWED_FIELDS = {"sample_id", *ASSOCIATION_OUTCOME_LABELS, "parkinsonism_label", "ad_control_split"}
PRIVATE_FIELDS = {"donor", "projid", "donor_private", "projid_private"}
CATEGORICAL_FIELDS = ("diagnosis_group", "study", "clusters", "cogdx", "braak_stage", "cerad_score",
                      "adnc", "parkinsonism", "sex_code", "apoe_genotype", "ad_control_split",
                      "available_tissues", "in_original_analysis")


def source_paths(dataset):
    if dataset == "brain":
        return [*(DATA_DIR / f"regional_modules/metadata/maximum_tissue/{t}.parquet" for t in BRAIN_TISSUES),
                DATA_DIR / "sample_metadata.parquet"]
    if dataset == "motor":
        return [DATA_DIR / f"{m}/sample_metadata.parquet" for m in ("muscle_control_l4", "muscle_all_donors_l4")]
    raise ValueError("Unknown donor metadata namespace")


def source_revision(dataset):
    entries = load_drive_index().get("files", {})
    revision = []
    for path in source_paths(dataset):
        if path.is_file():
            s = path.stat()
            revision.append((str(path), s.st_mtime_ns, s.st_size))
        else:
            key = str(path.relative_to(DATA_DIR))
            revision.append((key, entries.get(key, {}).get("md5", "unavailable")))
    study = Path(__file__).resolve().parents[1] / "data/study/sample_study.parquet"
    if study.is_file():
        revision.append(("study_overlay", study.stat().st_mtime_ns, study.stat().st_size))
    return tuple(revision)


def _read(path, namespace):
    if not data_path_available(path):
        raise FileNotFoundError("The public donor metadata for this source are not packaged yet")
    result = pd.read_parquet(ensure_data_path(path))
    if PRIVATE_FIELDS.intersection(result) or "sample_id" not in result:
        raise ValueError("Donor metadata failed its public-identifier schema check")
    result = result[[c for c in result if c in ALLOWED_FIELDS]].copy()
    if result.sample_id.isna().any() or result.sample_id.astype(str).duplicated().any():
        raise ValueError("Donor metadata do not contain one unique public sample ID per row")
    result["sample_id"] = result.sample_id.astype(str)
    if result.sample_id.str.strip().eq("").any():
        raise ValueError("Blank public sample identifier")
    return with_study_metadata(result, namespace)


def reconcile_regional_frames(frames):
    """Merge only matching regional snapshots, rejecting conflicting values."""
    combined = pd.concat(frames.values(), ignore_index=True)
    grouped = combined.groupby("sample_id", observed=True)
    conflicts = grouped.nunique(dropna=True).gt(1)
    for field in conflicts.columns[conflicts.any()]:
        if pd.api.types.is_numeric_dtype(combined[field]):
            bounds = grouped[field].agg(["min", "max"])
            conflicts[field] &= ~np.isclose(bounds["min"], bounds["max"], rtol=1e-10, atol=1e-12, equal_nan=True)
    if conflicts.any().any():
        fields = conflicts.columns[conflicts.any()].tolist()
        raise ValueError("Regional metadata conflict in fields: " + ", ".join(fields))
    result = combined.groupby("sample_id", as_index=False, observed=True).first()
    for tissue, frame in frames.items():
        result["has_" + tissue] = result.sample_id.isin(frame.sample_id)
    return result


def _decorate(frame, namespace, tissues, version):
    result = frame.copy()
    result["metadata_namespace"] = namespace
    result["metadata_snapshot"] = version
    result["available_tissues"] = result.apply(lambda r: " + ".join(t for t in tissues if r["has_" + t]), axis=1)
    result["n_available_tissues"] = result[["has_" + t for t in tissues]].sum(axis=1).astype(int)
    return result


def load_catalog(dataset):
    """Read at most four tiny brain metadata files or two motor metadata files."""
    paths = source_paths(dataset)
    if dataset == "brain":
        frames = {t: _read(p, "brain") for t, p in zip(BRAIN_TISSUES, paths[:3])}
        catalog = reconcile_regional_frames(frames)
        original = _read(paths[-1], "brain")
        if not set(original.sample_id).issubset(set(catalog.sample_id)):
            raise ValueError("Original analysis donors are absent from regional coverage metadata")
        catalog["in_original_analysis"] = catalog.sample_id.isin(original.sample_id)
        coverage = catalog[["sample_id", *("has_" + t for t in BRAIN_TISSUES)]]
        original = original.merge(coverage, on="sample_id", how="left", validate="one_to_one")
        original["in_original_analysis"] = True
        catalog = _decorate(catalog, "brain", BRAIN_TISSUES, "Regional / validation metadata")
        original = _decorate(original, "brain", BRAIN_TISSUES, "Original 450-donor analysis metadata")
        if not original.n_available_tissues.eq(3).all():
            raise ValueError("Original common cohort does not have all three regions")
        return catalog, original
    # Both motor partitions use the same public common cohort. Never double
    # count it, and never merge motor IDs with identically spelled brain IDs.
    a, b = (_read(p, "motor") for p in paths)
    if set(a.sample_id) != set(b.sample_id):
        raise ValueError("Motor partition metadata have different cohorts; explicit routing is required")
    reconcile_regional_frames({"control_partition": a, "all_donor_partition": b})
    for tissue in MOTOR_TISSUES:
        a["has_" + tissue] = True
    a["in_original_analysis"] = True
    catalog = _decorate(a, "motor", MOTOR_TISSUES, "Common motor cohort metadata (both partitions)")
    return catalog, pd.DataFrame()


def coverage_mask(frame, tissues, required, mode="at_least"):
    required = tuple(required)
    if set(required) - set(tissues) or mode not in {"at_least", "exactly", "any"}:
        raise ValueError("Invalid tissue coverage selection")
    if not required:
        return pd.Series(False, index=frame.index)
    chosen = frame[["has_" + t for t in required]].astype(bool)
    keep = chosen.any(axis=1) if mode == "any" else chosen.all(axis=1)
    if mode == "exactly":
        other = ["has_" + t for t in tissues if t not in required]
        if other:
            keep &= ~frame[other].any(axis=1)
    return keep


def cohort_options(dataset, frame, original):
    if dataset == "motor":
        return {"motor_common": f"Common VH / SMA / QUAD cohort ({len(frame):,})"}
    options = {
        "all": f"Any brain region ({len(frame):,})",
        "complete": f"All three brain regions ({int(frame.n_available_tissues.eq(3).sum()):,})",
        "original": f"Original analyzed cohort — original metadata ({len(original):,})",
        "incomplete": f"Incomplete-region validation donors ({int(frame.n_available_tissues.lt(3).sum()):,})",
    }
    for n in (1, 2):
        for selected in combinations(BRAIN_TISSUES, n):
            name = "–".join(selected)
            for mode, text in (("at_least", "available (may include other regions)"), ("exactly", "only")):
                count = int(coverage_mask(frame, BRAIN_TISSUES, selected, mode).sum())
                options[mode + ":" + "+".join(selected)] = f"{name} {text} ({count:,})"
    options["custom"] = "Custom region coverage"
    return options


def select_cohort(dataset, frame, original, preset, required=(), mode="at_least"):
    if preset not in cohort_options(dataset, frame, original):
        raise ValueError("Cohort is not registered for this dataset")
    if dataset == "motor" or preset == "all":
        return frame.copy()
    if preset == "original":
        return original.copy()
    if preset in {"complete", "incomplete"}:
        keep = frame.n_available_tissues.eq(3) if preset == "complete" else frame.n_available_tissues.lt(3)
    else:
        if preset != "custom":
            mode, text = preset.split(":")
            required = text.split("+")
        keep = coverage_mask(frame, BRAIN_TISSUES, required, mode)
    return frame.loc[keep].copy()


def category_values(frame, field):
    def label(value):
        if pd.isna(value):
            return MISSING
        if field == "clusters":
            return "Cluster " + str(int(float(value)))
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        return str(value)
    return frame[field].map(label)


def filter_donors(frame, categories=None, numeric_ranges=None, require_observed=()):
    keep = pd.Series(True, index=frame.index)
    for field, levels in (categories or {}).items():
        keep &= category_values(frame, field).isin(levels)
    for field, (low, high, include_missing) in (numeric_ranges or {}).items():
        values = pd.to_numeric(frame[field], errors="coerce")
        keep &= values.between(low, high) | (values.isna() if include_missing else False)
    for field in require_observed:
        values = frame[field]
        if pd.api.types.is_numeric_dtype(values):
            keep &= np.isfinite(values)
        else:
            keep &= values.notna()
    return frame.loc[keep].copy()


def coverage_summary(frame, tissues):
    rows = []
    for n in range(1, len(tissues) + 1):
        for selected in combinations(tissues, n):
            block = frame.loc[coverage_mask(frame, tissues, selected)]
            counts = category_values(block, "diagnosis_group").value_counts()
            row = {"Component": "–".join(selected), "Total": len(block)}
            row.update({g: int(counts.get(g, 0)) for g in ("Control", "MCI", "AD", "Unclassified", MISSING)})
            rows.append(row)
    return pd.DataFrame(rows)


def missingness(frame, fields):
    return pd.DataFrame([dict(field=c, label=ASSOCIATION_OUTCOME_LABELS.get(c, c), donors=len(frame),
                              observed=int(frame[c].notna().sum()), missing=int(frame[c].isna().sum()),
                              missing_percent=float(frame[c].isna().mean() * 100) if len(frame) else np.nan)
                         for c in fields])


def snapshot_differences(original, regional):
    """Describe differences without choosing, filling, or replacing a value."""
    fields = sorted((set(original) & set(regional) & ALLOWED_FIELDS) - {"sample_id"})
    merged = original.merge(regional, on="sample_id", suffixes=("_original", "_regional"), validate="one_to_one")
    rows = []
    for field in fields:
        a, b = merged[field + "_original"], merged[field + "_regional"]
        observed = a.notna() & b.notna()
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            different = observed & ~np.isclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True)
        else:
            different = observed & a.astype(str).ne(b.astype(str))
        rows.append(dict(field=field, label=ASSOCIATION_OUTCOME_LABELS.get(field, field),
                         matched_donors=len(merged), observed_in_both=int(observed.sum()),
                         differing_values=int(different.sum()),
                         missing_only_original=int((a.isna() & b.notna()).sum()),
                         missing_only_regional=int((b.isna() & a.notna()).sum())))
    return pd.DataFrame(rows)


def same_metadata_value(a, b):
    if pd.isna(a) or pd.isna(b):
        return bool(pd.isna(a) and pd.isna(b))
    if isinstance(a, (float, int, np.number)) and isinstance(b, (float, int, np.number)):
        return bool(np.isclose(a, b, rtol=1e-10, atol=1e-12))
    return str(a) == str(b)
