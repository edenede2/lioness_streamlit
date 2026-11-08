"""Read-only meeting evidence and pure donor-resampling diagnostics.

UI readers never reconstruct an edge or import the offline input layer. Public
tables contain aggregates only; detailed edge/matrix artifacts remain local.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

APP = Path(__file__).resolve().parents[1]
VERSION = "20261108_focused_validation_v1"
PUBLIC = APP / "docs/reports" / VERSION
PRIVATE = APP / "artifacts" / VERSION / "private"
REPORT = APP / "docs/reports/control_calibrated_validation_20261104"
FOCUS = {"control_derived": (747, 1064, 1116), "full_cohort": (1643,)}
COMPONENTS = ("TS_DLPFC", "CT_DLPFC__PCG")
COMPONENT_LABELS = {"TS_DLPFC": "DLPFC", "CT_DLPFC__PCG": "DLPFC–PCG"}
REPORT_TABLES = {
    "focused_all_endpoints",
    "focused_selection_adjustment",
    "interaction",
    "incremental",
    "comparators",
    "composition",
    "study_applicability",
    "effect_summary",
}
PUBLIC_TABLES = {
    "cohorts",
    "intersections",
    "mdc_overview",
    "structure",
    "mdc_decomposition",
    "edge_summary",
    "availability",
    "historical_mdc",
    "association_overview",
}
PRIVATE_TABLES = {"same_edges", "edge_candidates", "edge_contributions"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def manifest(root=PUBLIC):
    path = Path(root) / "manifest.json"
    if not path.is_file():
        return {
            "version": VERSION,
            "status": "unavailable",
            "reason": "Focused evidence bundle not installed",
        }
    value = json.loads(path.read_text())
    if value.get("version") != VERSION:
        raise ValueError("Evidence artifact version mismatch; no fallback is permitted")
    return value


@lru_cache(maxsize=48)
def _read(path, size, mtime, expected):
    if sha(path) != expected:
        raise ValueError("Evidence checksum mismatch")
    frame = pd.read_csv(path)
    if {"sample_id", "donor", "projid"}.intersection(frame):
        raise ValueError("Aggregate evidence cannot contain donor identifiers")
    return frame


def table(name, root=PUBLIC):
    if name not in PUBLIC_TABLES:
        raise ValueError("Unknown aggregate evidence table")
    m = manifest(root)
    entry = m.get("files", {}).get(name + ".csv")
    if not entry:
        return pd.DataFrame()
    path = Path(root) / (name + ".csv")
    stat = path.stat()
    return _read(str(path), stat.st_size, stat.st_mtime_ns, entry["sha256"]).copy()


def public_artifact(name, root=PUBLIC):
    """Only hash-verified meeting exports; never resolve a private/local path."""
    root = Path(root)
    name = str(name)
    allowed = name in {"RESULTS.md", "QA.json"} or (
        Path(name).parent == Path("figures") and Path(name).suffix in {".png", ".pdf"}
    )
    entry = manifest(root).get("artifacts", {}).get(name)
    path = root / name
    if (
        not allowed
        or not entry
        or not path.is_file()
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError("Unknown public meeting artifact")
    if sha(path) != entry["sha256"]:
        raise ValueError("Meeting artifact checksum mismatch")
    return path.read_bytes()


def report_table(name, root=REPORT):
    """Exact already-public report, including its original correction columns."""
    if name not in REPORT_TABLES:
        raise ValueError("Unknown adjusted-report table")
    root = Path(root)
    path = root / "report_manifest.json"
    if not path.is_file():
        return pd.DataFrame()
    m = json.loads(path.read_text())
    expected = m.get("files", {}).get(name + ".csv")
    if not m.get("aggregate_only") or not expected:
        raise ValueError("Unverified aggregate report table")
    source = root / (name + ".csv")
    stat = source.stat()
    frame = _read(str(source), stat.st_size, stat.st_mtime_ns, expected).copy()
    if "analysis_id" in frame and not frame.analysis_id.eq(m["analysis_id"]).all():
        raise ValueError("Report analysis identity mismatch")
    return frame


def private_table(name, root=None):
    if name not in PRIVATE_TABLES:
        raise ValueError("Unknown local-only edge table")
    root = PRIVATE if root is None else Path(root)
    path = Path(root) / (name + ".parquet")
    receipt = Path(root) / "manifest.json"
    if not path.is_file() or not receipt.is_file():
        return pd.DataFrame()
    m = json.loads(receipt.read_text())
    if m.get("version") != VERSION or not m.get("local_only"):
        raise ValueError("Local edge evidence identity mismatch")
    expected = m.get("files", {}).get(path.name)
    if not expected or sha(path) != expected:
        raise ValueError("Local edge evidence checksum mismatch")
    return pd.read_parquet(path)


def canonical_component(component):
    return str(component).replace("PCGBA23", "PCG")


def intersection_audit(left, right):
    if left is None or right is None:
        return dict(
            status="independence not verified",
            overlap_n=np.nan,
            original_n=np.nan if left is None else len(set(left)),
            validation_n=np.nan if right is None else len(set(right)),
        )
    left, right = set(left), set(right)
    return dict(
        status="verified disjoint" if not left & right else "overlapping inputs",
        overlap_n=len(left & right),
        original_n=len(left),
        validation_n=len(right),
    )


def cohort_selection(meta, component, cohort, *, independent=False):
    component = canonical_component(component)
    tissues = component[3:].split("__")
    if any("has_" + t not in meta for t in tissues):
        raise ValueError("Required tissue-presence inclusion record missing")
    present = meta[["has_" + t for t in tissues]].all(axis=1)
    complete_columns = ["has_" + t for t in ("AC", "DLPFC", "PCG")]
    if cohort == "original_complete":
        keep = meta.in_original_analysis.astype(bool)
    elif cohort == "incomplete":
        keep = ~meta[complete_columns].all(axis=1)
    elif cohort == "legacy_heldout":
        keep = meta.in_original_analysis & meta.ad_control_split.eq("Validation")
    else:
        raise ValueError("Unknown cohort; no original/validation substitution")
    if independent:
        keep &= meta.role.eq("target")
    return (keep & present).to_numpy()


def concordance(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def group_edges(values, rows, columns, power):
    from app_helpers.camel_mdc import standardized, adjacency_edges

    if len(values) < 2:
        return np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    z = standardized(values)
    raw = np.clip((z.T @ z)[rows, columns], -1, 1)
    return raw, adjacency_edges(z, rows, columns, power)


def hub_concordance(first, second, rows, columns, node_tissues):
    n = len(node_tissues)
    one = np.bincount(rows, weights=abs(first), minlength=n) + np.bincount(
        columns, weights=abs(first), minlength=n
    )
    two = np.bincount(rows, weights=abs(second), minlength=n) + np.bincount(
        columns, weights=abs(second), minlength=n
    )
    return {
        "hub_"
        + t: concordance(
            one[np.asarray(node_tissues) == t], two[np.asarray(node_tissues) == t]
        )
        for t in dict.fromkeys(node_tissues)
    }


def density_ratio(ad, control):
    """No epsilon or denominator flooring; undefined is not zero."""
    numerator, denominator = float(np.mean(abs(ad))), float(np.mean(abs(control)))
    return (
        numerator,
        denominator,
        (
            numerator / denominator
            if denominator > 0 and np.isfinite(denominator)
            else np.nan
        ),
    )


def structure_metrics(original, validation, rows, columns, tissues, power):
    r0, a0 = group_edges(original, rows, columns, power)
    r1, a1 = group_edges(validation, rows, columns, power)
    return dict(
        mean_abs_original_control=float(np.mean(abs(a0))),
        mean_abs_validation_control=float(np.mean(abs(a1))),
        signed_r_concordance=concordance(r0, r1),
        signed_adjacency_concordance=concordance(a0, a1),
        magnitude_concordance=concordance(abs(a0), abs(a1)),
        **hub_concordance(a0, a1, rows, columns, tissues),
    )


def resample_indices(meta, rng):
    labels = meta.study.fillna("Missing study").astype(str).to_numpy()
    return np.concatenate(
        [
            rng.choice(ix, len(ix), replace=True)
            for label in sorted(set(labels))
            for ix in [np.flatnonzero(labels == label)]
        ]
    )


def interval(values, budget):
    values = np.asarray(values, float)
    valid = values[np.isfinite(values)]
    bounds = (
        np.quantile(valid, [0.025, 0.975])
        if len(valid) >= 0.8 * budget
        else [np.nan, np.nan]
    )
    return float(bounds[0]), float(bounds[1]), len(valid)


def evidence_status(original, validation, p=np.nan, q=np.nan, alpha=0.05):
    if not np.isfinite(original) or not np.isfinite(validation):
        return "Unavailable / not tested"
    if np.sign(original) != np.sign(validation) and original != 0 and validation != 0:
        return "Direction reversal"
    if original == 0 or validation == 0:
        return "Zero direction / descriptive"
    if np.isfinite(q) and q < alpha:
        return "Same direction + adjusted support"
    if np.isfinite(p) and p < alpha:
        return "Same direction + nominal support"
    return "Same direction only"


def mdc_summary(frame, alpha=0.05):
    """Preserve full-family q and every planned row, including empty masks."""
    f = frame.copy()
    original = pd.to_numeric(f.original_mdc, errors="coerce")
    validation = pd.to_numeric(f.validation_mdc, errors="coerce")
    valid = (
        original.gt(0)
        & validation.gt(0)
        & np.isfinite(original)
        & np.isfinite(validation)
    )
    original_signal = pd.to_numeric(f.original_directional_fdr, errors="coerce").lt(
        alpha
    )
    log_original, log_validation = np.log2(original.where(original.gt(0))), np.log2(
        validation.where(validation.gt(0))
    )
    same = valid & (log_original * log_validation).gt(0)
    reversed_ = valid & (log_original * log_validation).lt(0)
    p = np.where(
        validation.ge(1),
        np.maximum(f.p_loss_sample, f.p_loss_gene),
        np.maximum(f.p_gain_sample, f.p_gain_gene),
    )
    tested = f.inferential.fillna(False).astype(bool) & np.isfinite(
        f.validation_directional_fdr
    )
    return dict(
        planned=len(f),
        comparable=int(valid.sum()),
        unavailable=int((~valid).sum()),
        original_significant=int(original_signal.sum()),
        original_significant_evaluable=int((original_signal & valid).sum()),
        same_direction=int(same.sum()),
        direction_reversed=int(reversed_.sum()),
        zero_direction=int((valid & ~same & ~reversed_).sum()),
        original_signal_same_direction=int((original_signal & same).sum()),
        original_signal_nominal_support=int(
            (original_signal & same & tested & (p < alpha)).sum()
        ),
        original_signal_adjusted_support=int(
            (
                original_signal & same & tested & f.validation_directional_fdr.lt(alpha)
            ).sum()
        ),
        validation_tested=int(tested.sum()),
        descriptive_only=int((~f.inferential.fillna(False).astype(bool)).sum()),
    )
