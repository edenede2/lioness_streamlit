"""Read completed LOCAL validation bundles; never redirects a hosted release."""

from pathlib import Path
from functools import lru_cache
import json
import pandas as pd
import pyarrow.parquet as pq
from app_helpers import drive_data as D
from .registry import APP, RESEARCH_ROOT
from .inputs import sha

BASE = APP / "data/control_calibrated_validation_v1"


def local_root():
    if D.DATA_DIR != D.LOCAL_DATA_DIR.resolve():
        return None
    path = BASE / "current_local.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    identity = value.get("analysis_id", "")
    if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("Invalid local validation identity")
    root = BASE / identity
    complete = root / "complete.json"
    if not complete.exists():
        return None
    record = json.loads(complete.read_text())
    if record.get("status") != "complete" or record.get("analysis_id") != identity:
        return None
    return root


@lru_cache(maxsize=64)
def _verified(path, mtime, size, digest):
    if sha(path) != digest:
        raise ValueError("Immutable validation result checksum mismatch")


def verified(root, relative):
    root = Path(root)
    path = root / relative
    manifest = json.loads((root / "complete.json").read_text())
    expected = manifest["files"][relative]
    s = path.stat()
    _verified(str(path), s.st_mtime_ns, s.st_size, expected)
    return path


def manifest():
    root = local_root()
    return json.loads((root / "complete.json").read_text()) if root else None


def read_json(relative):
    root = local_root()
    if root is None:
        raise FileNotFoundError("Local validation analysis is not available")
    return json.loads(verified(root, relative).read_text())


def read_table(relative, filters=None):
    root = local_root()
    if root is None:
        raise FileNotFoundError("Local validation analysis is not available")
    frame = pd.read_parquet(verified(root, relative), filters=filters)
    if "bootstrap_draws" in frame:
        frame["bootstrap_budget"] = frame.bootstrap_draws
        frame["bootstrap_attempted"] = (
            frame.bootstrap_draws.where(frame.n.ge(3), 0)
            if "bootstrap_valid" in frame
            else 0
        )
        if "bootstrap_valid" in frame:
            frame["bootstrap_degenerate"] = (
                frame.bootstrap_attempted - frame.bootstrap_valid
            )
    return frame


def baseline_path(score_root, dataset, scope, version="auto"):
    if version not in ("auto", "repaired", "original"):
        raise ValueError("Unknown baseline inference version")
    root = (
        local_root() if Path(score_root).resolve() == RESEARCH_ROOT.resolve() else None
    )
    if version == "repaired" and root is None:
        raise FileNotFoundError(
            "Repaired inference is available only in the completed local validation bundle"
        )
    if version != "original" and root is not None:
        return verified(root, f"repaired_baseline/{dataset}_{scope}.parquet")
    return None


def results(
    family,
    beta,
    cohorts=("original", "incomplete"),
    models=None,
    strata=None,
    outcome=None,
    modules=None,
):
    tables = []
    record = read_json("analysis_spec.json")
    dataset = record["families"][family]["dataset"]
    filters = [("family", "=", family), ("beta", "=", int(beta))]
    if models:
        filters.append(("model", "in", list(models)))
    if strata:
        filters.append(("study_stratum", "in", list(strata)))
    if outcome:
        filters.append(("outcome", "=", outcome))
    for cohort in cohorts:
        tables.append(read_table(f"results/{dataset}_{cohort}.parquet", filters))
    if not tables:
        return pd.DataFrame()
    result = pd.concat(tables, ignore_index=True)
    result["correction_scope"] = result.model.map(
        lambda m: (
            "within-focused age-spline model/cohort; 4 named modules, 2 components, 8 endpoints, 4 metrics, both betas"
            if m == "C_age_spline"
            else "within catalog model/cohort/study stratum; all canonical registered hypotheses, 8 endpoints, 4 metrics, both betas"
        )
    )
    # Add the requested module-only family without altering sealed inference.
    # Read all modules before selecting named examples; focused-only panels do
    # not establish a complete-partition BH denominator.
    from .association_fdr import add_module_fdr
    full = result.loc[~result.model.eq("C_age_spline")]
    focused = result.loc[result.model.eq("C_age_spline")].copy()
    if not focused.empty:
        focused["q_across_modules"] = float("nan")
        focused["module_family_n"] = 0
        focused["module_fdr_scope"] = "unavailable: focused panel, not the complete module partition"
    result = pd.concat([add_module_fdr(full), focused], ignore_index=True)
    if modules:
        result = result.loc[result.module.isin(list(map(int, modules)))].copy()
    return result
