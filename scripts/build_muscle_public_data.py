#!/usr/bin/env python3
"""Build the public non-prediction bundle for VH/SMA/QUAD Control L4 modules.

The source modules are biological L4 (zero-indexed ``table_3``) discovered in
Controls with Method 5 extended SFN.  This builder packages the separate
signedAlt LIONESS analysis, tissue-expanded/regional KEGG, composition and
entropy, and (when available) the full-M5 observed MDC tables.  Donor and
``projid`` identifiers never enter the public directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, rankdata, spearmanr, t as student_t
from sklearn.utils.extmath import randomized_svd

_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

from app_helpers.correlations import (
    add_categorical_across_module_fdr,
    calculate_categorical_associations,
)


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[1]
RUN_ROOT = (
    REPO_ROOT
    / "out/lioness_reference_comparison_rosmap/"
    "20261012_muscle_control_l4_nonprediction"
)
ASSIGNMENTS = (
    REPO_ROOT
    / "results/se2_all_genes/20260319_090116/M5_all_genes/"
    "M5_all_genes_table_3.csv"
)
MUSCLE_METADATA = (
    REPO_ROOT / "data/external/muscle_samples_with_clinical_groups_by_IID_cogdx_RADC.csv"
)
MUSCLE_EXPRESSION_ROOT = REPO_ROOT / "data/raw/ROSMAP_muscle_split"
KEGG_ROOT = REPO_ROOT / "out/kegg_enrichment_muscle_control_l4_table3"
MDC_ROOT = REPO_ROOT / "out/mdc_muscle_control_l4/20261012_m5_ad_vs_control"
MDC_SIGNIFICANCE_ROOT = MDC_ROOT / "significance_200perms"
OUTPUT = APP_ROOT / "data/muscle_control_l4"
PRIVATE_MAP = REPO_ROOT / "out/muscle_control_l4/private_public_sample_mapping.tsv"
# This partition was learned in Controls. Only Control-referenced LIONESS is a
# supported analysis or public method for this module definition.
METHODS = ("control_anchored",)
MODULE_SET = "muscle_control_l4"
MODULE_COUNT = 155
ASSIGNMENT_COUNT = 33_491
MODULE_LABEL = "Control-derived VH/SMA/QUAD M5 L4 modules (155)"
PARTITION_COHORT = "Control only"
BETAS_TS = {"VH": 5, "SMA": 7, "QUAD": 13}
BETAS_CT = {"VH-SMA": 8, "VH-QUAD": 10, "SMA-QUAD": 10}
FEATURES = (
    "connectivity",
    "positive_density",
    "negative_density",
    "abs_sum",
    "positive_abs_sum",
    "negative_abs_sum",
)
PHENOTYPES = (
    "cogn_global",
    "cogng_demog_slope",
    "cogng_path_slope",
    "motor10_demog_slope",
    "sqrt_parksc_demog_slope",
)
OUTCOMES = (
    *PHENOTYPES,
    "age_at_death",
    "education_years",
    "cogdx",
    "braak_stage",
    "cerad_score",
    "adnc",
    "parkinsonism",
)
COMPONENTS = (
    "CT_VH__SMA",
    "CT_VH__QUAD",
    "CT_SMA__QUAD",
    "TS_VH",
    "TS_SMA",
    "TS_QUAD",
)
COMPONENT_LABELS = {
    "CT_VH__SMA": "CT: VH - SMA",
    "CT_VH__QUAD": "CT: VH - QUAD",
    "CT_SMA__QUAD": "CT: SMA - QUAD",
    "TS_VH": "TS: VH",
    "TS_SMA": "TS: SMA",
    "TS_QUAD": "TS: QUAD",
}
PRIORITY = re.compile(
    r"lipid|fatty acid|cholesterol|glycerolipid|glycerophospholipid|"
    r"sphingolipid|metaboli|alzheimer|infection|infectious|viral|virus",
    re.I,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    frame.to_parquet(
        temporary, index=False, compression="zstd", row_group_size=25_000
    )
    os.replace(temporary, path)


def atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def bh(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & np.isfinite(numeric)
    if not valid.any():
        return result
    selected = numeric.loc[valid].to_numpy(float)
    order = np.argsort(selected, kind="stable")
    ranked = selected[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    result.loc[valid] = restored
    return result


def rint(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & np.isfinite(numeric)
    count = int(valid.sum())
    if count:
        ranks = rankdata(numeric.loc[valid].to_numpy(float), method="average")
        result.loc[valid] = norm.ppf((ranks - 0.5) / count)
    return result


def robust_scale(values: pd.Series | np.ndarray) -> float:
    numeric = np.asarray(values, dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if not len(numeric):
        return 1.0
    median = float(np.median(numeric))
    candidates = (
        float(np.median(np.abs(numeric))),
        float(np.median(np.abs(numeric - median))),
        float(np.subtract(*np.quantile(numeric, [0.75, 0.25])) / 1.349),
        float(np.std(numeric, ddof=1)) if len(numeric) > 1 else np.nan,
    )
    minimum = math.sqrt(np.finfo(float).eps)
    return next(
        (value for value in candidates if np.isfinite(value) and value > minimum),
        1.0,
    )


def recover_public_map() -> pd.DataFrame:
    public = pd.read_parquet(APP_ROOT / "data/sample_metadata.parquet")
    private = pd.read_parquet(
        REPO_ROOT
        / "out/lioness_reference_comparison_rosmap/"
        "20260817_standard_control_anchored_allmodules_5phenotypes_6features/"
        "standard/data/phenotypes.parquet"
    ).rename(
        columns={
            "age_death.x": "age_at_death",
            "educ.x": "education_years",
            "braaksc": "braak_stage",
            "ceradsc": "cerad_score",
        }
    )
    keys = [
        "diagnosis_group",
        *PHENOTYPES,
        "age_at_death",
        "education_years",
        "braak_stage",
        "cerad_score",
        "adnc",
    ]
    if public[keys].duplicated(keep=False).any() or private[keys].duplicated(keep=False).any():
        raise ValueError("Existing pseudonym fingerprint is not unique")
    joined = public[["sample_id", *keys]].merge(
        private[["donor", *keys]], on=keys, how="left", validate="one_to_one"
    )
    if joined["donor"].isna().any() or joined["donor"].nunique() != 450:
        raise ValueError("Could not recover all existing public pseudonyms")
    return joined[["donor", "sample_id"]].astype(str)


def sample_map(donors: set[str]) -> dict[str, str]:
    existing = recover_public_map()
    if PRIVATE_MAP.exists():
        mapping = pd.read_csv(PRIVATE_MAP, sep="\t", dtype=str)
    else:
        mapping = existing.copy()
    check = mapping.merge(existing, on="donor", suffixes=("", "_existing"))
    if len(check) != 450 or not check["sample_id"].eq(check["sample_id_existing"]).all():
        raise ValueError("The private map does not preserve the deployed sample IDs")
    known = set(mapping["donor"])
    used = set(mapping["sample_id"])
    rows = []
    for donor in sorted(donors - known):
        candidate = f"S-{secrets.token_hex(6)}"
        while candidate in used:
            candidate = f"S-{secrets.token_hex(6)}"
        used.add(candidate)
        rows.append({"donor": donor, "sample_id": candidate})
    if rows:
        mapping = pd.concat([mapping, pd.DataFrame(rows)], ignore_index=True)
    if mapping["donor"].duplicated().any() or mapping["sample_id"].duplicated().any():
        raise ValueError("Pseudonym map is not one-to-one")
    PRIVATE_MAP.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(PRIVATE_MAP, sep="\t", index=False)
    missing = donors.difference(mapping["donor"])
    if missing:
        raise ValueError(f"Missing pseudonyms for {len(missing)} donors")
    return dict(zip(mapping["donor"], mapping["sample_id"], strict=True))


def collapsed_metadata(metrics: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    source = pd.read_csv(MUSCLE_METADATA, low_memory=False)
    source["ProjID"] = source["ProjID"].astype(str).str.replace(r"\.0$", "", regex=True)
    target = metrics[["donor", "projid", "diagnosis_group"]].drop_duplicates()
    target["projid"] = target["projid"].astype(str).str.replace(r"\.0$", "", regex=True)
    source = source.loc[source["ProjID"].isin(set(target["projid"]))].copy()
    fields = {
        "cogn_global": "cogn_global",
        "cogng_demog_slope": "cogng_demog_slope",
        "cogng_path_slope": "cogng_path_slope",
        "motor10_demog_slope": "motor10_demog_slope",
        "sqrt_parksc_demog_slope": "sqrt_parksc_demog_slope",
        "age_death": "age_at_death",
        "educ": "education_years",
        "cogdx": "cogdx",
        "braaksc": "braak_stage",
        "ceradsc": "cerad_score",
        "adnc": "adnc",
        "parkinsonism_yn_lv": "parkinsonism",
        "msex": "sex_code",
        "apoe_genotype": "apoe_genotype",
        "clusters": "clusters",
    }
    for column in fields:
        conflicts = source.groupby("ProjID", observed=True)[column].nunique(dropna=True)
        if conflicts.gt(1).any():
            raise ValueError(f"Conflicting muscle metadata values for {column}")
    collapsed = source.drop_duplicates("ProjID")[["ProjID", *fields]].rename(columns=fields)
    result = target.merge(
        collapsed, left_on="projid", right_on="ProjID", how="left", validate="one_to_one"
    ).drop(columns=["ProjID", "projid"])
    if len(result) != 357 or result["diagnosis_group"].value_counts().to_dict() != {
        "AD": 155,
        "Control": 120,
        "MCI": 82,
    }:
        raise ValueError("Unexpected muscle cohort counts")
    from scripts.build_pathology_outcomes import augment_metadata
    result = augment_metadata(result, result["donor"])
    result.insert(0, "sample_id", result["donor"].map(mapping))
    result = result.drop(columns="donor")
    result["sex_code"] = result["sex_code"].map({0: "Code 0", 1: "Code 1"}).astype("string")

    def apoe(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        text = str(int(float(value)))
        return f"ε{text[0]}/ε{text[1]}" if len(text) == 2 else f"Code {text}"

    result["apoe_genotype"] = result["apoe_genotype"].map(apoe).astype("string")
    result["parkinsonism_label"] = result["parkinsonism"].map({0.0: "No", 1.0: "Yes"}).astype("string")
    result["clusters"] = pd.to_numeric(result["clusters"], errors="coerce").astype("Int64")
    result["ad_control_split"] = pd.NA
    for column in OUTCOMES:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
    return result


def transformed_tables(
    source: pd.DataFrame, metadata: pd.DataFrame, mapping: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = source[["donor", "module", "diagnosis_group", "lioness_method"]].copy()
    base["sample_id"] = base["donor"].map(mapping)
    embedded = metadata[["sample_id", *PHENOTYPES]]
    base = base.drop(columns="donor").merge(
        embedded, on="sample_id", how="left", validate="many_to_one"
    )
    aggregate_parts = []
    resolved_parts = []
    for feature in FEATURES:
        aggregate = base.copy()
        aggregate["metric_family"] = feature
        for component in ("CT", "TS"):
            raw = pd.to_numeric(source[f"{feature}_{component}"], errors="coerce")
            scale_values = (
                pd.DataFrame(
                    {
                        "lioness_method": source["lioness_method"],
                        "module": source["module"],
                        "value": raw,
                    }
                )
                .groupby(["lioness_method", "module"], observed=True, sort=False)["value"]
                .transform(robust_scale)
            )
            aggregate[f"{component}_raw"] = raw
            aggregate[f"{component}_asinh"] = np.arcsinh(raw / scale_values)
            aggregate[f"{component}_rint"] = (
                pd.DataFrame(
                    {
                        "lioness_method": source["lioness_method"],
                        "module": source["module"],
                        "value": raw,
                    }
                )
                .groupby(["lioness_method", "module"], observed=True, sort=False)["value"]
                .transform(rint)
            )
        aggregate_parts.append(aggregate)
        for component in COMPONENTS:
            resolved = base.copy()
            resolved["metric_family"] = feature
            resolved["component"] = component
            resolved["component_class"] = component.split("_", 1)[0]
            resolved["component_label"] = COMPONENT_LABELS[component]
            raw = pd.to_numeric(source[f"{feature}_{component}"], errors="coerce")
            scale_values = (
                pd.DataFrame(
                    {
                        "lioness_method": source["lioness_method"],
                        "module": source["module"],
                        "value": raw,
                    }
                )
                .groupby(["lioness_method", "module"], observed=True, sort=False)["value"]
                .transform(robust_scale)
            )
            resolved["metric_raw"] = raw
            resolved["metric_asinh"] = np.arcsinh(raw / scale_values)
            resolved["metric_rint"] = (
                pd.DataFrame(
                    {
                        "lioness_method": source["lioness_method"],
                        "module": source["module"],
                        "value": raw,
                    }
                )
                .groupby(["lioness_method", "module"], observed=True, sort=False)["value"]
                .transform(rint)
            )
            resolved_parts.append(resolved)
    aggregate = pd.concat(aggregate_parts, ignore_index=True)
    resolved = pd.concat(resolved_parts, ignore_index=True)
    aggregate = aggregate.sort_values(
        ["lioness_method", "module", "metric_family", "sample_id"], kind="stable"
    ).reset_index(drop=True)
    resolved = resolved.sort_values(
        ["lioness_method", "module", "metric_family", "component", "sample_id"],
        kind="stable",
    ).reset_index(drop=True)
    return aggregate, resolved


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> tuple[int, float, float]:
    pair = pd.DataFrame(
        {"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}
    ).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(pair)
    if n < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return n, np.nan, np.nan
    result = pearsonr(pair["x"], pair["y"]) if method == "pearson" else spearmanr(pair["x"], pair["y"])
    return n, float(result.statistic), float(result.pvalue)


def slope_values(x: pd.Series, y: pd.Series) -> tuple[float, float, float, float]:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair.x.nunique() < 2 or pair.y.nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(pair.x.to_numpy(float), pair.y.to_numpy(float), 1)
    return float(intercept), float(slope), float(pair.x.min()), float(pair.x.max())


def winsor(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return numeric
    low, high = finite.quantile([0.05, 0.95])
    return numeric.clip(low, high)


def _fast_corr(x: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(valid.sum())
    if n < 4:
        return n, np.nan, np.nan
    selected_x = x[valid]
    selected_y = y[valid]
    centered_x = selected_x - selected_x.mean()
    centered_y = selected_y - selected_y.mean()
    denominator = float(
        np.sqrt(np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y))
    )
    if denominator <= 0 or not np.isfinite(denominator):
        return n, np.nan, np.nan
    coefficient = float(np.clip(np.dot(centered_x, centered_y) / denominator, -1, 1))
    residual = max(0.0, 1.0 - coefficient * coefficient)
    if residual <= np.finfo(float).eps:
        p_value = 0.0
    else:
        statistic = abs(coefficient) * math.sqrt((n - 2) / residual)
        p_value = float(2.0 * student_t.sf(statistic, n - 2))
    return n, coefficient, p_value


def _fast_spearman(x: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(valid.sum())
    if n < 4:
        return n, np.nan, np.nan
    return _fast_corr(
        rankdata(x[valid], method="average"),
        rankdata(y[valid], method="average"),
    )


def _fast_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 4:
        return np.nan, np.nan, np.nan, np.nan
    selected_x = x[valid]
    selected_y = y[valid]
    centered_x = selected_x - selected_x.mean()
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 0 or np.unique(selected_y).size < 2:
        return np.nan, np.nan, np.nan, np.nan
    slope = float(np.dot(centered_x, selected_y - selected_y.mean()) / denominator)
    intercept = float(selected_y.mean() - slope * selected_x.mean())
    return intercept, slope, float(selected_x.min()), float(selected_x.max())


def _robust_skew(values: np.ndarray) -> float:
    selected = values[np.isfinite(values)]
    if len(selected) < 3:
        return np.nan
    standard_deviation = float(np.std(selected, ddof=1))
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return np.nan
    return float(np.mean(np.power(selected - selected.mean(), 3)) / standard_deviation**3)


def _winsorize_asinh(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if valid.any():
        lower, upper = np.quantile(
            values[valid], [0.025, 0.975], method="median_unbiased"
        )
        result[valid] = np.clip(values[valid], lower, upper)
    return result


def _leave_one_out(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    selected_x = x[valid]
    selected_y = y[valid]
    n = len(selected_x)
    if n < 6 or np.unique(selected_x).size < 2 or np.unique(selected_y).size < 2:
        return np.nan, np.nan, np.nan
    sum_x, sum_y = selected_x.sum(), selected_y.sum()
    sum_x2, sum_y2 = np.square(selected_x).sum(), np.square(selected_y).sum()
    sum_xy = (selected_x * selected_y).sum()
    full_covariance = sum_xy - sum_x * sum_y / n
    full_variance_x = sum_x2 - sum_x**2 / n
    full_variance_y = sum_y2 - sum_y**2 / n
    full_denominator_sq = max(full_variance_x, 0.0) * max(full_variance_y, 0.0)
    if not np.isfinite(full_denominator_sq) or full_denominator_sq <= 0:
        return np.nan, np.nan, np.nan
    full = full_covariance / math.sqrt(full_denominator_sq)
    n_minus = n - 1
    sum_x_minus = sum_x - selected_x
    sum_y_minus = sum_y - selected_y
    covariance = (
        sum_xy - selected_x * selected_y
        - sum_x_minus * sum_y_minus / n_minus
    )
    variance_x = sum_x2 - selected_x**2 - sum_x_minus**2 / n_minus
    variance_y = sum_y2 - selected_y**2 - sum_y_minus**2 / n_minus
    denominator = np.sqrt(np.maximum(variance_x, 0) * np.maximum(variance_y, 0))
    estimates = np.divide(
        covariance,
        denominator,
        out=np.full(n, np.nan),
        where=denominator > 0,
    )
    estimates = estimates[np.isfinite(estimates)]
    if not len(estimates):
        return np.nan, np.nan, np.nan
    return (
        float(np.max(np.abs(estimates - full))),
        float(np.min(np.abs(estimates))),
        float(np.any(np.sign(estimates) != np.sign(full))),
    )


def _williams_test(
    r_y_ct: float, r_y_ts: float, r_ct_ts: float, n: int
) -> tuple[float, float]:
    values = np.asarray([r_y_ct, r_y_ts, r_ct_ts], dtype=float)
    if n < 6 or not np.isfinite(values).all():
        return np.nan, np.nan
    determinant = (
        1.0 - r_y_ct**2 - r_y_ts**2 - r_ct_ts**2
        + 2.0 * r_y_ct * r_y_ts * r_ct_ts
    )
    denominator_sq = (
        2.0 * determinant * (n - 1) / (n - 3)
        + ((r_y_ct + r_y_ts) ** 2 / 4.0) * (1.0 - r_ct_ts) ** 3
    )
    if not np.isfinite(denominator_sq) or denominator_sq <= 0:
        return np.nan, np.nan
    statistic = (
        (r_y_ct - r_y_ts)
        * math.sqrt((n - 1) * (1.0 + r_ct_ts))
        / math.sqrt(denominator_sq)
    )
    return float(statistic), float(2.0 * student_t.sf(abs(statistic), n - 3))


def resolved_statistics(resolved: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    joined = resolved.merge(
        metadata[["sample_id", *[value for value in OUTCOMES if value not in PHENOTYPES]]],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    for column in ["metric_raw", "metric_asinh", "metric_rint", *OUTCOMES]:
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
    rows = []
    group_cols = ["module", "metric_family", "component", "component_class", "component_label", "diagnosis_group", "lioness_method"]
    for keys, group in joined.groupby(group_cols, observed=True, sort=False):
        base = dict(zip(group_cols, keys, strict=True))
        raw = group["metric_raw"].to_numpy(float, copy=False)
        asinh = group["metric_asinh"].to_numpy(float, copy=False)
        zscore = group["metric_rint"].to_numpy(float, copy=False)
        finite_raw = raw[np.isfinite(raw)]
        finite_asinh = asinh[np.isfinite(asinh)]
        finite_rint = zscore[np.isfinite(zscore)]
        winsorized = _winsorize_asinh(asinh)
        raw_scale = robust_scale(finite_raw)
        raw_constant = len(finite_raw) == 0 or np.unique(finite_raw).size <= 1
        skew_raw = _robust_skew(finite_raw)
        skew_asinh = _robust_skew(finite_asinh)
        skew_rint = _robust_skew(finite_rint)
        for outcome in OUTCOMES:
            y = group[outcome].to_numpy(float, copy=False)
            n, r_raw, p_raw = _fast_corr(raw, y)
            _, r_asinh, p_asinh = _fast_corr(asinh, y)
            _, rho, p_spearman = _fast_spearman(raw, y)
            _, r_winsor, p_winsor = _fast_corr(winsorized, y)
            _, r_rint, p_rint = _fast_corr(zscore, y)
            intercept, slope, xmin, xmax = _fast_slope(zscore, y)
            loo_asinh = _leave_one_out(asinh, y)
            loo_rint = _leave_one_out(zscore, y)
            rows.append(
                {
                    **base,
                    "estimator": "lioness",
                    "network_method": base["lioness_method"],
                    "phenotype": outcome,
                    "n": n,
                    "scale": raw_scale,
                    "metric_zero_variance": raw_constant,
                    "skew_raw": skew_raw,
                    "skew_asinh": skew_asinh,
                    "skew_rint": skew_rint,
                    "r_raw": r_raw,
                    "p_raw": p_raw,
                    "r_asinh": r_asinh,
                    "p_asinh": p_asinh,
                    "rho": rho,
                    "p_spearman": p_spearman,
                    "r_winsor": r_winsor,
                    "p_winsor": p_winsor,
                    "r_rint": r_rint,
                    "p_rint": p_rint,
                    "intercept_rint": intercept,
                    "slope_rint": slope,
                    "x_min_rint": xmin,
                    "x_max_rint": xmax,
                    "loo_asinh_max_delta": loo_asinh[0],
                    "loo_asinh_min_abs": loo_asinh[1],
                    "loo_asinh_sign_flip": loo_asinh[2],
                    "loo_rint_max_delta": loo_rint[0],
                    "loo_rint_min_abs": loo_rint[1],
                    "loo_rint_sign_flip": loo_rint[2],
                }
            )
    result = pd.DataFrame(rows)
    family = ["lioness_method", "metric_family", "component", "diagnosis_group", "phenotype"]
    grouped = result.groupby(family, observed=True, dropna=False, sort=False)
    result["q_rint_within_phenotype"] = grouped["p_rint"].transform(bh)
    result["q_spearman_within_phenotype"] = grouped["p_spearman"].transform(bh)
    for suffix in ("all12_global", "primary5_global", "global"):
        result[f"q_rint_{suffix}"] = result["q_rint_within_phenotype"]
        result[f"q_spearman_{suffix}"] = result["q_spearman_within_phenotype"]
    return result


def aggregate_statistics(aggregate: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    # Reuse the resolved calculator, then pivot CT/TS fields to the legacy schema.
    parts = []
    for component in ("CT", "TS"):
        frame = aggregate.copy()
        frame["component"] = component
        frame["component_class"] = component
        frame["component_label"] = f"{component} aggregate"
        frame["metric_raw"] = frame[f"{component}_raw"]
        frame["metric_asinh"] = frame[f"{component}_asinh"]
        frame["metric_rint"] = frame[f"{component}_rint"]
        parts.append(frame)
    long = resolved_statistics(pd.concat(parts, ignore_index=True), metadata)
    identifiers = [
        "module", "metric_family", "diagnosis_group", "estimator",
        "lioness_method", "network_method", "phenotype",
    ]
    value_columns = [
        column for column in long.columns
        if column not in {*identifiers, "component", "component_class", "component_label"}
    ]
    wide_parts = []
    for component in ("CT", "TS"):
        part = long.loc[long["component"].eq(component), [*identifiers, *value_columns]].copy()
        part = part.rename(columns={column: f"{column}_{component}" for column in value_columns})
        wide_parts.append(part)
    result = wide_parts[0].merge(wide_parts[1], on=identifiers, validate="one_to_one")
    joined = aggregate.merge(
        metadata[["sample_id", *[value for value in OUTCOMES if value not in PHENOTYPES]]],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    component_rows = []
    component_group_columns = [
        "module", "metric_family", "diagnosis_group", "lioness_method"
    ]
    for keys, group in joined.groupby(
        component_group_columns, observed=True, sort=False
    ):
        base = dict(zip(component_group_columns, keys, strict=True))
        ct_raw = group["CT_raw"].to_numpy(float, copy=False)
        ts_raw = group["TS_raw"].to_numpy(float, copy=False)
        ct_asinh = group["CT_asinh"].to_numpy(float, copy=False)
        ts_asinh = group["TS_asinh"].to_numpy(float, copy=False)
        ct_rint = group["CT_rint"].to_numpy(float, copy=False)
        ts_rint = group["TS_rint"].to_numpy(float, copy=False)
        for outcome in OUTCOMES:
            y = pd.to_numeric(group[outcome], errors="coerce").to_numpy(float)
            valid = np.isfinite(y) & np.isfinite(ct_raw) & np.isfinite(ts_raw)
            n = int(valid.sum())
            if n:
                selected_y = y[valid]
                selected_ct_raw, selected_ts_raw = ct_raw[valid], ts_raw[valid]
                selected_ct_asinh, selected_ts_asinh = ct_asinh[valid], ts_asinh[valid]
                selected_ct_rint, selected_ts_rint = ct_rint[valid], ts_rint[valid]
            else:
                selected_y = selected_ct_raw = selected_ts_raw = np.empty(0)
                selected_ct_asinh = selected_ts_asinh = np.empty(0)
                selected_ct_rint = selected_ts_rint = np.empty(0)

            _, r_asinh_ct, _ = _fast_corr(selected_ct_asinh, selected_y)
            _, r_asinh_ts, _ = _fast_corr(selected_ts_asinh, selected_y)
            _, r_asinh_components, _ = _fast_corr(
                selected_ct_asinh, selected_ts_asinh
            )
            asinh_test = _williams_test(
                r_asinh_ct, r_asinh_ts, r_asinh_components, n
            )

            _, r_rint_ct, _ = _fast_corr(selected_ct_rint, selected_y)
            _, r_rint_ts, _ = _fast_corr(selected_ts_rint, selected_y)
            _, r_rint_components, _ = _fast_corr(
                selected_ct_rint, selected_ts_rint
            )
            rint_test = _williams_test(
                r_rint_ct, r_rint_ts, r_rint_components, n
            )

            _, rho_ct, _ = _fast_spearman(selected_ct_raw, selected_y)
            _, rho_ts, _ = _fast_spearman(selected_ts_raw, selected_y)
            if n >= 4:
                rank_ct = rankdata(selected_ct_raw, method="average")
                rank_ts = rankdata(selected_ts_raw, method="average")
                _, rank_components, _ = _fast_corr(rank_ct, rank_ts)
            else:
                rank_components = np.nan
            rank_test = _williams_test(rho_ct, rho_ts, rank_components, n)
            component_rows.append(
                {
                    **base,
                    "phenotype": outcome,
                    "n_joint": n,
                    "component_t_asinh": asinh_test[0],
                    "p_component_asinh": asinh_test[1],
                    "component_t_rank": rank_test[0],
                    "p_component_rank": rank_test[1],
                    "component_t_rint": rint_test[0],
                    "p_component_rint": rint_test[1],
                }
            )
    component_tests = pd.DataFrame(component_rows)
    component_family = [
        "lioness_method", "metric_family", "diagnosis_group", "phenotype"
    ]
    for transformation in ("asinh", "rank", "rint"):
        p_column = f"p_component_{transformation}"
        component_tests[f"q_component_{transformation}_within_phenotype"] = (
            component_tests.groupby(
                component_family, observed=True, sort=False
            )[p_column].transform(bh)
        )
    result = result.merge(
        component_tests,
        on=["module", "metric_family", "diagnosis_group", "lioness_method", "phenotype"],
        how="left",
        validate="one_to_one",
    )
    result["n"] = result.pop("n_joint")
    # Legacy aliases used by the statistics and CT-vs-TS screens.
    for component in ("CT", "TS"):
        result[f"q_rint_{component}_within_phenotype"] = result[f"q_rint_within_phenotype_{component}"]
        result[f"q_spearman_{component}_within_phenotype"] = result[f"q_spearman_within_phenotype_{component}"]
        for suffix in ("all12_global", "primary5_global", "global"):
            result[f"q_rint_{component}_{suffix}"] = result[f"q_rint_within_phenotype_{component}"]
            result[f"q_spearman_{component}_{suffix}"] = result[f"q_spearman_within_phenotype_{component}"]
    for transformation in ("rint", "asinh", "rank"):
        within = result[f"q_component_{transformation}_within_phenotype"]
        result[f"q_component_{transformation}_all12_global"] = within
        result[f"q_component_{transformation}_primary5_global"] = within
        if transformation != "rank":
            result[f"q_component_{transformation}_global"] = within
    return result


def categorical_statistics(
    aggregate: pd.DataFrame,
    resolved: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Build compact nominal Cluster 1–4 statistics for low-I/O app reads."""

    metadata_columns = metadata[["sample_id", "clusters"]]
    pieces: list[pd.DataFrame] = []
    for resolution in ("aggregate", "resolved"):
        if resolution == "aggregate":
            component_frames = []
            for component in ("CT", "TS"):
                part = aggregate[
                    [
                        "sample_id", "module", "metric_family", "diagnosis_group",
                        "lioness_method", f"{component}_raw",
                    ]
                ].copy()
                part = part.rename(columns={f"{component}_raw": "metric_value"})
                part["component"] = component
                part["component_label"] = f"{component} aggregate"
                component_frames.append(part)
            base = pd.concat(component_frames, ignore_index=True)
        else:
            base = resolved[
                [
                    "sample_id", "module", "metric_family", "component",
                    "component_label", "diagnosis_group", "lioness_method", "metric_raw",
                ]
            ].rename(columns={"metric_raw": "metric_value"})
        base = base.merge(metadata_columns, on="sample_id", how="left", validate="many_to_one")
        pooled = base.copy()
        pooled["diagnosis_group"] = "All donors"
        analysis = pd.concat([base, pooled], ignore_index=True)
        group_columns = [
            "module", "metric_family", "component", "component_label",
            "diagnosis_group", "lioness_method",
        ]
        result = calculate_categorical_associations(
            analysis,
            group_columns,
            category_column="clusters",
            min_group_n=5,
        )
        result = add_categorical_across_module_fdr(
            result,
            family_columns=[
                "lioness_method", "metric_family", "component",
                "diagnosis_group", "outcome",
            ],
        )
        result.insert(0, "module_set", MODULE_SET)
        result.insert(1, "estimator", "lioness")
        result.insert(2, "network_method", result["lioness_method"])
        result.insert(3, "edge_rule", "all")
        result.insert(4, "resolution", resolution)
        result = result.drop(columns="lioness_method")
        result["clusters_are_nominal"] = True
        result["numeric_correlation_used"] = False
        result["effect_definition"] = "max(0,(Kruskal H-k+1)/(n-k))"
        pieces.append(result)
    return pd.concat(pieces, ignore_index=True)


def module_eigengenes(
    assignments: pd.DataFrame,
    donors: set[str],
    mapping: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Calculate outcome-independent PCA1 summaries per module and tissue."""

    expressions: dict[str, pd.DataFrame] = {}
    source_files: list[dict[str, object]] = []
    for tissue in ("VH", "SMA", "QUAD"):
        parts = []
        for diagnosis in ("Control", "MCI", "AD"):
            path = MUSCLE_EXPRESSION_ROOT / (
                f"ROSMAP_tissue_{tissue.lower()}_{diagnosis}.csv"
            )
            frame = pd.read_csv(path)
            donor_column = frame.columns[0]
            frame = frame.rename(columns={donor_column: "donor"})
            frame["donor"] = frame["donor"].astype(str)
            parts.append(frame)
            source_files.append(
                {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path)}
            )
        expression = pd.concat(parts, ignore_index=True)
        if expression["donor"].duplicated().any():
            raise ValueError(f"{tissue}: duplicate expression donors")
        if not donors.issubset(set(expression["donor"])):
            missing_donors = donors.difference(expression["donor"])
            raise ValueError(
                f"{tissue}: {len(missing_donors)} complete-cohort donors are missing"
            )
        expressions[tissue] = expression.set_index("donor").loc[sorted(donors)]

    symbol_lookup = gene_symbol_mapping()
    rows: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for (module, tissue), group in assignments.groupby(
        ["Cluster ID", "Tissue"], observed=True, sort=True
    ):
        genes = group["Gene ID"].astype(str).tolist()
        expression = expressions[str(tissue)]
        missing = sorted(set(genes).difference(expression.columns))
        if missing:
            raise ValueError(f"M{module} {tissue}: {len(missing)} genes missing from expression")
        matrix = expression[genes].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        medians = np.nanmedian(matrix, axis=0)
        missing_cells = np.where(~np.isfinite(matrix))
        if len(missing_cells[0]):
            matrix[missing_cells] = medians[missing_cells[1]]
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0, ddof=0)
        variable = np.isfinite(scales) & (scales > 0)
        if not variable.any():
            score = np.zeros(len(matrix), dtype=float)
            variance_explained = 0.0
            anchor_gene = genes[0]
            applied_sign = 1.0
        else:
            standardized = (matrix[:, variable] - means[variable]) / scales[variable]
            u, singular, vt = randomized_svd(
                standardized,
                n_components=1,
                n_iter=7,
                random_state=42,
                flip_sign=False,
            )
            loadings = vt[0]
            anchor_position = int(np.argmax(np.abs(loadings)))
            applied_sign = 1.0 if loadings[anchor_position] >= 0 else -1.0
            score = u[:, 0] * singular[0] * applied_sign
            variable_genes = np.asarray(genes, dtype=object)[variable]
            anchor_gene = str(variable_genes[anchor_position])
            total_variance = float(np.square(standardized).sum())
            variance_explained = (
                float(singular[0] ** 2 / total_variance) if total_variance > 0 else 0.0
            )
        component = f"TS_{tissue}"
        part = pd.DataFrame(
            {
                "sample_id": [mapping[donor] for donor in expression.index],
                "module": int(module),
                "metric_family": "eigengene",
                "component": component,
                "component_class": "TS",
                "component_label": f"TS: {tissue}",
                "metric_raw": score,
                "metric_asinh": np.arcsinh(score),
                "metric_rint": rint(pd.Series(score)).to_numpy(float),
            }
        )
        rows.append(part)
        audits.append(
            {
                "module": int(module),
                "tissue": str(tissue),
                "n_genes": len(genes),
                "n_variable_genes": int(variable.sum()),
                "imputed_cells": int(len(missing_cells[0])),
                "variance_explained_pc1": variance_explained,
                "orientation_rule": "largest absolute PCA1 loading is positive",
                "orientation_anchor_gene_symbol": symbol_lookup.get(
                    anchor_gene, "Unmapped gene"
                ),
                "applied_sign": applied_sign,
            }
        )
    result = pd.concat(rows, ignore_index=True).sort_values(
        ["module", "component", "sample_id"], kind="stable"
    ).reset_index(drop=True)
    if result["sample_id"].nunique() != 357 or result["module"].nunique() != MODULE_COUNT:
        raise AssertionError("Unexpected eigengene donor or module coverage")
    if {"donor", "projid", "ProjID"}.intersection(result.columns):
        raise AssertionError("Private donor identifiers entered eigengene output")
    manifest = {
        "status": "complete",
        "created_utc": utc_now(),
        "method": "PCA1 of within-tissue donor-standardized expression",
        "fit_cohort": "357 donors with complete VH, SMA, and QUAD expression",
        "orientation_rule": "largest absolute PCA1 loading is positive",
        "module_tissue_fits": len(audits),
        "rows": len(result),
        "modules": int(result["module"].nunique()),
        "donors": int(result["sample_id"].nunique()),
        "component_counts": result["component"].value_counts().to_dict(),
        "sources": source_files,
        "fit_audit": audits,
    }
    return result, manifest


def module_details(assignments: pd.DataFrame) -> pd.DataFrame:
    records = []
    for module, group in assignments.groupby("Cluster ID", observed=True, sort=True):
        counts = group["Tissue"].value_counts()
        total = len(group)
        proportions = {tissue: int(counts.get(tissue, 0)) / total for tissue in ("VH", "SMA", "QUAD")}
        positive = np.asarray([value for value in proportions.values() if value > 0])
        entropy = float(-(positive * np.log2(positive)).sum())
        dominant = max(proportions, key=proportions.get)
        records.append(
            {
                "module": int(module),
                "module_size": total,
                "cluster_type": "TS" if proportions[dominant] >= 0.95 else "CT",
                "tissues": ", ".join(tissue for tissue in ("VH", "SMA", "QUAD") if counts.get(tissue, 0)),
                "n_tissues": int((counts > 0).sum()),
                "dominant_tissue": dominant,
                "n_genes_vh": int(counts.get("VH", 0)),
                "n_genes_sma": int(counts.get("SMA", 0)),
                "n_genes_quad": int(counts.get("QUAD", 0)),
                "proportion_vh": proportions["VH"],
                "proportion_sma": proportions["SMA"],
                "proportion_quad": proportions["QUAD"],
                "tissue_entropy": entropy,
                "tissue_entropy_normalized": entropy / math.log2(3),
            }
        )
    return pd.DataFrame(records)


def gene_symbol_mapping() -> dict[str, str]:
    frame = pd.read_csv(
        REPO_ROOT
        / "data/proccessed/M5_all_genes_muscles/target_annotations/gene_target_annotations.tsv",
        sep="\t",
        dtype=str,
    )
    symbols = frame["hgnc_symbol"].astype("string").str.strip()
    symbols = symbols.mask(
        symbols.isna() | symbols.eq("") | symbols.str.startswith("ENSG", na=False),
        "Unmapped gene",
    )
    return dict(
        zip(
            frame["ensembl_gene_id_base"].astype(str),
            symbols.astype(str),
            strict=True,
        )
    )


def replace_overlap_genes(value: object, symbols: dict[str, str]) -> object:
    if pd.isna(value):
        return value
    tokens = []
    for token in str(value).split(","):
        match = re.match(r"(ENSG\d+)(.*)", token.strip())
        if match:
            tokens.append(symbols.get(match.group(1), "Unmapped gene") + match.group(2))
        else:
            tokens.append(token.strip())
    return ",".join(tokens)


def kegg_tables(modules: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    expanded_source = pd.read_csv(
        KEGG_ROOT / "method4_tissue_expanded_kegg_annotated.tsv", sep="\t"
    )
    regional = pd.read_csv(KEGG_ROOT / "method2_meta_kegg_annotated.tsv", sep="\t")
    expanded_source = expanded_source.loc[
        expanded_source["cluster_id"].isin(modules)
    ].copy()
    regional = regional.loc[regional["cluster_id"].isin(modules)].copy()
    regional_columns = [
        "cluster_id", "term", "p_VH", "fdr_VH", "significant_VH",
        "p_SMA", "fdr_SMA", "significant_SMA",
        "p_QUAD", "fdr_QUAD", "significant_QUAD",
        "overlap_VH", "overlap_SMA", "overlap_QUAD",
        "pathway_id", "pathway_num", "pathway_name",
        "category_level1", "category_level2",
    ]
    expanded_payload = expanded_source.drop(
        columns=[
            "overlap_VH", "overlap_SMA", "overlap_QUAD", "pathway_id",
            "pathway_num", "pathway_name", "category_level1", "category_level2",
        ]
    )
    # The regional analysis evaluates the stable pathway universe for every
    # module.  Keep that complete frame and left-join tissue-expanded ORA so a
    # module/pathway with no qualifying expanded overlap remains explicit.
    expanded = regional[regional_columns].merge(
        expanded_payload,
        on=["cluster_id", "term"],
        how="left",
        validate="one_to_one",
    )
    expanded["significant"] = expanded["significant"].eq(True)
    symbols = gene_symbol_mapping()
    expanded["overlap_genes"] = expanded["overlap_genes"].map(
        lambda value: replace_overlap_genes(value, symbols)
    )
    ordered = [
        "cluster_id", "cluster_size_expanded", "pathway_id", "pathway_num",
        "pathway_name", "category_level1", "category_level2", "term", "p", "fdr",
        "significant", "p_VH", "fdr_VH", "significant_VH", "p_SMA", "fdr_SMA",
        "significant_SMA", "p_QUAD", "fdr_QUAD", "significant_QUAD", "overlap_k",
        "term_size_expanded", "term_size_original", "universe_expanded", "n_tissues",
        "overlap_VH", "overlap_SMA", "overlap_QUAD", "overlap_genes", "neglog10_p",
        "neglog10_fdr",
    ]
    expanded = expanded[ordered].sort_values(["cluster_id", "fdr"], kind="stable")
    annotations = []
    for module in sorted(modules):
        rows = expanded.loc[expanded["cluster_id"].eq(module)].copy()
        rows = rows.loc[pd.to_numeric(rows["fdr"], errors="coerce").notna()].copy()
        if rows.empty:
            annotations.append({"module": module, "annotation_available": False, "subtitle_text": "KEGG enrichment: unavailable"})
            continue
        rows["priority"] = (
            rows["category_level1"].fillna("") + " "
            + rows["category_level2"].fillna("") + " "
            + rows["pathway_name"].fillna("")
        ).str.contains(PRIORITY)
        preferred = rows.loc[rows["priority"] & rows["fdr"].lt(0.05)]
        displayed = (preferred if not preferred.empty else rows).sort_values("fdr", kind="stable").iloc[0]
        top = rows.sort_values("fdr", kind="stable").iloc[0]
        annotations.append(
            {
                "module": module,
                "annotation_available": True,
                "top_category": top["category_level1"],
                "top_subcategory": top["category_level2"],
                "top_pathway": top["pathway_name"],
                "top_pathway_id": top["pathway_id"],
                "top_fdr": top["fdr"],
                "top_significant": bool(top["fdr"] < 0.05),
                "priority_theme": "priority" if displayed["priority"] else "none",
                "priority_category": displayed["category_level1"],
                "priority_subcategory": displayed["category_level2"],
                "priority_pathway": displayed["pathway_name"],
                "priority_pathway_id": displayed["pathway_id"],
                "priority_fdr": displayed["fdr"],
                "priority_significant": bool(displayed["fdr"] < 0.05),
                "priority_is_top": bool(displayed["pathway_id"] == top["pathway_id"]),
                "displayed_selection": "priority_significant_lowest_fdr" if displayed["priority"] and displayed["fdr"] < 0.05 else "overall_lowest_fdr",
                "displayed_category": displayed["category_level1"],
                "displayed_subcategory": displayed["category_level2"],
                "displayed_pathway": displayed["pathway_name"],
                "displayed_pathway_id": displayed["pathway_id"],
                "displayed_fdr": displayed["fdr"],
                "subtitle_text": (
                    f"KEGG enrichment: {displayed['category_level1']} / "
                    f"{displayed['category_level2']} / {displayed['pathway_name']} | "
                    f"FDR={float(displayed['fdr']):.3g}"
                ),
            }
        )
    return expanded, pd.DataFrame(annotations)


def mdc_tables(details: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    significance_path = MDC_SIGNIFICANCE_ROOT / "mdc_significance.parquet"
    source_path = (
        significance_path
        if significance_path.exists()
        else MDC_ROOT / "mdc_observed.parquet"
    )
    modules = details["module"].astype(int)
    if not source_path.exists():
        summary = pd.DataFrame({"module": modules, "module_size_mapped": details["module_size"]})
        for scope in ("total", "ts", "ct"):
            summary[f"n_{scope}_edges" if scope != "total" else "n_total_edges"] = 0
            summary[f"mdc_{scope}"] = np.nan
            summary[f"log2_mdc_{scope}"] = np.nan
            summary[f"direction_{scope}"] = "Unavailable — M5 recovery running"
            summary[f"directional_fdr_{scope}"] = np.nan
            summary[f"significant_{scope}_fdr05"] = False
            summary[f"significant_{scope}_fdr10"] = False
        summary["n_ts_edges"] = 0
        summary["n_ct_edges"] = 0
        summary["ts_minus_ct_log2_mdc"] = np.nan
        summary["any_significant_fdr05"] = False
        summary["any_significant_fdr10"] = False
        resolved = pd.DataFrame(
            columns=[
                "module_definition", "module", "component", "component_class",
                "n_features", "n_edges", "mdc", "log2_mdc", "direction",
                "directional_fdr", "significant_fdr_05", "significant_fdr_10",
                "component_label",
            ]
        )
        return summary, resolved
    observed = pd.read_parquet(source_path)
    significance_available = "directional_fdr" in observed.columns
    observed["mdc_stable_for_inference"] = (
        pd.to_numeric(observed["n_edges"], errors="coerce").ge(10)
        & pd.to_numeric(observed["mean_abs_ad"], errors="coerce").gt(1e-8)
        & pd.to_numeric(observed["mean_abs_control"], errors="coerce").gt(1e-8)
        & pd.to_numeric(observed["mdc_raw"], errors="coerce").notna()
    )
    observed["mdc_stability_reason"] = ""
    observed.loc[
        pd.to_numeric(observed["n_edges"], errors="coerce").lt(10),
        "mdc_stability_reason",
    ] = "fewer than 10 edges"
    observed.loc[
        pd.to_numeric(observed["mean_abs_ad"], errors="coerce").le(1e-8)
        | pd.to_numeric(observed["mean_abs_control"], errors="coerce").le(1e-8),
        "mdc_stability_reason",
    ] = "near-zero AD or Control mean absolute weight"
    observed.loc[
        pd.to_numeric(observed["mdc_raw"], errors="coerce").isna(),
        "mdc_stability_reason",
    ] = "structurally unavailable or zero denominator"
    aggregate = observed.loc[observed["scope"].isin(["total", "TS", "CT"])].copy()
    pivot = aggregate.pivot(index="module", columns="scope")
    summary = pd.DataFrame({"module": sorted(modules)})
    size_lookup = details.set_index("module")["module_size"]
    summary["module_size_mapped"] = summary["module"].map(size_lookup)
    for scope, name in (("total", "total"), ("TS", "ts"), ("CT", "ct")):
        lookup = aggregate.loc[aggregate["scope"].eq(scope)].set_index("module")
        summary[f"n_{name}_edges"] = summary["module"].map(lookup["n_edges"])
        summary[f"mdc_{name}"] = summary["module"].map(lookup["mdc_raw"])
        summary[f"log2_mdc_{name}"] = summary["module"].map(lookup["mdc_log2"])
        summary[f"mean_abs_ad_{name}"] = summary["module"].map(lookup["mean_abs_ad"])
        summary[f"mean_abs_control_{name}"] = summary["module"].map(
            lookup["mean_abs_control"]
        )
        summary[f"direction_{name}"] = np.where(
            summary[f"mdc_{name}"].ge(1), "Higher in AD", "Higher in Control"
        )
        if significance_available:
            for statistic in (
                "directional_p_sample",
                "directional_p_gene",
                "directional_fdr",
            ):
                summary[f"{statistic}_{name}"] = summary["module"].map(
                    lookup[statistic]
                )
            summary[f"inference_available_{name}"] = summary["module"].map(
                lookup["inference_available"]
            ).fillna(False)
            summary[f"inference_unavailable_reason_{name}"] = summary["module"].map(
                lookup["inference_unavailable_reason"]
            ).fillna("")
        else:
            summary[f"directional_p_sample_{name}"] = np.nan
            summary[f"directional_p_gene_{name}"] = np.nan
            summary[f"directional_fdr_{name}"] = np.nan
            summary[f"inference_available_{name}"] = False
            summary[f"inference_unavailable_reason_{name}"] = (
                "directional permutation inference not evaluated"
            )
        summary[f"mdc_stable_for_inference_{name}"] = summary["module"].map(
            lookup["mdc_stable_for_inference"]
        ).fillna(False)
        summary[f"mdc_stability_reason_{name}"] = summary["module"].map(
            lookup["mdc_stability_reason"]
        ).fillna("")
        summary[f"significant_{name}_fdr05"] = summary[
            f"directional_fdr_{name}"
        ].lt(0.05)
        summary[f"significant_{name}_fdr10"] = summary[
            f"directional_fdr_{name}"
        ].lt(0.10)
    summary["n_ts_edges"] = summary.pop("n_ts_edges")
    summary["n_ct_edges"] = summary.pop("n_ct_edges")
    summary["ts_minus_ct_log2_mdc"] = summary["log2_mdc_ts"] - summary["log2_mdc_ct"]
    summary["any_significant_fdr05"] = summary[
        ["significant_total_fdr05", "significant_ts_fdr05", "significant_ct_fdr05"]
    ].any(axis=1)
    summary["any_significant_fdr10"] = summary[
        ["significant_total_fdr10", "significant_ts_fdr10", "significant_ct_fdr10"]
    ].any(axis=1)
    selected = observed.loc[observed["scope"].isin(COMPONENTS)].copy()
    selected = selected.rename(columns={"scope": "component", "mdc_raw": "mdc", "mdc_log2": "log2_mdc"})
    selected.insert(0, "module_definition", MODULE_SET)
    selected["component_class"] = selected["component"].str.split("_", n=1).str[0]
    selected["n_features"] = selected["module"].map(size_lookup)
    selected["direction"] = np.where(selected["mdc"].ge(1), "Higher in AD", "Higher in Control")
    if not significance_available:
        selected["directional_fdr"] = np.nan
        selected["significant_fdr_05"] = False
        selected["significant_fdr_10"] = False
        selected["inference_available"] = False
        selected["inference_unavailable_reason"] = (
            "directional permutation inference not evaluated"
        )
    selected["component_label"] = selected["component"].map(COMPONENT_LABELS)
    for column in (
        "p_loss_sample", "p_loss_gene", "q_loss_sample", "q_loss_gene", "q_loss_max",
        "p_gain_sample", "p_gain_gene", "q_gain_sample", "q_gain_gene", "q_gain_max",
        "directional_p_sample", "directional_p_gene",
    ):
        selected[column] = np.nan
    columns = [
        "module_definition", "module", "component", "component_class", "n_features",
        "n_edges", "mdc", "log2_mdc", "direction", "p_loss_sample", "p_loss_gene",
        "q_loss_sample", "q_loss_gene", "q_loss_max", "p_gain_sample", "p_gain_gene",
        "q_gain_sample", "q_gain_gene", "q_gain_max", "directional_p_sample",
        "directional_p_gene", "directional_fdr", "significant_fdr_05",
        "significant_fdr_10", "mean_abs_ad", "mean_abs_control",
        "inference_available", "inference_unavailable_reason",
        "mdc_stable_for_inference", "mdc_stability_reason", "component_label",
    ]
    for optional in (
        "sample_permutations",
        "gene_permutations",
        "sample_null_finite_n",
        "gene_null_finite_n",
    ):
        if optional in selected.columns:
            columns.append(optional)
    return summary, selected[columns]


def mdc_manifest() -> dict[str, object]:
    significance_path = MDC_SIGNIFICANCE_ROOT / "mdc_significance.parquet"
    run_manifest_path = MDC_SIGNIFICANCE_ROOT / "run_manifest.json"
    run_manifest = (
        json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest_path.exists()
        else {}
    )
    complete = significance_path.exists() and run_manifest.get("status") == "complete"
    return {
        "functional": "M5 extended SFN recovered-module subnetwork",
        "observed_complete": (MDC_ROOT / "mdc_observed.parquet").exists(),
        "permutation_significance_complete": complete,
        "sample_permutations": int(run_manifest.get("sample_permutations", 0)),
        "gene_permutations": int(run_manifest.get("gene_permutations", 0)),
        "completed_sample_permutations": int(
            run_manifest.get("completed_sample_checkpoints", 0)
        ),
        "completed_gene_permutations": int(
            run_manifest.get("completed_gene_checkpoints", 0)
        ),
        "minimum_inference_edges": int(
            run_manifest.get("minimum_inference_edges", 10)
        ),
        "minimum_group_mean_absolute_weight": float(
            run_manifest.get("minimum_group_mean_absolute_weight", 1e-8)
        ),
        "note": (
            "Directional FDR is the conservative maximum of component- and direction-"
            "specific BH values from the sample and tissue-count-preserving gene nulls."
            if complete
            else "Directional FDR is unavailable until the full-M5 null run completes."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--skip-statistics", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--statistics-only",
        action="store_true",
        help="Build statistics from the already packaged transformed-data files.",
    )
    mode.add_argument(
        "--refresh-mdc-only",
        action="store_true",
        help="Refresh MDC tables from the completed observed full-M5 result.",
    )
    mode.add_argument(
        "--refresh-kegg-only",
        action="store_true",
        help="Refresh complete tissue-expanded and regional KEGG tables.",
    )
    mode.add_argument(
        "--categorical-statistics-only",
        action="store_true",
        help="Build compact nominal Cluster 1–4 statistics from packaged scores.",
    )
    mode.add_argument(
        "--eigengenes-only",
        action="store_true",
        help="Build descriptive within-tissue module eigengenes for the 357 donors.",
    )
    mode.add_argument(
        "--register-existing",
        action="store_true",
        help="Refresh hashes and register an already built validated bundle.",
    )
    parser.add_argument(
        "--update-app-manifest",
        action="store_true",
        help="Register the validated module set in data/data_manifest.json.",
    )
    return parser.parse_args()


def refresh_existing_manifest(output: Path) -> dict[str, object]:
    path = output / "module_set_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing module-set manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["created_utc"] = utc_now()
    manifest["status"] = (
        "complete"
        if (output / "aggregate_statistics.parquet").exists()
        and (output / "resolved_statistics.parquet").exists()
        else "partial"
    )
    manifest["mdc"] = mdc_manifest()
    manifest.setdefault("capabilities", {})["descriptive_eigengenes"] = (
        output / "module_eigengenes.parquet"
    ).exists()
    files = {}
    for file_path in sorted(output.rglob("*")):
        if file_path.is_file() and file_path != path:
            files[str(file_path.relative_to(output))] = {
                "bytes": file_path.stat().st_size,
                "sha256": sha256(file_path),
            }
    manifest["files"] = files
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    return manifest


def update_app_manifest(module_manifest: dict[str, object], output: Path) -> None:
    path = APP_ROOT / "data/data_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = {
        "aggregate_rows": len(pd.read_parquet(output / "aggregate_plot_data.parquet", columns=["module"])),
        "resolved_rows": len(pd.read_parquet(output / "resolved_plot_data.parquet", columns=["module"])),
        "aggregate_stat_rows": (
            len(pd.read_parquet(output / "aggregate_statistics.parquet", columns=["module"]))
            if (output / "aggregate_statistics.parquet").exists()
            else 0
        ),
        "resolved_stat_rows": (
            len(pd.read_parquet(output / "resolved_statistics.parquet", columns=["module"]))
            if (output / "resolved_statistics.parquet").exists()
            else 0
        ),
        "assignment_rows": ASSIGNMENT_COUNT,
        "module_details_rows": MODULE_COUNT,
        "entropy_rows": MODULE_COUNT,
        "mdc_rows": MODULE_COUNT,
        "resolved_mdc_rows": len(pd.read_csv(output / "mdc_resolved_ad_vs_control.tsv", sep="\t")),
    }
    kegg_frame = pd.read_parquet(output / "kegg_tissue_expanded_full.parquet")
    expanded_modules = set(
        pd.to_numeric(
            kegg_frame.loc[kegg_frame["fdr"].notna(), "cluster_id"], errors="coerce"
        ).dropna().astype(int)
    )
    all_modules = set(pd.read_csv(output / "module_details.tsv", sep="\t")["module"].astype(int))
    regional_counts = kegg_frame.groupby("cluster_id", observed=True).size()
    manifest.setdefault("module_sets", {})[MODULE_SET] = {
        "status": "complete",
        "key": MODULE_SET,
        "label": MODULE_LABEL,
        "relative_data_dir": MODULE_SET,
        "source_run": str(RUN_ROOT.relative_to(REPO_ROOT)),
        "modules": MODULE_COUNT,
        "assignments": ASSIGNMENT_COUNT,
        "sentinel_module": 36 if 36 in all_modules else min(all_modules),
        "methods": list(METHODS),
        "estimators": ["lioness"],
        "feature_families": list(FEATURES),
        "feature_families_by_estimator": {"lioness": list(FEATURES)},
        "phenotypes": list(PHENOTYPES),
        "outcomes": list(OUTCOMES),
        "tissues": ["VH", "SMA", "QUAD"],
        "resolved_components": list(COMPONENTS),
        "donors_per_method": {method: 357 for method in METHODS},
        "diagnosis_counts_per_method": {
            method: {"Control": 120, "MCI": 82, "AD": 155}
            for method in METHODS
        },
        "fdr_test_families": (
            f"Association BH is calculated across the {MODULE_COUNT} modules only within each "
            "fixed method, feature, component, outcome, diagnosis/grouping level, and scale."
        ),
        "capabilities": module_manifest["capabilities"],
        "partition_network": module_manifest.get("partition_network", {}),
        "mdc_sensitivity": module_manifest.get("mdc_sensitivity", {}),
        "mdc_label_validation": module_manifest.get("mdc_label_validation", {}),
        "categorical_statistics_local": True,
        "module_details": {
            "classification": "TS when one tissue contributes at least 95%; CT otherwise",
            "tissue_entropy": "Shannon entropy base 2 over VH/SMA/QUAD proportions",
        },
        "kegg": {
            "status": "complete",
            "tissues": ["VH", "SMA", "QUAD"],
            "min_cluster_size": 30,
            "min_term_size": 10,
            "min_overlap": 3,
            "input_modules": MODULE_COUNT,
            "admitted_modules": MODULE_COUNT,
            "modules_with_reported_pathways": len(expanded_modules),
            "modules_without_reported_pathways": sorted(all_modules - expanded_modules),
            "join_coverage": 1.0,
            "join_keys": ["cluster_id", "term"],
            "rows": len(kegg_frame),
            "source_sha256": sha256(
                KEGG_ROOT / "method4_tissue_expanded_kegg_annotated.tsv"
            ),
            "per_region_source_sha256": sha256(
                KEGG_ROOT / "method2_meta_kegg_annotated.tsv"
            ),
            "per_region_input_modules": MODULE_COUNT,
            "per_region_pathways_per_module_min": int(regional_counts.min()),
            "per_region_pathways_per_module_max": int(regional_counts.max()),
            "region_statistics": {
                "VH": ["p_VH", "fdr_VH", "significant_VH"],
                "SMA": ["p_SMA", "fdr_SMA", "significant_SMA"],
                "QUAD": ["p_QUAD", "fdr_QUAD", "significant_QUAD"],
            },
            "whole_expanded_fdr_definition": (
                "Benjamini-Hochberg within each module across pathways meeting "
                "the expanded tissue-gene overlap threshold."
            ),
            "per_region_fdr_definition": (
                "Benjamini-Hochberg separately within each module and region across "
                "the stable 328-pathway universe; sub-threshold overlaps receive p=1."
            ),
        },
        "mdc": module_manifest["mdc"],
        "rows": rows,
        "files": module_manifest["files"],
    }
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.statistics_only:
        metadata = pd.read_parquet(args.output / "sample_metadata.parquet")
        aggregate = pd.read_parquet(args.output / "aggregate_plot_data.parquet")
        resolved = pd.read_parquet(args.output / "resolved_plot_data.parquet")
        atomic_parquet(
            aggregate_statistics(aggregate, metadata),
            args.output / "aggregate_statistics.parquet",
        )
        atomic_parquet(
            resolved_statistics(resolved, metadata),
            args.output / "resolved_statistics.parquet",
        )
        manifest = refresh_existing_manifest(args.output)
        if args.update_app_manifest:
            update_app_manifest(manifest, args.output)
        print(json.dumps({"output": str(args.output), "statistics": "complete"}, indent=2))
        return
    if args.refresh_mdc_only:
        details = pd.read_csv(args.output / "module_details.tsv", sep="\t")
        mdc_summary, mdc_resolved = mdc_tables(details)
        atomic_tsv(mdc_summary, args.output / "mdc_ad_vs_control_summary.tsv")
        atomic_tsv(mdc_resolved, args.output / "mdc_resolved_ad_vs_control.tsv")
        manifest = refresh_existing_manifest(args.output)
        if args.update_app_manifest:
            update_app_manifest(manifest, args.output)
        print(json.dumps({"output": str(args.output), "mdc": "refreshed"}, indent=2))
        return
    if args.refresh_kegg_only:
        details = pd.read_csv(args.output / "module_details.tsv", sep="\t")
        kegg, annotations = kegg_tables(set(details["module"].astype(int)))
        atomic_parquet(kegg, args.output / "kegg_tissue_expanded_full.parquet")
        atomic_tsv(kegg, args.output / "kegg_tissue_expanded_full.tsv")
        atomic_tsv(annotations, args.output / "module_kegg_annotations.tsv")
        manifest = refresh_existing_manifest(args.output)
        if args.update_app_manifest:
            update_app_manifest(manifest, args.output)
        print(json.dumps({"output": str(args.output), "kegg_rows": len(kegg)}, indent=2))
        return
    if args.categorical_statistics_only:
        metadata = pd.read_parquet(args.output / "sample_metadata.parquet")
        aggregate = pd.read_parquet(args.output / "aggregate_plot_data.parquet")
        resolved = pd.read_parquet(args.output / "resolved_plot_data.parquet")
        result = categorical_statistics(aggregate, resolved, metadata)
        atomic_parquet(result, args.output / "cluster_association_statistics.parquet")
        manifest = refresh_existing_manifest(args.output)
        if args.update_app_manifest:
            update_app_manifest(manifest, args.output)
        print(json.dumps({"output": str(args.output), "categorical_rows": len(result)}, indent=2))
        return
    if args.eigengenes_only:
        source = pd.read_parquet(
            RUN_ROOT / METHODS[0] / "donor_module_metrics.parquet",
            columns=["donor"],
        )
        donors = set(source["donor"].astype(str))
        mapping = sample_map(donors)
        assignments = pd.read_csv(ASSIGNMENTS, dtype=str)
        assignments["Cluster ID"] = pd.to_numeric(
            assignments["Cluster ID"], errors="raise"
        )
        sizes = assignments.groupby("Cluster ID", observed=True).size()
        assignments = assignments.loc[
            assignments["Cluster ID"].isin(sizes.loc[sizes.ge(30)].index)
        ].copy()
        scores, eigengene_manifest = module_eigengenes(assignments, donors, mapping)
        atomic_parquet(scores, args.output / "module_eigengenes.parquet")
        manifest_path = args.output / "module_eigengenes_manifest.json"
        manifest_path.write_text(
            json.dumps(eigengene_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = refresh_existing_manifest(args.output)
        if args.update_app_manifest:
            update_app_manifest(manifest, args.output)
        print(json.dumps({"output": str(args.output), "eigengene_rows": len(scores)}, indent=2))
        return
    if args.register_existing:
        manifest = refresh_existing_manifest(args.output)
        update_app_manifest(manifest, args.output)
        print(json.dumps({"output": str(args.output), "registered": True}, indent=2))
        return
    sources = []
    for method in METHODS:
        frame = pd.read_parquet(RUN_ROOT / method / "donor_module_metrics.parquet")
        if len(frame) != MODULE_COUNT * 357 or frame["module"].nunique() != MODULE_COUNT:
            raise ValueError(f"{method}: unexpected source dimensions")
        sources.append(frame)
    source = pd.concat(sources, ignore_index=True)
    donors = set(source["donor"].astype(str))
    mapping = sample_map(donors)
    metadata = collapsed_metadata(source, mapping)
    atomic_parquet(metadata, args.output / "sample_metadata.parquet")

    aggregate, resolved = transformed_tables(source, metadata, mapping)
    if len(aggregate) != MODULE_COUNT * 357 * 6 * len(METHODS):
        raise AssertionError("Unexpected aggregate row count")
    if len(resolved) != MODULE_COUNT * 357 * 6 * 6 * len(METHODS):
        raise AssertionError("Unexpected resolved row count")
    atomic_parquet(aggregate, args.output / "aggregate_plot_data.parquet")
    atomic_parquet(resolved, args.output / "resolved_plot_data.parquet")
    if not args.skip_statistics:
        atomic_parquet(
            aggregate_statistics(aggregate, metadata),
            args.output / "aggregate_statistics.parquet",
        )
        atomic_parquet(
            resolved_statistics(resolved, metadata),
            args.output / "resolved_statistics.parquet",
        )
        atomic_parquet(
            categorical_statistics(aggregate, resolved, metadata),
            args.output / "cluster_association_statistics.parquet",
        )

    assignments = pd.read_csv(ASSIGNMENTS, dtype=str)
    assignments["Cluster ID"] = pd.to_numeric(assignments["Cluster ID"], errors="raise")
    sizes = assignments.groupby("Cluster ID", observed=True).size()
    assignments = assignments.loc[assignments["Cluster ID"].isin(sizes.loc[sizes.ge(30)].index)].copy()
    details = module_details(assignments)
    atomic_tsv(details, args.output / "module_details.tsv")
    kegg, annotations = kegg_tables(set(details["module"].astype(int)))
    atomic_parquet(kegg, args.output / "kegg_tissue_expanded_full.parquet")
    atomic_tsv(kegg, args.output / "kegg_tissue_expanded_full.tsv")
    atomic_tsv(annotations, args.output / "module_kegg_annotations.tsv")
    mdc_summary, mdc_resolved = mdc_tables(details)
    atomic_tsv(mdc_summary, args.output / "mdc_ad_vs_control_summary.tsv")
    atomic_tsv(mdc_resolved, args.output / "mdc_resolved_ad_vs_control.tsv")

    files = {}
    for path in sorted(args.output.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(args.output))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    manifest = {
        "status": "complete" if (args.output / "aggregate_statistics.parquet").exists() else "partial",
        "created_utc": utc_now(),
        "module_set": MODULE_SET,
        "module_definition": MODULE_LABEL,
        "modules": MODULE_COUNT,
        "assignments": ASSIGNMENT_COUNT,
        "donors": 357,
        "diagnosis_counts": {"Control": 120, "MCI": 82, "AD": 155},
        "methods": list(METHODS),
        "estimators": ["lioness"],
        "tissues": ["VH", "SMA", "QUAD"],
        "components": list(COMPONENTS),
        "partition_network": {
            "method": "M5 extended SFN",
            "cohort": PARTITION_COHORT,
            "biological_level": 4,
            "zero_indexed_table": 3,
            "beta_ts": BETAS_TS,
            "beta_ct": BETAS_CT,
        },
        "lioness_network": {
            "functional": "Pearson signedAlt with frozen block-specific powers",
            "note": "LIONESS is distinct from the M5 partition and M5 group-MDC functional.",
        },
        "mdc": mdc_manifest(),
        "capabilities": {
            "prediction": False,
            "bonobo": False,
            "differential_edges": False,
            "edge_summaries": False,
            "expanded_components": False,
            "edge_expression": False,
            "descriptive_eigengenes": (
                args.output / "module_eigengenes.parquet"
            ).exists(),
        },
        "privacy": "Public rows contain random pseudonyms only; donor and projid are excluded.",
        "files": files,
    }
    (args.output / "module_set_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.update_app_manifest:
        update_app_manifest(manifest, args.output)
    print(json.dumps({"output": str(args.output), "files": len(files), "rows": {"aggregate": len(aggregate), "resolved": len(resolved)}}, indent=2))


if __name__ == "__main__":
    main()
