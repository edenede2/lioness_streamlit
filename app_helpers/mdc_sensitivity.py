"""Definitions and compact public tables for muscle MDC normalization sensitivity."""
from __future__ import annotations

import numpy as np
import pandas as pd

VARIANTS = {
    "m5_high_no_q90": {"label": "M5: no q90 scaling (same high powers)", "family": "high_m5", "scale": "none", "clip": False},
    "original": {"label": "Current M5: high powers, separate scales, clipped", "family": "high_m5", "scale": "separate", "clip": True},
    "pearson": {"label": "Unpowered Pearson correlations", "family": "pearson", "scale": "none", "clip": False},
    "signed_alt_low": {"label": "SignedAlt: CT beta 2 / TS beta 3", "family": "signed_alt_low", "scale": "none", "clip": False},
    "m5_high_shared_unclipped": {"label": "M5 high powers: shared Control scale, unclipped", "family": "high_m5", "scale": "control", "clip": False},
    "m5_high_shared_clipped": {"label": "M5 high powers: shared Control scale, clipped", "family": "high_m5", "scale": "control", "clip": True},
    "m5_high_separate_unclipped": {"label": "M5 high powers: separate scales, unclipped", "family": "high_m5", "scale": "separate", "clip": False},
    "m5_low_separate_clipped": {"label": "M5 low powers: separate scales, clipped", "family": "low_m5", "scale": "separate", "clip": True},
    "m5_low_shared_unclipped": {"label": "M5 low powers: shared Control scale, unclipped", "family": "low_m5", "scale": "control", "clip": False},
    "m5_low_shared_clipped": {"label": "M5 low powers: shared Control scale, clipped", "family": "low_m5", "scale": "control", "clip": True},
}

COMPONENT_LABELS = {
    "TS_VH": "TS: VH", "TS_SMA": "TS: SMA", "TS_QUAD": "TS: QUAD",
    "CT_VH__SMA": "CT: VH - SMA", "CT_VH__QUAD": "CT: VH - QUAD",
    "CT_SMA__QUAD": "CT: SMA - QUAD",
}


def public_mdc_tables(frame: pd.DataFrame, details: pd.DataFrame):
    """Preserve actual p/FDR values and expose inference availability explicitly."""
    if frame.duplicated(["module", "scope"]).any():
        raise ValueError("Duplicate module-scope rows in MDC sensitivity")
    data = frame.copy()
    data["mdc_stable_for_inference"] = (
        data.n_edges.ge(10) & data.mean_abs_ad.gt(1e-8)
        & data.mean_abs_control.gt(1e-8) & np.isfinite(data.mdc_raw)
    )
    data["mdc_stability_reason"] = ""
    data.loc[data.n_edges.lt(10), "mdc_stability_reason"] = "fewer than 10 edges"
    data.loc[data.mean_abs_ad.le(1e-8) | data.mean_abs_control.le(1e-8), "mdc_stability_reason"] = "near-zero AD or Control mean absolute weight"
    data.loc[data.mdc_raw.isna(), "mdc_stability_reason"] = "structurally unavailable or zero denominator"
    for column in ["directional_fdr", "directional_p_sample", "directional_p_gene"]:
        if column not in data:
            data[column] = np.nan
    if "inference_available" not in data:
        data["inference_available"] = False
        data["inference_unavailable_reason"] = "Permutation inference is pending for this functional"
    data["direction"] = np.select(
        [data.mdc_raw.gt(1), data.mdc_raw.lt(1), data.mdc_raw.eq(1)],
        ["Higher in AD", "Higher in Control", "Equal"], default="Not available",
    )
    data["significant_fdr_05"] = data.directional_fdr.lt(.05)
    data["significant_fdr_10"] = data.directional_fdr.lt(.10)
    size_lookup = details.set_index("module")["module_size"]
    summary = pd.DataFrame({"module": sorted(data.module.unique())})
    summary["module_size_mapped"] = summary.module.map(size_lookup)
    mappings = {"mdc_raw": "mdc", "mdc_log2": "log2_mdc", "n_edges": "n"}
    for scope, suffix in [("total", "total"), ("TS", "ts"), ("CT", "ct")]:
        selected = data[data.scope.eq(scope)].set_index("module")
        for column in ["mdc_raw", "mdc_log2", "n_edges", "mean_abs_ad", "mean_abs_control",
                       "direction", "directional_fdr", "directional_p_sample", "directional_p_gene",
                       "inference_available", "inference_unavailable_reason", "mdc_stable_for_inference",
                       "mdc_stability_reason"]:
            name = f"{mappings.get(column,column)}_{suffix}"
            if column == "n_edges":
                name = f"n_{suffix}_edges"
            summary[name] = summary.module.map(selected[column])
        for alpha, label in [(.05, "05"), (.10, "10")]:
            summary[f"significant_{suffix}_fdr{label}"] = summary[f"directional_fdr_{suffix}"].lt(alpha)
    summary["ts_minus_ct_log2_mdc"] = summary.log2_mdc_ts - summary.log2_mdc_ct
    for label in ["05", "10"]:
        summary[f"any_significant_fdr{label}"] = summary[[f"significant_{s}_fdr{label}" for s in ["total","ts","ct"]]].any(axis=1)
    resolved = data[data.scope.isin(COMPONENT_LABELS)].rename(
        columns={"scope": "component", "mdc_raw": "mdc", "mdc_log2": "log2_mdc"}
    ).copy()
    resolved["module_definition"] = "muscle_control_l4"
    resolved["component_class"] = resolved.component.str.split("_").str[0]
    resolved["component_label"] = resolved.component.map(COMPONENT_LABELS)
    resolved["n_features"] = resolved.module.map(size_lookup)
    return summary, resolved
