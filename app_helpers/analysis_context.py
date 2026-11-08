"""Analysis identity is separate from appearance and never stores donor frames."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral, Real
from typing import Mapping

import streamlit as st

# Appearance fields deliberately do not invalidate an applied analysis.
APPEARANCE_FIELDS = frozenset(
    {
        "color_by",
        "continuous_colorscale",
        "reverse_colorscale",
        "single_point_color",
        "point_color_labels",
        "point_color_options",
        "annotation_fields",
        "show_group_trend_lines",
        "show_pooled_association",
        "pooled_line_dash",
        "trend_line_rule",
        "association_significance_cutoff",
        "association_plot_type",
        "match_effect_kegg_subtitle",
    }
)

ANALYSIS_FIELDS = frozenset(
    """active_view module_set module estimator method cohort_scope feature phenotype
association_interpretation resolution resolved scale correlation_method diagnoses selected_components
grouping_variable selected_group_levels minimum_group_n distribution_grouping_variable
selected_distribution_levels distribution_minimum_group_n heatmap_grouping_variable
selected_heatmap_group_levels heatmap_minimum_group_n edge_rule bonobo_edge_subset edge_subset_basis
differential_edge_rule differential_fdr_scope differential_fdr_threshold score_normalization analysis_subset
effect_statistic effect_cutoff_mode effect_cutoff_value effect_direction
source source_label score_variant network_method tissues selected_modules
interpretation selected_levels outcome grouping levels min_n scope mask_id metrics components
feature_scale normalization rule cohort categorical association_type selected_outcomes category_variable
minimum_category_n regional_tissue baseline_inference_version norm""".split()
)

WIDGET_STATE_FIELDS = {
    "module_set": "analysis_module_set",
    "module": "analysis_module",
    "phenotype": "analysis_phenotype",
    "feature": "analysis_feature",
    "resolution": "analysis_resolution",
    "scale": "analysis_scale",
    "correlation_method": "analysis_correlation_method",
    "method": "analysis_method",
    "cohort_scope": "analysis_cohort_scope",
    "diagnoses": "analysis_diagnoses",
    "estimator": "network_estimator",
    **{
        name: "analysis_" + name
        for name in (
            "edge_subset_basis",
            "edge_rule",
            "bonobo_edge_subset",
            "effect_statistic",
            "effect_cutoff_mode",
            "effect_cutoff_value",
            "effect_direction",
            "differential_fdr_scope",
            "differential_fdr_threshold",
            "score_normalization",
            "analysis_subset",
            "grouping_variable",
            "selected_group_levels",
            "minimum_group_n",
            "selected_components",
        )
    },
}


def prime_controls(view: str) -> None:
    transfer = (
        st.session_state.pop("_analysis_transfer", None)
        if view == "Associations"
        else None
    )
    saved = (
        transfer
        or st.session_state.get(f"analysis_context:{view}")
        or st.session_state.get("_last_exploration_context", {})
    )
    if transfer:
        st.session_state[f"analysis_context:{view}"] = dict(transfer)
        st.session_state.pop(f"analysis_labels:{view}", None)
    returning = st.session_state.get("_active_controls_view") != view
    st.session_state["_active_controls_view"] = view
    for field, key in WIDGET_STATE_FIELDS.items():
        if field in saved and (transfer or returning):
            value = saved[field]
            st.session_state[key] = (
                list(value)
                if isinstance(value, tuple)
                and field in {"diagnoses", "selected_group_levels"}
                else value
            )


def preserve_control_widgets() -> None:
    # Native pages clean up widgets that disappear. Keep only small, known
    # selection/appearance fields; never buttons, cached data, or donor frames.
    for key in (
        *WIDGET_STATE_FIELDS.values(),
        "mdc_display_scale",
        "association_plot_type",
    ):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


@dataclass(frozen=True)
class AnalysisContext:
    values: tuple[tuple[str, object], ...]

    @classmethod
    def from_controls(cls, controls: Mapping) -> "AnalysisContext":
        # Small JSON-safe controls only. No sample data, manifests, readers, or UI objects.
        values = []

        def plain(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, Integral):
                return int(value)
            if isinstance(value, Real):
                return float(value)
            return value

        for key, value in controls.items():
            if key not in ANALYSIS_FIELDS or key in APPEARANCE_FIELDS:
                continue
            value = plain(value)
            if value is None or isinstance(value, (str, bool, int, float)):
                values.append((key, value))
            elif isinstance(value, (tuple, list)) and all(
                v is None or isinstance(plain(v), (str, bool, int, float))
                for v in value
            ):
                values.append((key, tuple(plain(v) for v in value)))
            elif key == "selected_modules" and isinstance(value, dict):
                values.append((key, tuple(sorted(value.items()))))
        return cls(tuple(sorted(values)))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.values, sort_keys=True, default=str).encode()
        ).hexdigest()

    def as_dict(self) -> dict:
        return dict(self.values)


def apply_controls(controls: dict, *, view: str) -> dict:
    """Metadata controls may rerun; analysis keeps its last applied scalar settings."""
    with controls.get("analysis_panel", st.sidebar):
        with st.form(f"apply_analysis:{view}", border=False):
            submitted = st.form_submit_button(
                "Apply analysis", type="primary", width="stretch"
            )
        return resolve_controls(controls, view=view, submitted=submitted)


def resolve_controls(controls: dict, *, view: str, submitted: bool) -> dict:
    """Shared apply-state policy for source-specific presentation adapters."""
    draft = AnalysisContext.from_controls(controls)
    key = f"analysis_context:{view}"
    saved = st.session_state.get(key)
    # Never combine a pending source's metadata with the previous source's scores.
    namespace = tuple(
        controls.get(k) for k in ("module_set", "estimator", "method", "cohort_scope")
    )
    namespace_changed = (
        saved is not None
        and tuple(
            saved.get(k) for k in ("module_set", "estimator", "method", "cohort_scope")
        )
        != namespace
    )
    labels_key = f"analysis_labels:{view}"
    if saved is None:
        saved = draft.as_dict()
        st.session_state[key] = saved
        st.session_state[labels_key] = dict(controls.get("active_feature_labels", {}))
    pending = AnalysisContext.from_controls(saved).fingerprint != draft.fingerprint
    if submitted:
        saved = draft.as_dict()
        st.session_state[key] = saved
        st.session_state[labels_key] = dict(controls.get("active_feature_labels", {}))
        pending = False
    if pending:
        st.info(
            "Unapplied settings. Figures and downloads still use the applied analysis."
        )
    if namespace_changed and not submitted:
        st.info(
            "Apply the new dataset/method/cohort before loading its results. No other source will be substituted."
        )
        st.stop()
    result = dict(controls)
    result.update(saved)
    if ":" not in view:
        st.session_state["_last_exploration_context"] = dict(saved)
    result["active_feature_labels"] = st.session_state.get(
        labels_key, controls.get("active_feature_labels", {})
    )
    # Preserve the original collection types expected by existing scientific helpers.
    for name, value in controls.items():
        if isinstance(value, list) and isinstance(result.get(name), tuple):
            result[name] = list(result[name])
        elif isinstance(value, dict) and isinstance(result.get(name), tuple):
            result[name] = dict(result[name])
    return result
