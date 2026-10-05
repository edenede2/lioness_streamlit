"""Manifest-driven Streamlit view for Experiment 0210 / Protocol v1.1."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app_helpers.drive_data import DATA_DIR, data_path_available, ensure_data_path
from app_helpers.streamlit_compat import (
    normalize_width_kwargs,
    plotly_chart,
    stateful_tabs,
    tab_is_open,
)
from app_helpers.table_controls import filterable_dataframe


PROTOCOL_DATA_DIR = DATA_DIR / "protocol_v1_1"
PROTOCOL_MANIFEST = PROTOCOL_DATA_DIR / "manifest.json"

RESULT_FILES = {
    "performance": "performance.parquet",
    "comparisons": "descriptive_comparisons.parquet",
    "coefficients": "coefficient_summary.parquet",
    "shap": "shap_summary.parquet",
    "permutation": "permutation_importance.parquet",
    "prediction_completion": "prediction_completion.tsv",
    "linear_model_completion": "linear_model_completion.tsv",
    "explanation_completion": "explanation_completion.tsv",
    "qa": "qa_gates.tsv",
    "resources": "resource_measurements.tsv",
    "null_progress": "null_progress.tsv",
}


def protocol_v11_available() -> bool:
    return data_path_available(PROTOCOL_MANIFEST)


@st.cache_data(show_spinner=False)
def load_protocol_manifest() -> dict[str, object]:
    if not data_path_available(PROTOCOL_MANIFEST):
        return {}
    path = ensure_data_path(PROTOCOL_MANIFEST)
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_protocol_table(name: str) -> pd.DataFrame:
    if name in RESULT_FILES:
        relative = RESULT_FILES[name]
    else:
        relative = name
    requested = PROTOCOL_DATA_DIR / relative
    if not data_path_available(requested):
        return pd.DataFrame()
    path = ensure_data_path(requested)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t")


def _gate_frame(manifest: dict[str, object]) -> pd.DataFrame:
    gates = manifest.get("gates", {})
    return pd.DataFrame(
        [
            {
                "gate": key,
                "status": "passed" if bool(value) else "pending",
            }
            for key, value in gates.items()
        ]
    )


def _render_status(manifest: dict[str, object]) -> None:
    verified = list(manifest.get("verified_outcomes", []))
    blocked = list(manifest.get("blocked_outcomes", []))
    authorizations = manifest.get("execution_authorizations", {})
    columns = st.columns(4)
    columns[0].metric("Complete-expression donors", manifest.get("complete_expression_n", "—"))
    columns[1].metric("Verified outcomes", len(verified))
    columns[2].metric("Blocked definitions", len(blocked))
    columns[3].metric(
        "Catalog",
        "Interim" if bool(manifest.get("interim", True)) else "Complete",
    )
    st.caption(
        "Level B and Level C have independent release gates. A pending Level-C null "
        "does not prevent release of Level-B results whose own gates have passed."
    )
    authorization_frame = pd.DataFrame(
        [
            {
                "execution lane": key,
                "authorized": bool(value),
            }
            for key, value in authorizations.items()
        ]
    )
    if not authorization_frame.empty:
        st.dataframe(
            authorization_frame,
            hide_index=True,
            **normalize_width_kwargs({"width": "stretch"}),
        )
    gates = _gate_frame(manifest)
    if not gates.empty:
        filterable_dataframe(
            gates,
            table_key="protocol_v11_gates",
            table_name="Protocol v1.1 acceptance gates",
            hide_index=True,
        )


def _render_outcomes() -> None:
    outcomes = load_protocol_table("outcome_dictionary.tsv")
    exclusions = load_protocol_table("outcome_exclusions.tsv")
    splits = load_protocol_table("split_registry_summary.tsv")
    if outcomes.empty:
        st.info("The source-audit outcome dictionary is not packaged yet.")
        return
    status_order = {"verified": 0, "blocked_definition": 1, "unavailable": 2}
    outcomes["_status_order"] = outcomes["audit_status"].map(status_order).fillna(9)
    outcomes = outcomes.sort_values(["_status_order", "evidence_tier", "key"]).drop(
        columns="_status_order"
    )
    filterable_dataframe(
        outcomes,
        table_key="protocol_v11_outcomes",
        table_name="Audited Protocol v1.1 outcomes",
        hide_index=True,
        height=520,
    )
    if not exclusions.empty:
        counts = exclusions.loc[
            exclusions["reason"].isin(
                ["eligible_observed_target", "missing_or_excluded_target"]
            )
        ].copy()
        if not counts.empty:
            figure = px.bar(
                counts,
                x="outcome",
                y="n",
                color="reason",
                barmode="stack",
                labels={"outcome": "Outcome", "n": "Donors", "reason": "Eligibility"},
                title="Outcome-specific eligibility and missingness",
            )
            figure.update_layout(legend={"orientation": "h", "y": 1.12})
            plotly_chart(figure, key="protocol_v11_outcome_eligibility")
    if not splits.empty:
        filterable_dataframe(
            splits,
            table_key="protocol_v11_split_summary",
            table_name="Outcome-specific split-registry status",
            hide_index=True,
        )


def _render_results(manifest: dict[str, object]) -> None:
    performance = load_protocol_table("performance")
    if performance.empty:
        st.info(
            "No Protocol v1.1 prediction estimates are public yet. The app exposes the "
            "audited outcomes and gates now; validated Level-B targets will appear here "
            "incrementally without waiting for Level C."
        )
        return
    selectors = st.columns(4)
    selected = performance.copy()
    for column, label, container in (
        ("outcome", "Outcome", selectors[0]),
        ("validation_level", "Validation level", selectors[1]),
        ("representation", "Representation", selectors[2]),
        ("learner", "Learner", selectors[3]),
    ):
        if column not in selected:
            continue
        values = selected[column].dropna().astype(str).drop_duplicates().tolist()
        if not values:
            continue
        choice = container.selectbox(
            label,
            values,
            key=f"protocol_v11_{column}",
        )
        selected = selected.loc[selected[column].astype(str).eq(choice)]
    if {"metric", "value", "representation"}.issubset(selected.columns):
        figure = px.bar(
            selected,
            x="representation",
            y="value",
            color="model_variant" if "model_variant" in selected else None,
            facet_col="metric",
            title="Protocol v1.1 validated performance",
        )
        plotly_chart(figure, key="protocol_v11_performance")
    filterable_dataframe(
        selected,
        table_key="protocol_v11_performance_table",
        table_name="Protocol v1.1 performance",
        hide_index=True,
    )
    comparisons = load_protocol_table("comparisons")
    if not comparisons.empty:
        st.markdown("#### Paired descriptive comparisons")
        st.caption(
            "Intervals are descriptive. Algorithm-comparison p-values and adjusted "
            "p-values remain unavailable until a separate inferential procedure is approved."
        )
        filterable_dataframe(
            comparisons,
            table_key="protocol_v11_comparisons",
            table_name="Protocol v1.1 descriptive model comparisons",
            hide_index=True,
        )


def _render_importance() -> None:
    coefficients = load_protocol_table("coefficients")
    shap = load_protocol_table("shap")
    permutation = load_protocol_table("permutation")
    if coefficients.empty and shap.empty and permutation.empty:
        st.info(
            "Feature-explanation completion is tracked separately from prediction "
            "completion. No validated explanation artifact is public yet."
        )
        return
    for title, frame, key in (
        ("Linear coefficient stability", coefficients, "coefficients"),
        ("Outer-test SHAP summaries", shap, "shap"),
        ("Held-out permutation importance", permutation, "permutation"),
    ):
        if frame.empty:
            continue
        st.markdown(f"#### {title}")
        filterable_dataframe(
            frame,
            table_key=f"protocol_v11_{key}",
            table_name=title,
            hide_index=True,
            height=480,
        )


def render_protocol_v11_view() -> None:
    manifest = load_protocol_manifest()
    st.subheader("Experiment 0210 — Protocol v1.1")
    st.warning(
        "This is a new versioned experiment. Existing prediction catalogs are unchanged. "
        "Only completed, validated target/representation combinations are selectable."
    )
    if not manifest:
        st.info(
            "The Protocol v1.1 app interface is installed, but its public audit manifest "
            "has not been built yet."
        )
        return
    status_tab, outcome_tab, result_tab, importance_tab, methods_tab = stateful_tabs(
        ["Status & gates", "Outcomes", "Results", "Feature importance", "Methods"],
        key="protocol_v11_tabs",
    )
    with status_tab:
        if tab_is_open(status_tab):
            _render_status(manifest)
            qa = load_protocol_table("qa")
            resources = load_protocol_table("resources")
            null_progress = load_protocol_table("null_progress")
            if not qa.empty:
                filterable_dataframe(
                    qa,
                    table_key="protocol_v11_qa",
                    table_name="Formula, null, invariance, and leakage QA",
                    hide_index=True,
                )
            if not resources.empty:
                filterable_dataframe(
                    resources,
                    table_key="protocol_v11_resources",
                    table_name="Measured and projected resources, including null suites",
                    hide_index=True,
                )
            if not null_progress.empty:
                st.caption(
                    "Only the controlled role-artifact experiment has a chance-level gate. "
                    "Study×sex permutations are a descriptive conditional null."
                )
                filterable_dataframe(
                    null_progress,
                    table_key="protocol_v11_null_progress",
                    table_name="Level-B null-experiment progress",
                    hide_index=True,
                )
    with outcome_tab:
        if tab_is_open(outcome_tab):
            _render_outcomes()
    with result_tab:
        if tab_is_open(result_tab):
            _render_results(manifest)
    with importance_tab:
        if tab_is_open(importance_tab):
            _render_importance()
    with methods_tab:
        if tab_is_open(methods_tab):
            st.markdown(
                "The controlled role-artifact null permutes diagnosis across the eligible "
                "AD/Control cohort and has chance-level AUC gates. The separate study×sex "
                "conditional permutation preserves stratum-specific prevalence; its pooled "
                "AUC is summarized empirically and is not required to lie in 0.48–0.52."
            )
            st.markdown(
                "Changing inner-validation labels cannot alter inner-training references, "
                "features, preprocessing, or fixed candidate models, but may change validation "
                "scores and hyperparameter selection. Outer-test outcomes never influence "
                "fitting or selection."
            )
            st.markdown(
                "If the role-artifact gate fails, every donor uses the same deterministic "
                "label-independent reference-fold assignment. Equal-size Control references "
                "are built only from the applicable training partition, and a training "
                "Control's assigned fold is excluded from its own reference at inner and "
                "outer levels."
            )
            st.json(manifest, expanded=False)
