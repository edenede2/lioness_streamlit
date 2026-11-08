"""Compact Streamlit entrypoint for six task-focused ROSMAP workspaces."""

from __future__ import annotations

import streamlit as st

from app_helpers.app_shell import render_navigation
from app_helpers.ui_cache import initialize_bundle
from app_helpers import ui_runtime as runtime
from app_helpers.analysis_context import preserve_control_widgets

st.set_page_config(
    page_title="ROSMAP Network Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    "<style>.block-container{padding-top:1.6rem;padding-bottom:3rem;max-width:1500px}"
    "[data-testid='stMetricValue']{font-size:1.35rem}.app-note{color:#526273;font-size:.93rem}"
    ".kegg-note{padding:.6rem .85rem;background:#f3f7fb;border-left:4px solid #2c7fb8}</style>",
    unsafe_allow_html=True,
)
runtime.refresh_helper_apis()
preserve_control_widgets()

try:
    data_manifest = initialize_bundle()
except (FileNotFoundError, RuntimeError, ValueError) as error:
    st.error(f"Could not initialize the data bundle: {error}")
    st.stop()


def render_view(view):
    if view == "Prediction":
        from app_helpers.workspaces.prediction import render_prediction_view

        render_prediction_view()
        return
    if view == "Module–outcome comparison":
        from app_helpers.score_comparison_ui import render_view as render

        render(data_manifest)
        return
    if view == "Partition comparison":
        from app_helpers.workspaces.compare import render_partition_comparison_view

        render_partition_comparison_view()
        return
    if view in {
        "Gene & edge associations",
        "Donor explorer",
        "Validation & replication",
        "Matched-control results",
    }:
        import importlib

        adapter = {
            "Gene & edge associations": "outcome_screening_ui",
            "Donor explorer": "donor_explorer_ui",
            "Validation & replication": "validation_review_ui",
            "Matched-control results": "matched_brain_results_ui",
        }[view]
        render = importlib.import_module("app_helpers." + adapter).render_view
        (
            render()
            if view == "Gene & edge associations"
            else render(runtime.render_plotly_chart)
        )
        return
    is_distribution_search = view == "Distribution differentiation"
    active_view = "Feature distributions" if is_distribution_search else view
    if active_view in runtime.REGIONAL_ANALYSIS_VIEWS or active_view in {
        "Module details",
        "MDC",
        "KEGG enrichment",
        "Statistics",
    }:
        with st.sidebar:
            from app_helpers.matched_brain_results import available as matched_available

            organizations = ["matched_multitissue"]
            if active_view in runtime.REGIONAL_ANALYSIS_VIEWS:
                organizations.append("independent_regional")
            if active_view != "Correlation heatmaps" and matched_available():
                organizations.append("matched_discovery")
            organization = st.radio(
                "Module organization",
                organizations,
                format_func=lambda v: {
                    "matched_multitissue": "Matched multi-tissue modules",
                    "independent_regional": "Independent single-region modules",
                    "matched_discovery": "Matched 70/30 Control-discovery modules (247)",
                }[v],
            )
        if organization == "independent_regional":
            runtime.render_regional_analysis_view(active_view)
            return
        if organization == "matched_discovery":
            import importlib
            from app_helpers import matched_brain_results_ui as matched_ui

            if getattr(matched_ui, "WORKSPACE_SECTION_API_VERSION", 0) < 1:
                importlib.reload(matched_ui)
            render_matched = matched_ui.render_view
            section = {
                "Associations": "Donor scatter plots",
                "Feature distributions": "Donor feature distributions",
                "Module details": "Module composition",
                "MDC": "MDC",
                "KEGG enrichment": "KEGG enrichment",
                "Statistics": "CAMEL associations",
            }[active_view]
            render_matched(runtime.render_plotly_chart, section=section)
            return
    from app_helpers.workspaces.controls import build_controls
    from app_helpers.analysis_context import apply_controls, prime_controls
    from app_helpers.workspaces.shared import (
        prepare_metadata,
        prepare_donors,
        prepare_enrichment,
    )

    prime_controls(active_view)
    context = apply_controls(
        build_controls(active_view, data_manifest), view=active_view
    )
    context["distribution_search"] = is_distribution_search
    context = prepare_metadata(context)
    needs_donors = active_view in {
        "Feature distributions",
        "Statistics",
        "Donor edge explorer",
    } or (
        active_view == "Associations" and context["association_plot_type"] == "Scatter"
    )
    if needs_donors:
        context = prepare_donors(context)
    if active_view == "Associations":
        context = prepare_enrichment(context)
    import importlib

    renderers = {
        "Associations": ("module", "render_associations"),
        "Feature distributions": ("module", "render_distributions"),
        "MDC": ("module", "render_mdc"),
        "Statistics": ("module", "render_statistics"),
        "KEGG enrichment": ("module", "render_kegg"),
        "Module details": ("module", "render_details"),
        "Correlation heatmaps": ("discover", "render_heatmaps"),
        "CT–TS screen": ("discover", "render_ct_ts"),
        "Module finder": ("discover", "render_finder"),
        "Edge summaries": ("donors", "render_summaries"),
        "Donor edge explorer": ("donors", "render_edge_explorer"),
        "Edge volcano": ("donors", "render_volcano"),
        "Methods & data": ("methods", "render_methods"),
    }
    workspace_name, renderer = renderers[active_view]
    getattr(
        importlib.import_module("app_helpers.workspaces." + workspace_name), renderer
    )(context)


render_navigation(render_view)
