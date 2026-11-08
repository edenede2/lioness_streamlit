"""Six task workspaces and explicit aliases for the existing twenty analysis links."""

from __future__ import annotations

from dataclasses import dataclass
import streamlit as st


@dataclass(frozen=True)
class Workspace:
    key: str
    label: str
    tools: tuple[str, ...]


WORKSPACES = (
    Workspace(
        "discover",
        "Discover",
        (
            "Module finder",
            "Correlation heatmaps",
            "CT–TS screen",
            "Distribution differentiation",
        ),
    ),
    Workspace(
        "module",
        "Module",
        (
            "Associations",
            "Feature distributions",
            "MDC",
            "KEGG enrichment",
            "Module details",
            "Statistics",
        ),
    ),
    Workspace(
        "compare",
        "Compare",
        (
            "Module–outcome comparison",
            "Partition comparison",
            "Validation & replication",
            "Matched-control results",
        ),
    ),
    Workspace("prediction", "Prediction", ("Prediction",)),
    Workspace(
        "donors",
        "Donors & edges",
        (
            "Donor explorer",
            "Gene & edge associations",
            "Edge summaries",
            "Edge volcano",
            "Donor edge explorer",
        ),
    ),
    Workspace("methods", "Data & methods", ("Methods & data",)),
)
VIEW_ROUTES = {
    view: workspace.key for workspace in WORKSPACES for view in workspace.tools
}


def route_for(view: str) -> str:
    return VIEW_ROUTES.get(view, "discover")


def open_module(module: int, context: dict, *, diagnosis="All donors") -> None:
    """Move a compact applied analysis, never a donor frame, into Module."""
    from app_helpers.analysis_context import AnalysisContext

    transfer = AnalysisContext.from_controls(context).as_dict()
    transfer.update(
        active_view="Associations",
        module=int(module),
        grouping_variable="__all__" if diagnosis == "All donors" else "diagnosis_group",
        selected_group_levels=(
            ("__all__",) if diagnosis == "All donors" else (diagnosis,)
        ),
    )
    if diagnosis != "All donors":
        transfer["diagnoses"] = (diagnosis,)
    # Stored screening Pearson effects use RINT, not an arbitrary displayed scale.
    if context.get("correlation_method") == "Pearson":
        transfer["scale"] = "rint"
    st.session_state["_analysis_transfer"] = transfer
    st.session_state["_discover_return_view"] = context["active_view"]
    st.query_params.update(
        view="Associations",
        screen_partition=context["module_set"],
        screen_method=context["method"],
        screen_module=str(module),
    )
    st.rerun()


def render_navigation(render_view) -> None:
    requested = st.query_params.get("view")
    changed_link = requested in VIEW_ROUTES and requested != st.session_state.get(
        "_last_legacy_view"
    )
    if changed_link:
        st.session_state["_last_legacy_view"] = requested
        st.session_state[f"workspace_tool:{route_for(requested)}"] = requested

    def page_renderer(workspace):
        def run():
            st.title("ROSMAP Network Explorer")
            if workspace.key == "module" and st.session_state.get(
                "_discover_return_view"
            ):
                if st.button("← Back to discovery", key="back_to_discover"):
                    st.query_params["view"] = st.session_state["_discover_return_view"]
                    st.rerun()
            if len(workspace.tools) > 1:
                key = f"workspace_tool:{workspace.key}"
                tool = (
                    st.pills(
                        "Analysis view",
                        workspace.tools,
                        key=key,
                        default=None if key in st.session_state else workspace.tools[0],
                        selection_mode="single",
                    )
                    or workspace.tools[0]
                )
            else:
                tool = workspace.tools[0]
            render_view(tool)

        return run

    initial = route_for(requested) if requested in VIEW_ROUTES else "discover"
    pages = {
        workspace.key: st.Page(
            page_renderer(workspace),
            title=workspace.label,
            url_path=workspace.key,
            default=workspace.key == initial,
        )
        for workspace in WORKSPACES
    }
    selected = st.navigation(list(pages.values()), position="sidebar")
    if changed_link and selected is not pages[route_for(requested)]:
        st.switch_page(pages[route_for(requested)])
    selected.run()
