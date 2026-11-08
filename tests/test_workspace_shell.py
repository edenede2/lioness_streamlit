from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app_helpers.analysis_context import AnalysisContext
from app_helpers.app_shell import VIEW_ROUTES, WORKSPACES, route_for
from app_helpers import ui_cache

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"
LEGACY_VIEWS = (
    "Associations",
    "Feature distributions",
    "Correlation heatmaps",
    "Module–outcome comparison",
    "Gene & edge associations",
    "Prediction",
    "Module finder",
    "CT–TS screen",
    "Edge summaries",
    "Edge volcano",
    "Donor edge explorer",
    "MDC",
    "Statistics",
    "KEGG enrichment",
    "Module details",
    "Partition comparison",
    "Validation & replication",
    "Matched-control results",
    "Donor explorer",
    "Methods & data",
)


def app_for(view="Associations", **params):
    app = AppTest.from_file(APP, default_timeout=180)
    app.query_params.update(view=view, **params)
    return app.run()


def apply(app):
    return (
        next(button for button in app.button if button.label == "Apply analysis")
        .click()
        .run()
    )


def widget(app, kind, label):
    return next(value for value in getattr(app, kind) if value.label == label)


def test_all_twenty_old_views_have_a_reachable_destination():
    assert len(WORKSPACES) == 6
    assert set(LEGACY_VIEWS) <= set(VIEW_ROUTES)
    assert route_for("unknown") == "discover"
    assert len(VIEW_ROUTES) == len(set(VIEW_ROUTES))


def test_context_excludes_donor_frames_and_appearance():
    controls = dict(
        module_set="control_derived",
        module=121,
        method="control_anchored",
        diagnoses=["Control", "AD"],
        color_by="age",
        show_group_trend_lines=True,
        frame=pd.DataFrame({"sample_id": ["private-test"]}),
    )
    first = AnalysisContext.from_controls(controls)
    controls.update(color_by="clusters", show_group_trend_lines=False)
    assert first == AnalysisContext.from_controls(controls)
    assert "frame" not in first.as_dict()
    controls["module"] = 935
    assert first != AnalysisContext.from_controls(controls)


def test_comparison_can_seed_applied_score_without_overwriting_existing_boxes(
    monkeypatch,
):
    from app_helpers import score_comparison_ui as comparison

    registry = {"control_derived": {}}
    monkeypatch.setattr(comparison, "availability_reason", lambda spec, registry: None)
    existing = [{"uid": "existing", "source": "full_cohort"}]
    state = {comparison.BOX_STATE: existing}
    monkeypatch.setattr(comparison.st, "session_state", state)
    context = dict(
        module_set="control_derived",
        module=121,
        method="control_anchored",
        feature="connectivity",
        scale="asinh",
        selected_components=["CT_AC__DLPFC"],
        differential_edge_rule="all",
        diagnoses=["AD"],
    )
    comparison.add_applied_score(context, registry)
    assert state[comparison.BOX_STATE][0] == existing[0]
    assert len(existing) == 1
    added = state[comparison.BOX_STATE][1]
    assert (added["source"], added["component"], added["scale"]) == (
        "control_derived",
        "CT_AC__DLPFC",
        "asinh",
    )
    assert "module" not in added and "diagnoses" not in added
    assert comparison.applied_score_box({"module_set": "unknown"}, registry) is None
    assert comparison.applied_score_box(dict(context, estimator="bonobo"), registry) is None


def test_association_header_keeps_publication_legend_below_title():
    import plotly.graph_objects as go
    from app_helpers.workspaces.shared import association_header
    from app_helpers.figure_export import (
        PlotExportSpec,
        PlotContentConfig,
        preset_config,
    )
    from app_helpers.figure_export.adapters import build_artifact, fit_automatic_config

    figure = association_header(go.Figure(go.Scatter(x=[1, 2], y=[2, 1], name="AD")))
    figure.update_layout(
        title_text="Module M935: cognition vs connectivity", legend_orientation="h"
    )
    spec = PlotExportSpec.from_figure("association_header_contract", figure)
    content = PlotContentConfig()
    config = fit_automatic_config(
        spec, preset_config("Manuscript – single column", spec.aspect_ratio), content
    )
    exported = build_artifact(spec, config, content).figure
    assert exported.layout.legend.yref == "container"
    assert exported.layout.legend.y < exported.layout.title.y


def test_dataset_cache_versions_are_independent():
    original = dict(ui_cache._versions), ui_cache._manifest_stamp
    try:
        manifest = {"module_sets": {"a": {"hash": "a1"}, "b": {"hash": "b1"}}}
        ui_cache.register_manifest(manifest, ("first",))
        a, b = ui_cache.version_for("exploration", "a"), ui_cache.version_for(
            "exploration", "b"
        )
        manifest["module_sets"]["b"]["hash"] = "b2"
        ui_cache.register_manifest(manifest, ("second",))
        assert ui_cache.version_for("exploration", "a") == a
        assert ui_cache.version_for("exploration", "b") != b
        with patch.object(
            ui_cache, "ensure_data_path", side_effect=AssertionError("prediction read")
        ):
            assert ui_cache.version_for("exploration", "a") == a
    finally:
        ui_cache._versions.clear()
        ui_cache._versions.update(original[0])
        ui_cache._manifest_stamp = original[1]


@pytest.mark.parametrize("view", LEGACY_VIEWS)
def test_legacy_deep_links_render(view):
    app = app_for(view)
    assert not app.exception, [(x.type, x.message) for x in app.exception]


def test_default_is_discover_with_compact_tool_navigation():
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert not app.exception
    selector = widget(app, "pills", "Analysis view")
    assert selector.value == "Module finder"
    assert len(selector.options) == 4


@pytest.mark.parametrize("view", ["Module details", "KEGG enrichment", "MDC"])
def test_metadata_views_never_request_donor_scores(view):
    from app_helpers.workspaces import shared

    with patch.object(
        shared, "cached_aggregate", side_effect=AssertionError("unrelated donor read")
    ), patch.object(
        shared, "cached_resolved", side_effect=AssertionError("unrelated donor read")
    ):
        app = app_for(view)
        assert not app.exception


def test_unapplied_module_does_not_change_figure_or_download_identity():
    app = app_for("Associations")
    assert not app.exception
    before = json.loads(app.get("plotly_chart")[0].proto.spec)
    module = widget(app, "selectbox", "Module")
    replacement = next(value for value in module.options if value != module.value)
    # Options are formatted labels in AppTest; select_index preserves the integer value.
    app = module.select_index(0 if module.index != 0 else 1).run()
    assert not app.exception
    after = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert before["data"] == after["data"]
    assert any("Unapplied settings" in item.value for item in app.info)
    app = apply(app)
    assert not app.exception
    assert json.loads(app.get("plotly_chart")[0].proto.spec)["data"] != before["data"]


def test_source_change_requires_apply_and_never_reuses_the_old_source():
    app = app_for("Associations")
    app = (
        widget(app, "selectbox", "Module definition").set_value("control_derived").run()
    )
    assert not app.exception
    assert not app.get("plotly_chart")
    assert any("Apply the new dataset" in item.value for item in app.info)
    app = apply(app)
    assert not app.exception
    assert app.get("plotly_chart")


def test_appearance_change_does_not_query_scores_or_statistics(monkeypatch):
    app = app_for("Associations")
    assert not app.exception
    from app_helpers.workspaces import module, shared

    def forbidden(*args, **kwargs):
        raise AssertionError("appearance caused an analysis read")

    # Already-cached active results are permitted; uncached scientific readers are not.
    from app_helpers import ui_runtime

    monkeypatch.setattr(ui_runtime, "load_aggregate", forbidden)
    monkeypatch.setattr(ui_runtime, "load_resolved", forbidden)
    app = (
        widget(app, "checkbox", "Show group-specific OLS trend lines")
        .set_value(False)
        .run()
    )
    assert not app.exception
    assert app.get("plotly_chart")


def test_versioned_reader_matches_direct_known_module():
    from app_helpers import data, ui_runtime

    actual = ui_runtime.cached_resolved(
        "control_derived", "lioness", "control_anchored", 121, "negative_density", "all"
    )
    expected = data.load_resolved(
        "control_anchored", 121, "negative_density", module_set="control_derived"
    )
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_unavailable_dataset_link_never_falls_back():
    app = app_for("Associations", screen_partition="unavailable_partition")
    assert not app.exception and not app.get("plotly_chart")
    assert any(
        "No other dataset will be substituted" in error.value for error in app.error
    )


def test_discover_to_module_preserves_context_and_back_link():
    app = app_for("Module finder")
    assert not app.exception
    chosen = widget(app, "selectbox", "Inspect ranked module").value
    app = (
        next(
            button
            for button in app.button
            if button.label == "Open module associations"
        )
        .click()
        .run()
    )
    assert not app.exception
    assert widget(app, "selectbox", "Module").value == chosen
    assert widget(app, "selectbox", "Group correlations by").value == "__all__"
    assert app.get("plotly_chart")
    app = (
        next(button for button in app.button if button.label == "← Back to discovery")
        .click()
        .run()
    )
    assert not app.exception
    assert widget(app, "pills", "Analysis view").value == "Module finder"


def test_regional_catalog_cache_holds_all_three_tissues(monkeypatch):
    from app_helpers import regional_modules as regional

    source = "single_region_complete_case_l3"
    regional.regional_correlation_catalog.clear()
    args = (source, "common_450", "eigengene", "not_applicable")
    # Use the manifest's actual stored eigengene-method token.
    manifest = regional.completed_combinations(regional.load_regional_manifest())
    row = manifest.loc[
        (manifest.partition_source == source)
        & (manifest.cohort_scope == "common_450")
        & (manifest.score_variant == "eigengene")
    ].iloc[0]

    def catalog(tissue):
        return regional.regional_correlation_catalog(
            source,
            "common_450",
            "eigengene",
            row.network_method,
            tissue,
            "raw",
            ("Control", "MCI", "AD"),
            "__all__",
            ("__all__",),
            10,
            ("cogn_global",),
            True,
        )

    before = {tissue: catalog(tissue) for tissue in ("AC", "DLPFC", "PCG")}

    def forbidden(*args, **kwargs):
        raise AssertionError("three-tissue cache evicted an active catalog")

    monkeypatch.setattr(regional, "load_regional_scores", forbidden)
    for tissue, expected in before.items():
        pd.testing.assert_frame_equal(catalog(tissue), expected, check_exact=True)
