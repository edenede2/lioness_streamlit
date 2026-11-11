"""Integration tests against the generated local catalog; no prediction training."""

import json
from pathlib import Path
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from app_helpers.control_calibrated import registry as CC, adapter as A
from ui_interactions import apply_driver

APP = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not CC.ready(), reason="Run the full offline calibrated catalog first"
)


def app_for(family="full_cohort", view="Associations", beta=3, module=None):
    app = apply_driver(AppTest.from_file(APP / "streamlit_app.py", default_timeout=180))
    app.query_params.update(
        view=view, screen_partition=family, score_source="calibrated_beta" + str(beta)
    )
    if module is not None:
        app.query_params["screen_module"] = str(module)
    app = app.run()
    if view == "Gene & edge associations":
        app = (
            app.radio(key="screening_calibrated_source")
            .set_value("calibrated_beta" + str(beta))
            .run()
        )
        app = (
            app.selectbox(key="screening_calibrated_partition").set_value(family).run()
        )
    return app


@pytest.mark.parametrize(
    "family",
    ["full_cohort", "control_derived", "muscle_control_l4", "muscle_all_donors_l4"],
)
def test_complete_application_flow_per_family(family):
    app = app_for(family)
    assert not app.exception
    assert app.multiselect(key="cc_metrics").value == ["gain", "loss"]
    assert app.selectbox(key="cc_scale").value == "raw"
    plots = app.get("plotly_chart")
    assert plots
    meta = json.loads(plots[0].proto.spec)["layout"]["meta"]
    assert (
        meta["family"] == family and meta["beta"] == 3 and meta["transform"] == "native"
    )
    assert app.get("download_button")
    module = app.selectbox(key="cc_module").value
    frame = A.scores(module_set=family, module=module)
    assert frame.role.eq("target").all()
    assert frame.sample_id.nunique() == (267 if "muscle" in family else 327)


def test_incomplete_scope_and_beta_change_do_not_reuse_results():
    app = app_for("full_cohort")
    app = app.radio(key="cc_scope").set_value("incomplete_region_validation").run()
    app = app.multiselect(key="cc_components").set_value(["TS_AC"]).run()
    assert not app.exception
    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert spec["layout"]["meta"]["cohort_scope"] == "incomplete_region_validation"
    module = app.selectbox(key="cc_module").value
    first = A.scores(
        module_set="full_cohort",
        module=module,
        component="TS_AC",
        metric_family="gain",
        cohort_scope="incomplete_region_validation",
    )
    assert first.metric_raw.notna().any()
    assert first.unavailable_reason.ne("ok").any()
    network = next(w for w in app.radio if w.label == "Network estimator")
    app = network.set_value("calibrated_beta1").run()
    assert not app.exception
    assert (
        json.loads(app.get("plotly_chart")[0].proto.spec)["layout"]["meta"]["beta"] == 1
    )


@pytest.mark.parametrize(
    "view",
    [
        "Feature distributions",
        "Correlation heatmaps",
        "Statistics",
        "Module finder",
        "CT–TS screen",
        "Edge summaries",
        "Donor edge explorer",
        "Methods & data",
        "Module details",
        "KEGG enrichment",
    ],
)
def test_existing_analysis_views_accept_calibrated_source(view):
    app = app_for(view=view)
    assert not app.exception
    if view in {"Module details", "KEGG enrichment"}:
        assert any("shared historical" in item.value for item in app.caption)
        assert app.get("plotly_chart") or app.dataframe
        return
    assert any(
        "Control-calibrated connectivity" in item.value for item in app.subheader
    )


@pytest.mark.parametrize("family,beta", [("full_cohort", 3), ("muscle_control_l4", 1)])
def test_heatmap_all_modules_preserves_baseline_after_outcome_filters(family, beta):
    app = app_for(
        family, view="Correlation heatmaps", beta=beta, module=smallest_module(family)
    )
    assert not app.exception
    app = app.radio(key="cc_heatmap_scope").set_value("All modules").run()
    assert not app.exception
    shown = next(
        d.value
        for d in app.dataframe
        if {"q_scan", "outcome", "module_key"} <= set(d.value.columns)
    )
    jobs = [j for j in A.catalog()["modules"] if j["family"] == family]
    assert set(shown.module_key) == {j["module_key"] for j in jobs}
    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert spec["layout"]["meta"]["module"] is None
    assert spec["layout"]["meta"]["heatmap_scope"] == "all_modules"
    # The complete module grid retains undefined physical components as NA.
    assert len(spec["data"][0]["y"]) == len(jobs) * 4
    app = app.multiselect(key="cc_heatmap_outcomes").set_value(["cogn_global"]).run()
    assert not app.exception
    filtered = next(
        d.value
        for d in app.dataframe
        if {"q_scan", "outcome", "module_key"} <= set(d.value.columns)
    )
    assert filtered.outcome.eq("cogn_global").all()
    expected = shown.loc[shown.outcome.eq("cogn_global")].reset_index(drop=True)
    import pandas as pd

    pd.testing.assert_frame_equal(
        filtered.reset_index(drop=True), expected, check_exact=True
    )
    # A new transform must never reuse native baseline statistics.
    app = app.selectbox(key="cc_scale").set_value("asinh").run()
    assert not app.exception and not app.get("plotly_chart")
    assert app.button(key="cc_run_heatmap_search")


def test_regional_all_module_heatmap_keeps_selected_tissue_namespace():
    family = "single_region_complete_case_l3"
    app = app_for(view="Correlation heatmaps")
    organization = next(w for w in app.radio if w.label == "Module organization")
    app = organization.set_value("independent_regional").run()
    app = app.selectbox(key="regional_partition_source").set_value(family).run()
    app = (
        app.radio(key="regional_calibrated_source").set_value("calibrated_beta3").run()
    )
    app = app.radio(key="cc_heatmap_scope").set_value("All modules").run()
    assert not app.exception
    tissue = app.selectbox(key="cc_regional_tissue").value
    expected = {
        j["module_key"]
        for j in A.catalog()["modules"]
        if j["family"] == family and j["regional_tissue"] == tissue
    }
    shown = next(
        d.value
        for d in app.dataframe
        if {"q_scan", "outcome", "module_key"} <= set(d.value.columns)
    )
    assert set(shown.module_key) == expected
    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert spec["layout"]["meta"]["displayed_modules"] == len(expected)
    assert spec["layout"]["meta"]["regional_tissue"] == tissue


@pytest.mark.parametrize(
    "family",
    [
        "single_region_complete_case_l3",
        "single_region_full_tissue_l3",
        "single_motor_tissue_complete_case_l3",
    ],
)
def test_regional_source_uses_explicit_tissue_namespace(family):
    app = app_for()
    organization = next(w for w in app.radio if w.label == "Module organization")
    app = organization.set_value("independent_regional").run()
    app = app.selectbox(key="regional_partition_source").set_value(family).run()
    app = (
        app.radio(key="regional_calibrated_source").set_value("calibrated_beta3").run()
    )
    assert not app.exception
    assert app.selectbox(key="cc_regional_tissue").value
    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert spec["layout"]["meta"]["regional_tissue"]
    assert spec["layout"]["meta"]["family"] == family


def test_transform_and_descriptive_model_change_hide_saved_result():
    app = app_for()
    app = app.button(key="cc_calculate_analysis").click().run()
    assert not app.exception
    assert "cc_current_analysis" in app.session_state
    app = app.selectbox(key="cc_scale").set_value("asinh").run()
    assert not app.exception
    assert (
        json.loads(app.get("plotly_chart")[0].proto.spec)["layout"]["meta"]["transform"]
        == "asinh"
    )
    assert not any(
        "q_search" in d.value for d in app.dataframe if hasattr(d.value, "columns")
    )


def smallest_module(family="full_cohort"):
    jobs = [j for j in A.catalog()["modules"] if j["family"] == family]
    return min(jobs, key=lambda j: j["physical_nodes"])["module"]


@pytest.mark.parametrize("beta", [1, 3])
def test_calibrated_scatter_uses_separate_measures_and_historical_guides(beta):
    from app_helpers.charts import DIAGNOSIS_COLORS
    app = app_for("control_derived", beta=beta, module=747)
    assert not app.exception
    specs = [json.loads(plot.proto.spec) for plot in app.get("plotly_chart")]
    assert [spec["layout"]["meta"]["metric_family"] for spec in specs] == ["gain", "loss"]
    for spec in specs:
        assert spec["layout"]["meta"]["components"] == ["CT", "TS"]
        assert "Global cognition" in spec["layout"]["yaxis"]["title"]["text"]
        points = [t for t in spec["data"] if t.get("mode") == "markers"]
        assert points
        assert all(t["marker"]["color"] == DIAGNOSIS_COLORS[t["name"]] for t in points)
        assert any(t.get("mode") == "lines" and t["line"]["color"] == "#000000" for t in spec["data"])
    app = app.checkbox(key="cc_plot_group_lines").uncheck().run()
    app = app.radio(key="cc_plot_pooled_dash").set_value("solid").run()
    assert not app.exception
    for plot in app.get("plotly_chart"):
        spec = json.loads(plot.proto.spec)
        lines = [t for t in spec["data"] if t.get("mode") == "lines"]
        assert len(lines) == 2
        assert all(t["line"]["color"] == "#000000" and t["line"]["dash"] == "solid" for t in lines)
        assert any(t.get("mode") == "markers" for t in spec["data"])


def test_calibrated_distribution_views_keep_measures_separate():
    app = app_for("control_derived", view="Feature distributions", module=747)
    for view in ("Histogram", "Violin", "Raincloud", "ECDF"):
        app = app.radio(key="cc_distribution_view").set_value(view).run()
        assert not app.exception
        specs = [json.loads(p.proto.spec) for p in app.get("plotly_chart")]
        assert [p["layout"]["meta"]["metric_family"] for p in specs] == ["gain", "loss"]
    app = app.selectbox(key="cc_distribution_group").set_value("clusters").run()
    assert not app.exception
    assert app.multiselect(key="cc_distribution_levels_clusters").value


def test_grouped_module_only_fdr_matches_full_family_and_invalidates():
    app = app_for("control_derived", module=747)
    app = app.selectbox(key="cc_plot_grouping").set_value("clusters").run()
    app = app.multiselect(key="cc_plot_levels_clusters").set_value([4]).run()
    assert not app.exception
    app = app.button(key="cc_plot_family_fdr").click().run()
    assert not app.exception
    _, full = app.session_state["cc_plot_family_statistics"]
    from scipy.stats import false_discovery_control
    for _, group in full.groupby(["metric_family", "component", "grouping_level", "method"], sort=False):
        valid = group.p.notna()
        assert group.module_family_n.eq(valid.sum()).all()
        assert group.module_family_n.max() <= 186
        np.testing.assert_allclose(group.loc[valid, "q_across_modules"], false_discovery_control(group.loc[valid, "p"]))
    tables = [d.value for d in app.dataframe if "spearman_fdr_across_modules" in d.value]
    assert len(tables) == 2
    assert all(t.spearman_fdr_across_modules.notna().any() for t in tables)
    before = [t.spearman_fdr_across_modules.to_numpy().copy() for t in tables]
    app = app.checkbox(key="cc_plot_group_lines").uncheck().run()
    for expected, table in zip(before, [d.value for d in app.dataframe if "spearman_fdr_across_modules" in d.value]):
        np.testing.assert_array_equal(expected, table.spearman_fdr_across_modules)
    app = app.selectbox(key="cc_scale").set_value("asinh").run()
    assert not app.exception
    assert all(d.value.spearman_fdr_across_modules.isna().all() for d in app.dataframe if "spearman_fdr_across_modules" in d.value)


def test_edge_inspection_shows_calibration_and_real_denominator():
    app = app_for(view="Donor edge explorer", module=smallest_module())
    app = app.button(key="cc_inspect_edges").click().run()
    assert not app.exception
    detail = next(d.value for d in app.dataframe if "gain_contribution" in d.value)
    assert {
        "w",
        "abs_w",
        "delta_abs",
        "a",
        "nu",
        "tau",
        "calibration_n",
        "eligibility_reason",
        "sign_class",
        "aggregate_denominator",
    } <= set(detail)
    assert detail.calibration_n.eq(41).all()
    assert detail.role.eq("independent_target").all()


@pytest.mark.parametrize("view", ["Gene & edge associations", "Edge volcano"])
def test_full_selected_edge_search_uses_separate_correction(view):
    app = app_for(view=view, module=smallest_module())
    app = app.button(key="cc_run_edge_association").click().run()
    assert not app.exception
    _, results = app.session_state["cc_edge_association"]
    assert not results.empty and "q_search" in results
    assert "q_scan" not in results
    if view == "Edge volcano":
        assert results.method.eq("Kruskal-Wallis").all()
        assert "mean_AD_minus_Control" in results
        assert len(app.get("plotly_chart")) >= 1


def test_shared_donor_historical_comparison_and_mask_do_not_borrow_baseline_q():
    app = app_for()
    app.session_state['cc_legacy_comparison_panel'] = True
    app.session_state['cc_baseline_panel'] = True
    app = app.run()
    app = app.button(key="cc_calculate_legacy_comparison").click().run()
    assert not app.exception
    _, result, _ = app.session_state["cc_legacy_comparison"]
    assert "q_comparison_spearman" in result and "q_scan" not in result
    assert result.n.min() > 0 and result.n.max() <= 327
    assert result.n.nunique() == 1
    rule = "ad_control_discovery_effect__hedges_g__fixed_0.40__either"
    app = app.selectbox(key="cc_mask").set_value(rule).run()
    assert not app.exception
    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert spec["layout"]["meta"]["mask"] == rule
    baseline = [d.value for d in app.dataframe if "q_scan" in d.value]
    assert baseline and all(d["mask"].eq("all").all() for d in baseline)
    assert any(
        "All independent targets, native units, unmasked edges" in c.value
        for c in app.caption
    )


def test_ordinal_categorical_override_is_a_new_kw_analysis():
    from app_helpers import data as D

    outcome = sorted(D.SELECTABLE_ASSOCIATION_OUTCOMES)[0]
    app = app_for(module=smallest_module())
    app = app.selectbox(key="cc_outcome").set_value(outcome).run()
    app = app.radio(key="cc_outcome_interpretation").set_value("categorical").run()
    app = app.button(key="cc_calculate_analysis").click().run()
    assert not app.exception
    _, result = app.session_state["cc_current_analysis"]
    assert result.method.eq("Kruskal-Wallis").all()
    assert "q_across_modules" in result and "q_scan" not in result


@pytest.mark.parametrize("resolved", [True, False])
def test_streamed_real_scores_match_direct_spearman(resolved):
    from scipy.stats import spearmanr
    from app_helpers.streaming_associations import stream_grouped_correlations

    family = "full_cohort"
    module = smallest_module(family)
    metadata = A.metadata(family)
    components = ("TS_AC",) if resolved else ("TS", "CT")
    result = stream_grouped_correlations(
        [module],
        metadata,
        module_set=family,
        estimator="calibrated_beta3",
        method="control_anchored",
        resolved=resolved,
        feature="gain",
        phenotype="cogn_global",
        scale="raw",
        components=components,
        diagnoses=("Control", "AD", "MCI"),
        grouping_variable="__all__",
        grouping_levels=(),
        include_pooled=False,
        min_group_n=3,
        edge_rule="all",
        differential_edge_rule="all",
        differential_fdr_scope="global",
        differential_fdr_threshold=0.05,
        score_normalization="standard_pruned",
        analysis_subset="all_donors",
    )
    direct = A.scores(module_set=family, module=module, metric_family="gain")
    for component in components:
        part = direct[direct.component.eq(component)].dropna(
            subset=["metric_raw", "cogn_global"]
        )
        row = result[result.component.eq(component)].iloc[0]
        expected = spearmanr(part.metric_raw, part.cogn_global)
        assert row["n"] == len(part)
        np.testing.assert_allclose(
            [row.spearman_rho, row.spearman_p], expected, rtol=1e-11
        )


@pytest.mark.parametrize(
    "view,key",
    [
        ("Donor explorer", "donor_explorer_calibrated_source"),
        ("Validation & replication", "validation_calibrated_source"),
    ],
)
def test_dedicated_consumers_use_shared_calibrated_renderer(view, key):
    app = app_for(view=view)
    if view == "Validation & replication":
        app = next(w for w in app.radio if w.label == "Validation section").set_value("Technical checks").run()
    app = app.radio(key=key).set_value("calibrated_beta3").run()
    assert not app.exception and app.get("plotly_chart")
    meta = json.loads(app.get("plotly_chart")[0].proto.spec)["layout"]["meta"]
    assert meta["score_version"] == "control_calibrated_connectivity.v2.beta3"
    if view == "Validation & replication":
        assert meta["cohort_scope"] == "incomplete_region_validation"


def test_module_outcome_comparison_calibrated_box_and_display_reuse():
    app = app_for(view="Module–outcome comparison")
    rows = app.session_state["module_comparison_boxes"]
    app.session_state["module_comparison_boxes"] = [rows[0]]
    app = app.run()
    source = next(w for w in app.selectbox if w.label == "Donor score source")
    app = source.set_value("calibrated_beta3").run()
    app = app.button(key="module_comparison_calculate").click().run()
    assert not app.exception
    assert not app.error
    result = app.session_state["module_comparison_result"]
    catalog = next(iter(result["catalogs"].values()))
    assert "q_comparison_spearman" in catalog
    assert catalog.n.max() <= 327
    app = app.radio(key="module_comparison_selection").set_value("Top K").run()
    assert not app.exception
