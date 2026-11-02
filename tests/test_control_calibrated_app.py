"""Integration tests against the generated local catalog; no prediction training."""

import json
from pathlib import Path
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from app_helpers.control_calibrated import registry as CC, adapter as A

APP = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not CC.ready(), reason="Run the full offline calibrated catalog first"
)


def app_for(family="full_cohort", view="Associations", beta=3, module=None):
    app = AppTest.from_file(APP / "streamlit_app.py", default_timeout=180)
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
    assert any(
        "Control-calibrated connectivity" in item.value for item in app.subheader
    )


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
    assert "q_search" in result and "q_scan" not in result


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
