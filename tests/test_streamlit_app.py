from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

import app_helpers.charts as chart_helpers


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def widget_with_label(elements, label: str):
    return next(element for element in elements if element.label == label)


def assert_app_clean(app: AppTest) -> None:
    assert not app.exception
    assert not [error for error in app.error if "Traceback" in str(error.value)]


def select_view(app: AppTest, label: str) -> AppTest:
    return widget_with_label(app.pills, "Analysis view").set_value(label).run()


def test_streamlit_hot_reload_recovers_stale_chart_helper(monkeypatch) -> None:
    monkeypatch.delattr(chart_helpers, "CONTINUOUS_COLOR_SCALES")
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert_app_clean(app)
    assert hasattr(chart_helpers, "CONTINUOUS_COLOR_SCALES")


def test_streamlit_table_column_picker_filters_and_resets() -> None:
    assert "st.dataframe(" not in APP.read_text(encoding="utf-8")
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert_app_clean(app)
    picker = widget_with_label(app.multiselect, "Module composition columns")
    all_columns = list(picker.value)
    app = picker.set_value(["Tissue", "Genes"]).run()
    assert_app_clean(app)
    assert list(app.dataframe[0].value.columns) == ["Tissue", "Genes"]
    reset = next(button for button in app.button if button.label == "Reset to all columns")
    app = reset.click().run()
    assert_app_clean(app)
    assert list(app.dataframe[0].value.columns) == all_columns


def test_every_lazy_analysis_view_renders_cleanly() -> None:
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert_app_clean(app)
    view_selector = widget_with_label(app.pills, "Analysis view")
    assert view_selector.value == "Associations"
    for view in view_selector.options:
        app = select_view(app, view)
        assert_app_clean(app)


def test_streamlit_smoke_lioness_and_bonobo_module_definitions() -> None:
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert_app_clean(app)

    outcome = widget_with_label(app.selectbox, "Association outcome")
    assert len(outcome.options) == 12
    outcome.set_value("adnc")
    widget_with_label(app.selectbox, "Color points by").set_value(
        "age_at_death"
    ).run()
    widget_with_label(app.selectbox, "Continuous color scale").set_value("Viridis")
    widget_with_label(app.radio, "Network estimator").set_value("bonobo").run()
    assert_app_clean(app)

    edge_subset = widget_with_label(app.radio, "BONOBO edge subset")
    edge_subset.set_value("Significant edges").run()
    widget_with_label(app.radio, "Significant-edge rule").set_value("bh_fdr05").run()
    assert_app_clean(app)

    widget_with_label(app.selectbox, "Module definition").set_value(
        "control_derived"
    ).run()
    assert_app_clean(app)
    network_method = widget_with_label(app.radio, "Network method")
    assert network_method.value == "bonobo"
    assert network_method.options == ["All-donor empirical-Bayes BONOBO"]


def test_streamlit_differential_filter_can_switch_bh_scope_and_cutoff() -> None:
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert_app_clean(app)

    widget_with_label(app.radio, "AD–Control edge subset").set_value(
        "ad_control_discovery_fdr05"
    ).run()
    widget_with_label(app.radio, "Differential-edge FDR scope").set_value(
        "per_module"
    ).run()
    widget_with_label(app.radio, "Differential-edge FDR cutoff").set_value(
        0.10
    ).run()
    assert_app_clean(app)
    assert widget_with_label(app.selectbox, "Evaluation cohort").value == (
        "validation_ad_control"
    )

    widget_with_label(app.radio, "AD–Control edge subset").set_value("all").run()
    assert_app_clean(app)


def test_streamlit_module_finder_supports_each_ranking_criterion() -> None:
    app = AppTest.from_file(APP, default_timeout=180).run()
    app = select_view(app, "Module finder")
    assert_app_clean(app)
    criterion = widget_with_label(app.radio, "Ranking criterion")
    assert criterion.value == "ct_ts"
    criterion.set_value("ad_control").run()
    assert_app_clean(app)
    widget_with_label(app.radio, "Ranking criterion").set_value("both").run()
    assert_app_clean(app)
    widget_with_label(app.selectbox, "Association-difference FDR filter").set_value(
        0.10
    ).run()
    assert_app_clean(app)


def test_streamlit_mdc_can_switch_to_raw_scale() -> None:
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert_app_clean(app)
    app = select_view(app, "MDC")
    assert_app_clean(app)

    scale = widget_with_label(app.radio, "MDC display scale")
    assert scale.value == "log2"
    scale.set_value("raw").run()
    assert_app_clean(app)
    assert widget_with_label(app.radio, "MDC display scale").value == "raw"

    widget_with_label(app.selectbox, "Module definition").set_value(
        "control_derived"
    ).run()
    assert_app_clean(app)
    assert widget_with_label(app.radio, "MDC display scale").value == "raw"


def test_streamlit_pathway_resolved_mdc_controls_both_module_sets() -> None:
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert_app_clean(app)
    app = select_view(app, "MDC")
    assert_app_clean(app)
    assert any(tab.label == "Region-resolved MDC" for tab in app.tabs)
    assert any(tab.label == "Pathway-resolved MDC" for tab in app.tabs)
    assert widget_with_label(app.selectbox, "Pathway detail").value

    widget_with_label(app.radio, "KEGG enrichment threshold").set_value(0.10).run()
    widget_with_label(app.radio, "Module MDC rows").set_value(
        "MDC FDR-significant only"
    ).run()
    assert_app_clean(app)

    widget_with_label(app.selectbox, "Module definition").set_value(
        "control_derived"
    ).run()
    assert_app_clean(app)
    assert widget_with_label(app.selectbox, "Pathway detail").value
