"""Validation evidence must not conflate numeric reproduction with replication."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app_helpers import camel_validation as V
from app_helpers import validation_review as R
from app_helpers.validation_review_ui import scatter_figure

APP = Path(__file__).resolve().parents[1]
STAGE = APP.parents[1] / "out/camel_validation/20261021_frozen164_incomplete_regions/public_bundle"


def example():
    return pd.DataFrame(dict(
        module=[1, 2, 3, 4, 5], metric_family="connectivity", component="TS_AC",
        phenotype="cogn_global", diagnosis_group="AD", original_rho=[.6, .5, -.5, .3, .7],
        original_p_spearman=[.001, .002, .003, .2, .001], original_q_spearman=[.01, .02, .03, .5, .01],
        validation_rho_raw=[.5, .1, .4, .2, np.nan], validation_p_spearman_raw=[.01, .5, .01, .2, np.nan],
        validation_q_spearman_raw_across_modules=[.04, .8, .04, .5, np.nan],
        validation_rho_rint=[.4, .05, .3, .15, np.nan], validation_p_spearman_rint=[.02, .6, .03, .3, np.nan],
        validation_q_spearman_rint_across_modules=[.08, .9, .08, .6, np.nan],
        original_r_raw=[.5, .4, -.4, .2, .6], original_p_raw=[.01, .02, .03, .2, .01],
        original_q_raw=[.04, .05, .06, .5, .04], validation_r_raw=[.4, .1, .3, .1, np.nan],
        validation_p_raw=[.02, .6, .03, .3, np.nan], validation_q_raw_across_modules=[.08, .9, .08, .6, np.nan],
        original_family_n_spearman=186, validation_family_n_spearman_raw=180,
        validation_family_n_spearman_rint=179, original_family_n_raw=186, validation_family_n_raw=180,
    ))


def test_replication_criteria_require_original_signal_direction_and_validation_evidence():
    f = R.replication_labels(R.association_effects(example()), .05)
    assert f.replication_status.tolist() == ["Same direction + validation FDR", "Same direction only",
        "Original signal; opposite/zero direction", "No original FDR signal", "Unavailable comparison"]
    assert R.replication_counts(f) == dict(comparable_modules=4, original_signals=4,
        original_signals_evaluable=3, direction_agreement_all=3, same_direction=2, nominal_replication=1, fdr_replication=1)
    assert pd.isna(f.same_direction.iloc[-1])


def test_scale_specific_spearman_and_full_family_corrections_survive_display_filter():
    f = R.association_effects(example(), scale="rint")
    assert f.validation_effect.iloc[0] == .4
    assert f.validation_family_n.iloc[0] == 179
    displayed = R.replication_labels(f).iloc[:1]
    assert displayed.validation_fdr.iloc[0] == .08  # not recomputed over one displayed row
    assert displayed.original_family_n.iloc[0] == 186
    assert not displayed.validation_fdr_signal.iloc[0]
    pearson = R.association_effects(example(), "Pearson")
    assert pearson.original_effect.iloc[0] == .5
    assert pearson.validation_effect.iloc[0] == .4
    with pytest.raises(ValueError):
        R.association_effects(example(), scale="validation_fitted")


def test_scatter_has_actual_points_identity_and_external_legend():
    f = R.replication_labels(R.association_effects(example()))
    fig = scatter_figure(f, "original_effect", "validation_effect", "Original", "Validation", bounded=True)
    points = [t for t in fig.data if t.mode == "markers"]
    assert sum(len(t.x) for t in points) == 4
    assert list(fig.layout.xaxis.range) == [-1.02, 1.02]
    assert fig.layout.legend.y > 1
    assert fig.data[-1].name == "Equal effect"
    assert scatter_figure(f.iloc[4:], "original_effect", "validation_effect", "Original", "Validation") is None


@pytest.fixture
def stage(monkeypatch):
    if not (STAGE / "review/manifest.json").exists():
        pytest.skip("Source-backed private evidence has not been staged; never download in tests")
    monkeypatch.setenv("CAMEL_VALIDATION_DATA_ROOT", str(STAGE))
    return STAGE


def widget(elements, label):
    return next(e for e in elements if e.label == label)


def test_source_backed_shards_are_small_aggregate_and_keep_full_family_fdr(stage):
    manifest = R.manifest()
    assert manifest["modules"] == 186 and manifest["validation_donors"] == 793
    f = R.association_table("all", "standard_pruned", "connectivity", "cogn_global")
    assert len(f) == 186 * 6 * 5 and not f.duplicated(R.MATCH_KEYS).any()
    assert not {"sample_id", "donor", "projid"}.intersection(f)
    subset = f.loc[f.component.eq("CT_AC__DLPFC") & f.diagnosis_group.eq("AD")]
    from app_helpers.correlations import benjamini_hochberg
    np.testing.assert_allclose(benjamini_hochberg(subset.validation_p_spearman_raw),
                               subset.validation_q_spearman_raw_across_modules, equal_nan=True)
    for entry in manifest["association_shards"]:
        assert (stage / "review" / entry["path"]).stat().st_size < 2 * 1024**2


def test_real_review_sections_do_not_load_donor_scores(stage, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Review page must never read donor scores or metadata")
    for name in ("scores", "summaries", "metadata"):
        monkeypatch.setattr(V, name, forbidden)
    script = '''
from app_helpers.validation_review_ui import render_view
from app_helpers.figure_export import render_exportable_plot
render_view(render_exportable_plot)
'''
    app = AppTest.from_string(script, default_timeout=120).run()
    assert not app.exception
    assert any("Technical reproduction" in x.value for x in app.subheader)
    app = widget(app.radio, "Validation section").set_value("Association replication").run()
    assert not app.exception and app.get("plotly_chart")
    widget(app.selectbox, "Validation donor group").set_value("All donors")
    app.run()
    assert not app.exception and any("No comparable associations" in x.value for x in app.info)
    app = widget(app.radio, "Validation section").set_value("MDC replication").run()
    assert not app.exception and app.get("plotly_chart")
    app = widget(app.radio, "MDC comparison scale").set_value("Raw MDC").run()
    assert not app.exception and app.get("plotly_chart")
    app = widget(app.multiselect, "MDC validation components").set_value(["CT_AC__PCGBA23"]).run()
    assert not app.exception and any("descriptive only" in x.value for x in app.warning)
    app = widget(app.radio, "Validation section").set_value("Coverage & completion").run()
    assert not app.exception and app.get("plotly_chart")
    counts = next(x.value for x in app.dataframe if "component_label" in x.value)
    assert counts.n.tolist() == [266, 752, 195, 230, 14, 176]


def test_main_navigation_routes_review_before_single_partition_scoring(stage):
    app = AppTest.from_file(APP / "streamlit_app.py", default_timeout=180).run()
    app = widget(app.pills, "Analysis view").set_value("Validation & replication").run()
    assert not app.exception
    assert any(x.value == "Validation & replication" for x in app.header)
    assert not any(x.label == "Module definition" for x in app.selectbox)


def test_new_validation_cohort_and_completed_effect_mask_render_existing_views(stage):
    app = AppTest.from_file(APP / "streamlit_app.py", default_timeout=180).run()
    app = widget(app.selectbox, "Module definition").set_value("control_derived").run()
    app = widget(app.radio, "Donor cohort scope").set_value("incomplete_region_validation").run()
    assert not app.exception and app.get("plotly_chart")
    if V.completed_effect_options():
        app = widget(app.radio, "AD–Control edge subset").set_value("effect_size").run()
        assert not app.exception
    for view in ("Feature distributions", "Correlation heatmaps", "Statistics", "Edge summaries", "MDC"):
        app = widget(app.pills, "Analysis view").set_value(view).run()
        assert not app.exception, view


def test_uninstalled_review_is_explicit_not_fake_results(monkeypatch, tmp_path):
    monkeypatch.setenv("CAMEL_VALIDATION_DATA_ROOT", str(tmp_path))
    app = AppTest.from_string('''
from app_helpers.validation_review_ui import render_view
render_view(lambda *args, **kwargs: None)
''').run()
    assert not app.exception
    assert any("has not been installed" in x.value for x in app.info)
    assert not app.metric
