"""Historical plot contracts and independent module-only association families."""
import itertools
import json
import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control, pearsonr, spearmanr

from app_helpers.control_calibrated.association_fdr import add_module_fdr
from app_helpers.control_calibrated import presentation as P, adapter as A
from app_helpers.charts import DIAGNOSIS_COLORS, DIAGNOSIS_SYMBOLS, grouped_association_figure


def test_bh_holds_every_context_fixed_and_preserves_archived_values():
    rows = []
    for tissue, beta, metric, component, outcome, method, level in itertools.product(
            ("AC", "PCG"), (1, 3), ("gain", "loss"), ("CT", "TS"),
            ("cogn_global", "gpath"), ("Pearson", "Spearman"), ("AD", "Control", "4")):
        for module, p in zip((1, 2, 3, 4), (0.001, .04, .3, np.nan)):
            rows.append(dict(module_key=f"regional::{tissue}::M{module}", family="regional",
                beta=beta, metric_family=metric, component=component, outcome=outcome,
                method=method, grouping_level=level, p=p, q_scan=.99))
    result = add_module_fdr(pd.DataFrame(rows))
    expected = false_discovery_control([.001, .04, .3])
    for _, group in result.groupby(["regional_tissue", "beta", "metric_family", "component", "outcome", "method", "grouping_level"]):
        np.testing.assert_allclose(group.q_across_modules.iloc[:3], expected)
        assert group.module_family_n.eq(3).all()
        assert pd.isna(group.q_across_modules.iloc[3])
    assert result.q_scan.eq(.99).all()
    with pytest.raises(ValueError, match="Duplicate"):
        add_module_fdr(pd.concat([result, result.iloc[[0]]]))


def fixture_frame(module=1):
    rng = np.random.default_rng(112 + module)
    rows = []
    for component in ("CT", "TS"):
        for i in range(36):
            rows.append(dict(sample_id=f"test-{i}", module=module, module_key=f"full_cohort::M{module}",
                component=component, component_label=component, metric_family="gain",
                metric_raw=rng.normal(), cogn_global=rng.normal(),
                diagnosis_group=["Control", "MCI", "AD"][i % 3], clusters=1+i%2))
    return pd.DataFrame(rows)


def test_module_scan_matches_direct_scipy_and_never_bh_over_groups(monkeypatch):
    jobs = [dict(family="full_cohort", module=m, module_key=f"full_cohort::M{m}", regional_tissue=None) for m in (1, 2, 3)]
    frames = {m:fixture_frame(m) for m in (1, 2, 3)}
    frames[3]["metric_raw"] = 0  # excluded from the family, not counted as p=1
    monkeypatch.setattr(A, "catalog", lambda:dict(modules=jobs))
    monkeypatch.setattr(A, "scores", lambda **kw:frames[kw["module"]].copy())
    P.module_statistics.clear()
    result = P.module_statistics("full_cohort", "calibrated_beta3", "complete_450", ("all", "global", .05),
        None, "raw", ("CT", "TS"), ("gain",), "cogn_global", "diagnosis_group",
        ("Control", "AD", "__pooled__"), 10, tuple(frames[1].sample_id.unique()))
    for (component, level, method), group in result.groupby(["component", "grouping_level", "method"]):
        expected = []
        for m in (1, 2):
            f = frames[m].loc[frames[m].component.eq(component)]
            if level != "__pooled__":
                f = f.loc[f.diagnosis_group.eq(level)]
            statistic = (pearsonr if method == "Pearson" else spearmanr)(f.metric_raw, f.cogn_global)
            actual = group.loc[group.module.eq(m)].iloc[0]
            assert actual.p == pytest.approx(statistic.pvalue)
            expected.append(statistic.pvalue)
        np.testing.assert_allclose(group.q_across_modules.iloc[:2], false_discovery_control(expected))
        assert group.module_family_n.eq(2).all()
        assert pd.isna(group.q_across_modules.iloc[-1])
    P.module_statistics.clear()


def test_nominal_category_tests_do_not_correlate_arbitrary_codes(monkeypatch):
    monkeypatch.setattr(A, "catalog", lambda:dict(modules=[dict(family="full_cohort", module=1, module_key="full_cohort::M1", regional_tissue=None)]))
    monkeypatch.setattr(A, "scores", lambda **kw:fixture_frame())
    P.module_statistics.clear()
    result = P.module_statistics("full_cohort", "calibrated_beta1", "complete_450", ("all", "global", .05),
        None, "raw", ("CT", "TS"), ("gain",), "clusters", "clusters", (1, 2), 10,
        tuple(fixture_frame().sample_id.unique()), categorical=True)
    assert result.method.eq("Kruskal-Wallis").all()
    assert "spearman_rho" not in result
    assert result.module_family_n.eq(1).all()
    P.module_statistics.clear()


def test_ols_saved_associations_use_separate_component_and_group_families(monkeypatch):
    jobs = [dict(family="full_cohort", module=m, module_key=f"full_cohort::M{m}", regional_tissue=None) for m in (1, 2)]
    monkeypatch.setattr(A, "catalog", lambda:dict(modules=jobs))
    monkeypatch.setattr(A, "scores", lambda **kw:fixture_frame(kw["module"]))
    P.module_statistics.clear()
    result = P.module_statistics("full_cohort", "calibrated_beta3", "complete_450", ("all", "global", .05),
        None, "raw", ("CT", "TS"), ("gain",), "cogn_global", "diagnosis_group", ("Control", "AD"),
        10, tuple(fixture_frame().sample_id.unique()), ols_covariates=())
    assert result.method.eq("Descriptive OLS").all()
    for _, group in result.groupby(["component", "grouping_level"]):
        assert group.module_family_n.eq(2).all()
        np.testing.assert_allclose(group.q_across_modules, false_discovery_control(group.p))
    P.module_statistics.clear()


def test_plot_defaults_reuse_historical_colors_shapes_lines_and_axes():
    f = fixture_frame().assign(metric_value=lambda x:x.metric_raw)
    statistics = P.plot_statistics(f, "cogn_global", "diagnosis_group", ("Control", "MCI", "AD", "__pooled__"), 10)
    assert statistics.spearman_fdr_across_modules.isna().all()  # never one-module BH
    fig = grouped_association_figure(f, statistics, phenotype="cogn_global", phenotype_label="Global cognition",
        feature_label="Gain", scale_label="native", grouping_variable="diagnosis_group",
        grouping_levels=["Control", "MCI", "AD"], grouping_labels={}, module=1,
        color_by="diagnosis_group", color_label="Diagnosis", hover_fields={})
    for trace in fig.data:
        if trace.mode == "markers":
            assert trace.marker.color == DIAGNOSIS_COLORS[trace.name]
            assert trace.marker.symbol == DIAGNOSIS_SYMBOLS[trace.name]
    lines = [t for t in fig.data if t.mode == "lines"]
    assert len(lines) == 8  # three groups and the independent pooled fit, per panel
    assert any(t.line.color == "#000000" and t.line.dash == "dash" for t in lines)
    assert "Gain" in fig.layout.xaxis.title.text and "Global cognition" in fig.layout.yaxis.title.text
    assert fig.layout.legend.y >= 1
    assert fig.layout.legend.groupclick == "togglegroup"
