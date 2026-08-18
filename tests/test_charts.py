from __future__ import annotations

import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from app_helpers.charts import (  # noqa: E402
    aggregate_to_long,
    association_figure,
    correlation_heatmap_figure,
    distribution_figure,
)
from app_helpers.correlations import calculate_correlations  # noqa: E402
from app_helpers.data import load_aggregate, load_aggregate_statistics  # noqa: E402


def test_scatter_and_distribution_chart_paths() -> None:
    aggregate = load_aggregate("control_anchored", 935, "connectivity")
    long = aggregate_to_long(aggregate, "rint")
    stats = load_aggregate_statistics(
        "control_anchored", 935, "motor10_demog_slope", "connectivity"
    )
    scatter = association_figure(
        long,
        stats,
        phenotype="motor10_demog_slope",
        phenotype_label="Demographic-adjusted motor slope",
        feature_label="Connectivity",
        scale="rint",
        scale_label="Z-score",
        diagnoses=["Control", "MCI", "AD"],
        module=935,
        resolved=False,
        color_by="diagnosis_group",
        color_label="Diagnosis group",
        hover_fields={"cogn_global": "Global cognition"},
    )
    assert len(scatter.data) >= 6
    assert "Module M935" in scatter.layout.title.text
    assert all("projid" not in str(trace.hovertemplate) for trace in scatter.data)

    histogram = distribution_figure(
        long,
        feature_label="Connectivity",
        scale_label="Z-score",
        diagnoses=["Control", "MCI", "AD"],
        module=935,
        chart_type="Histogram",
        bins=30,
    )
    assert len(histogram.data) == 6
    assert all(trace.histnorm == "probability density" for trace in histogram.data)


def test_continuous_color_and_correlation_heatmap() -> None:
    aggregate = load_aggregate("control_anchored", 935, "connectivity")
    long = aggregate_to_long(aggregate, "rint")
    stats = load_aggregate_statistics(
        "control_anchored", 935, "cogn_global", "connectivity"
    )
    scatter = association_figure(
        long,
        stats,
        phenotype="cogn_global",
        phenotype_label="Global cognition",
        feature_label="Connectivity",
        scale="rint",
        scale_label="Z-score",
        diagnoses=["Control", "MCI", "AD"],
        module=935,
        resolved=False,
        color_by="motor10_demog_slope",
        color_label="Demographic-adjusted motor slope",
        hover_fields={
            "cogn_global": "Global cognition",
            "motor10_demog_slope": "Demographic-adjusted motor slope",
        },
    )
    assert scatter.layout.coloraxis.colorbar.title.text == "Demographic-adjusted motor slope"
    assert any("motor slope" in str(trace.hovertemplate) for trace in scatter.data)

    correlations = calculate_correlations(
        long,
        group_columns=[
            "module",
            "metric_family",
            "component",
            "component_label",
            "diagnosis_group",
        ],
        outcomes=["cogn_global", "motor10_demog_slope"],
    )
    correlations["outcome_label"] = correlations["outcome"].map(
        {
            "cogn_global": "Global cognition",
            "motor10_demog_slope": "Motor slope",
        }
    )
    correlations["heatmap_row"] = (
        correlations["metric_family"] + " · " + correlations["component_label"]
    )
    selected = correlations.loc[correlations["diagnosis_group"].eq("AD")]
    heatmap = correlation_heatmap_figure(
        selected,
        value_column="pearson_r",
        p_column="pearson_p",
        fdr_column="pearson_fdr_displayed_family",
        title="Test correlations",
    )
    assert len(heatmap.data) == 1
    assert heatmap.data[0].zmin == -1
    assert heatmap.data[0].zmax == 1
