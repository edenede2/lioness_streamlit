from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from app_helpers.charts import (  # noqa: E402
    aggregate_to_long,
    association_figure,
    correlation_heatmap_figure,
    distribution_figure,
    mdc_module_figure,
    mdc_overview_figure,
)
from app_helpers.correlations import calculate_correlations  # noqa: E402
from app_helpers.data import (  # noqa: E402
    load_aggregate,
    load_aggregate_statistics,
    load_mdc_summary,
)


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


def test_correlation_heatmap_can_cluster_rows_and_columns() -> None:
    rows = ["A", "B", "C", "D"]
    outcomes = ["outcome_1", "outcome_3", "outcome_2"]
    coefficients = {
        "A": [1.0, -1.0, 0.9],
        "B": [-1.0, 1.0, -0.9],
        "C": [0.9, -0.8, 1.0],
        "D": [-0.9, 0.8, -1.0],
    }
    records = []
    for row in rows:
        for outcome, coefficient in zip(outcomes, coefficients[row], strict=True):
            records.append(
                {
                    "heatmap_row": row,
                    "outcome": outcome,
                    "outcome_label": outcome,
                    "pearson_r": coefficient,
                    "pearson_p": 0.01,
                    "pearson_fdr_displayed_family": 0.02,
                    "n": 100,
                }
            )

    figure = correlation_heatmap_figure(
        pd.DataFrame.from_records(records),
        value_column="pearson_r",
        p_column="pearson_p",
        fdr_column="pearson_fdr_displayed_family",
        title="Clustered correlations",
        row_order=rows,
        cluster_rows=True,
        cluster_columns=True,
    )
    clustered_rows = list(figure.data[0].y)
    clustered_columns = list(figure.data[0].x)

    assert set(clustered_rows) == set(rows)
    assert set(clustered_columns) == set(outcomes)
    assert abs(clustered_rows.index("A") - clustered_rows.index("C")) == 1
    assert abs(clustered_rows.index("B") - clustered_rows.index("D")) == 1
    assert abs(clustered_columns.index("outcome_1") - clustered_columns.index("outcome_2")) == 1


def test_mdc_selected_module_and_overview_charts() -> None:
    mdc = load_mdc_summary()
    selected = mdc.loc[mdc["module"].eq(1918)].iloc[0]

    module_figure = mdc_module_figure(selected, threshold=0.05)
    assert len(module_figure.data) == 1
    assert list(module_figure.data[0].x) == [
        "Total",
        "Tissue-specific (TS)",
        "Cross-tissue (CT)",
    ]
    assert "M1918" in module_figure.layout.title.text

    overview = mdc_overview_figure(mdc, selected_module=1918, threshold=0.05)
    assert len(overview.data) >= 2
    assert any(trace.name == "Selected M1918" for trace in overview.data)
    assert overview.layout.xaxis.title.text == "TS log2 MDC (AD / Control)"
