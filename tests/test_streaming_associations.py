from __future__ import annotations

import numpy as np
import pandas as pd

from app_helpers import streaming_associations as streaming
from app_helpers.correlations import calculate_correlations


def _metadata() -> pd.DataFrame:
    sample_ids = [f"S-{index:02d}" for index in range(12)]
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "diagnosis_group": ["Control"] * 6 + ["AD"] * 6,
            "ad_control_split": ["Discovery"] * 12,
            "clusters": [1] * 3 + [2] * 3 + [1] * 3 + [2] * 3,
            "outcome": np.arange(12, dtype=float),
        }
    )


def _resolved_module(module: int) -> pd.DataFrame:
    sample_ids = [f"S-{index:02d}" for index in range(12)]
    rows = []
    for component, offset in (("TS_AC", 0.0), ("CT_AC__DLPFC", 2.0)):
        for index, sample_id in enumerate(sample_ids):
            rows.append(
                {
                    "sample_id": sample_id,
                    "module": module,
                    "metric_family": "connectivity",
                    "component": component,
                    "component_class": component[:2],
                    "component_label": component,
                    "metric_rint": float(index + offset + module),
                    "lioness_method": "control_anchored",
                }
            )
    return pd.DataFrame(rows)


def test_grouped_correlations_read_one_narrow_module_at_a_time(monkeypatch) -> None:
    calls: list[tuple[int | None, bool, str | None]] = []

    def fake_load(*args, module=None, metric_scale=None,
                  include_embedded_metadata=True, **kwargs):
        calls.append((module, include_embedded_metadata, metric_scale))
        return _resolved_module(int(module))

    monkeypatch.setattr(streaming, "load_resolved_scope", fake_load)
    result = streaming.stream_grouped_correlations(
        (1, 2), _metadata(), module_set="full_cohort", estimator="lioness",
        method="control_anchored", resolved=True, feature="connectivity",
        phenotype="outcome", scale="rint",
        components=("TS_AC", "CT_AC__DLPFC"),
        diagnoses=("Control", "AD"), grouping_variable="clusters",
        grouping_levels=(1, 2), include_pooled=True, min_group_n=3,
        edge_rule="all", differential_edge_rule="all",
        differential_fdr_scope="global", differential_fdr_threshold=0.05,
        score_normalization="standard_pruned", analysis_subset="all_donors",
    )

    assert calls == [(1, False, "rint"), (2, False, "rint")]
    assert result.shape[0] == 2 * 2 * 3
    assert set(result["grouping_level"].astype(str)) == {"1", "2", "__pooled__"}
    assert result["n"].min() == 6
    assert result.loc[
        result["grouping_level"].astype(str).eq("__pooled__"), "n"
    ].eq(12).all()


def test_grouped_correlation_matrix_keeps_level_fdr_families_separate(
    monkeypatch,
) -> None:
    calls: list[int] = []

    def fake_load(*args, module=None, **kwargs):
        calls.append(int(module))
        return _resolved_module(int(module))

    monkeypatch.setattr(streaming, "load_resolved_scope", fake_load)
    metadata = _metadata()
    metadata["outcome_two"] = np.square(metadata["outcome"])
    result = streaming.stream_grouped_correlation_matrix(
        (1, 2), metadata, module_set="full_cohort", estimator="lioness",
        method="control_anchored", resolved=True, feature="connectivity",
        component=None, outcomes=("outcome", "outcome_two"), scale="rint",
        diagnoses=("Control", "AD"), grouping_variable="clusters",
        grouping_levels=(1, 2), min_group_n=3, edge_rule="all",
        differential_edge_rule="all", differential_fdr_scope="global",
        differential_fdr_threshold=0.05,
        score_normalization="standard_pruned", analysis_subset="all_donors",
    )

    assert calls == [1, 2]
    assert result.shape[0] == 2 * 2 * 2 * 2
    assert set(result["grouping_level"]) == {1, 2}
    assert result["pearson_fdr_module_family_n"].eq(2).all()
    family_sizes = result.groupby(
        ["component", "outcome", "grouping_level"], observed=True
    )["module"].nunique()
    assert family_sizes.eq(2).all()


def test_pooled_correlations_apply_diagnosis_filter_per_module(monkeypatch) -> None:
    monkeypatch.setattr(
        streaming,
        "load_resolved_scope",
        lambda *args, module=None, **kwargs: _resolved_module(int(module)),
    )
    result = streaming.stream_pooled_correlations(
        (1, 2), _metadata(), module_set="full_cohort", estimator="lioness",
        method="control_anchored", resolved=True, feature="connectivity",
        component="TS_AC", outcomes=("outcome",), scale="rint",
        diagnoses=("AD",), edge_rule="all", differential_edge_rule="all",
        differential_fdr_scope="global", differential_fdr_threshold=0.05,
        score_normalization="standard_pruned", analysis_subset="all_donors",
    )

    assert result.shape[0] == 2
    assert result["diagnosis_group"].eq("All donors").all()
    assert result["n"].eq(6).all()


def test_vectorized_pooled_statistics_match_cellwise_correlations() -> None:
    scores = _resolved_module(1)
    metadata = _metadata()
    frame = scores.rename(columns={"metric_rint": "metric_value"}).merge(
        metadata[["sample_id", "outcome"]],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    frame["diagnosis_group"] = "All donors"
    # Exercise pairwise missingness and tied ranks, not only complete monotone data.
    frame.loc[
        frame["sample_id"].eq("S-02") & frame["component"].eq("TS_AC"),
        "metric_value",
    ] = np.nan
    frame.loc[frame["sample_id"].isin(["S-03", "S-04"]), "outcome"] = 4.0

    expected = calculate_correlations(
        frame,
        [
            "module", "metric_family", "component", "component_label",
            "diagnosis_group",
        ],
        ["outcome"],
    ).sort_values("component").reset_index(drop=True)
    observed = streaming._pooled_module_correlations(
        frame, ("outcome",)
    ).sort_values("component").reset_index(drop=True)

    assert observed[["module", "component", "outcome", "n"]].equals(
        expected[["module", "component", "outcome", "n"]]
    )
    for column in (
        "pearson_r", "pearson_p", "spearman_rho", "spearman_p",
        "pearson_fdr_displayed_family", "spearman_fdr_displayed_family",
    ):
        assert np.allclose(observed[column], expected[column], equal_nan=True)
