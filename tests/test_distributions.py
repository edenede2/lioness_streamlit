from __future__ import annotations

import numpy as np
import pandas as pd

from app_helpers.correlations import benjamini_hochberg
from app_helpers.distributions import (
    add_pairwise_across_module_fdr,
    calculate_pairwise_distribution_statistics,
    cliffs_delta,
    default_reference_level,
    distribution_contrasts,
)


def test_distribution_contrasts_and_biological_reference_defaults() -> None:
    levels = ["Control", "MCI", "AD"]
    assert distribution_contrasts(
        levels, mode="reference", reference_level="Control"
    ) == [("Control", "MCI"), ("Control", "AD")]
    assert distribution_contrasts(levels, mode="all_pairs") == [
        ("Control", "MCI"), ("Control", "AD"), ("MCI", "AD")
    ]
    assert default_reference_level(
        "diagnosis_group", levels, {"Control": 12, "MCI": 8, "AD": 14},
        minimum_group_n=10,
    ) == "Control"
    assert default_reference_level(
        "apoe_genotype", ["ε2/ε2", "ε3/ε3"],
        {"ε2/ε2": 3, "ε3/ε3": 20}, minimum_group_n=10,
    ) == "ε3/ε3"


def test_pairwise_statistics_direction_ties_and_deterministic_bootstrap() -> None:
    frame = pd.DataFrame(
        {
            "module": [1] * 12,
            "component": ["CT"] * 12,
            "component_label": ["CT aggregate"] * 12,
            "metric_family": ["connectivity"] * 12,
            "group": ["reference"] * 6 + ["comparison"] * 6,
            "metric_value": [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5],
        }
    )
    kwargs = dict(
        group_columns=["module", "component", "component_label", "metric_family"],
        category_column="group",
        contrasts=[("reference", "comparison")],
        minimum_group_n=5,
        bootstrap_resamples=100,
        seed=42,
        include_ks=True,
    )
    first = calculate_pairwise_distribution_statistics(frame, **kwargs).iloc[0]
    second = calculate_pairwise_distribution_statistics(frame, **kwargs).iloc[0]
    reference = frame.loc[frame["group"].eq("reference"), "metric_value"].to_numpy()
    comparison = frame.loc[frame["group"].eq("comparison"), "metric_value"].to_numpy()
    brute_delta = float(
        np.mean(comparison[:, None] > reference[None, :])
        - np.mean(comparison[:, None] < reference[None, :])
    )
    assert np.isclose(first["cliffs_delta"], brute_delta)
    assert np.isclose(first["cliffs_delta"], cliffs_delta(comparison, reference))
    assert first["cliffs_delta"] > 0
    assert np.isclose(
        first["probability_superiority"], (first["cliffs_delta"] + 1) / 2
    )
    assert first["cliffs_delta_ci_low"] == second["cliffs_delta_ci_low"]
    assert first["cliffs_delta_ci_high"] == second["cliffs_delta_ci_high"]
    assert 0 <= first["ks_d"] <= 1


def test_pairwise_fdr_is_separate_for_each_contrast_and_excludes_missing() -> None:
    rows = []
    for reference, comparison, p_values in (
        ("Control", "AD", [0.01, 0.04, np.nan]),
        ("Control", "MCI", [0.20, 0.03, 0.50]),
    ):
        for module, p_value in enumerate(p_values, start=1):
            rows.append(
                {
                    "module": module,
                    "metric_family": "connectivity",
                    "component": "CT",
                    "grouping_variable": "diagnosis_group",
                    "reference_level_key": reference,
                    "comparison_level_key": comparison,
                    "mann_whitney_p": p_value,
                    "ks_p": p_value,
                }
            )
    adjusted = add_pairwise_across_module_fdr(
        pd.DataFrame(rows),
        family_columns=["metric_family", "component", "grouping_variable"],
    )
    for comparison in ("AD", "MCI"):
        group = adjusted.loc[adjusted["comparison_level_key"].eq(comparison)]
        assert np.allclose(
            group["mann_whitney_fdr_across_modules"],
            benjamini_hochberg(group["mann_whitney_p"]),
            equal_nan=True,
        )
    ad = adjusted.loc[adjusted["comparison_level_key"].eq("AD")]
    mci = adjusted.loc[adjusted["comparison_level_key"].eq("MCI")]
    assert ad["mann_whitney_fdr_module_family_n"].eq(2).all()
    assert mci["mann_whitney_fdr_module_family_n"].eq(3).all()


def test_pairwise_sparse_and_constant_groups_remain_explicitly_unavailable() -> None:
    frame = pd.DataFrame(
        {
            "module": [1] * 9,
            "component": ["TS"] * 9,
            "metric_family": ["connectivity"] * 9,
            "group": ["A"] * 5 + ["B"] * 4,
            "metric_value": [1.0] * 9,
        }
    )
    sparse = calculate_pairwise_distribution_statistics(
        frame, ["module", "component", "metric_family"],
        category_column="group", contrasts=[("A", "B")], minimum_group_n=5,
    ).iloc[0]
    assert not bool(sparse["eligible"])
    assert sparse["unavailable_reason"] == "comparison n < 5"

    constant = calculate_pairwise_distribution_statistics(
        frame, ["module", "component", "metric_family"],
        category_column="group", contrasts=[("A", "B")], minimum_group_n=4,
    ).iloc[0]
    assert not bool(constant["eligible"])
    assert constant["unavailable_reason"] == "constant scores"


def test_rank_distribution_statistics_are_invariant_to_monotone_transform() -> None:
    raw = pd.DataFrame(
        {
            "module": [1] * 20,
            "component": ["CT"] * 20,
            "group": ["A"] * 10 + ["B"] * 10,
            "metric_value": np.linspace(-4, 5, 20),
        }
    )
    transformed = raw.copy()
    transformed["metric_value"] = np.arcsinh(transformed["metric_value"])
    kwargs = dict(
        group_columns=["module", "component"], category_column="group",
        contrasts=[("A", "B")], minimum_group_n=5,
        bootstrap_resamples=0, include_ks=True,
    )
    raw_result = calculate_pairwise_distribution_statistics(raw, **kwargs).iloc[0]
    transformed_result = calculate_pairwise_distribution_statistics(
        transformed, **kwargs
    ).iloc[0]
    for column in ("mann_whitney_u", "mann_whitney_p", "cliffs_delta", "ks_d", "ks_p"):
        assert np.isclose(raw_result[column], transformed_result[column])
