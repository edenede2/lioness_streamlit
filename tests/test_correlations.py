from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from app_helpers.correlations import (  # noqa: E402
    add_across_module_fdr,
    benjamini_hochberg,
    calculate_categorical_associations,
    calculate_correlations,
)
from app_helpers.data import load_aggregate_statistics  # noqa: E402


def test_across_module_fdr_keeps_pearson_and_spearman_families_separate() -> None:
    frame = pd.DataFrame(
        {
            "module": [1, 2, 3, 1, 2, 3],
            "component": ["CT"] * 3 + ["TS"] * 3,
            "diagnosis_group": ["AD"] * 6,
            "outcome": ["motor"] * 6,
            "pearson_p": [0.01, 0.04, 0.50, 0.20, 0.40, np.nan],
            "spearman_p": [0.03, 0.02, 0.90, 0.01, 0.80, 0.60],
        }
    )
    adjusted = add_across_module_fdr(
        frame,
        family_columns=["component", "diagnosis_group", "outcome"],
    )
    ct = adjusted.loc[adjusted["component"].eq("CT")]
    expected_pearson = benjamini_hochberg(ct["pearson_p"])
    expected_spearman = benjamini_hochberg(ct["spearman_p"])
    assert np.allclose(
        ct["pearson_fdr_across_modules"], expected_pearson, equal_nan=True
    )
    assert np.allclose(
        ct["spearman_fdr_across_modules"], expected_spearman, equal_nan=True
    )
    ts = adjusted.loc[adjusted["component"].eq("TS")]
    assert set(ts["pearson_fdr_module_family_n"]) == {2}
    assert set(ts["spearman_fdr_module_family_n"]) == {3}


def test_grouped_fdr_corrects_modules_separately_within_each_group_level() -> None:
    frame = pd.DataFrame(
        {
            "module": [1, 2, 3, 1, 2, 3],
            "component": ["CT"] * 6,
            "outcome": ["cogn_global"] * 6,
            "grouping_variable": ["clusters"] * 6,
            "grouping_level": [1] * 3 + [4] * 3,
            "pearson_p": [0.001, 0.020, 0.90, 0.040, 0.060, 0.80],
            "spearman_p": [0.002, 0.030, 0.70, 0.010, 0.20, 0.90],
        }
    )
    adjusted = add_across_module_fdr(
        frame,
        family_columns=[
            "component", "outcome", "grouping_variable", "grouping_level",
        ],
    )

    for level in (1, 4):
        group = adjusted.loc[adjusted["grouping_level"].eq(level)]
        assert np.allclose(
            group["pearson_fdr_across_modules"],
            benjamini_hochberg(group["pearson_p"]),
            equal_nan=True,
        )
        assert np.allclose(
            group["spearman_fdr_across_modules"],
            benjamini_hochberg(group["spearman_p"]),
            equal_nan=True,
        )
        assert group["pearson_fdr_module_family_n"].eq(3).all()
        assert group["spearman_fdr_module_family_n"].eq(3).all()


def test_real_aggregate_fdr_families_contain_each_module_definition() -> None:
    for module_set, expected_modules in (("full_cohort", 154), ("control_derived", 186)):
        stored = load_aggregate_statistics(
            "control_anchored",
            module=None,
            phenotype="cogn_global",
            metric_family="connectivity",
            module_set=module_set,
        )
        control = stored.loc[stored["diagnosis_group"].eq("Control")].copy()
        frame = pd.DataFrame(
            {
                "module": control["module"],
                "component": "CT",
                "diagnosis_group": "Control",
                "outcome": "cogn_global",
                "pearson_p": control["p_rint_CT"],
                "spearman_p": control["p_spearman_CT"],
            }
        )
        adjusted = add_across_module_fdr(
            frame,
            family_columns=["component", "diagnosis_group", "outcome"],
        )
        assert adjusted["module"].nunique() == expected_modules
        assert set(adjusted["pearson_fdr_module_family_n"]) == {
            int(frame["pearson_p"].notna().sum())
        }
        assert set(adjusted["spearman_fdr_module_family_n"]) == {
            int(frame["spearman_p"].notna().sum())
        }


def test_grouped_correlations_enforce_minimum_size_and_report_reason() -> None:
    frame = pd.DataFrame(
        {
            "module": [1] * 12,
            "group": ["large"] * 8 + ["small"] * 4,
            "metric_value": np.arange(12, dtype=float),
            "outcome": np.arange(12, dtype=float) * 2,
        }
    )
    result = calculate_correlations(
        frame, ["module", "group"], ["outcome"], min_group_n=5
    ).set_index("group")
    assert bool(result.loc["large", "eligible"])
    assert np.isclose(result.loc["large", "spearman_rho"], 1.0)
    assert not bool(result.loc["small", "eligible"])
    assert np.isnan(result.loc["small", "pearson_r"])
    assert result.loc["small", "unavailable_reason"] == "n < 5"


def test_generic_categorical_association_supports_string_levels() -> None:
    frame = pd.DataFrame(
        {
            "module": [1] * 18,
            "component": ["CT"] * 18,
            "apoe_genotype": ["ε2/ε3"] * 6 + ["ε3/ε3"] * 7 + ["ε4/ε4"] * 5,
            "metric_value": [*range(6), *range(10, 17), *range(20, 25)],
        }
    )
    result = calculate_categorical_associations(
        frame, ["module", "component"],
        category_column="apoe_genotype", min_group_n=6,
    ).iloc[0]
    assert set(result["levels_tested"].split(",")) == {"ε2/ε3", "ε3/ε3"}
    assert result["levels_excluded_small_n"] == "ε4/ε4"
    assert result["k_tested"] == 2
    assert 0 <= result["epsilon_squared"] <= 1
