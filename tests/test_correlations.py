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
