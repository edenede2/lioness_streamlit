from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_muscle_m5_mdc.py"
SPEC = importlib.util.spec_from_file_location("run_muscle_m5_mdc", SCRIPT)
assert SPEC and SPEC.loader
MDC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MDC
SPEC.loader.exec_module(MDC)


def test_scope_edges_partition_total_into_ts_and_ct() -> None:
    tissues = np.array(["VH", "VH", "SMA", "QUAD"])
    scopes = MDC.scope_edges(tissues)
    total = set(zip(*scopes["total"], strict=True))
    ts = set(zip(*scopes["TS"], strict=True))
    ct = set(zip(*scopes["CT"], strict=True))
    assert len(total) == 6
    assert ts.isdisjoint(ct)
    assert ts | ct == total
    assert len(scopes["TS_VH"][0]) == 1
    assert len(scopes["CT_VH__SMA"][0]) == 2


def test_observed_mdc_uses_ad_over_control_absolute_m5_weight() -> None:
    tissues = np.array(["VH", "VH", "SMA", "QUAD"])
    ad = np.zeros((4, 4), dtype=float)
    control = np.zeros((4, 4), dtype=float)
    upper_i, upper_j = np.triu_indices(4, k=1)
    ad[upper_i, upper_j] = np.array([2, -4, 6, 8, -10, 12], dtype=float)
    control[upper_i, upper_j] = np.array([1, 2, -3, 4, 5, -6], dtype=float)
    rows = MDC.summarize_module(
        7, np.arange(4, dtype=np.int64), tissues, ad, control
    )
    by_scope = {row["scope"]: row for row in rows}
    assert len(rows) == len(MDC.SCOPES)
    assert by_scope["total"]["mdc_raw"] == 2.0
    assert by_scope["total"]["mdc_log2"] == 1.0
    assert by_scope["TS"]["mdc_raw"] == 2.0
    assert by_scope["CT"]["mdc_raw"] == 2.0
    assert np.isnan(by_scope["TS_SMA"]["mdc_raw"])

