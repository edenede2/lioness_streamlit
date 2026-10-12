from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_muscle_lioness.py"
SPEC = importlib.util.spec_from_file_location("run_muscle_lioness", SCRIPT)
assert SPEC and SPEC.loader
MUSCLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MUSCLE
SPEC.loader.exec_module(MUSCLE)


def _adjacency(matrix: np.ndarray, tissues: np.ndarray) -> np.ndarray:
    upper_i, upper_j, powers, _ = MUSCLE.edge_metadata(tissues)
    correlation = np.corrcoef(matrix, rowvar=False)[upper_i, upper_j]
    correlation = np.nan_to_num(correlation, nan=0.0)
    return MUSCLE.signed_power(correlation, powers)


def _computed_weights(
    matrix: np.ndarray, diagnoses: np.ndarray, method: str
) -> np.ndarray:
    tissues = np.array(["VH", "VH", "SMA", "QUAD", "QUAD"])
    upper_i, upper_j, powers, _ = MUSCLE.edge_metadata(tissues)
    chunks = list(
        MUSCLE.method_weights(
            matrix,
            diagnoses,
            method,
            upper_i,
            upper_j,
            powers,
            chunk_size=3,
        )
    )
    result = np.empty((len(matrix), len(upper_i)))
    for donor_indices, weights in chunks:
        result[donor_indices] = weights
    return result


def test_standard_lioness_matches_direct_recalculation() -> None:
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(9, 5))
    tissues = np.array(["VH", "VH", "SMA", "QUAD", "QUAD"])
    diagnoses = np.array(["Control"] * 4 + ["MCI"] * 2 + ["AD"] * 3)
    observed = _computed_weights(matrix, diagnoses, "standard")
    reference = _adjacency(matrix, tissues)
    expected = np.vstack(
        [
            len(matrix) * reference
            - (len(matrix) - 1) * _adjacency(np.delete(matrix, donor, axis=0), tissues)
            for donor in range(len(matrix))
        ]
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-10)


def test_control_anchored_leave_one_out_and_add_one_match_direct() -> None:
    rng = np.random.default_rng(1042)
    matrix = rng.normal(size=(10, 5))
    tissues = np.array(["VH", "VH", "SMA", "QUAD", "QUAD"])
    diagnoses = np.array(["Control"] * 5 + ["MCI"] * 2 + ["AD"] * 3)
    observed = _computed_weights(matrix, diagnoses, "control_anchored")
    controls = matrix[:5]
    reference = _adjacency(controls, tissues)
    expected = []
    for donor in range(len(matrix)):
        if diagnoses[donor] == "Control":
            scored = (
                len(controls) * reference
                - (len(controls) - 1)
                * _adjacency(np.delete(controls, donor, axis=0), tissues)
            )
        else:
            scored = (
                (len(controls) + 1)
                * _adjacency(np.vstack([controls, matrix[donor]]), tissues)
                - len(controls) * reference
            )
        expected.append(scored)
    np.testing.assert_allclose(
        observed, np.vstack(expected), rtol=1e-10, atol=1e-10
    )


def test_block_powers_match_control_partition_parameters() -> None:
    tissues = np.array(["VH", "SMA", "QUAD"])
    _, _, powers, components = MUSCLE.edge_metadata(tissues)
    assert powers.tolist() == [8.0, 10.0, 10.0]
    assert components["CT_VH__SMA"].tolist() == [0]
    assert components["CT_VH__QUAD"].tolist() == [1]
    assert components["CT_SMA__QUAD"].tolist() == [2]

