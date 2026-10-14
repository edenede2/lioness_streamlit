from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[1]
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from models.differential_edges import EffectMaskSpec, effect_size_mask
from scripts.run_muscle_effect_size_edges import build_split, scope_context


def test_muscle_public_bundle_exposes_control_reference_only() -> None:
    effect_manifest = json.loads(
        (APP_ROOT / "data/muscle_control_l4/effect_size/manifest.json").read_text()
    )
    module_manifest = json.loads(
        (APP_ROOT / "data/muscle_control_l4/module_set_manifest.json").read_text()
    )
    app_manifest = json.loads((APP_ROOT / "data/data_manifest.json").read_text())
    assert effect_manifest["completed_networks"] == ["lioness/control_anchored"]
    assert module_manifest["methods"] == ["control_anchored"]
    assert app_manifest["module_sets"]["muscle_control_l4"]["methods"] == [
        "control_anchored"
    ]
    assert not (
        APP_ROOT / "data/muscle_control_l4/effect_size/lioness/standard"
    ).exists()
    for filename, method_column in {
        "aggregate_plot_data.parquet": "lioness_method",
        "aggregate_statistics.parquet": "lioness_method",
        "cluster_association_statistics.parquet": "network_method",
        "resolved_plot_data.parquet": "lioness_method",
        "resolved_statistics.parquet": "lioness_method",
    }.items():
        methods = pd.read_parquet(
            APP_ROOT / "data/muscle_control_l4" / filename,
            columns=[method_column],
        )[method_column].dropna().unique().tolist()
        assert methods == ["control_anchored"]


def test_muscle_effect_directions_form_exact_union() -> None:
    effect = np.asarray([-0.8, -0.4, -0.39, 0.0, 0.39, 0.4, 0.9, np.nan])
    edge_index = np.arange(len(effect))
    spec = EffectMaskSpec("hedges_g", "fixed", 0.40)
    ad = effect_size_mask(effect, edge_index, spec, direction="ad_higher")
    control = effect_size_mask(effect, edge_index, spec, direction="control_higher")
    either = effect_size_mask(effect, edge_index, spec, direction="either")
    assert np.array_equal(either, ad | control)
    assert np.flatnonzero(ad).tolist() == [5, 6]
    assert np.flatnonzero(control).tolist() == [0, 1]


def test_public_gene_sanitization_never_retains_ensembl_self_mapping(monkeypatch):
    from scripts import build_muscle_effect_size_public_data as build
    monkeypatch.setattr(build, 'gene_symbol_mapping', lambda: {'ENSG00000066379': 'ENSG00000066379'})
    original = pd.DataFrame({'gene_a': ['ENSG00000066379'], 'gene_b': ['SMAD4'],
                             'discovery_hedges_g': [0.413], 'module': [1000]})
    converted = build.sanitize(original, {})
    assert converted.gene_a.tolist() == ['ZNRD1']
    pd.testing.assert_frame_equal(original.drop(columns=['gene_a']), converted.drop(columns=['gene_a']))


def test_muscle_scope_context_recombines_ts_and_ct() -> None:
    tissues = np.asarray(["VH", "VH", "SMA", "QUAD"], dtype=object)
    upper_i, upper_j = np.triu_indices(len(tissues), k=1)
    names = []
    for left, right in zip(tissues[upper_i], tissues[upper_j], strict=True):
        if left == right:
            names.append(f"TS_{left}")
        else:
            ordered = sorted((str(left), str(right)), key=("VH", "SMA", "QUAD").index)
            names.append(f"CT_{ordered[0]}__{ordered[1]}")
    component_edges = {
        component: np.flatnonzero(np.asarray(names) == component)
        for component in (
            "TS_VH", "TS_SMA", "TS_QUAD", "CT_VH__SMA",
            "CT_VH__QUAD", "CT_SMA__QUAD",
        )
    }
    context = scope_context(tissues, upper_i, upper_j, component_edges)
    assert context.scopes["TS"].sum() == sum(
        context.scopes[name].sum() for name in component_edges if name.startswith("TS_")
    )
    assert context.scopes["CT"].sum() == sum(
        context.scopes[name].sum() for name in component_edges if name.startswith("CT_")
    )


def test_muscle_discovery_split_is_deterministic_and_keeps_mci_external(tmp_path) -> None:
    diagnoses = ("AD",) * 10 + ("Control",) * 10 + ("MCI",) * 4
    projids = tuple(str(index + 1) for index in range(len(diagnoses)))
    donors = tuple(f"R-{value}" for value in projids)
    metadata = pd.DataFrame(
        {
            "ProjID": projids,
            "msex": [index % 2 for index in range(len(diagnoses))],
        }
    )
    path = tmp_path / "metadata.csv"
    metadata.to_csv(path, index=False)
    cohort = SimpleNamespace(donors=donors, projids=projids, diagnoses=diagnoses)
    first = build_split(cohort, path, seed=42)
    second = build_split(cohort, path, seed=42)
    assert first["ad_control_split"].tolist() == second["ad_control_split"].tolist()
    assert first.loc[first["diagnosis_group"].eq("MCI"), "ad_control_split"].eq(
        "MCI_external"
    ).all()
    assert set(first.loc[first["diagnosis_group"].ne("MCI"), "ad_control_split"]) == {
        "Discovery", "Validation"
    }
