import numpy as np
import json
import pandas as pd
import pytest

from app_helpers.validation_datasets import dataset, manifest_matches
from app_helpers.validation_scoring import score_component
from app_helpers import camel_validation as V, validation_review as R


def test_datasets_are_isolated_and_have_audited_grains():
    assert dataset("full_cohort", "standard").expected_rows() == 7_795_942
    assert dataset("muscle_control_l4").expected_rows()+dataset("muscle_all_donors_l4").expected_rows() == 509_056
    assert dataset("muscle_control_l4").framework("control_anchored") == "CAMEL"
    assert dataset("muscle_all_donors_l4").framework("control_anchored") == "LIONESS"
    with pytest.raises(ValueError): dataset("muscle_control_l4", "standard")
    assert manifest_matches({}, "control_derived", "control_anchored")
    assert not manifest_matches({}, "full_cohort", "control_anchored")


@pytest.mark.parametrize("method", ["standard", "control_anchored"])
def test_shared_scoring_direct_formulas_and_original_only_maps(method):
    rng = np.random.default_rng(42)
    original = rng.normal(size=(15, 6))
    # Preserve a repeated assignment as one additional positional node.
    original[:, 5] = original[:, 2]
    new = rng.normal(size=(3, 6)); new[:, 5] = new[:, 2]
    ref = original if method == "standard" else original[:8]
    positions = np.arange(15) if method == "standard" else np.r_[np.arange(8), np.full(7, -1)]
    rows, cols = np.tril_indices(6, -1)
    ad = np.arange(len(rows)) % 3 == 0; control = np.arange(len(rows)) % 3 == 1
    masks = {"all": np.ones(len(rows), bool), "ad": ad, "control": control, "either": ad|control}
    kwargs = dict(reference=ref, original=original, reference_positions=positions,
        rows=rows, columns=cols, masks=masks, power=2, module=1918, component="CT_AC__DLPFC",
        n_features=6, original_ids=[f"S{i}" for i in range(15)], validation_ids=[f"V{i}" for i in range(3)])
    a, b, maps, checks = score_component(validation=new, edge_batch=4, **kwargs)
    _, changed, other_maps, _ = score_component(validation=new*100, edge_batch=4, **kwargs)
    assert maps.equals(other_maps)
    assert checks and max(c["max_abs_edge_error"] for c in checks) < 1e-10
    assert not b.signed_weight_sum.equals(changed.signed_weight_sum)
    parts = {r: b.loc[b.differential_edge_rule.eq(r)].reset_index(drop=True) for r in masks}
    np.testing.assert_allclose(parts["ad"].signed_weight_sum+parts["control"].signed_weight_sum, parts["either"].signed_weight_sum)
    np.testing.assert_allclose(parts["either"].absolute_weight_sum,
        parts["either"].positive_weight_sum+parts["either"].negative_weight_magnitude)
    assert a.n_features.eq(6).all() and len(b) == 12


@pytest.mark.parametrize("method", ["standard", "control_anchored"])
def test_bound_readers_never_mix_methods_or_legacy_partitions(tmp_path,monkeypatch,method):
    monkeypatch.setenv("PARTITION_VALIDATION_DATA_ROOT",str(tmp_path))
    identity=dict(module_set="full_cohort",network_method=method,status="complete",completed_rules=["all"],backend_verified=True)
    (tmp_path/"manifest.json").write_text(json.dumps(identity))
    (tmp_path/"review").mkdir(); (tmp_path/"review/manifest.json").write_text(json.dumps(identity))
    source=V.for_dataset("full_cohort",method)
    assert source.available() and source.root()==tmp_path
    assert R.for_dataset("full_cohort",method).manifest()["network_method"]==method
    other="standard" if method=="control_anchored" else "control_anchored"
    with pytest.raises(ValueError,match="different partition or method"):
        V.for_dataset("full_cohort",other).available()
    assert R.for_dataset("full_cohort",other).manifest()=={}
    assert ("full_cohort",other) not in V.available_datasets()
    assert source.config.framework(method)=="LIONESS"


def test_positional_expression_selection_does_not_multiply_repeated_rows():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    from run_partition_validation import matrix_for
    assignment=pd.DataFrame({"public_tissue":["AC","AC","DLPFC","AC"],"Gene Symbol":["A","A","B","C"]})
    state={"matrix":{"AC":pd.DataFrame([[1.,2.],[3.,4.]],index=["one","two"],columns=["A","C"]),
                     "DLPFC":pd.DataFrame([[5.],[6.]],index=["one","two"],columns=["B"])}}
    np.testing.assert_array_equal(matrix_for(assignment,["two","one"],state),[[3,3,6,4],[1,1,5,2]])
