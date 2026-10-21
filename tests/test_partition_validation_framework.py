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


def test_approved_legacy_node_conventions_reproduce_unique_and_repeated_networks():
    from app_helpers.camel import feature_values, frozen_edge_weights
    rng=np.random.default_rng(123)
    ref=rng.normal(size=(20,4)); new=rng.normal(size=(3,4))
    ref=np.c_[ref,ref[:,1]]; new=np.c_[new,new[:,1]]
    rows,cols=np.tril_indices(5,-1); unique=(rows!=4)&(cols!=4)
    masks={"all":unique,"effect":np.ones(len(rows),bool)}
    _, scored, _, _=score_component(reference=ref,original=ref,validation=new,
        reference_positions=np.arange(20),rows=rows,columns=cols,masks=masks,power=3,
        module=188,component="TS_AC",n_features={"all":4,"effect":5},
        possible_edges={"all":6,"effect":10},node_conventions={"all":"unique","effect":"positional"},
        source_n_features=5,original_ids=[f"S{i}" for i in range(20)],validation_ids=[f"V{i}" for i in range(3)])
    for rule,size,edges in [("all",4,6),("effect",5,10)]:
        part=scored.loc[scored.differential_edge_rule.eq(rule)]
        ri,ci=np.tril_indices(size,-1)
        weights=frozen_edge_weights(ref[:,:size],new[:,:size],ri,ci,np.full(len(ri),3))
        np.testing.assert_allclose(feature_values(part,"connectivity","standard_pruned"),2*weights.sum(axis=1)/size,atol=1e-12)
        assert part.n_features.eq(size).all() and part.n_possible_edges.eq(edges).all()
        assert part.n_retained_edges.eq(edges).all() and part.n_pruned_edges.eq(0).all()
        assert part.cross_filter_node_convention_warning.all()


@pytest.mark.parametrize("method", ["standard", "control_anchored"])
def test_validation_scores_and_maps_are_independent_of_other_validation_donors(method):
    rng=np.random.default_rng(6042)
    original=rng.normal(size=(18,5)); validation=rng.normal(size=(4,5))
    rows,columns=np.tril_indices(5,-1)
    reference=original if method=="standard" else original[:8]
    kwargs=dict(reference=reference,original=original,
        reference_positions=np.arange(18) if method=="standard" else np.r_[np.arange(8),np.full(10,-1)],
        rows=rows,columns=columns,masks={"all":np.ones(len(rows),bool)},power=2,
        module=1918,component="CT_AC__DLPFC",n_features=5,
        original_ids=[f"S{i}" for i in range(18)])
    _,baseline,maps,_=score_component(validation=validation,validation_ids=["V0","V1","V2","V3"],**kwargs)
    for indices in ([2,0,3,1],[0],[3,0]):
        _,changed,changed_maps,_=score_component(validation=validation[indices],validation_ids=[f"V{i}" for i in indices],**kwargs)
        pd.testing.assert_frame_equal(changed_maps.drop(columns=["x","z"]),maps.drop(columns=["x","z"]),
                                      check_exact=False,rtol=0,atol=1e-12)
        # BLAS reduction tiles can change roundoff when batch sizes differ;
        # the fitted mapping must be invariant to the numerical gate tolerance.
        for column in ("x","z"):
            for left,right in zip(changed_maps[column],maps[column]):
                np.testing.assert_allclose(left,right,rtol=0,atol=1e-12)
        pd.testing.assert_frame_equal(baseline.loc[baseline.sample_id.eq("V0")].reset_index(drop=True),
                                      changed.loc[changed.sample_id.eq("V0")].reset_index(drop=True),
                                      check_exact=False,rtol=0,atol=1e-12)
    # Diagnosis labels are deliberately absent from the scoring API. Every
    # validation donor follows add-one scoring, including validation Controls.
