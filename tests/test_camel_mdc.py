import numpy as np
from app_helpers.camel_mdc import standardized, adjacency_edges, mdc_from_edges, empirical


def test_mdc_signed_powers_direct_ratio_and_empty_masks():
    rng=np.random.default_rng(42)
    a=rng.normal(size=(23,7)); c=rng.normal(size=(19,7))
    ri,ci=np.tril_indices(7,-1)
    masks=np.array([np.ones(len(ri)),np.arange(len(ri))%2,np.zeros(len(ri))])
    for power in (2,3):
        ae=adjacency_edges(standardized(a),ri,ci,power)
        ce=adjacency_edges(standardized(c),ri,ci,power)
        expected=np.corrcoef(a,rowvar=False)[ri,ci]
        np.testing.assert_allclose(ae,np.sign(expected)*np.abs(expected)**power,atol=1e-14)
        ratio,ad,control=mdc_from_edges(ae,ce,masks)
        np.testing.assert_allclose(ratio[:2],[np.abs(ae[m.astype(bool)]).mean()/np.abs(ce[m.astype(bool)]).mean() for m in masks[:2]])
        assert np.isnan(ratio[-1]) and np.isnan(ad[-1]) and np.isnan(control[-1])
    empty=mdc_from_edges(np.array([]),np.array([]),np.zeros((3,0)))
    assert all(np.isnan(arr).all() for arr in empty)


def test_mdc_empirical_null_is_plus_one_and_never_fills_missing_inference():
    observed=np.array([2.,.5,np.nan,1.])
    draws=np.array([[1.,2.,3.],[1.,2.,3.],[1.,2.,3.],[np.nan]*3])
    p=empirical(observed,draws,True)
    np.testing.assert_allclose(p[:2],[.75,1.])
    assert np.isnan(p[2:]).all()


def test_mdc_shuffle_restart_seed_is_independent_of_execution_order():
    def null_draw(i):
        rng=np.random.default_rng(np.random.SeedSequence([42,771,3,i]))
        z=standardized(np.arange(40).reshape(10,4))
        return np.take_along_axis(z,np.argsort(rng.random(z.shape),axis=0),axis=0)
    first=[null_draw(i) for i in range(7)]
    resumed=[null_draw(i) for i in range(4,7)]
    for a,b in zip(first[4:],resumed): np.testing.assert_array_equal(a,b)
