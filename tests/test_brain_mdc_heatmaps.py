from __future__ import annotations
import numpy as np
import pytest
from scripts.build_brain_mdc_heatmaps import paired_matrix,ratio,figure


def test_split_triangles_use_one_order_and_correct_diagnosis_without_diagonal():
    ad=np.array([[0.,.1,.8],[.1,0.,.3],[.8,.3,0.]])
    control=np.array([[0.,.4,.2],[.4,0.,.7],[.2,.7,0.]])
    combined,order=paired_matrix(ad,control)
    assert order.tolist()==[2,0,1]
    a=ad[np.ix_(order,order)];c=control[np.ix_(order,order)]
    u=np.triu_indices(3,1);l=np.tril_indices(3,-1)
    np.testing.assert_array_equal(combined[u],np.abs(a[u]))
    np.testing.assert_array_equal(combined[l],np.abs(c[l]))
    assert np.isnan(np.diag(combined)).all()


def test_signed_weights_display_absolute_and_absent_ct_is_not_zero():
    ad=np.array([[0.,-.5],[-.5,0.]])
    control=np.array([[0.,.25],[.25,0.]])
    combined,_=paired_matrix(ad,control)
    assert combined[0,1]==.5 and combined[1,0]==.25
    stats=ratio(ad,control,np.array([0,0]))
    assert stats['total']==stats['ts']==2 and np.isnan(stats['ct'])


def test_invalid_networks_rejected_and_figure_labels_use_dlpf_not_mf():
    with pytest.raises(ValueError):paired_matrix(np.zeros((2,3)),np.zeros((2,3)))
    ad=np.array([[0.,.5],[.5,0.]])
    combined,order=paired_matrix(ad,ad)
    fig,cap=figure(combined,np.array([0,1]),order,'Control-derived L4 modules',7,{'total':1.,'ts':np.nan,'ct':1.})
    import matplotlib.pyplot as plt
    assert cap==.5
    labels=' '.join(t.get_text() for t in fig.texts)
    assert 'Upper triangle: AD' in labels and 'Lower triangle: Control' in labels
    legend=' '.join(t.get_text() for t in fig.legends[0].get_texts())
    assert 'DLPFC' in legend and 'MF' not in legend
    plt.close(fig)


def test_published_brain_catalog_routes_new_images_without_mixing_definitions(tmp_path, monkeypatch):
    import json
    from app_helpers import data
    for key in ('full_cohort','control_derived'):
        root=tmp_path/key
        monkeypatch.setitem(data.MODULE_SET_DIRS,key,root)
        folder=root/'mdc_split_heatmaps/brain_20261014'
        folder.mkdir(parents=True)
        (folder/'M121.png').write_bytes(b'fixture')
        (folder.parent/'catalog.json').write_text(json.dumps(dict(status='complete',module_set=key,image_directory='brain_20261014')))
        assert data.mdc_split_heatmap_available(key,121)
        assert data.load_mdc_split_heatmap(key,121)==folder/'M121.png'
        assert not data.mdc_split_heatmap_available(key,121,'ad_control_discovery_fdr05')
    assert data.load_mdc_split_heatmap('full_cohort',121)!=data.load_mdc_split_heatmap('control_derived',121)


def test_brain_catalog_rejects_path_traversal(tmp_path,monkeypatch):
    import json
    from app_helpers import data
    monkeypatch.setitem(data.MODULE_SET_DIRS,'full_cohort',tmp_path)
    folder=tmp_path/'mdc_split_heatmaps'
    folder.mkdir()
    (folder/'catalog.json').write_text(json.dumps(dict(status='complete',module_set='full_cohort',image_directory='../other')))
    with pytest.raises(ValueError,match='Unsafe'):
        data.mdc_split_heatmap_available('full_cohort',121)
