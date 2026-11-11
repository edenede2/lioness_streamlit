"""Incomplete-region MDC scatters and edge-count-only display exclusions."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app_helpers.validation_mdc_plots import (
    component_frame, paired_components, comparison_figure, entropy_figure,
    filter_retained_edges, filter_wide_components,
)

LABELS = {"TS_DLPFC":"DLPFC", "CT_DLPFC__PCGBA23":"DLPFC–PCG"}


def fixture():
    rows=[]
    for module in (1,2,3):
        for component in LABELS:
            value = (1e8 if module==2 else 1000 if module==3 else 1.2) if component.startswith("CT_") else .7+module*.2
            rows.append(dict(module=module, component=component, component_label=LABELS[component],
                mdc=value, log2_mdc=np.log2(value), n_retained_edges=1 if module==2 and component.startswith("CT_") else 500,
                n_possible_edges=500, n_ad=77, n_control=65, directional_fdr=.01 if module==1 else np.nan,
                differential_edge_rule="all", inferential=True))
    details=pd.DataFrame(dict(module=[1,2,3],module_size=[30,40,50],cluster_type=["CT"]*3,
        tissue_entropy=[.5,.8,1.],tissue_entropy_normalized=[.3,.5,.8]))
    return pd.DataFrame(rows),details


def test_edge_filter_removes_small_support_not_large_ratios_or_saved_q():
    frame,_=fixture(); original=frame.copy(deep=True)
    pd.testing.assert_frame_equal(filter_retained_edges(frame,0),frame)
    filtered=filter_retained_edges(frame,10)
    assert len(filtered)==5
    assert filtered.loc[filtered.component.str.startswith("CT_"),"module"].tolist()==[1,3]
    assert 1000 in filtered.mdc.tolist()  # no implicit MDC clipping
    pd.testing.assert_frame_equal(frame,original)
    assert filtered.set_index(["module","component"]).directional_fdr.equals(
        frame.set_index(["module","component"]).directional_fdr.loc[filtered.set_index(["module","component"]).index])
    wide=paired_components(frame,*LABELS)
    wide["n_retained_edges_ts"]=wide.n_retained_edges_ts
    wide["n_retained_edges_ct"]=wide.n_retained_edges_ct
    wide["n_edges"]=wide.n_retained_edges_ts+wide.n_retained_edges_ct
    result=filter_wide_components(wide,10)
    assert pd.isna(result.loc[result.module.eq(2),"mdc_ct"]).all()
    assert result.loc[result.module.eq(2),"mdc_ts"].notna().all()
    pd.testing.assert_series_equal(result.directional_fdr_ct,wide.directional_fdr_ct)
    historical=wide.drop(columns=["n_edges", "n_retained_edges_ts", "n_retained_edges_ct"]).assign(
        n_ts_edges=wide.n_retained_edges_ts, n_ct_edges=wide.n_retained_edges_ct)
    filtered_wide=filter_wide_components(historical,10)
    assert filtered_wide.loc[filtered_wide.module.eq(2),"mdc_ts"].notna().all()
    assert filtered_wide.loc[filtered_wide.module.eq(2),"mdc_ct"].isna().all()
    assert filter_retained_edges(frame,1000).empty
    adapted=frame.rename(columns={"n_retained_edges":"n_edges"})
    pd.testing.assert_frame_equal(filter_retained_edges(adapted,10),filtered.rename(columns={"n_retained_edges":"n_edges"}))


@pytest.mark.parametrize("scale",["raw","log2"])
def test_scatter_values_and_component_names_match_saved_data(scale):
    frame,details=fixture(); frame=filter_retained_edges(frame,10)
    fig=comparison_figure(frame,*LABELS,LABELS,2,.05,scale,"Control-derived L4")
    regular=[t for t in fig.data if t.customdata is not None]
    assert sum(len(t.x) for t in regular)==2
    assert {int(str(r[0])[1:]) for t in regular for r in t.customdata}=={1,3}
    assert "DLPFC–PCG" in fig.layout.yaxis.title.text
    assert "pooled" not in fig.layout.title.text
    assert not any((t.meta or {}).get("export_role")=="selected_module" for t in fig.data)
    for component,scope in zip(LABELS,("ts","ct")):
        fig=entropy_figure(frame,component,scope,LABELS,details,1,.05,scale,"Control-derived L4")
        points=[t for t in fig.data if t.mode=="markers"]
        expected=component_frame(frame,component,scope,details).set_index("module")
        for trace in points:
            for x,y,row in zip(trace.x,trace.y,trace.customdata):
                module=int(str(row[0])[1:])
                assert x==expected.loc[module,"tissue_entropy_normalized"]
                assert y==pytest.approx(expected.loc[module,("mdc_" if scale=="raw" else "log2_mdc_")+scope])
                assert row[8]==expected.loc[module,"n_retained_edges_"+scope]
        assert LABELS[component]+" MDC" in fig.layout.title.text
        assert any(t.name=="OLS trend" for t in fig.data) if len(expected)>=3 else True


@pytest.mark.parametrize("family,method",[("control_derived","control_anchored"),("full_cohort","control_anchored"),("full_cohort","standard")])
def test_validation_page_renders_same_three_scatters_and_filtered_download(monkeypatch,family,method):
    from app_helpers import camel_validation as V, data as D
    frame,details=fixture()
    def bound(module_set,network_method):
        assert (module_set,network_method)==(family,method)
        return SimpleNamespace(config=SimpleNamespace(labels=LABELS),mdc=lambda rule:frame.copy())
    monkeypatch.setattr(V,"for_dataset",bound)
    monkeypatch.setattr(D,"load_module_details",lambda module_set:details)
    app=AppTest.from_string(f'''
from app_helpers.camel_validation_ui import render_mdc
from app_helpers.figure_export import render_exportable_plot
render_mdc(1,"all",render_exportable_plot,module_set={family!r},network_method={method!r})
''',default_timeout=30).run()
    assert not app.exception
    assert len(app.get("plotly_chart"))==4  # selected bars + component scatter + two entropy plots
    for p in app.get("plotly_chart"):
        spec=json.loads(p.proto.spec)
        assert spec["layout"]["meta"]["module_set"]==family
        assert spec["layout"]["meta"]["method"]==method
    app=app.number_input(key="mdc_minimum_edges").set_value(10).run()
    assert not app.exception
    shown=next(d.value for d in app.dataframe if "n_retained_edges" in d.value)
    pd.testing.assert_frame_equal(shown,filter_retained_edges(frame,10))
    app=app.number_input(key="mdc_minimum_edges").set_value(1000).run()
    assert not app.exception and not app.get("plotly_chart")


@pytest.mark.parametrize("family,method",[("control_derived","control_anchored"),("full_cohort","standard"),("full_cohort","control_anchored")])
def test_real_app_mdc_route_restores_scatters_and_edge_filter(family,method):
    from pathlib import Path
    from app_helpers import camel_validation as V
    from ui_interactions import apply_driver
    source=V.for_dataset(family,method)
    if not (source.root()/"mdc/resolved.parquet").is_file():
        pytest.skip("Saved validation MDC unavailable locally; no remote download in test")
    app=apply_driver(AppTest.from_file(Path(__file__).resolve().parents[1]/"streamlit_app.py",default_timeout=90))
    app.query_params.update(view="MDC",screen_partition=family,screen_method=method,screen_module="747" if family=="control_derived" else "1918")
    app=app.run()
    assert not app.exception
    app=app.number_input(key="mdc_minimum_edges").set_value(50).run()
    assert not app.exception
    app=app.radio(key="analysis_cohort_scope").set_value("incomplete_region_validation").run()
    assert not app.exception and len(app.get("plotly_chart"))==4
    assert app.radio(key="validation_mdc_view").value=="MDC scatter plots"
    table=next(d.value for d in app.dataframe if "n_retained_edges" in d.value)
    assert table.n_retained_edges.ge(50).all()
