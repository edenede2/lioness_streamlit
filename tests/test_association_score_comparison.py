from dataclasses import replace
from pathlib import Path
import io
import os

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr
from PIL import Image
from streamlit.testing.v1 import AppTest

from app_helpers.association_score_comparison import (
    ScoreSeries, adjacent_series, common_case_scores, comparison_statistics,
    combine_panels, load_series, score_comparison_figure, score_series_catalog,
)
from app_helpers.figure_export import PlotExportSpec, PlotContentConfig, preset_config
from app_helpers.figure_export.adapters import build_artifact
from app_helpers.figure_export.config import figure_panels

APP = Path(__file__).resolve().parents[1]


def fixture_data():
    catalog = score_series_catalog(["AC", "DLPFC", "PCG"], {})
    metadata = pd.DataFrame({"sample_id":[f"S{i}" for i in range(15)],
        "outcome":np.arange(15,dtype=float), "adjusted":np.arange(15,dtype=float)*0.3,
        "diagnosis_group":["Control"]*8+["AD"]*7})
    columns = {s.key:pd.Series(np.arange(15,dtype=float)*(1 if i%2 else -1),
                             index=metadata.sample_id) for i,s in enumerate(catalog)}
    columns[catalog[0].key].loc["S1"] = np.nan
    columns[catalog[1].key].loc["S2"] = np.inf
    metadata.loc[3,"outcome"] = np.nan
    metadata.loc[4,"adjusted"] = np.nan
    return catalog,metadata,columns


def test_common_case_intersection_signs_n_and_no_display_only_fdr():
    series,metadata,columns=fixture_data()
    common=common_case_scores(metadata,columns,series,["outcome"],121)
    assert common.groupby("component").sample_id.nunique().eq(12).all()
    assert all(set(g.sample_id)==set(common.sample_id) for _,g in common.groupby("component"))
    stat=comparison_statistics(common,"outcome")
    assert stat.n.eq(12).all()
    assert stat.spearman_rho.min()==pytest.approx(-1)
    assert stat.spearman_rho.max()==pytest.approx(1)
    assert not any("fdr" in c for c in stat.columns)
    for key,part in common.groupby("component"):
        row=stat.set_index("component").loc[key]
        assert row.pearson_r==pytest.approx(pearsonr(part.metric_value,part.outcome).statistic)
        assert row.spearman_rho==pytest.approx(spearmanr(part.metric_value,part.outcome).statistic)


def test_optional_adjusted_outcome_and_donor_filters_enter_shared_intersection():
    series,metadata,columns=fixture_data()
    selected=metadata.loc[metadata.diagnosis_group.eq("AD")]
    common=common_case_scores(selected,columns,series,["outcome","adjusted"],121)
    assert common.sample_id.nunique()==7
    assert common.diagnosis_group.eq("AD").all()
    both=common_case_scores(metadata,columns,series,["outcome","adjusted"],121)
    assert both.sample_id.nunique()==11
    with pytest.raises(ValueError):
        common_case_scores(pd.concat([metadata,metadata.iloc[:1]]),columns,series,["outcome"],121)


def test_empty_and_constant_scores_are_not_encoded_as_zero_correlations():
    series,metadata,columns=fixture_data()
    columns[series[0].key][:]=np.nan
    assert common_case_scores(metadata,columns,series,["outcome"],121).empty
    columns[series[0].key][:]=0
    common=common_case_scores(metadata,columns,series,["outcome"],121)
    row=comparison_statistics(common,"outcome").set_index("component").loc[series[0].key]
    assert np.isnan(row.spearman_rho)
    assert "constant" in row.unavailable_reason


def test_loader_reuses_same_definition_families_scales_and_rejects_duplicate_nodes():
    series,metadata,_=fixture_data()
    calls=[]
    def loader(module,family,resolved):
        calls.append((module,family,resolved))
        if not resolved:
            return pd.DataFrame({"sample_id":metadata.sample_id,"module":module,
                "CT_raw":np.arange(15),"TS_raw":-np.arange(15)})
        return pd.concat([pd.DataFrame({"sample_id":metadata.sample_id,"module":module,
            "component":"TS_"+t,"metric_raw":np.arange(15)}) for t in ["AC","DLPFC","PCGBA23"]])
    columns=load_series(121,series,loader,"raw")
    assert len(calls)==2
    assert calls==[(121,"eigengene",True),(121,"connectivity",False)]
    assert len(columns)==5
    def wrong(*args):
        frame=loader(*args);frame["module"]=935;return frame
    with pytest.raises(ValueError,match="different module"):
        load_series(121,series,wrong,"raw")
    def duplicate(*args):
        frame=loader(*args);return pd.concat([frame,frame.iloc[:1]])
    with pytest.raises(ValueError,match="Multiple score"):
        load_series(121,series,duplicate,"raw")


def test_series_visibility_order_is_adjacent_not_stacked_or_summed():
    series,metadata,columns=fixture_data()
    ordered=adjacent_series([series[-1],series[1],series[-2],series[0]])
    assert [s.key for s in ordered]==[series[-1].key,series[1].key,series[0].key,series[-2].key]
    common=common_case_scores(metadata,columns,ordered,["outcome"],121)
    stats=comparison_statistics(common,"outcome")
    fig=score_comparison_figure(stats,ordered,method="spearman",module=121,
        outcome_label="Outcome",y_title="Correlation to Outcome",subtitle="KEGG enrichment: Metabolism / Lipid metabolism / Test | FDR=0.01")
    assert len(fig.data)==1 and len(fig.data[0].y)==4
    assert fig.layout.barmode not in {"stack","relative"}
    assert list(fig.data[0].marker.color)==[s.color for s in ordered]
    assert all(text == "1.000" for text in fig.data[0].text)
    spec=PlotExportSpec.from_figure("comparison",fig)
    artifact=build_artifact(spec,preset_config("Grant figure",spec.aspect_ratio),PlotContentConfig())
    np.testing.assert_allclose(artifact.figure.data[0].y,fig.data[0].y)
    selected_content=replace(PlotContentConfig(),bar_categories=(ordered[0].label,))
    filtered=build_artifact(spec,preset_config("Grant figure",spec.aspect_ratio),selected_content).figure
    assert list(filtered.data[0].x)==[ordered[0].label]
    key=next(a for a in filtered.layout.annotations if a.name=="publication-panel-title-score-groups")
    assert ordered[0].group in key.text
    assert ordered[-1].group not in key.text
    combined=combine_panels([("A",fig),("B",fig),("C",fig)],title="Combined")
    assert len(figure_panels(combined))==3
    assert len(combined.data)==3


def test_known_module_common_case_matches_direct_scipy():
    from app_helpers.data import load_aggregate,load_resolved,load_sample_metadata
    series=score_series_catalog(["AC","DLPFC","PCG"],{})
    def loader(module,family,resolved):
        function=load_resolved if resolved else load_aggregate
        return function("control_anchored",module,family,module_set="control_derived")
    columns=load_series(121,series,loader,"raw")
    metadata=load_sample_metadata("control_derived")
    common=common_case_scores(metadata,columns,series,["cogn_global"],121)
    assert common.sample_id.nunique()>=300
    stats=comparison_statistics(common,"cogn_global").set_index("component")
    assert stats.n.nunique()==1
    for spec in series:
        ids=common.loc[common.component.eq(spec.key),"sample_id"]
        x=columns[spec.key].reindex(ids).to_numpy()
        y=metadata.set_index("sample_id").loc[ids,"cogn_global"].to_numpy()
        assert stats.loc[spec.key,"pearson_r"]==pytest.approx(pearsonr(x,y).statistic,abs=1e-12)
        assert stats.loc[spec.key,"spearman_rho"]==pytest.approx(spearmanr(x,y).statistic,abs=1e-12)


@pytest.mark.parametrize("coefficients", [(.2,.6,.4),(-.2,-.6,-.4),(-.2,.6,-.4),(.4,.4,0),(.2,np.nan,-.4),(.19,.192,.3)])
def test_composite_equal_width_deltas_preserve_endpoints_and_never_sum(coefficients):
    series,metadata,columns=fixture_data()
    stats=comparison_statistics(common_case_scores(metadata,columns,series,["outcome"],121),"outcome")
    for spec,value in zip(series[:3],coefficients):
        stats.loc[stats.component.eq(spec.key),"spearman_rho"] = value
    original_stats=stats.copy(deep=True)
    fig=score_comparison_figure(stats,series,method="spearman",module=121,
        outcome_label="Outcome",y_title="Correlation",composite_keys=[s.key for s in series[:3]])
    assert fig.layout.barmode=="overlay"
    assert list(fig.layout.xaxis.categoryarray)==["Regional eigengenes","Pooled TS","Pooled CT"]
    overlaid=[t for t in fig.data if t.x[0]=="Regional eigengenes"]
    finite=[t for t in overlaid if np.isfinite(t.y[0])]
    assert all(t.width==.62 for t in overlaid)
    for positive in (True,False):
        part=sorted([t for t in finite if (t.meta['score_composite']['value']>=0)==positive],
            key=lambda t:abs(t.meta['score_composite']['value']))
        previous=0.
        for trace in part:
            assert trace.base==pytest.approx(previous)
            previous=trace.base+trace.y[0]
        assert sum(t.y[0] for t in part)==pytest.approx(previous)
    for spec,value in zip(series[:3],coefficients):
        trace=next(t for t in overlaid if t.name==spec.label)
        np.testing.assert_allclose([trace.base+trace.y[0]],[value],equal_nan=True)
        assert trace.marker.color==spec.color
        assert trace.text==("",)  # Exact values have collision-separated callouts.
    annotation=next(a for a in fig.layout.annotations if a.name=="publication-panel-title-score-groups")
    assert "N=" not in annotation.text and "ρ=" not in annotation.text
    callouts=[a for a in fig.layout.annotations if a.name.startswith('score-composite-value-')]
    assert len(callouts)==sum(np.isfinite(coefficients))
    assert len({round(a.ay,8) for a in callouts})==len(callouts)
    assert all('+' not in a.text and '-' not in a.text and 'N=' not in a.text for a in callouts)
    assert sorted(a.text for a in callouts)==sorted(f'{abs(v):.3f}' for v in coefficients if np.isfinite(v))
    spec=PlotExportSpec.from_figure('including_missing',fig)
    exported=build_artifact(spec,preset_config('Grant figure',spec.aspect_ratio),PlotContentConfig()).figure
    assert len([a for a in exported.layout.annotations if a.name.startswith('score-composite-value-')])==len(callouts)
    pd.testing.assert_frame_equal(stats,original_stats)


def test_composite_export_filter_colors_and_combined_panels_preserve_overlay():
    series,metadata,columns=fixture_data()
    stats=comparison_statistics(common_case_scores(metadata,columns,series,["outcome"],121),"outcome")
    fig=score_comparison_figure(stats,series,method="spearman",module=121,
        outcome_label="Outcome",y_title="Correlation",composite_keys=[s.key for s in series[:3]])
    spec=PlotExportSpec.from_figure("overlay",fig)
    config=preset_config("Grant figure",spec.aspect_ratio)
    content=replace(PlotContentConfig(),bar_series=(series[2].label,series[0].label),
        bar_categories=("Regional eigengenes",),bar_series_colors=((series[2].label,"#AA1177"),))
    exported=build_artifact(spec,config,content).figure
    assert len(exported.data)==2 and exported.layout.barmode=="overlay"
    for trace in exported.data:
        original=next(t for t in fig.data if t.name==trace.name)
        assert trace.zorder==original.zorder and trace.width==original.width
        np.testing.assert_allclose(trace.y,original.y)
    annotation=next(a for a in exported.layout.annotations if a.name=="publication-panel-title-score-groups")
    assert "#AA1177" in annotation.text and series[1].label not in annotation.text
    assert "N=" not in annotation.text and "ρ=" not in annotation.text
    callouts=[a for a in exported.layout.annotations if a.name.startswith('score-composite-value-')]
    assert len(callouts)==2 and any(a.arrowcolor=='#AA1177' for a in callouts)
    assert all(series[1].key not in a.name for a in callouts)
    # Color changes without bar selection must refresh the same legend/key too.
    recolored=build_artifact(spec,config,replace(PlotContentConfig(),bar_series_colors=((series[0].label,"#112233"),))).figure
    assert "#112233" in next(a.text for a in recolored.layout.annotations if a.name=="publication-panel-title-score-groups")
    combined=combine_panels([("Bars",fig),("Bars",fig)],title="Composite")
    assert combined.layout.barmode=="overlay"
    assert len(combined.data)==2*len(series)
    with pytest.raises(ValueError,match="at least two"):
        score_comparison_figure(stats,series,method="spearman",module=121,
            outcome_label="Outcome",y_title="Correlation",composite_keys=[series[0].key])
    with pytest.raises(ValueError,match="distinct"):
        score_comparison_figure(stats,series,method="spearman",module=121,
            outcome_label="Outcome",y_title="Correlation",composite_keys=[s.key for s in series[:3]],composite_label=series[-1].label)


@pytest.mark.skipif(os.environ.get("LIONESS_EXPORT_INTEGRATION")!="1",reason="Requires real Chrome/Kaleido")
def test_known_module_real_exports(tmp_path):
    from app_helpers.data import load_aggregate,load_resolved,load_sample_metadata,load_kegg,selected_kegg_table_annotation
    from app_helpers.association_score_comparison import scatter_statistics
    from app_helpers.charts import grouped_association_figure
    from app_helpers.figure_export.adapters import fit_automatic_config,render_figure
    series=score_series_catalog(["AC","DLPFC","PCG"],{})
    def loader(module,family,resolved):
        return (load_resolved if resolved else load_aggregate)("control_anchored",module,family,module_set="control_derived")
    metadata=load_sample_metadata("control_derived")
    columns=load_series(121,series,loader,"raw")
    common=common_case_scores(metadata,columns,series,["cogn_global"],121)
    statistics=comparison_statistics(common,"cogn_global")
    bars=score_comparison_figure(statistics,series,method="spearman",module=121,
        outcome_label="Global cognition",y_title="Correlation to clinical cognition",
        subtitle=selected_kegg_table_annotation(load_kegg(121,"control_derived")) or "KEGG enrichment: unavailable")
    ct=common.loc[common.component.eq("connectivity:CT")]
    options=dict(grouping_variable="diagnosis_group",grouping_levels=["Control","MCI","AD"],
        minimum_group_n=10,show_pooled=True)
    scatter=grouped_association_figure(ct,scatter_statistics(ct,"cogn_global",options),
        phenotype="cogn_global",phenotype_label="Global cognition",feature_label="CT connectivity",scale_label="Raw",
        module=121,color_by="diagnosis_group",color_label="Diagnosis",hover_fields={},
        grouping_labels={g:g for g in options["grouping_levels"]},
        categorical_color_fields=["diagnosis_group"],annotation_fields=["n","coefficient","p"],**options)
    combined=combine_panels([("CT connectivity",scatter),("Score comparison",bars)],title="Module M121: score–outcome comparison")
    overlay=score_comparison_figure(statistics,series,method="spearman",module=121,
        outcome_label="Global cognition",y_title="Correlation to clinical cognition",
        subtitle=selected_kegg_table_annotation(load_kegg(121,"control_derived")) or "KEGG enrichment: unavailable",
        composite_keys=[s.key for s in series if s.family=="eigengene"],show_significance=True)
    overlay_combined=combine_panels([("CT connectivity",scatter),("Score comparison",overlay)],title="Module M121: score–outcome comparison")
    absolute=score_comparison_figure(statistics,series,method="spearman",module=121,
        outcome_label="Global cognition",y_title="Absolute correlation to clinical cognition",
        composite_keys=[s.key for s in series if s.family=="eigengene"],show_significance=True,
        absolute_values=True)
    absolute_combined=combine_panels([("CT connectivity",scatter),("Score comparison",absolute)],title="Module M121: absolute score comparison")
    content_defaults=PlotContentConfig(statistic_fields=("n","coefficient","p"))
    for name,figure,content in [("bars",bars,content_defaults),("combined",combined,content_defaults),
        ("separate_bars",combined,replace(content_defaults,panel_order=("x2:y2",))),
        ("overlay",overlay,content_defaults),("overlay_combined",overlay_combined,content_defaults),
        ("overlay_separate",overlay_combined,replace(content_defaults,panel_order=("x2:y2",))),
        ("absolute_overlay",absolute,content_defaults),("absolute_overlay_combined",absolute_combined,content_defaults)]:
        spec=PlotExportSpec.from_figure(name,figure)
        config=replace(preset_config("Grant figure",spec.aspect_ratio),dpi=150,width=240,height=165,format="png")
        config=fit_automatic_config(spec,config,content)
        data,artifact=render_figure(spec,config,content)
        (tmp_path/f"M121_{name}.png").write_bytes(data)
        assert Image.open(io.BytesIO(data)).size==config.pixel_dimensions
        exported=[trace for trace in artifact.figure.data if trace.type=="bar"]
        expected=absolute if "absolute" in name else overlay if "overlay" in name else bars
        assert len(exported)==len(expected.data)
        for actual,original in zip(exported,expected.data):
            np.testing.assert_allclose(actual.y,original.y,atol=1e-12)
            if "overlay" in name:
                assert actual.zorder==original.zorder
        if "overlay" in name:
            assert artifact.figure.layout.barmode=="overlay"
    statistics.to_csv(tmp_path/"M121_common_donor_statistics.tsv",sep="\t",index=False)


def _widget(elements,label):
    return next(element for element in elements if element.label==label)


def test_associations_default_scatter_and_score_comparison_render():
    app=AppTest.from_file(APP/"streamlit_app.py",default_timeout=180).run()
    assert not app.exception
    picker=_widget(app.selectbox,"Plot type")
    assert picker.value=="Scatter"
    app=picker.set_value("Score comparison").run()
    assert not app.exception
    assert app.get("plotly_chart")
    app=_widget(app.radio,"Score comparison layout").set_value("CT scatter beside comparison bars").run()
    assert not app.exception
    app=_widget(app.checkbox,"Include pathology-adjusted cognitive-slope scatter").set_value(True).run()
    assert not app.exception
    app=_widget(app.checkbox,"Show score-scatter overview above").set_value(True).run()
    assert not app.exception and len(app.get("plotly_chart"))==2
    app=_widget(app.checkbox,"Show separate panels too").set_value(True).run()
    assert not app.exception and len(app.get("plotly_chart"))==5


def test_composite_column_controls_and_statistics_render_without_recalculation_changes():
    import json
    app=AppTest.from_file(APP/"streamlit_app.py",default_timeout=180).run()
    app=_widget(app.selectbox,"Module definition").set_value("control_derived").run()
    app=_widget(app.selectbox,"Module").set_value(121).run()
    app=_widget(app.selectbox,"Plot type").set_value("Score comparison").run()
    before=next(d.value for d in app.dataframe if "pearson_r" in d.value.columns).copy()
    assert _widget(app.radio,"Bar arrangement").value=="Separate bars"
    app=_widget(app.radio,"Bar arrangement").set_value("Vertical composite column").run()
    assert not app.exception and app.get("plotly_chart")
    assert _widget(app.multiselect,"Series in composite column").value==["eigengene:AC","eigengene:DLPFC","eigengene:PCG"]
    after=next(d.value for d in app.dataframe if "pearson_r" in d.value.columns)
    pd.testing.assert_frame_equal(before,after)
    figure=json.loads(app.get("plotly_chart")[0].proto.spec)
    assert figure["layout"]["barmode"]=="overlay" and len(figure["data"])==5
    assert len([a for a in figure['layout']['annotations'] if a.get('name','').startswith('score-composite-value-')])==3
    app=_widget(app.checkbox,"Show significance asterisks (nominal p < 0.05)").check().run()
    assert not app.exception
    figure=json.loads(app.get("plotly_chart")[0].proto.spec)
    assert any('*' in a['text'] for a in figure['layout']['annotations'] if a.get('name','').startswith('score-composite-value-'))
    pd.testing.assert_frame_equal(before,next(d.value for d in app.dataframe if 'pearson_r' in d.value))
    app=_widget(app.radio,"Score comparison layout").set_value("CT scatter beside comparison bars").run()
    assert not app.exception
    figure=json.loads(app.get("plotly_chart")[0].proto.spec)
    assert figure["layout"]["barmode"]=="overlay"
    app=_widget(app.multiselect,"Series in composite column").set_value(["eigengene:AC"]).run()
    assert not app.exception and not app.get("plotly_chart")
    assert any("at least two" in str(i.value) for i in app.info)


def test_composite_export_rebases_retained_series_without_changing_correlations():
    series,metadata,columns=fixture_data()
    stats=comparison_statistics(common_case_scores(metadata,columns,series,['outcome'],121),'outcome')
    for spec,value in zip(series[:3],[.2,.6,.4]):
        stats.loc[stats.component.eq(spec.key),'spearman_rho']=value
    fig=score_comparison_figure(stats,series,method='spearman',module=121,
        outcome_label='Outcome',y_title='Correlation',composite_keys=[s.key for s in series[:3]])
    spec=PlotExportSpec.from_figure('composite',fig)
    content=replace(PlotContentConfig(),bar_series=(series[1].label,series[2].label),bar_categories=('Regional eigengenes',))
    exported=build_artifact(spec,preset_config('Grant figure',spec.aspect_ratio),content).figure
    retained=sorted(exported.data,key=lambda t:t.meta['score_composite']['value'])
    assert [t.base for t in retained]==pytest.approx([0,.4])
    assert [t.y[0] for t in retained]==pytest.approx([.4,.2])
    assert [t.base+t.y[0] for t in retained]==pytest.approx([.4,.6])
    pooled=build_artifact(spec,preset_config('Grant figure',spec.aspect_ratio),
        replace(PlotContentConfig(),bar_categories=('Pooled TS',))).figure
    assert not any(a.name.startswith('score-composite-value-') for a in pooled.layout.annotations)


@pytest.mark.parametrize('method',['pearson','spearman'])
def test_value_labels_hide_sign_and_n_and_stars_use_selected_nominal_p(method):
    series,metadata,columns=fixture_data()
    stats=comparison_statistics(common_case_scores(metadata,columns,series,['outcome'],121),'outcome')
    stats['spearman_rho']=-.4;stats['pearson_r']=.6
    stats['spearman_p']=[.01,.05,np.nan,.001,.9]
    stats['pearson_p']=[.5,.001,.01,np.nan,.05]
    stats['spearman_fdr_across_modules']=.001  # Must never recycle this FDR.
    options=dict(method=method,module=121,outcome_label='Outcome',y_title='Correlation',show_significance=True)
    p=stats.set_index('component').reindex([s.key for s in series])[method+'_p']
    separate=score_comparison_figure(stats,series,**options)
    assert [text.endswith('*') for text in separate.data[0].text]==[pd.notna(v) and v<.05 for v in p]
    assert all('+' not in t and '-' not in t and 'N=' not in t and '=' not in t for t in separate.data[0].text)
    composite=score_comparison_figure(stats,series,composite_keys=[s.key for s in series[:3]],**options)
    assert len([a for a in composite.layout.annotations if a.name.startswith('score-composite-value-')])==3
    hidden=score_comparison_figure(stats,series,composite_keys=[s.key for s in series[:3]],show_values=False,**options)
    assert not any(a.name.startswith('score-composite-value-') for a in hidden.layout.annotations)
    assert all(t.text==('',) for t in hidden.data)


@pytest.mark.parametrize('method',['pearson','spearman'])
@pytest.mark.parametrize('composite',[False,True])
def test_absolute_bar_axis_preserves_signed_statistics_p_values_and_exports(method,composite):
    series,metadata,columns=fixture_data()
    stats=comparison_statistics(common_case_scores(metadata,columns,series,['outcome'],121),'outcome')
    coefficient='spearman_rho' if method=='spearman' else 'pearson_r'
    expected=np.array([-.2,.6,-.4,0,np.nan])
    for spec,value in zip(series,expected):
        stats.loc[stats.component.eq(spec.key),coefficient]=value
    before=stats.copy(deep=True)
    kwargs=dict(method=method,module=121,outcome_label='Outcome',y_title='Absolute correlation',
        composite_keys=[s.key for s in series[:3]] if composite else (),show_significance=True)
    signed=score_comparison_figure(stats,series,**kwargs)
    absolute=score_comparison_figure(stats,series,absolute_values=True,**kwargs)
    assert signed.layout.yaxis.range==(-1.15,1.15)
    assert absolute.layout.yaxis.range==(0,1.15)
    assert absolute.layout.yaxis.tickvals==(0,.25,.5,.75,1)
    assert absolute.layout.meta['score_comparison']['y_axis_mode']=='absolute'
    if composite:
        for trace,value in zip(absolute.data,expected):
            np.testing.assert_allclose([trace.base+trace.y[0]],[abs(value)],equal_nan=True)
        assert max(t.base+t.y[0] for t in absolute.data[:3])==pytest.approx(.6)
        assert all(a.ay>=0 for a in absolute.layout.annotations if a.name.startswith('score-composite-value-'))
    else:
        np.testing.assert_allclose(absolute.data[0].y,np.abs(expected),equal_nan=True)
    for original,displayed in zip(signed.data,absolute.data):
        pd.testing.assert_frame_equal(pd.DataFrame(original.customdata),pd.DataFrame(displayed.customdata))
        assert original.text==displayed.text  # Asterisks and labels are unchanged.
    spec=PlotExportSpec.from_figure('absolute',absolute)
    exported=build_artifact(spec,preset_config('Grant figure',spec.aspect_ratio),PlotContentConfig()).figure
    assert exported.layout.yaxis.range==(0,1.15)
    for trace,original in zip(exported.data,absolute.data):
        np.testing.assert_allclose(np.asarray(trace.y,float),np.asarray(original.y,float),equal_nan=True)
        if composite:assert trace.base==original.base
    combined=combine_panels([('Signed',signed),('Absolute',absolute)],title='Comparison')
    assert combined.layout.yaxis.range==signed.layout.yaxis.range
    assert combined.layout.yaxis2.range==absolute.layout.yaxis.range
    pd.testing.assert_frame_equal(stats,before)


def test_ui_absolute_y_axis_toggles_bars_and_keeps_statistics_and_composite():
    from app_helpers.figure_export.config import restore_figure
    app=AppTest.from_file(APP/'streamlit_app.py',default_timeout=180).run()
    app=_widget(app.selectbox,'Module definition').set_value('control_derived').run()
    app=_widget(app.selectbox,'Module').set_value(121).run()
    app=_widget(app.selectbox,'Plot type').set_value('Score comparison').run()
    assert not _widget(app.checkbox,'Show absolute correlations on Y-axis').value
    before=next(d.value for d in app.dataframe if 'pearson_r' in d.value).copy(deep=True)
    signed=restore_figure(app.get('plotly_chart')[0].proto.spec)
    assert min(signed.data[0].y)<0
    app=_widget(app.checkbox,'Show absolute correlations on Y-axis').check().run()
    assert not app.exception
    absolute=restore_figure(app.get('plotly_chart')[0].proto.spec)
    np.testing.assert_allclose(absolute.data[0].y,np.abs(signed.data[0].y))
    assert 'Absolute correlation' in absolute.layout.yaxis.title.text
    pd.testing.assert_frame_equal(before,next(d.value for d in app.dataframe if 'pearson_r' in d.value))
    app=_widget(app.radio,'Bar arrangement').set_value('Vertical composite column').run()
    assert not app.exception
    composite=restore_figure(app.get('plotly_chart')[0].proto.spec)
    assert all(t.y[0]>=0 and t.base>=0 for t in composite.data)
    app=_widget(app.radio,'Score comparison layout').set_value('CT scatter beside comparison bars').run()
    assert not app.exception
    combined=restore_figure(app.get('plotly_chart')[0].proto.spec)
    assert combined.layout.yaxis2.range==(0,1.15)
    assert all(a.ay>=0 for a in combined.layout.annotations if a.name.startswith('score-composite-value-'))
    app=_widget(app.text_input,'Correlation axis label').set_value('Custom axis label').run()
    app=_widget(app.checkbox,'Show absolute correlations on Y-axis').uncheck().run()
    restored=restore_figure(app.get('plotly_chart')[0].proto.spec)
    assert restored.layout.yaxis2.range==(-1.15,1.15)
    assert restored.layout.yaxis2.title.text=='Custom axis label'
    for original,current in zip([t for t in combined.data if t.type!='bar'],
                                [t for t in restored.data if t.type!='bar']):
        np.testing.assert_allclose(np.asarray(original.y,float),np.asarray(current.y,float),equal_nan=True)
    pd.testing.assert_frame_equal(before,next(d.value for d in app.dataframe if 'pearson_r' in d.value))


def test_comparison_reuses_outcome_group_scale_and_effect_filter_controls():
    app=AppTest.from_file(APP/"streamlit_app.py",default_timeout=180).run()
    app=_widget(app.selectbox,"Module definition").set_value("control_derived").run()
    app=_widget(app.selectbox,"Module").set_value(121).run()
    app=_widget(app.selectbox,"Plot type").set_value("Score comparison").run()
    assert not app.exception
    assert len(_widget(app.multiselect,"Same-module eigengene regions").value)==3
    app=_widget(app.selectbox,"Group correlations by").set_value("clusters").run()
    app=_widget(app.multiselect,"Group levels").set_value([4]).run()
    assert not app.exception and app.get("plotly_chart")
    stats=next(d.value for d in app.dataframe if "pearson_r" in d.value.columns)
    assert stats.n.nunique()==1 and stats.n.iloc[0]<=30
    app=_widget(app.selectbox,"Feature scale").set_value("asinh").run()
    app=_widget(app.radio,"Association correlation").set_value("Pearson").run()
    assert not app.exception and app.get("plotly_chart")
    app=_widget(app.selectbox,"Group correlations by").set_value("diagnosis_group").run()
    app=_widget(app.radio,"AD–Control edge subset").set_value("effect_size").run()
    assert not app.exception and app.get("plotly_chart")
    assert _widget(app.selectbox,"Plot type").value=="Score comparison"


def test_nominal_outcome_never_enters_comparison_correlations():
    app=AppTest.from_file(APP/"streamlit_app.py",default_timeout=180).run()
    app=_widget(app.selectbox,"Association outcome").set_value("clusters").run()
    app=_widget(app.selectbox,"Plot type").set_value("Score comparison").run()
    assert not app.exception
    assert any("numeric/ordinal" in str(i.value) for i in app.info)


def test_comparison_opens_existing_publication_export_dialog(monkeypatch):
    from app_helpers.figure_export import ui
    buffer=io.BytesIO()
    Image.new("RGB",(100,80),"white").save(buffer,format="PNG")
    calls=[]
    def preview(spec,config,content):
        calls.append(spec.plot_id)
        return buffer.getvalue(),()
    monkeypatch.setattr(ui,"_render_cached",preview)
    app=AppTest.from_file(APP/"streamlit_app.py",default_timeout=180).run()
    app=_widget(app.selectbox,"Plot type").set_value("Score comparison").run()
    app=_widget(app.button,"Export figure").click().run()
    assert not app.exception
    assert _widget(app.selectbox,"Publication preset")
    assert _widget(app.button,"Prepare download")
    assert calls and calls[-1].startswith("associations.score_comparison")
