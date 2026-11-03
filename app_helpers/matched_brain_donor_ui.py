"""Donor plots using approved saved scores and the existing chart/export stack."""
from __future__ import annotations
import hashlib
import html
import json
import re
import textwrap
import numpy as np
import pandas as pd
import streamlit as st
from app_helpers import matched_brain_donors as D,matched_brain_results as R
from app_helpers.data import (ASSOCIATION_GROUP_LABELS,ASSOCIATION_OUTCOME_LABELS,
    ASSOCIATION_LEVEL_ORDERS,CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES,NUMERIC_OUTCOMES,
    FEATURE_LABELS,HOVER_LABELS,association_level_label,selected_kegg_table_annotation,association_kegg_subtitles)
from app_helpers.charts import (grouped_association_figure,categorical_association_figure,
    distribution_figure,CONTINUOUS_COLOR_SCALES)
from app_helpers.distributions import calculate_pairwise_distribution_statistics,distribution_contrasts
from app_helpers.table_controls import filterable_dataframe

DEFINITION="Matched-Control brain L4 (247 modules; 80 discovery Controls)"
SCALES={"raw":"Raw","asinh":"asinh","rint":"Z-score"}


def donor_layout(fig):
    """Keep the existing panel semantics, with a narrow-safe header band."""
    title_depth=stat_depth=1
    for annotation in fig.layout.annotations:
        if str(annotation.name or "").startswith("publication-panel-title-"):
            lines=str(annotation.text).split("<br>")
            plain=[html.unescape(re.sub("<[^>]+>","",line)) for line in lines]
            wrapped=[part for line in plain[1:] for part in textwrap.wrap(line,32,break_long_words=False,break_on_hyphens=False)]
            annotation.text="<b>"+html.escape(plain[0])+"</b>"+"".join("<br>"+html.escape(line) for line in wrapped)
            title_depth=max(title_depth,len(wrapped)+1)
        elif str(annotation.name or "").startswith("publication-statistics-"):
            stat_depth=max(stat_depth,str(annotation.text).count("<br>")+1)
    header=15*title_depth+12*stat_depth+20
    top=max(float(fig.layout.margin.t or 0),header+115)
    bottom=max(float(fig.layout.margin.b or 0),65)
    height=max(float(fig.layout.height or 0),450+top+bottom)
    fig.update_layout(height=height,margin=dict(t=top,b=bottom),title=dict(y=.99,yanchor="top",yref="container",font=dict(size=14)),
        legend=dict(orientation="h",x=.5,xanchor="center",y=1+(header+12)/(height-top-bottom),yanchor="bottom"))
    meta=dict(fig.layout.meta or {});export=dict(meta.get("figure_export",{}))
    export["fixed_container_title"]=True;meta["figure_export"]=export;fig.layout.meta=meta
    return fig


def _table(frame,key,title):
    filterable_dataframe(frame,table_key=key,table_name=title,hide_index=True)
    st.download_button("Download "+title.lower(),frame.to_csv(sep="\t",index=False).encode(),
        file_name=key+".tsv",mime="text/tab-separated-values",key=key+"_download")


def _levels(frame,field):
    if field=="__all__":return ["__all__"]
    observed=frame[field].dropna().drop_duplicates().tolist()
    preferred=ASSOCIATION_LEVEL_ORDERS.get(field,[])
    return [v for v in preferred if v in observed]+sorted([v for v in observed if v not in preferred],key=str)


def _subtitles(module,components,rule):
    expanded=R.kegg(rule).loc[lambda f:f.cluster_id.eq(module)].copy()
    regional=R.kegg(rule,regional=True).loc[lambda f:f.cluster_id.eq(module)].copy()
    result={};annotation=selected_kegg_table_annotation(expanded)
    for comp in components:
        result.update(association_kegg_subtitles(regional,[comp],resolved=comp not in {"TS","CT","all"},aggregate_annotation=annotation))
        if rule!="all":result[comp]+=" | Frozen "+R.RULES[rule]
    return result


def render_view(render,module,*,distributions=False):
    st.subheader("Donor feature distributions" if distributions else "Donor association plots")
    if not D.available():
        st.info("The approved donor bundle is not installed yet. Aggregate results remain available.");return
    prefix="matched_donor_distribution" if distributions else "matched_donor_scatter"
    a,b,c=st.columns(3)
    cohorts=["Held-out validation","Discovery",*R.COHORTS.keys(),"Unmatched complete donors (exploratory)","MCI (exploratory)"]
    cohorts=list(dict.fromkeys(cohorts))
    cohort=a.selectbox("Donor cohort",cohorts,format_func=lambda v:R.COHORTS.get(v,v),key=prefix+"_cohort")
    feature=b.selectbox("Donor module feature",["connectivity","eigengene","positive_density","negative_density","abs_sum","positive_abs_sum","negative_abs_sum"],
        format_func=lambda v:FEATURE_LABELS.get(v,v),key=prefix+"_feature")
    scale=c.selectbox("Donor score scale",list(SCALES),format_func=SCALES.get,key=prefix+"_scale")
    a,b=st.columns(2)
    rule=a.selectbox("Donor edge subset",["all"] if feature=="eigengene" else list(R.RULES),format_func=R.RULES.get,key=prefix+"_rule")
    norm=b.selectbox("Donor score handling",["unfiltered"] if feature=="eigengene" else ["standard_pruned","retained_edge"],
        format_func=lambda v:{"unfiltered":"Unfiltered eigengene","standard_pruned":"Standard pruned","retained_edge":"Retained-edge normalized"}[v],key=prefix+"_norm")
    comps=list(R.COMPONENTS)
    if feature=="eigengene":comps=[c for c in comps if c.startswith("TS_")]
    if cohort in {"Incomplete-region validation",D.BALANCED,D.PROFILE}:comps=[c for c in comps if c not in {"TS","CT","all"}]
    default=["TS","CT"] if "CT" in comps else ["TS_DLPFC"]
    # Reset an invalid pooled selection when switching to regional eigengenes
    # or incomplete-region donors; never silently retain impossible components.
    component_context=("eigengene" if feature=="eigengene" else "network")+("_pooled" if "CT" in comps else "_resolved")
    components=st.multiselect("Donor plot components",comps,default=default,format_func=R.COMPONENTS.get,key=prefix+"_components_"+component_context)
    if not components:st.info("Select at least one component.");return
    frame=D.load_scores(module,cohort=cohort,components=components,feature=feature,rule=rule,normalization=norm,scale=scale)
    if frame.empty:st.info("No observed scores are available for this module and cohort.");return
    diagnoses=_levels(frame,"diagnosis_group")
    diagnoses=st.multiselect("Include diagnoses",diagnoses,default=diagnoses,key=prefix+"_diagnoses")
    frame=frame.loc[frame.diagnosis_group.isin(diagnoses)]
    if frame.empty:st.info("No donors match these diagnosis filters.");return
    a,b=st.columns(2)
    outcomes=[x for x in ASSOCIATION_OUTCOME_LABELS if x in frame]
    outcome=a.selectbox("Donor outcome",outcomes,format_func=ASSOCIATION_OUTCOME_LABELS.get,key=prefix+"_outcome") if not distributions else None
    categorical=outcome in CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES
    if outcome and outcome in {"cogdx","braak_stage","cerad_score","adnc","parkinsonism"}:
        categorical=b.radio("Outcome interpretation",["Numeric / ordinal correlation","Categorical comparison"],key=prefix+"_interpretation")=="Categorical comparison"
    fields=[x for x in ASSOCIATION_GROUP_LABELS if x in frame or x=="__all__"]
    if distributions:fields=[x for x in fields if x!="__all__"]
    if outcome in fields and not categorical:fields.remove(outcome)
    grouping=b.selectbox("Divide distributions by" if distributions else "Group donor correlations by",fields,
        format_func=ASSOCIATION_GROUP_LABELS.get,key=prefix+"_group") if not categorical else outcome
    levels=_levels(frame,grouping)
    if grouping!="__all__":
        level_context=grouping+"_"+hashlib.sha256(cohort.encode()).hexdigest()[:8]
        levels=st.multiselect("Included category levels",levels,default=levels,
            format_func=lambda v:association_level_label(grouping,v),key=prefix+"_levels_"+level_context)
        missing=int(frame[grouping].isna().sum())
        if missing:st.caption(f"{missing} component-score rows with missing grouping values are excluded; missing is not a biological category.")
    frame=D.filter_donors(frame,diagnoses,grouping,levels)
    if frame.empty or not np.isfinite(frame.metric_value).any():
        st.warning("No finite scores remain. This component may be structurally absent or the retained-edge set may be empty.");return
    min_n=st.selectbox("Minimum donors per tested group",[5,10,20],index=1,key=prefix+"_min_n")
    if cohort=="Discovery" and rule!="all":st.warning("Discovery-selected edge separation is selection-biased. Prefer held-out or incomplete-region validation.")
    st.caption(f"{frame.sample_id.nunique():,} distinct donors; {len(frame):,} observed donor–component rows. No missing tissue is imputed.")
    _table(frame,prefix+"_scores","Displayed pseudonymized donor scores")
    if distributions:
        chart_type=st.selectbox("Donor distribution view",["Raincloud","ECDF","Violin","Histogram"],key=prefix+"_chart")
        display=frame.copy();display["display_group"]=display[grouping].map(lambda x:association_level_label(grouping,x))
        fig=distribution_figure(display,FEATURE_LABELS.get(feature,feature),SCALES[scale],
            [association_level_label(grouping,v) for v in levels],module,chart_type,
            module_definition=DEFINITION,group_column="display_group",group_label=ASSOCIATION_GROUP_LABELS[grouping])
        render(donor_layout(fig),key=prefix+"_plot")
        stats=D.selected_statistics(frame,outcome=grouping,grouping=grouping,levels=levels,min_n=min_n,categorical=True)
        category_outcome=grouping
        if len(levels)>=2:
            reference=st.selectbox("Pairwise reference category",levels,format_func=lambda v:association_level_label(grouping,v),key=prefix+"_reference")
            pairs=calculate_pairwise_distribution_statistics(frame,D.KEYS,category_column=grouping,
                contrasts=distribution_contrasts(levels,reference_level=reference),minimum_group_n=min_n,
                bootstrap_resamples=1000,seed=42)
            _table(pairs,prefix+"_pairwise","Selected-module pairwise distribution tests")
    elif categorical:
        category_outcome=outcome
        stats=D.selected_statistics(frame,outcome=outcome,grouping=grouping,levels=levels,min_n=min_n,categorical=True)
        fig=categorical_association_figure(frame,stats,category_variable=outcome,category_label=ASSOCIATION_OUTCOME_LABELS[outcome],
            category_levels=levels,category_labels={str(v):association_level_label(outcome,v) for v in levels},
            feature_label=FEATURE_LABELS.get(feature,feature),scale_label=SCALES[scale],module=module,minimum_group_n=min_n,
            annotation_fields=("n","effect","p"),module_definition=DEFINITION,kegg_subtitles=_subtitles(module,components,rule),
            hover_fields={"sample_id":"Anonymous sample",**HOVER_LABELS})
        render(donor_layout(fig),key=prefix+"_plot")
    else:
        category_outcome=outcome
        stats=D.numeric_statistics(frame,outcome,grouping,levels,min_n)
    # Family correction is an explicit action, never a BH over plotted groups.
    context=dict(cohort=cohort,components=tuple(components),feature=feature,rule=rule,normalization=norm,scale=scale,
        diagnoses=tuple(diagnoses),grouping=grouping,levels=tuple(levels),outcome=category_outcome,min_n=min_n,categorical=bool(categorical or distributions))
    identity=hashlib.sha256(json.dumps(context,sort_keys=True,default=str).encode()).hexdigest()
    state=prefix+"_family"
    if st.button("Calculate module-set FDR (all 247 modules)",key=prefix+"_calculate"):
        with st.spinner("Reading one module at a time and correcting each fixed component/group separately…"):
            family=D.family_statistics(**context)
        st.session_state[state]=(identity,family)
    saved=st.session_state.get(state)
    if saved and saved[0]==identity:
        family=saved[1];stats=family.loc[family.module.eq(module)]
        st.caption("BH uses all valid modules separately within each fixed component and grouping level—not groups × modules. Display filters never redefine this family.")
        _table(family,prefix+"_family_statistics","Complete module-family statistics")
    else:
        st.caption("Module-set FDR is not calculated from one module or from displayed groups. Use the explicit all-module action to obtain it. These donor-plot tests are unadjusted; demographic-adjusted tests remain in CAMEL associations.")
    if not distributions and not categorical:
        a,b,c=st.columns(3)
        method=a.radio("Donor correlation statistic",["spearman","pearson"],format_func=str.capitalize,key=prefix+"_method")
        group_trends=b.checkbox("Show group-specific trend lines",value=True,key=prefix+"_trends")
        pooled=c.checkbox("Show pooled all-donor trend",value=grouping!="__all__",key=prefix+"_pooled")
        solid=st.checkbox("Solid black pooled line",value=False,key=prefix+"_solid")
        a,b=st.columns(2)
        color_by=a.selectbox("Donor point colors",["Single color",*list(dict.fromkeys([grouping if grouping!="__all__" else "diagnosis_group",*outcomes]))],
            index=1,format_func=lambda x:ASSOCIATION_OUTCOME_LABELS.get(x,x),key=prefix+"_color")
        single=b.color_picker("Point color","#0072B2",key=prefix+"_single") if color_by=="Single color" else None
        palette=st.selectbox("Continuous donor palette",list(CONTINUOUS_COLOR_SCALES),key=prefix+"_palette")
        annotation=("n","coefficient","p","fdr") if "pearson_fdr_across_modules" in stats else ("n","coefficient","p")
        fig=grouped_association_figure(frame,stats,phenotype=outcome,phenotype_label=ASSOCIATION_OUTCOME_LABELS[outcome],
            feature_label=FEATURE_LABELS.get(feature,feature),scale_label=SCALES[scale],grouping_variable=grouping,
            grouping_levels=levels,grouping_labels={str(v):association_level_label(grouping,v) for v in levels},module=module,
            color_by="diagnosis_group" if single else color_by,color_label=ASSOCIATION_OUTCOME_LABELS.get(color_by,color_by),
            hover_fields={"sample_id":"Anonymous sample",**HOVER_LABELS},correlation_method=method,annotation_fields=annotation,
            minimum_group_n=min_n,show_group_trends=group_trends,show_pooled=pooled,module_definition=DEFINITION,
            categorical_color_fields=set(ASSOCIATION_GROUP_LABELS)-{"__all__"},continuous_colorscale=palette,
            single_point_color=single,pooled_line_dash="solid" if solid else "dash",kegg_subtitles=_subtitles(module,components,rule))
        render(donor_layout(fig),key=prefix+"_plot")
    _table(stats,prefix+"_statistics","Displayed-module donor statistics")
