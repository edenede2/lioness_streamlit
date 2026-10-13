"""Lazy, small-table Streamlit view for fixed-module MDC validation."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_helpers.data import _read_filtered, data_path_available, ensure_data_path, module_set_path
from app_helpers.mdc_validation import VARIANTS, SCOPES
from app_helpers.table_controls import filterable_dataframe

DIRECTORY = "mdc_label_validation"
LABELS = {"total":"All edges", "TS":"TS pooled", "CT":"CT pooled",
          "TS_VH":"TS: VH", "TS_SMA":"TS: SMA", "TS_QUAD":"TS: QUAD",
          "CT_VH__SMA":"CT: VH–SMA", "CT_VH__QUAD":"CT: VH–QUAD", "CT_SMA__QUAD":"CT: SMA–QUAD",
          "CT_minus_TS":"CT minus TS (log2 MDC)"}


def load_manifest(module_set="muscle_control_l4"):
    path=module_set_path(f"{DIRECTORY}/manifest.json",module_set)
    return json.loads(ensure_data_path(path).read_text()) if data_path_available(path) else {}


@st.cache_data(max_entries=4, ttl=600, show_spinner=False)
def load_results(variant, revision, module_set="muscle_control_l4"):
    return _read_filtered(module_set_path(f"{DIRECTORY}/results.parquet",module_set),
                          [("normalization_variant","=",variant)])


@st.cache_data(max_entries=12, ttl=600, show_spinner=False)
def load_replicates(kind, variant, module, scope, revision, module_set="muscle_control_l4"):
    return _read_filtered(module_set_path(f"{DIRECTORY}/{kind}_replicates.parquet",module_set),
        [("normalization_variant","=",variant),("module","=",int(module)),("scope","=",scope)])


def heatmap_figure(frame, selected_module, top_n=20, significant_only=False):
    frame=frame.copy()
    frame.loc[~frame.test_eligible,["statistic","label_fdr"]]=np.nan
    frame["rank_score"]=frame.label_fdr.fillna(2)
    ranks=frame.groupby("module").rank_score.min().sort_values(kind="stable")
    if significant_only:
        ranks=ranks[ranks<.05]
    modules=list(ranks.head(top_n).index)
    scopes=[s for s in SCOPES if s!="CT_minus_TS"]
    cells=frame.set_index(["module","scope"]).reindex(pd.MultiIndex.from_product([modules,scopes]))
    shape=(len(modules),len(scopes))
    z=cells.statistic.to_numpy().reshape(shape)
    q=cells.label_fdr.to_numpy().reshape(shape)
    fig=go.Figure(go.Heatmap(z=z,x=[LABELS[s] for s in scopes],y=[f"M{m}" for m in modules],
        colorscale="RdBu_r",zmid=0,customdata=q,text=np.where(q<.05,"*",""),texttemplate="%{text}",
        hovertemplate="%{y} · %{x}<br>log2 MDC=%{z:.3g}<br>Label FDR=%{customdata:.4g}<extra></extra>",
        colorbar=dict(title="log2 MDC")))
    fig.update_layout(template="plotly_white",font=dict(size=13),height=max(320,25*len(modules)+180),
                      yaxis=dict(autorange="reversed",tickmode="array",tickvals=[f"M{m}" for m in modules]),
                      title="Observed MDC; * diagnosis-label FDR <0.05",margin=dict(l=70,r=80,t=60,b=130))
    return fig


def null_figure(replicates, observed, scope):
    values=replicates.statistic.dropna()
    fig=go.Figure(go.Histogram(x=values,nbinsx=30,marker_color="#4C78A8",name="Permuted diagnosis labels"))
    if np.isfinite(observed):
        fig.add_vline(x=observed,line_color="#D62728",line_width=3,annotation_text="Observed",annotation_position="top")
    fig.update_layout(template="plotly_white",font=dict(size=13),title="Diagnosis-label permutation distribution",xaxis_title=LABELS[scope] if scope=="CT_minus_TS" else "log2 MDC",
                      yaxis_title="Permutations",height=370,showlegend=False,margin=dict(t=70))
    return fig


def balanced_figure(frame, scope):
    selected=frame[frame.scope.eq(scope)].copy()
    selected=selected[selected.test_eligible]
    selected=selected[np.isfinite(selected.balanced_median)].sort_values("balanced_median",kind="stable").head(30)
    fig=go.Figure(go.Scatter(x=selected.balanced_median,y=[f"M{m}" for m in selected.module],mode="markers",
        marker=dict(color="#4C78A8",size=8),error_x=dict(type="data",symmetric=False,
            array=selected.balanced_high-selected.balanced_median,arrayminus=selected.balanced_median-selected.balanced_low),
        name="Balanced median + central 95%",customdata=selected.balanced_finite_n if 'balanced_finite_n' in selected else None,
        hovertemplate="%{y}<br>Median=%{x:.3g}<br>Finite resamples=%{customdata}<extra></extra>"))
    fig.add_trace(go.Scatter(x=selected.statistic,y=[f"M{m}" for m in selected.module],mode="markers",
        marker=dict(color="#D62728",symbol="diamond",size=8),name="Original 155 AD / 120 Control"))
    fig.update_layout(template="plotly_white",font=dict(size=13),title="Balanced-group sensitivity (not confidence intervals)",height=max(350,25*len(selected)+150),
        xaxis_title="CT–TS contrast" if scope=="CT_minus_TS" else "log2 MDC",
        yaxis=dict(autorange="reversed",tickmode="array",tickvals=[f"M{m}" for m in selected.module]),
        legend=dict(orientation="h",y=1.15),margin=dict(t=110))
    return fig


def render_mdc_validation(module, render_plot, module_set="muscle_control_l4"):
    manifest=load_manifest() if module_set=="muscle_control_l4" else load_manifest(module_set)
    st.caption("All-edge group-network MDC context; this catalog does not change donor-level LIONESS/BONOBO scores or prediction analyses.")
    definition="Control-derived" if module_set=="muscle_control_l4" else "all-donor"
    st.warning(f"Fixed {definition} modules: these checks do not correct discovery-selection bias (step 3). They are conditional validation, not independent replication.")
    if not manifest:
        st.info("Diagnosis-label MDC validation is being prepared. Existing MDC catalogs remain available.")
        return
    nlabel=manifest["completed_label_permutations"];nbal=manifest["completed_balanced_resamples"]
    st.caption(f"Published snapshot: label permutations {nlabel}/{manifest['label_permutations']}; balanced resamples {nbal}/{manifest['balanced_resamples']}. Each donor's expression profile is preserved. Label-dependent M5 terms are rebuilt.")
    variant=st.selectbox("Validation network weighting",list(VARIANTS),format_func=VARIANTS.get,key="mdc_validation_weighting")
    st.caption("The CT–TS specificity test is primary with unpowered Pearson weights. M5 uses different CT/TS weighting functions and is a sensitivity analysis.")
    revision=manifest["files"]["results.parquet"]["sha256"]
    frame=load_results(variant,revision) if module_set=="muscle_control_l4" else load_results(variant,revision,module_set)
    section=st.radio("MDC validation analysis",["Diagnosis-label tests","Balanced groups","Direct CT–TS test"],horizontal=True,key="mdc_validation_analysis")
    selected=frame[frame.module.eq(module)]
    if section=="Balanced groups":
        st.info("120 Controls versus 120 AD donors; Controls are fixed and AD donors are repeatedly sampled without replacement, preserving AD study/sex proportions. Intervals describe subsampling variability, not confidence intervals or p-values.")
        if nbal!=manifest["balanced_resamples"]:
            st.info("Balanced analysis is incomplete; final summaries will appear when all resamples finish.")
            return
        scope=st.selectbox("Balanced MDC component",list(SCOPES),format_func=LABELS.get)
        render_plot(balanced_figure(frame,scope),use_container_width=True,plot_id="mdc.validation.balance")
        table=frame[frame.scope.eq(scope)].copy()
    elif section=="Direct CT–TS test":
        st.markdown("D = log2(MDC CT) − log2(MDC TS). Negative D means a larger relative CT reduction in AD. Significance compares D with the same paired diagnosis-label permutations, not with zero alone.")
        scope="CT_minus_TS"
        table=frame[frame.scope.eq(scope)].copy()
        row=selected[selected.scope.eq(scope)].iloc[0]
        if nlabel==manifest["label_permutations"]:
            parameters=("label",variant,module,scope,manifest["files"]["label_replicates.parquet"]["sha256"])
            reps=load_replicates(*parameters) if module_set=="muscle_control_l4" else load_replicates(*parameters,module_set)
            render_plot(null_figure(reps,row.statistic,scope),use_container_width=True,plot_id="mdc.validation.contrast")
        else:
            st.info("Label-permutation inference is incomplete. No interim p-values or FDRs are shown.")
    else:
        st.caption("Two-sided equal-tail permutation p-values; BH across valid modules separately for each weighting and component. The null is not assumed to center on MDC=1. Blank heatmap cells are unavailable/unstable tests; raw descriptive values remain in the tables.")
        scope=st.selectbox("Label-test MDC component",list(SCOPES[:-1]),format_func=LABELS.get)
        row=selected[selected.scope.eq(scope)].iloc[0]
        if nlabel==manifest["label_permutations"]:
            parameters=("label",variant,module,scope,manifest["files"]["label_replicates.parquet"]["sha256"])
            reps=load_replicates(*parameters) if module_set=="muscle_control_l4" else load_replicates(*parameters,module_set)
            render_plot(null_figure(reps,row.statistic,scope),use_container_width=True,plot_id="mdc.validation.label_null")
            top_n=st.selectbox("Validation heatmap modules",list(dict.fromkeys([20,50,manifest['module_count']])))
            only_sig=st.toggle("Only modules with significant label tests",value=False)
            if only_sig and not frame.label_fdr.lt(.05).any():
                st.info("No module passes label-test FDR <0.05 for this weighting.")
            else:
                render_plot(heatmap_figure(frame,module,top_n,only_sig),use_container_width=True,plot_id="mdc.validation.heatmap")
        else:
            st.info("Label-permutation inference is incomplete. No interim p-values or FDRs are shown.")
        table=frame[frame.scope.eq(scope)].copy()
    st.subheader(f"Selected module M{module}")
    selected_table=selected[selected.scope.eq(scope)]
    if nbal!=manifest["balanced_resamples"]:
        pending=["balanced_low","balanced_median","balanced_high"]
        selected_table=selected_table.drop(columns=pending,errors="ignore")
        table=table.drop(columns=pending,errors="ignore")
    filterable_dataframe(selected_table,table_key="mdc.validation.selected",table_name="Selected-module validation",hide_index=True)
    filterable_dataframe(table,table_key="mdc.validation.all",table_name="All-module validation",hide_index=True)
    st.download_button("Download complete component validation table",table.to_csv(sep="\t",index=False).encode(),
        file_name=f"{module_set}_{variant}_{scope}_mdc_validation.tsv",mime="text/tab-separated-values")
