"""Resolved-only validation screens: matched-donor comparisons and group MDC."""
from __future__ import annotations
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from scipy.stats import rankdata
from app_helpers import camel_validation as V
from app_helpers.camel import COMPONENT_LABELS
from app_helpers.correlations import calculate_categorical_associations, benjamini_hochberg
from app_helpers.data import CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES, ASSOCIATION_OUTCOME_LABELS
from app_helpers.drive_data import ensure_data_path
from app_helpers.score_robustness import _fast_corr, _fast_spearman, _williams_test
from app_helpers.table_controls import filterable_dataframe


@st.cache_data(max_entries=1,ttl=1800,show_spinner=False)
def matched_screen(phenotype,feature,scale,rule,normalization,ct,ts,diagnoses,method,revision,
                   module_set="control_derived",network_method="control_anchored"):
    source=V.for_dataset(module_set,network_method)
    metadata=source.metadata(); metadata=metadata.loc[metadata.diagnosis_group.isin(diagnoses)]
    rows=[]; nominal=phenotype in CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES
    for module in source.manifest()["completed_module_ids"]:
        scores=source.scores(module,feature,rule,normalization,scale=scale,embedded=False)
        paired=scores.pivot(index="sample_id",columns="component",values=f"metric_{scale}")
        if ct not in paired or ts not in paired: continue
        paired=paired[[ct,ts]].dropna().merge(metadata,on="sample_id",how="inner",validate="one_to_one")
        paired=paired.dropna(subset=[phenotype])
        groups=[("All donors",paired),*list(paired.groupby("diagnosis_group",sort=False))]
        for diagnosis,group in groups:
            row=dict(module=module,diagnosis_group=diagnosis,n=len(group),ct_component=ct,ts_component=ts,
                     ct_effect=np.nan,ts_effect=np.nan,ct_p=np.nan,ts_p=np.nan,difference_p=np.nan)
            if nominal:
                for component,prefix in ((ct,"ct"),(ts,"ts")):
                    f=group.rename(columns={component:"metric_value"}).assign(test=0)
                    q=calculate_categorical_associations(f,["test"],category_column=phenotype,min_group_n=10).iloc[0] if len(f) else None
                    if q is not None: row.update({prefix+"_effect":q.epsilon_squared,prefix+"_p":q.categorical_p})
            elif len(group)>=10:
                a,b,y=group[ct].to_numpy(float),group[ts].to_numpy(float),pd.to_numeric(group[phenotype],errors="coerce").to_numpy(float)
                kernel=_fast_spearman if method=="Spearman" else _fast_corr
                _,row["ct_effect"],row["ct_p"]=kernel(a,y)
                _,row["ts_effect"],row["ts_p"]=kernel(b,y)
                if method=="Spearman": a,b,y=rankdata(a),rankdata(b),rankdata(y)
                _,ab,_=_fast_corr(a,b)
                _,row["difference_p"]=_williams_test(row["ct_effect"],row["ts_effect"],ab,len(group))
            row["effect_difference"]=row["ct_effect"]-row["ts_effect"]
            rows.append(row)
    frame=pd.DataFrame(rows)
    if frame.empty: return frame
    for prefix in ("ct","ts","difference"):
        frame[prefix+"_fdr"]=frame.groupby("diagnosis_group")[prefix+"_p"].transform(benjamini_hochberg)
    return frame


def render_component_screen(phenotype,feature,scale,rule,normalization,diagnoses,method,render,*,
                            module_set="control_derived",network_method="control_anchored"):
    source=V.for_dataset(module_set,network_method); labels=source.config.labels
    st.subheader("Validation CT–TS comparison on identical donors")
    st.caption("Choose a tissue pair and a tissue. Each module uses only donors with both scores and the outcome. A third-region score is unavailable for these donors. Nominal effect differences are descriptive.")
    c1,c2=st.columns(2)
    ct=c1.selectbox("Validation CT pair",[c for c in labels if c.startswith("CT_")],format_func=labels.get)
    ts=c2.selectbox("Validation TS region",[c for c in labels if c.startswith("TS_")],format_func=labels.get)
    if not st.button("Calculate matched-donor module comparison",key="camel_matched_screen"): return
    with st.spinner("Scanning one module at a time…"):
        frame=matched_screen(phenotype,feature,scale,rule,normalization,ct,ts,tuple(diagnoses),method,
                             str(source.manifest().get("generated_utc","")),module_set,network_method)
    if frame.empty or not frame[["ct_effect","ts_effect"]].notna().any().any():
        st.info("No eligible matched-donor tests. The selected region may be absent from this pair, or group sizes / score variation may be insufficient.")
    else:
        plotted=frame.loc[frame.diagnosis_group.eq("All donors")].dropna(subset=["ct_effect","ts_effect"])
        figure=go.Figure(go.Scatter(x=plotted.ct_effect,y=plotted.ts_effect,mode="markers",text="M"+plotted.module.astype(str),
            customdata=plotted[["n","ct_fdr","ts_fdr"]],hovertemplate="%{text}<br>CT=%{x:.3f}; TS=%{y:.3f}<br>n=%{customdata[0]}<br>CT FDR=%{customdata[1]:.3g}<br>TS FDR=%{customdata[2]:.3g}<extra></extra>"))
        figure.update_layout(template="plotly_white",xaxis_title=labels[ct]+" association effect",yaxis_title=labels[ts]+" association effect")
        render(figure,key="camel_matched_effects")
    filterable_dataframe(frame.sort_values("effect_difference",key=lambda s:s.abs(),ascending=False),table_key="camel_matched_module_table",table_name="Validation matched-donor effects",hide_index=True)
    st.download_button("Download matched-donor results",frame.to_csv(sep="\t",index=False).encode(),"camel_validation_matched_component_statistics.tsv")


def adjacency_figure(module,component,rule,*,module_set="control_derived",network_method="control_anchored"):
    source=V.for_dataset(module_set,network_method)
    path=ensure_data_path(source.root()/f"mdc/adjacency/{component}/M{module}.npz")
    with np.load(path,allow_pickle=False) as z:
        rules=z["rules"].tolist()
        if rule not in rules: raise ValueError("No matching frozen adjacency mask")
        n=int(z["n_nodes"]); e=int(z["n_edges"])
        ri,ci=z["row"],z["column"]
        mask=np.unpackbits(z["mask_bits"][rules.index(rule)])[:e].astype(bool)
        if rule=="all" and "canonical_nodes" in z.files:
            canonical=z["canonical_nodes"].astype(bool)
            positions=np.full(n,-1,int);positions[canonical]=np.arange(canonical.sum())
            mask &= canonical[ri]&canonical[ci]
            ri,ci=positions[ri],positions[ci];n=int(canonical.sum())
        matrix=np.full((n,n),np.nan,dtype=np.float32)
        matrix[ri[mask],ci[mask]]=np.abs(z["control"][mask])
        matrix[ci[mask],ri[mask]]=np.abs(z["ad"][mask])
    finite=matrix[np.isfinite(matrix)]
    cap=float(np.quantile(finite,.99)) if len(finite) else 1.
    if not np.isfinite(cap) or cap<=0: cap=1.
    scaled=np.nan_to_num(np.clip(matrix/cap,0,1),nan=0.)
    pixels=np.stack([np.full_like(scaled,255),255*(1-scaled),np.zeros_like(scaled)],axis=-1).astype(np.uint8)
    pixels[~np.isfinite(matrix)]=255
    image=Image.fromarray(pixels).resize((1200,1200),resample=Image.Resampling.NEAREST)
    blob=io.BytesIO(); image.save(blob,format="PNG")
    import base64
    f=go.Figure(go.Image(source="data:image/png;base64,"+base64.b64encode(blob.getvalue()).decode(),hoverinfo="skip"))
    f.add_trace(go.Scatter(x=[None,None],y=[None,None],mode="markers",showlegend=False,
        marker=dict(color=[0,cap],colorscale=[[0,"#FFFF00"],[.5,"#FF8000"],[1,"#FF0000"]],cmin=0,cmax=cap,showscale=True,
                    colorbar=dict(title="Absolute adjacency weight")),hoverinfo="skip"))
    f.update_layout(template="plotly_white",title=f"M{module} · {source.config.labels[component]} · validation<br>Upper: AD; lower: Control",height=700,
                    xaxis=dict(showticklabels=False,title="Module genes"),yaxis=dict(showticklabels=False,title="Module genes",scaleanchor="x"))
    return f


def render_mdc(module,rule,render,*,module_set="control_derived",network_method="control_anchored"):
    source=V.for_dataset(module_set,network_method); labels=source.config.labels
    st.subheader("Resolved MDC in non-discovery validation donors")
    st.caption("AD/Control mean absolute adjacency ratio; not a single-donor score. Masks remain frozen. Components with fewer than five AD or five Controls are descriptive only. No pooled three-tissue MDC is available.")
    frame=source.mdc(rule)
    if frame.empty:
        st.info("Validation MDC permutations are not yet complete for this selection. Original-cohort MDC will not be substituted.")
        return
    original = frame
    settings = st.columns(3)
    with settings[0]:
        use_raw=st.radio("Validation MDC scale",["Raw MDC","log2 MDC"],horizontal=True)
    with settings[1]:
        minimum = st.number_input("Minimum retained edges per MDC component", min_value=0,
            value=0, step=1, key="mdc_minimum_edges",
            help="0 keeps the original catalog. Try 10, 50 or 100 to hide ratios supported by very few retained edges. This is a display filter, not a new significance test.")
    with settings[2]:
        threshold = st.radio("Validation MDC significance threshold", [.05,.10], horizontal=True,
            key="validation_mdc_threshold")
    from app_helpers.validation_mdc_plots import filter_retained_edges, comparison_figure, entropy_figure
    frame = filter_retained_edges(original, minimum)
    st.caption(f"Showing {len(frame):,}/{len(original):,} module–component rows after the retained-edge filter. Saved MDC values and full-family FDRs are unchanged; no correction is recalculated after filtering.")
    st.caption("Directional FDR is the historical two-null screen, in the direction selected by the observed AD/Control ratio; it is not an established joint-FDR guarantee. Biological direction comes from the ratio, not q_loss/q_gain column names.")
    if frame.empty:
        st.info("No MDC components pass this edge-count threshold. Lower it to restore the plots.")
        return
    values="mdc" if use_raw=="Raw MDC" else "log2_mdc"
    scale="raw" if use_raw=="Raw MDC" else "log2"
    from app_helpers.data import MODULE_SET_LABELS, load_module_details
    definition=MODULE_SET_LABELS.get(module_set,module_set)
    metadata=dict(module_set=module_set, method=network_method, module=int(module),
        cohort_scope="incomplete_region_validation", differential_edge_rule=rule,
        scale=scale, minimum_retained_edges=int(minimum), significance_threshold=threshold)
    def draw(figure, key, metadata=metadata):
        figure.update_layout(meta={**dict(figure.layout.meta or {}), **metadata})
        render(figure,key=key,metadata=metadata)
    chosen=frame.loc[frame.module.eq(module)]
    fig=go.Figure(go.Bar(x=chosen.component_label,y=chosen[values],customdata=chosen[["n_ad","n_control","directional_fdr","n_retained_edges"]],
        hovertemplate="%{x}<br>MDC=%{y:.4g}<br>AD n=%{customdata[0]}; Control n=%{customdata[1]}<br>Saved FDR=%{customdata[2]:.3g}<br>Retained edges=%{customdata[3]:,}<extra></extra>"))
    fig.add_hline(y=1 if values=="mdc" else 0,line_dash="dash",line_color="black")
    fig.update_layout(template="plotly_white",yaxis_title=use_raw,title=f"Validation MDC · M{module}")
    draw(fig,key="camel_validation_mdc_bar")
    if chosen.empty:
        st.info("The selected module has no components above the edge-count threshold. Other qualifying modules remain available below.")
    view=st.radio("Validation MDC view",["Selected module","MDC scatter plots","All-module component heatmap"],
        index=1,horizontal=True,key="validation_mdc_view")
    if view=="All-module component heatmap":
        wide=frame.pivot(index="module",columns="component_label",values=values)
        sig=frame.pivot(index="module",columns="component_label",values="directional_fdr").reindex_like(wide)
        heat=go.Figure(go.Heatmap(z=wide.values,x=wide.columns,y=[f"M{m}" for m in wide.index],
            colorscale="Viridis" if values=="mdc" else "RdBu_r",zmid=None if values=="mdc" else 0,
            text=np.where(sig.values<threshold,"*",""),texttemplate="%{text}",colorbar=dict(title=use_raw)))
        heat.update_layout(template="plotly_white",height=950,title="Validation MDC by resolved component")
        draw(heat,key="camel_validation_mdc_heatmap")
    elif view=="MDC scatter plots":
        available = list(original.component.drop_duplicates())
        ts_options = [c for c in available if c.startswith("TS_")]
        ct_options = [c for c in available if c.startswith("CT_")]
        if not ts_options or not ct_options:
            st.info("Both a within-region and a between-region component are required for this comparison.")
        else:
            a,b=st.columns(2)
            ts=a.selectbox("MDC scatter TS tissue",ts_options,
                index=ts_options.index("TS_DLPFC") if "TS_DLPFC" in ts_options else 0,
                format_func=labels.get,key="validation_mdc_scatter_ts")
            preferred=next((c for c in ct_options if "DLPFC" in c and "PCG" in c),ct_options[0])
            ct=b.selectbox("MDC scatter CT pair",ct_options,index=ct_options.index(preferred),
                format_func=labels.get,key="validation_mdc_scatter_ct")
            st.caption("Each dot is one module. The comparison requires both selected components to pass the edge threshold. Component cohorts can differ; this is not a matched-donor test. No pooled three-region MDC is constructed.")
            figure=comparison_figure(frame,ts,ct,labels,module,threshold,scale,definition)
            draw(figure,key="camel_validation_mdc_scatter",metadata=dict(metadata,ts_component=ts,ct_component=ct))
            details=load_module_details(module_set=module_set)
            st.markdown("#### MDC and tissue-mixing entropy")
            for column,component,scope in zip(st.columns(2),(ts,ct),("ts","ct")):
                with column:
                    figure=entropy_figure(frame,component,scope,labels,details,module,threshold,scale,definition)
                    draw(figure,key="camel_validation_mdc_entropy_"+scope,
                        metadata=dict(metadata,component=component))
            st.caption("Entropy and MDC associations describe the displayed modules. OLS is a visual guide; the subtitle reports Spearman correlation across modules, not donors. Saved MDC FDRs keep their original families.")
    elif not chosen.empty:
        component=st.selectbox("Validation adjacency component",chosen.component.unique(),format_func=labels.get)
        if st.checkbox("Show validation AD/Control triangle heatmap",value=False):
            selected=chosen.loc[chosen.component.eq(component)].iloc[0]
            if selected.n_retained_edges==0: st.info("This module/component/mask has no retained edges.")
            else: draw(adjacency_figure(module,component,rule,module_set=module_set,network_method=network_method),key="camel_validation_adjacency",metadata=dict(metadata,component=component))
    filterable_dataframe(frame,table_key="camel_validation_mdc_table",table_name="Validation MDC",hide_index=True)
    st.download_button("Download validation MDC",frame.to_csv(sep="\t",index=False).encode(),"camel_validation_mdc.tsv")
