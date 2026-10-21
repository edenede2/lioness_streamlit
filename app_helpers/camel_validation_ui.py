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
def matched_screen(phenotype,feature,scale,rule,normalization,ct,ts,diagnoses,method,revision):
    metadata=V.metadata(); metadata=metadata.loc[metadata.diagnosis_group.isin(diagnoses)]
    rows=[]; nominal=phenotype in CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES
    for module in V.manifest()["completed_module_ids"]:
        scores=V.scores(module,feature,rule,normalization,scale=scale,embedded=False)
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


def render_component_screen(phenotype,feature,scale,rule,normalization,diagnoses,method,render):
    st.subheader("Validation CT–TS comparison on identical donors")
    st.caption("Choose a tissue pair and a tissue. Each module uses only donors with both scores and the outcome. A third-region score is unavailable for these donors. Nominal effect differences are descriptive.")
    c1,c2=st.columns(2)
    ct=c1.selectbox("Validation CT pair",[c for c in COMPONENT_LABELS if c.startswith("CT_")],format_func=COMPONENT_LABELS.get)
    ts=c2.selectbox("Validation TS region",[c for c in COMPONENT_LABELS if c.startswith("TS_")],format_func=COMPONENT_LABELS.get)
    if not st.button("Calculate matched-donor module comparison",key="camel_matched_screen"): return
    with st.spinner("Scanning one module at a time…"):
        frame=matched_screen(phenotype,feature,scale,rule,normalization,ct,ts,tuple(diagnoses),method,
                             str(V.manifest().get("generated_utc","")))
    if frame.empty or not frame[["ct_effect","ts_effect"]].notna().any().any():
        st.info("No eligible matched-donor tests. The selected region may be absent from this pair, or group sizes / score variation may be insufficient.")
    else:
        plotted=frame.loc[frame.diagnosis_group.eq("All donors")].dropna(subset=["ct_effect","ts_effect"])
        figure=go.Figure(go.Scatter(x=plotted.ct_effect,y=plotted.ts_effect,mode="markers",text="M"+plotted.module.astype(str),
            customdata=plotted[["n","ct_fdr","ts_fdr"]],hovertemplate="%{text}<br>CT=%{x:.3f}; TS=%{y:.3f}<br>n=%{customdata[0]}<br>CT FDR=%{customdata[1]:.3g}<br>TS FDR=%{customdata[2]:.3g}<extra></extra>"))
        figure.update_layout(template="plotly_white",xaxis_title=COMPONENT_LABELS[ct]+" association effect",yaxis_title=COMPONENT_LABELS[ts]+" association effect")
        render(figure,key="camel_matched_effects")
    filterable_dataframe(frame.sort_values("effect_difference",key=lambda s:s.abs(),ascending=False),table_key="camel_matched_module_table",table_name="Validation matched-donor effects",hide_index=True)
    st.download_button("Download matched-donor results",frame.to_csv(sep="\t",index=False).encode(),"camel_validation_matched_component_statistics.tsv")


def adjacency_figure(module,component,rule):
    path=ensure_data_path(V.root()/f"mdc/adjacency/{component}/M{module}.npz")
    with np.load(path,allow_pickle=False) as z:
        rules=z["rules"].tolist()
        if rule not in rules: raise ValueError("No matching frozen adjacency mask")
        n=int(z["n_nodes"]); e=int(z["n_edges"])
        ri,ci=z["row"],z["column"]
        mask=np.unpackbits(z["mask_bits"][rules.index(rule)])[:e].astype(bool)
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
    f.update_layout(template="plotly_white",title=f"M{module} · {COMPONENT_LABELS[component]} · validation<br>Upper: AD; lower: Control",height=700,
                    xaxis=dict(showticklabels=False,title="Module genes"),yaxis=dict(showticklabels=False,title="Module genes",scaleanchor="x"))
    return f


def render_mdc(module,rule,render):
    st.subheader("Resolved MDC in non-discovery validation donors")
    st.caption("AD/Control mean absolute adjacency ratio; not a single-donor CAMEL score. Masks remain frozen. AC–PCG is descriptive only. No pooled three-region MDC is available.")
    frame=V.mdc(rule)
    if frame.empty:
        st.info("Validation MDC permutations are not yet complete for this selection. Original-cohort MDC will not be substituted.")
        return
    use_raw=st.radio("Validation MDC scale",["Raw MDC","log2 MDC"],horizontal=True)
    values="mdc" if use_raw=="Raw MDC" else "log2_mdc"
    chosen=frame.loc[frame.module.eq(module)]
    fig=go.Figure(go.Bar(x=chosen.component_label,y=chosen[values],customdata=chosen[["n_ad","n_control","directional_fdr"]],
        hovertemplate="%{x}<br>MDC=%{y:.4g}<br>AD n=%{customdata[0]}; Control n=%{customdata[1]}<br>FDR=%{customdata[2]:.3g}<extra></extra>"))
    fig.add_hline(y=1 if values=="mdc" else 0,line_dash="dash",line_color="black")
    fig.update_layout(template="plotly_white",yaxis_title=use_raw,title=f"Validation MDC · M{module}")
    render(fig,key="camel_validation_mdc_bar")
    view=st.radio("Validation MDC view",["Selected module","All-module component heatmap"],horizontal=True)
    if view!="Selected module":
        wide=frame.pivot(index="module",columns="component_label",values=values)
        sig=frame.pivot(index="module",columns="component_label",values="directional_fdr").reindex_like(wide)
        heat=go.Figure(go.Heatmap(z=wide.values,x=wide.columns,y=[f"M{m}" for m in wide.index],
            colorscale="Viridis" if values=="mdc" else "RdBu_r",zmid=None if values=="mdc" else 0,
            text=np.where(sig.values<.05,"*",""),texttemplate="%{text}",colorbar=dict(title=use_raw)))
        heat.update_layout(template="plotly_white",height=950,title="Validation MDC by resolved component")
        render(heat,key="camel_validation_mdc_heatmap")
    else:
        component=st.selectbox("Validation adjacency component",chosen.component.unique(),format_func=COMPONENT_LABELS.get)
        if st.checkbox("Show validation AD/Control triangle heatmap",value=False):
            selected=chosen.loc[chosen.component.eq(component)].iloc[0]
            if selected.n_retained_edges==0: st.info("This module/component/mask has no retained edges.")
            else: render(adjacency_figure(module,component,rule),key="camel_validation_adjacency")
    filterable_dataframe(frame,table_key="camel_validation_mdc_table",table_name="Validation MDC",hide_index=True)
    st.download_button("Download validation MDC",frame.to_csv(sep="\t",index=False).encode(),"camel_validation_mdc.tsv")
