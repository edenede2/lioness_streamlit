"""Aggregate-only matched-Control release, independent of legacy module routing."""
from __future__ import annotations
import numpy as np
import textwrap
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app_helpers import matched_brain_results as D
from app_helpers.data import ASSOCIATION_OUTCOME_LABELS,FEATURE_LABELS,selected_kegg_table_annotation
from app_helpers.table_controls import filterable_dataframe
from app_helpers.charts import module_entropy_figure,module_size_distribution_figure,module_region_composition_figure

TITLE="Matched-control results"
SECTIONS=["CAMEL associations","CAMEL AD–Control differences","MDC","KEGG enrichment","Module composition","Cohorts & technical checks","Catalog & downloads"]
NORMALIZATIONS={"standard_pruned":"Standard pruned","retained_edge":"Retained-edge normalized","unfiltered":"Unfiltered eigengene"}
COHORT_SHORT={
    "Held-out validation":"Held-out",
    "Discovery-demography-matched incomplete-region validation (AD–Control)":"Discovery-like",
    "Balanced incomplete-region validation (AD–Control)":"AD–Control matched",
    "Incomplete-region validation":"All incomplete",
    "Discovery":"Discovery"}


def _table(frame,key,title):
    filterable_dataframe(frame,table_key="matched_"+key,table_name=title,hide_index=True)
    st.download_button("Download "+title.lower(),frame.to_csv(sep="\t",index=False).encode(),
        file_name="matched_"+key+".tsv",mime="text/tab-separated-values",key="matched_download_"+key)


def _layout(fig):
    fig.update_layout(template="plotly_white",legend=dict(orientation="h",y=1.12,yanchor="bottom",x=0),
        margin=dict(t=110,b=65,l=85,r=35),
        meta={"figure_export":{"fixed_container_title":True}})
    return fig


def _labels(series,mapping):
    # Parquet predicates retain dictionary/categorical columns. Convert only
    # these small display vectors, not the million-row underlying table.
    return series.astype(str).map(lambda value:mapping.get(value,value))


def effect_figure(frame,k=20,stars=True):
    f=frame.loc[frame.effect.notna()].copy()
    if f.empty: return None
    order=f.groupby("module").effect.apply(lambda v:abs(v).max()).sort_values(ascending=False,kind="stable").head(k).index
    f=f.loc[f.module.isin(order)].copy()
    f["module_label"]=f.module.map(lambda m:f"M{int(m)}")
    f["comparison"]=_labels(f.cohort_scope,COHORT_SHORT)+" · "+_labels(f.component,D.COMPONENTS)
    if f.comparison.nunique()==1:
        f=f.set_index("module").loc[order].reset_index()
        text=[f"{v:.3f}"+("*" if stars and q<.05 else "") for v,q in zip(f.effect,f.q_permutation)]
        fig=go.Figure(go.Bar(x=f.effect,y=f.module_label,orientation="h",text=text,textposition="outside",
            marker_color=np.where(f.effect>=0,"#D55E00","#0072B2"),name="Module effect",showlegend=False,cliponaxis=False,
            customdata=f[["n","p_permutation","q_permutation","q_permutation_family_n"]].to_numpy(),
            hovertemplate="%{y}<br>Effect=%{x:.4f}<br>N=%{customdata[0]}<br>p=%{customdata[1]:.3g}<br>FDR=%{customdata[2]:.3g}<br>Tested modules=%{customdata[3]}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        low=min(0.,float(f.effect.min()));high=max(0.,float(f.effect.max()))
        padding=.16*max(abs(low),abs(high),np.finfo(float).eps)
        fig.update_xaxes(range=[low-padding if low<0 else 0,high+padding if high>0 else 0] if low!=high else [-.05,.05])
    else:
        f["label"]=f.effect.map(lambda v:f"{v:.3f}")+np.where(stars&f.q_permutation.lt(.05),"*","")
        matrix=f.pivot(index="module_label",columns="comparison",values="effect").reindex([f"M{m}" for m in order])
        text=f.pivot(index="module_label",columns="comparison",values="label").reindex_like(matrix)
        extra=np.stack([f.pivot(index="module_label",columns="comparison",values=c).reindex_like(matrix).to_numpy()
            for c in ("n","p_permutation","q_permutation","q_permutation_family_n")],axis=-1)
        fig=go.Figure(go.Heatmap(z=matrix.to_numpy(),x=matrix.columns,y=matrix.index,text=text.to_numpy(),customdata=extra,
            texttemplate="%{text}",colorscale="RdBu_r",zmid=0,colorbar=dict(title="Effect"),
            hovertemplate="%{y}<br>%{x}<br>Effect=%{z:.4f}<br>N=%{customdata[0]}<br>p=%{customdata[1]:.3g}<br>FDR=%{customdata[2]:.3g}<br>Tested modules=%{customdata[3]}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
    label="AD − Control mean score" if frame.test_type.str.startswith("score_").all() else (
        "Partial Spearman ρ" if frame.correlation_method.eq("spearman").all() else "Partial Pearson r")
    fig.update_layout(height=max(430,min(1000,30*len(order)+160)),xaxis_title=label,yaxis_title="Module")
    outcome=ASSOCIATION_OUTCOME_LABELS.get(str(f.phenotype.iloc[0]),str(f.phenotype.iloc[0]))
    feature=FEATURE_LABELS.get(str(f.metric_family.iloc[0]),str(f.metric_family.iloc[0]))
    title=("AD–Control score difference: "+feature) if label.startswith("AD") else (feature+" vs "+outcome)
    context=str(f.comparison.iloc[0]) if f.comparison.nunique()==1 else "Selected cohorts and components"
    fig.update_layout(title=dict(text="<br>".join(textwrap.wrap(title+" · "+context+" · "+str(f.scale.iloc[0]),60)),
        x=.01,xanchor="left",automargin=True,font=dict(size=14)))
    return _layout(fig)


def _cohort_components(prefix,*,baseline=False):
    cohorts=list(D.COHORTS)
    if baseline: cohorts=["Discovery",*cohorts]
    default=[cohorts[1]] if not baseline else ["Discovery"]
    selected=st.multiselect("Comparison cohorts",cohorts,default=default,
        format_func=lambda c:D.COHORTS.get(c,c),key=prefix+"_cohorts")
    comps=list(D.COMPONENTS)
    if any(c!="Held-out validation" and c!="Discovery" for c in selected):
        comps=[c for c in comps if c not in {"all","TS","CT"}]
    components=st.multiselect("Components",comps,default=["TS_DLPFC"],format_func=D.COMPONENTS.get,key=prefix+"_components")
    return selected,components


def _edge_scale(prefix,feature=None):
    a,b,c=st.columns(3)
    rule=a.selectbox("Frozen discovery edge set",list(D.RULES),format_func=D.RULES.get,key=prefix+"_rule")
    norm=b.selectbox("Score handling",["unfiltered"] if feature=="eigengene" else ["standard_pruned","retained_edge"],
        format_func=NORMALIZATIONS.get,key=prefix+"_norm")
    scale=c.selectbox("Score transformation",["raw","asinh","rint"],format_func=lambda v:{"raw":"Raw","asinh":"asinh","rint":"RINT (frozen discovery mapping)"}[v],key=prefix+"_scale")
    if feature=="eigengene" and rule!="all":
        st.info("Eigengenes have no edge mask; this selection uses the unfiltered eigengene.")
        rule="all"
    return rule,norm,scale


def _display_effects(frame,render,prefix):
    if frame.empty:
        st.info("This exact combination is unavailable. No result from another cohort or module definition is substituted.")
        return
    a,b,c=st.columns(3)
    cutoff=a.selectbox("Display FDR filter",[None,.05,.10],format_func=lambda v:"No significance filter" if v is None else f"FDR < {v:.2f}",key=prefix+"_cutoff")
    k=b.number_input("Top modules in plot",min_value=1,max_value=247,value=20,step=1,key=prefix+"_k")
    stars=c.checkbox("Mark FDR < 0.05 with *",value=True,key=prefix+"_stars")
    focus=st.multiselect("Focus modules (optional)",sorted(frame.module.unique()),format_func=lambda m:f"M{m}",key=prefix+"_focus")
    shown=frame.loc[frame.module.isin(focus)] if focus else frame
    if cutoff: shown=shown.loc[shown.q_permutation.lt(cutoff)]
    a,b,c=st.columns(3)
    a.metric("Module–component–cohort rows",len(frame))
    b.metric("Valid tests",int(frame.p_permutation.notna().sum()))
    c.metric("FDR < 0.05",int(frame.q_permutation.lt(.05).sum()))
    fig=effect_figure(shown,int(k),stars)
    if fig is None: st.info("No nonconstant eligible effects remain in this display selection.")
    else: render(fig,key=prefix+"_effect")
    st.caption("Top K and table filters affect display only. All p-values, FDRs and tested-module counts come from the complete precomputed module family.")
    columns=[c for c in ["module","cohort_scope","component","metric_family","score_normalization","differential_edge_rule",
        "scale","phenotype","correlation_method","effect","r","n","n_ad","n_control","p_permutation","q_permutation",
        "q_permutation_family_n","p_nominal","q_nominal","q_nominal_family_n","covariates","df","permutations",
        "exact_enumeration","eligible","unavailable_reason"] if c in shown]
    _table(shown[columns],prefix,"Selected-context module statistics")
    st.download_button("Download all modules before display filters",frame.to_csv(sep="\t",index=False).encode(),
        file_name=prefix+"_all_modules.tsv",key=prefix+"_all_download")


def associations(render,group_differences=False):
    prefix="matched_difference" if group_differences else "matched_association"
    st.subheader("Frozen CAMEL AD–Control score differences" if group_differences else "Demographic-adjusted CAMEL and eigengene associations")
    cohorts,components=_cohort_components(prefix)
    test=st.selectbox("Label-swap test",["score_matched_pair_label","score_diagnosis_label"],format_func=D.TESTS.get,key=prefix+"_test") if group_differences else "covariate_adjusted_association"
    a,b=st.columns(2)
    feature_options=D.options("metric_family")
    feature=a.selectbox("Module score feature",feature_options,index=feature_options.index("connectivity") if "connectivity" in feature_options else 0,
        format_func=lambda v:FEATURE_LABELS.get(v,v),key=prefix+"_feature")
    if not group_differences:
        outcomes=[x for x in D.options("phenotype",test=test) if x!="diagnosis_AD_Control"]
        outcome=b.selectbox("Outcome",outcomes,index=outcomes.index("cogn_global") if "cogn_global" in outcomes else 0,
            format_func=lambda v:ASSOCIATION_OUTCOME_LABELS.get(v,v),key=prefix+"_outcome")
        method=st.radio("Correlation method",["spearman","pearson"],format_func=str.capitalize,horizontal=True,key=prefix+"_method")
    else: outcome="diagnosis_AD_Control";method="not_applicable"
    rule,norm,scale=_edge_scale(prefix,feature)
    if not cohorts or not components:
        st.info("Select at least one cohort and component.");return
    pieces=[D.camel(cohort,component,test_type=test,metric_family=feature,score_normalization=norm,
        differential_edge_rule=rule,scale=scale,phenotype=outcome,correlation_method=method)
        for cohort in cohorts for component in components]
    frame=pd.concat(pieces,ignore_index=True)
    if group_differences:
        st.caption("Donor scores, references and discovery masks stay frozen. The test permutes whole-donor diagnosis labels; matched swaps stay within demographic pairs. Positive effects mean higher scores in AD.")
    else:
        st.caption("Partial correlations adjust age, education, sex and ROS/MAP study. When age or education is the outcome, that field is not adjusted away. Supplied cognitive slopes are not re-estimated.")
        st.caption("Permutation p-values use reduced-model residual permutations within study/sex. This assumes an adequate nuisance model and residual exchangeability; it is not a causal test.")
    if rule!="all": st.caption("Edges were selected in discovery AD/Control donors; validation labels never relearn the mask.")
    _display_effects(frame,render,prefix)


def mdc_figure(frame,scale,stars=True,k=30):
    value="mdc" if scale=="Raw MDC" else "log2_mdc"
    f=frame.loc[np.isfinite(frame[value])].copy()
    if f.empty: return None
    f["_deviation"]=abs(f.log2_mdc)
    order=f.groupby("module")["_deviation"].max().sort_values(ascending=False,kind="stable").head(k).index
    f=f.loc[f.module.isin(order)]
    f["cell"]=_labels(f.cohort_scope,COHORT_SHORT)+" · "+_labels(f.component,D.COMPONENTS)
    f["module_label"]=f.module.map(lambda v:f"M{v}")
    matrix=f.pivot(index="module_label",columns="cell",values=value).reindex([f"M{m}" for m in order])
    sig=f.assign(marker=np.where(stars&f.display_fdr.lt(.05),"*","")).pivot(index="module_label",columns="cell",values="marker").reindex_like(matrix)
    diverging=scale!="Raw MDC" and np.nanmin(matrix)<0<np.nanmax(matrix)
    color="YlOrRd" if scale=="Raw MDC" else ("RdBu_r" if diverging else "Blues_r" if np.nanmax(matrix)<=0 else "Reds")
    fig=go.Figure(go.Heatmap(z=matrix.to_numpy(),x=matrix.columns,y=matrix.index,text=sig.to_numpy(),
        texttemplate="%{text}",colorscale=color,zmid=0 if diverging else None,colorbar=dict(title=scale),
        hovertemplate="%{y}<br>%{x}<br>MDC scale value=%{z:.4f}<extra></extra>"))
    fig.update_layout(height=max(450,min(1100,25*len(order)+180)),title="Module MDC · "+scale)
    fig.update_yaxes(autorange="reversed")
    return _layout(fig)


def mdc(render):
    st.subheader("MDC and donor-label permutation evidence")
    source=st.selectbox("MDC significance test",["matched_pair_label","diagnosis_label","legacy"],
        format_func=lambda v:D.TESTS.get(v,"Existing sample/gene nulls"),key="matched_mdc_source")
    cohorts,components=_cohort_components("matched_mdc",baseline=source=="legacy")
    a,b,c=st.columns(3)
    rule=a.selectbox("MDC edge set",list(D.RULES),format_func=D.RULES.get,key="matched_mdc_rule")
    direction=b.selectbox("MDC test direction",["two_sided","gain","loss"] if source!="legacy" else ["gain","loss"],
        format_func=lambda v:{"two_sided":"Two-sided","gain":"Higher in AD","loss":"Higher in Control"}[v],key="matched_mdc_direction")
    scale=c.radio("MDC scale",["Log2 MDC","Raw MDC"],horizontal=True,key="matched_mdc_scale")
    frame=D.table("mdc_baseline" if source=="legacy" else "mdc_labels")
    if source!="legacy": frame=frame.loc[frame.test_type.eq(source)]
    frame=frame.loc[frame.cohort_scope.isin(cohorts)&frame.component.isin(components)&frame.differential_edge_rule.eq(rule)].copy()
    if frame.empty:
        st.info("No MDC test exists for this exact selection. The discovery-demography subset has label-swap tests only; unrestricted incomplete donors have no matched-pair null.");return
    frame["display_p"]=frame["p_"+direction] if source!="legacy" else np.nan
    frame["display_fdr"]=frame["q_"+direction]
    st.caption("MDC is the AD/Control ratio of mean absolute group-network adjacency, not an average donor CAMEL score. Raw MDC=1 or log2 MDC=0 means equal connectivity.")
    st.caption("FDR is across valid modules separately for cohort, component, edge set and null. Existing sample/gene FDR remains the larger of the two null-specific FDRs; label-test FDRs are separate evidence.")
    if source=="matched_pair_label": st.caption("Pair membership is fixed. Sex and study match exactly; age and education use two-year calipers. Matching does not remove unmeasured confounding.")
    a,b,c=st.columns(3)
    cutoff=a.selectbox("MDC display FDR filter",[None,.05,.10],format_func=lambda v:"No significance filter" if v is None else f"FDR < {v:.2f}",key="matched_mdc_cutoff")
    k=b.number_input("MDC top modules",min_value=1,max_value=247,value=30,key="matched_mdc_top")
    stars=c.checkbox("MDC significance stars",value=True,key="matched_mdc_stars")
    shown=frame.loc[frame.display_fdr.lt(cutoff)] if cutoff else frame
    fig=mdc_figure(shown,scale,stars,int(k))
    if fig is None: st.info("No finite MDC values pass these display filters.")
    else: render(fig,key="matched_mdc_heatmap")
    _table(shown,"mdc_statistics","MDC statistics")
    if st.checkbox("Compare MDC with discovery",key="matched_mdc_compare"):
        baseline=D.table("mdc_baseline")
        baseline=baseline.loc[baseline.cohort_scope.eq("Discovery")&baseline.differential_edge_rule.eq(rule),["module","component","mdc","log2_mdc"]]
        paired=frame.merge(baseline,on=["module","component"],suffixes=("_validation","_discovery"),validate="many_to_one")
        v="mdc" if scale=="Raw MDC" else "log2_mdc"
        paired=paired.dropna(subset=[v+"_discovery",v+"_validation"])
        if paired.empty: st.info("No finite discovery comparison.")
        else:
            fig=px.scatter(paired,x=v+"_discovery",y=v+"_validation",color="component",
                hover_data=["module","cohort_scope","n_ad","n_control","display_fdr"])
            lo=float(paired[[v+"_discovery",v+"_validation"]].min().min());hi=float(paired[[v+"_discovery",v+"_validation"]].max().max())
            fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode="lines",name="Equal MDC",line=dict(color="black",dash="dash")))
            render(_layout(fig),key="matched_mdc_discovery_scatter")
            st.caption("This is descriptive concordance. Discovery and validation were not selected or tested identically; matching directions alone is not statistical replication.")


def kegg(render,module):
    st.subheader("Full-module and frozen-edge endpoint KEGG")
    a,b=st.columns(2)
    rule=a.selectbox("Enrichment gene set",list(D.RULES),format_func=D.RULES.get,key="matched_kegg_rule")
    regional=b.radio("Enrichment scope",["Tissue-expanded","Region-specific"],horizontal=True,key="matched_kegg_scope")=="Region-specific"
    f=D.kegg(rule,regional)
    if not st.checkbox("All modules in KEGG table",key="matched_kegg_all"): f=f.loc[f.cluster_id.eq(module)]
    a,b=st.columns(2)
    search=a.text_input("Pathway/category contains",key="matched_kegg_search")
    cutoff=b.selectbox("KEGG FDR cutoff",[.05,.10,None],format_func=lambda v:"No FDR filter" if v is None else str(v),key="matched_kegg_cutoff")
    if search:
        mask=f[["category_level1","category_level2","pathway_name"]].fillna("").astype(str).agg(" ".join,axis=1).str.contains(search,case=False,regex=False)
        f=f.loc[mask]
    if regional:
        alpha=cutoff if cutoff is not None else .05
        f=f.copy();f["significant_region_count"]=sum(f["fdr_"+t].lt(alpha) for t in ("AC","DLPFC","PCG"))
        shared=st.selectbox("Significant regional sharing",["Any","At least two regions","All three regions"],key="matched_kegg_shared")
        if shared!="Any": f=f.loc[f.significant_region_count.ge(2 if shared=="At least two regions" else 3)]
        elif cutoff is not None: f=f.loc[f.significant_region_count.gt(0)]
        plot=f.sort_values(["fdr_AC","fdr_DLPFC","fdr_PCG"]).head(35).copy()
        if not plot.empty:
            labels="M"+plot.cluster_id.astype(str)+" · "+plot.pathway_name.astype(str)
            z=-np.log10(plot[["fdr_AC","fdr_DLPFC","fdr_PCG"]].clip(lower=1e-300).to_numpy())
            fig=go.Figure(go.Heatmap(z=z,x=["AC","DLPFC","PCG"],y=labels,colorscale="YlOrRd",
                customdata=plot[["fdr_AC","fdr_DLPFC","fdr_PCG"]].to_numpy(),
                hovertemplate="%{y}<br>%{x}<br>FDR=%{customdata:.3g}<extra></extra>"))
            fig.update_layout(height=max(440,min(1100,len(plot)*25+160)))
            render(_layout(fig),key="matched_kegg_heatmap")
    else:
        if cutoff is not None: f=f.loc[f.fdr.lt(cutoff)]
        f=f.sort_values(["fdr","cluster_id"])
    st.caption("Enrichment FDR is separate from association and MDC FDR. Endpoint ORA is exploratory: genes with more retained edges are more likely to enter its gene set.")
    _table(f.head(2000),"kegg","Selected KEGG enrichments")
    st.caption(f"{len(f):,} matching rows; the displayed table is capped at 2,000 for memory. The download below contains all matching rows.")
    st.download_button("Download all matching KEGG rows",f.to_csv(sep="\t",index=False).encode(),"matched_kegg_full_selection.tsv",key="matched_kegg_full")


def composition(render,module):
    frame=D.table("module_details").copy()
    for tissue in ("AC","DLPFC","PCG"): frame["n_genes_"+tissue.lower()]=frame[tissue]
    types=st.multiselect("Module types",["CT","TS"],default=["CT","TS"],key="matched_composition_types")
    frame=frame.loc[frame.cluster_type.isin(types)]
    highlight=st.checkbox("Highlight inspected module",value=False,key="matched_entropy_highlight")
    render(module_entropy_figure(frame,module if highlight else -1,"Matched-Control discovery L4"),key="matched_entropy")
    render(module_size_distribution_figure(frame,"Matched-Control discovery L4"),key="matched_sizes")
    a,b=st.columns(2)
    orientation=a.radio("Composition orientation",["h","v"],format_func=lambda v:"Horizontal" if v=="h" else "Vertical",key="matched_composition_orientation")
    scale=b.radio("Composition scale",["counts","proportions"],format_func=lambda v:"Gene counts" if v=="counts" else "100% proportions",key="matched_composition_scale")
    render(module_region_composition_figure(frame,"Matched-Control discovery L4",orientation=orientation,composition_scale=scale),key="matched_composition")
    _table(frame,"composition","Module composition")


def cohorts():
    a=D.audits()
    st.subheader("Outcome-blind cohorts and technical safeguards")
    design=a["discovery_design"]
    rows=[dict(cohort=k,**v["counts"]) for k,v in design["cohorts"].items()]
    _table(pd.DataFrame(rows),"cohort_counts","Complete-cohort diagnosis counts")
    rows=[]
    for key,label in [("balanced_validation","AD–Control matched incomplete"),("discovery_demography_validation","Discovery-demography matched incomplete")]:
        for item in a[key]["components"]:
            rows.append(dict(cohort=label,component=D.COMPONENTS.get(item["component"],item["component"]),
                matched_pairs=item["matched_pairs"],AD=item.get("matched_AD",item.get("n_AD")),
                Control=item.get("matched_Control",item.get("n_Control")),unavailable_reason=item.get("insufficient_reason",item.get("unavailable_reason",""))))
    _table(pd.DataFrame(rows),"matching_counts","Component-specific validation pairs")
    st.caption("Component cohorts overlap and their counts must not be summed. All 464 complete-expression donors were excluded from incomplete-region validation; absent regions were never imputed.")
    balance=[]
    for group,label in [("balanced_validation","AD–Control matched"),("discovery_demography_validation","Discovery-demography matched")]:
        for item in a[group]["components"]:
            values=item.get("balance",item.get("within_validation_balance",[]))
            balance.extend(dict(cohort=label,component=D.COMPONENTS[item["component"]],**v) for v in values)
    _table(pd.DataFrame(balance),"matching_balance","Within-cohort demographic balance")
    cross=[]
    for item in a["discovery_demography_validation"]["components"]:
        cross.extend(dict(component=D.COMPONENTS[item["component"]],**v) for v in item["discovery_demographic_comparison"])
    _table(pd.DataFrame(cross),"discovery_balance","Validation versus discovery demographics")
    with st.expander("Numerical pilot and export provenance"):
        st.json({"pilot":a["pilot"],"export_recovery":a["export_recovery"]})
    st.caption("No discovery inferential label-test p-values were added: discovery diagnoses helped select the modules and edge masks. These validation tests condition on those frozen objects.")


def render_view(render):
    st.header(TITLE)
    if not D.available():
        st.info("The matched-Control result bundle is not installed; existing analyses remain unchanged.");return
    manifest=D.manifest()
    st.caption("247 new L4 modules learned from 80 matched discovery Controls. Module IDs belong to this partition and must not be equated with the older 186- or 154-module definitions.")
    section=st.radio("Matched-control results section",SECTIONS,horizontal=True,key="matched_results_section")
    details=D.table("module_details")
    module=st.selectbox("Inspect module",sorted(details.module.astype(int)),format_func=lambda m:f"M{m}",key="matched_inspected_module")
    with st.expander("Inspected-module enrichment"):
        annotation=selected_kegg_table_annotation(D.kegg().loc[lambda f:f.cluster_id.eq(module)])
        st.write(annotation or "No qualifying enrichment is available for this module.")
    if section=="CAMEL associations": associations(render)
    elif section=="CAMEL AD–Control differences": associations(render,True)
    elif section=="MDC": mdc(render)
    elif section=="KEGG enrichment": kegg(render,module)
    elif section=="Module composition": composition(render,module)
    elif section=="Cohorts & technical checks": cohorts()
    else:
        st.subheader("Complete aggregate catalog")
        st.write(f"{manifest['camel_rows']:,} precomputed CAMEL/eigengene test rows; all modules are queryable, not only saved top-ten rankings.")
        st.caption("Only the selected statistical context is loaded. Downloads in the analysis sections contain all selected-context modules before top-K filtering.")
        _table(pd.DataFrame(manifest["files"])[["path","kind","rows","bytes","sha256"]],"files","Result file manifest")
        st.json({k:v for k,v in manifest.items() if k!="files"})
    st.info("This release contains module-level results only. Donor scatter plots and distributions for this new partition are not published here; the existing donor-level views continue to use their previous datasets.")
