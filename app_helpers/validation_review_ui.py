"""Read-only CAMEL validation page; no donor-level reads or computations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_helpers import camel_validation as V
from app_helpers import validation_review as R
from app_helpers.camel import COMPONENT_LABELS
from app_helpers.data import ASSOCIATION_OUTCOME_LABELS, FEATURE_LABELS, MODULE_SET_LABELS, METHOD_LABELS
from app_helpers.validation_datasets import DATASETS, MOTOR_LABELS
from app_helpers.table_controls import filterable_dataframe

COLORS = {
    "No original FDR signal": "#A6A6A6", "Original signal; opposite/zero direction": "#CC6677",
    "Same direction only": "#DDCC77", "Same direction + validation nominal p": "#4477AA",
    "Same direction + validation FDR": "#228833", "Unavailable comparison": "#BBBBBB",
}


def rule_label(rule):
    if rule == "all":
        return "All edges"
    return rule.removeprefix("ad_control_discovery_effect__").replace("__", " · ").replace("_", " ")


def download_table(frame, key, title):
    filterable_dataframe(frame, table_key=key, table_name=title, hide_index=True)
    st.download_button(f"Download {title.lower()}", frame.to_csv(sep="\t", index=False).encode(),
                       file_name=key + ".tsv", mime="text/tab-separated-values", key=key + "_download")


def scatter_figure(frame, x, y, x_label, y_label, *, bounded=False):
    f = frame.dropna(subset=[x, y]).copy()
    if f.empty:
        return None
    f["module_label"] = "M" + f.module.astype(str)
    f["region"] = f.component.map({**COMPONENT_LABELS,**MOTOR_LABELS})
    color = "replication_status" if "replication_status" in f else "region"
    hover = [c for c in ["module_label", "region", "original_n", "validation_n", "original_p",
                         "validation_p", "original_fdr", "validation_fdr", "original_family_n",
                         "validation_family_n", "n_ad", "n_control", "inferential"] if c in f]
    fig = px.scatter(f, x=x, y=y, color=color, color_discrete_map=COLORS,
                     hover_data=hover, labels={x: x_label, y: y_label, color: "Evidence" if color == "replication_status" else "Component"})
    low, high = (-1., 1.) if bounded else (float(np.min(f[[x, y]].to_numpy())), float(np.max(f[[x, y]].to_numpy())))
    if low == high:
        low, high = low - .5, high + .5
    fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines",
                            line=dict(color="black", dash="dash", width=1), name="Equal effect",
                            hoverinfo="skip"))
    if bounded:
        fig.update_xaxes(range=[-1.02, 1.02], zeroline=True, zerolinecolor="#CCCCCC")
        fig.update_yaxes(range=[-1.02, 1.02], zeroline=True, zerolinecolor="#CCCCCC")
    fig.update_layout(template="plotly_white", height=610,
                      legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0, xanchor="left"),
                      margin=dict(t=125, b=65, l=75, r=30))
    return fig


def _technical(m, reader=R):
    st.subheader("Technical reproduction and safeguards")
    st.caption("Passing these checks means the calculation reproduced the intended algorithm. It does not mean an association replicated in different donors.")
    checks = reader.table("technical_checks")
    download_table(checks, "validation_technical_checks", "Technical checks")
    with st.expander("Original-score reproduction: sampled modules"):
        r = reader.table("score_reproduction")
        st.caption("Recomputed original-donor features versus the unchanged saved R catalog. Error is relative; the test covers sampled donors/modules, not every original value.")
        download_table(r, "validation_original_score_reproduction", "Original score reproduction")
    with st.expander("Direct formula checks by module and component"):
        f = reader.table("formula_checks")
        download_table(f, "validation_formula_checks", "Direct formula checks")
    with st.expander("Frozen inputs and interpretation limits"):
        for note in m.get("limitations", []):
            st.write(note)
        sources = pd.DataFrame([dict(source=k, sha256=v) for k, v in m.get("source_hashes", {}).items()])
        download_table(sources, "validation_frozen_sources", "Frozen source hashes")


def _associations(m, render, reader=R, labels=COMPONENT_LABELS):
    st.subheader("Did the original associations reproduce?")
    entries = m.get("association_shards", [])
    if not entries:
        st.info("Original-versus-validation association tables are not yet ready. No original-cohort result is substituted.")
        return
    a, b, c = st.columns(3)
    rule = a.selectbox("Validation edge set", list(dict.fromkeys(e["rule"] for e in entries)),
                       format_func=rule_label, key="validation_review_rule")
    norms = list(dict.fromkeys(e["normalization"] for e in entries if e["rule"] == rule))
    norm = b.selectbox("Score handling", norms, format_func=lambda x: {"standard_pruned": "Standard pruned", "retained_edge": "Retained-edge normalized"}[x], key="validation_review_norm")
    features = list(dict.fromkeys(e["feature"] for e in entries if e["rule"] == rule and e["normalization"] == norm))
    feature = c.selectbox("Validation feature", features, index=features.index("connectivity") if "connectivity" in features else 0,
                          format_func=lambda x: FEATURE_LABELS.get(x, x), key="validation_review_feature")
    choices = list(dict.fromkeys(e["outcome"] for e in entries if (e["rule"], e["normalization"], e["feature"]) == (rule, norm, feature)))
    a, b, c = st.columns(3)
    outcome = a.selectbox("Validation outcome", choices, index=choices.index("cogn_global") if "cogn_global" in choices else 0,
                          format_func=lambda x: ASSOCIATION_OUTCOME_LABELS.get(x, x), key="validation_review_outcome")
    method = b.selectbox("Validation correlation", ["Spearman", "Pearson"], key="validation_review_correlation")
    scale = c.selectbox("Validation score scale", ["raw", "asinh", "rint"], format_func=lambda x: x.upper() if x == "rint" else x.capitalize(), key="validation_review_scale")
    f = reader.association_table(rule, norm, feature, outcome)
    if f.empty:
        st.info("This exact comparison is unavailable.")
        return
    a, b, c = st.columns(3)
    group = a.selectbox("Validation donor group", ["All donors", "AD", "Control", "MCI", "Unclassified"],
                        index=0 if m.get("original_pooled_shards") else 1, key="validation_review_group")
    components = b.multiselect("Validation components", list(labels), default=[next(c for c in labels if c.startswith("CT_"))],
                               format_func=labels.get, key="validation_review_components")
    alpha = c.selectbox("Evidence cutoff", [.05, .10], format_func=lambda x: f"{x:.2f}" + (" (exploratory)" if x == .10 else ""), key="validation_review_alpha")
    f = R.replication_labels(R.association_effects(f, method, scale), alpha)
    f = f.loc[f.diagnosis_group.eq(group) & f.component.isin(components)].copy()
    st.caption("FDRs were calculated across valid modules in each fixed component/outcome/group/feature/scale/edge set—not across groups × modules. No correction is recalculated after these display filters. Original and validation FDRs have their own tested-module counts.")
    if group == "All donors":
        st.caption("Pooled correlations use all original-cohort donors with observed scores and outcomes, and all eligible validation donors for each component. Original pooled FDR uses the same module-only BH calculation as Associations; Pearson and Spearman remain separate. No diagnosis-specific correlations are averaged.")
    st.caption("Same direction alone is not statistical replication. Nominal replication requires original FDR below the cutoff, the same nonzero effect direction, and validation p below the cutoff; FDR replication replaces validation p with validation FDR. These are descriptive replication criteria, not a test that the two correlations are equal.")
    if rule != "all":
        st.warning("These are frozen discovery-selected edges. Original discovery associations can be selection-biased; no validation outcome was used to select edges.")
    if scale == "rint":
        st.caption("Validation RINT is frozen from original donors; boundary clipping can introduce ties, so its Spearman correlation need not equal Raw.")
    counts = R.replication_counts(f)
    columns = st.columns(4)
    columns[0].metric("Comparable module–component tests", counts["comparable_modules"])
    columns[1].metric("Original FDR signals", counts["original_signals"])
    columns[2].metric("Nominal replications", counts["nominal_replication"])
    columns[3].metric("FDR replications", counts["fdr_replication"])
    st.caption(f'Direction agrees for {counts["direction_agreement_all"]}/{counts["comparable_modules"]} comparable estimates (descriptive). Among original FDR signals, {counts["original_signals_evaluable"]} are evaluable and {counts["same_direction"]} retain direction. Counts are module–component tests when multiple components are selected.')
    display = st.selectbox("Show association evidence", ["All tests", "Original FDR signals", "Same direction + nominal replication", "Same direction + FDR replication", "Unavailable comparisons"], key="validation_review_evidence")
    shown = f
    if display == "Original FDR signals":
        shown = f.loc[f.original_signal]
    elif display == "Same direction + nominal replication":
        shown = f.loc[f.original_signal & f.same_direction.fillna(False) & f.validation_nominal_signal]
    elif display == "Same direction + FDR replication":
        shown = f.loc[f.original_signal & f.same_direction.fillna(False) & f.validation_fdr_signal]
    elif display == "Unavailable comparisons":
        shown = f.loc[~f.comparison_available]
    focus = st.multiselect("Focus modules (optional)", sorted(f.module.dropna().astype(int).unique()),
                           format_func=lambda module: f"M{module}", key="validation_review_module_focus",
                           help="Filter the plot and displayed table only. FDRs retain their complete module families; the complete-context download includes all modules.")
    if focus:
        shown = shown.loc[shown.module.isin(focus)]
    fig = scatter_figure(shown, "original_effect", "validation_effect", f"Original {method} effect", f"Validation {method} effect", bounded=True)
    if fig is None:
        st.info("No comparable associations in this selection. Unclassified original rows, uninstalled pooled baselines, small cohorts, constant scores, or empty masks may be unavailable; missing evidence is not counted as failed replication.")
    else:
        render(fig, key="validation_association_replication_scatter")
    columns = [*R.MATCH_KEYS, "original_n", "validation_n", "original_effect", "validation_effect", "effect_difference",
               "original_p", "validation_p", "original_fdr", "validation_fdr", "original_family_n", "validation_family_n",
               "same_direction", "replication_status", "unavailable_reason"]
    download_table(shown[columns], "validation_association_comparison", "Displayed association comparison")
    st.download_button("Download complete selected-context comparison", f.to_csv(sep="\t", index=False).encode(),
                       "validation_association_full_context.tsv", key="validation_review_full_associations")


def _mdc(render, reader=R, labels=COMPONENT_LABELS):
    st.subheader("MDC in original versus validation donors")
    f = reader.table("mdc_comparison")
    if f.empty:
        st.info("MDC comparison is not yet available.")
        return
    a, b, c = st.columns(3)
    rule = a.selectbox("MDC validation edge set", list(dict.fromkeys(f.differential_edge_rule)), format_func=rule_label, key="validation_review_mdc_rule")
    components = b.multiselect("MDC validation components", list(labels), default=[next(c for c in labels if c.startswith("CT_"))], format_func=labels.get, key="validation_review_mdc_components")
    scale = c.radio("MDC comparison scale", ["Log2 MDC", "Raw MDC"], key="validation_review_mdc_scale")
    f = f.loc[f.differential_edge_rule.eq(rule) & f.component.isin(components)].copy()
    baseline = 0. if scale == "Log2 MDC" else 1.
    suffix = "log2_mdc" if scale == "Log2 MDC" else "mdc"
    valid = np.isfinite(f.original_mdc) & np.isfinite(f.validation_mdc)
    f["same_mdc_direction"] = pd.Series(pd.NA, index=f.index, dtype="boolean")
    f.loc[valid, "same_mdc_direction"] = np.sign(f.loc[valid, "original_mdc"] - 1).eq(np.sign(f.loc[valid, "validation_mdc"] - 1))
    st.caption("MDC is the validation AD/Control ratio of mean absolute group-network weight, not an average donor CAMEL score. Ratios above 1 indicate higher AD connectivity. Original and validation values and FDRs are shown separately; effect-filtered original MDC comparisons are unavailable in this catalog.")
    if "CT_AC__PCGBA23" in components:
        st.warning("AC–PCG: 3 AD and 6 Controls. MDC is descriptive only; validation p-values/FDR are unavailable.")
    if not valid.any():
        st.info("No exact original MDC comparison for this selection. Validation-only values remain in the table.")
    else:
        fig = scatter_figure(f, "original_" + suffix, "validation_" + suffix, "Original " + scale, "Validation " + scale)
        if fig is not None:
            fig.add_hline(y=baseline, line_color="#CCCCCC")
            fig.add_vline(x=baseline, line_color="#CCCCCC")
            render(fig, key="validation_mdc_replication_scatter")
    download_table(f, "validation_mdc_comparison", "MDC comparison")


def render_view(render):
    st.header("Validation & replication")
    identities=[(key,method) for key,config in DATASETS.items() for method in config.methods
                if R.manifest(key,method)]
    if not identities:
        st.info("Validation-review evidence has not been installed yet. Existing analyses remain unchanged; no result is substituted.")
        return
    if len(identities)>1:
        chosen=st.selectbox("Validation dataset",identities,
            format_func=lambda item:MODULE_SET_LABELS[item[0]]+" · "+METHOD_LABELS[item[1]],key="validation_review_dataset")
    else:
        chosen=identities[0]
    source=V.for_dataset(*chosen); reader=R.for_dataset(*chosen)
    labels=source.config.labels
    st.caption(f"{MODULE_SET_LABELS[chosen[0]]}: {source.config.framework(chosen[1])} in donors excluded from the complete-expression cohort.")
    m = reader.manifest()
    if not m:
        st.info("Validation-review evidence has not been installed yet. Existing analyses remain unchanged; no result is substituted.")
        return
    c = st.columns(3)
    c[0].metric("Validation donors", m["validation_donors"])
    c[1].metric("Modules scored", f'{m["complete_modules"]}/{m["modules"]}')
    c[2].metric("Frozen reference donors" if chosen[1]=="standard" else "Frozen reference Controls",
                m.get("reference_donors",m["reference_controls"]))
    st.caption(f'Evidence snapshot: {m["generated_utc"]} · {m["run_id"]}. This is non-discovery-donor validation within ROSMAP, not external validation.')
    if not source.available():
        st.info("Aggregate validation evidence is available here. Donor-level plots become selectable after the Drive bundle passes readback verification; this page does not load those files.")
    section = st.radio("Validation section", ["Technical checks", "Association replication", "MDC replication", "Coverage & completion"], horizontal=True, key="validation_review_section")
    if section == "Technical checks":
        _technical(m,reader)
    elif section == "Association replication":
        _associations(m, render,reader,labels)
    elif section == "MDC replication":
        _mdc(render,reader,labels)
    else:
        st.subheader("Component-specific donor coverage")
        st.caption(f'All {m["excluded_complete_donors"]} complete-expression donors were excluded. Component cohorts overlap; their counts must not be summed. No absent region is imputed and no pooled three-region CT/TS score is constructed.')
        f = reader.table("cohorts")
        long = f.melt(id_vars=["component", "component_label", "n"], value_vars=["Control", "MCI", "AD", "Unclassified"], var_name="Diagnosis", value_name="Donors")
        fig = px.bar(long, x="component_label", y="Donors", color="Diagnosis", barmode="stack",
                     color_discrete_map={"Control": "#377EB8", "MCI": "#DDAA33", "AD": "#EE7733", "Unclassified": "#999999"},
                     labels={"component_label": "Component"})
        fig.update_layout(template="plotly_white", legend=dict(orientation="h", y=1.1), margin=dict(t=70))
        render(fig, key="validation_cohort_coverage")
        download_table(f, "validation_component_coverage", "Component coverage")
        st.subheader("Completion by frozen edge mask")
        download_table(reader.table("completion"), "validation_completion", "Validation completion")
        st.caption("Score completion does not imply association or publication completion. Only reconciled combinations are enabled in donor-level analysis views.")
