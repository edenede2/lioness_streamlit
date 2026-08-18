"""Public explorer for standard and control-referenced ROSMAP LIONESS results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app_helpers.charts import (
    aggregate_to_long,
    association_figure,
    distribution_figure,
    distribution_summary,
    resolved_to_long,
)
from app_helpers.data import (
    COMPONENT_ORDER,
    DATA_DIR,
    DIAGNOSIS_ORDER,
    FEATURE_LABELS,
    KEGG_TSV,
    METHOD_LABELS,
    PHENOTYPE_LABELS,
    SCALE_LABELS,
    dataframe_to_tsv_bytes,
    load_aggregate,
    load_aggregate_statistics,
    load_feature_definitions,
    load_kegg,
    load_module_annotations,
    load_resolved,
    load_resolved_statistics,
    load_tissue_mapping,
    module_label,
    require_data_files,
    selected_annotation,
)


st.set_page_config(
    page_title="ROSMAP LIONESS Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px;}
      [data-testid="stMetricValue"] {font-size: 1.35rem;}
      .app-note {color: #526273; font-size: 0.93rem;}
      .kegg-note {padding: .6rem .85rem; background: #f3f7fb; border-left: 4px solid #2c7fb8;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_annotations() -> pd.DataFrame:
    return load_module_annotations()


@st.cache_data(show_spinner=False, max_entries=48)
def cached_aggregate(method: str, module: int, feature: str) -> pd.DataFrame:
    return load_aggregate(method, module, feature)


@st.cache_data(show_spinner=False, max_entries=48)
def cached_resolved(method: str, module: int, feature: str) -> pd.DataFrame:
    return load_resolved(method, module, feature)


@st.cache_data(show_spinner=False, max_entries=96)
def cached_aggregate_stats(
    method: str,
    module: int | None,
    phenotype: str | None,
    feature: str | None,
) -> pd.DataFrame:
    return load_aggregate_statistics(method, module, phenotype, feature)


@st.cache_data(show_spinner=False, max_entries=48)
def cached_resolved_stats(
    method: str, module: int, phenotype: str, feature: str
) -> pd.DataFrame:
    return load_resolved_statistics(method, module, phenotype, feature)


@st.cache_data(show_spinner=False, max_entries=48)
def cached_kegg(module: int | None) -> pd.DataFrame:
    return load_kegg(module)


@st.cache_data(show_spinner=False)
def cached_kegg_tsv() -> bytes:
    return KEGG_TSV.read_bytes()


@st.cache_data(show_spinner=False)
def cached_feature_definitions() -> pd.DataFrame:
    return load_feature_definitions()


@st.cache_data(show_spinner=False)
def cached_tissue_mapping() -> pd.DataFrame:
    return load_tissue_mapping()


def readable_method(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def fdr_text() -> str:
    return (
        "LIONESS FDR values use Benjamini–Hochberg correction separately within each "
        "network method. For aggregate results, CT and TS correlation FDRs each cover "
        "13,860 tests; the global CT-vs-TS FDR covers 13,860 dependent-correlation "
        "tests, and the within-phenotype CT-vs-TS FDR covers 2,772 tests. Tissue-resolved "
        "global FDRs cover 83,160 component tests and within-phenotype FDRs cover 16,632. "
        "KEGG FDR is independent and comes directly from the supplied tissue-expanded "
        "enrichment analysis."
    )


try:
    require_data_files()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

annotations = cached_annotations()
modules = sorted(annotations["module"].astype(int).unique().tolist())

st.title("ROSMAP LIONESS Network Explorer")
st.markdown(
    "Explore donor-level module features from the standard and control-referenced "
    "LIONESS networks, including aggregate CT/TS and tissue-resolved components."
)

with st.sidebar:
    st.header("Plot controls")
    method = st.radio(
        "Network method",
        options=list(METHOD_LABELS),
        format_func=readable_method,
        index=1,
    )
    default_module = modules.index(935) if 935 in modules else 0
    module = st.selectbox(
        "Module",
        options=modules,
        index=default_module,
        format_func=module_label,
    )
    phenotype = st.selectbox(
        "Phenotype",
        options=list(PHENOTYPE_LABELS),
        format_func=lambda value: PHENOTYPE_LABELS[value],
    )
    feature = st.selectbox(
        "Module feature",
        options=list(FEATURE_LABELS),
        format_func=lambda value: FEATURE_LABELS[value],
    )
    resolution = st.radio("Resolution", ["Aggregate CT / TS", "Tissue resolved"])
    scale = st.selectbox(
        "Feature scale",
        options=list(SCALE_LABELS),
        format_func=lambda value: SCALE_LABELS[value],
        index=0,
    )
    diagnoses = st.multiselect(
        "Diagnosis groups",
        options=DIAGNOSIS_ORDER,
        default=DIAGNOSIS_ORDER,
    )

if not diagnoses:
    st.warning("Select at least one diagnosis group in the sidebar.")
    st.stop()

with st.spinner("Loading the selected module…"):
    if resolution == "Aggregate CT / TS":
        plot_data = cached_aggregate(method, module, feature)
        plot_data = aggregate_to_long(plot_data, scale)
        statistics = cached_aggregate_stats(method, module, phenotype, feature)
        resolved = False
    else:
        plot_data = cached_resolved(method, module, feature)
        plot_data = resolved_to_long(plot_data, scale)
        statistics = cached_resolved_stats(method, module, phenotype, feature)
        resolved = True

available_components = (
    plot_data[["component", "component_label"]]
    .drop_duplicates()
    .assign(
        component_order=lambda frame: frame["component"].map(
            {component: index for index, component in enumerate(COMPONENT_ORDER)}
        )
    )
    .sort_values(["component_order", "component"])
)
component_labels = dict(
    available_components[["component", "component_label"]].itertuples(index=False, name=None)
)
if resolved:
    selected_components = st.sidebar.multiselect(
        "Tissue components",
        options=available_components["component"].tolist(),
        default=available_components["component"].tolist(),
        format_func=lambda value: component_labels[value],
    )
else:
    selected_components = available_components["component"].tolist()

plot_data = plot_data.loc[
    plot_data["diagnosis_group"].isin(diagnoses)
    & plot_data["component"].isin(selected_components)
].copy()
statistics = statistics.loc[
    statistics["diagnosis_group"].isin(diagnoses)
].copy()
if resolved:
    statistics = statistics.loc[statistics["component"].isin(selected_components)]

if plot_data.empty or not selected_components:
    st.warning("No data remain for the current filters.")
    st.stop()

annotation = selected_annotation(annotations, module)
if annotation:
    st.markdown(f'<div class="kegg-note">{annotation}</div>', unsafe_allow_html=True)
else:
    st.caption(f"No KEGG enrichment row is available for {module_label(module)}.")

summary_cols = st.columns(4)
summary_cols[0].metric("Method", "Control-referenced" if method == "control_anchored" else "Standard")
summary_cols[1].metric("Module", module_label(module))
summary_cols[2].metric("Samples shown", plot_data["sample_id"].nunique())
summary_cols[3].metric("Components", plot_data["component"].nunique())

association_tab, distribution_tab, screen_tab, statistics_tab, kegg_tab, about_tab = st.tabs(
    [
        "Associations",
        "Feature distributions",
        "CT–TS screen",
        "Statistics",
        "KEGG enrichment",
        "Methods & data",
    ]
)

with association_tab:
    st.subheader("Phenotype association")
    st.caption(
        "Diagnosis-specific points and ordinary least-squares trends. Click a diagnosis "
        "in the legend to hide or show its points and trend together."
    )
    figure = association_figure(
        plot_data,
        statistics,
        phenotype=phenotype,
        phenotype_label=PHENOTYPE_LABELS[phenotype],
        feature_label=FEATURE_LABELS[feature],
        scale=scale,
        scale_label=SCALE_LABELS[scale],
        diagnoses=diagnoses,
        module=module,
        resolved=resolved,
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"M{module}_{phenotype}_{feature}_{method}",
                "scale": 3,
            },
        },
    )
    public_plot_data = plot_data[
        [
            "sample_id",
            "diagnosis_group",
            "module",
            "metric_family",
            "component",
            "component_label",
            "metric_value",
            phenotype,
            "lioness_method",
        ]
    ].rename(columns={"metric_value": f"feature_{scale}"})
    st.download_button(
        "Download displayed plot data (TSV)",
        data=dataframe_to_tsv_bytes(public_plot_data),
        file_name=f"M{module}_{phenotype}_{feature}_{method}_{resolution.replace(' ', '_')}.tsv",
        mime="text/tab-separated-values",
    )

with distribution_tab:
    st.subheader("Donor-level feature distributions")
    st.caption(
        "These views use only the selected LIONESS module feature; no phenotype is on an axis. "
        "Histogram heights are probability densities so diagnosis groups with different sample "
        "sizes remain comparable."
    )
    chart_col, bin_col = st.columns([1, 2])
    chart_type = chart_col.radio("Distribution view", ["Histogram", "Violin"], horizontal=True)
    bins = bin_col.slider("Histogram bins", 10, 80, 30, disabled=chart_type != "Histogram")
    distribution = distribution_figure(
        plot_data,
        feature_label=FEATURE_LABELS[feature],
        scale_label=SCALE_LABELS[scale],
        diagnoses=diagnoses,
        module=module,
        chart_type=chart_type,
        bins=bins,
    )
    st.plotly_chart(
        distribution,
        use_container_width=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"M{module}_{feature}_{method}_distribution",
                "scale": 3,
            },
        },
    )
    summary = distribution_summary(plot_data)
    with st.expander("Distribution summary table", expanded=False):
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.download_button(
            "Download distribution summary (TSV)",
            data=dataframe_to_tsv_bytes(summary),
            file_name=f"M{module}_{feature}_{method}_distribution_summary.tsv",
            mime="text/tab-separated-values",
        )

with screen_tab:
    st.subheader("Descriptive CT–TS pattern screen")
    st.caption(
        "Ranks all 154 modules for the selected phenotype, feature, diagnosis, and method by "
        "the absolute difference between CT and TS RINT correlations. This is an exploratory "
        "screen, not an independent validation ranking."
    )
    screen_diagnosis = st.selectbox(
        "Diagnosis for screen",
        options=diagnoses,
        index=diagnoses.index("AD") if "AD" in diagnoses else 0,
        key="screen_diagnosis",
    )
    screen = cached_aggregate_stats(method, None, phenotype, feature)
    screen = screen.loc[screen["diagnosis_group"].eq(screen_diagnosis)].copy()
    screen["delta_r_CT_minus_TS"] = screen["r_rint_CT"] - screen["r_rint_TS"]
    screen["abs_delta_r"] = screen["delta_r_CT_minus_TS"].abs()
    screen["opposite_CT_TS_sign"] = np.sign(screen["r_rint_CT"]) != np.sign(screen["r_rint_TS"])
    screen["max_abs_r"] = screen[["r_rint_CT", "r_rint_TS"]].abs().max(axis=1)
    annotation_columns = annotations[
        [
            "module",
            "displayed_category",
            "displayed_subcategory",
            "displayed_pathway",
            "displayed_fdr",
        ]
    ].copy()
    screen = screen.merge(annotation_columns, on="module", how="left")
    screen = screen.sort_values(
        ["abs_delta_r", "max_abs_r"], ascending=[False, False], na_position="last"
    )
    screen.insert(0, "screen_rank", np.arange(1, len(screen) + 1))
    n_screen = st.slider("Rows to show", 10, 154, 30, 10)
    screen_columns = [
        "screen_rank",
        "module",
        "r_rint_CT",
        "p_rint_CT",
        "q_rint_CT_global",
        "r_rint_TS",
        "p_rint_TS",
        "q_rint_TS_global",
        "delta_r_CT_minus_TS",
        "abs_delta_r",
        "opposite_CT_TS_sign",
        "p_component_rint",
        "q_component_rint_within_phenotype",
        "q_component_rint_global",
        "displayed_category",
        "displayed_subcategory",
        "displayed_pathway",
        "displayed_fdr",
    ]
    st.dataframe(
        screen[screen_columns].head(n_screen),
        use_container_width=True,
        hide_index=True,
        column_config={"module": st.column_config.NumberColumn(format="M%d")},
    )
    st.download_button(
        "Download complete 154-module screen (TSV)",
        data=dataframe_to_tsv_bytes(screen[screen_columns]),
        file_name=f"{method}_{screen_diagnosis}_{phenotype}_{feature}_CT_TS_screen.tsv",
        mime="text/tab-separated-values",
    )
    st.info(fdr_text())

with statistics_tab:
    st.subheader("Robust association statistics")
    st.caption(
        "The downloadable table retains raw, asinh, Winsorized, Spearman, RINT, "
        "leave-one-out sensitivity, dependent CT-vs-TS tests, and FDR fields."
    )
    if resolved:
        display_columns = [
            "component_label",
            "diagnosis_group",
            "n",
            "r_rint",
            "p_rint",
            "q_rint_global",
            "q_rint_within_phenotype",
            "rho",
            "p_spearman",
            "loo_rint_max_delta",
            "loo_rint_sign_flip",
        ]
    else:
        display_columns = [
            "diagnosis_group",
            "n",
            "r_rint_CT",
            "p_rint_CT",
            "q_rint_CT_global",
            "r_rint_TS",
            "p_rint_TS",
            "q_rint_TS_global",
            "rho_CT",
            "rho_TS",
            "p_component_rint",
            "q_component_rint_within_phenotype",
            "q_component_rint_global",
        ]
    st.dataframe(
        statistics[display_columns], use_container_width=True, hide_index=True
    )
    st.download_button(
        "Download all selected robust-statistics columns (TSV)",
        data=dataframe_to_tsv_bytes(statistics),
        file_name=f"M{module}_{phenotype}_{feature}_{method}_robust_statistics.tsv",
        mime="text/tab-separated-values",
    )
    st.info(fdr_text())

with kegg_tab:
    st.subheader(f"Full tissue-expanded KEGG table for {module_label(module)}")
    kegg = cached_kegg(module)
    if kegg.empty:
        st.info(
            "This module has no row in the supplied tissue-expanded KEGG table. "
            "The LIONESS module itself is still available in every plot and statistics view."
        )
    else:
        significant_count = int(kegg["significant"].fillna(False).astype(bool).sum())
        kegg_metrics = st.columns(4)
        kegg_metrics[0].metric("KEGG pathways", len(kegg))
        kegg_metrics[1].metric("FDR-significant", significant_count)
        kegg_metrics[2].metric("Module size", int(kegg["cluster_size_expanded"].iloc[0]))
        kegg_metrics[3].metric("Tissues represented", int(kegg["n_tissues"].max()))
        search = st.text_input(
            "Search this module's KEGG rows",
            placeholder="pathway, category, gene, tissue…",
        ).strip()
        shown_kegg = kegg
        if search:
            searchable = kegg.select_dtypes(include=["object", "string"]).fillna("")
            mask = searchable.astype(str).agg(" ".join, axis=1).str.contains(
                search, case=False, regex=False
            )
            shown_kegg = kegg.loc[mask]
        st.dataframe(
            shown_kegg.sort_values(["fdr", "p"], na_position="last"),
            use_container_width=True,
            hide_index=True,
            column_config={
                "p": st.column_config.NumberColumn(format="%.3e"),
                "fdr": st.column_config.NumberColumn(format="%.3e"),
            },
        )
        st.download_button(
            f"Download all KEGG rows for {module_label(module)} (TSV)",
            data=dataframe_to_tsv_bytes(kegg.sort_values(["fdr", "p"])),
            file_name=f"M{module}_tissue_expanded_KEGG.tsv",
            mime="text/tab-separated-values",
        )
    st.download_button(
        "Download the complete tissue-expanded KEGG table (all modules)",
        data=cached_kegg_tsv(),
        file_name="method4_tissue_expanded_kegg_annotated.tsv",
        mime="text/tab-separated-values",
    )
    st.caption(
        "KEGG tissue labels: AC, DLPFC, and PCG/BA23. The KEGG FDR column "
        "was supplied by the enrichment analysis and is not recalculated by this app."
    )

with about_tab:
    st.subheader("Analysis methods and deploy data")
    st.markdown(
        "**Standard LIONESS** scores each donor relative to the complete 450-donor "
        "reference network. **Control-referenced LIONESS** uses the 164 Controls as the "
        "reference and scores Controls by leave-one-out and MCI/AD by donor addition."
    )
    formula_col1, formula_col2 = st.columns(2)
    with formula_col1:
        st.markdown("Standard donor network")
        st.latex(r"A_i = N A_{all} - (N-1)A_{all\\setminus i},\\quad N=450")
        st.markdown("Control donor")
        st.latex(r"A_i = n_C A_C - (n_C-1)A_{C\\setminus i},\\quad n_C=164")
    with formula_col2:
        st.markdown("MCI or AD donor added to the Control reference")
        st.latex(r"A_i = (n_C+1)A_{C+i} - n_C A_C")
        st.markdown(
            "Networks use signedAlt adjacency with β=3 for tissue-specific edges and "
            "β=2 for cross-tissue edges. Each undirected edge is counted once and the "
            "diagonal is excluded."
        )

    st.markdown("#### Module feature definitions")
    st.dataframe(cached_feature_definitions(), use_container_width=True, hide_index=True)
    st.markdown("#### Tissue labels")
    st.dataframe(cached_tissue_mapping(), use_container_width=True, hide_index=True)
    st.markdown("#### Public deploy safeguards")
    st.markdown(
        "The deploy Parquet files do not contain `projid`, the source donor identifier, "
        "or detailed clinical metadata. Each point has a random-salted pseudonymous sample "
        "label that is stable within this bundle only. Diagnosis and the five plotted "
        "phenotypes remain because they are required for the figures. Public release still "
        "requires confirmation that the governing ROSMAP data-use agreement permits sharing "
        "these donor-level derived values."
    )
    manifest_path = DATA_DIR / "data_manifest.json"
    if manifest_path.exists():
        with st.expander("Deploy data manifest"):
            st.json(manifest_path.read_text(encoding="utf-8"))

st.markdown(
    '<p class="app-note">Exploratory research interface. Associations are not causal, and '
    "nominal p-values should be interpreted alongside FDR and leave-one-out sensitivity.</p>",
    unsafe_allow_html=True,
)
