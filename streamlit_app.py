"""Public explorer for standard and control-referenced ROSMAP LIONESS results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app_helpers.charts import (
    aggregate_to_long,
    association_figure,
    correlation_heatmap_figure,
    distribution_figure,
    distribution_summary,
    mdc_module_figure,
    mdc_overview_figure,
    resolved_to_long,
)
from app_helpers.correlations import calculate_correlations
from app_helpers.data import (
    COLOR_LABELS,
    COMPONENT_ORDER,
    DATA_DIR,
    DIAGNOSIS_ORDER,
    FEATURE_LABELS,
    HOVER_LABELS,
    KEGG_TSV,
    METHOD_LABELS,
    NUMERIC_OUTCOMES,
    OUTCOME_LABELS,
    PHENOTYPE_LABELS,
    SCALE_LABELS,
    dataframe_to_tsv_bytes,
    load_aggregate,
    load_aggregate_scope,
    load_aggregate_statistics,
    load_data_manifest,
    load_feature_definitions,
    load_kegg,
    load_mdc_summary,
    load_module_annotations,
    load_module_details,
    load_resolved,
    load_resolved_scope,
    load_resolved_statistics,
    load_sample_metadata,
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


@st.cache_data(show_spinner=False)
def cached_module_details() -> pd.DataFrame:
    return load_module_details()


@st.cache_data(show_spinner=False, max_entries=48)
def cached_aggregate(method: str, module: int, feature: str) -> pd.DataFrame:
    return load_aggregate(method, module, feature)


@st.cache_data(show_spinner=False, max_entries=48)
def cached_resolved(method: str, module: int, feature: str) -> pd.DataFrame:
    return load_resolved(method, module, feature)


@st.cache_data(show_spinner=False)
def cached_sample_metadata() -> pd.DataFrame:
    return load_sample_metadata()


@st.cache_data(show_spinner=False)
def cached_mdc_summary() -> pd.DataFrame:
    return load_mdc_summary()


@st.cache_data(show_spinner=False)
def cached_data_manifest() -> dict[str, object]:
    return load_data_manifest()


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


def attach_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    metadata = cached_sample_metadata()
    additional = [
        column for column in metadata.columns if column == "sample_id" or column not in frame.columns
    ]
    return frame.merge(metadata[additional], on="sample_id", how="left", validate="many_to_one")


def add_correlation_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["outcome_label"] = result["outcome"].map(OUTCOME_LABELS)
    result["feature_label"] = result["metric_family"].map(FEATURE_LABELS)
    result["heatmap_row"] = result["feature_label"] + " · " + result["component_label"]
    return result


@st.cache_data(show_spinner="Calculating the selected module correlation matrix…", max_entries=24)
def cached_module_correlations(method: str, module: int, resolved: bool) -> pd.DataFrame:
    if resolved:
        source = load_resolved_scope(method, module=module)
        long = resolved_to_long(source, "rint")
    else:
        source = load_aggregate_scope(method, module=module)
        long = aggregate_to_long(source, "rint")
    long = attach_metadata(long)
    all_donors = long.copy()
    all_donors["diagnosis_group"] = "All donors"
    long = pd.concat([long, all_donors], ignore_index=True)
    summary = calculate_correlations(
        long,
        group_columns=[
            "module",
            "metric_family",
            "component",
            "component_label",
            "diagnosis_group",
        ],
        outcomes=NUMERIC_OUTCOMES,
    )
    return add_correlation_labels(summary)


@st.cache_data(show_spinner="Calculating correlations across all 154 modules…", max_entries=24)
def cached_all_module_correlations(
    method: str,
    resolved: bool,
    feature: str,
    component: str,
    diagnosis: str,
) -> pd.DataFrame:
    if resolved:
        source = load_resolved_scope(
            method,
            metric_family=feature,
            component=component,
        )
        long = resolved_to_long(source, "rint")
    else:
        source = load_aggregate_scope(method, metric_family=feature)
        long = aggregate_to_long(source, "rint")
        long = long.loc[long["component"].eq(component)]
    long = attach_metadata(long)
    if diagnosis != "All donors":
        long = long.loc[long["diagnosis_group"].eq(diagnosis)]
    else:
        long = long.copy()
        long["diagnosis_group"] = "All donors"
    summary = calculate_correlations(
        long,
        group_columns=[
            "module",
            "metric_family",
            "component",
            "component_label",
            "diagnosis_group",
        ],
        outcomes=NUMERIC_OUTCOMES,
    )
    summary = add_correlation_labels(summary)
    summary["heatmap_row"] = summary["module"].map(lambda value: f"M{int(value)}")
    return summary


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
module_details = cached_module_details()
mdc_summary = cached_mdc_summary()
data_manifest = cached_data_manifest()

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
    color_by = st.selectbox(
        "Color points by",
        options=list(COLOR_LABELS),
        format_func=lambda value: COLOR_LABELS[value],
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

plot_data = attach_metadata(plot_data)

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

selected_details = module_details.loc[
    module_details["module"].astype(int).eq(int(module))
]
if len(selected_details) != 1:
    st.error(f"Expected exactly one module-details row for {module_label(module)}.")
    st.stop()
selected_details = selected_details.iloc[0]

annotation = selected_annotation(annotations, module)
if annotation:
    st.markdown(f'<div class="kegg-note">{annotation}</div>', unsafe_allow_html=True)
else:
    st.caption(f"No KEGG enrichment row is available for {module_label(module)}.")

summary_cols = st.columns(5)
summary_cols[0].metric("Method", "Control-referenced" if method == "control_anchored" else "Standard")
summary_cols[1].metric("Module", module_label(module))
summary_cols[2].metric("Module genes", int(selected_details["module_size"]))
summary_cols[3].metric("Samples shown", plot_data["sample_id"].nunique())
summary_cols[4].metric("Components", plot_data["component"].nunique())

tissue_composition = pd.DataFrame(
    [
        {
            "Tissue": tissue,
            "Genes": int(selected_details[count_column]),
            "Module proportion": float(selected_details[proportion_column]),
        }
        for tissue, count_column, proportion_column in [
            ("AC", "n_genes_ac", "proportion_ac"),
            ("DLPFC", "n_genes_dlpfc", "proportion_dlpfc"),
            ("PCG", "n_genes_pcg", "proportion_pcg"),
        ]
    ]
)
tissue_composition["Possible TS edges"] = tissue_composition["Genes"].map(
    lambda count: count * (count - 1) // 2
)
tissue_composition["TS feature"] = tissue_composition["Genes"].map(
    lambda count: "Available" if count >= 2 else "Unavailable (<2 genes)"
)

with st.expander(
    f"{module_label(module)} module composition · {int(selected_details['module_size'])} genes · "
    f"{selected_details['tissues']}",
    expanded=True,
):
    detail_cols = st.columns(4)
    detail_cols[0].metric("Module size", f"{int(selected_details['module_size']):,} genes")
    detail_cols[1].metric("Tissues represented", f"{int(selected_details['n_tissues'])} of 3")
    detail_cols[2].metric("Module type", str(selected_details["cluster_type"]))
    detail_cols[3].metric("Dominant tissue", str(selected_details["dominant_tissue"]))
    st.dataframe(
        tissue_composition,
        width="stretch",
        hide_index=True,
        column_config={
            "Genes": st.column_config.NumberColumn(format="%d"),
            "Module proportion": st.column_config.ProgressColumn(
                format="percent", min_value=0.0, max_value=1.0
            ),
            "Possible TS edges": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.caption(
        "Tissue proportions use total module genes as the denominator. A tissue needs at "
        "least two module genes for a within-tissue (TS) edge; one gene can still participate "
        "in cross-tissue (CT) edges."
    )

(
    association_tab,
    distribution_tab,
    correlation_tab,
    screen_tab,
    mdc_tab,
    statistics_tab,
    kegg_tab,
    module_details_tab,
    about_tab,
) = st.tabs(
    [
        "Associations",
        "Feature distributions",
        "Correlation heatmaps",
        "CT–TS screen",
        "MDC",
        "Statistics",
        "KEGG enrichment",
        "Module details",
        "Methods & data",
    ]
)

with association_tab:
    st.subheader("Phenotype association")
    st.caption(
        "Diagnosis-specific points and ordinary least-squares trends. Click a diagnosis "
        "in the legend to hide or show its points and trend together. Point shape identifies "
        "diagnosis; point color follows the selected color variable. Gray points have a missing "
        "value for a continuous color variable."
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
        color_by=color_by,
        color_label=COLOR_LABELS[color_by],
        hover_fields=HOVER_LABELS,
    )
    st.plotly_chart(
        figure,
        width="stretch",
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"M{module}_{phenotype}_{feature}_{method}",
                "scale": 3,
            },
        },
    )
    public_columns = [
        "sample_id",
        "diagnosis_group",
        "module",
        "metric_family",
        "component",
        "component_label",
        "metric_value",
        phenotype,
        "lioness_method",
        *[column for column in HOVER_LABELS if column != phenotype],
    ]
    public_plot_data = plot_data[public_columns].rename(
        columns={"metric_value": f"feature_{scale}"}
    )
    st.download_button(
        "Download displayed plot data (TSV)",
        data=dataframe_to_tsv_bytes(public_plot_data),
        file_name=f"M{module}_{phenotype}_{feature}_{method}_{resolution.replace(' ', '_')}.tsv",
        mime="text/tab-separated-values",
    )

with correlation_tab:
    st.subheader("LIONESS score correlations across phenotypes and outcomes")
    st.caption(
        "Heatmaps use donor-level Z-scored LIONESS features. Nominal fields such as sex code "
        "and APOE genotype remain available in hover but are excluded from Pearson/Spearman "
        "heatmaps because numeric correlation is not appropriate for unordered categories. "
        "CogDx, Braak, CERAD, ADNC, and Parkinsonism are source-coded ordinal/binary outcomes; "
        "Spearman is generally the more appropriate descriptive coefficient for the ordinal fields."
    )
    heatmap_mode = st.radio(
        "Heatmap scope",
        ["Selected module: all feature scores", "All 154 modules: selected score"],
        horizontal=True,
    )
    coefficient = st.radio(
        "Correlation",
        ["Pearson", "Spearman"],
        horizontal=True,
    )
    heatmap_diagnosis = st.selectbox(
        "Diagnosis in heatmap",
        options=["All donors", *DIAGNOSIS_ORDER],
        index=3,
        key="heatmap_diagnosis",
    )
    heatmap_clustering = st.selectbox(
        "Heatmap clustering",
        options=["None", "Rows", "Columns", "Rows and columns"],
        help=(
            "Reorders the selected axes using average-linkage hierarchical clustering "
            "of Euclidean distances between correlation profiles."
        ),
    )
    cluster_rows = heatmap_clustering in {"Rows", "Rows and columns"}
    cluster_columns = heatmap_clustering in {"Columns", "Rows and columns"}
    if coefficient == "Pearson":
        value_column = "pearson_r"
        p_column = "pearson_p"
        fdr_column = "pearson_fdr_displayed_family"
    else:
        value_column = "spearman_rho"
        p_column = "spearman_p"
        fdr_column = "spearman_fdr_displayed_family"

    if heatmap_mode.startswith("Selected module"):
        correlation_table = cached_module_correlations(method, module, resolved)
        heatmap_data = correlation_table.loc[
            correlation_table["diagnosis_group"].eq(heatmap_diagnosis)
        ].copy()
        component_rank = {value: index for index, value in enumerate(COMPONENT_ORDER)}
        feature_rank = {value: index for index, value in enumerate(FEATURE_LABELS)}
        row_order_frame = (
            heatmap_data[["metric_family", "component", "heatmap_row"]]
            .drop_duplicates()
            .assign(
                feature_rank=lambda frame: frame["metric_family"].map(feature_rank),
                component_rank=lambda frame: frame["component"].map(component_rank),
            )
            .sort_values(["feature_rank", "component_rank", "component"])
        )
        row_order = row_order_frame["heatmap_row"].tolist()
        heatmap_title = (
            f"Module M{module}: all LIONESS feature scores vs outcomes · {heatmap_diagnosis}"
        )
        fdr_scope = (
            "Displayed-family FDR is Benjamini–Hochberg correction across every feature, "
            "component, outcome, and diagnosis correlation calculated for this module."
        )
    else:
        all_feature = st.selectbox(
            "Feature for all-module heatmap",
            options=list(FEATURE_LABELS),
            format_func=lambda value: FEATURE_LABELS[value],
            index=list(FEATURE_LABELS).index(feature),
        )
        if resolved:
            component_options = available_components["component"].tolist()
            all_component = st.selectbox(
                "Component for all-module heatmap",
                options=component_options,
                format_func=lambda value: component_labels[value],
            )
        else:
            all_component = st.selectbox(
                "Component for all-module heatmap",
                options=["CT", "TS"],
                format_func=lambda value: f"{value} aggregate",
            )
        correlation_table = cached_all_module_correlations(
            method,
            resolved,
            all_feature,
            all_component,
            heatmap_diagnosis,
        )
        heatmap_data = correlation_table.copy()
        module_order = sorted(heatmap_data["module"].astype(int).unique())
        row_order = [f"M{value}" for value in module_order]
        heatmap_title = (
            f"All modules: {FEATURE_LABELS[all_feature]} · "
            f"{heatmap_data['component_label'].iloc[0]} · {heatmap_diagnosis}"
        )
        fdr_scope = (
            "Displayed-family FDR is Benjamini–Hochberg correction across all 154 modules "
            "and all numeric outcomes in this selected feature/component/diagnosis map."
        )

    correlation_heatmap = correlation_heatmap_figure(
        heatmap_data,
        value_column=value_column,
        p_column=p_column,
        fdr_column=fdr_column,
        title=heatmap_title,
        row_order=row_order,
        cluster_rows=cluster_rows,
        cluster_columns=cluster_columns,
    )
    st.plotly_chart(
        correlation_heatmap,
        width="stretch",
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"{method}_{heatmap_diagnosis}_correlation_heatmap",
                "scale": 3,
            },
        },
    )
    if heatmap_clustering != "None":
        st.caption(
            "Display order uses average-linkage hierarchical clustering with Euclidean "
            "distance. Missing coefficients are treated as zero only when determining "
            "the order; displayed correlations, p-values, FDRs, and downloads are unchanged."
        )
    st.caption(fdr_scope)
    correlation_columns = [
        "module",
        "metric_family",
        "component",
        "component_label",
        "diagnosis_group",
        "outcome",
        "outcome_label",
        "n",
        "pearson_r",
        "pearson_p",
        "pearson_fdr_displayed_family",
        "spearman_rho",
        "spearman_p",
        "spearman_fdr_displayed_family",
    ]
    with st.expander("Complete correlation table", expanded=False):
        st.dataframe(
            correlation_table[correlation_columns],
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Download complete correlation table (TSV)",
            data=dataframe_to_tsv_bytes(correlation_table[correlation_columns]),
            file_name=f"{method}_{heatmap_diagnosis}_lioness_outcome_correlations.tsv",
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
        width="stretch",
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
        st.dataframe(summary, width="stretch", hide_index=True)
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
        width="stretch",
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

with mdc_tab:
    st.subheader("Module differential connectivity (MDC)")
    st.caption(
        "MDC compares the mean absolute signedAlt adjacency in AD with Control. "
        "Values above 1 indicate higher connectivity in AD; values below 1 indicate "
        "higher connectivity in Control. Total uses all edges, TS uses same-tissue edges, "
        "and CT uses cross-tissue edges."
    )
    mdc_metadata = data_manifest.get("mdc", {})
    st.warning(
        "Cohort scope differs from the donor-complete LIONESS analysis. The MDC source "
        f"assembled {mdc_metadata.get('reference_assembled_donors', 517)} AD and "
        f"{mdc_metadata.get('target_assembled_donors', 408)} Control donors across the "
        "tissue union, including the 167 AD and 164 Control complete-three-tissue donors "
        "used here plus donors with partial tissue availability. MCI was not included. "
        "MDC is module-level context and does not change with the LIONESS method selector."
    )
    mdc_threshold = st.radio(
        "MDC significance threshold",
        options=[0.05, 0.10],
        format_func=lambda value: f"FDR < {value:.2f}",
        horizontal=True,
        key="mdc_threshold",
    )

    mdc_view = mdc_summary.copy()
    for scope in ["total", "ts", "ct"]:
        mdc_view[f"significant_{scope}"] = mdc_view[
            f"directional_fdr_{scope}"
        ].lt(mdc_threshold)
    mdc_view["any_significant"] = mdc_view[
        ["significant_total", "significant_ts", "significant_ct"]
    ].any(axis=1)

    count_columns = st.columns(4)
    count_columns[0].metric(
        "Total significant", int(mdc_view["significant_total"].sum())
    )
    count_columns[1].metric("TS significant", int(mdc_view["significant_ts"].sum()))
    count_columns[2].metric("CT significant", int(mdc_view["significant_ct"].sum()))
    count_columns[3].metric("Any MDC significant", int(mdc_view["any_significant"].sum()))

    selected_mdc = mdc_view.loc[mdc_view["module"].astype(int).eq(int(module))]
    if selected_mdc.empty:
        st.error(f"No MDC row is available for {module_label(module)}.")
    else:
        selected_mdc_row = selected_mdc.iloc[0]
        st.markdown(f"#### Selected module: {module_label(module)}")
        selected_columns = st.columns(3)
        for container, scope, label in zip(
            selected_columns,
            ["total", "ts", "ct"],
            ["Total MDC", "TS MDC", "CT MDC"],
            strict=True,
        ):
            ratio = selected_mdc_row[f"mdc_{scope}"]
            directional_fdr = selected_mdc_row[f"directional_fdr_{scope}"]
            direction = selected_mdc_row[f"direction_{scope}"]
            ratio_text = "NA" if pd.isna(ratio) else f"{float(ratio):.3f}"
            fdr_value_text = (
                "NA" if pd.isna(directional_fdr) else f"{float(directional_fdr):.3g}"
            )
            significance_text = (
                "Significant"
                if pd.notna(directional_fdr) and directional_fdr < mdc_threshold
                else "Not significant"
            )
            if pd.isna(ratio):
                significance_text = "Not available"
            with container:
                st.metric(label, ratio_text)
                st.caption(
                    f"{direction} · directional FDR={fdr_value_text} · {significance_text}"
                )

        selected_mdc_chart = mdc_module_figure(selected_mdc_row, mdc_threshold)
        st.plotly_chart(
            selected_mdc_chart,
            width="stretch",
            config={
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": f"M{module}_AD_Control_MDC",
                    "scale": 3,
                },
            },
        )
        st.caption(
            "Bars use log2 MDC so equal AD/Control connectivity is centered at zero. "
            "Orange indicates higher connectivity in AD, blue indicates higher connectivity "
            "in Control, and ★ marks significance at the selected threshold."
        )

    significance_filter = st.selectbox(
        "Modules to show in the MDC overview",
        options=[
            "All modules",
            "Any significant",
            "Total significant",
            "TS significant",
            "CT significant",
            "No significant MDC",
        ],
    )
    filter_masks = {
        "All modules": pd.Series(True, index=mdc_view.index),
        "Any significant": mdc_view["any_significant"],
        "Total significant": mdc_view["significant_total"],
        "TS significant": mdc_view["significant_ts"],
        "CT significant": mdc_view["significant_ct"],
        "No significant MDC": ~mdc_view["any_significant"],
    }
    displayed_mdc = mdc_view.loc[filter_masks[significance_filter]].copy()
    mdc_overview = mdc_overview_figure(displayed_mdc, module, mdc_threshold)
    st.plotly_chart(
        mdc_overview,
        width="stretch",
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "AD_Control_TS_CT_MDC_overview",
                "scale": 3,
            },
        },
    )
    st.caption(
        "Dashed zero lines separate AD-higher from Control-higher MDC. The dotted diagonal "
        "marks equal TS and CT effects. Six single-tissue modules have no CT edges and therefore "
        "no CT MDC point."
    )

    displayed_mdc["minimum_directional_fdr"] = displayed_mdc[
        ["directional_fdr_total", "directional_fdr_ts", "directional_fdr_ct"]
    ].min(axis=1, skipna=True)
    displayed_mdc["maximum_abs_log2_mdc"] = displayed_mdc[
        ["log2_mdc_total", "log2_mdc_ts", "log2_mdc_ct"]
    ].abs().max(axis=1, skipna=True)
    displayed_mdc = displayed_mdc.sort_values(
        ["minimum_directional_fdr", "maximum_abs_log2_mdc"],
        ascending=[True, False],
        na_position="last",
    )
    mdc_columns = [
        "module",
        "module_size_mapped",
        "mdc_total",
        "direction_total",
        "directional_fdr_total",
        "significant_total",
        "mdc_ts",
        "direction_ts",
        "directional_fdr_ts",
        "significant_ts",
        "mdc_ct",
        "direction_ct",
        "directional_fdr_ct",
        "significant_ct",
        "ts_minus_ct_log2_mdc",
        "n_ts_edges",
        "n_ct_edges",
    ]
    st.dataframe(
        displayed_mdc[mdc_columns],
        width="stretch",
        hide_index=True,
        column_config={
            "module": st.column_config.NumberColumn(format="M%d"),
            "mdc_total": st.column_config.NumberColumn(format="%.3f"),
            "mdc_ts": st.column_config.NumberColumn(format="%.3f"),
            "mdc_ct": st.column_config.NumberColumn(format="%.3f"),
            "directional_fdr_total": st.column_config.NumberColumn(format="%.3g"),
            "directional_fdr_ts": st.column_config.NumberColumn(format="%.3g"),
            "directional_fdr_ct": st.column_config.NumberColumn(format="%.3g"),
            "ts_minus_ct_log2_mdc": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.download_button(
        "Download displayed MDC table (TSV)",
        data=dataframe_to_tsv_bytes(displayed_mdc[mdc_columns]),
        file_name=f"AD_Control_MDC_{significance_filter.replace(' ', '_')}.tsv",
        mime="text/tab-separated-values",
    )
    st.info(
        "For each scope and direction, sample-permutation and gene-permutation p-values were "
        "Benjamini–Hochberg adjusted separately across the 154 modules. The displayed "
        "directional FDR is the larger of those two adjusted values for the observed MDC "
        "direction, making significance require support from both null models. The latest "
        "source used 200 sample permutations and 200 gene permutations."
    )

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
        statistics[display_columns], width="stretch", hide_index=True
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
            width="stretch",
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

with module_details_tab:
    st.subheader("Level-4 module composition")
    st.caption(
        "Module size, represented tissues, per-tissue gene counts, and proportions from "
        "the level-4 SE2 module-details file. Tissue proportions use module size as the denominator."
    )
    detail_columns = [
        "module",
        "module_size",
        "cluster_type",
        "tissues",
        "n_tissues",
        "dominant_tissue",
        "n_genes_ac",
        "proportion_ac",
        "n_genes_dlpfc",
        "proportion_dlpfc",
        "n_genes_pcg",
        "proportion_pcg",
    ]
    st.markdown(f"#### Selected module: {module_label(module)}")
    st.dataframe(
        module_details.loc[module_details["module"].astype(int).eq(int(module)), detail_columns],
        width="stretch",
        hide_index=True,
        column_config={
            "module": st.column_config.NumberColumn("Module", format="M%d"),
            "module_size": st.column_config.NumberColumn("Module genes", format="%d"),
            "cluster_type": "Module type",
            "tissues": "Tissues",
            "n_tissues": st.column_config.NumberColumn("N tissues", format="%d"),
            "dominant_tissue": "Dominant tissue",
            "n_genes_ac": st.column_config.NumberColumn("AC genes", format="%d"),
            "proportion_ac": st.column_config.NumberColumn("AC proportion", format="percent"),
            "n_genes_dlpfc": st.column_config.NumberColumn("DLPFC genes", format="%d"),
            "proportion_dlpfc": st.column_config.NumberColumn(
                "DLPFC proportion", format="percent"
            ),
            "n_genes_pcg": st.column_config.NumberColumn("PCG genes", format="%d"),
            "proportion_pcg": st.column_config.NumberColumn("PCG proportion", format="percent"),
        },
    )
    st.markdown("#### All 154 modules")
    st.dataframe(
        module_details[detail_columns],
        width="stretch",
        hide_index=True,
        height=520,
        column_config={
            "module": st.column_config.NumberColumn("Module", format="M%d"),
            "module_size": st.column_config.NumberColumn("Module genes", format="%d"),
            "cluster_type": "Module type",
            "tissues": "Tissues",
            "n_tissues": st.column_config.NumberColumn("N tissues", format="%d"),
            "dominant_tissue": "Dominant tissue",
            "n_genes_ac": st.column_config.NumberColumn("AC genes", format="%d"),
            "proportion_ac": st.column_config.NumberColumn("AC proportion", format="percent"),
            "n_genes_dlpfc": st.column_config.NumberColumn("DLPFC genes", format="%d"),
            "proportion_dlpfc": st.column_config.NumberColumn(
                "DLPFC proportion", format="percent"
            ),
            "n_genes_pcg": st.column_config.NumberColumn("PCG genes", format="%d"),
            "proportion_pcg": st.column_config.NumberColumn("PCG proportion", format="percent"),
        },
    )
    st.download_button(
        "Download complete module-details table (TSV)",
        data=dataframe_to_tsv_bytes(module_details[detail_columns]),
        file_name="se2_level4_module_details.tsv",
        mime="text/tab-separated-values",
    )
    st.info(
        "A tissue with one module gene can contribute CT edges but has no within-tissue "
        "gene pair, so its tissue-specific LIONESS feature is unavailable."
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

    st.markdown("#### Module differential connectivity")
    st.markdown(
        "MDC is an independent group-network comparison supplied as module-level context: "
        "mean absolute AD adjacency divided by mean absolute Control adjacency. The app "
        "shows total, same-tissue (TS), and cross-tissue (CT) ratios and their conservative "
        "directional permutation FDR. Its broader tissue-union AD/Control cohort and absence "
        "of MCI are stated in the MDC tab."
    )

    st.markdown("#### Module feature definitions")
    st.dataframe(cached_feature_definitions(), width="stretch", hide_index=True)
    st.markdown("#### Tissue labels")
    st.dataframe(cached_tissue_mapping(), width="stretch", hide_index=True)
    st.markdown("#### Public deploy safeguards")
    st.markdown(
        "The deploy Parquet files do not contain `projid`, the source donor identifier, "
        "or a donor-to-pseudonym key. Each point has a random-salted pseudonymous sample "
        "label that is stable within this bundle only. Diagnosis, the five primary phenotypes, "
        "and selected age, education, APOE, cognitive diagnosis, neuropathology, and Parkinsonism "
        "fields are included for color, hover, and correlation views. Public release still "
        "requires confirmation that the governing ROSMAP data-use agreement permits sharing "
        "these donor-level derived and metadata values."
    )
    st.caption(
        "Sex is displayed as source Code 0/1 because this deploy bundle does not infer a label "
        "without an accompanying reviewed data dictionary. Age at death is rounded to one decimal."
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
