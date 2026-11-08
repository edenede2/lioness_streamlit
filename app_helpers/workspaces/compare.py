from __future__ import annotations

from app_helpers.ui_runtime import (
    cached_module_details,
    cached_partition_comparison_summary,
    cached_partition_comparison_table,
    cached_partition_module_mapping,
    filterable_dataframe,
    partition_association_concordance_figure,
    partition_comparison_available,
    partition_entropy_figure,
    partition_mdc_concordance_figure,
    partition_overlap_figure,
    partition_proportion_figure,
    partition_size_figure,
    pd,
    render_plotly_chart,
    st,
    stateful_tabs,
    tab_is_open,
)


def render_partition_comparison_view() -> None:
    st.header("Partition comparison")
    st.caption(
        "All-donor CorShrink M1 biological-L4 modules versus the existing matched-donor "
        "full-cohort L4 modules. This contrast combines cohort expansion and correlation "
        "shrinkage; it does not isolate either change. Module numbers are never treated "
        "as equivalent across definitions."
    )
    if not partition_comparison_available():
        st.info("The validated partition-comparison bundle is not available yet.")
        return
    summary = cached_partition_comparison_summary()
    structure = summary.get("partition_comparison", summary)
    columns = st.columns(5)
    metrics = [
        ("New modules", structure.get("new_modules")),
        ("Matched modules", structure.get("matched_modules")),
        ("Common tissue–genes", structure.get("common_nodes")),
        ("ARI", structure.get("adjusted_rand_index")),
        ("NMI", structure.get("normalized_mutual_information")),
    ]
    for column, (label, value) in zip(columns, metrics):
        if isinstance(value, float):
            column.metric(label, f"{value:.3f}")
        else:
            column.metric(label, f"{int(value):,}" if value is not None else "NA")
    st.caption(
        "ARI and NMI are calculated on the tissue–gene nodes retained by both "
        "partitions. Coverage and overlap columns retain separate denominators for the "
        "new and matched-donor definitions."
    )

    mapping = cached_partition_module_mapping()
    tissue_overlap = cached_partition_comparison_table("tissue_gene_overlap")
    if not tissue_overlap.empty:
        st.subheader("Retained tissue–gene coverage")
        filterable_dataframe(
            tissue_overlap,
            table_key="partition_tissue_gene_overlap",
            table_name="Tissue-gene overlap",
        )
    new_details = cached_module_details("all_donor_corrshrink_l4").assign(
        partition="All-donor CorShrink M1"
    )
    matched_details = cached_module_details("full_cohort").assign(
        partition="Matched-donor full cohort"
    )
    structural = pd.concat([new_details, matched_details], ignore_index=True)
    size_tab, entropy_tab, type_tab = stateful_tabs(
        ["Module sizes", "Tissue-mixing entropy", "CT/TS composition"],
        key="partition_structure_tabs",
    )
    with size_tab:
        if tab_is_open(size_tab):
            render_plotly_chart(
                partition_size_figure(
                    structural,
                    x="module_size",
                    color="partition",
                    barmode="overlay",
                    opacity=0.65,
                    marginal="box",
                    nbins=35,
                    labels={"module_size": "Genes per module"},
                    title="Module-size distributions",
                ),
                key="partition_size_distribution",
            )
    with entropy_tab:
        if tab_is_open(entropy_tab):
            render_plotly_chart(
                partition_entropy_figure(
                    structural,
                    x="tissue_entropy_normalized",
                    color="partition",
                    barmode="overlay",
                    opacity=0.65,
                    marginal="box",
                    nbins=25,
                    labels={"tissue_entropy_normalized": "Normalized Shannon entropy"},
                    title="Continuous tissue-mixing distributions",
                ),
                key="partition_entropy_distribution",
            )
    with type_tab:
        if tab_is_open(type_tab):
            type_counts = (
                structural.groupby(["partition", "cluster_type"], observed=True)
                .size()
                .rename("modules")
                .reset_index()
            )
            type_counts["proportion"] = type_counts["modules"] / type_counts.groupby(
                "partition", observed=True
            )["modules"].transform("sum")
            render_plotly_chart(
                partition_proportion_figure(
                    type_counts,
                    x="partition",
                    y="proportion",
                    color="cluster_type",
                    barmode="stack",
                    text="modules",
                    title="CT/TS module proportions",
                    labels={
                        "proportion": "Proportion of modules",
                        "cluster_type": "Type",
                    },
                ),
                key="partition_ct_ts_proportions",
            )
    mapping_mode = st.radio(
        "Module correspondence",
        options=["best_many_to_one", "maximum_weight_one_to_one"],
        format_func=lambda value: {
            "best_many_to_one": "Best overlap (many-to-one)",
            "maximum_weight_one_to_one": "Maximum-weight one-to-one",
        }[value],
        horizontal=True,
    )
    selected_mapping = mapping.loc[mapping["mapping"].eq(mapping_mode)].copy()
    left, right = st.columns([1.1, 1])
    with left:
        figure = partition_overlap_figure(selected_mapping)
        render_plotly_chart(figure, key=f"partition_jaccard_{mapping_mode}")
    with right:
        selected_new_module = st.selectbox(
            "New all-donor CorShrink module",
            options=sorted(selected_mapping["new_module"].astype(int).unique()),
            format_func=lambda value: f"New M{value}",
        )
        selected_pair = selected_mapping.loc[
            selected_mapping["new_module"].astype(int).eq(int(selected_new_module))
        ]
        filterable_dataframe(
            selected_pair,
            table_key="partition_selected_pair",
            table_name="Selected partition pair",
            use_container_width=True,
            hide_index=True,
        )
    filterable_dataframe(
        selected_mapping.sort_values("jaccard", ascending=False),
        table_key=f"partition_mapping_{mapping_mode}",
        table_name="Module correspondence table",
    )

    association = cached_partition_comparison_table("association_concordance")
    mdc = cached_partition_comparison_table("mdc_concordance")
    kegg = cached_partition_comparison_table("kegg_concordance")
    association_tab, mdc_tab, kegg_tab = stateful_tabs(
        ["Association concordance", "MDC concordance", "KEGG overlap"],
        key="partition_concordance_tabs",
    )
    with association_tab:
        if tab_is_open(association_tab):
            if association.empty:
                st.info(
                    "Association concordance will appear after the new statistics finish."
                )
            else:
                filters = {}
                filter_columns = [
                    column
                    for column in (
                        "lioness_method",
                        "metric_family",
                        "component",
                        "phenotype",
                        "diagnosis_group",
                        "scale",
                        "correlation_method",
                    )
                    if column in association
                ]
                controls = st.columns(min(3, max(1, len(filter_columns))))
                for index, column in enumerate(filter_columns):
                    options = sorted(association[column].dropna().astype(str).unique())
                    filters[column] = controls[index % len(controls)].selectbox(
                        column.replace("_", " ").title(),
                        options,
                        key=f"partition_assoc_{column}",
                    )
                view = association.copy()
                for column, value in filters.items():
                    view = view.loc[view[column].astype(str).eq(value)]
                if {"new_effect", "matched_effect"}.issubset(view.columns):
                    figure = partition_association_concordance_figure(view)
                    render_plotly_chart(figure, key="partition_association_concordance")
                filterable_dataframe(
                    view,
                    table_key="partition_association_table",
                    table_name="Association concordance table",
                )
    with mdc_tab:
        if tab_is_open(mdc_tab):
            if mdc.empty:
                st.info(
                    "MDC concordance will appear after the new permutation run finishes."
                )
            else:
                if {"new_log2_mdc", "matched_log2_mdc"}.issubset(mdc.columns):
                    figure = partition_mdc_concordance_figure(mdc)
                    render_plotly_chart(figure, key="partition_mdc_concordance")
                filterable_dataframe(
                    mdc,
                    table_key="partition_mdc_table",
                    table_name="MDC concordance table",
                )
    with kegg_tab:
        if tab_is_open(kegg_tab):
            if kegg.empty:
                st.info("KEGG overlap will appear after enrichment finishes.")
            else:
                filterable_dataframe(
                    kegg,
                    table_key="partition_kegg_table",
                    table_name="KEGG overlap table",
                )
