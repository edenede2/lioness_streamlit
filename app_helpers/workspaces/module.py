from __future__ import annotations

from app_helpers.ui_runtime import (
    ASSOCIATION_GROUP_LABELS,
    ASSOCIATION_OUTCOME_LABELS,
    CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES,
    DIAGNOSIS_ORDER,
    EDGE_SCOPE_LABELS,
    EFFECT_DIRECTION_LABELS,
    ESTIMATOR_LABELS,
    HOVER_LABELS,
    MDC_ENRICHMENT_RESOLUTION_LABELS,
    MUSCLE_MODULE_SETS,
    OUTCOME_LABELS,
    PATHOLOGY_OUTCOMES,
    SCALE_LABELS,
    VALIDATION_SCOPE,
    _chart_helpers,
    association_kegg_subtitles,
    association_level_label,
    cached_aggregate,
    cached_categorical_module_set_associations,
    cached_distribution_pairwise_associations,
    cached_effect_endpoint_kegg,
    cached_effect_endpoint_kegg_tsv,
    cached_effect_endpoint_summary,
    cached_grouped_module_set_associations,
    cached_kegg,
    cached_kegg_tsv,
    cached_mdc_resolved,
    cached_mdc_summary,
    cached_module_set_cluster_associations,
    cached_pathway_mdc_rows,
    cached_resolved,
    cached_selected_distribution_pairwise,
    categorical_association_figure,
    collapse_pathway_mdc_rows,
    dataframe_to_tsv_bytes,
    default_reference_level,
    descriptive_eigengene_data_available,
    differential_mdc_data_available,
    distribution_contrasts,
    distribution_feature_heatmap_figure,
    distribution_figure,
    distribution_module_ranking_figure,
    distribution_omnibus_component_figure,
    distribution_pairwise_forest_figure,
    distribution_pairwise_heatmap_figure,
    distribution_summary,
    effect_endpoint_kegg_available,
    fdr_text,
    filter_kegg_enrichments,
    filter_kegg_region_sharing,
    filterable_dataframe,
    grouped_association_figure,
    kegg_region_combination_figure,
    kegg_region_heatmap_figure,
    load_mdc_normalization_manifest,
    load_mdc_normalization_summary,
    load_mdc_split_heatmap,
    mdc_entropy_figure,
    mdc_module_figure,
    mdc_overview_figure,
    mdc_resolved_heatmap_figure,
    mdc_resolved_module_figure,
    mdc_split_heatmap_available,
    module_entropy_figure,
    module_label,
    module_region_composition_figure,
    module_size_distribution_figure,
    np,
    pathway_mdc_detail_figure,
    pathway_mdc_heatmap_figure,
    pd,
    readable_method,
    render_plotly_chart,
    render_streamlit_image,
    selected_annotation,
    selected_kegg_table_annotation,
    st,
    stateful_tabs,
    summarize_pathway_mdc_rows,
    tab_is_open,
)


def render_associations(context):
    active_component_order = context["active_component_order"]
    active_feature_labels = context["active_feature_labels"]
    active_tissues = context["active_tissues"]
    active_view = context["active_view"]
    analysis_subset = context["analysis_subset"]
    annotation_fields = context["annotation_fields"]
    annotations = context["annotations"]
    association_interpretation = context["association_interpretation"]
    association_kegg_context = context["association_kegg_context"]
    association_metadata = context["association_metadata"]
    association_plot_type = context["association_plot_type"]
    association_significance_cutoff = context["association_significance_cutoff"]
    association_subtitles = context["association_subtitles"]
    cohort_scope = context["cohort_scope"]
    color_by = context["color_by"]
    continuous_colorscale = context["continuous_colorscale"]
    correlation_method = context["correlation_method"]
    diagnoses = context["diagnoses"]
    differential_edge_rule = context["differential_edge_rule"]
    differential_fdr_scope = context["differential_fdr_scope"]
    differential_fdr_threshold = context["differential_fdr_threshold"]
    download_prefix = context["download_prefix"]
    edge_rule = context["edge_rule"]
    effect_direction = context["effect_direction"]
    estimator = context["estimator"]
    feature = context["feature"]
    grouping_variable = context["grouping_variable"]
    match_effect_kegg_subtitle = context["match_effect_kegg_subtitle"]
    method = context["method"]
    minimum_group_n = context["minimum_group_n"]
    module = context["module"]
    module_count = context["module_count"]
    module_set = context["module_set"]
    module_set_label = context["module_set_label"]
    modules = context["modules"]
    phenotype = context["phenotype"]
    plot_data = context.get("plot_data")
    point_color_labels = context["point_color_labels"]
    pooled_line_dash = context["pooled_line_dash"]
    resolution = context["resolution"]
    resolved = context["resolved"]
    reverse_colorscale = context["reverse_colorscale"]
    scale = context["scale"]
    score_normalization = context["score_normalization"]
    selected_components = context["selected_components"]
    selected_group_levels = context["selected_group_levels"]
    show_group_trend_lines = context["show_group_trend_lines"]
    show_pooled_association = context["show_pooled_association"]
    single_point_color = context["single_point_color"]
    tissue_composition = context["tissue_composition"]
    trend_line_rule = context["trend_line_rule"]
    if active_view == "Associations" and association_plot_type == "Score comparison":
        if association_interpretation != "numeric":
            st.info(
                "Score comparison requires a numeric/ordinal correlation outcome. Nominal category codes are never correlated."
            )
            st.stop()
        from app_helpers.association_score_comparison import render_score_comparison

        comparison_metadata = association_metadata.copy()
        if grouping_variable != "__all__":
            comparison_metadata = comparison_metadata.loc[
                comparison_metadata[grouping_variable].notna()
                & comparison_metadata[grouping_variable].isin(selected_group_levels)
            ].copy()

        def comparison_load_scores(comparison_module, family, tissue_resolved):
            if family == "eigengene":
                return cached_resolved(
                    module_set,
                    "lioness",
                    method,
                    comparison_module,
                    family,
                    "all",
                    "all",
                    "global",
                    0.05,
                    "standard_pruned",
                    cohort_scope,
                )
            common_args = (
                module_set,
                estimator,
                method,
                comparison_module,
                family,
                edge_rule,
                differential_edge_rule,
                differential_fdr_scope,
                differential_fdr_threshold,
                score_normalization,
            )
            return (
                cached_resolved(*common_args, cohort_scope)
                if tissue_resolved
                else cached_aggregate(*common_args)
            )

        def comparison_enrichment(comparison_module, components):
            table = (
                cached_effect_endpoint_kegg(
                    module_set, method, effect_direction, comparison_module
                )
                if match_effect_kegg_subtitle
                else cached_kegg(module_set, comparison_module)
            )
            aggregate_annotation = (
                selected_kegg_table_annotation(table)
                if match_effect_kegg_subtitle
                else selected_annotation(annotations, comparison_module)
            )
            subtitles = {}
            for is_resolved in (False, True):
                selected = [
                    c for c in components if (c not in {"CT", "TS"}) == is_resolved
                ]
                subtitles.update(
                    association_kegg_subtitles(
                        table,
                        selected,
                        resolved=is_resolved,
                        aggregate_annotation=aggregate_annotation,
                    )
                )
            if association_kegg_context:
                subtitles = {
                    c: s.replace(
                        "KEGG enrichment",
                        f"KEGG enrichment [{association_kegg_context}]",
                        1,
                    )
                    for c, s in subtitles.items()
                }
            return subtitles

        render_score_comparison(
            modules=modules,
            selected_module=module,
            tissues=active_tissues,
            default_regions=tissue_composition.loc[
                tissue_composition["Genes"].gt(0), "Tissue"
            ].tolist(),
            default_network_components=tuple(
                c
                for c, available in (
                    ("TS", tissue_composition["Genes"].ge(2).any()),
                    ("CT", tissue_composition["Genes"].gt(0).sum() >= 2),
                )
                if available
            ),
            component_labels={c: EDGE_SCOPE_LABELS[c] for c in active_component_order},
            cohort_scope=cohort_scope,
            eigengenes_available=cohort_scope == "complete_450"
            and descriptive_eigengene_data_available(module_set),
            metadata=comparison_metadata,
            load_scores=comparison_load_scores,
            outcome=phenotype,
            outcome_labels=OUTCOME_LABELS,
            scale=scale,
            scale_label=SCALE_LABELS[scale],
            correlation_method=correlation_method.lower(),
            minimum_n=minimum_group_n,
            enrichment=comparison_enrichment,
            render_plot=render_plotly_chart,
            table_renderer=filterable_dataframe,
            module_definition=module_set_label,
            analysis_context=dict(
                module_set=module_set,
                estimator=estimator,
                network_method=method,
                cohort_scope=cohort_scope,
                score_scale=scale,
                edge_rule=edge_rule,
                differential_edge_rule=differential_edge_rule,
                differential_fdr_scope=differential_fdr_scope,
                differential_fdr_threshold=differential_fdr_threshold,
                score_normalization=score_normalization,
                evaluation_subset=analysis_subset,
                diagnosis_filter=" / ".join(diagnoses),
                grouping_variable=grouping_variable,
                grouping_levels=" / ".join(map(str, selected_group_levels)),
            ),
            scatter_options=dict(
                grouping_variable=grouping_variable,
                grouping_levels=selected_group_levels,
                grouping_labels={
                    str(v): association_level_label(grouping_variable, v)
                    for v in selected_group_levels
                },
                color_by=color_by,
                color_label=point_color_labels[color_by],
                hover_fields=HOVER_LABELS,
                correlation_method=correlation_method.lower(),
                annotation_fields=annotation_fields,
                trend_line_rule=trend_line_rule,
                significance_cutoff=association_significance_cutoff,
                minimum_group_n=minimum_group_n,
                show_group_trends=show_group_trend_lines,
                show_pooled=show_pooled_association,
                pooled_label="All displayed donors (pooled)",
                continuous_colorscale=continuous_colorscale,
                reverse_colorscale=reverse_colorscale,
                single_point_color=(
                    single_point_color if color_by == "__single_color__" else None
                ),
                pooled_line_dash=pooled_line_dash,
                categorical_color_fields=CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES,
            ),
        )
        st.stop()

    if active_view == "Associations":
        if association_interpretation == "numeric":
            module_set_associations = cached_grouped_module_set_associations(
                module_set,
                module_count,
                estimator,
                method,
                resolved,
                feature,
                phenotype,
                scale,
                tuple(selected_components),
                tuple(diagnoses),
                grouping_variable,
                tuple(selected_group_levels),
                True,
                minimum_group_n,
                edge_rule,
                differential_edge_rule,
                differential_fdr_scope,
                differential_fdr_threshold,
                score_normalization,
                analysis_subset,
                cohort_scope,
            )
            if not show_pooled_association and grouping_variable != "__all__":
                module_set_associations = module_set_associations.loc[
                    ~module_set_associations["grouping_level"].eq("__pooled__")
                ].copy()
        else:
            module_set_associations = cached_categorical_module_set_associations(
                module_set,
                module_count,
                estimator,
                method,
                resolved,
                feature,
                phenotype,
                scale,
                tuple(selected_components),
                tuple(diagnoses),
                tuple(selected_group_levels),
                minimum_group_n,
                edge_rule,
                differential_edge_rule,
                differential_fdr_scope,
                differential_fdr_threshold,
                score_normalization,
                analysis_subset,
                cohort_scope,
            )
        displayed_association_statistics = module_set_associations.loc[
            module_set_associations["module"].astype(int).eq(int(module))
        ].copy()
        pooled_label = (
            "All donors (pooled)"
            if set(diagnoses) == set(DIAGNOSIS_ORDER)
            and analysis_subset == "all_donors"
            else "All displayed donors (pooled)"
        )
        st.subheader("Phenotype association")
        if association_interpretation == "categorical":
            st.caption(
                (
                    "Cluster labels are nominal categories, not numbers. "
                    if phenotype == "clusters"
                    else f"{ASSOCIATION_OUTCOME_LABELS[phenotype]} is treated as a categorical outcome. "
                )
                + "Each panel shows its network-score distributions; category codes are never "
                "entered into Pearson or Spearman calculations. The Kruskal–Wallis omnibus "
                f"test includes levels with at least {minimum_group_n} displayed donors; smaller "
                "levels remain visible but are identified as excluded. Epsilon-squared is the effect "
                f"size. FDR is Benjamini–Hochberg adjusted only across the {module_count} "
                "modules in the selected definition for the fixed estimator, network method, "
                "feature, component, cohort, edge rule, and score normalization. Pearson, "
                "Spearman, and regression lines are intentionally not calculated."
            )
        else:
            st.caption(
                f"{correlation_method} · grouped by {ASSOCIATION_GROUP_LABELS[grouping_variable]} · OLS lines are visual guides."
            )
            with st.expander("Statistics and interpretation", expanded=False):
                st.caption(
                    "Selected levels control points, lines, annotations, legend groups, and downloads. "
                    "Diagnosis is a donor-inclusion filter and marker shape. "
                    f"Pearson and Spearman BH FDR uses only the {module_count} modules in this definition, "
                    "with estimator, method, feature, component, outcome, cohort, edge rules, score handling, "
                    "grouping variable and level held fixed. Missing and constant tests are excluded. "
                    "The pooled association is recalculated after donor and level filtering. Legend entries "
                    "toggle each group's points and line together. KEGG subtitles are component-specific."
                )
        grouping_labels = {
            str(value): association_level_label(
                (
                    phenotype
                    if association_interpretation == "categorical"
                    else grouping_variable
                ),
                value,
            )
            for value in selected_group_levels
        }
        figure = (
            categorical_association_figure(
                plot_data,
                displayed_association_statistics,
                category_variable=phenotype,
                category_label=ASSOCIATION_OUTCOME_LABELS[phenotype],
                category_levels=selected_group_levels,
                category_labels=grouping_labels,
                feature_label=active_feature_labels[feature],
                scale_label=SCALE_LABELS[scale],
                module=module,
                minimum_group_n=minimum_group_n,
                annotation_fields=annotation_fields,
                module_definition=module_set_label,
                kegg_subtitles=association_subtitles,
                hover_fields=HOVER_LABELS,
                single_point_color=(
                    single_point_color if color_by == "__single_color__" else None
                ),
            )
            if association_interpretation == "categorical"
            else grouped_association_figure(
                plot_data,
                displayed_association_statistics,
                phenotype=phenotype,
                phenotype_label=OUTCOME_LABELS[phenotype],
                feature_label=active_feature_labels[feature],
                scale_label=SCALE_LABELS[scale],
                grouping_variable=grouping_variable,
                grouping_levels=selected_group_levels,
                grouping_labels=grouping_labels,
                module=module,
                color_by=color_by,
                color_label=point_color_labels[color_by],
                hover_fields=HOVER_LABELS,
                correlation_method=correlation_method.lower(),
                annotation_fields=annotation_fields,
                trend_line_rule=trend_line_rule,
                significance_cutoff=association_significance_cutoff,
                minimum_group_n=minimum_group_n,
                show_group_trends=show_group_trend_lines,
                show_pooled=show_pooled_association,
                pooled_label=pooled_label,
                module_definition=module_set_label,
                continuous_colorscale=continuous_colorscale,
                reverse_colorscale=reverse_colorscale,
                single_point_color=(
                    single_point_color if color_by == "__single_color__" else None
                ),
                pooled_line_dash=pooled_line_dash,
                categorical_color_fields=CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES,
                kegg_subtitles=association_subtitles,
            )
        )
        from app_helpers.workspaces.shared import association_header

        figure = association_header(figure)
        render_plotly_chart(
            figure,
            use_container_width=True,
            config={
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": (
                        f"{download_prefix}M{module}_{phenotype}_{feature}_{method}_"
                        f"{association_interpretation}_{correlation_method.lower()}"
                    ),
                    "scale": 3,
                },
            },
            plot_id="main.figure",
            metadata={
                "module_set": module_set,
                "method": method,
                "module": f"M{module}",
                "outcome": phenotype,
                "feature": feature,
            },
        )
        public_columns = list(
            dict.fromkeys(
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
                    *[column for column in HOVER_LABELS if column != phenotype],
                ]
            )
        )
        public_plot_data = plot_data[public_columns].rename(
            columns={
                "metric_value": f"feature_{scale}",
                "lioness_method": "network_method",
            }
        )
        public_plot_data.insert(0, "module_definition", module_set_label)
        st.download_button(
            "Download displayed plot data (TSV)",
            data=dataframe_to_tsv_bytes(public_plot_data),
            file_name=(
                f"{download_prefix}M{module}_{phenotype}_{feature}_{method}_"
                f"{resolution.replace(' ', '_')}.tsv"
            ),
            mime="text/tab-separated-values",
        )
        association_table = displayed_association_statistics.copy()
        if association_interpretation == "numeric":
            association_table["grouping_label"] = association_table[
                "grouping_label"
            ].replace({"All displayed donors (pooled)": pooled_label})
            if "diagnosis_group" in association_table:
                association_table["diagnosis_group"] = association_table[
                    "diagnosis_group"
                ].replace({"All donors": pooled_label})
            else:
                association_table["diagnosis_group"] = " / ".join(diagnoses)
        association_table.insert(0, "module_definition", module_set_label)
        association_columns = (
            [
                "module_definition",
                "module",
                "metric_family",
                "component_label",
                "category_variable",
                "outcome",
                "n",
                "n_tested",
                "k_tested",
                "level_counts",
                "level_medians",
                "levels_tested",
                "levels_excluded_small_n",
                "kruskal_h",
                "kruskal_df",
                "epsilon_squared",
                "categorical_p",
                "categorical_fdr_across_modules",
                "categorical_fdr_module_family_n",
            ]
            if association_interpretation == "categorical"
            else [
                "module_definition",
                "module",
                "metric_family",
                "component_label",
                "diagnosis_group",
                "grouping_variable",
                "grouping_level",
                "grouping_label",
                "is_pooled",
                "outcome",
                "n",
                "minimum_group_n",
                "eligible",
                "unavailable_reason",
                "pearson_r",
                "pearson_p",
                "pearson_fdr_across_modules",
                "pearson_fdr_module_family_n",
                "spearman_rho",
                "spearman_p",
                "spearman_fdr_across_modules",
                "spearman_fdr_module_family_n",
            ]
        )
        with st.expander(
            (
                "Kruskal–Wallis statistics · BH across modules"
                if association_interpretation == "categorical"
                else "Pearson and Spearman statistics · BH across modules"
            ),
            expanded=False,
        ):
            filterable_dataframe(
                association_table[association_columns],
                table_key="association_module_set_fdr",
                table_name="Association statistics",
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download displayed association statistics (TSV)",
                data=dataframe_to_tsv_bytes(association_table[association_columns]),
                file_name=(
                    f"{download_prefix}M{module}_{phenotype}_{feature}_{method}_"
                    "association_module_set_fdr.tsv"
                ),
                mime="text/tab-separated-values",
            )


def render_distributions(context):
    active_component_order = context["active_component_order"]
    active_feature_labels = context["active_feature_labels"]
    analysis_subset = context["analysis_subset"]
    cohort_scope = context["cohort_scope"]
    descriptive_eigengenes_available = context["descriptive_eigengenes_available"]
    diagnoses = context["diagnoses"]
    differential_edge_rule = context["differential_edge_rule"]
    differential_fdr_scope = context["differential_fdr_scope"]
    differential_fdr_threshold = context["differential_fdr_threshold"]
    distribution_grouping_variable = context["distribution_grouping_variable"]
    distribution_minimum_group_n = context["distribution_minimum_group_n"]
    download_prefix = context["download_prefix"]
    edge_rule = context["edge_rule"]
    estimator = context["estimator"]
    feature = context["feature"]
    method = context["method"]
    module = context["module"]
    module_count = context["module_count"]
    module_set = context["module_set"]
    module_set_label = context["module_set_label"]
    phenotype = context["phenotype"]
    plot_data = context["plot_data"]
    resolved = context["resolved"]
    scale = context["scale"]
    score_normalization = context["score_normalization"]
    selected_components = context["selected_components"]
    selected_distribution_levels = context["selected_distribution_levels"]
    st.subheader("Donor-level feature distributions")
    if not selected_distribution_levels:
        st.warning("Select at least one distribution group level in the sidebar.")
        st.stop()
    distribution_plot_data = plot_data.copy()
    if distribution_grouping_variable == "__all__":
        distribution_plot_data["distribution_group"] = "All displayed donors"
        distribution_groups = ["All displayed donors"]
    else:
        distribution_plot_data["distribution_group"] = distribution_plot_data[
            distribution_grouping_variable
        ].map(
            lambda value: association_level_label(distribution_grouping_variable, value)
        )
        distribution_groups = [
            association_level_label(distribution_grouping_variable, value)
            for value in selected_distribution_levels
        ]
    st.caption(
        "These views use only the selected module feature; no phenotype is on an axis. "
        f"Distributions are grouped by {ASSOCIATION_GROUP_LABELS[distribution_grouping_variable]}. "
        "Rank-based differentiation analyses are exploratory and use only the displayed "
        "donors and selected category levels."
    )
    distribution_analysis = st.radio(
        "Distribution analysis",
        [
            "Distribution shape",
            "Overall differentiation",
            "Pairwise differentiation",
            "Feature comparison heatmaps",
            "Find differentiated modules",
        ],
        horizontal=True,
        help="Only the selected analysis is calculated and rendered.",
        index=4 if context.get("distribution_search") else 0,
    )

    def distribution_omnibus_catalog() -> pd.DataFrame:
        return cached_categorical_module_set_associations(
            module_set,
            module_count,
            estimator,
            method,
            resolved,
            feature,
            distribution_grouping_variable,
            scale,
            tuple(selected_components),
            tuple(diagnoses),
            tuple(selected_distribution_levels),
            distribution_minimum_group_n,
            edge_rule,
            differential_edge_rule,
            differential_fdr_scope,
            differential_fdr_threshold,
            score_normalization,
            analysis_subset,
            cohort_scope,
        )

    def distribution_pairwise_setup(
        key_prefix: str,
        *,
        show_ks_control: bool = True,
    ) -> tuple[tuple[tuple[object, object], ...], bool]:
        unique_donors = plot_data[
            ["sample_id", distribution_grouping_variable]
        ].drop_duplicates()
        counts = unique_donors[distribution_grouping_variable].value_counts()
        default_reference = default_reference_level(
            distribution_grouping_variable,
            selected_distribution_levels,
            counts,
            minimum_group_n=distribution_minimum_group_n,
        )
        contrast_mode_label = st.radio(
            "Pairwise contrasts",
            ["Reference vs other groups", "All selected pairs"],
            horizontal=True,
            key=f"{key_prefix}_contrast_mode",
        )
        contrast_mode = (
            "reference"
            if contrast_mode_label == "Reference vs other groups"
            else "all_pairs"
        )
        reference_level = None
        if contrast_mode == "reference":
            eligible_references = [
                value
                for value in selected_distribution_levels
                if int(counts.get(value, 0)) >= distribution_minimum_group_n
            ]
            if eligible_references:
                reference_level = st.selectbox(
                    "Reference group",
                    options=eligible_references,
                    index=(
                        eligible_references.index(default_reference)
                        if default_reference in eligible_references
                        else 0
                    ),
                    format_func=lambda value: association_level_label(
                        distribution_grouping_variable, value
                    ),
                    key=f"{key_prefix}_reference_level",
                )
        include_ks = (
            st.checkbox(
                "Include advanced KS shape diagnostic",
                value=False,
                key=f"{key_prefix}_include_ks",
                help=(
                    "The two-sided Kolmogorov–Smirnov test detects any empirical-distribution "
                    "difference, including spread and tail changes."
                ),
            )
            if show_ks_control
            else False
        )
        if contrast_mode == "reference" and reference_level is None:
            return (), include_ks
        return (
            tuple(
                distribution_contrasts(
                    selected_distribution_levels,
                    mode=contrast_mode,
                    reference_level=reference_level,
                )
            ),
            include_ks,
        )

    def label_pairwise(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["reference_label"] = result["reference_level"].map(
            lambda value: association_level_label(distribution_grouping_variable, value)
        )
        result["comparison_label"] = result["comparison_level"].map(
            lambda value: association_level_label(distribution_grouping_variable, value)
        )
        result["contrast_label"] = (
            result["comparison_label"] + " vs " + result["reference_label"]
        )
        return result

    def module_set_pairwise_catalog(
        contrasts: tuple[tuple[object, object], ...],
    ) -> pd.DataFrame:
        return cached_distribution_pairwise_associations(
            module_set,
            estimator,
            method,
            resolved,
            feature,
            distribution_grouping_variable,
            scale,
            tuple(selected_components),
            tuple(diagnoses),
            tuple(selected_distribution_levels),
            contrasts,
            distribution_minimum_group_n,
            edge_rule,
            differential_edge_rule,
            differential_fdr_scope,
            differential_fdr_threshold,
            score_normalization,
            analysis_subset,
            cohort_scope,
        )

    if distribution_analysis == "Distribution shape":
        st.caption(
            "Histogram heights are probability densities with bins shared across groups. "
            "Rainclouds combine donor points, density, median, and IQR; ECDFs are bin-free."
        )
        chart_col, bin_col = st.columns([2, 1])
        chart_type = chart_col.radio(
            "Distribution view",
            ["Histogram", "Violin", "Raincloud", "ECDF"],
            horizontal=True,
        )
        bins = bin_col.slider(
            "Histogram bins", 10, 80, 30, disabled=chart_type != "Histogram"
        )
        distribution = distribution_figure(
            distribution_plot_data,
            feature_label=active_feature_labels[feature],
            scale_label=SCALE_LABELS[scale],
            diagnoses=distribution_groups,
            module=module,
            chart_type=chart_type,
            bins=bins,
            module_definition=module_set_label,
            group_column="distribution_group",
            group_label=ASSOCIATION_GROUP_LABELS[distribution_grouping_variable],
        )
        render_plotly_chart(
            distribution,
            use_container_width=True,
            config={
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": (
                        f"{download_prefix}M{module}_{feature}_{method}_distribution"
                    ),
                    "scale": 3,
                },
            },
            plot_id="main.distribution",
            metadata={
                "module_set": module_set,
                "method": method,
                "module": f"M{module}",
                "outcome": phenotype,
                "feature": feature,
            },
        )
        summary = distribution_summary(
            distribution_plot_data,
            group_column="distribution_group",
            group_label="distribution_group",
        )
        summary.insert(0, "module_definition", module_set_label)
        summary.insert(1, "grouping_variable", distribution_grouping_variable)
        with st.expander("Distribution summary table", expanded=False):
            filterable_dataframe(
                summary,
                table_key="distribution_summary",
                table_name="Distribution summary",
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download distribution summary (TSV)",
                data=dataframe_to_tsv_bytes(summary),
                file_name=(
                    f"{download_prefix}M{module}_{feature}_{method}_"
                    f"{distribution_grouping_variable}_distribution_summary.tsv"
                ),
                mime="text/tab-separated-values",
            )

    elif distribution_grouping_variable == "__all__":
        st.info(
            "Choose a categorical grouping variable to calculate distribution "
            "differentiation statistics."
        )

    elif distribution_analysis == "Feature comparison heatmaps":
        st.caption(
            "Compare group-differentiation effect sizes across module scores. Each "
            "asterisk marks a component-specific module-set FDR below the selected "
            "cutoff; missing structural combinations remain blank, not zero."
        )
        heatmap_feature_options = [
            value for value in active_feature_labels if resolved or value != "eigengene"
        ]
        default_heatmap_features = (
            [feature] if feature in heatmap_feature_options else []
        )
        if "connectivity" in heatmap_feature_options:
            default_heatmap_features = ["connectivity"]
        if resolved and "eigengene" in heatmap_feature_options:
            default_heatmap_features.append("eigengene")
        default_heatmap_features = list(dict.fromkeys(default_heatmap_features))
        selected_heatmap_features = st.multiselect(
            "Features to compare",
            options=heatmap_feature_options,
            default=default_heatmap_features,
            format_func=lambda value: active_feature_labels[value],
            help=(
                "Select Connectivity and Module eigengene to compare network topology "
                "with tissue-level transcriptomic variation directly."
            ),
        )
        if not resolved and descriptive_eigengenes_available:
            st.info(
                "Module eigengenes are tissue-specific. Switch Resolution to Tissue "
                "resolved to include eigengenes in this heatmap."
            )
        heatmap_component_options = (
            list(active_component_order) if resolved else ["CT", "TS"]
        )
        selected_heatmap_components = st.multiselect(
            "Components to compare",
            options=heatmap_component_options,
            default=heatmap_component_options,
            format_func=lambda value: EDGE_SCOPE_LABELS.get(value, value),
        )
        heatmap_scope = st.radio(
            "Differentiation heatmap scope",
            ["Selected module", f"All {module_count} modules"],
            horizontal=True,
        )
        heatmap_significance_threshold = st.radio(
            "Heatmap significance cutoff",
            [0.05, 0.10],
            horizontal=True,
            format_func=lambda value: (
                f"FDR < {value:.2f}" + (" (exploratory)" if value == 0.10 else "")
            ),
        )
        all_module_layout = "Compact feature summary"
        all_module_row_filter = "No FDR filter"
        all_module_top_n = min(20, module_count)
        if heatmap_scope.startswith("All"):
            all_module_layout = st.radio(
                "All-module heatmap layout",
                ["Compact feature summary", "Detailed feature × component"],
                horizontal=True,
                help=(
                    "Compact mode shows the strongest selected component for each "
                    "module-feature pair. Detailed mode preserves every feature-component "
                    "cell. Hover always identifies the underlying component."
                ),
            )
            all_module_row_filter = st.radio(
                "Modules to include",
                [
                    "No FDR filter",
                    "At least one FDR < 0.05",
                    "At least one FDR < 0.10 (exploratory)",
                ],
                horizontal=True,
            )
            top_options = list(dict.fromkeys([20, 50, 100, module_count]))
            all_module_top_n = st.selectbox(
                "Rows in all-module heatmap",
                top_options,
                index=0,
                format_func=lambda value: (
                    f"All {module_count} modules"
                    if value == module_count
                    else f"Top {value} modules"
                ),
            )
        run_feature_heatmap = st.toggle(
            "Calculate feature differentiation heatmap",
            value=False,
            help=(
                "Runs one memory-bounded scan across the module definition and caches "
                "all available features for subsequent heatmap selections."
            ),
        )
        if not selected_heatmap_features or not selected_heatmap_components:
            st.info("Select at least one feature and one component.")
        elif run_feature_heatmap:
            feature_heatmap_catalog = cached_categorical_module_set_associations(
                module_set,
                module_count,
                estimator,
                method,
                resolved,
                "__all__",
                distribution_grouping_variable,
                scale,
                tuple(selected_heatmap_components),
                tuple(diagnoses),
                tuple(selected_distribution_levels),
                distribution_minimum_group_n,
                edge_rule,
                differential_edge_rule,
                differential_fdr_scope,
                differential_fdr_threshold,
                score_normalization,
                analysis_subset,
                cohort_scope,
            ).copy()
            feature_heatmap_catalog = feature_heatmap_catalog.loc[
                feature_heatmap_catalog["metric_family"].isin(selected_heatmap_features)
                & feature_heatmap_catalog["component"].isin(selected_heatmap_components)
            ].copy()
            feature_heatmap_catalog["feature_label"] = feature_heatmap_catalog[
                "metric_family"
            ].map(active_feature_labels)
            feature_heatmap_catalog["component_label"] = (
                feature_heatmap_catalog["component"]
                .map(EDGE_SCOPE_LABELS)
                .fillna(feature_heatmap_catalog["component_label"])
            )
            feature_heatmap_catalog["epsilon_squared"] = pd.to_numeric(
                feature_heatmap_catalog["epsilon_squared"], errors="coerce"
            )
            feature_order = [
                active_feature_labels[value] for value in selected_heatmap_features
            ]
            component_order = [
                EDGE_SCOPE_LABELS.get(value, value)
                for value in selected_heatmap_components
            ]
            if feature_heatmap_catalog.empty:
                st.info("No differentiation statistics are available for these scores.")
            else:
                if heatmap_scope == "Selected module":
                    displayed_heatmap = feature_heatmap_catalog.loc[
                        feature_heatmap_catalog["module"].astype(int).eq(int(module))
                    ].copy()
                    displayed_heatmap["heatmap_row"] = displayed_heatmap[
                        "feature_label"
                    ]
                    displayed_heatmap["heatmap_column"] = displayed_heatmap[
                        "component_label"
                    ]
                    row_order = feature_order
                    column_order = component_order
                    heatmap_title = (
                        f"Feature differentiation for module M{module}<br><sup>"
                        f"{module_set_label} · Kruskal–Wallis epsilon-squared · "
                        f"* module-set FDR < {heatmap_significance_threshold:.2f}</sup>"
                    )
                    heatmap_table = displayed_heatmap.copy()
                else:
                    module_peaks = (
                        feature_heatmap_catalog.groupby("module", observed=True)[
                            "epsilon_squared"
                        ]
                        .max()
                        .sort_values(ascending=False, na_position="last")
                    )
                    fdr_limit = {
                        "No FDR filter": None,
                        "At least one FDR < 0.05": 0.05,
                        "At least one FDR < 0.10 (exploratory)": 0.10,
                    }[all_module_row_filter]
                    if fdr_limit is not None:
                        eligible_modules = feature_heatmap_catalog.loc[
                            pd.to_numeric(
                                feature_heatmap_catalog[
                                    "categorical_fdr_across_modules"
                                ],
                                errors="coerce",
                            ).lt(fdr_limit),
                            "module",
                        ].unique()
                        module_peaks = module_peaks.loc[
                            module_peaks.index.isin(eligible_modules)
                        ]
                    shown_modules = module_peaks.head(
                        int(all_module_top_n)
                    ).index.tolist()
                    source_heatmap = feature_heatmap_catalog.loc[
                        feature_heatmap_catalog["module"].isin(shown_modules)
                    ].copy()
                    source_heatmap["module_peak_epsilon_squared"] = source_heatmap[
                        "module"
                    ].map(module_peaks)
                    if all_module_layout == "Compact feature summary":
                        valid = source_heatmap.loc[
                            source_heatmap["epsilon_squared"].notna()
                        ].copy()
                        strongest_index = valid.groupby(
                            ["module", "metric_family"], observed=True, sort=False
                        )["epsilon_squared"].idxmax()
                        displayed_heatmap = valid.loc[strongest_index].copy()
                        displayed_heatmap["heatmap_column"] = displayed_heatmap[
                            "feature_label"
                        ]
                        column_order = feature_order
                        layout_note = (
                            "Each cell is the strongest selected component; this "
                            "post-hoc maximum is descriptive. The star and hover FDR "
                            "belong to that winning component, not to a max-adjusted test."
                        )
                    else:
                        displayed_heatmap = source_heatmap.copy()
                        displayed_heatmap["heatmap_column"] = (
                            displayed_heatmap["feature_label"].astype(str)
                            + " · "
                            + displayed_heatmap["component_label"].astype(str)
                        )
                        column_order = [
                            f"{feature_label} · {component_label}"
                            for feature_label in feature_order
                            for component_label in component_order
                        ]
                        layout_note = (
                            "Each cell is one prespecified feature-component test; "
                            "blank cells are structurally unavailable."
                        )
                    displayed_heatmap["heatmap_row"] = "M" + displayed_heatmap[
                        "module"
                    ].astype(int).astype(str)
                    row_order = [f"M{int(value)}" for value in shown_modules]
                    heatmap_title = (
                        f"All-module feature differentiation<br><sup>{module_set_label} · "
                        f"modules ordered by maximum epsilon-squared · "
                        f"* module-set FDR < {heatmap_significance_threshold:.2f}</sup>"
                    )
                    heatmap_table = feature_heatmap_catalog.copy()
                    heatmap_table["module_peak_epsilon_squared"] = heatmap_table[
                        "module"
                    ].map(module_peaks)
                    heatmap_table["shown_in_heatmap"] = heatmap_table["module"].isin(
                        shown_modules
                    )
                    st.caption(layout_note)
                if displayed_heatmap.empty:
                    st.info("No modules meet the selected heatmap filters.")
                else:
                    render_plotly_chart(
                        distribution_feature_heatmap_figure(
                            displayed_heatmap,
                            title=heatmap_title,
                            row_order=row_order,
                            column_order=column_order,
                            significance_threshold=heatmap_significance_threshold,
                        ),
                        use_container_width=True,
                        config={"displaylogo": False},
                        plot_id="main.distribution_feature_heatmap_figure",
                        metadata={
                            "module_set": module_set,
                            "method": method,
                            "module": f"M{module}",
                            "outcome": phenotype,
                            "feature": feature,
                        },
                    )
                st.caption(
                    "BH is calculated across valid modules only, separately for each "
                    "fixed feature and component. Features, components, or group contrasts "
                    "are never multiplied into the same FDR family."
                )
                table_columns = [
                    "module",
                    "metric_family",
                    "feature_label",
                    "component",
                    "component_label",
                    "n",
                    "n_tested",
                    "k_tested",
                    "levels_tested",
                    "levels_excluded_small_n",
                    "level_counts",
                    "level_medians",
                    "kruskal_h",
                    "epsilon_squared",
                    "categorical_p",
                    "categorical_fdr_across_modules",
                    "categorical_fdr_module_family_n",
                    "eligible",
                    "unavailable_reason",
                ]
                table_columns.extend(
                    column
                    for column in ("module_peak_epsilon_squared", "shown_in_heatmap")
                    if column in heatmap_table
                )
                heatmap_table = heatmap_table[
                    [column for column in table_columns if column in heatmap_table]
                ].copy()
                heatmap_table.insert(0, "module_definition", module_set_label)
                filterable_dataframe(
                    heatmap_table,
                    table_key="distribution_feature_heatmap_table",
                    table_name="Feature differentiation heatmap source",
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download feature differentiation matrix source (TSV)",
                    data=dataframe_to_tsv_bytes(heatmap_table),
                    file_name=(
                        f"{download_prefix}{method}_{distribution_grouping_variable}_"
                        "feature_differentiation_heatmap.tsv"
                    ),
                    mime="text/tab-separated-values",
                )

    elif distribution_analysis == "Overall differentiation":
        st.caption(
            f"Kruskal–Wallis tests include selected levels with at least "
            f"{distribution_minimum_group_n} donors. FDR is BH-adjusted across valid "
            f"modules only within the selected {module_set_label} definition."
        )
        omnibus = distribution_omnibus_catalog()
        selected_omnibus = omnibus.loc[
            omnibus["module"].astype(int).eq(int(module))
        ].copy()
        if selected_omnibus.empty:
            st.info("Overall differentiation is unavailable for the selected data.")
        else:
            render_plotly_chart(
                distribution_omnibus_component_figure(
                    selected_omnibus,
                    module=module,
                    module_definition=module_set_label,
                ),
                use_container_width=True,
                config={"displaylogo": False},
                plot_id="main.distribution_omnibus_component_figure",
                metadata={
                    "module_set": module_set,
                    "method": method,
                    "module": f"M{module}",
                    "outcome": phenotype,
                    "feature": feature,
                },
            )
            selected_omnibus.insert(0, "module_definition", module_set_label)
            filterable_dataframe(
                selected_omnibus,
                table_key="distribution_omnibus_table",
                table_name="Overall distribution differentiation",
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download overall differentiation (TSV)",
                data=dataframe_to_tsv_bytes(selected_omnibus),
                file_name=(
                    f"{download_prefix}M{module}_{feature}_{method}_"
                    f"{distribution_grouping_variable}_omnibus.tsv"
                ),
                mime="text/tab-separated-values",
            )

    elif distribution_analysis == "Pairwise differentiation":
        contrasts, include_ks = distribution_pairwise_setup("distribution_pairwise")
        if not contrasts:
            st.info(
                f"At least two groups with {distribution_minimum_group_n} donors are "
                "required for pairwise analysis."
            )
        else:
            module_set_pairwise = module_set_pairwise_catalog(contrasts)
            selected_module_pairwise = module_set_pairwise.loc[
                module_set_pairwise["module"].astype(int).eq(int(module))
            ].copy()
            selected_intervals = cached_selected_distribution_pairwise(
                plot_data,
                distribution_grouping_variable,
                contrasts,
                distribution_minimum_group_n,
                include_ks,
            )
            interval_columns = [
                "module",
                "component",
                "reference_level_key",
                "comparison_level_key",
                "cliffs_delta_ci_low",
                "cliffs_delta_ci_high",
                "bootstrap_resamples",
            ]
            selected_module_pairwise = selected_module_pairwise.drop(
                columns=[
                    "cliffs_delta_ci_low",
                    "cliffs_delta_ci_high",
                    "bootstrap_resamples",
                ],
                errors="ignore",
            ).merge(
                selected_intervals[interval_columns],
                on=[
                    "module",
                    "component",
                    "reference_level_key",
                    "comparison_level_key",
                ],
                how="left",
                validate="one_to_one",
            )
            selected_module_pairwise = label_pairwise(selected_module_pairwise)
            if selected_module_pairwise.empty:
                st.info(
                    "Pairwise differentiation is unavailable for the selected data."
                )
            else:
                st.caption(
                    "Positive Cliff’s delta means the comparison group tends to have larger "
                    "scores than the reference. Mann–Whitney and optional KS BH corrections "
                    "are each calculated across modules separately for every fixed contrast "
                    "and component."
                )
                render_plotly_chart(
                    distribution_pairwise_forest_figure(
                        selected_module_pairwise,
                        module=module,
                        module_definition=module_set_label,
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                    plot_id="main.distribution_pairwise_forest_figure",
                    metadata={
                        "module_set": module_set,
                        "method": method,
                        "module": f"M{module}",
                        "outcome": phenotype,
                        "feature": feature,
                    },
                )
                render_plotly_chart(
                    distribution_pairwise_heatmap_figure(
                        selected_module_pairwise,
                        module=module,
                        module_definition=module_set_label,
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                    plot_id="main.distribution_pairwise_heatmap_figure",
                    metadata={
                        "module_set": module_set,
                        "method": method,
                        "module": f"M{module}",
                        "outcome": phenotype,
                        "feature": feature,
                    },
                )
                pairwise_columns = [
                    "module",
                    "metric_family",
                    "component",
                    "component_label",
                    "grouping_variable",
                    "reference_label",
                    "comparison_label",
                    "n_reference",
                    "n_comparison",
                    "eligible",
                    "unavailable_reason",
                    "median_reference",
                    "q1_reference",
                    "q3_reference",
                    "median_comparison",
                    "q1_comparison",
                    "q3_comparison",
                    "median_difference",
                    "mann_whitney_u",
                    "mann_whitney_p",
                    "mann_whitney_fdr_across_modules",
                    "mann_whitney_fdr_module_family_n",
                    "cliffs_delta",
                    "cliffs_delta_ci_low",
                    "cliffs_delta_ci_high",
                    "probability_superiority",
                ]
                if include_ks:
                    pairwise_columns.extend(
                        [
                            "ks_d",
                            "ks_p",
                            "ks_fdr_across_modules",
                            "ks_fdr_module_family_n",
                        ]
                    )
                pairwise_columns = [
                    column
                    for column in pairwise_columns
                    if column in selected_module_pairwise
                ]
                pairwise_table = selected_module_pairwise[pairwise_columns].copy()
                pairwise_table.insert(0, "module_definition", module_set_label)
                filterable_dataframe(
                    pairwise_table,
                    table_key="distribution_pairwise_table",
                    table_name="Pairwise distribution differentiation",
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download pairwise differentiation (TSV)",
                    data=dataframe_to_tsv_bytes(pairwise_table),
                    file_name=(
                        f"{download_prefix}M{module}_{feature}_{method}_"
                        f"{distribution_grouping_variable}_pairwise.tsv"
                    ),
                    mime="text/tab-separated-values",
                )

    else:
        rank_options = {
            "Omnibus epsilon-squared": "epsilon_squared",
            "Maximum absolute Cliff’s delta": "cliffs_delta",
        }
        include_ranking_ks = st.checkbox(
            "Enable KS shape-difference ranking",
            value=False,
            help="Adds maximum pairwise KS D as an advanced ranking option.",
        )
        if include_ranking_ks:
            rank_options["Maximum KS D"] = "ks_d"
        ranking_label = st.selectbox(
            "Rank modules by", options=list(rank_options), index=0
        )
        ranking_statistic = rank_options[ranking_label]
        top_n = st.selectbox("Top module components", [10, 20, 50], index=1)
        fdr_filter = st.radio(
            "Ranking significance filter",
            ["No FDR filter", "FDR < 0.05", "FDR < 0.10 (exploratory)"],
            horizontal=True,
        )
        ranking_contrasts: tuple[tuple[object, object], ...] = ()
        if ranking_statistic != "epsilon_squared":
            ranking_contrasts, _ = distribution_pairwise_setup(
                "distribution_ranking", show_ks_control=False
            )
        run_ranking = st.toggle(
            "Calculate across all modules",
            value=False,
            help=(
                "This streams one module at a time and caches the result. Leave disabled "
                "when only the selected-module distribution is needed."
            ),
        )
        if run_ranking:
            if ranking_statistic == "epsilon_squared":
                ranking_catalog = distribution_omnibus_catalog().copy()
                ranking_catalog["ranking_value"] = ranking_catalog["epsilon_squared"]
                ranking_catalog["ranking_abs"] = ranking_catalog["epsilon_squared"]
                ranking_catalog["ranking_p"] = ranking_catalog["categorical_p"]
                ranking_catalog["ranking_fdr"] = ranking_catalog[
                    "categorical_fdr_across_modules"
                ]
                ranking_catalog["ranking_family_n"] = ranking_catalog[
                    "categorical_fdr_module_family_n"
                ]
                ranking_summary = ranking_catalog.copy()
                signed_ranking = False
                ranking_axis_label = "Epsilon-squared"
            elif not ranking_contrasts:
                ranking_catalog = pd.DataFrame()
                ranking_summary = pd.DataFrame()
                signed_ranking = ranking_statistic == "cliffs_delta"
                ranking_axis_label = ranking_label
            else:
                ranking_catalog = label_pairwise(
                    module_set_pairwise_catalog(ranking_contrasts)
                )
                ranking_catalog = ranking_catalog.loc[
                    ranking_catalog[ranking_statistic].notna()
                ].copy()
                ranking_catalog["ranking_abs"] = ranking_catalog[
                    ranking_statistic
                ].abs()
                p_prefix = (
                    "mann_whitney" if ranking_statistic == "cliffs_delta" else "ks"
                )
                ranking_catalog["ranking_value"] = ranking_catalog[ranking_statistic]
                ranking_catalog["ranking_p"] = ranking_catalog[f"{p_prefix}_p"]
                ranking_catalog["ranking_fdr"] = ranking_catalog[
                    f"{p_prefix}_fdr_across_modules"
                ]
                ranking_catalog["ranking_family_n"] = ranking_catalog[
                    f"{p_prefix}_fdr_module_family_n"
                ]
                strongest_index = ranking_catalog.groupby(
                    ["module", "component"], observed=True, sort=False
                )["ranking_abs"].idxmax()
                ranking_summary = ranking_catalog.loc[strongest_index].copy()
                signed_ranking = ranking_statistic == "cliffs_delta"
                ranking_axis_label = (
                    "Cliff’s delta (strongest contrast)"
                    if signed_ranking
                    else "KS D (strongest contrast)"
                )
            if ranking_summary.empty:
                st.info("No module comparisons are available for the selected filters.")
            else:
                fdr_limit = {
                    "No FDR filter": None,
                    "FDR < 0.05": 0.05,
                    "FDR < 0.10 (exploratory)": 0.10,
                }[fdr_filter]
                if fdr_limit is not None:
                    ranking_summary = ranking_summary.loc[
                        ranking_summary["ranking_fdr"].lt(fdr_limit)
                    ].copy()
                ranking_summary = ranking_summary.sort_values(
                    "ranking_abs", ascending=False, na_position="last"
                )
                ranking_summary["ranking_label"] = (
                    "M"
                    + ranking_summary["module"].astype(int).astype(str)
                    + " · "
                    + ranking_summary["component_label"].astype(str)
                )
                ranking_summary["ranking_label"] += np.where(
                    ranking_summary["ranking_fdr"].lt(0.05), " *", ""
                )
                ranking_plot = ranking_summary.head(int(top_n)).copy()
                if ranking_plot.empty:
                    st.info("No modules meet the selected FDR filter.")
                else:
                    render_plotly_chart(
                        distribution_module_ranking_figure(
                            ranking_plot,
                            value_label=ranking_axis_label,
                            signed=signed_ranking,
                            module_definition=module_set_label,
                        ),
                        use_container_width=True,
                        config={"displaylogo": False},
                        plot_id="main.distribution_module_ranking_figure",
                        metadata={
                            "module_set": module_set,
                            "method": method,
                            "module": f"M{module}",
                            "outcome": phenotype,
                            "feature": feature,
                        },
                    )
                ranking_catalog.insert(0, "module_definition", module_set_label)
                filterable_dataframe(
                    ranking_catalog,
                    table_key="distribution_module_ranking_table",
                    table_name="Complete module differentiation ranking",
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download complete differentiation ranking (TSV)",
                    data=dataframe_to_tsv_bytes(ranking_catalog),
                    file_name=(
                        f"{download_prefix}{feature}_{method}_"
                        f"{distribution_grouping_variable}_{ranking_statistic}_ranking.tsv"
                    ),
                    mime="text/tab-separated-values",
                )


def render_mdc(context):
    download_prefix = context["download_prefix"]
    component_labels = context["component_labels"]
    module_set_label = context["module_set_label"]
    active_component_order = context["active_component_order"]
    active_tissues = context["active_tissues"]
    cohort_scope = context["cohort_scope"]
    data_manifest = context["data_manifest"]
    differential_edge_rule = context["differential_edge_rule"]
    differential_fdr_scope = context["differential_fdr_scope"]
    differential_fdr_threshold = context["differential_fdr_threshold"]
    estimator = context["estimator"]
    feature = context["feature"]
    method = context["method"]
    module = context["module"]
    module_count = context["module_count"]
    module_details = context["module_details"]
    module_manifest = context["module_manifest"]
    module_set = context["module_set"]
    phenotype = context["phenotype"]
    selected_details = context["selected_details"]
    if cohort_scope == VALIDATION_SCOPE:
        from app_helpers.camel_validation_ui import render_mdc as render_validation_mdc

        render_validation_mdc(
            module,
            differential_edge_rule,
            render_plotly_chart,
            module_set=module_set,
            network_method=method,
        )
        st.stop()
    st.subheader("Module differential connectivity (MDC)")
    if module_set in MUSCLE_MODULE_SETS and differential_edge_rule == "all":
        mdc_catalog = st.radio(
            "MDC result catalog",
            ["Existing MDC catalogs", "Diagnosis-label validation (steps 1, 2, 4)"],
            horizontal=True,
            key="muscle_mdc_result_catalog",
        )
        if mdc_catalog != "Existing MDC catalogs":
            from app_helpers.mdc_validation_ui import render_mdc_validation

            if module_set == "muscle_control_l4":
                render_mdc_validation(module, render_plotly_chart)
            else:
                render_mdc_validation(
                    module, render_plotly_chart, module_set=module_set
                )
            st.stop()
        st.caption(
            "Existing FDRs compare against disrupted-coexpression and random-gene nulls. "
            "They remain available for those hypotheses; use Diagnosis-label validation "
            "for the separate group-label tests and CT–TS comparison."
        )
    mdc_normalization_variant = "original"
    mdc_sensitivity_manifest = {}
    if module_set in MUSCLE_MODULE_SETS and differential_edge_rule == "all":
        mdc_sensitivity_manifest = load_mdc_normalization_manifest(module_set)
        available_variants = {
            name: entry
            for name, entry in mdc_sensitivity_manifest.get("variants", {}).items()
            if entry.get("observed_complete", False)
        }
        if available_variants:
            mdc_normalization_variant = st.selectbox(
                "MDC network weighting and normalization",
                options=[
                    *[
                        value
                        for value in ("m5_high_no_q90", "original")
                        if value in available_variants
                    ],
                    *[
                        value
                        for value in available_variants
                        if value not in ("m5_high_no_q90", "original")
                    ],
                ],
                format_func=lambda value: available_variants[value]["label"],
                key="muscle_mdc_normalization_variant_v2",
            )
            beta_text = (
                "TS VH=6, SMA=6, QUAD=10; CT VH–SMA=6, VH–QUAD=6, SMA–QUAD=7. "
                if module_set == "muscle_all_donors_l4"
                else "TS VH=5, SMA=7, QUAD=13; CT VH–SMA=8, VH–QUAD=10, SMA–QUAD=10. "
            )
            st.caption(
                "The same beta is used in AD and Control for every tissue block. "
                + "Current M5: "
                + beta_text
                + "Low-power variants: TS=3 and CT=2. "
                "Shared scale uses one Control-derived q90 per block for both diagnoses."
            )
            if mdc_normalization_variant == "m5_high_no_q90":
                st.caption(
                    "No-q90 MDC uses the unscaled M5 weights directly, without the block rescaling or its clipping step. Betas, bootstrap stability, and CT null-excess are unchanged. Pooled scores give more weight to stronger blocks; use the resolved views to inspect each tissue and pair."
                )
            if mdc_normalization_variant != "original":
                module_set_label += (
                    " · " + available_variants[mdc_normalization_variant]["label"]
                )
                download_prefix += mdc_normalization_variant + "_"
    mdc_data_revision = "|".join(
        mdc_sensitivity_manifest.get("files", {}).get(name, {}).get("sha256", "")
        for name in ("summary.parquet", "resolved.parquet")
    )
    if differential_edge_rule != "all" and not differential_mdc_data_available(
        module_set
    ):
        st.info(
            "AD–Control-filtered MDC is currently being generated for this module "
            "definition. Select **All edges** to use the existing MDC catalog now; "
            "the filtered option will activate automatically when its validated public "
            "tables are deployed."
        )
        st.stop()
    mdc_summary = cached_mdc_summary(
        module_set,
        estimator,
        method,
        differential_edge_rule,
        differential_fdr_scope,
        differential_fdr_threshold,
        mdc_normalization_variant,
        mdc_data_revision,
    )
    if mdc_summary.empty:
        st.error(
            "No MDC rows are available for the selected edge mask and network method."
        )
        st.stop()
    mdc_network_label = (
        "full Method-5 extended-SFN adjacency"
        if module_set in MUSCLE_MODULE_SETS
        else "signedAlt adjacency"
    )
    if mdc_normalization_variant != "original":
        mdc_network_label = available_variants[mdc_normalization_variant]["label"]
    st.caption(
        f"MDC compares the mean absolute {mdc_network_label} in AD with Control. "
        "Values above 1 indicate higher connectivity in AD; values below 1 indicate "
        "higher connectivity in Control. Total uses all edges, TS uses same-tissue edges, "
        "and CT uses cross-tissue edges."
    )
    mdc_metadata = (
        data_manifest.get("mdc_differential_edges", {})
        if differential_edge_rule != "all"
        else module_manifest.get("mdc", data_manifest.get("mdc", {}))
    )
    if mdc_normalization_variant != "original":
        mdc_metadata = {
            **mdc_sensitivity_manifest,
            **available_variants[mdc_normalization_variant],
        }
    estimator_note = (
        "All-edge MDC is module-level context and does not change with the estimator "
        "or network-method selector."
        if differential_edge_rule == "all"
        else (
            f"This view applies the {ESTIMATOR_LABELS[estimator]} / "
            f"{readable_method(method)} discovery mask selected above. The mask is held "
            "fixed during MDC calculation; BONOBO donor-native edge significance is not "
            "an additional group-MDC filter."
        )
    )
    if module_set in MUSCLE_MODULE_SETS:
        discovery_context = (
            "Modules were discovered in the all-donor cohort"
            if module_set == "muscle_all_donors_l4"
            else "Modules were discovered in Controls"
        )
        st.warning(
            "The muscle MDC uses the 155 AD and 120 Control donors with complete VH, "
            "SMA, and QUAD expression. The weighting selector applies to every MDC "
            "view, table, and download below. M5 variants retain bootstrap stability "
            "and CT null-excess calibration; Pearson and SignedAlt are separate "
            "sensitivity baselines. Current M5 retains the full M5 normalization. "
            + discovery_context
            + ", so these "
            "comparisons remain conditional on that module definition. MCI was not included. "
            + estimator_note
        )
        if not mdc_metadata.get("permutation_significance_complete", False):
            st.info(
                "Observed MDC is available for this sensitivity, while its own "
                "permutation inference is pending. Existing M5 FDRs are not reused. "
                f"Completed nulls: sample {mdc_metadata.get('completed_sample_permutations', 0)}/200; "
                f"gene {mdc_metadata.get('completed_gene_permutations', 0)}/200. "
                "Counts refer to this published snapshot. NA FDR means not evaluated."
            )
        if mdc_sensitivity_manifest and st.toggle(
            "Compare MDC distributions across weighting variants",
            value=False,
            key="muscle_mdc_compare_normalizations",
        ):
            from app_helpers.charts import mdc_normalization_comparison_figure

            comparison_scope = st.radio(
                "Comparison component",
                options=["ct", "ts", "total"],
                format_func=lambda value: {
                    "ct": "CT pooled",
                    "ts": "TS pooled",
                    "total": "All edges",
                }[value],
                horizontal=True,
            )
            comparison_data = load_mdc_normalization_summary(module_set)
            render_plotly_chart(
                mdc_normalization_comparison_figure(
                    comparison_data, comparison_scope, module
                ),
                use_container_width=True,
                plot_id="main.mdc_normalization_comparison",
            )
            st.caption(
                "Boxes describe the distribution across modules with at least ten edges. Stars mark the selected module. This comparison does not supply a test across weighting variants."
            )
    else:
        st.warning(
            "Cohort scope differs from the donor-complete LIONESS analysis. The MDC source "
            f"assembled {mdc_metadata.get('reference_assembled_donors', 517)} AD and "
            f"{mdc_metadata.get('target_assembled_donors', 408)} Control donors across the "
            "tissue union, including the 167 AD and 164 Control complete-three-tissue donors "
            "used here plus donors with partial tissue availability. MCI was not included. "
            + estimator_note
        )
    if differential_edge_rule != "all":
        st.warning(
            "Filtered MDC is exploratory post-selection context. The AD–Control discovery "
            "mask is fixed in the 200 sample and 200 tissue-count-preserving gene "
            "permutations, so its directional FDR does not account for adaptive edge "
            "selection and is not independent validation. Modules with no retained edge "
            "in a scope remain unavailable rather than being assigned MDC=0."
        )
    mdc_control_columns = st.columns(2)
    with mdc_control_columns[0]:
        mdc_threshold = st.radio(
            "MDC significance threshold",
            options=[0.05, 0.10],
            format_func=lambda value: f"FDR < {value:.2f}",
            horizontal=True,
            key="mdc_threshold",
        )
    with mdc_control_columns[1]:
        mdc_scale = st.radio(
            "MDC display scale",
            options=["log2", "raw"],
            format_func=lambda value: {
                "log2": "log₂ ratio (equality = 0)",
                "raw": "Raw AD/Control ratio (equality = 1)",
            }[value],
            horizontal=True,
            key="mdc_display_scale",
        )
    if mdc_scale == "raw":
        st.caption(
            "Raw heatmap cells and hovers show MDC ratios. Heatmap colors are "
            "log-symmetric around MDC=1, so reciprocal changes such as 0.5 and 2 "
            "have equal color intensity in opposite directions."
        )
    mdc_inference_complete = bool(
        mdc_metadata.get("permutation_significance_complete", False)
    )

    mdc_view = mdc_summary.copy()
    if "module_definition" in mdc_view:
        mdc_view.pop("module_definition")
    mdc_view.insert(0, "module_definition", module_set_label)
    mdc_view = mdc_view.merge(
        module_details[
            [
                "module",
                "module_size",
                "cluster_type",
                "tissue_entropy",
                "tissue_entropy_normalized",
            ]
        ],
        on="module",
        how="left",
        validate="one_to_one",
    )
    if mdc_view["tissue_entropy_normalized"].isna().any():
        st.warning(
            "Some MDC modules lack module-composition entropy and will be omitted from "
            "the entropy relationship plots."
        )
    for scope in ["total", "ts", "ct"]:
        mdc_view[f"significant_{scope}"] = mdc_view[f"directional_fdr_{scope}"].lt(
            mdc_threshold
        )
    mdc_view["any_significant"] = mdc_view[
        ["significant_total", "significant_ts", "significant_ct"]
    ].any(axis=1)
    if differential_edge_rule != "all":
        available_counts = {
            scope.upper(): int(mdc_view[f"mdc_{scope}"].notna().sum())
            for scope in ("total", "ts", "ct")
        }
        st.caption(
            "Modules with at least one retained edge and an available MDC ratio: "
            + " · ".join(
                f"{scope}={count}/{module_count}"
                for scope, count in available_counts.items()
            )
        )

    count_columns = st.columns(4)
    count_columns[0].metric(
        "Total significant",
        (
            int(mdc_view["significant_total"].sum())
            if mdc_inference_complete
            else "Pending"
        ),
    )
    count_columns[1].metric(
        "TS significant",
        int(mdc_view["significant_ts"].sum()) if mdc_inference_complete else "Pending",
    )
    count_columns[2].metric(
        "CT significant",
        int(mdc_view["significant_ct"].sum()) if mdc_inference_complete else "Pending",
    )
    count_columns[3].metric(
        "Any MDC significant",
        int(mdc_view["any_significant"].sum()) if mdc_inference_complete else "Pending",
    )

    selected_mdc = mdc_view.loc[mdc_view["module"].astype(int).eq(int(module))]
    if selected_mdc.empty:
        st.error(f"No MDC row is available for {module_label(module)}.")
    else:
        selected_mdc_row = selected_mdc.iloc[0]
        st.markdown(f"#### Selected module: {module_label(module)}")
        if differential_edge_rule != "all":
            st.caption(
                "Retained discovery edges: "
                f"total={int(selected_mdc_row['n_retained_edges_total']):,}/"
                f"{int(selected_mdc_row['n_possible_edges_total']):,}, "
                f"TS={int(selected_mdc_row['n_retained_edges_ts']):,}/"
                f"{int(selected_mdc_row['n_possible_edges_ts']):,}, "
                f"CT={int(selected_mdc_row['n_retained_edges_ct']):,}/"
                f"{int(selected_mdc_row['n_possible_edges_ct']):,}."
            )
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
                "Not evaluated"
                if pd.isna(directional_fdr)
                else (
                    "Significant"
                    if directional_fdr < mdc_threshold
                    else "Not significant"
                )
            )
            if pd.isna(ratio):
                significance_text = "Not available"
            with container:
                st.metric(label, ratio_text)
                st.caption(
                    f"{direction} · directional FDR={fdr_value_text} · {significance_text}"
                )
                inference_column = f"inference_available_{scope}"
                reason_column = f"inference_unavailable_reason_{scope}"
                if (
                    mdc_inference_complete
                    and inference_column in selected_mdc_row.index
                    and not bool(selected_mdc_row[inference_column])
                ):
                    reason = str(selected_mdc_row.get(reason_column, "")).strip()
                    st.warning(
                        "Descriptive MDC only" + (f": {reason}." if reason else ".")
                    )

        selected_mdc_chart = mdc_module_figure(
            selected_mdc_row,
            mdc_threshold,
            module_definition=module_set_label,
            scale=mdc_scale,
        )
        render_plotly_chart(
            selected_mdc_chart,
            use_container_width=True,
            config={
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": f"{download_prefix}M{module}_AD_Control_MDC",
                    "scale": 3,
                },
            },
            plot_id="main.selected_mdc_chart",
            metadata={
                "module_set": module_set,
                "method": method,
                "module": f"M{module}",
                "outcome": phenotype,
                "feature": feature,
            },
        )
        st.caption(
            (
                "Bars use log2 MDC, so equal AD/Control connectivity is centered at zero. "
                if mdc_scale == "log2"
                else "Bars use the raw AD/Control MDC ratio, so equal connectivity is marked at one. "
            )
            + "Orange indicates higher connectivity in AD, blue indicates higher connectivity "
            "in Control, and ★ marks significance at the selected threshold."
        )

        st.markdown("##### AD and Control module adjacency")
        st.caption(
            "The upper triangle shows the AD group network and the lower triangle "
            "shows the Control group network. Both triangles use the same absolute "
            f"{mdc_network_label} color scale and the AD-derived node ordering."
        )
        if mdc_normalization_variant not in ("original", "m5_high_no_q90"):
            st.info(
                "The existing split-triangle adjacency image represents Current M5. Select that weighting to view its matching network image."
            )
        elif differential_edge_rule != "all":
            st.info(
                "The split-triangle adjacency image is currently available for the "
                "all-edge MDC network only. Select **All edges** to view it."
            )
        elif mdc_split_heatmap_available(
            module_set, module, differential_edge_rule, mdc_normalization_variant
        ):
            show_mdc_adjacency = st.toggle(
                "Load the selected module adjacency heatmap",
                value=False,
                key=f"show_mdc_split_heatmap_{module_set}_{module}",
                help=(
                    "Images are fetched one module at a time from the Drive backend to "
                    "keep the Streamlit app within its memory limit."
                ),
            )
            if show_mdc_adjacency:
                with st.spinner("Loading the selected AD/Control adjacency heatmap…"):
                    mdc_heatmap_path = load_mdc_split_heatmap(
                        module_set,
                        module,
                        differential_edge_rule,
                        mdc_normalization_variant,
                    )
                try:
                    from app_helpers.mdc_adjacency_ui import (
                        render_configurable_mdc_heatmap,
                    )

                    render_configurable_mdc_heatmap(
                        mdc_heatmap_path,
                        int(module),
                        module_set,
                        module_set_label,
                        mdc_normalization_variant,
                        {
                            tissue: int(
                                selected_details.get(f"n_genes_{tissue.lower()}", 0)
                            )
                            for tissue in active_tissues
                        },
                    )
                except (ValueError, OSError, KeyError) as error:
                    st.warning(
                        "The configurable frame is unavailable for this image. "
                        "Showing the original heatmap instead. " + str(error)
                    )
                    render_streamlit_image(
                        str(mdc_heatmap_path),
                        caption=(
                            f"{module_label(module)}: upper triangle = AD; "
                            "lower triangle = Control. Yellow (low) → orange → red (high)."
                        ),
                    )
                st.download_button(
                    "Download original adjacency heatmap (PNG)",
                    data=mdc_heatmap_path.read_bytes(),
                    file_name=(
                        f"{download_prefix}M{module}_AD_Control_"
                        "split_triangle_adjacency.png"
                    ),
                    mime="image/png",
                )
        else:
            st.info(
                "The split-triangle adjacency image has not yet been published for "
                f"{module_label(module)} in this module definition."
            )

    significance_options = ["All modules"]
    if mdc_inference_complete:
        significance_options.extend(
            [
                "Any significant",
                "Total significant",
                "TS significant",
                "CT significant",
                "Evaluated, no significant MDC",
                "Inference unavailable",
            ]
        )
    significance_filter = st.selectbox(
        "Modules to show in the MDC overview and entropy plots",
        options=significance_options,
    )
    aggregate_inference_columns = [
        column
        for column in (
            "inference_available_total",
            "inference_available_ts",
            "inference_available_ct",
        )
        if column in mdc_view.columns
    ]
    any_inference_available = (
        mdc_view[aggregate_inference_columns].any(axis=1)
        if aggregate_inference_columns
        else pd.Series(False, index=mdc_view.index)
    )
    filter_masks = {
        "All modules": pd.Series(True, index=mdc_view.index),
        "Any significant": mdc_view["any_significant"],
        "Total significant": mdc_view["significant_total"],
        "TS significant": mdc_view["significant_ts"],
        "CT significant": mdc_view["significant_ct"],
        "Evaluated, no significant MDC": (
            any_inference_available & ~mdc_view["any_significant"]
        ),
        "Inference unavailable": ~any_inference_available,
    }
    displayed_mdc = mdc_view.loc[filter_masks[significance_filter]].copy()
    mdc_overview = mdc_overview_figure(
        displayed_mdc,
        module,
        mdc_threshold,
        module_definition=module_set_label,
        scale=mdc_scale,
    )
    render_plotly_chart(
        mdc_overview,
        use_container_width=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"{download_prefix}AD_Control_TS_CT_MDC_overview",
                "scale": 3,
            },
        },
        plot_id="main.mdc_overview",
        metadata={
            "module_set": module_set,
            "method": method,
            "module": f"M{module}",
            "outcome": phenotype,
            "feature": feature,
        },
    )
    if differential_edge_rule == "all":
        ct_unavailable_count = len(
            module_manifest.get(
                "ct_unavailable_modules",
                mdc_summary.loc[
                    pd.to_numeric(mdc_summary["n_ct_edges"], errors="coerce")
                    .fillna(0)
                    .eq(0),
                    "module",
                ]
                .astype(int)
                .tolist(),
            )
        )
        unavailable_reason = "have no structural CT edges"
    else:
        ct_unavailable_count = int(
            pd.to_numeric(mdc_summary["n_ct_edges"], errors="coerce")
            .fillna(0)
            .eq(0)
            .sum()
        )
        unavailable_reason = "retain no CT edge under the selected discovery mask"
    st.caption(
        (
            "Dashed zero lines separate AD-higher from Control-higher MDC. "
            if mdc_scale == "log2"
            else "Dashed equality lines at one separate AD-higher from Control-higher MDC. "
        )
        + "The dotted diagonal "
        f"marks equal TS and CT effects. {ct_unavailable_count} module(s) {unavailable_reason} "
        "and therefore no CT MDC point."
    )

    st.markdown("#### MDC and tissue-mixing entropy")
    st.caption(
        "Each point is one module after applying the module filter above. Diamonds are "
        f"directionally significant at FDR < {mdc_threshold:.2f}; open circles have no "
        "significant result or have unavailable inference. "
        "The dotted line is an OLS visual guide, while the subtitle reports the Spearman "
        "association across the displayed modules."
    )
    entropy_chart_columns = st.columns(2)
    for chart_column, entropy_scope in zip(
        entropy_chart_columns, ["ts", "ct"], strict=True
    ):
        with chart_column:
            render_plotly_chart(
                mdc_entropy_figure(
                    displayed_mdc,
                    scope=entropy_scope,
                    selected_module=module,
                    threshold=mdc_threshold,
                    scale=mdc_scale,
                    module_definition=module_set_label,
                ),
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": (
                            f"{download_prefix}{entropy_scope.upper()}_MDC_"
                            f"normalized_Shannon_entropy_{mdc_scale}"
                        ),
                        "scale": 3,
                    },
                },
                plot_id=f"main.mdc_entropy_figure.{entropy_scope}",
                metadata={
                    "module_set": module_set,
                    "method": method,
                    "module": f"M{module}",
                    "outcome": phenotype,
                    "feature": feature,
                    "component": entropy_scope,
                },
            )

    displayed_mdc["minimum_directional_fdr"] = displayed_mdc[
        ["directional_fdr_total", "directional_fdr_ts", "directional_fdr_ct"]
    ].min(axis=1, skipna=True)
    displayed_mdc["maximum_abs_log2_mdc"] = (
        displayed_mdc[["log2_mdc_total", "log2_mdc_ts", "log2_mdc_ct"]]
        .abs()
        .max(axis=1, skipna=True)
    )
    displayed_mdc = displayed_mdc.sort_values(
        ["minimum_directional_fdr", "maximum_abs_log2_mdc"],
        ascending=[True, False],
        na_position="last",
    )
    mdc_columns = [
        "module_definition",
        "module",
        "module_size_mapped",
        "module_size",
        "cluster_type",
        "tissue_entropy",
        "tissue_entropy_normalized",
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
    for scope in ("total", "ts", "ct"):
        mdc_columns.extend(
            column
            for column in (
                f"mean_abs_ad_{scope}",
                f"mean_abs_control_{scope}",
                f"inference_available_{scope}",
                f"inference_unavailable_reason_{scope}",
                f"mdc_stable_for_inference_{scope}",
                f"mdc_stability_reason_{scope}",
            )
            if column in displayed_mdc.columns
        )
    if differential_edge_rule != "all":
        mdc_columns.extend(
            [
                "estimator",
                "network_method",
                "differential_fdr_scope",
                "differential_fdr_threshold",
                "n_possible_edges_total",
                "n_possible_edges_ts",
                "n_possible_edges_ct",
                "n_retained_edges_total",
                "n_retained_edges_ts",
                "n_retained_edges_ct",
                "retained_edge_fraction_total",
                "retained_edge_fraction_ts",
                "retained_edge_fraction_ct",
                "mean_abs_ad_total",
                "mean_abs_control_total",
                "mean_abs_ad_ts",
                "mean_abs_control_ts",
                "mean_abs_ad_ct",
                "mean_abs_control_ct",
                "inference_scope",
            ]
        )
    filterable_dataframe(
        displayed_mdc[mdc_columns],
        table_key="mdc_summary",
        table_name="MDC summary",
        use_container_width=True,
        hide_index=True,
        column_config={
            "module": st.column_config.NumberColumn(format="M%d"),
            "module_size": st.column_config.NumberColumn(format="%d"),
            "tissue_entropy": st.column_config.NumberColumn(format="%.4f"),
            "tissue_entropy_normalized": st.column_config.ProgressColumn(
                format="%.3f", min_value=0.0, max_value=1.0
            ),
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
        file_name=(
            f"{download_prefix}AD_Control_MDC_"
            f"{significance_filter.replace(' ', '_')}.tsv"
        ),
        mime="text/tab-separated-values",
    )
    if module_set in MUSCLE_MODULE_SETS and not mdc_metadata.get(
        "permutation_significance_complete", False
    ):
        st.info(
            "These rows contain observed MDC ratios for the selected weighting. Directional permutation p-values "
            "and FDRs have not been evaluated for this functional, so the FDR fields remain "
            "NA and no significance claim is made."
        )
    else:
        st.info(
            "For each scope and direction, sample-permutation and gene-permutation p-values were "
            f"Benjamini–Hochberg adjusted separately across the {module_count} modules. The "
            "directional FDR is the larger of those two adjusted values for the observed MDC "
            "direction, making significance require support from both null models. The latest "
            "source used 200 sample permutations and 200 gene permutations."
        )

    region_mdc_tab, pathway_mdc_tab = stateful_tabs(
        ["Region-resolved MDC", "Pathway-resolved MDC"],
        key="mdc_resolution_tabs",
    )
    with region_mdc_tab:
        if tab_is_open(region_mdc_tab):
            st.markdown("#### Tissue-resolved MDC")
            st.caption(
                "Resolved MDC separates the three within-tissue blocks and the three cross-tissue "
                "pairs. Gene permutations, when available for the selected module definition, "
                "preserve each module's "
                + ", ".join(active_tissues)
                + " feature counts. For each component and direction, BH correction uses only "
                "modules where that edge block structurally exists; unavailable or unevaluated "
                "blocks remain missing."
            )
            resolved_mdc = cached_mdc_resolved(
                module_set,
                estimator,
                method,
                differential_edge_rule,
                differential_fdr_scope,
                differential_fdr_threshold,
                mdc_normalization_variant,
                mdc_data_revision,
            ).copy()
            resolved_components_available = resolved_mdc[
                ["component", "component_label"]
            ].drop_duplicates()
            selected_mdc_components = st.multiselect(
                "Resolved MDC components",
                options=resolved_components_available["component"].tolist(),
                default=resolved_components_available["component"].tolist(),
                format_func=lambda value: resolved_components_available.loc[
                    resolved_components_available["component"].eq(value),
                    "component_label",
                ].iloc[0],
                key=f"resolved_mdc_components_{module_set}",
            )
            resolved_mdc = resolved_mdc.loc[
                resolved_mdc["component"].isin(selected_mdc_components)
            ].copy()
            resolved_significance = st.selectbox(
                "Resolved MDC rows",
                options=[
                    "All",
                    "FDR significant",
                    "Evaluated, not FDR significant",
                    "FDR unavailable",
                ],
                key=f"resolved_mdc_significance_{module_set}",
            )
            resolved_fdr = pd.to_numeric(
                resolved_mdc["directional_fdr"], errors="coerce"
            )
            resolved_is_significant = resolved_fdr.lt(mdc_threshold)
            if resolved_significance == "FDR significant":
                resolved_mdc = resolved_mdc.loc[resolved_is_significant]
            elif resolved_significance == "Evaluated, not FDR significant":
                resolved_mdc = resolved_mdc.loc[
                    resolved_fdr.notna() & ~resolved_is_significant
                ]
            elif resolved_significance == "FDR unavailable":
                resolved_mdc = resolved_mdc.loc[resolved_fdr.isna()]
            if mdc_inference_complete and "mdc_stable_for_inference" in resolved_mdc:
                unstable_count = int(
                    (~resolved_mdc["mdc_stable_for_inference"].fillna(False)).sum()
                )
                show_descriptive_only = st.checkbox(
                    "Include descriptive-only unstable components",
                    value=False,
                    key=f"resolved_mdc_unstable_{module_set}",
                    help=(
                        "These components have fewer than 10 edges or a near-zero AD/Control "
                        "mean absolute weight. Their raw ratios may be extreme, so no FDR is "
                        "reported."
                    ),
                )
                if unstable_count:
                    st.caption(
                        f"{unstable_count:,} displayed component rows are descriptive-only "
                        "before applying this stability filter."
                    )
                if not show_descriptive_only:
                    resolved_mdc = resolved_mdc.loc[
                        resolved_mdc["mdc_stable_for_inference"].fillna(False)
                    ]
            selected_resolved_mdc = resolved_mdc.loc[
                resolved_mdc["module"].astype(int).eq(int(module))
            ]
            if selected_resolved_mdc.empty:
                st.info(
                    "The selected module has no resolved MDC row under the current filters."
                )
            else:
                render_plotly_chart(
                    mdc_resolved_module_figure(
                        selected_resolved_mdc,
                        mdc_threshold,
                        module_definition=module_set_label,
                        scale=mdc_scale,
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                    plot_id="main.mdc_resolved_module_figure",
                    metadata={
                        "module_set": module_set,
                        "method": method,
                        "module": f"M{module}",
                        "outcome": phenotype,
                        "feature": feature,
                    },
                )
            if resolved_mdc.empty:
                st.info("No resolved MDC rows remain under the current filters.")
            else:
                render_plotly_chart(
                    mdc_resolved_heatmap_figure(
                        resolved_mdc,
                        mdc_threshold,
                        selected_module=module,
                        module_definition=module_set_label,
                        scale=mdc_scale,
                    ),
                    use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True},
                    plot_id="main.mdc_resolved_heatmap_figure",
                    metadata={
                        "module_set": module_set,
                        "method": method,
                        "module": f"M{module}",
                        "outcome": phenotype,
                        "feature": feature,
                    },
                )
                resolved_mdc.insert(0, "module_definition_label", module_set_label)
                filterable_dataframe(
                    resolved_mdc,
                    table_key="resolved_mdc",
                    table_name="Resolved MDC",
                    use_container_width=True,
                    hide_index=True,
                    height=520,
                )
                st.download_button(
                    "Download displayed resolved MDC rows (TSV)",
                    data=dataframe_to_tsv_bytes(resolved_mdc),
                    file_name=f"{download_prefix}AD_Control_resolved_MDC.tsv",
                    mime="text/tab-separated-values",
                )

    with pathway_mdc_tab:
        if tab_is_open(pathway_mdc_tab):
            st.markdown("#### Pathway-annotated regional MDC")
            st.caption(
                "MDC remains a module-level statistic. This view annotates each module-component "
                "MDC with KEGG pathways enriched in the matching region, then summarizes MDC "
                "across enriched modules. For a CT tissue pair, both regions must meet the KEGG "
                "threshold; the displayed pair FDR is the larger regional FDR and is not a new "
                "combined p-value."
            )
            pathway_control_columns = st.columns(4)
            with pathway_control_columns[0]:
                pathway_mdc_resolution = st.selectbox(
                    "MDC enrichment resolution",
                    options=list(MDC_ENRICHMENT_RESOLUTION_LABELS),
                    format_func=MDC_ENRICHMENT_RESOLUTION_LABELS.get,
                    key=f"pathway_mdc_resolution_{module_set}",
                )
            with pathway_control_columns[1]:
                pathway_kegg_threshold = st.radio(
                    "KEGG enrichment threshold",
                    options=[0.05, 0.10],
                    format_func=lambda value: (
                        f"FDR < {value:.2f}"
                        + (" (exploratory)" if np.isclose(value, 0.10) else "")
                    ),
                    horizontal=True,
                    key=f"pathway_mdc_kegg_threshold_{module_set}",
                )
            with pathway_control_columns[2]:
                minimum_pathway_modules = st.slider(
                    "Minimum enriched modules per cell",
                    min_value=1,
                    max_value=10,
                    value=1,
                    key=f"pathway_mdc_minimum_modules_{module_set}",
                )
            with pathway_control_columns[3]:
                pathway_mdc_row_scope = st.radio(
                    "Module MDC rows",
                    options=["All", "MDC FDR-significant only"],
                    horizontal=True,
                    key=f"pathway_mdc_row_scope_{module_set}",
                )

            pathway_rows = cached_pathway_mdc_rows(
                module_set,
                pathway_kegg_threshold,
                estimator,
                method,
                differential_edge_rule,
                differential_fdr_scope,
                differential_fdr_threshold,
                mdc_normalization_variant,
                mdc_data_revision,
            ).copy()
            component_preference = [
                *[value for value in active_component_order if value.startswith("TS_")],
                *[value for value in active_component_order if value.startswith("CT_")],
                "total",
                "TS",
                "CT",
            ]
            component_labels = (
                pathway_rows[["component", "component_label"]]
                .drop_duplicates("component")
                .set_index("component")["component_label"]
                .to_dict()
            )
            available_pathway_components = [
                value for value in component_preference if value in component_labels
            ]
            default_pathway_components = available_pathway_components[:6]
            selected_pathway_components = st.multiselect(
                "Regions and tissue pairs",
                options=available_pathway_components,
                default=default_pathway_components,
                format_func=lambda value: component_labels[value],
                key=f"pathway_mdc_components_{module_set}",
            )
            pathway_rows = pathway_rows.loc[
                pathway_rows["component"].isin(selected_pathway_components)
            ].copy()
            if pathway_mdc_row_scope == "MDC FDR-significant only":
                pathway_rows = pathway_rows.loc[
                    pd.to_numeric(pathway_rows["directional_fdr"], errors="coerce").lt(
                        mdc_threshold
                    )
                ].copy()

            category_options = sorted(
                pathway_rows["category_level1"].dropna().astype(str).unique().tolist()
            )
            filter_columns = st.columns([2, 3])
            selected_pathway_categories = filter_columns[0].multiselect(
                "KEGG categories",
                options=category_options,
                default=[],
                placeholder="All categories",
                key=f"pathway_mdc_categories_{module_set}",
            )
            pathway_search = (
                filter_columns[1]
                .text_input(
                    "Search pathways",
                    placeholder="Pathway, category, sub-category, or KEGG ID",
                    key=f"pathway_mdc_search_{module_set}",
                )
                .strip()
            )
            if selected_pathway_categories:
                pathway_rows = pathway_rows.loc[
                    pathway_rows["category_level1"].isin(selected_pathway_categories)
                ].copy()
            if pathway_search:
                searchable = (
                    pathway_rows[
                        [
                            "pathway_id",
                            "pathway_label",
                            "category_level1",
                            "category_level2",
                        ]
                    ]
                    .fillna("")
                    .astype(str)
                )
                pathway_rows = pathway_rows.loc[
                    searchable.agg(" ".join, axis=1).str.contains(
                        pathway_search, case=False, regex=False
                    )
                ].copy()

            pathway_group_rows = collapse_pathway_mdc_rows(
                pathway_rows,
                resolution=pathway_mdc_resolution,
            )
            pathway_summary = summarize_pathway_mdc_rows(
                pathway_group_rows,
                mdc_fdr_threshold=mdc_threshold,
                minimum_modules=minimum_pathway_modules,
                resolution=pathway_mdc_resolution,
            )
            if pathway_summary.empty:
                st.info("No KEGG-annotated MDC cells remain under the current filters.")
            else:
                retained_pathway_ids = set(pathway_summary["pathway_id"].astype(str))
                pathway_group_rows = pathway_group_rows.loc[
                    pathway_group_rows["pathway_id"]
                    .astype(str)
                    .isin(retained_pathway_ids)
                ].copy()
                pathway_ranking = (
                    pathway_summary.assign(
                        absolute_mean_log2_mdc=pathway_summary["mean_log2_mdc"].abs()
                    )
                    .groupby(["pathway_id", "pathway_label"], observed=True)
                    .agg(
                        maximum_absolute_mean_log2_mdc=(
                            "absolute_mean_log2_mdc",
                            "max",
                        ),
                        minimum_enrichment_fdr=("minimum_enrichment_fdr", "min"),
                        maximum_module_support=("n_modules", "max"),
                    )
                    .reset_index()
                    .sort_values(
                        [
                            "maximum_absolute_mean_log2_mdc",
                            "minimum_enrichment_fdr",
                            "maximum_module_support",
                        ],
                        ascending=[False, True, False],
                        kind="stable",
                    )
                )
                pathway_metric_columns = st.columns(4)
                resolution_singular = MDC_ENRICHMENT_RESOLUTION_LABELS[
                    pathway_mdc_resolution
                ]
                resolution_plural = {
                    "pathway": "Pathways",
                    "subcategory": "KEGG sub-categories",
                    "category": "KEGG categories",
                }[pathway_mdc_resolution]
                pathway_metric_columns[0].metric(
                    resolution_plural,
                    f"{pathway_summary['pathway_id'].nunique():,}",
                )
                pathway_metric_columns[1].metric(
                    "Modules", f"{pathway_group_rows['module'].nunique():,}"
                )
                pathway_metric_columns[2].metric(
                    f"Region/{resolution_singular.lower()} cells",
                    f"{len(pathway_summary):,}",
                )
                pathway_metric_columns[3].metric(
                    "Module-component annotations", f"{len(pathway_group_rows):,}"
                )
                pathway_options = pathway_ranking["pathway_id"].astype(str).tolist()
                pathway_label_map = pathway_ranking.set_index("pathway_id")[
                    "pathway_label"
                ].to_dict()
                pathway_selection_columns = st.columns([2, 1])
                selected_pathway_id = pathway_selection_columns[0].selectbox(
                    f"{resolution_singular} detail",
                    options=pathway_options,
                    format_func=lambda value: (
                        f"{pathway_label_map[value]} ({value})"
                        if pathway_mdc_resolution == "pathway"
                        else pathway_label_map[value]
                    ),
                    key=(
                        f"pathway_mdc_selected_{module_set}_"
                        f"{pathway_mdc_resolution}"
                    ),
                )
                maximum_top_pathways = min(60, len(pathway_options))
                top_pathways = pathway_selection_columns[1].slider(
                    f"{resolution_plural} in heatmap",
                    min_value=1,
                    max_value=maximum_top_pathways,
                    value=min(25, maximum_top_pathways),
                    key=(
                        f"pathway_mdc_top_n_{module_set}_" f"{pathway_mdc_resolution}"
                    ),
                )
                render_plotly_chart(
                    pathway_mdc_heatmap_figure(
                        pathway_summary,
                        scale=mdc_scale,
                        top_n=top_pathways,
                        selected_pathway_id=selected_pathway_id,
                        module_definition=module_set_label,
                        resolution=pathway_mdc_resolution,
                    ),
                    use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True},
                    plot_id="main.pathway_mdc_heatmap_figure",
                    metadata={
                        "module_set": module_set,
                        "method": method,
                        "module": f"M{module}",
                        "outcome": phenotype,
                        "feature": feature,
                    },
                )
                st.caption(
                    "Each heatmap cell gives equal weight to every qualifying enriched module. "
                    "At category and sub-category resolution, a module with several supporting "
                    "pathways is counted once and uses its smallest component-matched KEGG FDR. "
                    "On log2 scale it is the arithmetic mean log2 MDC; on raw scale it is the "
                    "equivalent geometric mean MDC ratio. Cell n is the number of enriched modules; "
                    "hover also reports the number of distinct supporting pathways."
                )
                selected_pathway_rows = pathway_group_rows.loc[
                    pathway_group_rows["pathway_id"].astype(str).eq(selected_pathway_id)
                ].copy()
                render_plotly_chart(
                    pathway_mdc_detail_figure(
                        selected_pathway_rows,
                        pathway_id=selected_pathway_id,
                        selected_module=module,
                        threshold=mdc_threshold,
                        scale=mdc_scale,
                        module_definition=module_set_label,
                        resolution=pathway_mdc_resolution,
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                    plot_id="main.pathway_mdc_detail_figure",
                    metadata={
                        "module_set": module_set,
                        "method": method,
                        "module": f"M{module}",
                        "outcome": phenotype,
                        "feature": feature,
                    },
                )
                pathway_detail_columns = [
                    "enrichment_resolution_label",
                    "pathway_label",
                    "category_level1",
                    "category_level2",
                    "supporting_pathway_count",
                    "supporting_pathway_ids",
                    "supporting_pathway_names",
                    "supporting_subcategories",
                    "best_supporting_pathway_id",
                    "best_supporting_pathway_name",
                    "module",
                    "component_label",
                    "enrichment_scope",
                    "enrichment_p",
                    "enrichment_fdr",
                    "enrichment_region_a",
                    "enrichment_fdr_region_a",
                    "enrichment_region_b",
                    "enrichment_fdr_region_b",
                    "mdc",
                    "log2_mdc",
                    "direction",
                    "directional_fdr",
                    "n_edges",
                ]
                if pathway_mdc_resolution == "pathway":
                    pathway_detail_columns[1:1] = ["pathway_id"]
                    pathway_detail_columns.extend(["overlap_k", "overlap_genes"])
                selected_pathway_rows = selected_pathway_rows.sort_values(
                    ["component_label", "directional_fdr", "enrichment_fdr"],
                    kind="stable",
                )
                filterable_dataframe(
                    selected_pathway_rows[pathway_detail_columns],
                    table_key=f"pathway_mdc_detail_{pathway_mdc_resolution}",
                    table_name=f"{resolution_singular}-resolved MDC detail",
                    use_container_width=True,
                    hide_index=True,
                    height=480,
                    column_config={
                        **(
                            {"overlap_genes": "Overlap gene symbols"}
                            if pathway_mdc_resolution == "pathway"
                            else {}
                        ),
                        "supporting_pathway_names": "Supporting pathway names",
                        "best_supporting_pathway_name": "Best-FDR supporting pathway",
                    },
                )
                download_columns = [
                    "enrichment_resolution",
                    "enrichment_resolution_label",
                    "pathway_id",
                    "pathway_label",
                    "category_level1",
                    "category_level2",
                    "component",
                    "component_label",
                    "enrichment_scope",
                    "n_modules",
                    "n_pathways",
                    "mean_log2_mdc",
                    "geometric_mean_mdc",
                    "median_mdc",
                    "minimum_enrichment_fdr",
                    "median_enrichment_fdr",
                    "n_mdc_significant",
                    "proportion_mdc_significant",
                    "minimum_mdc_fdr",
                    "total_edges",
                ]
                download_columns = [
                    column for column in download_columns if column in pathway_summary
                ]
                pathway_download_columns = st.columns(2)
                pathway_download_columns[0].download_button(
                    f"Download {resolution_singular.lower()}-component summary (TSV)",
                    data=dataframe_to_tsv_bytes(pathway_summary[download_columns]),
                    file_name=(
                        f"{download_prefix}{pathway_mdc_resolution}_resolved_"
                        "MDC_summary.tsv"
                    ),
                    mime="text/tab-separated-values",
                )
                pathway_download_columns[1].download_button(
                    f"Download selected {resolution_singular.lower()} module rows (TSV)",
                    data=dataframe_to_tsv_bytes(
                        selected_pathway_rows[pathway_detail_columns]
                    ),
                    file_name=(
                        f"{download_prefix}selected_{pathway_mdc_resolution}_"
                        "regional_MDC_modules.tsv"
                    ),
                    mime="text/tab-separated-values",
                )


def render_statistics(context):
    statistics = context["statistics"]
    analysis_subset = context["analysis_subset"]
    cohort_scope = context["cohort_scope"]
    correlation_method = context["correlation_method"]
    diagnoses = context["diagnoses"]
    differential_edge_rule = context["differential_edge_rule"]
    differential_fdr_scope = context["differential_fdr_scope"]
    differential_fdr_threshold = context["differential_fdr_threshold"]
    download_prefix = context["download_prefix"]
    edge_rule = context["edge_rule"]
    estimator = context["estimator"]
    feature = context["feature"]
    method = context["method"]
    module = context["module"]
    module_count = context["module_count"]
    module_manifest = context["module_manifest"]
    module_set = context["module_set"]
    module_set_label = context["module_set_label"]
    phenotype = context["phenotype"]
    resolved = context["resolved"]
    scale = context["scale"]
    score_normalization = context["score_normalization"]
    selected_components = context["selected_components"]
    st.subheader("Robust association statistics")
    if phenotype in {"clusters", "study"}:
        cluster_statistics = cached_module_set_cluster_associations(
            module_set,
            module_count,
            estimator,
            method,
            resolved,
            feature,
            tuple(selected_components),
            tuple(diagnoses),
            True,
            edge_rule,
            differential_edge_rule,
            differential_fdr_scope,
            differential_fdr_threshold,
            score_normalization,
            analysis_subset,
            cohort_scope,
            category_variable=phenotype,
        )
        cluster_statistics = cluster_statistics.loc[
            cluster_statistics["module"].astype(int).eq(int(module))
        ].copy()
        cluster_statistics.insert(0, "module_definition", module_set_label)
        st.caption(
            f"{ASSOCIATION_OUTCOME_LABELS[phenotype]} statistics use Kruskal–Wallis and epsilon-squared. "
            "Categories with fewer than five non-missing donors are excluded from the test "
            "but reported. FDR is BH across modules only; no Pearson/Spearman statistic is "
            "calculated from nominal category codes. A single-study cohort cannot support a study comparison."
        )
        filterable_dataframe(
            cluster_statistics,
            table_key="cluster_robust_statistics",
            table_name=f"{ASSOCIATION_OUTCOME_LABELS[phenotype]} statistics",
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            f"Download nominal {phenotype} statistics (TSV)",
            data=dataframe_to_tsv_bytes(cluster_statistics),
            file_name=f"{download_prefix}M{module}_{phenotype}_{feature}_{method}_statistics.tsv",
            mime="text/tab-separated-values",
        )
        st.stop()
    st.caption(
        "Pathology corrections are BH across valid modules only within each outcome, "
        "feature, component and donor group. Legacy all-12/primary-five corrections "
        "are not applicable. The download retains all robustness and module-only FDR fields."
        if phenotype in PATHOLOGY_OUTCOMES
        else f"The primary table follows the sidebar selection: {correlation_method}. The "
        "download retains raw, asinh, Winsorized, Spearman, RINT, leave-one-out "
        "sensitivity, dependent CT-vs-TS tests, and all stored FDR fields. All-12 "
        "global FDR is the current primary correction; primary-five global FDR is "
        "retained for backward comparison and is unavailable for the seven added outcomes."
    )
    if cohort_scope == VALIDATION_SCOPE:
        display_columns = [
            "component_label",
            "diagnosis_group",
            "n",
            "unavailable_reason",
            f"rho_{scale}",
            f"p_spearman_{scale}",
            f"q_spearman_{scale}_across_modules",
            f"r_{scale}",
            f"p_{scale}",
            f"q_{scale}_across_modules",
            "r_winsor",
            "p_winsor",
            "loo_rint_max_delta",
            "loo_rint_sign_flip",
        ]
    elif resolved and correlation_method == "Spearman":
        display_columns = [
            "component_label",
            "diagnosis_group",
            "n",
            "rho",
            "p_spearman",
            "q_spearman_all12_global",
            "q_spearman_primary5_global",
        ]
    elif resolved:
        display_columns = [
            "component_label",
            "diagnosis_group",
            "n",
            "r_rint",
            "p_rint",
            "q_rint_all12_global",
            "q_rint_primary5_global",
            "q_rint_within_phenotype",
            "rho",
            "p_spearman",
            "loo_rint_max_delta",
            "loo_rint_sign_flip",
        ]
    elif correlation_method == "Spearman":
        display_columns = [
            "diagnosis_group",
            "n",
            "rho_CT",
            "p_spearman_CT",
            "q_spearman_CT_all12_global",
            "q_spearman_CT_primary5_global",
            "rho_TS",
            "p_spearman_TS",
            "q_spearman_TS_all12_global",
            "q_spearman_TS_primary5_global",
            "component_t_rank",
            "p_component_rank",
            "q_component_rank_within_phenotype",
            "q_component_rank_all12_global",
            "q_component_rank_primary5_global",
        ]
    else:
        display_columns = [
            "diagnosis_group",
            "n",
            "r_rint_CT",
            "p_rint_CT",
            "q_rint_CT_all12_global",
            "q_rint_CT_primary5_global",
            "r_rint_TS",
            "p_rint_TS",
            "q_rint_TS_all12_global",
            "q_rint_TS_primary5_global",
            "rho_CT",
            "rho_TS",
            "p_component_rint",
            "q_component_rint_within_phenotype",
            "q_component_rint_all12_global",
            "q_component_rint_primary5_global",
        ]
    statistics = statistics.copy()
    if phenotype in PATHOLOGY_OUTCOMES:
        display_columns = [
            column.replace("_all12_global", "_across_modules")
            for column in display_columns
            if "_primary5_global" not in column
        ]
        if not resolved:
            display_columns[
                display_columns.index("n") : display_columns.index("n") + 1
            ] = ["n_CT", "n_TS", "n"]
    statistics.insert(0, "module_definition", module_set_label)
    filterable_dataframe(
        statistics[display_columns],
        table_key="robust_statistics",
        table_name="Robust statistics",
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download all selected robust-statistics columns (TSV)",
        data=dataframe_to_tsv_bytes(statistics),
        file_name=(
            f"{download_prefix}M{module}_{phenotype}_{feature}_{method}_"
            "robust_statistics.tsv"
        ),
        mime="text/tab-separated-values",
    )
    st.info(fdr_text(module_manifest, module_count))


def render_kegg(context):
    active_tissues = context["active_tissues"]
    download_prefix = context["download_prefix"]
    estimator = context["estimator"]
    feature = context["feature"]
    method = context["method"]
    module = context["module"]
    module_count = context["module_count"]
    module_details = context["module_details"]
    module_manifest = context["module_manifest"]
    module_set = context["module_set"]
    module_set_label = context["module_set_label"]
    phenotype = context["phenotype"]
    st.subheader("Tissue-expanded KEGG enrichments")
    endpoint_kegg_ready = estimator == "lioness" and effect_endpoint_kegg_available(
        module_set, method, manifest=context["data_manifest"]
    )
    kegg_source_options = ["full_module"]
    if endpoint_kegg_ready:
        kegg_source_options.extend(["ad_higher", "control_higher", "either"])
    kegg_source = st.selectbox(
        "KEGG gene set",
        options=kegg_source_options,
        format_func=lambda value: {
            "full_module": "Full module genes",
            "ad_higher": "Hedges’ g endpoints: Higher in AD",
            "control_higher": "Hedges’ g endpoints: Higher in Control",
            "either": "Hedges’ g endpoints: Either direction",
        }[value],
        help=(
            "Endpoint catalogs contain unique tissue-gene nodes incident to at least "
            "one discovery AD–Control edge passing the fixed |Hedges’ g| ≥ 0.40 rule."
        ),
        key=f"kegg_gene_set_{module_set}_{method}",
    )
    if estimator != "lioness":
        st.caption("Effect-endpoint KEGG enrichment is available for LIONESS only.")
    elif not endpoint_kegg_ready:
        st.caption(
            "No completed effect-endpoint KEGG catalog is available for this module "
            "definition and network method. Full-module enrichment remains available."
        )
    kegg_scope = st.radio(
        "Enrichment table scope",
        options=["Selected module", "All modules"],
        horizontal=True,
        key=f"kegg_scope_{module_set}",
        help=(
            "Selected module follows the module chosen in the sidebar. All modules "
            "loads every reported enrichment row for the selected module definition."
        ),
    )
    kegg_module = module if kegg_scope == "Selected module" else None
    if kegg_source == "full_module":
        source_kegg = cached_kegg(module_set, kegg_module).copy()
        kegg_source_label = "Full module genes"
        kegg_effect_direction = None
    else:
        kegg_effect_direction = kegg_source
        source_kegg = cached_effect_endpoint_kegg(
            module_set, method, kegg_effect_direction, kegg_module
        ).copy()
        kegg_source_label = (
            "Hedges’ g endpoint genes · "
            + EFFECT_DIRECTION_LABELS[kegg_effect_direction]
        )
    if not source_kegg.empty:
        if "module_definition" in source_kegg:
            source_kegg["module_definition"] = module_set_label
        else:
            source_kegg.insert(0, "module_definition", module_set_label)

    if source_kegg.empty:
        unavailable_detail = "No pathway reached the minimum overlap."
        if kegg_source != "full_module" and kegg_effect_direction is not None:
            endpoint_summary = cached_effect_endpoint_summary(
                module_set, method, kegg_effect_direction, kegg_module
            )
            if len(endpoint_summary) == 1:
                reason = str(endpoint_summary.iloc[0]["availability_reason"])
                unavailable_detail = {
                    "no_retained_edges": "No edge passed the selected endpoint rule.",
                    "no_qualifying_pathway": (
                        "Edges passed, but no KEGG pathway reached the minimum overlap."
                    ),
                    "available": "No reported pathway remains after the current query.",
                }.get(reason, "Endpoint enrichment is unavailable.")
        st.info(
            f"{module_label(module)} has no reported pathway for {kegg_source_label}. "
            f"{unavailable_detail} Choose All modules to browse the available "
            "enrichments."
        )
    else:
        st.markdown("#### Filters")
        filter_row_one = st.columns([1.5, 1.25, 1.1, 0.9])
        if kegg_scope == "All modules":
            selected_kegg_modules = filter_row_one[0].multiselect(
                "Modules",
                options=sorted(module_details["module"].astype(int).unique()),
                default=[],
                format_func=module_label,
                placeholder="All modules",
                key=f"kegg_modules_{module_set}",
                help="Leave empty to include all modules in this definition.",
            )
        else:
            selected_kegg_modules = [int(module)]
            filter_row_one[0].text_input(
                "Module",
                value=module_label(module),
                disabled=True,
                key=f"kegg_selected_module_{module_set}",
            )
        kegg_region_suffix = {
            "AC": "AC",
            "DLPFC": "DLPFC",
            "PCG": "PCGBA23",
            "VH": "VH",
            "SMA": "SMA",
            "QUAD": "QUAD",
        }
        statistical_scope = filter_row_one[1].selectbox(
            "Statistical scope",
            options=[
                "All tissues (expanded)",
                "Any individual tissue",
                *active_tissues,
            ],
            key=f"kegg_statistical_scope_{module_set}_{kegg_scope}",
            help=(
                "Choose which FDR and significance columns control the significance "
                "and maximum-FDR filters."
            ),
        )
        significance_label = filter_row_one[2].selectbox(
            "Significance",
            options=["All rows", "FDR-significant only", "Not FDR-significant"],
            key=f"kegg_significance_{module_set}_{kegg_scope}",
        )
        maximum_fdr = filter_row_one[3].number_input(
            "Maximum FDR",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
            format="%.3f",
            key=f"kegg_maximum_fdr_{module_set}_{kegg_scope}",
        )

        filter_row_two = st.columns([1.1, 1.3, 1.6])
        category_options = sorted(
            source_kegg["category_level1"].dropna().astype(str).unique()
        )
        selected_categories = filter_row_two[0].multiselect(
            "KEGG categories",
            options=category_options,
            default=[],
            placeholder="All categories",
            key=f"kegg_categories_{module_set}_{kegg_scope}",
        )
        subcategory_source = source_kegg
        if selected_categories:
            subcategory_source = source_kegg.loc[
                source_kegg["category_level1"].isin(selected_categories)
            ]
        subcategory_options = sorted(
            subcategory_source["category_level2"].dropna().astype(str).unique()
        )
        selected_subcategories = filter_row_two[1].multiselect(
            "KEGG sub-categories",
            options=subcategory_options,
            default=[],
            placeholder="All sub-categories",
            key=f"kegg_subcategories_{module_set}_{kegg_scope}",
        )
        search = filter_row_two[2].text_input(
            "Search pathways or genes",
            placeholder="Alzheimer, lipid, infection, APOE…",
            key=f"kegg_search_{module_set}_{kegg_scope}",
        )

        filter_row_three = st.columns([1.45, 1.0, 1.0])
        regional_sharing = filter_row_three[0].selectbox(
            "Regional sharing",
            options=["any", "exactly_one", "at_least_two", "all_three"],
            format_func=lambda value: {
                "any": "Any regional pattern",
                "exactly_one": "Exactly one significant region",
                "at_least_two": "At least two significant regions",
                "all_three": "All three regions",
            }[value],
            key=f"kegg_regional_sharing_{module_set}_{kegg_scope}_{kegg_source}",
        )
        regional_sharing_threshold = filter_row_three[1].radio(
            "Regional FDR threshold",
            options=[0.05, 0.10],
            horizontal=True,
            format_func=lambda value: ("FDR ≤ 0.05" if value == 0.05 else "FDR ≤ 0.10"),
            key=f"kegg_regional_threshold_{module_set}_{kegg_scope}_{kegg_source}",
        )
        regional_top_n = filter_row_three[2].selectbox(
            "Visualization rows",
            options=[20, 50, 100],
            index=1,
            key=f"kegg_regional_topn_{module_set}_{kegg_scope}_{kegg_source}",
        )

        significance_value = {
            "All rows": "all",
            "FDR-significant only": "significant",
            "Not FDR-significant": "not_significant",
        }[significance_label]
        regional_significance_columns = [
            f"significant_{kegg_region_suffix[tissue]}" for tissue in active_tissues
        ]
        regional_fdr_columns = [
            f"fdr_{kegg_region_suffix[tissue]}" for tissue in active_tissues
        ]
        statistical_scope_columns = {
            "All tissues (expanded)": (["significant"], ["fdr"]),
            "Any individual tissue": (
                regional_significance_columns,
                regional_fdr_columns,
            ),
            **{
                tissue: (
                    [f"significant_{kegg_region_suffix[tissue]}"],
                    [f"fdr_{kegg_region_suffix[tissue]}"],
                )
                for tissue in active_tissues
            },
        }
        significance_columns, fdr_columns = statistical_scope_columns[statistical_scope]
        shown_kegg = filter_kegg_enrichments(
            source_kegg,
            modules=selected_kegg_modules if kegg_scope == "All modules" else None,
            categories=selected_categories,
            subcategories=selected_subcategories,
            significance=significance_value,
            maximum_fdr=maximum_fdr,
            search=search,
            significance_columns=significance_columns,
            fdr_columns=fdr_columns,
        )
        shown_kegg = filter_kegg_region_sharing(
            shown_kegg,
            mode=regional_sharing,
            fdr_threshold=float(regional_sharing_threshold),
        )

        significant_count = int(
            shown_kegg[significance_columns]
            .fillna(False)
            .astype(bool)
            .any(axis=1)
            .sum()
        )
        shown_module_count = int(shown_kegg["cluster_id"].nunique())
        source_module_count = int(source_kegg["cluster_id"].nunique())
        best_fdr = (
            shown_kegg[fdr_columns]
            .apply(pd.to_numeric, errors="coerce")
            .min(axis=1)
            .min()
            if not shown_kegg.empty
            else np.nan
        )
        shared_region_count = int(
            pd.to_numeric(
                shown_kegg.get("significant_region_count", pd.Series(dtype=float)),
                errors="coerce",
            )
            .ge(2)
            .sum()
        )
        kegg_metrics = st.columns(5)
        kegg_metrics[0].metric(
            "Rows shown", f"{len(shown_kegg):,} / {len(source_kegg):,}"
        )
        kegg_metrics[1].metric(
            "Modules shown",
            f"{shown_module_count:,} / "
            f"{module_count if kegg_scope == 'All modules' else 1:,}",
        )
        kegg_metrics[2].metric(
            f"Significant: {statistical_scope}", f"{significant_count:,}"
        )
        kegg_metrics[3].metric(
            "Rows significant in ≥2 regions", f"{shared_region_count:,}"
        )
        kegg_metrics[4].metric(
            f"Best FDR: {statistical_scope}",
            "NA" if pd.isna(best_fdr) else f"{best_fdr:.3e}",
        )
        st.caption(f"Enrichment gene set: {kegg_source_label}.")

        if kegg_scope == "All modules":
            st.caption(
                f"The supplied KEGG table contains reported rows for "
                f"{source_module_count} of {module_count} modules in this definition. "
                "Modules without a pathway meeting the enrichment input thresholds have no "
                "table row."
            )
        if shown_kegg.empty:
            st.info("No KEGG enrichment rows match the current filters.")
        else:
            st.markdown("#### Cross-region enrichment")
            regional_visualization = st.radio(
                "Cross-region visualization",
                options=["module_counts", "regional_heatmap"],
                format_func=lambda value: {
                    "module_counts": "Shared-pathway counts by module",
                    "regional_heatmap": "Module–pathway regional FDR heatmap",
                }[value],
                horizontal=True,
                key=f"kegg_regional_visualization_{module_set}_{kegg_scope}_{kegg_source}",
            )
            if regional_visualization == "module_counts":
                regional_figure = kegg_region_combination_figure(
                    shown_kegg,
                    top_n=int(regional_top_n),
                    module_definition=f"{module_set_label} · {kegg_source_label}",
                )
            else:
                regional_figure = kegg_region_heatmap_figure(
                    shown_kegg,
                    fdr_threshold=float(regional_sharing_threshold),
                    top_n=int(regional_top_n),
                    module_definition=f"{module_set_label} · {kegg_source_label}",
                )
            render_plotly_chart(
                regional_figure,
                use_container_width=True,
                config={"displaylogo": False},
                key=(
                    f"kegg_regional_{regional_visualization}_{module_set}_{method}_"
                    f"{kegg_source}_{kegg_scope}"
                ),
                metadata={
                    "module_set": module_set,
                    "method": method,
                    "module": f"M{module}",
                    "outcome": phenotype,
                    "feature": feature,
                },
            )
        regional_priority_columns = [
            column
            for tissue in active_tissues
            for column in (
                f"p_{kegg_region_suffix[tissue]}",
                f"fdr_{kegg_region_suffix[tissue]}",
                f"significant_{kegg_region_suffix[tissue]}",
            )
        ]
        kegg_priority_columns = [
            "module_definition",
            "cluster_id",
            "significant_region_count",
            "significant_regions",
            "regional_sharing_fdr_threshold",
            "pathway_name",
            "category_level1",
            "category_level2",
            "p",
            "fdr",
            "significant",
            *regional_priority_columns,
            "retained_edge_count",
            "endpoint_count",
            "endpoint_coverage",
            "effect_direction",
        ]
        kegg_priority_columns = [
            column for column in kegg_priority_columns if column in shown_kegg
        ]
        displayed_kegg = shown_kegg[
            kegg_priority_columns
            + [
                column
                for column in shown_kegg.columns
                if column not in kegg_priority_columns
            ]
        ]
        regional_column_config = {
            f"p_{kegg_region_suffix[tissue]}": st.column_config.NumberColumn(
                f"{tissue} p-value", format="%.3e"
            )
            for tissue in active_tissues
        }
        regional_column_config.update(
            {
                f"fdr_{kegg_region_suffix[tissue]}": st.column_config.NumberColumn(
                    f"{tissue} FDR", format="%.3e"
                )
                for tissue in active_tissues
            }
        )
        regional_column_config.update(
            {
                f"significant_{kegg_region_suffix[tissue]}": f"{tissue} significant"
                for tissue in active_tissues
            }
        )
        filterable_dataframe(
            displayed_kegg,
            table_key="kegg_enrichments",
            table_name="KEGG enrichments",
            use_container_width=True,
            hide_index=True,
            height=650,
            column_config={
                "cluster_id": st.column_config.NumberColumn("Module", format="M%d"),
                "p": st.column_config.NumberColumn(
                    "All regions p (expanded)", format="%.3e"
                ),
                "fdr": st.column_config.NumberColumn(
                    "All regions FDR (expanded)", format="%.3e"
                ),
                "significant": "All regions significant (expanded)",
                **regional_column_config,
                "significant_region_count": st.column_config.NumberColumn(
                    "Significant regions", format="%d"
                ),
                "significant_regions": "Significant region combination",
                "regional_sharing_fdr_threshold": st.column_config.NumberColumn(
                    "Sharing FDR threshold", format="%.2f"
                ),
                "retained_edge_count": st.column_config.NumberColumn(
                    "Retained edges", format="%d"
                ),
                "endpoint_count": st.column_config.NumberColumn(
                    "Endpoint tissue-genes", format="%d"
                ),
                "endpoint_coverage": st.column_config.NumberColumn(
                    "Endpoint coverage", format="%.3f"
                ),
                "overlap_genes": "Overlap gene symbols",
            },
        )
        st.download_button(
            "Download currently shown KEGG rows (TSV)",
            data=dataframe_to_tsv_bytes(shown_kegg),
            file_name=(
                f"{download_prefix}"
                + (
                    f"M{module}_{kegg_source}_filtered_tissue_expanded_KEGG.tsv"
                    if kegg_scope == "Selected module"
                    else f"allmodules_{kegg_source}_filtered_tissue_expanded_KEGG.tsv"
                )
            ),
            mime="text/tab-separated-values",
        )
    if kegg_source != "full_module" and kegg_effect_direction is not None:
        st.download_button(
            "Download complete selected endpoint KEGG catalog",
            data=cached_effect_endpoint_kegg_tsv(
                module_set, method, kegg_effect_direction
            ),
            file_name=(
                f"{download_prefix}{method}_hedges_g_040_{kegg_effect_direction}_"
                "endpoint_KEGG.tsv"
            ),
            mime="text/tab-separated-values",
        )
    st.download_button(
        "Download complete original full-module KEGG catalog",
        data=cached_kegg_tsv(module_set),
        file_name=f"{download_prefix}method4_tissue_expanded_kegg_annotated.tsv",
        mime="text/tab-separated-values",
    )
    kegg_pathway_space = int(
        module_manifest.get("kegg", {}).get("per_region_pathways_per_module_max", 350)
    )
    st.caption(
        f"All-tissues p/FDR uses the tissue-expanded enrichment test. {', '.join(active_tissues)} "
        "p-values use separate tissue-specific ORA tests against their corresponding "
        "expression backgrounds. Each regional FDR is Benjamini-Hochberg adjusted within "
        f"that module and region across {kegg_pathway_space} KEGG pathways; pathways below the minimum "
        "overlap receive p=1. Values are supplied by the enrichment analyses and are not "
        "recalculated by this app. Endpoint enrichments are exploratory because genes "
        "incident to many retained edges are more likely to enter the tested gene set."
    )


def render_details(context):
    module_details = context["module_details"]
    active_tissue_detail_columns = context["active_tissue_detail_columns"]
    active_tissues = context["active_tissues"]
    download_prefix = context["download_prefix"]
    feature = context["feature"]
    method = context["method"]
    module = context["module"]
    module_count = context["module_count"]
    module_set = context["module_set"]
    module_set_label = context["module_set_label"]
    phenotype = context["phenotype"]
    from app_helpers.workspaces.shared import render_composition

    render_composition(context)
    st.subheader("Level-4 module composition")
    st.caption(
        "Module size, represented tissues, per-tissue gene counts, and proportions from "
        "the level-4 SE2 module-details file. Tissue proportions use module size as the denominator."
    )
    st.markdown("#### Module landscape")
    module_type_scope = st.radio(
        "Module types to show",
        options=["Both", "CT only", "TS only"],
        horizontal=True,
        key=f"module_details_scope_{module_set}",
        help=(
            "This filter uses the CT/TS module classification in the selected level-4 "
            "SE2 module-details file and controls both charts below."
        ),
    )
    entropy_range = st.slider(
        "Normalized tissue-entropy range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.01,
        key=f"module_entropy_range_{module_set}",
        help=(
            f"Zero is a single-tissue module; one is equal {'/'.join(active_tissues)} "
            "composition. This filter combines with the CT/TS filter."
        ),
    )
    if module_type_scope == "Both":
        chart_module_details = module_details.copy()
    else:
        selected_module_type = module_type_scope.split()[0]
        chart_module_details = module_details.loc[
            module_details["cluster_type"]
            .astype(str)
            .str.upper()
            .eq(selected_module_type)
        ].copy()
    chart_module_details = chart_module_details.loc[
        chart_module_details["tissue_entropy_normalized"].between(
            entropy_range[0], entropy_range[1], inclusive="both"
        )
    ].copy()

    if chart_module_details.empty:
        st.warning(f"No {module_type_scope} modules are available in this definition.")
    else:
        landscape_metrics = st.columns(5)
        landscape_metrics[0].metric("Modules shown", f"{len(chart_module_details):,}")
        landscape_metrics[1].metric(
            "Median size", f"{chart_module_details['module_size'].median():,.0f} genes"
        )
        largest_module = chart_module_details.sort_values(
            ["module_size", "module"], ascending=[False, True]
        ).iloc[0]
        landscape_metrics[2].metric(
            "Largest module", f"M{int(largest_module['module'])}"
        )
        landscape_metrics[3].metric(
            "Largest size", f"{int(largest_module['module_size']):,} genes"
        )
        landscape_metrics[4].metric(
            "Median normalized entropy",
            f"{chart_module_details['tissue_entropy_normalized'].median():.3f}",
        )
        render_plotly_chart(
            module_size_distribution_figure(
                chart_module_details,
                module_definition=module_set_label,
            ),
            use_container_width=True,
            config={"displaylogo": False, "responsive": True},
            key=f"module_size_distribution_{module_set}_{module_type_scope}",
            metadata={
                "module_set": module_set,
                "method": method,
                "module": f"M{module}",
                "outcome": phenotype,
                "feature": feature,
            },
        )
        composition_controls = st.columns(2)
        composition_orientation = composition_controls[0].selectbox(
            "Composition orientation",
            ["h", "v"],
            format_func=lambda value: "Horizontal" if value == "h" else "Vertical",
            key=f"module_composition_orientation_{module_set}",
        )
        composition_scale = composition_controls[1].selectbox(
            "Composition values",
            ["counts", "proportions"],
            format_func=lambda value: (
                "Gene counts" if value == "counts" else "Tissue proportions (100%)"
            ),
            key=f"module_composition_scale_{module_set}",
            help="Proportions divide each tissue's gene count by the full module size. Hiding a tissue does not renormalize the others.",
        )
        with st.expander("Composition axis ticks", expanded=False):
            tick_columns = st.columns(2)
            composition_x_ticks = tick_columns[0].checkbox(
                "Show X-axis ticks",
                value=True,
                key=f"module_composition_x_ticks_{module_set}",
                help="Show or hide tick marks and tick labels without hiding the axis title or hover information.",
            )
            composition_y_ticks = tick_columns[1].checkbox(
                "Show Y-axis ticks",
                value=True,
                key=f"module_composition_y_ticks_{module_set}",
                help="Show or hide tick marks and tick labels without hiding the axis title or hover information.",
            )
        composition_colors = {}
        with st.expander("Composition tissue colors", expanded=False):
            color_columns = st.columns(len(active_tissues))
            for column, tissue in zip(color_columns, active_tissues):
                composition_colors[tissue] = column.color_picker(
                    f"{tissue} tissue color",
                    value=_chart_helpers.TISSUE_COMPOSITION_COLORS[tissue],
                    key=f"module_composition_color_{module_set}_{tissue}",
                )
        render_plotly_chart(
            module_region_composition_figure(
                chart_module_details,
                module_definition=module_set_label,
                orientation=composition_orientation,
                composition_scale=composition_scale,
                tissue_colors=composition_colors,
                show_x_ticks=composition_x_ticks,
                show_y_ticks=composition_y_ticks,
            ),
            use_container_width=True,
            config={
                "displaylogo": False,
                "responsive": True,
                "scrollZoom": True,
            },
            key=f"module_region_composition_{module_set}_{module_type_scope}",
            metadata={
                "module_set": module_set,
                "method": method,
                "module": f"M{module}",
                "outcome": phenotype,
                "feature": feature,
            },
        )
        render_plotly_chart(
            module_entropy_figure(
                chart_module_details,
                selected_module=module,
                module_definition=module_set_label,
            ),
            use_container_width=True,
            config={"displaylogo": False, "responsive": True},
            key=f"module_entropy_{module_set}_{module_type_scope}",
            metadata={
                "module_set": module_set,
                "method": method,
                "module": f"M{module}",
                "outcome": phenotype,
                "feature": feature,
            },
        )
        composition_description = (
            "Each complete bar represents 100% of the module's genes; colored segments show tissue proportions. "
            if composition_scale == "proportions"
            else "Total bar length is module size; colored segments show gene counts. "
        )
        st.caption(
            composition_description + f"Tissues are {', '.join(active_tissues[:-1])}, "
            f"and {active_tissues[-1]}. Modules are ordered largest to smallest. Hover for exact "
            "counts and proportions; drag or scroll to zoom. Entropy is the continuous "
            "tissue-mixing complement to the discrete CT/TS module classification: zero "
            "means one tissue only and one means equal representation of all three tissues."
        )
        landscape_download = chart_module_details.copy()
        landscape_download.insert(0, "module_definition", module_set_label)
        st.download_button(
            "Download modules shown in the landscape (TSV)",
            data=dataframe_to_tsv_bytes(landscape_download),
            file_name=(
                f"{download_prefix}module_landscape_"
                f"{module_type_scope.replace(' ', '_')}_entropy_filtered.tsv"
            ),
            mime="text/tab-separated-values",
        )

    tissue_detail_columns = [
        column
        for _tissue, count_column, proportion_column in active_tissue_detail_columns
        for column in (count_column, proportion_column)
    ]
    detail_columns = [
        "module_definition",
        "module",
        "module_size",
        "cluster_type",
        "tissues",
        "n_tissues",
        "dominant_tissue",
        "tissue_entropy",
        "tissue_entropy_normalized",
        *tissue_detail_columns,
    ]
    tissue_column_config = {
        count_column: st.column_config.NumberColumn(f"{tissue} genes", format="%d")
        for tissue, count_column, _proportion_column in active_tissue_detail_columns
    }
    tissue_column_config.update(
        {
            proportion_column: st.column_config.NumberColumn(
                f"{tissue} proportion", format="percent"
            )
            for tissue, _count_column, proportion_column in active_tissue_detail_columns
        }
    )
    module_details = module_details.copy()
    module_details.insert(0, "module_definition", module_set_label)
    st.markdown(f"#### Selected module: {module_label(module)}")
    filterable_dataframe(
        module_details.loc[
            module_details["module"].astype(int).eq(int(module)), detail_columns
        ],
        table_key="selected_module_details",
        table_name="Selected module details",
        use_container_width=True,
        hide_index=True,
        column_config={
            "module": st.column_config.NumberColumn("Module", format="M%d"),
            "module_size": st.column_config.NumberColumn("Module genes", format="%d"),
            "cluster_type": "Module type",
            "tissues": "Tissues",
            "n_tissues": st.column_config.NumberColumn("N tissues", format="%d"),
            "dominant_tissue": "Dominant tissue",
            "tissue_entropy": st.column_config.NumberColumn(
                "Tissue entropy (bits)", format="%.4f"
            ),
            "tissue_entropy_normalized": st.column_config.ProgressColumn(
                "Normalized tissue entropy", format="%.3f", min_value=0.0, max_value=1.0
            ),
            **tissue_column_config,
        },
    )
    st.markdown(f"#### All {module_count} modules")
    filterable_dataframe(
        module_details[detail_columns],
        table_key="all_module_details",
        table_name="All module details",
        use_container_width=True,
        hide_index=True,
        height=520,
        column_config={
            "module": st.column_config.NumberColumn("Module", format="M%d"),
            "module_size": st.column_config.NumberColumn("Module genes", format="%d"),
            "cluster_type": "Module type",
            "tissues": "Tissues",
            "n_tissues": st.column_config.NumberColumn("N tissues", format="%d"),
            "dominant_tissue": "Dominant tissue",
            "tissue_entropy": st.column_config.NumberColumn(
                "Tissue entropy (bits)", format="%.4f"
            ),
            "tissue_entropy_normalized": st.column_config.ProgressColumn(
                "Normalized tissue entropy", format="%.3f", min_value=0.0, max_value=1.0
            ),
            **tissue_column_config,
        },
    )
    st.download_button(
        "Download complete module-details table (TSV)",
        data=dataframe_to_tsv_bytes(module_details[detail_columns]),
        file_name=f"{download_prefix}se2_level4_module_details.tsv",
        mime="text/tab-separated-values",
    )
    st.info(
        "A tissue with one module gene can contribute CT edges but has no within-tissue "
        "gene pair, so its tissue-specific feature is unavailable."
    )
