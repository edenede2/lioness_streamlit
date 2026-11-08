from __future__ import annotations

from app_helpers.ui_runtime import (
    ASSOCIATION_GROUP_LABELS,
    ASSOCIATION_OUTCOME_LABELS,
    DIAGNOSIS_ORDER,
    FINDER_CRITERIA,
    NUMERIC_OUTCOMES,
    OUTCOME_LABELS,
    PATHOLOGY_OUTCOMES,
    VALIDATION_SCOPE,
    association_level_label,
    benjamini_hochberg,
    build_module_finder_table,
    cached_aggregate_stats,
    cached_all_module_correlations,
    cached_grouped_all_module_correlations,
    cached_grouped_module_correlations,
    cached_module_correlations,
    cached_module_set_cluster_associations,
    cached_pooled_module_finder_stats,
    cached_sample_metadata,
    cluster_association_heatmap_figure,
    clustered_correlation_group_order,
    correlation_heatmap_figure,
    dataframe_to_tsv_bytes,
    fdr_text,
    filterable_dataframe,
    module_finder_figure,
    np,
    pd,
    render_plotly_chart,
    st,
)


def render_heatmaps(context):
    active_component_order = context["active_component_order"]
    active_feature_labels = context["active_feature_labels"]
    analysis_subset = context["analysis_subset"]
    available_components = context["available_components"]
    cohort_scope = context["cohort_scope"]
    component_labels = context["component_labels"]
    correlation_method = context["correlation_method"]
    diagnoses = context["diagnoses"]
    diagnosis_options = context["diagnosis_options"]
    differential_edge_rule = context["differential_edge_rule"]
    differential_fdr_scope = context["differential_fdr_scope"]
    differential_fdr_threshold = context["differential_fdr_threshold"]
    download_prefix = context["download_prefix"]
    edge_rule = context["edge_rule"]
    estimator = context["estimator"]
    feature = context["feature"]
    heatmap_grouping_variable = context["heatmap_grouping_variable"]
    heatmap_minimum_group_n = context["heatmap_minimum_group_n"]
    method = context["method"]
    module = context["module"]
    module_count = context["module_count"]
    module_set = context["module_set"]
    module_set_label = context["module_set_label"]
    phenotype = context["phenotype"]
    resolved = context["resolved"]
    score_normalization = context["score_normalization"]
    selected_components = context["selected_components"]
    selected_heatmap_group_levels = context["selected_heatmap_group_levels"]
    heatmap_feature_labels = {
        key: label
        for key, label in active_feature_labels.items()
        if resolved or key != "eigengene"
    }
    heatmap_statistic_type = st.radio(
        "Association type",
        [
            "Numeric correlations",
            "Nominal ROSMAP clusters",
            "Study / project (ROS / MAP)",
        ],
        horizontal=True,
        key="heatmap_statistic_type",
    )
    if heatmap_statistic_type != "Numeric correlations":
        heatmap_category = (
            "study"
            if heatmap_statistic_type == "Study / project (ROS / MAP)"
            else "clusters"
        )
        category_name = "Study" if heatmap_category == "study" else "Cluster"
        st.subheader(
            f"Network-score associations with {ASSOCIATION_OUTCOME_LABELS[heatmap_category]}"
        )
        st.caption(
            "This is a separate nonnegative epsilon-squared heatmap. Asterisks mark "
            "Kruskal–Wallis FDR < 0.05 after BH across modules only. Nominal study/cluster labels are "
            "never passed to Pearson or Spearman correlation. Single-study cohorts have unavailable study tests."
        )
        cluster_heatmap_scope = st.radio(
            f"{category_name} heatmap scope",
            ["Selected module", f"All {module_count} modules"],
            horizontal=True,
        )
        cluster_heatmap_diagnosis = st.selectbox(
            f"Diagnosis/cohort in {category_name.lower()} heatmap",
            ["All donors", *diagnosis_options, "All diagnosis groups"],
            index=0,
        )
        cluster_features = st.multiselect(
            "Module scores",
            options=list(heatmap_feature_labels),
            default=(
                [feature]
                if heatmap_category == "study"
                else list(heatmap_feature_labels)
            ),
            format_func=lambda value: heatmap_feature_labels[value],
        )
        cluster_components = tuple(selected_components)
        if cluster_heatmap_diagnosis == "All donors":
            requested_diagnoses = tuple(diagnosis_options)
            include_cluster_pooled = True
            display_groups = ["All donors"]
        elif cluster_heatmap_diagnosis == "All diagnosis groups":
            requested_diagnoses = tuple(diagnosis_options)
            include_cluster_pooled = True
            display_groups = ["All donors", *diagnosis_options]
        else:
            requested_diagnoses = (cluster_heatmap_diagnosis,)
            include_cluster_pooled = False
            display_groups = [cluster_heatmap_diagnosis]
        cluster_frames = [
            cached_module_set_cluster_associations(
                module_set,
                module_count,
                estimator,
                method,
                resolved,
                selected_feature,
                cluster_components,
                requested_diagnoses,
                include_cluster_pooled,
                edge_rule,
                differential_edge_rule,
                differential_fdr_scope,
                differential_fdr_threshold,
                score_normalization,
                analysis_subset,
                cohort_scope,
                category_variable=heatmap_category,
            )
            for selected_feature in cluster_features
        ]
        cluster_table = (
            pd.concat(cluster_frames, ignore_index=True)
            if cluster_frames
            else pd.DataFrame()
        )
        if not cluster_table.empty:
            cluster_table = cluster_table.loc[
                cluster_table["diagnosis_group"].isin(display_groups)
            ].copy()
            if cluster_heatmap_scope == "Selected module":
                cluster_table = cluster_table.loc[
                    cluster_table["module"].astype(int).eq(int(module))
                ]
            cluster_table["feature_label"] = cluster_table["metric_family"].map(
                heatmap_feature_labels
            )
            cluster_table["base_component_label"] = cluster_table["component_label"]
            if len(display_groups) > 1:
                cluster_table["component_label"] = (
                    cluster_table["diagnosis_group"].astype(str)
                    + " · "
                    + cluster_table["component_label"].astype(str)
                )
            cluster_table["heatmap_row"] = (
                "M"
                + cluster_table["module"].astype(int).astype(str)
                + " · "
                + cluster_table["feature_label"].astype(str)
            )
        show_cluster_significant = st.checkbox(
            f"Show only rows with at least one {category_name.lower()} FDR < 0.05",
            value=False,
        )
        if show_cluster_significant and not cluster_table.empty:
            keep_rows = cluster_table.loc[
                cluster_table["categorical_fdr_across_modules"].lt(0.05),
                "heatmap_row",
            ].unique()
            cluster_table = cluster_table.loc[
                cluster_table["heatmap_row"].isin(keep_rows)
            ]
        cluster_ordering = st.selectbox(
            f"{category_name} heatmap clustering",
            ["None", "Rows", "Columns", "Rows and columns"],
        )
        if cluster_table.empty:
            st.info(
                f"No nominal {category_name.lower()} associations meet the selected filters."
            )
        elif not cluster_table["epsilon_squared"].notna().any():
            st.info(
                "No between-category test is available for the displayed rows. Check that at least two studies/categories meet the minimum sample size; the motor cohort is MAP-only."
            )
        else:
            render_plotly_chart(
                cluster_association_heatmap_figure(
                    cluster_table,
                    title=f"{ASSOCIATION_OUTCOME_LABELS[heatmap_category]} association effect sizes",
                    cluster_rows=cluster_ordering in {"Rows", "Rows and columns"},
                    cluster_columns=cluster_ordering in {"Columns", "Rows and columns"},
                ),
                use_container_width=True,
                config={"displaylogo": False},
                plot_id="main.cluster_association_heatmap_figure",
                metadata={
                    "module_set": module_set,
                    "method": method,
                    "module": f"M{module}",
                    "outcome": heatmap_category,
                    "feature": feature,
                },
            )
            filterable_dataframe(
                cluster_table,
                table_key="complete_cluster_association_table",
                table_name=f"Complete nominal {category_name.lower()} association table",
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                f"Download nominal {category_name.lower()} association table (TSV)",
                data=dataframe_to_tsv_bytes(cluster_table),
                file_name=f"{download_prefix}{heatmap_category}_association_statistics.tsv",
                mime="text/tab-separated-values",
            )
        st.stop()
    st.subheader("Module-score correlations across phenotypes and outcomes")
    st.caption(
        "Heatmaps use donor-level Z-scored network features and, when selected, "
        "tissue-specific module eigengenes. Nominal fields such as sex code "
        "and APOE genotype remain available in hover but are excluded from Pearson/Spearman "
        "heatmaps because numeric correlation is not appropriate for unordered categories. "
        "CogDx, Braak, CERAD, ADNC, and Parkinsonism are source-coded ordinal/binary outcomes; "
        "Spearman is generally the more appropriate descriptive coefficient for the ordinal fields."
    )
    heatmap_mode = st.radio(
        "Heatmap scope",
        [
            "Selected module: all feature scores",
            f"All {module_count} modules: selected or all feature scores",
        ],
        horizontal=True,
    )
    st.caption(
        f"Correlation method: **{correlation_method}** (controlled in the sidebar)."
    )
    if heatmap_grouping_variable == "diagnosis_group":
        heatmap_diagnosis = st.selectbox(
            "Diagnosis in heatmap",
            options=["All donors", *diagnosis_options, "All diagnosis groups"],
            index=3,
            key="heatmap_diagnosis",
            help=(
                "All diagnosis groups displays pooled donors, Control, MCI, and AD together; "
                "the group is included in each heatmap row and in the complete table."
            ),
        )
        heatmap_group_order = ["All donors", *diagnosis_options]
        heatmap_context_label = heatmap_diagnosis
        heatmap_file_group = heatmap_diagnosis
    else:
        if not selected_heatmap_group_levels:
            st.warning("Select at least one heatmap group level in the sidebar.")
            st.stop()
        heatmap_diagnosis = "Selected category levels"
        heatmap_group_order = [
            association_level_label(heatmap_grouping_variable, value)
            for value in selected_heatmap_group_levels
        ]
        heatmap_context_label = (
            f"{ASSOCIATION_GROUP_LABELS[heatmap_grouping_variable]}: "
            + ", ".join(heatmap_group_order)
        )
        heatmap_file_group = heatmap_grouping_variable
        st.caption(
            f"Within-category correlations for **{heatmap_context_label}**. "
            "Diagnosis selections in the sidebar remain donor-inclusion filters."
        )
    clustering_options = ["None", "Rows", "Columns", "Rows and columns"]
    if heatmap_mode.startswith("All"):
        clustering_options.extend(["Modules", "Modules and columns"])
    heatmap_clustering = st.selectbox(
        "Heatmap clustering",
        options=clustering_options,
        help=(
            "Reorders the selected axes using average-linkage hierarchical clustering "
            "of Euclidean distances between correlation profiles. Module clustering "
            "clusters whole module profiles, keeps all score rows from a module together, "
            "and outlines each module block."
        ),
    )
    cluster_rows = heatmap_clustering in {"Rows", "Rows and columns"}
    cluster_columns = heatmap_clustering in {
        "Columns",
        "Rows and columns",
        "Modules and columns",
    }
    cluster_modules = heatmap_clustering in {"Modules", "Modules and columns"}
    if correlation_method == "Pearson":
        value_column = "pearson_r"
        p_column = "pearson_p"
    else:
        value_column = "spearman_rho"
        p_column = "spearman_p"

    heatmap_outcome_options = [
        outcome for outcome in NUMERIC_OUTCOMES if outcome in OUTCOME_LABELS
    ]
    selected_heatmap_outcomes = st.multiselect(
        "Outcome columns in heatmap and table",
        options=heatmap_outcome_options,
        default=heatmap_outcome_options,
        format_func=lambda value: OUTCOME_LABELS[value],
        key=(
            "correlation_heatmap_outcomes_selected_module"
            if heatmap_mode.startswith("Selected module")
            else "correlation_heatmap_outcomes_all_modules"
        ),
        help=(
            "Restricts the displayed heatmap columns, complete table, and download. "
            "The underlying module-family FDR corrections are preserved and are not "
            "recalculated after display filtering."
        ),
    )
    if not selected_heatmap_outcomes:
        st.warning("Select at least one outcome column for the correlation heatmap.")
        st.stop()
    st.caption(
        f"Displaying {len(selected_heatmap_outcomes)} of "
        f"{len(heatmap_outcome_options)} numeric outcomes."
    )

    row_group_labels: dict[str, str] | None = None

    if heatmap_mode.startswith("Selected module"):
        if heatmap_grouping_variable == "diagnosis_group":
            correlation_table = cached_module_correlations(
                module_set,
                estimator,
                method,
                module,
                resolved,
                edge_rule,
                differential_edge_rule,
                differential_fdr_scope,
                differential_fdr_threshold,
                score_normalization,
                analysis_subset,
                cohort_scope,
            )
            if heatmap_diagnosis == "All diagnosis groups":
                heatmap_data = correlation_table.copy()
                heatmap_data["heatmap_row"] = (
                    heatmap_data["heatmap_row"]
                    + " · "
                    + heatmap_data["diagnosis_group"].astype(str)
                )
            else:
                heatmap_data = correlation_table.loc[
                    correlation_table["diagnosis_group"].eq(heatmap_diagnosis)
                ].copy()
        else:
            correlation_table = cached_grouped_module_correlations(
                module_set,
                estimator,
                method,
                module,
                resolved,
                tuple(diagnoses),
                heatmap_grouping_variable,
                tuple(selected_heatmap_group_levels),
                heatmap_minimum_group_n,
                edge_rule,
                differential_edge_rule,
                differential_fdr_scope,
                differential_fdr_threshold,
                score_normalization,
                analysis_subset,
                cohort_scope,
            )
            heatmap_data = correlation_table.copy()
            if len(heatmap_group_order) > 1:
                heatmap_data["heatmap_row"] = (
                    heatmap_data["heatmap_row"]
                    + " · "
                    + heatmap_data["grouping_label"].astype(str)
                )
        correlation_table = correlation_table.loc[
            correlation_table["outcome"].isin(selected_heatmap_outcomes)
        ].copy()
        heatmap_data = heatmap_data.loc[
            heatmap_data["outcome"].isin(selected_heatmap_outcomes)
        ].copy()
        if correlation_table.empty:
            st.info(
                "No correlations are available for the selected outcome columns and "
                "category levels."
            )
            st.stop()
        fdr_column = (
            "pearson_fdr_displayed_family"
            if correlation_method == "Pearson"
            else "spearman_fdr_displayed_family"
        )
        component_rank = {
            value: index for index, value in enumerate(active_component_order)
        }
        feature_rank = {
            value: index for index, value in enumerate(heatmap_feature_labels)
        }
        diagnosis_rank = {
            value: index for index, value in enumerate(heatmap_group_order)
        }
        row_order_frame = (
            heatmap_data[
                ["metric_family", "component", "diagnosis_group", "heatmap_row"]
            ]
            .drop_duplicates()
            .assign(
                feature_rank=lambda frame: frame["metric_family"].map(feature_rank),
                component_rank=lambda frame: frame["component"].map(component_rank),
                diagnosis_rank=lambda frame: frame["diagnosis_group"].map(
                    diagnosis_rank
                ),
            )
            .sort_values(
                ["feature_rank", "component_rank", "component", "diagnosis_rank"]
            )
        )
        row_order = row_order_frame["heatmap_row"].tolist()
        heatmap_title = (
            f"{module_set_label} · Module M{module}: all available module scores vs "
            f"outcomes · {heatmap_context_label}"
        )
        fdr_scope = (
            "Displayed-family FDR is Benjamini–Hochberg correction across every feature, "
            "component, and outcome calculated for this module, separately within each "
            "selected category level."
            if heatmap_grouping_variable != "diagnosis_group"
            else (
                "Displayed-family FDR is Benjamini–Hochberg correction across every feature, "
                "component, outcome, and diagnosis correlation calculated for this module."
            )
        )
    else:
        all_feature = st.selectbox(
            "Features for all-module heatmap and table",
            options=["__all__", *list(heatmap_feature_labels)],
            format_func=lambda value: (
                "All available module scores"
                if value == "__all__"
                else heatmap_feature_labels[value]
            ),
            index=1 + list(heatmap_feature_labels).index(feature),
        )
        if resolved:
            component_options = available_components["component"].tolist()
            all_component = st.selectbox(
                "Components for all-module heatmap and table",
                options=["__all__", *component_options],
                format_func=lambda value: (
                    f"All {len(component_options)} tissue-resolved components"
                    if value == "__all__"
                    else component_labels[value]
                ),
                index=(
                    1 + component_options.index("TS_AC")
                    if all_feature == "eigengene" and "TS_AC" in component_options
                    else 1
                ),
                help=(
                    "All components combines "
                    + ", ".join(component_labels[value] for value in component_options)
                    + " in the same heatmap and complete table."
                ),
            )
        else:
            all_component = st.selectbox(
                "Components for all-module heatmap and table",
                options=["__all__", "CT", "TS"],
                format_func=lambda value: (
                    "Both aggregate components (CT and TS)"
                    if value == "__all__"
                    else f"{value} aggregate"
                ),
                index=1,
                help=(
                    "Both aggregate components displays CT and TS rows together in "
                    "the heatmap and complete table."
                ),
            )
        if heatmap_grouping_variable == "diagnosis_group":
            correlation_table = cached_all_module_correlations(
                module_set,
                estimator,
                method,
                resolved,
                all_feature,
                all_component,
                heatmap_diagnosis,
                edge_rule,
                differential_edge_rule,
                differential_fdr_scope,
                differential_fdr_threshold,
                score_normalization,
                analysis_subset,
                cohort_scope,
            )
        else:
            correlation_table = cached_grouped_all_module_correlations(
                module_set,
                estimator,
                method,
                resolved,
                all_feature,
                all_component,
                tuple(diagnoses),
                heatmap_grouping_variable,
                tuple(selected_heatmap_group_levels),
                heatmap_minimum_group_n,
                edge_rule,
                differential_edge_rule,
                differential_fdr_scope,
                differential_fdr_threshold,
                score_normalization,
                analysis_subset,
                cohort_scope,
            )
        correlation_table = correlation_table.loc[
            correlation_table["outcome"].isin(selected_heatmap_outcomes)
        ].copy()
        if correlation_table.empty:
            st.info(
                "No correlations are available for the selected outcome columns and "
                "category levels."
            )
            st.stop()
        correlation_table["feature_label"] = correlation_table["metric_family"].map(
            heatmap_feature_labels
        )
        heatmap_data = correlation_table.copy()
        include_feature_in_row = all_feature == "__all__"
        include_component_in_row = all_component == "__all__"
        include_group_in_row = (
            heatmap_diagnosis == "All diagnosis groups"
            if heatmap_grouping_variable == "diagnosis_group"
            else len(heatmap_group_order) > 1
        )
        heatmap_data["heatmap_row"] = heatmap_data["module"].map(
            lambda value: f"M{int(value)}"
        )
        if include_feature_in_row:
            heatmap_data["heatmap_row"] += " · " + heatmap_data["feature_label"].astype(
                str
            )
        if include_component_in_row:
            heatmap_data["heatmap_row"] += " · " + heatmap_data[
                "component_label"
            ].astype(str)
        if include_group_in_row:
            heatmap_data["heatmap_row"] += " · " + heatmap_data[
                "diagnosis_group"
            ].astype(str)

        if cluster_modules:
            module_order = clustered_correlation_group_order(
                heatmap_data,
                value_column=value_column,
                group_column="module",
                subgroup_columns=("metric_family", "component", "diagnosis_group"),
            )
        else:
            module_order = sorted(heatmap_data["module"].astype(int).unique())
        module_rank = {int(value): index for index, value in enumerate(module_order)}
        feature_rank = {
            value: index for index, value in enumerate(heatmap_feature_labels)
        }
        diagnosis_rank = {
            value: index for index, value in enumerate(heatmap_group_order)
        }
        component_order = active_component_order if resolved else ["CT", "TS"]
        component_rank = {value: index for index, value in enumerate(component_order)}
        row_order_frame = (
            heatmap_data[
                [
                    "module",
                    "metric_family",
                    "component",
                    "diagnosis_group",
                    "heatmap_row",
                ]
            ]
            .drop_duplicates()
            .assign(
                module_rank=lambda frame: frame["module"].map(module_rank),
                feature_rank=lambda frame: frame["metric_family"].map(feature_rank),
                component_rank=lambda frame: frame["component"].map(component_rank),
                diagnosis_rank=lambda frame: frame["diagnosis_group"].map(
                    diagnosis_rank
                ),
            )
            .sort_values(
                ["module_rank", "feature_rank", "component_rank", "diagnosis_rank"]
            )
        )
        row_order = row_order_frame["heatmap_row"].tolist()
        if cluster_modules:
            row_group_labels = dict(
                row_order_frame[["heatmap_row", "module"]]
                .assign(
                    module=lambda frame: frame["module"].map(
                        lambda value: f"M{int(value)}"
                    )
                )
                .itertuples(index=False, name=None)
            )
        feature_title = (
            f"all {len(heatmap_feature_labels)} available module scores"
            if all_feature == "__all__"
            else heatmap_feature_labels[all_feature]
        )
        if all_component == "__all__":
            component_title = (
                f"all {len(component_options)} tissue-resolved components"
                if resolved
                else "both aggregate components (CT and TS)"
            )
        else:
            component_title = (
                component_labels[all_component]
                if resolved
                else f"{all_component} aggregate"
            )
        heatmap_title = (
            f"{module_set_label}: {feature_title} · "
            f"{component_title} · {heatmap_context_label}"
        )
        fdr_column = (
            "pearson_fdr_across_modules"
            if correlation_method == "Pearson"
            else "spearman_fdr_across_modules"
        )
        fdr_scope = (
            "FDR is Benjamini–Hochberg correction across only the eligible tested "
            f"modules from the {module_count}-module definition, separately for each "
            "fixed feature, component, outcome, grouping "
            "variable and grouping level, and Pearson/Spearman method. Category levels "
            "are never combined into a groups-times-modules family. Missing or constant "
            "tests and structurally unavailable tissue eigengenes are excluded; the "
            "complete table reports the resulting family size."
        )

    available_heatmap_rows = int(heatmap_data["heatmap_row"].nunique())
    heatmap_row_view = st.radio(
        "Rows displayed in heatmap",
        options=[
            "All rows",
            "At least one FDR < 0.05",
            "Top K by absolute correlation",
        ],
        horizontal=True,
        key=(
            "correlation_heatmap_row_view_selected_module"
            if heatmap_mode.startswith("Selected module")
            else "correlation_heatmap_row_view_all_modules"
        ),
        help=(
            "Significant rows require at least one displayed outcome with FDR < 0.05. "
            "Top K ranks score rows by their strongest absolute selected-method "
            "correlation across the displayed outcomes."
        ),
    )
    table_data = correlation_table.copy()
    if heatmap_row_view == "At least one FDR < 0.05":
        significant_rows = set(
            heatmap_data.loc[
                pd.to_numeric(heatmap_data[fdr_column], errors="coerce").lt(0.05),
                "heatmap_row",
            ]
        )
        heatmap_data = heatmap_data.loc[
            heatmap_data["heatmap_row"].isin(significant_rows)
        ].copy()
        row_order = [row for row in row_order if row in significant_rows]
        table_data = table_data.loc[
            pd.to_numeric(table_data[fdr_column], errors="coerce").lt(0.05)
        ].copy()
        if row_group_labels:
            row_group_labels = {
                row: group
                for row, group in row_group_labels.items()
                if row in significant_rows
            }
    elif heatmap_row_view == "Top K by absolute correlation":
        top_k = int(
            st.number_input(
                "Number of top heatmap rows (K)",
                min_value=1,
                max_value=max(1, available_heatmap_rows),
                value=min(50, max(1, available_heatmap_rows)),
                step=1,
                key=(
                    "correlation_heatmap_top_k_selected_module"
                    if heatmap_mode.startswith("Selected module")
                    else "correlation_heatmap_top_k_all_modules"
                ),
                help=(
                    "Ranking uses the maximum absolute correlation for each displayed "
                    "score row across only the selected outcome columns."
                ),
            )
        )
        strongest_rows = (
            heatmap_data.assign(
                _absolute_correlation=pd.to_numeric(
                    heatmap_data[value_column], errors="coerce"
                ).abs()
            )
            .groupby("heatmap_row", observed=True, as_index=False)[
                "_absolute_correlation"
            ]
            .max()
            .sort_values(
                ["_absolute_correlation", "heatmap_row"],
                ascending=[False, True],
                na_position="last",
            )
            .head(top_k)["heatmap_row"]
            .tolist()
        )
        strongest_set = set(strongest_rows)
        heatmap_data = heatmap_data.loc[
            heatmap_data["heatmap_row"].isin(strongest_set)
        ].copy()
        row_order = [row for row in row_order if row in strongest_set]
        if row_group_labels:
            row_group_labels = {
                row: group
                for row, group in row_group_labels.items()
                if row in strongest_set
            }
        st.caption(
            f"Showing the top {len(strongest_set)} of {available_heatmap_rows} available "
            "heatmap rows. The complete correlation table remains exhaustive."
        )

    correlation_table = correlation_table.copy()
    correlation_table.insert(0, "module_definition", module_set_label)
    table_data = table_data.copy()
    table_data.insert(0, "module_definition", module_set_label)
    table_data["correlation_method"] = correlation_method
    table_data["correlation"] = pd.to_numeric(table_data[value_column], errors="coerce")
    table_data["absolute_correlation"] = table_data["correlation"].abs()
    table_data["p_value"] = pd.to_numeric(table_data[p_column], errors="coerce")
    table_data["fdr"] = pd.to_numeric(table_data[fdr_column], errors="coerce")
    table_data = table_data.sort_values(
        ["absolute_correlation", "module", "metric_family", "component"],
        ascending=[False, True, True, True],
        na_position="last",
    )

    if heatmap_data.empty or not row_order:
        st.info("No correlations meet the selected heatmap filters.")
    else:
        if cluster_rows and len(row_order) > 1000:
            st.warning(
                "Row clustering is disabled above 1,000 rows to avoid a large pairwise-distance "
                "matrix. Choose a Top-row view or use Modules clustering."
            )
            cluster_rows = False
        correlation_heatmap = correlation_heatmap_figure(
            heatmap_data,
            value_column=value_column,
            p_column=p_column,
            fdr_column=fdr_column,
            title=heatmap_title,
            row_order=row_order,
            cluster_rows=cluster_rows,
            cluster_columns=cluster_columns,
            row_group_labels=row_group_labels,
            significance_threshold=0.05,
        )
        render_plotly_chart(
            correlation_heatmap,
            use_container_width=True,
            config={
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": (
                        f"{download_prefix}{method}_{heatmap_file_group}_"
                        f"{correlation_method.lower()}_"
                        "correlation_heatmap"
                    ),
                    "scale": 3,
                },
            },
            plot_id="main.correlation_heatmap",
            metadata={
                "module_set": module_set,
                "method": method,
                "module": f"M{module}",
                "outcome": phenotype,
                "feature": feature,
            },
        )
    if heatmap_clustering != "None":
        st.caption(
            "Display order uses average-linkage hierarchical clustering with Euclidean "
            "distance. Missing coefficients are treated as zero only when determining "
            "the order; displayed correlations, p-values, FDRs, and downloads are unchanged."
        )
    st.caption(fdr_scope)
    st.caption("* indicates selected-method FDR < 0.05 for that heatmap cell.")
    correlation_columns = [
        "module_definition",
        "module",
        "feature_label",
        "metric_family",
        "component",
        "component_label",
        "diagnosis_group",
        "grouping_variable",
        "grouping_level",
        "grouping_label",
        "outcome",
        "outcome_label",
        "n",
        "minimum_group_n",
        "eligible",
        "unavailable_reason",
        "correlation_method",
        "correlation",
        "absolute_correlation",
        "p_value",
        "fdr",
        "pearson_r",
        "pearson_p",
        "pearson_fdr_across_modules",
        "pearson_fdr_module_family_n",
        "spearman_rho",
        "spearman_p",
        "spearman_fdr_across_modules",
        "spearman_fdr_module_family_n",
    ]
    correlation_columns = [
        column for column in correlation_columns if column in table_data.columns
    ]
    with st.expander("Complete correlation table", expanded=False):
        filterable_dataframe(
            table_data[correlation_columns],
            table_key="complete_correlation_table",
            table_name="Complete correlation table",
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download complete correlation table (TSV)",
            data=dataframe_to_tsv_bytes(table_data[correlation_columns]),
            file_name=(
                f"{download_prefix}{method}_{heatmap_file_group}_"
                "network_outcome_correlations.tsv"
            ),
            mime="text/tab-separated-values",
        )


def render_ct_ts(context):
    active_view = context["active_view"]
    analysis_subset = context["analysis_subset"]
    annotations = context["annotations"]
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
    module_count = context["module_count"]
    module_manifest = context["module_manifest"]
    module_set = context["module_set"]
    module_set_label = context["module_set_label"]
    phenotype = context["phenotype"]
    scale = context["scale"]
    score_normalization = context["score_normalization"]
    if cohort_scope == VALIDATION_SCOPE and active_view in {
        "CT–TS screen",
        "Module finder",
    }:
        from app_helpers.camel_validation_ui import render_component_screen

        render_component_screen(
            phenotype,
            feature,
            scale,
            differential_edge_rule,
            score_normalization,
            tuple(diagnoses),
            correlation_method,
            render_plotly_chart,
            module_set=module_set,
            network_method=method,
        )
        st.stop()

    st.subheader("Descriptive CT–TS pattern screen")
    st.caption(
        f"Ranks all {module_count} modules for the selected phenotype, feature, diagnosis, "
        "and method by "
        + (
            "the absolute difference between CT and TS categorical epsilon-squared effects. "
            if phenotype in {"clusters", "study"}
            else f"the absolute difference between CT and TS {correlation_method} correlations. "
        )
        + "This is an exploratory screen, not an independent validation ranking."
    )
    screen_diagnosis = st.selectbox(
        "Diagnosis for screen",
        options=diagnoses,
        index=diagnoses.index("AD") if "AD" in diagnoses else 0,
        key="screen_diagnosis",
    )
    if (
        phenotype == "study"
        and cached_sample_metadata(module_set, cohort_scope)["study"].dropna().nunique()
        < 2
    ):
        st.info(
            "Between-study CT–TS effects are unavailable: this cohort contains fewer than two observed studies (the motor cohort is MAP-only)."
        )
        st.stop()
    if phenotype in {"clusters", "study"}:
        cluster_screen = cached_module_set_cluster_associations(
            module_set,
            module_count,
            estimator,
            method,
            False,
            feature,
            ("CT", "TS"),
            (screen_diagnosis,),
            False,
            edge_rule,
            differential_edge_rule,
            differential_fdr_scope,
            differential_fdr_threshold,
            score_normalization,
            analysis_subset,
            cohort_scope,
            category_variable=phenotype,
        )
        wide = cluster_screen.pivot_table(
            index="module",
            columns="component",
            values=[
                "epsilon_squared",
                "categorical_p",
                "categorical_fdr_across_modules",
            ],
            aggfunc="first",
            dropna=False,
        )
        wide.columns = [f"{value}_{component}" for value, component in wide.columns]
        screen = wide.reset_index()
        screen["delta_epsilon_squared_CT_minus_TS"] = (
            screen["epsilon_squared_CT"] - screen["epsilon_squared_TS"]
        )
        screen["abs_delta_epsilon_squared"] = screen[
            "delta_epsilon_squared_CT_minus_TS"
        ].abs()
        screen = screen.merge(
            annotations[
                [
                    "module",
                    "displayed_category",
                    "displayed_subcategory",
                    "displayed_pathway",
                    "displayed_fdr",
                ]
            ],
            on="module",
            how="left",
            validate="one_to_one",
        ).sort_values("abs_delta_epsilon_squared", ascending=False, na_position="last")
        screen.insert(0, "screen_rank", range(1, len(screen) + 1))
        screen.insert(1, "module_definition", module_set_label)
        n_screen = st.slider(
            "Rows to show", 10, module_count, min(30, module_count), 10
        )
        st.warning(
            "CT−TS epsilon-squared differences are descriptive effect-size contrasts. "
            "No permutation p-value or FDR is assigned to the difference itself; the table "
            "shows each component's omnibus p-value and across-module FDR."
        )
        filterable_dataframe(
            screen.head(n_screen),
            table_key="cluster_ct_ts_screen",
            table_name=f"{ASSOCIATION_OUTCOME_LABELS[phenotype]} CT–TS screen",
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            f"Download complete {module_count}-module {phenotype} screen (TSV)",
            data=dataframe_to_tsv_bytes(screen),
            file_name=f"{download_prefix}{method}_{screen_diagnosis}_{phenotype}_{feature}_CT_TS_screen.tsv",
            mime="text/tab-separated-values",
        )
        st.stop()
    screen = cached_aggregate_stats(
        module_set,
        estimator,
        method,
        None,
        phenotype,
        feature,
        edge_rule,
        differential_edge_rule,
        differential_fdr_scope,
        differential_fdr_threshold,
        score_normalization,
        analysis_subset,
    )
    screen = screen.loc[screen["diagnosis_group"].eq(screen_diagnosis)].copy()
    if correlation_method == "Spearman":
        screen["correlation_CT"] = screen["rho_CT"]
        screen["p_CT"] = screen["p_spearman_CT"]
        screen["fdr_CT_global"] = screen["q_spearman_CT_all12_global"]
        screen["correlation_TS"] = screen["rho_TS"]
        screen["p_TS"] = screen["p_spearman_TS"]
        screen["fdr_TS_global"] = screen["q_spearman_TS_all12_global"]
        screen["ct_ts_test_statistic"] = screen["component_t_rank"]
        screen["p_CT_vs_TS"] = screen["p_component_rank"]
        screen["fdr_CT_vs_TS_within_phenotype"] = screen[
            "q_component_rank_within_phenotype"
        ]
        screen["fdr_CT_vs_TS_global"] = screen["q_component_rank_all12_global"]
    else:
        screen["correlation_CT"] = screen["r_rint_CT"]
        screen["p_CT"] = screen["p_rint_CT"]
        screen["fdr_CT_global"] = screen["q_rint_CT_all12_global"]
        screen["correlation_TS"] = screen["r_rint_TS"]
        screen["p_TS"] = screen["p_rint_TS"]
        screen["fdr_TS_global"] = screen["q_rint_TS_all12_global"]
        screen["ct_ts_test_statistic"] = screen["component_t_rint"]
        screen["p_CT_vs_TS"] = screen["p_component_rint"]
        screen["fdr_CT_vs_TS_within_phenotype"] = screen[
            "q_component_rint_within_phenotype"
        ]
        screen["fdr_CT_vs_TS_global"] = screen["q_component_rint_all12_global"]
    screen["correlation_method"] = correlation_method
    if phenotype in PATHOLOGY_OUTCOMES:
        family = "spearman" if correlation_method == "Spearman" else "rint"
        test = "rank" if correlation_method == "Spearman" else "rint"
        for component in ("CT", "TS"):
            screen[f"fdr_{component}_global"] = screen[
                f"q_{family}_{component}_across_modules"
            ]
        screen["fdr_CT_vs_TS_global"] = screen[f"q_component_{test}_across_modules"]
        st.caption(
            "Pathology FDRs are BH across valid modules only, separately for this outcome and donor group."
        )
    screen["delta_correlation_CT_minus_TS"] = (
        screen["correlation_CT"] - screen["correlation_TS"]
    )
    screen["abs_delta_correlation"] = screen["delta_correlation_CT_minus_TS"].abs()
    paired_correlations = (
        screen["correlation_CT"].notna() & screen["correlation_TS"].notna()
    )
    screen["opposite_CT_TS_sign"] = pd.Series(
        pd.NA, index=screen.index, dtype="boolean"
    )
    screen.loc[paired_correlations, "opposite_CT_TS_sign"] = np.sign(
        screen.loc[paired_correlations, "correlation_CT"]
    ) != np.sign(screen.loc[paired_correlations, "correlation_TS"])
    screen["max_abs_correlation"] = (
        screen[["correlation_CT", "correlation_TS"]].abs().max(axis=1)
    )
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
        ["abs_delta_correlation", "max_abs_correlation"],
        ascending=[False, False],
        na_position="last",
    )
    screen.insert(0, "screen_rank", np.arange(1, len(screen) + 1))
    screen.insert(1, "module_definition", module_set_label)
    n_screen = st.slider("Rows to show", 10, module_count, min(30, module_count), 10)
    screen_columns = [
        "screen_rank",
        "module_definition",
        "module",
        "correlation_method",
        "correlation_CT",
        "p_CT",
        "correlation_TS",
        "p_TS",
        "delta_correlation_CT_minus_TS",
        "abs_delta_correlation",
        "opposite_CT_TS_sign",
        "ct_ts_test_statistic",
        "p_CT_vs_TS",
        "displayed_category",
        "displayed_subcategory",
        "displayed_pathway",
        "displayed_fdr",
    ]
    ct_p_index = screen_columns.index("p_CT") + 1
    screen_columns[ct_p_index:ct_p_index] = ["fdr_CT_global"]
    ts_p_index = screen_columns.index("p_TS") + 1
    screen_columns[ts_p_index:ts_p_index] = ["fdr_TS_global"]
    ct_ts_p_index = screen_columns.index("p_CT_vs_TS") + 1
    screen_columns[ct_ts_p_index:ct_ts_p_index] = [
        "fdr_CT_vs_TS_within_phenotype",
        "fdr_CT_vs_TS_global",
    ]
    filterable_dataframe(
        screen[screen_columns].head(n_screen),
        table_key="ct_ts_screen",
        table_name="CT–TS screen",
        use_container_width=True,
        hide_index=True,
        column_config={"module": st.column_config.NumberColumn(format="M%d")},
    )
    st.download_button(
        f"Download complete {module_count}-module screen (TSV)",
        data=dataframe_to_tsv_bytes(screen[screen_columns]),
        file_name=(
            f"{download_prefix}{method}_{screen_diagnosis}_{phenotype}_{feature}_"
            f"{correlation_method.lower()}_CT_TS_screen.tsv"
        ),
        mime="text/tab-separated-values",
    )
    if correlation_method == "Spearman":
        st.caption(
            "Spearman and rank-based CT-vs-TS FDR values use the expanded all-12-outcome "
            "Benjamini–Hochberg families for the selected module definition, estimator, "
            "network method, component family, and BONOBO edge rule."
        )
    st.info(fdr_text(module_manifest, module_count))


def render_finder(context):
    from app_helpers.app_shell import open_module

    active_feature_labels = context["active_feature_labels"]
    active_view = context["active_view"]
    analysis_subset = context["analysis_subset"]
    annotations = context["annotations"]
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
    module_details = context["module_details"]
    module_set = context["module_set"]
    module_set_label = context["module_set_label"]
    phenotype = context["phenotype"]
    scale = context["scale"]
    score_normalization = context["score_normalization"]
    if cohort_scope == VALIDATION_SCOPE and active_view in {
        "CT–TS screen",
        "Module finder",
    }:
        from app_helpers.camel_validation_ui import render_component_screen

        render_component_screen(
            phenotype,
            feature,
            scale,
            differential_edge_rule,
            score_normalization,
            tuple(diagnoses),
            correlation_method,
            render_plotly_chart,
            module_set=module_set,
            network_method=method,
        )
        st.stop()

    st.subheader("Find modules with contrasting association patterns")
    st.caption(
        "Categorical outcomes are ranked by descriptive CT–TS and Control–AD differences in Kruskal epsilon-squared. Omnibus p-values/FDRs belong to each underlying component, not to the effect-size difference."
        if phenotype in {"clusters", "study"}
        else "Rank modules by the difference between CT and TS correlations across all donors "
        "or within one diagnosis, by the difference between Control and AD correlations, "
        "or by both. "
        "All comparisons use the selected phenotype, feature, estimator, network method, "
        "edge rule, and Spearman/Pearson setting. These are association-pattern "
        "differences; they do not test a raw mean difference in the module score."
    )
    if (
        phenotype == "study"
        and cached_sample_metadata(module_set, cohort_scope)["study"].dropna().nunique()
        < 2
    ):
        st.info(
            "Between-study module ranking is unavailable: this cohort contains fewer than two observed studies (the motor cohort is MAP-only)."
        )
        st.stop()
    if phenotype in {"clusters", "study"}:
        cluster_finder_cohort = st.selectbox(
            "Cohort for nominal CT–TS comparison",
            ["All donors", *DIAGNOSIS_ORDER],
            index=0,
        )
        cluster_finder_stats = cached_module_set_cluster_associations(
            module_set,
            module_count,
            estimator,
            method,
            False,
            feature,
            ("CT", "TS"),
            tuple(DIAGNOSIS_ORDER),
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
        effect = cluster_finder_stats.pivot_table(
            index="module",
            columns=["diagnosis_group", "component"],
            values="epsilon_squared",
            aggfunc="first",
            dropna=False,
        )
        effect.columns = [
            f"epsilon_squared_{group}_{component}"
            for group, component in effect.columns
        ]
        fdr = cluster_finder_stats.pivot_table(
            index="module",
            columns=["diagnosis_group", "component"],
            values="categorical_fdr_across_modules",
            aggfunc="first",
            dropna=False,
        )
        fdr.columns = [f"fdr_{group}_{component}" for group, component in fdr.columns]
        finder = effect.join(fdr).reset_index()
        finder["ct_ts_delta_epsilon_squared"] = (
            finder[f"epsilon_squared_{cluster_finder_cohort}_CT"]
            - finder[f"epsilon_squared_{cluster_finder_cohort}_TS"]
        )
        finder["ct_ts_abs_delta_epsilon_squared"] = finder[
            "ct_ts_delta_epsilon_squared"
        ].abs()
        for component in ("CT", "TS"):
            finder[f"ad_control_delta_epsilon_squared_{component}"] = (
                finder[f"epsilon_squared_AD_{component}"]
                - finder[f"epsilon_squared_Control_{component}"]
            )
        finder["ad_control_max_abs_delta_epsilon_squared"] = (
            finder[
                [
                    "ad_control_delta_epsilon_squared_CT",
                    "ad_control_delta_epsilon_squared_TS",
                ]
            ]
            .abs()
            .max(axis=1)
        )
        finder["combined_descriptive_score"] = (
            finder["ct_ts_abs_delta_epsilon_squared"]
            + finder["ad_control_max_abs_delta_epsilon_squared"]
        )
        cluster_finder_criterion = st.radio(
            "Ranking criterion",
            ["CT–TS difference", "Control–AD difference", "Both"],
            horizontal=True,
        )
        sort_column = {
            "CT–TS difference": "ct_ts_abs_delta_epsilon_squared",
            "Control–AD difference": "ad_control_max_abs_delta_epsilon_squared",
            "Both": "combined_descriptive_score",
        }[cluster_finder_criterion]
        finder = finder.merge(
            annotations[
                [
                    "module",
                    "displayed_category",
                    "displayed_subcategory",
                    "displayed_pathway",
                    "displayed_fdr",
                ]
            ],
            on="module",
            how="left",
            validate="one_to_one",
        ).sort_values(sort_column, ascending=False, na_position="last")
        finder.insert(0, "finder_rank", range(1, len(finder) + 1))
        finder.insert(1, "module_definition", module_set_label)
        finder_rows = st.slider(
            "Top modules", 10, module_count, min(30, module_count), 10
        )
        st.warning(
            "These epsilon-squared differences are descriptive. FDR columns belong to the "
            "underlying component omnibus tests, not to a formal test of the difference."
        )
        filterable_dataframe(
            finder.head(finder_rows),
            table_key="cluster_module_finder",
            table_name=f"{ASSOCIATION_OUTCOME_LABELS[phenotype]} module finder",
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            f"Download complete {module_count}-module {phenotype} finder (TSV)",
            data=dataframe_to_tsv_bytes(finder),
            file_name=f"{download_prefix}{phenotype}_{feature}_module_finder.tsv",
            mime="text/tab-separated-values",
        )
        st.stop()
    finder_statistics = cached_aggregate_stats(
        module_set,
        estimator,
        method,
        None,
        phenotype,
        feature,
        edge_rule,
        differential_edge_rule,
        differential_fdr_scope,
        differential_fdr_threshold,
        score_normalization,
        analysis_subset,
    )
    available_finder_diagnoses = [
        cohort
        for cohort in DIAGNOSIS_ORDER
        if cohort in set(finder_statistics["diagnosis_group"].dropna().astype(str))
    ]
    if not {"Control", "AD"}.issubset(available_finder_diagnoses):
        st.warning(
            "This evaluation cohort does not contain both Control and AD association "
            "statistics. Choose All donors, Discovery AD/Control, or Held-out validation "
            "AD/Control in the sidebar."
        )
        st.stop()

    finder_controls = st.columns([1.2, 1.0, 1.0, 0.8])
    with finder_controls[0]:
        finder_criterion = st.radio(
            "Ranking criterion",
            options=list(FINDER_CRITERIA),
            format_func=lambda value: FINDER_CRITERIA[value],
            horizontal=True,
        )
    with finder_controls[1]:
        finder_ct_ts_diagnosis = st.selectbox(
            "Cohort for CT–TS comparison",
            options=["All donors", *available_finder_diagnoses],
            index=0,
            help=(
                "All donors pools Control, MCI, and AD donors for the phenotype–score "
                "correlations. Diagnosis choices calculate the same CT–TS contrast "
                "within the selected group only."
            ),
        )
    with finder_controls[2]:
        finder_fdr_filter = st.selectbox(
            "Association-difference FDR filter",
            options=[None, 0.10, 0.05],
            format_func=lambda value: (
                "No FDR filter" if value is None else f"FDR ≤ {value:.2f}"
            ),
            help=(
                "For Both criteria, the selected cutoff must be met by both the CT–TS "
                "test and at least one Control–AD component comparison."
            ),
        )
    with finder_controls[3]:
        finder_rows = st.slider(
            "Top modules",
            min_value=10,
            max_value=module_count,
            value=min(30, module_count),
            step=10,
        )

    finder_effect_controls = st.columns(2)
    with finder_effect_controls[0]:
        minimum_ct_ts_difference = st.slider(
            "Minimum absolute CT–TS Δ correlation",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.05,
            disabled=finder_criterion == "ad_control",
        )
    with finder_effect_controls[1]:
        minimum_ad_control_difference = st.slider(
            "Minimum absolute Control–AD Δ correlation",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.05,
            disabled=finder_criterion == "ct_ts",
        )

    if finder_ct_ts_diagnosis == "All donors":
        pooled_finder_statistics = cached_pooled_module_finder_stats(
            module_set,
            estimator,
            method,
            phenotype,
            feature,
            edge_rule,
            differential_edge_rule,
            differential_fdr_scope,
            differential_fdr_threshold,
            score_normalization,
            analysis_subset,
        )
        finder_statistics = pd.concat(
            [finder_statistics, pooled_finder_statistics], ignore_index=True
        )

    finder = build_module_finder_table(
        finder_statistics,
        ct_ts_diagnosis=finder_ct_ts_diagnosis,
        correlation_method=correlation_method,
        criterion=finder_criterion,
    )
    if phenotype in PATHOLOGY_OUTCOMES:
        finder["ct_ts_fdr_across_modules"] = finder["ct_ts_fdr_within_phenotype"]
        for component in ("CT", "TS"):
            finder[f"ad_control_fdr_{component}"] = benjamini_hochberg(
                finder[f"ad_control_p_value_{component}"]
            )
        finder["ad_control_min_fdr"] = finder[
            ["ad_control_fdr_CT", "ad_control_fdr_TS"]
        ].min(axis=1)
    finder = finder.merge(
        annotations[
            [
                "module",
                "displayed_category",
                "displayed_subcategory",
                "displayed_pathway",
                "displayed_fdr",
            ]
        ],
        on="module",
        how="left",
        validate="one_to_one",
    ).merge(
        module_details[
            [
                "module",
                "module_size",
                "cluster_type",
                "tissue_entropy_normalized",
            ]
        ],
        on="module",
        how="left",
        validate="one_to_one",
    )
    finder.insert(1, "module_definition", module_set_label)
    finder.insert(2, "phenotype", phenotype)
    finder.insert(3, "metric_family", feature)
    finder_filtered = finder.copy()
    if finder_criterion in {"ct_ts", "both"}:
        finder_filtered = finder_filtered.loc[
            finder_filtered["ct_ts_abs_delta_correlation"].ge(minimum_ct_ts_difference)
        ]
    if finder_criterion in {"ad_control", "both"}:
        finder_filtered = finder_filtered.loc[
            finder_filtered["ad_control_max_abs_delta_correlation"].ge(
                minimum_ad_control_difference
            )
        ]
    if finder_fdr_filter is not None:
        ct_ts_significant = finder_filtered["ct_ts_fdr_within_phenotype"].le(
            finder_fdr_filter
        )
        ad_control_significant = finder_filtered["ad_control_min_fdr"].le(
            finder_fdr_filter
        )
        if finder_criterion == "ct_ts":
            finder_filtered = finder_filtered.loc[ct_ts_significant]
        elif finder_criterion == "ad_control":
            finder_filtered = finder_filtered.loc[ad_control_significant]
        else:
            finder_filtered = finder_filtered.loc[
                ct_ts_significant & ad_control_significant
            ]
    finder_filtered = finder_filtered.sort_values(
        "finder_rank", kind="stable"
    ).reset_index(drop=True)
    finder_filtered.insert(0, "display_rank", np.arange(1, len(finder_filtered) + 1))

    if finder_filtered.empty:
        st.warning(
            "No modules meet the current effect-size and FDR filters. Relax one or more "
            "filters to restore the exploratory ranking."
        )
    else:
        top_finder = finder_filtered.iloc[0]
        finder_summary = st.columns(4)
        finder_summary[0].metric("Modules retained", f"{len(finder_filtered):,}")
        finder_summary[1].metric("Top module", f"M{int(top_finder['module'])}")
        finder_summary[2].metric(
            "Top CT–TS |Δr|",
            f"{top_finder['ct_ts_abs_delta_correlation']:.3f}",
        )
        finder_summary[3].metric(
            "Top Control–AD |Δr|",
            f"{top_finder['ad_control_max_abs_delta_correlation']:.3f}",
        )
        finder_figure = module_finder_figure(
            finder_filtered,
            phenotype_label=OUTCOME_LABELS[phenotype],
            feature_label=active_feature_labels[feature],
            correlation_method=correlation_method,
            ct_ts_diagnosis=finder_ct_ts_diagnosis,
            criterion_label=FINDER_CRITERIA[finder_criterion],
            selected_module=module,
            label_count=min(10, finder_rows),
            minimum_ct_ts_difference=minimum_ct_ts_difference,
            minimum_ad_control_difference=minimum_ad_control_difference,
        )
        render_plotly_chart(
            finder_figure,
            use_container_width=True,
            config={
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": f"{download_prefix}{phenotype}_{feature}_module_finder",
                    "scale": 3,
                },
            },
            plot_id="main.finder_figure",
            metadata={
                "module_set": module_set,
                "method": method,
                "module": f"M{module}",
                "outcome": phenotype,
                "feature": feature,
            },
        )
        finder_columns = [
            "display_rank",
            "module_definition",
            "module",
            "finder_criterion",
            "finder_score",
            "correlation_method",
            "ct_ts_diagnosis",
            "ct_ts_n",
            "ct_ts_correlation_CT",
            "ct_ts_correlation_TS",
            "ct_ts_delta_correlation",
            "ct_ts_abs_delta_correlation",
            "ct_ts_test_statistic",
            "ct_ts_p_value",
            "ct_ts_fdr_within_phenotype",
            (
                "ct_ts_fdr_across_modules"
                if phenotype in PATHOLOGY_OUTCOMES
                else "ct_ts_fdr_all12_global"
            ),
            "n_control",
            "n_ad",
            "correlation_CT_control",
            "correlation_CT_ad",
            "ad_control_delta_correlation_CT",
            "ad_control_fisher_z_CT",
            "ad_control_p_value_CT",
            "ad_control_fdr_CT",
            "correlation_TS_control",
            "correlation_TS_ad",
            "ad_control_delta_correlation_TS",
            "ad_control_fisher_z_TS",
            "ad_control_p_value_TS",
            "ad_control_fdr_TS",
            "ad_control_best_component",
            "ad_control_max_abs_delta_correlation",
            "ad_control_min_fdr",
            "module_size",
            "cluster_type",
            "tissue_entropy_normalized",
            "displayed_category",
            "displayed_subcategory",
            "displayed_pathway",
            "displayed_fdr",
        ]
        finder_columns = [
            column for column in finder_columns if column in finder_filtered.columns
        ]
        filterable_dataframe(
            finder_filtered[finder_columns].head(finder_rows),
            table_key="module_finder",
            table_name="Module finder ranking",
            on_row_select=lambda row: open_module(
                int(row["module"]), context, diagnosis=finder_ct_ts_diagnosis
            ),
            use_container_width=True,
            hide_index=True,
            height=560,
            column_config={
                "module": st.column_config.NumberColumn(format="M%d"),
                "finder_score": st.column_config.ProgressColumn(
                    "Ranking score", min_value=0.0, max_value=1.0, format="%.3f"
                ),
                "tissue_entropy_normalized": st.column_config.ProgressColumn(
                    "Normalized tissue entropy",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.3f",
                ),
            },
        )
        st.download_button(
            "Download current module-finder ranking (TSV)",
            data=dataframe_to_tsv_bytes(finder_filtered[finder_columns]),
            file_name=(
                f"{download_prefix}{phenotype}_{feature}_{correlation_method.lower()}_"
                f"{finder_criterion}_module_finder.tsv"
            ),
            mime="text/tab-separated-values",
        )
    st.info(
        "Pathology CT–TS FDR is BH across modules for this outcome and cohort. "
        "Control–AD contrasts use approximate Fisher-z tests with BH across modules "
        "separately for CT and TS. These are descriptive association-pattern screens."
        if phenotype in PATHOLOGY_OUTCOMES
        else "CT–TS FDR uses the existing dependent-component test corrected across modules "
        "within the selected phenotype. For All donors this test is calculated from the "
        "pooled donor rows; its legacy all-12-outcome FDR field remains unavailable. "
        "Control–AD differences use an approximate "
        "independent-groups Fisher-z test for CT and TS correlations, with BH across "
        f"both components and all {module_count} displayed modules. The Both score is "
        "the lower of the two percentile scores, so a module must rank well on both "
        "criteria rather than excelling on only one."
    )
    from app_helpers.app_shell import open_module

    inspect_rows = finder_filtered if "finder_filtered" in locals() else finder
    if not inspect_rows.empty:
        choice = st.selectbox(
            "Inspect ranked module",
            inspect_rows["module"].astype(int).tolist(),
            format_func=lambda value: f"M{value}",
            key="discover_inspect_module",
        )
        if st.button("Open module associations", key="discover_open_module"):
            open_module(choice, context, diagnosis=finder_ct_ts_diagnosis)
