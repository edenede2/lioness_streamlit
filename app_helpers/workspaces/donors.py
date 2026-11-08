from __future__ import annotations

from app_helpers.ui_runtime import (
    BONOBO_EDGE_RULE_LABELS,
    CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES,
    COLOR_LABELS,
    EDGE_COMPONENT_LABELS,
    EFFECT_DIRECTION_LABELS,
    ESTIMATOR_LABELS,
    FDR_SCOPE_LABELS,
    _edge_expression_color_kwargs,
    _edge_expression_mask,
    _expression_ols_fits,
    _render_expression_residuals,
    association_level_label,
    attach_metadata,
    cached_edge_expression,
    cached_edge_expression_candidates,
    cached_edge_expression_nodes,
    cached_edge_summaries,
    cached_effect_drivers,
    cached_kegg,
    cached_volcano_bins,
    cached_volcano_candidates,
    dataframe_to_tsv_bytes,
    donor_edge_composition_figure,
    donor_edge_difference_figure,
    donor_edge_driver_figure,
    donor_edge_scatter_figure,
    edge_expression_data_available,
    edge_summary_figure,
    edge_volcano_figure,
    effect_mask_column,
    effect_rule_label,
    endpoint_expression_2d_figure,
    endpoint_expression_3d_figure,
    filterable_dataframe,
    find_edge_triangles,
    gene_pathway_membership,
    np,
    pd,
    rank_edges,
    readable_method,
    render_plotly_chart,
    scope_edges,
    st,
    stateful_tabs,
    tab_is_open,
)


def render_summaries(context):
    analysis_subset = context["analysis_subset"]
    cohort_scope = context["cohort_scope"]
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
    module_set = context["module_set"]
    module_set_label = context["module_set_label"]
    phenotype = context["phenotype"]
    selected_split = context["selected_split"]
    st.subheader("Donor- and diagnosis-level edge summaries")
    st.caption(
        "Each undirected edge is counted once and the diagonal is excluded. LIONESS sign "
        "counts are recovered from the stored density and weight-sum identities and validated "
        "as integers. BONOBO counts are computed directly from the selected all-edge or "
        "significant-edge mask. Structurally unavailable scopes remain missing. These rows "
        "summarize each underlying network/edge rule once, so they do not change when the "
        "derived feature dropdown changes."
    )
    edge_data = attach_metadata(
        cached_edge_summaries(
            module_set,
            estimator,
            method,
            module,
            edge_rule,
            differential_edge_rule,
            differential_fdr_scope,
            differential_fdr_threshold,
            cohort_scope,
        ),
        module_set=module_set,
        cohort_scope=cohort_scope,
    )
    if differential_edge_rule != "all" and analysis_subset != "all_donors":
        edge_data = edge_data.loc[edge_data["ad_control_split"].eq(selected_split)]
    edge_data = edge_data.loc[edge_data["diagnosis_group"].isin(diagnoses)].copy()
    edge_data["retained_edge_fraction"] = pd.to_numeric(
        edge_data["n_retained_edges"], errors="coerce"
    ) / pd.to_numeric(edge_data["n_possible_edges"], errors="coerce")
    edge_scope_options = edge_data["scope"].drop_duplicates().tolist()
    selected_edge_scopes = st.multiselect(
        "Edge scopes",
        options=edge_scope_options,
        default=edge_scope_options,
        format_func=lambda value: edge_data.loc[
            edge_data["scope"].eq(value), "scope_label"
        ].iloc[0],
        key=f"edge_scopes_{module_set}_{estimator}_{method}_{edge_rule}",
    )
    edge_data = edge_data.loc[edge_data["scope"].isin(selected_edge_scopes)]
    edge_metric_labels = {
        "n_possible_edges": "Possible edges",
        "n_retained_edges": "Retained edges",
        "n_positive_edges": "Positive edges",
        "n_negative_edges": "Negative edges",
        "n_zero_edges": "Zero-weight edges",
        "n_pruned_edges": "Pruned edges",
        "signed_weight_sum": "Signed weight sum",
        "positive_weight_sum": "Positive weight sum",
        "negative_weight_magnitude": "Negative weight magnitude",
        "absolute_weight_sum": "Absolute weight sum",
        "retained_edge_fraction": "Retained-edge proportion",
    }
    for column, label in {
        "weight_mean": "Mean retained edge weight",
        "weight_sd": "Retained edge-weight SD",
        "weight_median": "Median retained edge weight",
        "cancellation_index": "Positive/negative cancellation index",
    }.items():
        if column in edge_data:
            edge_metric_labels[column] = label
    edge_metric = st.selectbox(
        "Edge summary metric",
        options=list(edge_metric_labels),
        format_func=lambda value: edge_metric_labels[value],
        key=f"edge_metric_{module_set}_{estimator}",
    )
    if edge_data.empty:
        st.info("No edge scopes remain for the current filters.")
    else:
        render_plotly_chart(
            edge_summary_figure(
                edge_data,
                edge_metric,
                edge_metric_labels[edge_metric],
                module,
                module_definition=module_set_label,
            ),
            use_container_width=True,
            config={"displaylogo": False},
            plot_id="main.edge_summary_figure",
            metadata={
                "module_set": module_set,
                "method": method,
                "module": f"M{module}",
                "outcome": phenotype,
                "feature": feature,
            },
        )
        edge_group_summary = (
            edge_data.groupby(
                ["scope", "scope_label", "diagnosis_group"], observed=True
            )[edge_metric]
            .agg(
                group_n="count",
                mean="mean",
                sd="std",
                median="median",
                q1=lambda values: values.quantile(0.25),
                q3=lambda values: values.quantile(0.75),
                minimum="min",
                maximum="max",
            )
            .reset_index()
        )
        edge_group_summary["iqr"] = edge_group_summary["q3"] - edge_group_summary["q1"]
        edge_group_summary.insert(0, "module_definition", module_set_label)
        edge_group_summary.insert(1, "estimator", ESTIMATOR_LABELS[estimator])
        edge_group_summary.insert(2, "network_method", readable_method(method))
        edge_group_summary.insert(3, "edge_rule", edge_rule)
        filterable_dataframe(
            edge_group_summary,
            table_key="edge_group_summary",
            table_name="Diagnosis-group edge summary",
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download diagnosis-group summary (TSV)",
            data=dataframe_to_tsv_bytes(edge_group_summary),
            file_name=f"{download_prefix}M{module}_{edge_metric}_group_summary.tsv",
            mime="text/tab-separated-values",
        )
        with st.expander("Donor-level edge-summary rows", expanded=False):
            filterable_dataframe(
                edge_data,
                table_key="donor_edge_summary",
                table_name="Donor-level edge summary",
                use_container_width=True,
                hide_index=True,
                height=480,
            )
            st.download_button(
                "Download donor-level edge summaries (TSV)",
                data=dataframe_to_tsv_bytes(edge_data),
                file_name=f"{download_prefix}M{module}_donor_edge_summaries.tsv",
                mime="text/tab-separated-values",
            )


def render_edge_explorer(context):
    active_feature_labels = context["active_feature_labels"]
    analysis_subset = context["analysis_subset"]
    cohort_scope = context["cohort_scope"]
    color_by = context["color_by"]
    component_labels = context["component_labels"]
    continuous_colorscale = context["continuous_colorscale"]
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
    module_set = context["module_set"]
    phenotype = context["phenotype"]
    plot_data = context["plot_data"]
    reverse_colorscale = context["reverse_colorscale"]
    selected_components = context["selected_components"]
    st.subheader("Donors with similar module scores and different phenotypes")
    st.caption(
        "Module connectivity is a one-dimensional signed sum. Two donors can have "
        "the same value even when different tissue components or positive and negative "
        "edges cancel. This view is descriptive and does not establish that an edge "
        "pattern caused the phenotype difference."
    )
    explorer_component = st.selectbox(
        "Comparison component",
        options=list(selected_components),
        format_func=lambda value: component_labels.get(value, value),
        key="donor_explorer_component",
    )
    candidates = plot_data.loc[
        plot_data["component"].eq(explorer_component),
        ["sample_id", "diagnosis_group", "metric_value", phenotype],
    ].drop_duplicates("sample_id")
    candidates["metric_value"] = pd.to_numeric(
        candidates["metric_value"], errors="coerce"
    )
    numeric_outcome = phenotype not in CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES
    if numeric_outcome:
        candidates[phenotype] = pd.to_numeric(candidates[phenotype], errors="coerce")
    candidates = candidates.dropna(subset=["metric_value", phenotype]).reset_index(
        drop=True
    )
    selection_mode = st.radio(
        "Donor selection",
        ["Find discordant pair", "Select manually"],
        horizontal=True,
    )
    selected_pair: tuple[str, str] | None = None
    pair_table = pd.DataFrame()
    if len(candidates) < 2:
        st.info(
            "At least two donors with non-missing score and phenotype are required."
        )
    elif selection_mode == "Select manually":
        donor_options = candidates["sample_id"].astype(str).tolist()
        pair = st.multiselect(
            "Pseudonymous donors",
            donor_options,
            default=donor_options[:2],
            max_selections=2,
        )
        if len(pair) == 2:
            selected_pair = (pair[0], pair[1])
    else:
        x = candidates["metric_value"].to_numpy(dtype=float)
        x_sd = float(np.nanstd(x, ddof=1)) or 1.0
        rows = []
        if numeric_outcome:
            y = candidates[phenotype].to_numpy(dtype=float)
            y_sd = float(np.nanstd(y, ddof=1)) or 1.0
            tolerance = st.slider(
                "Maximum score difference (SD)", 0.01, 0.50, 0.10, 0.01
            )
            phenotype_gap = st.slider(
                "Minimum phenotype difference (SD)", 0.0, 3.0, 1.0, 0.1
            )
        else:
            tolerance = st.slider(
                "Maximum score difference (SD)", 0.01, 0.50, 0.10, 0.01
            )
            phenotype_gap = 0.0
        for left in range(len(candidates) - 1):
            dx = np.abs(x[left + 1 :] - x[left]) / x_sd
            if numeric_outcome:
                dy = np.abs(y[left + 1 :] - y[left]) / y_sd
                eligible = (dx <= tolerance) & (dy >= phenotype_gap)
            else:
                dy = (
                    candidates.loc[left + 1 :, phenotype].to_numpy()
                    != candidates.loc[left, phenotype]
                ).astype(float)
                eligible = (dx <= tolerance) & dy.astype(bool)
            for offset in np.flatnonzero(eligible):
                right = left + 1 + int(offset)
                rows.append(
                    {
                        "sample_a": str(candidates.loc[left, "sample_id"]),
                        "sample_b": str(candidates.loc[right, "sample_id"]),
                        "score_gap_sd": float(dx[offset]),
                        "phenotype_gap_sd": (
                            float(dy[offset]) if numeric_outcome else np.nan
                        ),
                        "phenotype_a": candidates.loc[left, phenotype],
                        "phenotype_b": candidates.loc[right, phenotype],
                    }
                )
        pair_table = pd.DataFrame(rows)
        if pair_table.empty:
            st.info("No donor pair meets both thresholds; relax one of the filters.")
        else:
            pair_table = pair_table.sort_values(
                (
                    ["phenotype_gap_sd", "score_gap_sd"]
                    if numeric_outcome
                    else ["score_gap_sd"]
                ),
                ascending=[False, True] if numeric_outcome else [True],
                kind="stable",
            ).head(100)
            labels = [
                f"{row.sample_a} vs {row.sample_b} · score gap {row.score_gap_sd:.3f} SD"
                + (
                    f" · phenotype gap {row.phenotype_gap_sd:.2f} SD"
                    if numeric_outcome
                    else ""
                )
                for row in pair_table.itertuples(index=False)
            ]
            selected_label = st.selectbox("Candidate donor pair", labels)
            selected_row = pair_table.iloc[labels.index(selected_label)]
            selected_pair = (
                str(selected_row["sample_a"]),
                str(selected_row["sample_b"]),
            )

    explorer_scatter = donor_edge_scatter_figure(
        active_feature_labels,
        candidates,
        component_labels,
        explorer_component,
        feature,
        module,
        phenotype,
        selected_pair,
    )
    render_plotly_chart(
        explorer_scatter,
        key="donor_edge_explorer_scatter",
        metadata={
            "module_set": module_set,
            "method": method,
            "module": f"M{module}",
            "outcome": phenotype,
            "feature": feature,
        },
    )

    if selected_pair:
        edge_data = attach_metadata(
            cached_edge_summaries(
                module_set,
                estimator,
                method,
                module,
                edge_rule,
                differential_edge_rule,
                differential_fdr_scope,
                differential_fdr_threshold,
                cohort_scope,
            ),
            module_set=module_set,
            cohort_scope=cohort_scope,
        )
        edge_data = edge_data.loc[edge_data["sample_id"].isin(selected_pair)].copy()
        edge_data["retained_edge_fraction"] = pd.to_numeric(
            edge_data["n_retained_edges"], errors="coerce"
        ) / pd.to_numeric(edge_data["n_possible_edges"], errors="coerce")
        comparison_columns = [
            "sample_id",
            "diagnosis_group",
            "scope_label",
            "n_possible_edges",
            "n_retained_edges",
            "retained_edge_fraction",
            "n_positive_edges",
            "n_negative_edges",
            "signed_weight_sum",
            "positive_weight_sum",
            "negative_weight_magnitude",
            "absolute_weight_sum",
            "weight_mean",
            "weight_sd",
            "weight_q05",
            "weight_q25",
            "weight_median",
            "weight_q75",
            "weight_q95",
            "cancellation_index",
        ]
        comparison_columns = [
            column for column in comparison_columns if column in edge_data
        ]
        component_rows = edge_data.loc[~edge_data["scope"].eq("total")].copy()
        component_chart = component_rows.melt(
            id_vars=["sample_id", "scope_label"],
            value_vars=["positive_weight_sum", "negative_weight_magnitude"],
            var_name="edge_sign",
            value_name="weight_sum",
        )
        component_chart.loc[
            component_chart["edge_sign"].eq("negative_weight_magnitude"), "weight_sum"
        ] *= -1
        component_chart["edge_sign"] = component_chart["edge_sign"].map(
            {"positive_weight_sum": "Positive", "negative_weight_magnitude": "Negative"}
        )
        composition_figure = donor_edge_composition_figure(component_chart)
        render_plotly_chart(
            composition_figure,
            key="donor_edge_composition",
            metadata={
                "module_set": module_set,
                "method": method,
                "module": f"M{module}",
                "outcome": phenotype,
                "feature": feature,
            },
        )
        difference_metrics = [
            "signed_weight_sum",
            "positive_weight_sum",
            "negative_weight_magnitude",
            "absolute_weight_sum",
            "n_retained_edges",
            "cancellation_index",
        ]
        difference_metrics = [
            column for column in difference_metrics if column in component_rows.columns
        ]
        if difference_metrics:
            differences = component_rows.pivot_table(
                index="scope_label",
                columns="sample_id",
                values=difference_metrics,
                aggfunc="first",
            )
            if all(
                sample in differences.columns.get_level_values(1)
                for sample in selected_pair
            ):
                delta_rows = []
                for metric_name in difference_metrics:
                    delta = pd.to_numeric(
                        differences[(metric_name, selected_pair[1])], errors="coerce"
                    ) - pd.to_numeric(
                        differences[(metric_name, selected_pair[0])], errors="coerce"
                    )
                    delta_rows.append(
                        pd.DataFrame(
                            {
                                "scope_label": delta.index,
                                "metric": metric_name,
                                "difference": delta.to_numpy(),
                            }
                        )
                    )
                delta_frame = pd.concat(delta_rows, ignore_index=True)
                selected_difference_metric = st.selectbox(
                    f"Difference metric ({selected_pair[1]} minus {selected_pair[0]})",
                    difference_metrics,
                    format_func=lambda value: {
                        "signed_weight_sum": "Signed weight sum",
                        "positive_weight_sum": "Positive weight sum",
                        "negative_weight_magnitude": "Negative weight magnitude",
                        "absolute_weight_sum": "Absolute weight sum",
                        "n_retained_edges": "Retained edge count",
                        "cancellation_index": "Cancellation index",
                    }.get(value, value),
                )
                delta_plot = delta_frame.loc[
                    delta_frame["metric"].eq(selected_difference_metric)
                ]
                difference_figure = donor_edge_difference_figure(
                    delta_plot, selected_pair
                )
                render_plotly_chart(
                    difference_figure,
                    key="donor_edge_component_difference",
                    metadata={
                        "module_set": module_set,
                        "method": method,
                        "module": f"M{module}",
                        "outcome": phenotype,
                        "feature": feature,
                    },
                )
        filterable_dataframe(
            edge_data[comparison_columns],
            table_key="donor_edge_comparison",
            table_name="Selected-donor edge decomposition",
            height=460,
        )
        if differential_edge_rule.startswith("ad_control_discovery_effect__"):
            drivers = cached_effect_drivers(
                module_set,
                estimator,
                method,
                module,
                edge_rule,
                differential_edge_rule,
                tuple(selected_pair),
            )
            if drivers.empty:
                st.info(
                    "No retained gene-pair driver is available for this donor pair and mask."
                )
            else:
                drivers["edge"] = (
                    drivers["tissue_a"].astype(str)
                    + ":"
                    + drivers["gene_a"].astype(str)
                    + " ↔ "
                    + drivers["tissue_b"].astype(str)
                    + ":"
                    + drivers["gene_b"].astype(str)
                )
                driver_figure = donor_edge_driver_figure(drivers)
                render_plotly_chart(
                    driver_figure,
                    key="donor_edge_drivers",
                    metadata={
                        "module_set": module_set,
                        "method": method,
                        "module": f"M{module}",
                        "outcome": phenotype,
                        "feature": feature,
                    },
                )
                filterable_dataframe(
                    drivers,
                    table_key="donor_edge_driver_table",
                    table_name="Top gene-pair drivers",
                    height=520,
                )
                st.download_button(
                    "Download selected-donor gene-pair drivers (TSV)",
                    data=dataframe_to_tsv_bytes(drivers),
                    file_name=f"{download_prefix}M{module}_selected_donor_edge_drivers.tsv",
                    mime="text/tab-separated-values",
                )

    st.divider()
    st.subheader("Top-edge endpoint expression")
    st.caption(
        "Choose the top K exact edges in this module, then inspect the two endpoint "
        "genes or a genuine three-edge gene triangle. Axes are within-gene expression "
        "Z-scores across the 450 complete-tissue donors. OLS describes co-expression "
        "between axes; it does not model the selected phenotype or establish causality."
    )
    if not edge_expression_data_available(module_set, module):
        st.info(
            "Endpoint-expression data are not available for this module definition in "
            "the deployed bundle yet. The existing donor-decomposition tools above remain available."
        )
    else:
        exact_edges = cached_edge_expression_candidates(
            module_set, estimator, method, module
        )
        exact_edges = scope_edges(exact_edges, explorer_component)
        exact_edges = _edge_expression_mask(
            exact_edges,
            differential_edge_rule,
            differential_fdr_scope,
            differential_fdr_threshold,
        )
        if exact_edges.empty:
            st.info(
                "No exact edge row remains under the selected component and edge mask. "
                "Try All edges, a less restrictive mask, or another component."
            )
        else:
            rank_left, rank_middle, rank_right = st.columns(3)
            with rank_left:
                expression_rank_by = st.selectbox(
                    "Rank top edges by",
                    options=[
                        "absolute_hedges_g",
                        "absolute_mean_difference",
                        "global_fdr",
                        "per_module_fdr",
                    ],
                    format_func=lambda value: {
                        "absolute_hedges_g": "Absolute discovery Hedges’ g",
                        "absolute_mean_difference": "Absolute AD−Control mean difference",
                        "global_fdr": "Smallest discovery global FDR",
                        "per_module_fdr": "Smallest discovery per-module FDR",
                    }[value],
                )
            with rank_middle:
                expression_rank_direction = st.selectbox(
                    "Ranked-edge direction",
                    options=["either", "ad_higher", "control_higher"],
                    format_func=lambda value: EFFECT_DIRECTION_LABELS[value],
                )
            with rank_right:
                maximum_k = min(500, len(exact_edges))
                top_k = int(
                    st.number_input(
                        "Top K edges",
                        min_value=1,
                        max_value=max(1, maximum_k),
                        value=min(25, maximum_k),
                        step=1,
                    )
                )
            top_edges = rank_edges(
                exact_edges,
                expression_rank_by,
                top_k,
                direction=expression_rank_direction,
            )
            st.caption(
                f"Showing {len(top_edges):,} ranked edges from {len(exact_edges):,} "
                "available exact rows in the selected scope. The exact catalog contains "
                "all supported effect-mask edges plus significance-oriented candidates; "
                "it is not a dump of every structural edge."
            )
            ranked_columns = [
                "edge_rank",
                "edge_label",
                "component",
                "discovery_hedges_g",
                "discovery_mean_difference",
                "discovery_p_value",
                "discovery_fdr_global",
                "discovery_fdr_per_module",
                "validation_hedges_g",
                "validation_fdr_global",
                "validation_fdr_per_module",
                "validation_direction_concordant",
            ]
            filterable_dataframe(
                top_edges[[column for column in ranked_columns if column in top_edges]],
                table_key="top_endpoint_expression_edges",
                table_name="Top K exact edges",
                height=390,
            )

            node_metadata = cached_edge_expression_nodes(module_set, module)
            node_lookup = node_metadata.set_index("node_index")
            endpoint_tabs = stateful_tabs(
                ["2D edge", "3D gene triangle"], key="edge_expression_tabs"
            )
            with endpoint_tabs[0]:
                if tab_is_open(endpoint_tabs[0]):
                    edge_options = top_edges["edge_index"].astype(int).tolist()
                    selected_edge_index = st.selectbox(
                        "Edge for 2D expression plot",
                        options=edge_options,
                        format_func=lambda value: (
                            f"#{int(top_edges.loc[top_edges['edge_index'].eq(value), 'edge_rank'].iloc[0])} · "
                            f"{top_edges.loc[top_edges['edge_index'].eq(value), 'edge_label'].iloc[0]}"
                        ),
                        key="endpoint_expression_2d_edge",
                    )
                    edge = top_edges.loc[
                        top_edges["edge_index"].eq(selected_edge_index)
                    ].iloc[0]
                    endpoint_nodes = [int(edge["row_index"]), int(edge["column_index"])]
                    endpoint_labels = {
                        int(edge["row_index"]): f"{edge['tissue_a']}:{edge['gene_a']}",
                        int(
                            edge["column_index"]
                        ): f"{edge['tissue_b']}:{edge['gene_b']}",
                    }
                    right_node = st.selectbox(
                        "Gene predicted by the 2D OLS line (Y axis)",
                        options=endpoint_nodes,
                        index=1,
                        format_func=lambda value: endpoint_labels[value],
                        key="endpoint_expression_2d_response",
                    )
                    left_node = next(
                        node for node in endpoint_nodes if node != right_node
                    )
                    left_column, right_column = (
                        f"node_{left_node}",
                        f"node_{right_node}",
                    )
                    left_label = endpoint_labels[left_node]
                    right_label = endpoint_labels[right_node]
                    expression_2d = cached_edge_expression(
                        module_set, module, tuple(sorted((left_node, right_node)))
                    )
                    expression_2d = attach_metadata(
                        expression_2d, module_set=module_set
                    )
                    expression_2d = expression_2d.loc[
                        expression_2d["diagnosis_group"].isin(diagnoses)
                    ].copy()
                    if (
                        differential_edge_rule != "all"
                        and analysis_subset != "all_donors"
                    ):
                        selected_split = {
                            "discovery_ad_control": "Discovery",
                            "validation_ad_control": "Validation",
                            "mci_external": "MCI_external",
                        }[analysis_subset]
                        expression_2d = expression_2d.loc[
                            expression_2d["ad_control_split"].eq(selected_split)
                        ].copy()
                    expression_2d = expression_2d.dropna(
                        subset=[left_column, right_column]
                    )
                    expression_fit_scope_2d = st.radio(
                        "OLS fit scope",
                        options=["pooled", "diagnosis"],
                        format_func=lambda value: {
                            "pooled": "All displayed donors (pooled)",
                            "diagnosis": "Separate diagnosis-group fits",
                        }[value],
                        horizontal=True,
                        key="endpoint_expression_2d_fit_scope",
                    )
                    fits_2d, fit_summary_2d, coefficients_2d = _expression_ols_fits(
                        expression_2d,
                        right_column,
                        [left_column],
                        expression_fit_scope_2d,
                    )
                    categorical_color = (
                        color_by in CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES
                    )
                    expression_2d["plot_color"] = (
                        expression_2d[color_by].map(
                            lambda value: association_level_label(color_by, value)
                        )
                        if categorical_color
                        else pd.to_numeric(expression_2d[color_by], errors="coerce")
                    )
                    hover_columns = [
                        column
                        for column in (
                            "sample_id",
                            "diagnosis_group",
                            "study",
                            phenotype,
                            "clusters",
                            "cogdx",
                            "braak_stage",
                            "cerad_score",
                            "adnc",
                            "parkinsonism",
                        )
                        if column in expression_2d
                    ]
                    point_figure_2d = endpoint_expression_2d_figure(
                        COLOR_LABELS,
                        _edge_expression_color_kwargs,
                        color_by,
                        continuous_colorscale,
                        expression_2d,
                        fits_2d,
                        hover_columns,
                        left_column,
                        left_label,
                        module,
                        reverse_colorscale,
                        right_column,
                        right_label,
                        selected_pair,
                    )
                    render_plotly_chart(
                        point_figure_2d,
                        key="endpoint_expression_2d_plot",
                        metadata={
                            "module_set": module_set,
                            "method": method,
                            "module": f"M{module}",
                            "outcome": phenotype,
                            "feature": feature,
                        },
                    )
                    _render_expression_residuals(fits_2d, key="endpoint_expression_2d")
                    if not fit_summary_2d.empty:
                        fit_tables_2d = st.columns(2)
                        with fit_tables_2d[0]:
                            filterable_dataframe(
                                fit_summary_2d,
                                table_key="endpoint_expression_2d_fit_summary",
                                table_name="2D OLS fit summary",
                            )
                        with fit_tables_2d[1]:
                            filterable_dataframe(
                                coefficients_2d,
                                table_key="endpoint_expression_2d_coefficients",
                                table_name="2D OLS coefficients",
                            )
                    significant_kegg_2d = st.checkbox(
                        "Show only module KEGG enrichments with FDR < 0.05",
                        value=True,
                        key="endpoint_expression_2d_significant_kegg",
                    )
                    predictor_metadata = node_lookup.loc[left_node]
                    response_metadata_2d = node_lookup.loc[right_node]
                    genes_2d = pd.DataFrame(
                        [
                            {
                                "role": "X axis / OLS predictor",
                                "gene_symbol": predictor_metadata["gene_symbol"],
                                "tissue": predictor_metadata["tissue"],
                            },
                            {
                                "role": "Y axis / OLS response",
                                "gene_symbol": response_metadata_2d["gene_symbol"],
                                "tissue": response_metadata_2d["tissue"],
                            },
                        ]
                    )
                    memberships_2d = gene_pathway_membership(
                        genes_2d,
                        cached_kegg(module_set, module),
                        significant_only=significant_kegg_2d,
                    )
                    filterable_dataframe(
                        memberships_2d,
                        table_key="endpoint_expression_2d_gene_kegg",
                        table_name="Selected genes and module KEGG memberships",
                        height=420,
                    )

            with endpoint_tabs[1]:
                if tab_is_open(endpoint_tabs[1]):
                    if not st.checkbox(
                        "Render the 3D triangle and OLS plane",
                        value=False,
                        key="endpoint_expression_render_3d",
                        help=(
                            "3D rendering is opt-in because Streamlit tabs otherwise execute "
                            "hidden charts and consume memory even while the 2D tab is open."
                        ),
                    ):
                        st.caption(
                            "Enable this option to search the current top-K graph for complete "
                            "three-edge triangles and render the interactive OLS plane."
                        )
                        st.stop()
                    triangles = find_edge_triangles(top_edges)
                    if triangles.empty:
                        st.info(
                            "No three-edge triangle exists among the current top K edges. "
                            "Increase K or select TS pooled/a tissue block. A single CT tissue "
                            "pair is bipartite, so it cannot contain a three-edge triangle."
                        )
                    else:
                        triangle_index = st.selectbox(
                            "Triangle for 3D expression plot",
                            options=list(range(len(triangles))),
                            format_func=lambda value: (
                                f"{triangles.loc[value, 'triangle_label']} · worst edge rank "
                                f"{int(triangles.loc[value, 'worst_edge_rank'])}"
                            ),
                            key="endpoint_expression_3d_triangle",
                        )
                        triangle = triangles.loc[triangle_index]
                        triangle_nodes = [
                            int(triangle["node_x"]),
                            int(triangle["node_y"]),
                            int(triangle["node_z"]),
                        ]
                        triangle_labels = {
                            int(triangle["node_x"]): str(triangle["gene_x"]),
                            int(triangle["node_y"]): str(triangle["gene_y"]),
                            int(triangle["node_z"]): str(triangle["gene_z"]),
                        }
                        response_node = st.selectbox(
                            "Gene predicted by the OLS plane (Z axis)",
                            options=triangle_nodes,
                            format_func=lambda value: triangle_labels[value],
                            key="endpoint_expression_3d_response",
                        )
                        predictor_nodes = [
                            node for node in triangle_nodes if node != response_node
                        ]
                        response_column = f"node_{response_node}"
                        predictor_columns = [f"node_{node}" for node in predictor_nodes]
                        expression_3d = cached_edge_expression(
                            module_set, module, tuple(sorted(triangle_nodes))
                        )
                        expression_3d = attach_metadata(
                            expression_3d, module_set=module_set
                        )
                        expression_3d = (
                            expression_3d.loc[
                                expression_3d["diagnosis_group"].isin(diagnoses)
                            ]
                            .dropna(subset=[response_column, *predictor_columns])
                            .copy()
                        )
                        if (
                            differential_edge_rule != "all"
                            and analysis_subset != "all_donors"
                        ):
                            selected_split = {
                                "discovery_ad_control": "Discovery",
                                "validation_ad_control": "Validation",
                                "mci_external": "MCI_external",
                            }[analysis_subset]
                            expression_3d = expression_3d.loc[
                                expression_3d["ad_control_split"].eq(selected_split)
                            ].copy()
                        expression_fit_scope_3d = st.radio(
                            "OLS plane scope",
                            options=["pooled", "diagnosis"],
                            format_func=lambda value: {
                                "pooled": "All displayed donors (pooled)",
                                "diagnosis": "Separate diagnosis-group planes",
                            }[value],
                            horizontal=True,
                            key="endpoint_expression_3d_fit_scope",
                        )
                        fits_3d, fit_summary_3d, coefficients_3d = _expression_ols_fits(
                            expression_3d,
                            response_column,
                            predictor_columns,
                            expression_fit_scope_3d,
                        )
                        categorical_color_3d = (
                            color_by in CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES
                        )
                        expression_3d["plot_color"] = (
                            expression_3d[color_by].map(
                                lambda value: association_level_label(color_by, value)
                            )
                            if categorical_color_3d
                            else pd.to_numeric(expression_3d[color_by], errors="coerce")
                        )
                        point_figure_3d = endpoint_expression_3d_figure(
                            COLOR_LABELS,
                            _edge_expression_color_kwargs,
                            color_by,
                            continuous_colorscale,
                            expression_3d,
                            fits_3d,
                            module,
                            phenotype,
                            predictor_columns,
                            predictor_nodes,
                            response_column,
                            response_node,
                            reverse_colorscale,
                            selected_pair,
                            triangle_labels,
                        )
                        render_plotly_chart(
                            point_figure_3d,
                            key="endpoint_expression_3d_plot",
                            metadata={
                                "module_set": module_set,
                                "method": method,
                                "module": f"M{module}",
                                "outcome": phenotype,
                                "feature": feature,
                            },
                        )
                        _render_expression_residuals(
                            fits_3d, key="endpoint_expression_3d"
                        )
                        if not fit_summary_3d.empty:
                            fit_tables_3d = st.columns(2)
                            with fit_tables_3d[0]:
                                filterable_dataframe(
                                    fit_summary_3d,
                                    table_key="endpoint_expression_3d_fit_summary",
                                    table_name="3D-plane OLS fit summary",
                                )
                            with fit_tables_3d[1]:
                                filterable_dataframe(
                                    coefficients_3d,
                                    table_key="endpoint_expression_3d_coefficients",
                                    table_name="3D-plane OLS coefficients",
                                )
                        significant_kegg_3d = st.checkbox(
                            "Show only triangle-gene KEGG enrichments with FDR < 0.05",
                            value=True,
                            key="endpoint_expression_3d_significant_kegg",
                        )
                        triangle_gene_rows = []
                        for node in predictor_nodes:
                            metadata_row = node_lookup.loc[node]
                            triangle_gene_rows.append(
                                {
                                    "role": "OLS predictor",
                                    "gene_symbol": metadata_row["gene_symbol"],
                                    "tissue": metadata_row["tissue"],
                                }
                            )
                        response_metadata = node_lookup.loc[response_node]
                        triangle_gene_rows.append(
                            {
                                "role": "Z axis / OLS response",
                                "gene_symbol": response_metadata["gene_symbol"],
                                "tissue": response_metadata["tissue"],
                            }
                        )
                        memberships_3d = gene_pathway_membership(
                            pd.DataFrame(triangle_gene_rows),
                            cached_kegg(module_set, module),
                            significant_only=significant_kegg_3d,
                        )
                        filterable_dataframe(
                            memberships_3d,
                            table_key="endpoint_expression_3d_gene_kegg",
                            table_name="Triangle genes and module KEGG memberships",
                            height=420,
                        )


def render_volcano(context):
    active_component_order = context["active_component_order"]
    differential_available = context["differential_available"]
    differential_edge_rule = context["differential_edge_rule"]
    differential_fdr_scope = context["differential_fdr_scope"]
    differential_fdr_threshold = context["differential_fdr_threshold"]
    download_prefix = context["download_prefix"]
    effect_direction = context["effect_direction"]
    estimator = context["estimator"]
    feature = context["feature"]
    method = context["method"]
    module = context["module"]
    module_set = context["module_set"]
    module_set_label = context["module_set_label"]
    phenotype = context["phenotype"]
    st.subheader("AD–Control differential edges")
    effect_volcano = differential_edge_rule.startswith("ad_control_discovery_effect__")
    if effect_volcano:
        st.caption(
            "Each exact point is one undirected module edge retained in the union of "
            "the supported effect masks. The active feature mask uses discovery-set "
            f"effect size rather than significance: **{effect_rule_label(differential_edge_rule)}**. "
            "Hedges’ g and mean difference are AD minus Control. The held-out validation "
            "statistics never change the discovery mask."
        )
    else:
        st.caption(
            "Each point is one undirected module edge. The mask is discovered with a "
            "two-sided Welch test in 117 AD and 114 Control donors; Hedges’ g and mean "
            "difference are AD minus Control. Both global and per-module BH values are "
            "stored for every edge; this view and the filtered-score mask use the "
            f"sidebar selection (**{FDR_SCOPE_LABELS[differential_fdr_scope]}**). The "
            "validation view uses the held-out 50 AD and 50 Control donors and never "
            "changes the discovery mask."
        )
    if not differential_available and not effect_volcano:
        st.info(
            "The differential-edge volcano bundle is not available in this deployment."
        )
    else:
        volcano_candidates = cached_volcano_candidates(
            module_set, estimator, method, module, differential_edge_rule
        )
        if effect_volcano and not differential_available:
            # An effect-only release keeps exact retained points without forcing
            # the much larger all-edge density/FDR bundle into Streamlit memory.
            volcano_bins = pd.DataFrame(
                columns=[
                    "analysis_set",
                    "fdr_scope",
                    "scope",
                    "x_metric",
                    "y_metric",
                    "x_edges",
                    "y_edges",
                    "counts",
                ]
            )
        else:
            volcano_bins = cached_volcano_bins(
                module_set, estimator, method, module, differential_fdr_scope
            )
        volcano_left, volcano_middle, volcano_right = st.columns(3)
        with volcano_left:
            volcano_analysis_set = st.radio(
                "Edge-test cohort",
                options=["Discovery", "Validation"],
                horizontal=True,
                key=f"volcano_set_{module_set}_{estimator}_{method}_{module}",
            )
            volcano_x_metric = st.selectbox(
                "Effect axis",
                options=["hedges_g", "mean_difference"],
                format_func=lambda value: {
                    "hedges_g": "Hedges’ g",
                    "mean_difference": "Mean edge-weight difference",
                }[value],
            )
        with volcano_middle:
            volcano_y_metric = st.selectbox(
                "Significance axis",
                options=["fdr", "p_value"],
                format_func=lambda value: {
                    "fdr": "BH FDR",
                    "p_value": "Welch p-value",
                }[value],
            )
            volcano_threshold = st.selectbox(
                "Volcano display threshold",
                options=[0.05, 0.10],
                index=0 if np.isclose(differential_fdr_threshold, 0.05) else 1,
                help=(
                    "Defaults to the sidebar feature-mask cutoff. Changing this only "
                    "changes the volcano display, not the calculated module scores."
                ),
            )
        with volcano_right:
            volcano_scope_options = (
                ["total", "TS", "CT", *active_component_order]
                if volcano_bins.empty
                else volcano_bins["scope"].drop_duplicates().tolist()
            )
            volcano_scope = st.selectbox(
                "Edge scope",
                options=volcano_scope_options,
                format_func=lambda value: {
                    "total": "Total",
                    "TS": "TS pooled",
                    "CT": "CT pooled",
                    "TS_AC": "TS: AC",
                    "TS_DLPFC": "TS: DLPFC",
                    "TS_PCGBA23": "TS: PCG",
                    "CT_AC__DLPFC": "CT: AC - DLPFC",
                    "CT_AC__PCGBA23": "CT: AC - PCG",
                    "CT_DLPFC__PCGBA23": "CT: DLPFC - PCG",
                }.get(value, value),
            )
            volcano_direction = st.selectbox(
                "Effect direction",
                ["Either", "Higher in AD", "Higher in Control"],
                index=(
                    0
                    if effect_volcano
                    else {"either": 0, "ad_higher": 1, "control_higher": 2}.get(
                        effect_direction, 0
                    )
                ),
                help=(
                    "This is an additional display filter. For effect-size analyses, "
                    "the sidebar mask direction remains defined by discovery donors."
                ),
            )
        significant_only = st.checkbox(
            (
                "Hide edges outside the active effect-size mask"
                if effect_volcano
                else "Hide nonsignificant edges"
            ),
            value=False,
            help=(
                "When selected, nonsignificant edges are removed from the display. Their "
                "effect sizes are never replaced by zero."
            ),
        )
        prevalence_column = None
        minimum_prevalence = 0.0
        if estimator == "bonobo":
            available_prevalence_rules = [
                rule
                for rule in ("native_p05", "bh_fdr05")
                if any(
                    f"bonobo_{rule}_prevalence_{group}" in volcano_candidates.columns
                    for group in ("all", "control", "mci", "ad")
                )
            ]
            if available_prevalence_rules:
                prevalence_cols = st.columns(3)
                with prevalence_cols[0]:
                    prevalence_rule = st.selectbox(
                        "BONOBO prevalence rule",
                        options=available_prevalence_rules,
                        format_func=lambda value: BONOBO_EDGE_RULE_LABELS[value],
                    )
                available_prevalence_groups = [
                    group
                    for group in ("all", "control", "mci", "ad")
                    if f"bonobo_{prevalence_rule}_prevalence_{group}"
                    in volcano_candidates.columns
                ]
                with prevalence_cols[1]:
                    prevalence_group = st.selectbox(
                        "Prevalence donor group",
                        options=available_prevalence_groups,
                        format_func=lambda value: {
                            "all": "All donors",
                            "control": "Control",
                            "mci": "MCI",
                            "ad": "AD",
                        }[value],
                    )
                with prevalence_cols[2]:
                    minimum_prevalence = st.slider(
                        "Minimum native significance prevalence",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.05,
                        format="%.0f%%",
                    )
                prevalence_column = (
                    f"bonobo_{prevalence_rule}_prevalence_{prevalence_group}"
                )
            else:
                st.caption(
                    "BONOBO native-significance prevalence is unavailable in this "
                    "volcano shard; effect and AD–Control significance filters remain active."
                )

        volcano_point_data = volcano_candidates
        if effect_volcano and significant_only:
            mask_column = effect_mask_column(differential_edge_rule)
            if mask_column not in volcano_point_data:
                st.error(
                    f"Effect candidate shard lacks required membership field: {mask_column}"
                )
                st.stop()
            volcano_point_data = volcano_point_data.loc[
                volcano_point_data[mask_column].fillna(False)
            ].copy()
            if effect_direction == "ad_higher":
                volcano_point_data = volcano_point_data.loc[
                    volcano_point_data["discovery_mean_difference"].gt(0)
                ]
            elif effect_direction == "control_higher":
                volcano_point_data = volcano_point_data.loc[
                    volcano_point_data["discovery_mean_difference"].lt(0)
                ]
        volcano = edge_volcano_figure(
            volcano_point_data,
            volcano_bins,
            module=module,
            scope=volcano_scope,
            analysis_set=volcano_analysis_set,
            fdr_scope=differential_fdr_scope,
            x_metric=volcano_x_metric,
            y_metric=volcano_y_metric,
            significant_only=significant_only and not effect_volcano,
            significance_threshold=volcano_threshold,
            direction=volcano_direction,
            prevalence_column=prevalence_column,
            minimum_prevalence=minimum_prevalence,
            module_definition=module_set_label,
        )
        render_plotly_chart(
            volcano,
            use_container_width=True,
            config={
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": f"{download_prefix}M{module}_edge_volcano",
                    "scale": 3,
                },
            },
            plot_id="main.volcano",
            metadata={
                "module_set": module_set,
                "method": method,
                "module": f"M{module}",
                "outcome": phenotype,
                "feature": feature,
            },
        )
        st.caption(
            "Exact dots are colored by tissue component: circles are within-tissue "
            "edges and diamonds are cross-tissue pairs. "
            + (
                "This effect-only catalog omits the all-edge density background to keep "
                "the public app lightweight. "
                if volcano_bins.empty
                else "The gray density background includes every edge in the selected scope. "
            )
            + (
                "Exact hoverable rows are the union of all supported effect-size masks; "
                "the checkbox narrows them to the active criterion and its discovery direction."
                if effect_volcano
                else "Exact hoverable rows include all discovery/validation FDR≤0.10 or "
                "p≤0.05 edges plus the top 500 remaining edges per resolved component."
            )
        )
        prefix = volcano_analysis_set.lower()
        probability_column = (
            f"{prefix}_fdr_{differential_fdr_scope}"
            if volcano_y_metric == "fdr"
            else f"{prefix}_p_value"
        )
        displayed_edges = volcano_candidates.copy()
        if volcano_scope in {"CT", "TS"}:
            displayed_edges = displayed_edges.loc[
                displayed_edges["component_class"].eq(volcano_scope)
            ]
        elif volcano_scope != "total":
            displayed_edges = displayed_edges.loc[
                displayed_edges["component"].eq(volcano_scope)
            ]
        if volcano_direction == "Higher in AD":
            displayed_edges = displayed_edges.loc[
                displayed_edges[f"{prefix}_mean_difference"].gt(0)
            ]
        elif volcano_direction == "Higher in Control":
            displayed_edges = displayed_edges.loc[
                displayed_edges[f"{prefix}_mean_difference"].lt(0)
            ]
        if significant_only:
            if effect_volcano:
                displayed_edges = displayed_edges.loc[
                    displayed_edges[effect_mask_column(differential_edge_rule)].fillna(
                        False
                    )
                ]
                if effect_direction == "ad_higher":
                    displayed_edges = displayed_edges.loc[
                        displayed_edges["discovery_mean_difference"].gt(0)
                    ]
                elif effect_direction == "control_higher":
                    displayed_edges = displayed_edges.loc[
                        displayed_edges["discovery_mean_difference"].lt(0)
                    ]
            else:
                displayed_edges = displayed_edges.loc[
                    displayed_edges[probability_column].le(volcano_threshold)
                ]
        if prevalence_column and prevalence_column in displayed_edges.columns:
            displayed_edges = displayed_edges.loc[
                displayed_edges[prevalence_column].ge(minimum_prevalence)
            ]
        displayed_edges = displayed_edges.sort_values(
            [probability_column, f"{prefix}_hedges_g"],
            ascending=[True, False],
            kind="stable",
        )
        displayed_edges.insert(
            displayed_edges.columns.get_loc("component") + 1,
            "tissue_component",
            displayed_edges["component"]
            .map(EDGE_COMPONENT_LABELS)
            .fillna(displayed_edges["component"].astype(str)),
        )
        filterable_dataframe(
            displayed_edges,
            table_key="volcano_edges",
            table_name="Displayed volcano edges",
            use_container_width=True,
            hide_index=True,
            height=460,
            column_config={
                "gene_a": "Gene A symbol",
                "gene_b": "Gene B symbol",
                "tissue_component": "Tissue component",
            },
        )
        st.download_button(
            "Download displayed exact edge rows (TSV)",
            data=dataframe_to_tsv_bytes(displayed_edges),
            file_name=f"{download_prefix}M{module}_displayed_edge_statistics.tsv",
            mime="text/tab-separated-values",
        )
