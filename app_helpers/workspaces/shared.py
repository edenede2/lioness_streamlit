from __future__ import annotations

from app_helpers.ui_runtime import (
    EFFECT_DIRECTION_LABELS,
    FDR_SCOPE_LABELS,
    FDR_THRESHOLD_LABELS,
    SCALE_LABELS,
    VALIDATION_SCOPE,
    aggregate_to_long,
    association_kegg_subtitles,
    attach_metadata,
    cached_aggregate,
    cached_aggregate_stats,
    cached_effect_endpoint_kegg,
    cached_kegg,
    cached_resolved,
    cached_resolved_stats,
    filterable_dataframe,
    module_label,
    pd,
    resolved_to_long,
    selected_annotation,
    selected_kegg_table_annotation,
    st,
)


def prepare_metadata(context):
    active_tissue_detail_columns = context["active_tissue_detail_columns"]
    annotations = context["annotations"]
    cohort_scope = context["cohort_scope"]
    current_score_framework = context["current_score_framework"]
    diagnoses = context["diagnoses"]
    differential_edge_rule = context["differential_edge_rule"]
    differential_fdr_scope = context["differential_fdr_scope"]
    differential_fdr_threshold = context["differential_fdr_threshold"]
    edge_rule = context["edge_rule"]
    estimator = context["estimator"]
    module = context["module"]
    module_details = context["module_details"]
    module_set = context["module_set"]
    module_set_label = context["module_set_label"]
    scale = context["scale"]
    score_normalization = context["score_normalization"]
    # Component choices are metadata, not a reason to read a donor table.
    resolved = context["resolved"]
    component_options = (
        list(context["active_component_order"]) if resolved else ["CT", "TS"]
    )
    if context["feature"] == "eigengene":
        component_options = [
            component for component in component_options if component.startswith("TS_")
        ]
    component_labels = {
        component: context["component_labels"].get(component, component)
        for component in component_options
    }
    available_components = pd.DataFrame(
        {
            "component": component_options,
            "component_label": [component_labels[c] for c in component_options],
        }
    )
    selected_split = {
        "discovery_ad_control": "Discovery",
        "validation_ad_control": "Validation",
        "mci_external": "MCI_external",
    }.get(context["analysis_subset"])
    if differential_edge_rule == "all":
        differential_caption = "not applied to module scores"
    elif differential_edge_rule.startswith("ad_control_discovery_effect__"):
        differential_caption = "effect-size selection; FDR not used"
    else:
        differential_caption = (
            f"{FDR_SCOPE_LABELS[differential_fdr_scope]}, "
            f"{FDR_THRESHOLD_LABELS[differential_fdr_threshold]}"
        )
    download_prefix = (
        f"{module_set}__{estimator}__{edge_rule}__{differential_edge_rule}__"
        f"{differential_fdr_scope}__fdr{differential_fdr_threshold:.2f}__"
        f"{score_normalization}__{cohort_scope}__"
    )

    if not diagnoses:
        st.warning("Select at least one diagnosis group in the sidebar.")
        st.stop()

    selected_details = module_details.loc[
        module_details["module"].astype(int).eq(int(module))
    ]
    if len(selected_details) != 1:
        st.error(f"Expected exactly one module-details row for {module_label(module)}.")
        st.stop()
    selected_details = selected_details.iloc[0]

    annotation = selected_annotation(annotations, module)

    tissue_composition = pd.DataFrame(
        [
            {
                "Tissue": tissue,
                "Genes": int(selected_details[count_column]),
                "Module proportion": float(selected_details[proportion_column]),
            }
            for tissue, count_column, proportion_column in active_tissue_detail_columns
        ]
    )
    tissue_composition["Possible TS edges"] = tissue_composition["Genes"].map(
        lambda count: count * (count - 1) // 2
    )
    tissue_composition["TS feature"] = tissue_composition["Genes"].map(
        lambda count: "Available" if count >= 2 else "Unavailable (<2 genes)"
    )

    from app_helpers.ui_runtime import readable_method

    identity = [
        module_set_label,
        readable_method(context["method"]),
        (
            "Incomplete-region validation"
            if cohort_scope == VALIDATION_SCOPE
            else (
                "Maximum component cohort"
                if cohort_scope == "maximum_component"
                else "Complete-tissue cohort"
            )
        ),
    ]
    if context["active_view"] not in {
        "Module finder",
        "CT–TS screen",
        "Methods & data",
    }:
        identity.append(module_label(module))
    if context["active_view"] in {
        "Associations",
        "Feature distributions",
        "Correlation heatmaps",
        "Statistics",
        "Donor edge explorer",
    }:
        identity.append(SCALE_LABELS[scale])
    st.caption("Applied: " + " · ".join(identity))
    if context["active_view"] == "Associations":
        from app_helpers.ui_runtime import cached_sample_metadata

        association_metadata = cached_sample_metadata(module_set, cohort_scope)
        association_metadata = association_metadata.loc[
            association_metadata["diagnosis_group"].isin(diagnoses)
        ].copy()
        if selected_split is not None and differential_edge_rule != "all":
            association_metadata = association_metadata.loc[
                association_metadata["ad_control_split"].eq(selected_split)
            ].copy()

    return {
        **context,
        **{key: value for key, value in locals().items() if key != "context"},
    }


def prepare_donors(context):
    resolved = context["resolved"]
    active_view = context["active_view"]
    analysis_subset = context["analysis_subset"]
    association_interpretation = context["association_interpretation"]
    cohort_scope = context["cohort_scope"]
    component_labels = context["component_labels"]
    diagnoses = context["diagnoses"]
    differential_edge_rule = context["differential_edge_rule"]
    differential_fdr_scope = context["differential_fdr_scope"]
    differential_fdr_threshold = context["differential_fdr_threshold"]
    distribution_grouping_variable = context["distribution_grouping_variable"]
    edge_rule = context["edge_rule"]
    estimator = context["estimator"]
    feature = context["feature"]
    grouping_variable = context["grouping_variable"]
    method = context["method"]
    module = context["module"]
    module_set = context["module_set"]
    phenotype = context["phenotype"]
    resolution = context["resolution"]
    scale = context["scale"]
    score_normalization = context["score_normalization"]
    selected_components = context["selected_components"]
    selected_distribution_levels = context["selected_distribution_levels"]
    selected_group_levels = context["selected_group_levels"]
    with st.spinner("Loading the selected module…"):
        if resolution == "Aggregate CT / TS":
            plot_data = cached_aggregate(
                module_set,
                estimator,
                method,
                module,
                feature,
                edge_rule,
                differential_edge_rule,
                differential_fdr_scope,
                differential_fdr_threshold,
                score_normalization,
            )
            plot_data = aggregate_to_long(plot_data, scale)
            statistics = (
                cached_aggregate_stats(
                    module_set,
                    estimator,
                    method,
                    module,
                    phenotype,
                    feature,
                    edge_rule,
                    differential_edge_rule,
                    differential_fdr_scope,
                    differential_fdr_threshold,
                    score_normalization,
                    analysis_subset,
                )
                if active_view == "Statistics"
                and association_interpretation == "numeric"
                else pd.DataFrame()
            )
            resolved = False
        else:
            plot_data = cached_resolved(
                module_set,
                estimator,
                method,
                module,
                feature,
                edge_rule,
                differential_edge_rule,
                differential_fdr_scope,
                differential_fdr_threshold,
                score_normalization,
                cohort_scope,
            )
            plot_data = resolved_to_long(plot_data, scale)
            statistics = (
                cached_resolved_stats(
                    module_set,
                    estimator,
                    method,
                    module,
                    phenotype,
                    feature,
                    edge_rule,
                    differential_edge_rule,
                    differential_fdr_scope,
                    differential_fdr_threshold,
                    score_normalization,
                    analysis_subset,
                    cohort_scope,
                )
                if active_view == "Statistics"
                and association_interpretation == "numeric"
                else pd.DataFrame()
            )
            resolved = True

    plot_data = attach_metadata(
        plot_data, module_set=module_set, cohort_scope=cohort_scope
    )
    if differential_edge_rule != "all" and analysis_subset != "all_donors":
        selected_split = {
            "discovery_ad_control": "Discovery",
            "validation_ad_control": "Validation",
            "mci_external": "MCI_external",
        }[analysis_subset]
        plot_data = plot_data.loc[
            plot_data["ad_control_split"].eq(selected_split)
        ].copy()
    if differential_edge_rule != "all":
        st.warning(
            "The edge mask is frozen from the original discovery cohort. Every displayed donor is outside the complete-expression cohort; no validation labels select edges. This remains within-ROSMAP validation, not external-cohort validation."
            if cohort_scope == VALIDATION_SCOPE
            else "The AD–Control edge mask was selected in the discovery donors. Use the held-out "
            "validation view for diagnostic-group evaluation. This validates edge selection, "
            "not a fully external network model: Control-referenced LIONESS still uses its "
            "established Control reference."
        )

    plot_data = plot_data.loc[
        plot_data["diagnosis_group"].isin(diagnoses)
        & plot_data["component"].isin(selected_components)
    ].copy()
    if active_view == "Associations":
        if association_interpretation == "numeric" and grouping_variable != "__all__":
            plot_data = plot_data.loc[
                plot_data[grouping_variable].notna()
                & plot_data[grouping_variable].isin(selected_group_levels)
            ].copy()
        elif association_interpretation == "categorical":
            plot_data = plot_data.loc[
                plot_data[phenotype].notna()
                & plot_data[phenotype].isin(selected_group_levels)
            ].copy()
    elif (
        active_view == "Feature distributions"
        and distribution_grouping_variable != "__all__"
    ):
        plot_data = plot_data.loc[
            plot_data[distribution_grouping_variable].notna()
            & plot_data[distribution_grouping_variable].isin(
                selected_distribution_levels
            )
        ].copy()
    if not statistics.empty:
        statistics = statistics.loc[
            statistics["diagnosis_group"].isin(diagnoses)
        ].copy()
        if resolved:
            statistics = statistics.loc[
                statistics["component"].isin(selected_components)
            ]

    if plot_data.empty or not selected_components:
        st.warning("No data remain for the current filters.")
        st.stop()

    component_variation = plot_data.groupby("component", observed=True)[
        "metric_value"
    ].agg(nonmissing="count", distinct=lambda values: values.nunique(dropna=True))
    if cohort_scope == VALIDATION_SCOPE and "n_retained_edges" in plot_data:
        empty_validation_components = plot_data.groupby("component")[
            "n_retained_edges"
        ].max()
        empty_validation_components = empty_validation_components.loc[
            empty_validation_components.eq(0)
        ].index
        if len(empty_validation_components):
            st.warning(
                "No retained edges in: "
                + ", ".join(
                    component_labels.get(c, c) for c in empty_validation_components
                )
                + ". Standard pruned scores are zero for valid empty masks; retained-edge means are unavailable. Structurally nonexistent blocks are always unavailable."
            )
    structurally_unavailable_components = component_variation.loc[
        component_variation["nonmissing"].eq(0)
    ].index.tolist()
    diagnosis_variation = plot_data.groupby(
        ["component", "diagnosis_group"], observed=True
    )["metric_value"].agg(
        nonmissing="count", distinct=lambda values: values.nunique(dropna=True)
    )
    zero_variance_states = diagnosis_variation.loc[
        diagnosis_variation["nonmissing"].gt(0) & diagnosis_variation["distinct"].le(1)
    ].reset_index()
    if estimator == "bonobo" and structurally_unavailable_components:
        unavailable_labels = [
            component_labels.get(value, value)
            for value in structurally_unavailable_components
        ]
        st.warning(
            "The selected module has no structurally possible edges in: "
            + ", ".join(unavailable_labels)
            + ". These scopes remain missing rather than being encoded as zero."
        )
    if estimator == "bonobo" and not zero_variance_states.empty:
        zero_labels = [
            f"{component_labels.get(row.component, row.component)} ({row.diagnosis_group})"
            for row in zero_variance_states.itertuples(index=False)
        ]
        st.warning(
            "BONOBO has no donor-to-donor variation for the selected sparse-network score in: "
            + ", ".join(zero_labels)
            + ". Correlations are intentionally unavailable for constant states; for sparse "
            "features these are generally zero because no edge passed the selected rule."
        )

    return {
        **context,
        **{key: value for key, value in locals().items() if key != "context"},
    }


def prepare_enrichment(context):
    annotation = context["annotation"]
    effect_direction = context["effect_direction"]
    match_effect_kegg_subtitle = context["match_effect_kegg_subtitle"]
    method = context["method"]
    module = context["module"]
    module_set = context["module_set"]
    resolved = context["resolved"]
    selected_components = context["selected_components"]
    module_kegg = cached_kegg(module_set, int(module))
    association_kegg_context = ""
    if match_effect_kegg_subtitle:
        module_kegg = cached_effect_endpoint_kegg(
            module_set, method, effect_direction, int(module)
        )
        annotation = selected_kegg_table_annotation(module_kegg)
        association_kegg_context = (
            "Hedges’ g endpoint genes, "
            + EFFECT_DIRECTION_LABELS[effect_direction].lower()
        )
    association_subtitles = association_kegg_subtitles(
        module_kegg,
        selected_components,
        resolved=resolved,
        aggregate_annotation=annotation,
    )
    if association_kegg_context:
        association_subtitles = {
            component: subtitle.replace(
                "KEGG enrichment",
                f"KEGG enrichment [{association_kegg_context}]",
                1,
            )
            for component, subtitle in association_subtitles.items()
        }

    return {
        **context,
        **{key: value for key, value in locals().items() if key != "context"},
    }


def association_header(figure):
    """Give title and legend separate container rows, including narrow screens."""
    import html
    import textwrap

    extra_lines = 0
    for annotation in figure.layout.annotations:
        if not str(annotation.name or "").startswith("publication-panel-title-"):
            continue
        parts = str(annotation.text or "").split("<br>")
        wrapped = parts[:1]
        for part in parts[1:]:
            wrapped.extend(
                html.escape(line)
                for line in textwrap.wrap(
                    html.unescape(part),
                    width=42,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
        extra_lines = max(extra_lines, len(wrapped) - len(parts))
        annotation.text = "<br>".join(wrapped)
    if extra_lines:
        extra = 14 * extra_lines
        figure.update_layout(
            height=(figure.layout.height or 600) + extra,
            margin=dict(t=(figure.layout.margin.t or 200) + extra),
        )
    figure.update_layout(
        title=dict(y=0.90, yref="container", yanchor="top"),
        # y>1 retains the existing exporter's outside-legend contract. Using
        # exactly 1 would put the publication legend on top of its title.
        legend=dict(y=1.001, yref="container", yanchor="top", x=0.0, xanchor="left"),
    )
    return figure


def render_composition(context):
    active_tissues = context["active_tissues"]
    module = context["module"]
    selected_details = context["selected_details"]
    tissue_composition = context["tissue_composition"]
    with st.expander(
        f"{module_label(module)} module composition · {int(selected_details['module_size'])} genes · "
        f"{selected_details['tissues']}",
        expanded=False,
    ):
        detail_cols = st.columns(4)
        detail_cols[0].metric(
            "Module size", f"{int(selected_details['module_size']):,} genes"
        )
        detail_cols[1].metric(
            "Tissues represented",
            f"{int(selected_details['n_tissues'])} of {len(active_tissues)}",
        )
        detail_cols[2].metric("Module type", str(selected_details["cluster_type"]))
        detail_cols[3].metric(
            "Dominant tissue", str(selected_details["dominant_tissue"])
        )
        filterable_dataframe(
            tissue_composition,
            table_key="module_composition",
            table_name="Module composition",
            use_container_width=True,
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
