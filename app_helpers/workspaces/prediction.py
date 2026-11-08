from __future__ import annotations

from app_helpers.ui_runtime import (
    EIGENGENE_SOURCE_LABELS,
    MODULE_SET_LABELS,
    MODULE_SET_METHODS,
    PREDICTION_BLOCK_LABELS,
    PREDICTION_BLOCK_ORDER,
    PREDICTION_DESIGN_LABELS,
    PREDICTION_MASK_LABELS,
    PREDICTION_MODEL_LABELS,
    PREDICTION_MODULE_SETS,
    PREDICTION_OUTCOME_LABELS,
    PREDICTION_REFERENCE_LABELS,
    SCORE_NORMALIZATION_LABELS,
    SCORE_TRANSFORM_LABELS,
    TARGETED_HEDGES_MASKS,
    TARGETED_PANEL_LABELS,
    TARGETED_PREDICTION_MODE_LABELS,
    TARGETED_TIER_LABELS,
    _targeted_filters,
    annotate_prediction_coefficients,
    cached_coefficient_kegg_scopes,
    cached_prediction_bootstrap,
    cached_prediction_coefficients,
    cached_prediction_confusion,
    cached_prediction_curves,
    cached_prediction_diagnostics,
    cached_prediction_performance,
    cached_prediction_whole_network,
    cached_targeted_prediction_table,
    classification_diagnostic_rows,
    dataframe_to_tsv_bytes,
    filterable_dataframe,
    load_prediction_manifest,
    load_targeted_prediction_manifest,
    ordered_prediction_blocks,
    pd,
    prediction_coefficient_figure,
    prediction_confusion_figure,
    prediction_ct_ts_figure,
    prediction_curve_figure,
    prediction_data_available,
    prediction_error_figure,
    prediction_heatmap_figure,
    prediction_observed_figure,
    prediction_performance_figure,
    prediction_threshold_figure,
    protocol_v11_available,
    readable_method,
    render_plotly_chart,
    render_protocol_v11_view,
    st,
    stateful_tabs,
    tab_is_open,
    targeted_eigengene_source_comparison_figure,
    targeted_fold_robustness_figure,
    targeted_panel_overlap_figure,
    targeted_prediction_data_available,
    targeted_primary_comparison_figure,
    targeted_selection_frequency_figure,
    targeted_transform_comparison_figure,
    targeted_transform_heatmap_figure,
)


def render_targeted_prediction_view() -> None:
    """Render repeated nested-CV targeted-module results without touching the benchmark."""

    manifest = load_targeted_prediction_manifest()
    transcriptomic_variants = {
        "transcriptomics_only",
        "covariates_plus_transcriptomics",
        "network_plus_transcriptomics",
        "covariates_plus_network_plus_transcriptomics",
    }
    masked_available = bool(manifest.get("masked_sensitivity_available", False))
    hedges_manifest = manifest.get("hedges_g04_sensitivity", {})
    hedges_available = bool(manifest.get("hedges_g04_sensitivity_available", False))
    with st.sidebar:
        edge_options = ["all"]
        if masked_available:
            edge_options.extend(
                ["global_fdr05", "global_fdr10", "per_module_fdr05", "per_module_fdr10"]
            )
        if hedges_available:
            completed_hedges_masks = set(hedges_manifest.get("edge_masks", []))
            edge_options.extend(
                value
                for value in TARGETED_HEDGES_MASKS
                if value in completed_hedges_masks
            )
        edge_mask = st.selectbox(
            "Targeted edge set",
            options=edge_options,
            format_func=lambda value: PREDICTION_MASK_LABELS.get(value, value),
            key="targeted_edge_mask",
        )
        hedges_selection = edge_mask in TARGETED_HEDGES_MASKS
        fdr_masked_selection = edge_mask not in {"all", *TARGETED_HEDGES_MASKS}
        if edge_mask == "all" or hedges_selection:
            score_normalization = "standard_pruned"
        else:
            score_normalization = st.radio(
                "Targeted score normalization",
                options=["standard_pruned", "retained_edge"],
                format_func=lambda value: SCORE_NORMALIZATION_LABELS[value],
                key="targeted_score_normalization",
            )
    masked_selection = edge_mask != "all"
    performance_table = (
        "hedges_oof_performance"
        if hedges_selection
        else "masked_oof_performance" if fdr_masked_selection else "oof_performance"
    )
    fold_table = (
        "hedges_fold_performance"
        if hedges_selection
        else "masked_fold_performance" if fdr_masked_selection else "fold_performance"
    )
    coefficient_table = (
        "hedges_coefficients"
        if hedges_selection
        else "masked_coefficients" if fdr_masked_selection else "coefficients"
    )
    diagnostic_table = (
        "hedges_oof_predictions"
        if hedges_selection
        else "masked_oof_predictions" if fdr_masked_selection else "oof_predictions"
    )
    performance_catalog = cached_targeted_prediction_table(
        performance_table,
        _targeted_filters(
            edge_mask=edge_mask,
            score_normalization=score_normalization,
        ),
    )
    if performance_catalog.empty:
        st.info(
            "The targeted-prediction manifest is present, but no completed OOF results exist."
        )
        return
    if fdr_masked_selection:
        score_transform = "asinh"
    elif hedges_selection:
        score_transform = "raw"
    else:
        available_transforms = [
            value
            for value in ("raw", "asinh", "rint")
            if value in set(performance_catalog["score_transform"].astype(str))
        ]
        with st.sidebar:
            score_transform = st.selectbox(
                "Module-score transformation",
                options=available_transforms,
                format_func=lambda value: SCORE_TRANSFORM_LABELS.get(value, value),
                index=(
                    available_transforms.index("raw")
                    if "raw" in available_transforms
                    else 0
                ),
                key="targeted_score_transform",
                help=(
                    "Raw is the prespecified primary scale. asinh and RINT are exploratory "
                    "robustness analyses fitted within the nested-CV training partitions."
                ),
            )
        performance_catalog = performance_catalog.loc[
            performance_catalog["score_transform"].eq(score_transform)
        ].copy()
    source_registry = manifest.get("eigengene_source_registry", {})
    observed_sources = [
        value
        for value in performance_catalog.loc[
            performance_catalog["model_variant"].isin(transcriptomic_variants),
            "eigengene_source",
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
        if value != "not_applicable"
    ]
    selectable_sources = [
        source
        for source in source_registry
        if bool(source_registry[source].get("selectable", False))
        and source in observed_sources
    ]
    selectable_sources.extend(
        source
        for source in observed_sources
        if source not in source_registry and source not in selectable_sources
    )
    if hedges_selection:
        validated_hedges_sources = set(hedges_manifest.get("eigengene_sources", []))
        selectable_sources = [
            source for source in observed_sources if source in validated_hedges_sources
        ]
    if fdr_masked_selection or score_transform != "raw":
        eigengene_source = (
            "matched_multitissue"
            if "matched_multitissue" in observed_sources
            else observed_sources[0] if observed_sources else "not_applicable"
        )
    else:
        with st.sidebar:
            eigengene_source = st.selectbox(
                "Eigengene source",
                options=selectable_sources or ["matched_multitissue"],
                format_func=lambda value: EIGENGENE_SOURCE_LABELS.get(value, value),
                index=(
                    (selectable_sources or ["matched_multitissue"]).index(
                        "matched_multitissue"
                    )
                    if "matched_multitissue"
                    in (selectable_sources or ["matched_multitissue"])
                    else 0
                ),
                key="targeted_eigengene_source",
                help=(
                    "This selection changes transcriptomic and joint models only. Dummy, "
                    "demographic, and network-only baselines remain source-neutral."
                ),
            )
    performance_catalog = performance_catalog.loc[
        ~performance_catalog["model_variant"].isin(transcriptomic_variants)
        | performance_catalog["eigengene_source"].eq(eigengene_source)
    ].copy()
    st.subheader(
        "Targeted Hedges’ g edge-effect sensitivity"
        if hedges_selection
        else (
            "Targeted differential-edge sensitivity"
            if fdr_masked_selection
            else "Fully nested targeted-module LIONESS prediction"
        )
    )
    st.warning(
        "Repeated nested cross-validation is internal validation conditional on fixed module "
        "definitions. The earlier 70/30 cohort has already been inspected and is shown only "
        "as sensitivity evidence; it is not independent validation."
    )
    if not bool(manifest.get("complete", False)):
        st.warning(
            "This is an incremental transformation catalog. Only completed configurations are "
            "shown; the full Raw/asinh/RINT reconciliation is still interim."
        )
    if hedges_selection:
        st.warning(
            "Exploratory effect-size sensitivity: each outer fold learns its Hedges’ g "
            "mask only from outer-training AD and Control donor-edge weights. All-edge "
            "panels, K, and regularization are frozen; MCI and outer-test labels never "
            "construct the mask."
        )
        st.caption(
            "Raw standard-pruned connectivity is fixed for this analysis. The three "
            "directions are g ≥ 0.40, |g| ≥ 0.40, and g ≤ −0.40."
        )
    elif fdr_masked_selection:
        st.warning(
            "Exploratory differential-edge sensitivity: masks are learned from outer-training "
            "AD/Control donors in one five-fold outer cycle. The fold-specific all-edge panel "
            "and regularization settings are frozen; no new selection search is performed."
        )
        st.caption(
            "The differential-edge masked catalog is preserved unchanged and uses the existing "
            "asinh-based module scores."
        )
    elif score_transform != "raw":
        st.info(
            f"{SCORE_TRANSFORM_LABELS.get(score_transform, score_transform)} is an exploratory "
            "robustness analysis. The three prespecified primary hypotheses remain Raw-only."
        )
    with st.sidebar:
        st.header("Targeted prediction controls")
        tier_order = [
            value
            for value in ("primary", "secondary", "exploratory")
            if value in set(performance_catalog["evidence_tier"])
        ]
        evidence_tier = st.selectbox(
            "Evidence tier",
            options=tier_order,
            format_func=lambda value: TARGETED_TIER_LABELS.get(value, value),
            key="targeted_evidence_tier",
        )
        tier_catalog = performance_catalog.loc[
            performance_catalog["evidence_tier"].eq(evidence_tier)
        ]
        module_options = tier_catalog["module_definition"].drop_duplicates().tolist()
        module_definition = st.selectbox(
            "Targeted module definition",
            options=module_options,
            format_func=lambda value: MODULE_SET_LABELS.get(value, value),
            index=(
                module_options.index("control_derived")
                if "control_derived" in module_options
                else 0
            ),
            key="targeted_module_definition",
        )
        module_catalog = tier_catalog.loc[
            tier_catalog["module_definition"].eq(module_definition)
        ]
        methods = module_catalog["network_method"].drop_duplicates().tolist()
        network_method = st.selectbox(
            "Targeted LIONESS method",
            options=methods,
            format_func=readable_method,
            key="targeted_network_method",
        )
        method_catalog = module_catalog.loc[
            module_catalog["network_method"].eq(network_method)
        ]
        outcome_options = method_catalog["model_outcome"].drop_duplicates().tolist()
        outcome = st.selectbox(
            "Targeted prediction outcome",
            options=outcome_options,
            format_func=lambda value: PREDICTION_OUTCOME_LABELS.get(value, value),
            index=(
                outcome_options.index("diagnosis_binary")
                if "diagnosis_binary" in outcome_options
                else 0
            ),
            key="targeted_outcome",
        )
        outcome_catalog = method_catalog.loc[
            method_catalog["model_outcome"].eq(outcome)
        ]
        panel_options = outcome_catalog["panel_strategy"].drop_duplicates().tolist()
        panel_strategy = st.selectbox(
            "Module panel",
            options=panel_options,
            format_func=lambda value: TARGETED_PANEL_LABELS.get(value, value),
            index=(
                panel_options.index("tissue_neutral_ad")
                if "tissue_neutral_ad" in panel_options
                else 0
            ),
            key="targeted_panel_strategy",
        )

    if fdr_masked_selection and outcome == "cogn_global":
        st.info(
            "Global cognition was not an outcome in the completed all-edge prediction "
            "catalog. Its diagnosis-derived module panel and K remain frozen; only "
            "elastic-net regularization was calibrated by inner CV within each "
            "outer-training fold on all-edge scores. Outer-test outcomes and masked scores "
            "were not used for that calibration."
        )
    if outcome == "clusters":
        st.warning(
            "Cluster prediction is exploratory. The reused tissue-neutral panel was selected "
            "for AD-versus-Control, not for cluster membership, and the four nominal clusters "
            "are strongly associated with diagnosis in this cohort. This analysis describes "
            "transfer of a diagnosis-derived panel; it is not an independent replication."
        )

    selected = performance_catalog.loc[
        performance_catalog["evidence_tier"].eq(evidence_tier)
        & performance_catalog["module_definition"].eq(module_definition)
        & performance_catalog["network_method"].eq(network_method)
        & performance_catalog["model_outcome"].eq(outcome)
        & performance_catalog["panel_strategy"].eq(panel_strategy)
        & performance_catalog["edge_mask"].eq(edge_mask)
        & performance_catalog["score_normalization"].eq(score_normalization)
        & performance_catalog["score_transform"].eq(score_transform)
    ].copy()
    if selected.empty:
        st.warning("No repeated-CV models match the selected targeted analysis.")
        return
    primary_metric = str(
        selected.loc[selected["metric"].notna(), "metric"].iloc[0]
        if "primary_metric" not in selected
        else selected["primary_metric"].dropna().iloc[0]
    )
    if "primary_metric" not in selected:
        preferred = {
            "diagnosis_binary": "roc_auc",
            "diagnosis_three_class": "macro_roc_auc",
            "cogdx": "mae",
            "parkinsonism": "roc_auc",
        }.get(outcome, "r2")
        if preferred in set(selected["metric"]):
            primary_metric = preferred
    repeats = (
        1
        if fdr_masked_selection
        else int(manifest.get("selection", {}).get("outer_repeats", 5))
    )
    folds = int(manifest.get("selection", {}).get("outer_folds", 5))
    metrics = st.columns(6)
    metrics[0].metric(
        "Evaluation",
        (
            f"{repeats} × {folds} outer CV sensitivity"
            if masked_selection
            else f"{repeats} × {folds} nested CV"
        ),
    )
    metrics[1].metric("OOF donors", int(selected["n_oof"].max()))
    metrics[2].metric("Primary metric", primary_metric)
    metrics[3].metric(
        "Score scale", SCORE_TRANSFORM_LABELS.get(score_transform, score_transform)
    )
    metrics[4].metric(
        "Edge set",
        "All edges" if set(selected["edge_mask"]) == {"all"} else "Sensitivity mask",
    )
    metrics[5].metric(
        "Eigengenes",
        EIGENGENE_SOURCE_LABELS.get(eigengene_source, eigengene_source),
    )

    (
        summary_tab,
        comparison_tab,
        transformation_tab,
        source_tab,
        panel_tab,
        diagnostic_tab,
        coefficient_tab,
        tables_tab,
        methods_tab,
    ) = stateful_tabs(
        [
            "Summary",
            "CT versus TS",
            "Transformation sensitivity",
            "Eigengene sources",
            "Panel selection",
            "OOF diagnostics",
            "Coefficients & KEGG",
            "Tables",
            "Methods",
        ],
        key="targeted_prediction_tabs",
    )
    with summary_tab:
        if tab_is_open(summary_tab):
            st.caption(
                "Performance is calculated from fold-specific panels and outer-test predictions. "
                "The consensus panel below is display-only and is never used to estimate performance."
            )
            if (
                not masked_selection
                and score_transform == "raw"
                and evidence_tier == "primary"
                and module_definition == "control_derived"
                and outcome == "diagnosis_binary"
            ):
                absolute = performance_catalog.loc[
                    performance_catalog["evidence_tier"].eq("primary")
                    & performance_catalog["score_transform"].eq("raw")
                    & performance_catalog["edge_mask"].eq("all")
                    & performance_catalog["score_normalization"].eq("standard_pruned")
                    & performance_catalog["module_definition"].eq("control_derived")
                    & performance_catalog["network_method"].eq("control_anchored")
                    & performance_catalog["model_outcome"].eq("diagnosis_binary")
                    & performance_catalog["predictor_block"].eq("CT_pooled")
                    & performance_catalog["metric"].eq("roc_auc")
                ]
                absolute_specs = (
                    (
                        "Unadjusted comparators",
                        (
                            ("Covariates", "tissue_neutral_ad", "covariates"),
                            ("Connectivity", "tissue_neutral_ad", "network_only"),
                            (
                                "Transcriptomics",
                                "tissue_neutral_ad",
                                "transcriptomics_only",
                            ),
                            (
                                "Connectivity + transcriptomics",
                                "tissue_neutral_ad",
                                "network_plus_transcriptomics",
                            ),
                        ),
                    ),
                    (
                        "Covariate-adjusted comparators",
                        (
                            (
                                "Adjusted connectivity",
                                "tissue_neutral_ad",
                                "covariates_plus_network",
                            ),
                            (
                                "Adjusted transcriptomics",
                                "tissue_neutral_ad",
                                "covariates_plus_transcriptomics",
                            ),
                            (
                                "Fully adjusted joint",
                                "tissue_neutral_ad",
                                "covariates_plus_network_plus_transcriptomics",
                            ),
                            (
                                "All-module adjusted connectivity",
                                "all_modules",
                                "covariates_plus_network",
                            ),
                        ),
                    ),
                )
                for row_label, specifications in absolute_specs:
                    st.caption(row_label)
                    absolute_metrics = st.columns(len(specifications))
                    for column, (label, strategy, variant) in zip(
                        absolute_metrics, specifications, strict=True
                    ):
                        value = absolute.loc[
                            absolute["panel_strategy"].eq(strategy)
                            & absolute["model_variant"].eq(variant),
                            "value",
                        ]
                        column.metric(
                            f"{label} ROC-AUC",
                            f"{float(value.iloc[0]):.3f}" if len(value) else "NA",
                        )
            render_plotly_chart(
                prediction_performance_figure(
                    selected,
                    metric=primary_metric,
                    block_labels=PREDICTION_BLOCK_LABELS,
                    block_order=PREDICTION_BLOCK_ORDER,
                    model_labels=PREDICTION_MODEL_LABELS,
                    title=f"Nested-CV OOF performance: {PREDICTION_OUTCOME_LABELS.get(outcome, outcome)}",
                ),
                use_container_width=True,
                config={"displaylogo": False},
                plot_id="render_targeted_prediction_view.prediction_performance_figure",
            )
            if (
                not masked_selection
                and score_transform == "raw"
                and evidence_tier == "primary"
                and outcome == "diagnosis_binary"
            ):
                comparisons = cached_targeted_prediction_table("primary_comparisons")
                if not comparisons.empty:
                    render_plotly_chart(
                        targeted_primary_comparison_figure(
                            comparisons,
                            title="Three prespecified primary paired hypotheses",
                        ),
                        use_container_width=True,
                        config={"displaylogo": False},
                        plot_id="render_targeted_prediction_view.targeted_primary_comparison_figure",
                    )
                    st.caption(
                        "Positive differences favor the first model. BH FDR is corrected across "
                        "exactly these three primary hypotheses."
                    )
                    filterable_dataframe(
                        comparisons,
                        table_key="targeted_primary_comparisons",
                        table_name="Three primary paired comparisons",
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        "Download primary comparisons (TSV)",
                        data=dataframe_to_tsv_bytes(comparisons),
                        file_name="targeted_lioness_primary_comparisons.tsv",
                        mime="text/tab-separated-values",
                    )

    with comparison_tab:
        if tab_is_open(comparison_tab):
            adjusted = selected.loc[
                selected["model_variant"].isin(
                    [
                        "network_only",
                        "covariates_plus_network",
                        "transcriptomics_only",
                        "covariates_plus_transcriptomics",
                        "network_plus_transcriptomics",
                        "covariates_plus_network_plus_transcriptomics",
                    ]
                )
            ].copy()
            if adjusted.empty:
                st.info("No CT/TS comparison is available for this selection.")
            else:
                render_plotly_chart(
                    prediction_performance_figure(
                        adjusted,
                        metric=primary_metric,
                        block_labels=PREDICTION_BLOCK_LABELS,
                        block_order=PREDICTION_BLOCK_ORDER,
                        model_labels=PREDICTION_MODEL_LABELS,
                        title="CT, TS, and resolved-component OOF comparison",
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                    plot_id="render_targeted_prediction_view.prediction_performance_figure.2",
                )
                fold_performance = cached_targeted_prediction_table(
                    fold_table,
                    _targeted_filters(
                        evidence_tier=evidence_tier,
                        module_definition=module_definition,
                        network_method=network_method,
                        panel_strategy=panel_strategy,
                        model_outcome=outcome,
                        model_variant="covariates_plus_network",
                        edge_mask=edge_mask,
                        score_normalization=score_normalization,
                        score_transform=score_transform,
                    ),
                )
                if not fold_performance.empty:
                    render_plotly_chart(
                        targeted_fold_robustness_figure(
                            fold_performance,
                            metric=primary_metric,
                            block_labels=PREDICTION_BLOCK_LABELS,
                            block_order=PREDICTION_BLOCK_ORDER,
                            title="Outer-fold robustness",
                        ),
                        use_container_width=True,
                        config={"displaylogo": False},
                        plot_id="render_targeted_prediction_view.targeted_fold_robustness_figure",
                    )
                if not masked_selection and evidence_tier in {
                    "secondary",
                    "exploratory",
                }:
                    tier_comparisons = cached_targeted_prediction_table(
                        "tier_comparisons",
                        _targeted_filters(
                            evidence_tier=evidence_tier,
                            module_definition=module_definition,
                            network_method=network_method,
                            model_outcome=outcome,
                            score_transform=score_transform,
                        ),
                    )
                    if not tier_comparisons.empty:
                        st.caption(
                            "Secondary and exploratory paired comparisons use separate BH families; "
                            "exploratory rows retain both global-tier and within-outcome FDR."
                        )
                        filterable_dataframe(
                            tier_comparisons,
                            table_key="targeted_tier_comparisons",
                            table_name="Nested-CV paired performance comparisons",
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.download_button(
                            "Download paired comparisons (TSV)",
                            data=dataframe_to_tsv_bytes(tier_comparisons),
                            file_name="targeted_lioness_nested_cv_paired_comparisons.tsv",
                            mime="text/tab-separated-values",
                        )

    with transformation_tab:
        if tab_is_open(transformation_tab):
            if hedges_selection:
                st.info(
                    "Score-transformation sensitivity applies to the all-edge targeted catalog. "
                    "This Hedges’ g sensitivity is intentionally fixed to Raw standard-pruned "
                    "connectivity so the edge-direction comparison is not mixed with another "
                    "robustness axis."
                )
            elif fdr_masked_selection:
                st.info(
                    "Transformation sensitivity applies to the all-edge targeted catalog. "
                    "The differential-edge masked catalog remains the existing asinh analysis."
                )
            elif int(manifest.get("schema_version", 2)) < 3:
                st.info(
                    "This deployed snapshot contains the corrected legacy asinh catalog only. "
                    "Raw will appear here after its first validated incremental publication."
                )
            else:
                transform_status = cached_targeted_prediction_table(
                    "transformation_status"
                )
                if not transform_status.empty:
                    filterable_dataframe(
                        transform_status,
                        table_key="targeted_transform_status",
                        table_name="Transformation analysis status",
                        use_container_width=True,
                        hide_index=True,
                    )
                transform_comparisons = cached_targeted_prediction_table(
                    "transformation_comparisons",
                    _targeted_filters(
                        module_definition=module_definition,
                        network_method=network_method,
                        panel_strategy=panel_strategy,
                        model_outcome=outcome,
                    ),
                )
                if transform_comparisons.empty:
                    st.info(
                        "Paired transformed-versus-Raw comparisons will appear after a robustness "
                        "transformation completes for this configuration."
                    )
                else:
                    st.caption(
                        "These comparisons are exploratory. Positive oriented differences favor "
                        "asinh or RINT over Raw; global and within-outcome BH FDRs are both shown."
                    )
                    render_plotly_chart(
                        targeted_transform_comparison_figure(
                            transform_comparisons,
                            block_labels=PREDICTION_BLOCK_LABELS,
                            model_labels=PREDICTION_MODEL_LABELS,
                            title="Transformation sensitivity: paired donor-averaged OOF performance",
                        ),
                        use_container_width=True,
                        config={"displaylogo": False},
                        plot_id="render_targeted_prediction_view.targeted_transform_comparison_figure",
                    )
                    render_plotly_chart(
                        targeted_transform_heatmap_figure(
                            transform_comparisons,
                            block_labels=PREDICTION_BLOCK_LABELS,
                            block_order=PREDICTION_BLOCK_ORDER,
                            model_labels=PREDICTION_MODEL_LABELS,
                            title="Oriented primary-metric difference versus Raw",
                        ),
                        use_container_width=True,
                        config={"displaylogo": False},
                        plot_id="render_targeted_prediction_view.targeted_transform_heatmap_figure",
                    )
                    filterable_dataframe(
                        transform_comparisons,
                        table_key="targeted_transform_comparisons",
                        table_name="Paired transformation comparisons",
                        use_container_width=True,
                        hide_index=True,
                    )
                if panel_strategy != "all_modules":
                    selection_outcome = str(
                        outcome_catalog.loc[
                            outcome_catalog["panel_strategy"].eq(panel_strategy),
                            "selection_outcome",
                        ].iloc[0]
                    )
                    overlap = cached_targeted_prediction_table(
                        "panel_transform_overlap",
                        _targeted_filters(
                            module_definition=module_definition,
                            network_method=network_method,
                            panel_strategy=panel_strategy,
                            selection_outcome=selection_outcome,
                        ),
                    )
                    if not overlap.empty:
                        render_plotly_chart(
                            targeted_panel_overlap_figure(
                                overlap,
                                title="Fold-specific panel overlap with Raw",
                            ),
                            use_container_width=True,
                            config={"displaylogo": False},
                            plot_id="render_targeted_prediction_view.targeted_panel_overlap_figure",
                        )
                        k_summary = (
                            overlap.groupby("transformed_scale", observed=True)
                            .agg(
                                folds=("outer_fold", "size"),
                                median_raw_k=("raw_k", "median"),
                                median_transformed_k=("transformed_k", "median"),
                                median_jaccard=("jaccard", "median"),
                            )
                            .reset_index()
                        )
                        filterable_dataframe(
                            k_summary,
                            table_key="targeted_transform_k_summary",
                            table_name="K and panel-overlap summary",
                            use_container_width=True,
                            hide_index=True,
                        )
                if not transform_comparisons.empty:
                    st.download_button(
                        "Download transformation sensitivity (TSV)",
                        data=dataframe_to_tsv_bytes(transform_comparisons),
                        file_name="targeted_lioness_transformation_sensitivity.tsv",
                        mime="text/tab-separated-values",
                    )

    with source_tab:
        if tab_is_open(source_tab):
            source_rows = []
            for source_id, values in source_registry.items():
                module_counts = values.get("module_counts", {})
                assignment_counts = values.get("assignment_counts", {})
                source_rows.append(
                    {
                        "eigengene_source": source_id,
                        "label": values.get(
                            "label", EIGENGENE_SOURCE_LABELS.get(source_id, source_id)
                        ),
                        "status": values.get("status", "unavailable"),
                        "selectable": bool(values.get("selectable", False)),
                        "partition_level": values.get("partition_level"),
                        "tissue_scoped_modules": values.get("tissue_scoped_modules"),
                        "ac_modules": module_counts.get("AC"),
                        "dlpfc_modules": module_counts.get("DLPFC"),
                        "pcg_modules": module_counts.get("PCG"),
                        "assigned_tissue_genes": (
                            sum(assignment_counts.values())
                            if assignment_counts
                            else None
                        ),
                        "raw_configurations": values.get("raw_configurations", 0),
                    }
                )
            if source_rows:
                filterable_dataframe(
                    pd.DataFrame(source_rows),
                    table_key="targeted_eigengene_source_registry",
                    table_name="Eigengene-source availability",
                    use_container_width=True,
                    hide_index=True,
                )
            if fdr_masked_selection or score_transform != "raw":
                st.info(
                    "Independent eigengene-source comparisons are available only for the "
                    "all-edge Raw catalog. This selection retains the matched multi-tissue source."
                )
            else:
                source_table = (
                    "hedges_oof_performance" if hedges_selection else "oof_performance"
                )
                source_performance = cached_targeted_prediction_table(
                    source_table,
                    _targeted_filters(
                        evidence_tier=evidence_tier,
                        module_definition=module_definition,
                        network_method=network_method,
                        panel_strategy=panel_strategy,
                        model_outcome=outcome,
                        edge_mask=edge_mask,
                        score_normalization="standard_pruned",
                        score_transform="raw",
                    ),
                )
                completed_sources = (
                    set(hedges_manifest.get("eigengene_sources", []))
                    if hedges_selection
                    else {
                        source
                        for source, values in source_registry.items()
                        if bool(values.get("selectable", False))
                    }
                )
                source_performance = source_performance.loc[
                    source_performance["model_variant"].isin(transcriptomic_variants)
                    & source_performance["eigengene_source"].isin(completed_sources)
                ].copy()
                source_variant_order = [
                    value
                    for value in (
                        "transcriptomics_only",
                        "covariates_plus_transcriptomics",
                        "network_plus_transcriptomics",
                        "covariates_plus_network_plus_transcriptomics",
                    )
                    if value in set(source_performance["model_variant"].astype(str))
                ]
                if source_variant_order:
                    source_variant = st.selectbox(
                        "Source-comparison model",
                        options=source_variant_order,
                        format_func=lambda value: PREDICTION_MODEL_LABELS.get(
                            value, value
                        ),
                        key="targeted_eigengene_source_model_variant",
                    )
                    source_metric_rows = source_performance.loc[
                        source_performance["model_variant"].eq(source_variant)
                        & source_performance["metric"].eq(primary_metric)
                    ].copy()
                    if not source_metric_rows.empty:
                        source_metric_rows["model_variant"] = source_metric_rows[
                            "eigengene_source"
                        ]
                        render_plotly_chart(
                            prediction_performance_figure(
                                source_metric_rows,
                                metric=primary_metric,
                                block_labels=PREDICTION_BLOCK_LABELS,
                                block_order=PREDICTION_BLOCK_ORDER,
                                model_labels=EIGENGENE_SOURCE_LABELS,
                                title="OOF performance by eigengene source",
                            ),
                            use_container_width=True,
                            config={"displaylogo": False},
                            plot_id="render_targeted_prediction_view.prediction_performance_figure.3",
                        )
                    predictor_counts = source_performance.loc[
                        source_performance["model_variant"].eq(source_variant),
                        [
                            "eigengene_source",
                            "predictor_block",
                            "n_transcriptomic_predictors",
                        ],
                    ].drop_duplicates()
                    if not predictor_counts.empty:
                        predictor_counts["eigengene_source"] = predictor_counts[
                            "eigengene_source"
                        ].map(lambda value: EIGENGENE_SOURCE_LABELS.get(value, value))
                        predictor_counts["predictor_block"] = predictor_counts[
                            "predictor_block"
                        ].map(lambda value: PREDICTION_BLOCK_LABELS.get(value, value))
                        filterable_dataframe(
                            predictor_counts.sort_values(
                                ["eigengene_source", "n_transcriptomic_predictors"]
                            ),
                            table_key="targeted_eigengene_source_predictor_counts",
                            table_name="Eigengene predictor counts",
                            use_container_width=True,
                            hide_index=True,
                        )
                    if st.checkbox(
                        "Load outer-fold source robustness",
                        value=False,
                        key="targeted_load_eigengene_source_folds",
                        help="Loads the larger fold-level table only when requested.",
                    ):
                        source_folds = cached_targeted_prediction_table(
                            (
                                "hedges_fold_performance"
                                if hedges_selection
                                else "fold_performance"
                            ),
                            _targeted_filters(
                                evidence_tier=evidence_tier,
                                module_definition=module_definition,
                                network_method=network_method,
                                panel_strategy=panel_strategy,
                                model_outcome=outcome,
                                edge_mask=edge_mask,
                                score_normalization="standard_pruned",
                                score_transform="raw",
                                model_variant=source_variant,
                                metric=primary_metric,
                            ),
                        )
                        source_folds = source_folds.loc[
                            source_folds["eigengene_source"].isin(completed_sources)
                        ].copy()
                        if not source_folds.empty:
                            source_folds["source_block"] = source_folds.apply(
                                lambda row: (
                                    f"{row['predictor_block']}__{row['eigengene_source']}"
                                ),
                                axis=1,
                            )
                            source_block_labels = {
                                value: (
                                    f"{PREDICTION_BLOCK_LABELS.get(value.split('__', 1)[0], value)} · "
                                    f"{EIGENGENE_SOURCE_LABELS.get(value.split('__', 1)[1], value)}"
                                )
                                for value in source_folds["source_block"]
                                .astype(str)
                                .unique()
                            }
                            source_folds["predictor_block"] = source_folds[
                                "source_block"
                            ]
                            render_plotly_chart(
                                targeted_fold_robustness_figure(
                                    source_folds,
                                    metric=primary_metric,
                                    block_labels=source_block_labels,
                                    title="Outer-fold robustness by eigengene source",
                                ),
                                use_container_width=True,
                                config={"displaylogo": False},
                                plot_id="render_targeted_prediction_view.targeted_fold_robustness_figure.2",
                            )
                if hedges_selection:
                    st.info(
                        "The chart and fold view above compare the two independent regional "
                        "eigengene sources under the selected fold-local Hedges’ g mask. Paired "
                        "bootstrap source intervals will be added at final reconciliation."
                    )
                    source_comparisons = pd.DataFrame()
                elif (
                    "targeted_eigengene_source_comparisons.parquet"
                    not in manifest.get("files", {})
                ):
                    st.info(
                        "Paired source-comparison intervals will appear after the first "
                        "independent regional eigengene source finishes and passes validation."
                    )
                    source_comparisons = pd.DataFrame()
                else:
                    source_comparisons = cached_targeted_prediction_table(
                        "eigengene_source_comparisons",
                        _targeted_filters(
                            evidence_tier=evidence_tier,
                            module_definition=module_definition,
                            network_method=network_method,
                            panel_strategy=panel_strategy,
                            model_outcome=outcome,
                            score_transform="raw",
                        ),
                    )
                    source_comparisons = source_comparisons.loc[
                        source_comparisons["source_a"].eq(eigengene_source)
                        | source_comparisons["source_b"].eq(eigengene_source)
                    ].copy()
                if source_comparisons.empty:
                    if (
                        not hedges_selection
                        and "targeted_eigengene_source_comparisons.parquet"
                        in manifest.get("files", {})
                    ):
                        st.info(
                            "No paired source comparison matches the selected analysis."
                        )
                else:
                    st.caption(
                        "Exploratory paired comparisons use identical donors and outer folds. "
                        "Positive differences favor Source A. Global and within-outcome BH FDRs "
                        "are retained; these comparisons do not change the prespecified primary tests."
                    )
                    render_plotly_chart(
                        targeted_eigengene_source_comparison_figure(
                            source_comparisons,
                            block_labels=PREDICTION_BLOCK_LABELS,
                            model_labels=PREDICTION_MODEL_LABELS,
                            title="Eigengene-source sensitivity: paired OOF performance",
                        ),
                        use_container_width=True,
                        config={"displaylogo": False},
                        plot_id="render_targeted_prediction_view.targeted_eigengene_source_comparison_figure",
                    )
                    filterable_dataframe(
                        source_comparisons,
                        table_key="targeted_eigengene_source_comparisons",
                        table_name="Paired eigengene-source comparisons",
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        "Download eigengene-source comparisons (TSV)",
                        data=dataframe_to_tsv_bytes(source_comparisons),
                        file_name="targeted_lioness_eigengene_source_comparisons.tsv",
                        mime="text/tab-separated-values",
                    )

    with panel_tab:
        if tab_is_open(panel_tab):
            if panel_strategy == "all_modules":
                st.info("All-module benchmark uses no data-derived targeted panel.")
            else:
                panel_filters = _targeted_filters(
                    module_definition=module_definition,
                    network_method=network_method,
                    panel_strategy=panel_strategy,
                    selection_outcome=str(
                        outcome_catalog.loc[
                            outcome_catalog["panel_strategy"].eq(panel_strategy),
                            "selection_outcome",
                        ].iloc[0]
                    ),
                    score_transform=score_transform,
                )
                consensus = cached_targeted_prediction_table(
                    "consensus_panels", panel_filters
                )
                selection = cached_targeted_prediction_table(
                    "panel_selection", panel_filters
                )
                if consensus.empty:
                    st.info(
                        "Consensus selection results are unavailable for this panel."
                    )
                else:
                    render_plotly_chart(
                        targeted_selection_frequency_figure(
                            consensus,
                            title="Display-only consensus panel stability",
                        ),
                        use_container_width=True,
                        config={"displaylogo": False},
                        plot_id="render_targeted_prediction_view.targeted_selection_frequency_figure",
                    )
                    st.caption(
                        "KEGG annotations were joined only after selection and are interpretive only."
                    )
                    filterable_dataframe(
                        consensus,
                        table_key="targeted_consensus_panel",
                        table_name="Targeted consensus panel",
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        "Download consensus panel (TSV)",
                        data=dataframe_to_tsv_bytes(consensus),
                        file_name="targeted_lioness_consensus_panel.tsv",
                        mime="text/tab-separated-values",
                    )
                if not selection.empty:
                    k_distribution = (
                        selection[["outer_repeat", "outer_fold", "selected_k"]]
                        .drop_duplicates()
                        .value_counts("selected_k")
                        .rename("outer_folds")
                        .reset_index()
                    )
                    filterable_dataframe(
                        k_distribution,
                        table_key="targeted_k_distribution",
                        table_name="One-standard-error K distribution",
                        use_container_width=True,
                        hide_index=True,
                    )

    with diagnostic_tab:
        if tab_is_open(diagnostic_tab):
            blocks = ordered_prediction_blocks(selected["predictor_block"])
            diagnostic_block = st.selectbox(
                "OOF diagnostic predictor block",
                options=blocks,
                format_func=lambda value: PREDICTION_BLOCK_LABELS.get(value, value),
                key="targeted_diagnostic_block",
            )
            variants = (
                selected.loc[
                    selected["predictor_block"].eq(diagnostic_block), "model_variant"
                ]
                .drop_duplicates()
                .tolist()
            )
            diagnostic_variant = st.selectbox(
                "OOF diagnostic model",
                options=variants,
                format_func=lambda value: PREDICTION_MODEL_LABELS.get(value, value),
                index=(
                    variants.index("covariates_plus_network")
                    if "covariates_plus_network" in variants
                    else 0
                ),
                key="targeted_diagnostic_variant",
            )
            diagnostics = cached_targeted_prediction_table(
                diagnostic_table,
                _targeted_filters(
                    evidence_tier=evidence_tier,
                    module_definition=module_definition,
                    network_method=network_method,
                    panel_strategy=panel_strategy,
                    model_outcome=outcome,
                    predictor_block=diagnostic_block,
                    model_variant=diagnostic_variant,
                    edge_mask=edge_mask,
                    score_normalization=score_normalization,
                    score_transform=score_transform,
                    eigengene_source=(
                        eigengene_source
                        if diagnostic_variant in transcriptomic_variants
                        else "not_applicable"
                    ),
                ),
            )
            if diagnostics.empty:
                st.info("OOF donor diagnostics are unavailable for this model.")
            elif outcome in {
                "diagnosis_binary",
                "diagnosis_three_class",
                "parkinsonism",
                "clusters",
            }:
                confusion = (
                    diagnostics.groupby(["target", "predicted"], observed=True)
                    .size()
                    .rename("n")
                    .reset_index()
                    .rename(
                        columns={"target": "actual", "predicted": "predicted_class"}
                    )
                )
                render_plotly_chart(
                    prediction_confusion_figure(
                        confusion, title="Donor-averaged OOF confusion matrix"
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                    plot_id="render_targeted_prediction_view.prediction_confusion_figure",
                )
                diagnostic_curves = classification_diagnostic_rows(diagnostics)
                if not diagnostic_curves.empty:
                    curve_columns = st.columns(3)
                    for curve_column, curve_name in zip(
                        curve_columns,
                        ("ROC", "Precision-recall", "Calibration"),
                        strict=True,
                    ):
                        with curve_column:
                            render_plotly_chart(
                                prediction_curve_figure(
                                    diagnostic_curves,
                                    curve=curve_name,
                                    title=f"OOF {curve_name}",
                                ),
                                use_container_width=True,
                                config={"displaylogo": False},
                                plot_id=f"targeted_prediction.curve.{curve_name}",
                            )
                if (
                    len(
                        [
                            column
                            for column in diagnostics
                            if column.startswith("probability_")
                        ]
                    )
                    == 2
                ):
                    render_plotly_chart(
                        prediction_threshold_figure(
                            diagnostics,
                            title="Donor-averaged OOF threshold diagnostics",
                        ),
                        use_container_width=True,
                        config={"displaylogo": False},
                        plot_id="render_targeted_prediction_view.prediction_threshold_figure",
                    )
            else:
                render_plotly_chart(
                    prediction_observed_figure(
                        diagnostics,
                        title="Observed versus donor-averaged OOF prediction",
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                    plot_id="render_targeted_prediction_view.prediction_observed_figure",
                )
                render_plotly_chart(
                    prediction_error_figure(
                        diagnostics, title="OOF residual diagnostics"
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                    plot_id="render_targeted_prediction_view.prediction_error_figure",
                )
            if not diagnostics.empty:
                filterable_dataframe(
                    diagnostics,
                    table_key="targeted_oof_diagnostics",
                    table_name="Sanitized donor-averaged OOF predictions",
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download selected OOF predictions (TSV)",
                    data=dataframe_to_tsv_bytes(diagnostics),
                    file_name="targeted_lioness_oof_predictions.tsv",
                    mime="text/tab-separated-values",
                )

    with coefficient_tab:
        if tab_is_open(coefficient_tab):
            coefficients = cached_targeted_prediction_table(
                coefficient_table,
                _targeted_filters(
                    evidence_tier=evidence_tier,
                    module_definition=module_definition,
                    network_method=network_method,
                    panel_strategy=panel_strategy,
                    model_outcome=outcome,
                    edge_mask=edge_mask,
                    score_normalization=score_normalization,
                    score_transform=score_transform,
                ),
            )
            coefficients = coefficients.loc[
                ~coefficients["model_variant"].isin(transcriptomic_variants)
                | coefficients["eigengene_source"].eq(eigengene_source)
            ].copy()
            if coefficients.empty:
                st.info("No nonzero coefficients are available for this model family.")
            else:
                coefficient_block = st.selectbox(
                    "Targeted coefficient block",
                    options=ordered_prediction_blocks(coefficients["predictor_block"]),
                    format_func=lambda value: PREDICTION_BLOCK_LABELS.get(value, value),
                    key="targeted_coefficient_block",
                )
                coefficient_variants = (
                    coefficients.loc[
                        coefficients["predictor_block"].eq(coefficient_block),
                        "model_variant",
                    ]
                    .drop_duplicates()
                    .tolist()
                )
                coefficient_variant = st.selectbox(
                    "Targeted coefficient model",
                    options=coefficient_variants,
                    format_func=lambda value: PREDICTION_MODEL_LABELS.get(value, value),
                    index=(
                        coefficient_variants.index("covariates_plus_network")
                        if "covariates_plus_network" in coefficient_variants
                        else 0
                    ),
                    key="targeted_coefficient_variant",
                )
                shown = coefficients.loc[
                    coefficients["predictor_block"].eq(coefficient_block)
                    & coefficients["model_variant"].eq(coefficient_variant)
                ].copy()
                shown = annotate_prediction_coefficients(
                    shown, cached_coefficient_kegg_scopes(module_definition)
                )
                render_plotly_chart(
                    prediction_coefficient_figure(
                        shown,
                        title="Largest standardized outer-fold coefficients",
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                    plot_id="render_targeted_prediction_view.prediction_coefficient_figure",
                )
                st.caption(
                    "KEGG enrichment is interpretive only and never enters ranking or fitting."
                )
                filterable_dataframe(
                    shown,
                    table_key="targeted_coefficients",
                    table_name="Fold coefficients with post-selection KEGG annotations",
                    use_container_width=True,
                    hide_index=True,
                )
                module_coefficients = shown.loc[shown["module"].notna()].copy()
                if not module_coefficients.empty:
                    module_coefficients["component"] = (
                        module_coefficients["feature_name"]
                        .astype(str)
                        .str.split("__", n=1)
                        .str[1]
                        .str.replace("MFBA9BA46", "DLPFC", regex=False)
                    )
                    component_summary = (
                        module_coefficients.groupby(
                            ["coefficient_label", "component"], observed=True
                        )
                        .agg(
                            outer_fold_coefficients=(
                                "standardized_coefficient",
                                "size",
                            ),
                            coefficient_sum=("standardized_coefficient", "sum"),
                            absolute_coefficient_sum=(
                                "abs_standardized_coefficient",
                                "sum",
                            ),
                            median_absolute_coefficient=(
                                "abs_standardized_coefficient",
                                "median",
                            ),
                        )
                        .reset_index()
                    )
                    filterable_dataframe(
                        component_summary,
                        table_key="targeted_component_coefficients",
                        table_name="Component-level coefficient summary",
                        use_container_width=True,
                        hide_index=True,
                    )
                st.download_button(
                    "Download selected coefficients (TSV)",
                    data=dataframe_to_tsv_bytes(shown),
                    file_name="targeted_lioness_coefficients.tsv",
                    mime="text/tab-separated-values",
                )

    with tables_tab:
        if tab_is_open(tables_tab):
            filterable_dataframe(
                selected,
                table_key="targeted_oof_performance",
                table_name="Donor-averaged OOF performance",
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download selected OOF metrics (TSV)",
                data=dataframe_to_tsv_bytes(selected),
                file_name="targeted_lioness_nested_cv_metrics.tsv",
                mime="text/tab-separated-values",
            )

    with methods_tab:
        if tab_is_open(methods_tab):
            st.markdown(
                "Each outer-test fold is excluded from the LIONESS reference, panel ranking, "
                "redundancy pruning, K selection, preprocessing, and elastic-net tuning. Standard "
                "LIONESS uses outer-training donors; Control-referenced LIONESS uses outer-training "
                "Controls. Test donors are add-one scored against the frozen reference. Panel size "
                "and regularization are selected inside the outer-training partition."
            )
            st.markdown(
                "**Score transformations.** Raw uses `metric_raw` and is the only primary scale. "
                "asinh applies `arcsinh(metric_raw)`. RINT uses average ranks and is fitted anew "
                "inside every stability-selection subsample, inner-training fold, and outer-training "
                "fold; validation/test values are mapped through the training-derived monotone "
                "interpolation and clipped outside the training range. Panels and K may differ by "
                "transformation."
            )
            st.markdown(
                "**Transcriptomic comparator.** For every module, tissue-specific PCA1 "
                "eigengenes use training-only gene medians, scaling, loadings, and deterministic "
                "sign alignment. A single-tissue block uses that tissue; a tissue-pair block uses "
                "the two represented tissues; pooled CT/TS and resolved blocks concatenate AC, "
                "DLPFC, and PCG eigengenes without a second PCA. The matched source uses the same "
                "frozen LIONESS panel. Independent single-region sources instead use all level-3 "
                "regional modules from the represented tissues; joint models combine those "
                "eigengenes with the unchanged fold-selected LIONESS panel. Expression never "
                "contributes to LIONESS panel selection."
            )
            st.caption(
                "Regional partitions are fixed unsupervised structures discovered from broader "
                "expression cohorts. Eigengene preprocessing and PCA loadings are outer-training "
                "only, but partition discovery itself is not external validation."
            )
            st.caption(
                "The current fixed 70/30 held-out results are previously inspected sensitivity "
                "data and cannot sort, tune, or modify a targeted panel. Module discovery used the "
                "broader cohort, so this is not external validation."
            )
            st.json(manifest, expanded=False)


def render_prediction_view() -> None:
    st.caption(
        "Prediction catalogs are available only for the matched full-cohort and "
        "Control-derived module definitions. The all-donor CorShrink definition was "
        "intentionally added as a non-prediction analysis."
    )
    """Render the leakage-reduced LIONESS held-out prediction catalog lazily."""

    targeted_available = targeted_prediction_data_available()
    protocol_available = protocol_v11_available()
    if targeted_available or protocol_available:
        mode_options = []
        if targeted_available:
            mode_options.append("targeted")
        mode_options.append("benchmark")
        if protocol_available:
            mode_options.append("protocol_v11")
        prediction_mode = st.radio(
            "Prediction mode",
            options=mode_options,
            format_func=lambda value: (
                "Experiment 0210 — Protocol v1.1"
                if value == "protocol_v11"
                else TARGETED_PREDICTION_MODE_LABELS[value]
            ),
            horizontal=True,
            key="prediction_mode",
        )
        if prediction_mode == "protocol_v11":
            render_protocol_v11_view()
            return
        if prediction_mode == "targeted":
            render_targeted_prediction_view()
            return

    st.subheader("Leakage-reduced LIONESS prediction")
    if targeted_available:
        st.warning(
            "Previously inspected sensitivity data: this benchmark catalog remains useful for "
            "comparison, but it is not independent validation and cannot define or tune a "
            "targeted module panel."
        )
    st.caption(
        "The module definitions are fixed from the prior analysis, but reference networks, "
        "AD–Control edge masks, preprocessing, and elastic-net tuning use development data "
        "only. Held-out donors are add-one scored against frozen references. This is therefore "
        "leakage-reduced conditional on the module definitions, not a fully external validation."
    )
    if not prediction_data_available():
        st.info(
            "Prediction results have not yet been added to this deployed data bundle. "
            "The rest of the network explorer remains available."
        )
        return
    manifest = load_prediction_manifest()
    with st.sidebar:
        st.header("Prediction controls")
        reference_provenance = st.selectbox(
            "Prediction reference design",
            options=list(PREDICTION_REFERENCE_LABELS),
            format_func=lambda value: PREDICTION_REFERENCE_LABELS[value],
            index=0,
            key="prediction_reference_provenance",
        )
        module_definition = st.selectbox(
            "Prediction module definition",
            options=list(PREDICTION_MODULE_SETS),
            format_func=lambda value: MODULE_SET_LABELS[value],
            key="prediction_module_definition",
        )
        methods = MODULE_SET_METHODS[module_definition]
        network_method = st.selectbox(
            "LIONESS method",
            options=methods,
            format_func=lambda value: (
                (
                    "Development-standard LIONESS (314-donor frozen reference)"
                    if value == "standard"
                    else "Development-Control LIONESS (114-Control frozen reference)"
                )
                if reference_provenance == "development_frozen"
                else readable_method(value)
            ),
            key="prediction_network_method",
        )
        predictor_design = st.radio(
            "Predictor design",
            options=list(PREDICTION_DESIGN_LABELS),
            format_func=lambda value: PREDICTION_DESIGN_LABELS[value],
            key="prediction_design",
        )
        edge_mask = st.selectbox(
            "Development AD–Control edge mask",
            options=list(PREDICTION_MASK_LABELS),
            format_func=lambda value: PREDICTION_MASK_LABELS[value],
            index=list(PREDICTION_MASK_LABELS).index("per_module_fdr10"),
            key="prediction_edge_mask",
        )
        if predictor_design == "module_connectivity":
            score_normalization = st.radio(
                "Connectivity score",
                options=["standard_pruned", "retained_edge"],
                format_func=lambda value: SCORE_NORMALIZATION_LABELS[value],
                key="prediction_normalization",
            )
        else:
            score_normalization = "not_applicable"
        outcome = st.selectbox(
            "Prediction target",
            options=[
                value for value in PREDICTION_OUTCOME_LABELS if value != "clusters"
            ],
            format_func=lambda value: PREDICTION_OUTCOME_LABELS[value],
            key="prediction_outcome",
        )

    performance = cached_prediction_performance(
        reference_provenance,
        module_definition,
        network_method,
        predictor_design,
        edge_mask,
        score_normalization,
        None,
    )
    if performance.empty:
        st.warning("No fitted models are available for these prediction controls.")
        return
    if reference_provenance == "existing_sensitivity":
        st.warning(
            "Exploratory sensitivity only: these scores use the previously established "
            "all-donor Standard reference or all-Control Control reference. Held-out donor "
            "expression therefore contributed to the unsupervised reference construction; "
            "do not interpret this option as the primary leakage-reduced evaluation."
        )
    selected_performance = performance.loc[performance["outcome"].eq(outcome)].copy()
    if selected_performance.empty:
        st.warning(
            "The selected target is unavailable for this predictor configuration."
        )
        return
    primary_metric = str(selected_performance["primary_metric"].dropna().iloc[0])
    unavailable = selected_performance.loc[
        selected_performance["status"].ne("available")
    ]
    if not unavailable.empty:
        st.warning(
            f"{len(unavailable)} model result(s) are unavailable because the selected "
            "edge mask left no variable development predictors or fitting was not estimable."
        )
    st.info(
        "CogDx and diagnosis are not independent replications: diagnosis is essentially a "
        "grouped version of CogDx in this cohort. Outcomes with missing values are evaluated "
        "only in donors with an observed target; outcomes are never imputed."
    )
    summary_columns = st.columns(4)
    summary_columns[0].metric(
        "Development donors", int(selected_performance["n_development"].max())
    )
    summary_columns[1].metric(
        "Held-out donors", int(selected_performance["n_held_out"].max())
    )
    summary_columns[2].metric("Primary metric", primary_metric)
    summary_columns[3].metric("Edge mask", PREDICTION_MASK_LABELS[edge_mask])

    (
        overview_tab,
        comparison_tab,
        diagnostics_tab,
        coefficient_tab,
        tables_tab,
        methods_tab,
    ) = stateful_tabs(
        [
            "Performance",
            "CT versus TS",
            "Diagnostics",
            "Coefficients",
            "Tables",
            "Methods",
        ],
        key="benchmark_prediction_tabs",
    )
    with overview_tab:
        if tab_is_open(overview_tab):
            figure = prediction_performance_figure(
                selected_performance,
                metric=primary_metric,
                block_labels=PREDICTION_BLOCK_LABELS,
                block_order=PREDICTION_BLOCK_ORDER,
                model_labels=PREDICTION_MODEL_LABELS,
                title=f"Held-out {PREDICTION_OUTCOME_LABELS[outcome]} performance",
            )
            render_plotly_chart(
                figure,
                use_container_width=True,
                config={"displaylogo": False},
                plot_id="render_prediction_view.figure",
            )
            st.caption(
                "Lower is better for CogDx MAE. Higher is better for all other primary "
                "metrics; held-out R² may be negative when a model performs worse than "
                "predicting the held-out outcome mean."
            )
            adjusted = performance.loc[
                performance["model_variant"].eq("covariates_plus_network")
                & performance["status"].eq("available")
                & performance["metric"].eq(performance["primary_metric"])
            ].copy()
            adjusted.loc[~adjusted["higher_is_better"].astype(bool), "value"] *= -1
            adjusted["metric"] = "primary_performance"
            if not adjusted.empty:
                render_plotly_chart(
                    prediction_heatmap_figure(
                        adjusted,
                        metric="primary_performance",
                        block_labels=PREDICTION_BLOCK_LABELS,
                        block_order=PREDICTION_BLOCK_ORDER,
                        outcome_labels=PREDICTION_OUTCOME_LABELS,
                        title="Covariate-adjusted held-out primary performance by outcome and component block",
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                    plot_id="render_prediction_view.prediction_heatmap_figure",
                )
                st.caption(
                    "Cells use each outcome's prespecified primary metric. CogDx is displayed as "
                    "negative MAE so that larger values consistently mean better performance; the "
                    "numerical scales are still not directly interchangeable across columns."
                )
            if predictor_design == "edge_sums":
                with st.expander("Whole-network retained-edge sums", expanded=False):
                    whole_network = cached_prediction_whole_network(
                        reference_provenance=reference_provenance,
                        module_definition=module_definition,
                        network_method=network_method,
                        edge_mask=edge_mask,
                    )
                    st.caption(
                        "Positive and negative magnitudes are the two independent model inputs. "
                        "Signed and absolute sums are retained here for interpretation and download. "
                        "These totals aggregate only retained within-module edges; no between-module "
                        "edges are introduced."
                    )
                    filterable_dataframe(
                        whole_network,
                        table_key="prediction_whole_network_features",
                        table_name="Whole-network edge-sum predictors",
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        "Download whole-network predictor values (TSV)",
                        data=dataframe_to_tsv_bytes(whole_network),
                        file_name="lioness_prediction_whole_network_edge_sums.tsv",
                        mime="text/tab-separated-values",
                    )

    with comparison_tab:
        if tab_is_open(comparison_tab):
            bootstrap = cached_prediction_bootstrap(
                reference_provenance,
                module_definition,
                network_method,
                predictor_design,
                edge_mask,
                score_normalization,
                None,
            )
            if bootstrap.empty:
                st.info(
                    "CT-versus-TS bootstrap comparisons are unavailable for this selection."
                )
            else:
                render_plotly_chart(
                    prediction_ct_ts_figure(
                        bootstrap,
                        outcome_labels=PREDICTION_OUTCOME_LABELS,
                        title="Paired held-out CT-minus-TS performance with 95% bootstrap intervals",
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                    plot_id="render_prediction_view.prediction_ct_ts_figure",
                )
                st.caption(
                    "Positive values favor CT. For CogDx, the sign is reversed after calculating "
                    "MAE so that positive still means better CT performance. Global BH covers all "
                    "displayed prediction comparisons; within-outcome BH is also retained."
                )
                filterable_dataframe(
                    bootstrap,
                    table_key="prediction_bootstrap",
                    table_name="CT versus TS bootstrap comparisons",
                    use_container_width=True,
                    hide_index=True,
                )

    with diagnostics_tab:
        if tab_is_open(diagnostics_tab):
            available_blocks = [
                block
                for block in PREDICTION_BLOCK_ORDER
                if block in set(selected_performance["predictor_block"])
            ]
            diagnostic_block = st.selectbox(
                "Diagnostic predictor block",
                options=available_blocks,
                format_func=lambda value: PREDICTION_BLOCK_LABELS[value],
            )
            diagnostic_variant = st.radio(
                "Diagnostic model",
                options=[
                    value
                    for value in (
                        "network_only",
                        "covariates_plus_network",
                        "transcriptomics_only",
                        "covariates_plus_transcriptomics",
                        "network_plus_transcriptomics",
                        "covariates_plus_network_plus_transcriptomics",
                    )
                    if value in set(selected_performance["model_variant"])
                ],
                format_func=lambda value: PREDICTION_MODEL_LABELS[value],
                horizontal=True,
            )
            filters = {
                "reference_provenance": reference_provenance,
                "module_definition": module_definition,
                "network_method": network_method,
                "predictor_design": predictor_design,
                "edge_mask": edge_mask,
                "score_normalization": score_normalization,
                "outcome": outcome,
                "predictor_block": diagnostic_block,
                "model_variant": diagnostic_variant,
            }
            diagnostics = cached_prediction_diagnostics(**filters)
            if outcome in {"diagnosis_binary", "diagnosis_three_class", "parkinsonism"}:
                curves = cached_prediction_curves(**filters)
                confusion = cached_prediction_confusion(**filters)
                if curves.empty:
                    st.info(
                        "Classification curves are unavailable for this fitted model."
                    )
                else:
                    curve_columns = st.columns(3)
                    for column, curve_name in zip(
                        curve_columns,
                        ["ROC", "Precision-recall", "Calibration"],
                        strict=True,
                    ):
                        with column:
                            render_plotly_chart(
                                prediction_curve_figure(
                                    curves, curve=curve_name, title=curve_name
                                ),
                                use_container_width=True,
                                config={"displaylogo": False},
                                plot_id=f"prediction.curve.{curve_name}",
                            )
                if not confusion.empty:
                    render_plotly_chart(
                        prediction_confusion_figure(
                            confusion, title="Held-out confusion matrix"
                        ),
                        use_container_width=True,
                        config={"displaylogo": False},
                        plot_id="render_prediction_view.prediction_confusion_figure",
                    )
                if (
                    outcome in {"diagnosis_binary", "parkinsonism"}
                    and not diagnostics.empty
                ):
                    render_plotly_chart(
                        prediction_threshold_figure(
                            diagnostics, title="Held-out threshold diagnostics"
                        ),
                        use_container_width=True,
                        config={"displaylogo": False},
                        plot_id="render_prediction_view.prediction_threshold_figure",
                    )
            elif diagnostics.empty:
                st.info(
                    "Held-out donor diagnostics are unavailable in the local cache."
                )
            else:
                render_plotly_chart(
                    prediction_observed_figure(
                        diagnostics,
                        title=f"Observed versus held-out prediction: {PREDICTION_OUTCOME_LABELS[outcome]}",
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                    plot_id="render_prediction_view.prediction_observed_figure",
                )
                render_plotly_chart(
                    prediction_error_figure(
                        diagnostics,
                        title="Held-out residual and error diagnostics",
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                    plot_id="render_prediction_view.prediction_error_figure",
                )
            if not diagnostics.empty:
                filterable_dataframe(
                    diagnostics,
                    table_key="prediction_diagnostics",
                    table_name="Held-out prediction diagnostics",
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download selected held-out diagnostics (TSV)",
                    data=dataframe_to_tsv_bytes(diagnostics),
                    file_name="lioness_prediction_heldout_diagnostics.tsv",
                    mime="text/tab-separated-values",
                )

    with coefficient_tab:
        if tab_is_open(coefficient_tab):
            coefficient_filters = {
                "reference_provenance": reference_provenance,
                "module_definition": module_definition,
                "network_method": network_method,
                "predictor_design": predictor_design,
                "edge_mask": edge_mask,
                "score_normalization": score_normalization,
                "outcome": outcome,
            }
            coefficients = cached_prediction_coefficients(**coefficient_filters)
            coefficients = coefficients.loc[
                coefficients["model_variant"].isin(
                    [
                        "network_only",
                        "covariates_plus_network",
                        "transcriptomics_only",
                        "covariates_plus_transcriptomics",
                        "network_plus_transcriptomics",
                        "covariates_plus_network_plus_transcriptomics",
                    ]
                )
            ]
            if coefficients.empty:
                st.info(
                    "No nonzero influential coefficients are available for this selection."
                )
            else:
                coefficient_block = st.selectbox(
                    "Coefficient predictor block",
                    options=ordered_prediction_blocks(coefficients["predictor_block"]),
                    format_func=lambda value: PREDICTION_BLOCK_LABELS.get(value, value),
                )
                coefficient_variant = st.radio(
                    "Coefficient model",
                    options=sorted(coefficients["model_variant"].unique()),
                    format_func=lambda value: PREDICTION_MODEL_LABELS.get(value, value),
                    horizontal=True,
                    key="prediction_coefficient_variant",
                )
                shown = coefficients.loc[
                    coefficients["predictor_block"].eq(coefficient_block)
                    & coefficients["model_variant"].eq(coefficient_variant)
                ].copy()
                shown = annotate_prediction_coefficients(
                    shown, cached_coefficient_kegg_scopes(module_definition)
                )
                render_plotly_chart(
                    prediction_coefficient_figure(
                        shown,
                        title="Largest standardized module/component coefficients",
                    ),
                    use_container_width=True,
                    config={"displaylogo": False},
                    plot_id="render_prediction_view.prediction_coefficient_figure",
                )
                filterable_dataframe(
                    shown,
                    table_key="prediction_coefficients",
                    table_name="Prediction coefficients and KEGG annotations",
                    use_container_width=True,
                    hide_index=True,
                )
                module_coefficients = shown.loc[shown["module"].notna()].copy()
                if not module_coefficients.empty:
                    module_coefficients["component"] = (
                        module_coefficients["feature_name"]
                        .astype(str)
                        .str.split("__", n=1)
                        .str[1]
                        .str.replace("MFBA9BA46", "DLPFC", regex=False)
                    )
                    component_summary = (
                        module_coefficients.groupby(
                            ["coefficient_label", "component"], observed=True
                        )
                        .agg(
                            modules_shown=("module", "nunique"),
                            coefficient_sum=("standardized_coefficient", "sum"),
                            absolute_coefficient_sum=(
                                "abs_standardized_coefficient",
                                "sum",
                            ),
                            median_absolute_coefficient=(
                                "abs_standardized_coefficient",
                                "median",
                            ),
                        )
                        .reset_index()
                    )
                    filterable_dataframe(
                        component_summary,
                        table_key="prediction_component_coefficients",
                        table_name="Component-level coefficient summary",
                        use_container_width=True,
                        hide_index=True,
                    )

    with tables_tab:
        if tab_is_open(tables_tab):
            filterable_dataframe(
                selected_performance,
                table_key="prediction_performance",
                table_name="Held-out model metrics",
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download selected held-out metrics (TSV)",
                data=dataframe_to_tsv_bytes(selected_performance),
                file_name="lioness_prediction_heldout_metrics.tsv",
                mime="text/tab-separated-values",
            )

    with methods_tab:
        if tab_is_open(methods_tab):
            st.markdown(
                "Development-standard LIONESS uses all 314 development donors as its frozen "
                "reference. Development donors are leave-one-out scored and held-out donors are "
                "add-one scored. Development-Control LIONESS uses the 114 development Controls; "
                "all other donors are add-one scored without using their outcome label. Differential "
                "edge masks use only the 117 development AD and 114 development Controls. Models "
                "use elastic-net regularization with five-fold development-only cross-validation "
                "before one held-out evaluation. Logistic models use a 50/50 L1/L2 mix and tune "
                "three regularization strengths; continuous models tune both the mixing fraction "
                "and regularization strength."
            )
            st.markdown(
                "Module eigengenes are fitted on development expression only and applied to the "
                "held-out donors with frozen gene preprocessing and PCA loadings. Tissue and "
                "tissue-pair blocks use only their represented regions; pooled blocks concatenate "
                "AC, DLPFC, and PCG eigengenes. Transcriptomic-only and joint network–transcriptomic "
                "models are displayed beside the dummy, demographic/APOE, and network models."
            )
            st.caption(
                "For prediction only, 30 exact repeated full-cohort tissue–gene assignment "
                "rows were removed before scoring in both reference designs. Existing "
                "association analyses elsewhere in the app were not changed."
            )
            st.json(manifest, expanded=False)
