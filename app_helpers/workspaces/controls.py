from __future__ import annotations

from app_helpers.ui_runtime import (
    ANALYSIS_SUBSET_LABELS,
    ASSOCIATION_GROUP_LABELS,
    ASSOCIATION_OUTCOME_LABELS,
    BONOBO_EDGE_RULE_LABELS,
    BONOBO_FEATURE_LABELS,
    CAMEL_EXPANSION,
    CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES,
    COHORT_SCOPE_LABELS,
    COLOR_LABELS,
    CONTINUOUS_COLOR_SCALES,
    DIAGNOSIS_ORDER,
    EDGE_SCOPE_LABELS,
    EFFECT_DIRECTION_LABELS,
    EFFECT_STATISTIC_LABELS,
    EIGENGENE_FEATURE_LABELS,
    FDR_SCOPE_LABELS,
    FDR_THRESHOLD_LABELS,
    FEATURE_LABELS,
    HEDGES_FIXED_THRESHOLDS,
    MODULE_SET_LABELS,
    MODULE_SET_METHODS,
    MODULE_TOP_FRACTIONS,
    MUSCLE_MODULE_SETS,
    OUTCOME_LABELS,
    PATHOLOGY_OUTCOMES,
    SCALE_LABELS,
    SCORE_NORMALIZATION_LABELS,
    SELECTABLE_ASSOCIATION_OUTCOMES,
    VALIDATION_SCOPE,
    _camel_validation,
    association_level_label,
    available_module_sets,
    cached_annotations,
    cached_effect_size_manifest,
    cached_module_details,
    cached_sample_metadata,
    data_source_label,
    descriptive_eigengene_data_available,
    differential_data_available,
    effect_endpoint_kegg_available,
    effect_rule_key,
    effect_size_data_available,
    module_label,
    module_set_components,
    module_set_tissue_detail_columns,
    module_set_tissues,
    np,
    ordered_association_levels,
    pd,
    readable_method,
    render_plotly_chart,
    score_framework,
    st,
)


def _choice(control, visible, *args, **kwargs):
    """Reset incompatible source-scoped widget values before rendering them."""
    options = list(kwargs.get("options", args[1] if len(args) > 1 else ()))
    key = kwargs.get("key")
    if visible and key and key in st.session_state:
        current = st.session_state[key]
        if control.__name__ == "multiselect":
            valid = [value for value in current if value in options]
            if current and not valid:
                valid = list(kwargs.get("default", []))
            if list(current) != valid:
                st.session_state[key] = valid
        elif current not in options:
            del st.session_state[key]
    if visible:
        return control(*args, **kwargs)
    if control.__name__ == "multiselect":
        return list(kwargs.get("default", []))
    index = kwargs.get("index", 0)
    return options[index] if options and index is not None else None


def build_controls(active_view, data_manifest):
    analysis_slot = st.empty()
    calibrated_body = st.container()
    with st.sidebar:
        st.header("Dataset")
        st.caption(f"Data source: {data_source_label()}")
        requested_partition = st.query_params.get("screen_partition")
        if requested_partition and requested_partition not in available_module_sets(
            data_manifest
        ):
            st.error(
                "The linked module definition is unavailable. No other dataset will be substituted."
            )
            if st.button("Clear unavailable dataset link"):
                st.query_params.pop("screen_partition", None)
                st.rerun()
            st.stop()
        module_set = _choice(
            st.selectbox,
            True,
            "Module definition",
            options=list(available_module_sets(data_manifest)),
            format_func=lambda value: MODULE_SET_LABELS[value],
            index=(
                list(available_module_sets(data_manifest)).index(
                    st.query_params["screen_partition"]
                )
                if st.query_params.get("screen_partition")
                in available_module_sets(data_manifest)
                else 0
            ),
            help=(
                "Identically numbered modules in different definitions have different gene "
                "memberships and are always loaded from separate data bundles."
            ),
            key="analysis_module_set",
        )
        module_set_label = MODULE_SET_LABELS[module_set]
        active_tissues = module_set_tissues(module_set)
        active_component_order = list(module_set_components(module_set))
        active_tissue_detail_columns = module_set_tissue_detail_columns(module_set)
        module_manifest = data_manifest.get("module_sets", {}).get(module_set, {})
        if module_set in MUSCLE_MODULE_SETS:
            st.info(
                "These muscle partitions were discovered with Method 5 extended "
                "SFN (bootstrap stability, CT null excess, signed block powers, and blockwise "
                "q90 normalization). The MDC context is rebuilt with that same M5 functional. "
                "The historical donor-specific LIONESS comparator uses Pearson signedAlt "
                "with the partition's frozen block powers. Control-calibrated connectivity "
                "reuses these fixed memberships with uniform scoring beta3 or beta1."
            )
        annotations = cached_annotations(module_set)
        modules = sorted(annotations["module"].astype(int).unique().tolist())
        module_count = int(module_manifest.get("modules", len(modules)))
        if module_count != len(modules):
            st.error(
                f"The {module_set_label} manifest declares {module_count} modules, but its "
                f"annotation table contains {len(modules)}."
            )
            st.stop()
        module_details = cached_module_details(module_set)
        with st.expander("Score source and method", expanded=False):
            allowed_estimators = list(
                module_manifest.get("estimators", ["lioness", "bonobo"])
            )
            from app_helpers.control_calibrated import registry as calibrated_registry

            if active_view != "MDC" and calibrated_registry.ready(module_set):
                allowed_estimators += list(calibrated_registry.ESTIMATORS)
            if (
                "network_estimator" in st.session_state
                and st.session_state["network_estimator"] not in allowed_estimators
            ):
                del st.session_state["network_estimator"]
            estimator = _choice(
                st.radio,
                True,
                "Network estimator",
                options=allowed_estimators,
                format_func=lambda value: score_framework(
                    module_set, "control_anchored", value
                ),
                horizontal=True,
                key="network_estimator",
                index=(
                    allowed_estimators.index(
                        st.query_params.get("score_source", "lioness")
                    )
                    if st.query_params.get("score_source", "lioness")
                    in allowed_estimators
                    else 0
                ),
            )
            if calibrated_registry.is_calibrated(estimator):
                from app_helpers.control_calibrated.ui import (
                    render_view as render_calibrated_view,
                )

                with calibrated_body:
                    render_calibrated_view(
                        active_view, module_set, estimator, render_plotly_chart
                    )
                st.stop()
            allowed_methods = (
                list(module_manifest.get("methods", MODULE_SET_METHODS[module_set]))
                if estimator == "lioness"
                else ["bonobo"]
            )
            capabilities = module_manifest.get("capabilities", {})
            if active_view in {
                "Edge summaries",
                "Edge volcano",
                "Donor edge explorer",
            } and not capabilities.get(
                (
                    "differential_edges"
                    if active_view == "Edge volcano"
                    else "edge_summaries"
                ),
                True,
            ):
                st.info(
                    f"{active_view} was intentionally not calculated for {module_set_label}. "
                    "Select one of the existing module definitions to use this view."
                )
                st.stop()
            method = _choice(
                st.radio,
                True,
                "Network method",
                options=allowed_methods,
                format_func=readable_method,
                index=(
                    allowed_methods.index(st.query_params["screen_method"])
                    if st.query_params.get("screen_method") in allowed_methods
                    else (
                        allowed_methods.index("control_anchored")
                        if "control_anchored" in allowed_methods
                        else 0
                    )
                ),
                key="analysis_method",
            )
            current_score_framework = score_framework(module_set, method, estimator)
            validation_source = (
                _camel_validation.for_dataset(module_set, method)
                if estimator == "lioness"
                and _camel_validation.supports_dataset(module_set, method)
                else None
            )
            if current_score_framework == "CAMEL":
                st.caption(
                    f"CAMEL: {CAMEL_EXPANSION}. Existing estimator, formulas and internal identifiers are unchanged."
                )
                if (
                    module_set == "control_derived"
                    and not _camel_validation.available()
                    and _camel_validation.manifest()
                ):
                    st.caption(
                        "Incomplete-region CAMEL validation is being prepared. It becomes selectable only after the matching data and statistics are verified in the backend."
                    )
        cohort_scope_options = ["complete_450"]
        if (
            module_set == "all_donor_corrshrink_l4"
            and capabilities.get("expanded_components", False)
            and estimator == "lioness"
        ):
            cohort_scope_options.append("maximum_component")
        if validation_source is not None and validation_source.available():
            cohort_scope_options.append(VALIDATION_SCOPE)
        cohort_scope = _choice(
            st.radio,
            True,
            "Donor cohort scope",
            options=cohort_scope_options,
            format_func=lambda value: (
                "Complete VH/SMA/QUAD comparison cohort (357)"
                if module_set in MUSCLE_MODULE_SETS and value == "complete_450"
                else COHORT_SCOPE_LABELS[value]
            ),
            help="Validation excludes the audited complete-expression discovery cohort. Each resolved component has its own donor set; missing regions are never imputed.",
            key="analysis_cohort_scope",
        )
        if cohort_scope == VALIDATION_SCOPE:
            validation_manifest = validation_source.manifest()
            st.info(
                f"{validation_source.config.validation_donors} non-discovery donors within ROSMAP; {validation_manifest['reference_policy']}. This is not external-cohort validation. Component cohorts overlap; no pooled three-tissue scores are available."
            )
            if not validation_source.config.motor:
                st.caption(
                    "AC–PCG has only 14 donors (3 AD / 6 Control): group MDC is descriptive only."
                )
            if active_view in {"Prediction", "Edge volcano", "Donor edge explorer"}:
                st.info(
                    "No prediction, new differential-edge discovery, or public donor-expression catalog was generated for this validation cohort."
                )
                st.stop()
        analysis_panel = analysis_slot.container(border=True)
        analysis_cols = analysis_panel.columns(3)
        discovery_screen = active_view in {"Module finder", "CT–TS screen"}
        outcome_column = analysis_cols[0] if discovery_screen else analysis_cols[1]
        feature_column = analysis_cols[1] if discovery_screen else analysis_cols[2]
        default_module = modules.index(935) if 935 in modules else 0
        requested_module = st.query_params.get("screen_module", "")
        if (
            requested_partition == module_set
            and requested_module
            and (
                not str(requested_module).isdigit()
                or int(requested_module) not in modules
            )
        ):
            st.error(
                "The linked module is not in this partition. Select a valid module after clearing the link."
            )
            if st.button("Clear unavailable module link"):
                st.query_params.pop("screen_module", None)
                st.rerun()
            st.stop()
        if (
            st.query_params.get("screen_partition") == module_set
            and str(st.query_params.get("screen_module", "")).isdigit()
            and int(st.query_params["screen_module"]) in modules
        ):
            default_module = modules.index(int(st.query_params["screen_module"]))
        module = _choice(
            analysis_cols[0].selectbox,
            active_view not in {"Module finder", "CT–TS screen", "Methods & data"},
            "Module",
            options=modules,
            index=default_module,
            format_func=module_label,
            key="analysis_module",
        )
        if (
            cohort_scope == VALIDATION_SCOPE
            and module
            in validation_source.manifest().get(
                "cross_filter_node_convention_modules", []
            )
        ):
            st.warning(
                "Legacy node-convention difference: this module's unfiltered scores use unique gene–tissue nodes, while effect-filtered scores retain repeated assignment rows. Each reproduces its original analysis. Treat cross-filter differences cautiously; effective node counts are included in score/edge-summary downloads."
            )
        phenotype = _choice(
            outcome_column.selectbox,
            active_view
            in {
                "Associations",
                "Feature distributions",
                "Correlation heatmaps",
                "Module finder",
                "CT–TS screen",
                "Statistics",
                "Donor edge explorer",
            },
            "Association outcome",
            options=(
                list(ASSOCIATION_OUTCOME_LABELS)
                if active_view == "Associations"
                else [*OUTCOME_LABELS, "clusters", "study"]
            ),
            format_func=lambda value: ASSOCIATION_OUTCOME_LABELS[value],
            key="analysis_phenotype",
        )
        if (
            active_view == "Associations"
            and phenotype in SELECTABLE_ASSOCIATION_OUTCOMES
        ):
            association_interpretation = st.radio(
                "Outcome interpretation",
                options=["numeric", "categorical"],
                format_func=lambda value: (
                    "Numeric / ordinal correlation"
                    if value == "numeric"
                    else "Categorical comparison"
                ),
                horizontal=True,
                help=(
                    "Numeric mode preserves ordered score information. Categorical mode treats "
                    "the codes as labels and uses a Kruskal–Wallis omnibus comparison."
                ),
            )
        elif phenotype in CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES:
            association_interpretation = "categorical"
        else:
            association_interpretation = "numeric"
        if phenotype in PATHOLOGY_OUTCOMES:
            st.caption(
                "Postmortem pathology, August 2025 release. Amyloid and tangles use the "
                "supplied square-root 8-region estimates; no additional outcome transform is applied."
            )
        if phenotype == "study":
            st.caption(
                "ROS/MAP is nominal study membership. Comparisons use Kruskal–Wallis and epsilon-squared, not numeric correlation codes. A cohort containing only one study cannot support a between-study test."
            )
        active_feature_labels = dict(
            FEATURE_LABELS if estimator == "lioness" else BONOBO_FEATURE_LABELS
        )
        descriptive_eigengenes_available = (
            estimator == "lioness"
            and cohort_scope == "complete_450"
            and active_view
            in {
                "Associations",
                "Feature distributions",
                "Correlation heatmaps",
            }
            and descriptive_eigengene_data_available(module_set)
        )
        if descriptive_eigengenes_available:
            active_feature_labels.update(EIGENGENE_FEATURE_LABELS)
        feature = _choice(
            feature_column.selectbox,
            active_view
            in {
                "Associations",
                "Feature distributions",
                "Correlation heatmaps",
                "Module finder",
                "CT–TS screen",
                "Statistics",
                "Donor edge explorer",
            },
            "Module feature",
            options=list(active_feature_labels),
            format_func=lambda value: active_feature_labels[value],
            key="analysis_feature",
        )
        with st.expander("Edge selection", expanded=False):
            if estimator == "bonobo":
                bonobo_edge_subset = _choice(
                    st.radio,
                    True,
                    "BONOBO edge subset",
                    options=["All edges", "Significant edges"],
                    help=(
                        "Significance tests whether a donor-specific BONOBO covariance edge "
                        "differs from zero; it is not a diagnostic-group comparison."
                    ),
                    key="analysis_bonobo_edge_subset",
                )
                if bonobo_edge_subset == "Significant edges":
                    edge_rule = _choice(
                        st.radio,
                        True,
                        "Significant-edge rule",
                        options=list(BONOBO_EDGE_RULE_LABELS),
                        format_func=lambda value: BONOBO_EDGE_RULE_LABELS[value],
                        key="analysis_edge_rule",
                    )
                else:
                    edge_rule = "all"
            else:
                bonobo_edge_subset = "All edges"
                edge_rule = "all"
            differential_available = (
                cohort_scope == "complete_450"
                and feature != "eigengene"
                and differential_data_available(module_set)
            )
            effect_available = (
                feature != "eigengene"
                and (active_view != "MDC" or cohort_scope == VALIDATION_SCOPE)
                and effect_size_data_available(module_set, estimator, method)
            )
            if cohort_scope == VALIDATION_SCOPE:
                effect_available = any(
                    r != "all"
                    for r in validation_source.manifest().get("completed_rules", [])
                )
            subset_options = ["all"]
            if differential_available:
                subset_options.append("ad_control_discovery_fdr05")
            if effect_available:
                subset_options.append("effect_size")
            edge_subset_basis = _choice(
                st.radio,
                True,
                "AD–Control edge subset",
                options=subset_options,
                format_func=lambda value: {
                    "all": "All edges",
                    "ad_control_discovery_fdr05": "AD–Control FDR-filtered edges",
                    "effect_size": "AD–Control effect-size-filtered edges",
                }[value],
                help=(
                    "AD–Control masks are learned only in the diagnosis-and-sex-stratified "
                    "discovery donors. All-edge and existing FDR analyses remain unchanged."
                ),
                key="analysis_edge_subset_basis",
            )
            differential_fdr_scope = "global"
            differential_fdr_threshold = 0.05
            effect_statistic = "hedges_g"
            effect_cutoff_mode = "fixed"
            effect_cutoff_value = 0.40
            effect_direction = "either"
            if edge_subset_basis == "all":
                differential_edge_rule = "all"
            elif edge_subset_basis == "ad_control_discovery_fdr05":
                differential_edge_rule = "ad_control_discovery_fdr05"
                differential_fdr_scope = _choice(
                    st.radio,
                    True,
                    "Differential-edge FDR scope",
                    options=list(FDR_SCOPE_LABELS),
                    format_func=lambda value: FDR_SCOPE_LABELS[value],
                    index=0,
                    help=(
                        "Global BH adjusts all tested edges within the selected module definition, "
                        "estimator, network method, and discovery/validation family. Per-module BH "
                        "adjusts only across the edges of each module."
                    ),
                    key="analysis_differential_fdr_scope",
                )
                differential_fdr_threshold = _choice(
                    st.radio,
                    True,
                    "Differential-edge FDR cutoff",
                    options=list(FDR_THRESHOLD_LABELS),
                    format_func=lambda value: FDR_THRESHOLD_LABELS[value],
                    index=0,
                    help=(
                        "FDR < 0.10 is an exploratory, less restrictive mask. The selected "
                        "cutoff is applied after the chosen global or per-module BH correction."
                    ),
                    key="analysis_differential_fdr_threshold",
                )
            else:
                effect_manifest = cached_effect_size_manifest(module_set)
                validation_effect_options = (
                    validation_source.completed_effect_options()
                    if cohort_scope == VALIDATION_SCOPE
                    else []
                )
                if validation_effect_options:
                    effect_manifest = {
                        "available_effect_statistics": list(
                            dict.fromkeys(
                                e["statistic"] for e in validation_effect_options
                            )
                        ),
                        "available_cutoff_modes": {
                            s: list(
                                dict.fromkeys(
                                    e["mode"]
                                    for e in validation_effect_options
                                    if e["statistic"] == s
                                )
                            )
                            for s in {e["statistic"] for e in validation_effect_options}
                        },
                    }
                available_effect_statistics = effect_manifest.get(
                    "available_effect_statistics", list(EFFECT_STATISTIC_LABELS)
                )
                available_effect_statistics = [
                    value
                    for value in available_effect_statistics
                    if value in EFFECT_STATISTIC_LABELS
                ] or list(EFFECT_STATISTIC_LABELS)
                effect_statistic = _choice(
                    st.radio,
                    True,
                    "Effect statistic",
                    options=available_effect_statistics,
                    format_func=lambda value: EFFECT_STATISTIC_LABELS[value],
                    horizontal=True,
                    key="analysis_effect_statistic",
                )
                available_cutoff_modes = effect_manifest.get(
                    "available_cutoff_modes", {}
                )
                cutoff_modes = available_cutoff_modes.get(effect_statistic)
                if not cutoff_modes:
                    cutoff_modes = (
                        ["fixed", "module_top_fraction"]
                        if effect_statistic == "hedges_g"
                        else ["module_top_fraction"]
                    )
                effect_cutoff_mode = _choice(
                    st.radio,
                    True,
                    "Effect threshold type",
                    options=cutoff_modes,
                    format_func=lambda value: {
                        "fixed": "Fixed absolute effect",
                        "module_top_fraction": "Top fraction within module",
                    }[value],
                    horizontal=True,
                    key="analysis_effect_cutoff_mode",
                )
                if effect_cutoff_mode == "fixed":
                    fixed_options = effect_manifest.get(
                        "available_fixed_hedges_g", list(HEDGES_FIXED_THRESHOLDS)
                    )
                    fixed_options = [float(value) for value in fixed_options]
                    if validation_effect_options:
                        fixed_options = sorted(
                            {
                                e["value"]
                                for e in validation_effect_options
                                if e["statistic"] == effect_statistic
                                and e["mode"] == "fixed"
                            }
                        )
                    default_fixed = 0.40 if 0.40 in fixed_options else fixed_options[0]
                    effect_cutoff_value = _choice(
                        st.select_slider,
                        True,
                        "Absolute Hedges’ g cutoff",
                        options=fixed_options,
                        value=default_fixed,
                        format_func=lambda value: f"|g| ≥ {value:.2f}",
                        key="analysis_effect_cutoff_value",
                    )
                else:
                    fraction_options = effect_manifest.get(
                        "available_module_top_fractions", list(MODULE_TOP_FRACTIONS)
                    )
                    fraction_options = [float(value) for value in fraction_options]
                    if validation_effect_options:
                        fraction_options = sorted(
                            {
                                e["value"]
                                for e in validation_effect_options
                                if e["statistic"] == effect_statistic
                                and e["mode"] == "module_top_fraction"
                            }
                        )
                    effect_cutoff_value = _choice(
                        st.radio,
                        True,
                        "Retained module edges",
                        options=fraction_options,
                        format_func=lambda value: f"Top {int(round(100 * value))}%",
                        index=min(1, len(fraction_options) - 1),
                        horizontal=True,
                        key="analysis_effect_cutoff_value",
                    )
                available_directions = effect_manifest.get(
                    "available_directions", list(EFFECT_DIRECTION_LABELS)
                )
                available_directions = [
                    value
                    for value in available_directions
                    if value in EFFECT_DIRECTION_LABELS
                ] or list(EFFECT_DIRECTION_LABELS)
                if validation_effect_options:
                    available_directions = [
                        d
                        for d in available_directions
                        if any(
                            e["statistic"] == effect_statistic
                            and e["mode"] == effect_cutoff_mode
                            and e["value"] == effect_cutoff_value
                            and e["direction"] == d
                            for e in validation_effect_options
                        )
                    ]
                effect_direction = _choice(
                    st.radio,
                    True,
                    "Effect direction",
                    options=available_directions,
                    format_func=lambda value: EFFECT_DIRECTION_LABELS[value],
                    horizontal=True,
                    key="analysis_effect_direction",
                )
                differential_edge_rule = effect_rule_key(
                    effect_statistic,
                    effect_cutoff_mode,
                    float(effect_cutoff_value),
                    effect_direction,
                )
                # Effect-only catalogs are guaranteed to retain the exact module-level
                # Welch/BH statistics. Global BH remains available only where the full
                # differential-edge catalog was explicitly generated.
                if not differential_available:
                    differential_fdr_scope = "per_module"
            if not differential_available and not effect_available:
                if feature == "eigengene":
                    st.caption(
                        "Edge-subset controls do not apply to expression eigengenes."
                    )
                else:
                    st.caption(
                        "AD–Control filtered scores are not present in this deployed bundle."
                    )
            if differential_edge_rule == "all":
                score_normalization = "standard_pruned"
                analysis_subset = "all_donors"
                if cohort_scope == VALIDATION_SCOPE and active_view != "MDC":
                    score_normalization = _choice(
                        st.radio,
                        True,
                        "Score handling",
                        list(SCORE_NORMALIZATION_LABELS),
                        format_func=SCORE_NORMALIZATION_LABELS.get,
                        key="analysis_score_normalization",
                    )
            else:
                available_score_normalizations = (
                    cached_effect_size_manifest(module_set).get(
                        "available_score_normalizations",
                        list(SCORE_NORMALIZATION_LABELS),
                    )
                    if differential_edge_rule.startswith(
                        "ad_control_discovery_effect__"
                    )
                    else list(SCORE_NORMALIZATION_LABELS)
                )
                available_score_normalizations = [
                    value
                    for value in available_score_normalizations
                    if value in SCORE_NORMALIZATION_LABELS
                ] or list(SCORE_NORMALIZATION_LABELS)
                score_normalization = _choice(
                    st.radio,
                    True,
                    "Score handling",
                    options=available_score_normalizations,
                    format_func=lambda value: SCORE_NORMALIZATION_LABELS[value],
                    help=(
                        "Standard pruning preserves the module-gene denominator. Retained-edge "
                        "normalization divides edge sums by the number of retained edges."
                    ),
                    key="analysis_score_normalization",
                )
                if cohort_scope == VALIDATION_SCOPE:
                    analysis_subset = "all_donors"
                    st.caption(
                        "Evaluation uses only non-discovery validation donors. Original Discovery/Validation split controls do not apply."
                    )
                else:
                    analysis_subset = _choice(
                        st.selectbox,
                        True,
                        "Evaluation cohort",
                        options=list(ANALYSIS_SUBSET_LABELS),
                        format_func=lambda value: ANALYSIS_SUBSET_LABELS[value],
                        index=list(ANALYSIS_SUBSET_LABELS).index(
                            "validation_ad_control"
                        ),
                        key="analysis_analysis_subset",
                    )
                if score_normalization == "retained_edge":
                    active_feature_labels.update(
                        {
                            "connectivity": "Mean retained signed edge weight",
                            "abs_sum": "Mean retained absolute edge weight",
                            "positive_abs_sum": "Positive contribution per retained edge",
                            "negative_abs_sum": "Negative contribution per retained edge",
                        }
                    )
            if cohort_scope == VALIDATION_SCOPE and not validation_source.available(
                differential_edge_rule
            ):
                st.info(
                    "This validation mask is not ready yet; original-cohort results will not be substituted."
                )
                st.stop()
            if cohort_scope == "maximum_component" and (
                feature == "eigengene" or differential_edge_rule != "all"
            ):
                st.info(
                    "Maximum-component data are available only for all-edge network features."
                )
                st.stop()
        resolution_options = (
            ["Tissue resolved"]
            if cohort_scope != "complete_450" or feature == "eigengene"
            else ["Aggregate CT / TS", "Tissue resolved"]
        )
        resolution = _choice(
            analysis_cols[0].radio,
            active_view
            in {
                "Associations",
                "Feature distributions",
                "Correlation heatmaps",
                "Statistics",
                "Donor edge explorer",
            }
            or cohort_scope != "complete_450",
            "Resolution",
            resolution_options,
            key="analysis_resolution",
        )
        if feature == "eigengene":
            st.caption(
                "Module eigengenes are tissue-specific PCA1 expression summaries for "
                + ", ".join(active_tissues)
                + ". They are independent of the selected LIONESS method and "
                "edge filter; a cross-tissue pair eigengene is not defined."
            )
        if cohort_scope == "maximum_component" and active_view in {
            "CT–TS screen",
            "Module finder",
        }:
            st.info(
                "This screen compares pooled CT with pooled TS and is therefore available "
                "only in the common 450-donor cohort. The maximum-component sensitivity "
                "contains resolved tissues and tissue pairs with different donor sets."
            )
            st.stop()
        scale = _choice(
            analysis_cols[1].selectbox,
            active_view
            in {
                "Associations",
                "Feature distributions",
                "Correlation heatmaps",
                "Statistics",
                "Donor edge explorer",
            }
            or cohort_scope == VALIDATION_SCOPE,
            "Feature scale",
            options=list(SCALE_LABELS),
            format_func=lambda value: SCALE_LABELS[value],
            index=(
                list(SCALE_LABELS).index("raw")
                if cohort_scope == VALIDATION_SCOPE
                else 0
            ),
            key="analysis_scale",
        )
        correlation_method = _choice(
            analysis_cols[2].radio,
            active_view
            in {
                "Associations",
                "Correlation heatmaps",
                "Module finder",
                "CT–TS screen",
                "Statistics",
                "Donor edge explorer",
            },
            "Association correlation",
            options=["Spearman", "Pearson"],
            index=0,
            help=(
                "Controls association scatter annotations, correlation heatmaps, the CT–TS "
                "screen, and the primary statistics table. Spearman is the default."
            ),
            disabled=association_interpretation == "categorical",
            key="analysis_correlation_method",
        )
        diagnosis_options = [*DIAGNOSIS_ORDER]
        if cohort_scope != "complete_450":
            diagnosis_options.append("Unclassified")
        diagnoses = _choice(
            st.multiselect,
            active_view
            not in {
                "MDC",
                "KEGG enrichment",
                "Module details",
                "Methods & data",
                "Edge volcano",
            },
            "Diagnosis groups",
            options=diagnosis_options,
            default=diagnosis_options,
            key="analysis_diagnoses",
        )
        with (
            st.expander("Groups and statistics", expanded=False)
            if active_view
            in {"Associations", "Feature distributions", "Correlation heatmaps"}
            else st.container()
        ):
            grouping_variable = "diagnosis_group"
            selected_group_levels: list[object] = list(diagnoses)
            distribution_grouping_variable = "diagnosis_group"
            selected_distribution_levels: list[object] = list(diagnoses)
            distribution_minimum_group_n = 10
            heatmap_grouping_variable = "diagnosis_group"
            selected_heatmap_group_levels: list[object] = list(diagnoses)
            heatmap_minimum_group_n = 10
            minimum_group_n = 10
            annotation_fields: list[str] = ["n", "coefficient", "p", "fdr"]
            trend_line_rule = "all"
            show_group_trend_lines = True
            association_significance_cutoff = 0.05
            show_pooled_association = False
            pooled_line_dash = "dash"
            single_point_color = "#2C7FB8"
            match_effect_kegg_subtitle = False
            exact_endpoint_kegg_context = (
                estimator == "lioness"
                and edge_subset_basis == "effect_size"
                and effect_statistic == "hedges_g"
                and effect_cutoff_mode == "fixed"
                and np.isclose(float(effect_cutoff_value), 0.40)
                and effect_endpoint_kegg_available(
                    module_set, method, manifest=data_manifest
                )
            )
            if active_view == "Associations" and exact_endpoint_kegg_context:
                match_effect_kegg_subtitle = st.checkbox(
                    "Match KEGG subtitle to current edge filter",
                    value=True,
                    help=(
                        "Use enrichment calculated only from tissue-gene endpoints of the "
                        "currently selected Hedges’ g ≥ 0.40 edge set. Turn this off to retain "
                        "the original full-module enrichment subtitle."
                    ),
                )
            if active_view == "Associations":
                association_plot_type = _choice(
                    st.selectbox,
                    True,
                    "Plot type",
                    ["Scatter", "Score comparison"],
                    index=0,
                    key="association_plot_type",
                    help="Compare same-module eigengenes and coordination on common complete-case donors.",
                )
                association_metadata = cached_sample_metadata(module_set, cohort_scope)
                association_metadata = association_metadata.loc[
                    association_metadata["diagnosis_group"].isin(diagnoses)
                ].copy()
                if differential_edge_rule != "all" and analysis_subset != "all_donors":
                    selected_split = {
                        "discovery_ad_control": "Discovery",
                        "validation_ad_control": "Validation",
                        "mci_external": "MCI_external",
                    }[analysis_subset]
                    association_metadata = association_metadata.loc[
                        association_metadata["ad_control_split"].eq(selected_split)
                    ]
                minimum_group_n = _choice(
                    st.selectbox,
                    True,
                    "Minimum category size",
                    options=[5, 10, 20],
                    index=1,
                    help=(
                        "Smaller groups remain visible, but their statistics and trend lines are "
                        "marked unavailable."
                    ),
                    key="analysis_minimum_group_n",
                )
                if association_interpretation == "numeric":
                    grouping_options = [
                        variable
                        for variable in ASSOCIATION_GROUP_LABELS
                        if variable != phenotype
                    ]
                    grouping_variable = _choice(
                        st.selectbox,
                        True,
                        "Group correlations by",
                        options=grouping_options,
                        format_func=lambda value: ASSOCIATION_GROUP_LABELS[value],
                        index=grouping_options.index("diagnosis_group"),
                        key="analysis_grouping_variable",
                    )
                    available_group_levels = ordered_association_levels(
                        association_metadata, grouping_variable
                    )
                    selected_group_levels = _choice(
                        st.multiselect,
                        True,
                        "Group levels",
                        options=available_group_levels,
                        default=available_group_levels,
                        format_func=lambda value: association_level_label(
                            grouping_variable, value
                        ),
                        disabled=grouping_variable == "__all__",
                        help="The selection controls points, trends, annotations, legends, and downloads.",
                        key="analysis_selected_group_levels",
                    )
                    if grouping_variable == "__all__":
                        selected_group_levels = ["__all__"]
                    show_pooled_association = (
                        st.checkbox(
                            "Show pooled association across displayed donors",
                            value=True,
                            help=(
                                "The dashed pooled fit is recalculated after diagnosis, cohort, "
                                "component, grouping-level, and missing-value filters."
                            ),
                        )
                        if grouping_variable != "__all__"
                        else False
                    )
                    pooled_line_dash = st.radio(
                        "Pooled correlation line style",
                        options=["dash", "solid"],
                        index=0,
                        horizontal=True,
                        format_func=lambda value: {
                            "dash": "Dashed black",
                            "solid": "Solid black",
                        }[value],
                        disabled=not show_pooled_association,
                        help="Changes only the pooled all-donor correlation line.",
                    )
                    annotation_fields = st.multiselect(
                        "Plot annotation fields",
                        options=["n", "coefficient", "p", "fdr"],
                        default=["n", "coefficient", "p", "fdr"],
                        format_func=lambda value: {
                            "n": "Sample size (n)",
                            "coefficient": "Correlation coefficient",
                            "p": "Nominal p-value",
                            "fdr": "Module-set FDR",
                        }[value],
                    )
                    trend_line_rule = st.selectbox(
                        "Trend-line rule",
                        options=["all", "p", "fdr", "none"],
                        format_func=lambda value: {
                            "all": "All eligible groups",
                            "p": "Nominal p below cutoff",
                            "fdr": "Module-set FDR below cutoff",
                            "none": "No trend lines",
                        }[value],
                    )
                    show_group_trend_lines = st.checkbox(
                        "Show group-specific OLS trend lines",
                        value=True,
                        help=(
                            "Turn this off to retain group points, legends, and statistics while "
                            "hiding only their OLS guides. The pooled dashed trend is controlled "
                            "separately above."
                        ),
                        disabled=trend_line_rule == "none",
                    )
                    if not show_group_trend_lines and show_pooled_association:
                        st.caption(
                            "Group-specific lines are hidden; only the "
                            f"{'solid' if pooled_line_dash == 'solid' else 'dashed'} black "
                            "pooled all-donor trend remains."
                        )
                    association_significance_cutoff = st.radio(
                        "Association significance cutoff",
                        options=[0.05, 0.10],
                        index=0,
                        horizontal=True,
                        format_func=lambda value: f"{value:.2f}"
                        + (" (exploratory)" if value == 0.10 else ""),
                        disabled=trend_line_rule not in {"p", "fdr"},
                    )
                else:
                    available_group_levels = ordered_association_levels(
                        association_metadata, phenotype
                    )
                    selected_group_levels = _choice(
                        st.multiselect,
                        True,
                        "Category levels",
                        options=available_group_levels,
                        default=available_group_levels,
                        format_func=lambda value: association_level_label(
                            phenotype, value
                        ),
                        help="The selection controls displayed points and the omnibus test.",
                        key="analysis_selected_group_levels",
                    )
                    annotation_fields = st.multiselect(
                        "Plot annotation fields",
                        options=["n", "h", "effect", "p", "fdr"],
                        default=["n", "h", "effect", "p", "fdr"],
                        format_func=lambda value: {
                            "n": "Sample size (n)",
                            "h": "Kruskal–Wallis H",
                            "effect": "Epsilon-squared",
                            "p": "Nominal p-value",
                            "fdr": "Module-set FDR",
                        }[value],
                    )
            if active_view == "Feature distributions":
                distribution_metadata = cached_sample_metadata(module_set, cohort_scope)
                distribution_metadata = distribution_metadata.loc[
                    distribution_metadata["diagnosis_group"].isin(diagnoses)
                ].copy()
                if differential_edge_rule != "all" and analysis_subset != "all_donors":
                    selected_split = {
                        "discovery_ad_control": "Discovery",
                        "validation_ad_control": "Validation",
                        "mci_external": "MCI_external",
                    }[analysis_subset]
                    distribution_metadata = distribution_metadata.loc[
                        distribution_metadata["ad_control_split"].eq(selected_split)
                    ]
                distribution_grouping_variable = st.selectbox(
                    "Group distributions by",
                    options=list(ASSOCIATION_GROUP_LABELS),
                    format_func=lambda value: ASSOCIATION_GROUP_LABELS[value],
                    index=list(ASSOCIATION_GROUP_LABELS).index("diagnosis_group"),
                    help=(
                        "Diagnosis remains the donor-inclusion filter. This selection controls "
                        "how the retained donors are divided into distribution traces and summary rows."
                    ),
                )
                available_distribution_levels = ordered_association_levels(
                    distribution_metadata, distribution_grouping_variable
                )
                selected_distribution_levels = st.multiselect(
                    "Distribution group levels",
                    options=available_distribution_levels,
                    default=available_distribution_levels,
                    format_func=lambda value: association_level_label(
                        distribution_grouping_variable, value
                    ),
                    disabled=distribution_grouping_variable == "__all__",
                    help="Selected levels control traces, legend entries, summaries, and downloads.",
                )
                if distribution_grouping_variable == "__all__":
                    selected_distribution_levels = ["__all__"]
                distribution_minimum_group_n = st.selectbox(
                    "Distribution minimum group size",
                    options=[5, 10, 20],
                    index=1,
                    disabled=distribution_grouping_variable == "__all__",
                    help=(
                        "Smaller groups remain visible in distribution plots but are excluded "
                        "from omnibus and pairwise inference."
                    ),
                )
            if active_view == "Correlation heatmaps":
                heatmap_metadata = cached_sample_metadata(module_set, cohort_scope)
                heatmap_metadata = heatmap_metadata.loc[
                    heatmap_metadata["diagnosis_group"].isin(diagnoses)
                ].copy()
                if differential_edge_rule != "all" and analysis_subset != "all_donors":
                    selected_split = {
                        "discovery_ad_control": "Discovery",
                        "validation_ad_control": "Validation",
                        "mci_external": "MCI_external",
                    }[analysis_subset]
                    heatmap_metadata = heatmap_metadata.loc[
                        heatmap_metadata["ad_control_split"].eq(selected_split)
                    ]
                heatmap_grouping_variable = st.selectbox(
                    "Group heatmap correlations by",
                    options=list(ASSOCIATION_GROUP_LABELS),
                    format_func=lambda value: ASSOCIATION_GROUP_LABELS[value],
                    index=list(ASSOCIATION_GROUP_LABELS).index("diagnosis_group"),
                    help=(
                        "Diagnosis remains the donor-inclusion filter. Non-diagnosis groups "
                        "are calculated lazily, one module at a time."
                    ),
                )
                if heatmap_grouping_variable == "diagnosis_group":
                    st.caption(
                        "Choose the diagnosis/cohort display in the heatmap controls."
                    )
                else:
                    available_heatmap_group_levels = ordered_association_levels(
                        heatmap_metadata, heatmap_grouping_variable
                    )
                    selected_heatmap_group_levels = st.multiselect(
                        "Heatmap group levels",
                        options=available_heatmap_group_levels,
                        default=available_heatmap_group_levels,
                        format_func=lambda value: association_level_label(
                            heatmap_grouping_variable, value
                        ),
                        disabled=heatmap_grouping_variable == "__all__",
                        help=(
                            "Only these category levels enter the correlations, heatmap rows, "
                            "FDR families, table, and download."
                        ),
                    )
                    if heatmap_grouping_variable == "__all__":
                        selected_heatmap_group_levels = ["__all__"]
                    heatmap_minimum_group_n = st.selectbox(
                        "Heatmap minimum group size",
                        options=[5, 10, 20],
                        index=1,
                        help=(
                            "Smaller groups remain represented by unavailable statistics and "
                            "do not enter Benjamini–Hochberg correction."
                        ),
                    )
        if active_view in {"Associations", "Donor edge explorer"}:
            with st.expander("Appearance", expanded=False):
                point_color_labels = {
                    "__single_color__": "Single color",
                    **COLOR_LABELS,
                }
                point_color_options = (
                    list(point_color_labels)
                    if active_view == "Associations"
                    else list(COLOR_LABELS)
                )
                color_by = st.selectbox(
                    "Color points by",
                    options=point_color_options,
                    index=point_color_options.index("diagnosis_group"),
                    format_func=lambda value: point_color_labels[value],
                )
                if active_view == "Associations" and color_by == "__single_color__":
                    single_point_color = st.color_picker(
                        "Point color",
                        value="#2C7FB8",
                        help="Applies the selected color to every donor point.",
                    )
                continuous_colorscale = st.selectbox(
                    "Continuous color scale",
                    options=list(CONTINUOUS_COLOR_SCALES),
                    index=0,
                    disabled=(
                        color_by == "__single_color__"
                        or color_by in CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES
                        or (
                            active_view == "Associations"
                            and association_interpretation == "numeric"
                            and color_by == grouping_variable
                        )
                    ),
                )
                reverse_colorscale = st.checkbox(
                    "Reverse continuous color scale",
                    value=False,
                    disabled=(
                        color_by == "__single_color__"
                        or color_by in CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES
                        or (
                            active_view == "Associations"
                            and association_interpretation == "numeric"
                            and color_by == grouping_variable
                        )
                    ),
                )
        else:
            point_color_labels = dict(COLOR_LABELS)
            color_by = "diagnosis_group"
            continuous_colorscale = next(iter(CONTINUOUS_COLOR_SCALES))
            reverse_colorscale = False

    resolved = resolution != "Aggregate CT / TS"
    component_options = active_component_order if resolved else ["CT", "TS"]
    if feature == "eigengene":
        component_options = [c for c in component_options if c.startswith("TS_")]
    component_labels = {c: EDGE_SCOPE_LABELS.get(c, c) for c in component_options}
    available_components = pd.DataFrame(
        {
            "component": component_options,
            "component_label": [component_labels[c] for c in component_options],
        }
    )
    selected_components = (
        analysis_panel.multiselect(
            "Tissue components",
            component_options,
            default=component_options,
            format_func=component_labels.get,
        )
        if resolved
        else component_options
    )
    return locals()
