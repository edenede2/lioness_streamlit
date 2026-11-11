"""Historical plot contracts for calibrated scores; no score fitting or writes.

Measures are separate figures, components are panels, and color encodes donor
groups/phenotypes rather than mixing gain and loss into the same scatter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app_helpers import data as D
from app_helpers.charts import (
    CONTINUOUS_COLOR_SCALES,
    categorical_association_figure,
    distribution_figure,
    grouped_association_figure,
)
from app_helpers.correlations import calculate_correlations, calculate_categorical_associations
from .association_fdr import add_module_fdr, from_baseline
from .registry import feature_labels


def ordered_levels(frame, variable):
    if variable == "__all__":
        return ["__all__"]
    observed = list(frame[variable].dropna().unique())
    preferred = D.ASSOCIATION_LEVEL_ORDERS.get(variable, [])
    return [v for v in preferred if v in observed] + sorted(
        [v for v in observed if v not in preferred], key=str
    )


def panel_frame(frame, components):
    """Honor component selection order without altering any numeric value."""
    return frame.assign(_panel_order=frame.component.map({c: i for i, c in enumerate(components)})).sort_values(
        "_panel_order", kind="stable"
    ).drop(columns="_panel_order")


def definition_label(context):
    return f"{D.MODULE_SET_LABELS.get(context['family'], context['family'])} · beta{context['beta']}"


@st.cache_data(show_spinner=False, max_entries=8, ttl=600)
def plot_statistics(frame, outcome, grouping, levels, minimum_n, baseline=None):
    """Historical Pearson/Spearman tests; no BH on displayed groups alone.

    Exact matching native pooled tests may reuse the archived permutation test.
    Their module-only q is calculated from the entire matching baseline family.
    """
    if baseline is not None and baseline.empty:
        baseline = None
    pieces = []
    for level in levels:
        part = frame if grouping == "__all__" or level == "__pooled__" else frame.loc[frame[grouping].eq(level)]
        if part.empty:
            continue
        part = part.copy()
        part["grouping_level"] = level
        pieces.append(part)
    if not pieces:
        return pd.DataFrame()
    stacked = pd.concat(pieces, ignore_index=True)
    result = calculate_correlations(
        stacked, ["component", "grouping_level"], [outcome], min_group_n=minimum_n
    )
    # The helper's displayed-family q is NOT the calibrated catalog correction.
    result = result.drop(columns=[c for c in result if "fdr_displayed_family" in c])
    result["grouping_variable"] = grouping
    result["q_scan"] = np.nan
    result["q_GL_outcome"] = np.nan
    result["spearman_p_method"] = "historical SciPy Spearman (asymptotic)"
    result["pearson_p_method"] = "historical SciPy Pearson (two-sided)"
    for method in ("pearson", "spearman"):
        result[method + "_fdr_across_modules"] = np.nan
        result[method + "_fdr_module_family_n"] = 0
    result["fdr_scope"] = "unavailable until the complete module family is calculated"
    for index, row in result.iterrows():
        if not row.eligible:
            continue
        saved = pd.DataFrame() if baseline is None else baseline.loc[baseline.component.eq(row.component)]
        if row.grouping_level in {"__pooled__", "__all__"} and len(saved) == 1:
            source = saved.iloc[0]
            if int(source.n) == int(row.n) and np.isclose(source.effect, row.spearman_rho, atol=1e-12):
                result.loc[index, "spearman_p"] = source.p
                result.loc[index, "q_scan"] = source.get("q_scan", np.nan)
                result.loc[index, "q_GL_outcome"] = source.get("q_GL_outcome", np.nan)
                result.loc[index, "spearman_fdr_across_modules"] = source.get("q_across_modules", np.nan)
                result.loc[index, "spearman_fdr_module_family_n"] = source.get("module_family_n", 0)
                result.loc[index, "fdr_scope"] = "BH across all valid modules for this fixed component, measure, outcome and pooled cohort"
                result.loc[index, "spearman_p_method"] = str(source.get("p_method", "saved baseline inference"))
    return result


@st.cache_data(show_spinner=False, max_entries=4, ttl=600)
def module_statistics(family, estimator, scope, mask, tissue, transform, components,
                      metrics, outcome, grouping, levels, minimum_n, donors,
                      categorical=False, revision=None, use_baseline=False,
                      inference_version="auto", ols_covariates=None):
    """Serialized, bounded module reads; cache compact statistics, never scores.

    Missing artifacts abort the family (no silently partial correction). Empty or
    constant biological components remain explicit unavailable tests.
    """
    from . import adapter as A
    parts = []
    jobs = [j for j in A.catalog()["modules"] if j["family"] == family
            and (tissue is None or j["regional_tissue"] == tissue)]
    if not jobs:
        raise A.ScoreUnavailable("No modules in this partition/tissue")
    baseline = None
    if use_baseline:
        baseline = A.baseline_associations(module_set=family, estimator=estimator,
            cohort_scope=scope, phenotype=outcome, regional_tissue=tissue,
            inference_version=inference_version)
        baseline.loc[baseline.n.lt(minimum_n), "p"] = np.nan
        baseline = from_baseline(baseline)
    for job in jobs:
        frame = A.scores(module_set=family, estimator=estimator, module=job["module"],
            cohort_scope=scope, regional_tissue=tissue, metric_families=metrics,
            differential_edge_rule=mask[0], differential_fdr_scope=mask[1],
            differential_fdr_threshold=mask[2])
        frame = frame.loc[frame.component.isin(components) & frame.sample_id.isin(donors)].copy()
        frame["metric_value"] = frame["metric_" + transform]
        for metric in metrics:
            part = frame.loc[frame.metric_family.eq(metric)]
            if categorical:
                part = part.loc[part[outcome].isin(levels)]
                result = calculate_categorical_associations(part, ["component"], category_column=outcome, min_group_n=minimum_n)
                if result.empty:
                    continue
                result["method"] = "Kruskal-Wallis"
                result["p"] = result.categorical_p
            elif ols_covariates is not None:
                from .ui import _ols
                rows = []
                for level in levels:
                    selected = part if grouping == "__all__" or level == "__pooled__" else part.loc[part[grouping].eq(level)]
                    for component, group in selected.groupby("component", sort=False):
                        fitted = _ols(group.metric_value.to_numpy(float),
                            pd.to_numeric(group[outcome], errors="coerce").to_numpy(float), group, list(ols_covariates))
                        if fitted["n"] < minimum_n:
                            fitted.update(p=np.nan, unavailable_reason=f"n < {minimum_n}")
                        rows.append(dict(component=component, grouping_variable=grouping,
                            grouping_level=level, outcome=outcome, method="Descriptive OLS", **fitted))
                result = pd.DataFrame(rows)
                if result.empty:
                    continue
            else:
                saved = None if baseline is None else baseline.loc[
                    baseline.module.eq(job["module"]) & baseline.metric_family.eq(metric)
                    & baseline.method.eq("Spearman")]
                result = plot_statistics(part, outcome, grouping, levels, minimum_n, saved)
                if result.empty:
                    continue
                result = pd.concat([result.assign(method=method, p=result[prefix + "_p"])
                    for method, prefix in (("Pearson", "pearson"), ("Spearman", "spearman"))], ignore_index=True)
            result["module_key"], result["module"] = job["module_key"], job["module"]
            result["metric_family"], result["family"] = metric, family
            result["beta"] = int(estimator.removeprefix("calibrated_beta"))
            result["cohort_scope"], result["transform"] = scope, transform
            result["mask"] = mask[0]
            result["differential_fdr_scope"], result["differential_fdr_threshold"] = mask[1:]
            result["minimum_group_n"] = minimum_n
            result["score_normalization"] = "common_eligible_physical_edges"
            parts.append(result)
    if not parts:
        raise A.ScoreUnavailable("No eligible module tests for these selections")
    return add_module_fdr(pd.concat(parts, ignore_index=True))


def enrichment_subtitles(frame, context):
    """Reuse the historical tissue-expanded/regional KEGG subtitle chooser."""
    try:
        if context.get("regional_tissue"):
            from app_helpers.regional_modules import load_regional_kegg
            kegg = load_regional_kegg()
            kegg = kegg.loc[kegg.partition_source.eq(context["family"]) & kegg.tissue.eq(context["regional_tissue"]) & kegg.module.eq(context["module"])]
        else:
            kegg = D.load_kegg(module=context["module"], module_set=context["family"])
        aggregate = D.selected_kegg_table_annotation(kegg)
        output = {}
        for resolved in (False, True):
            components = [c for c in frame.component.unique() if c.startswith(("TS_", "CT_")) == resolved]
            names = {c: c.replace("PCG", "PCGBA23") if "PCGBA23" not in c else c for c in components}
            subtitles = D.association_kegg_subtitles(kegg, names.values(), resolved=resolved, aggregate_annotation=aggregate)
            output.update({c: subtitles.get(names[c], "KEGG enrichment: unavailable") for c in components})
        return output
    except (OSError, ValueError, KeyError):
        return {c: "KEGG enrichment: unavailable" for c in frame.component.unique()}


def render_associations(frame, context, baseline, render, draw, table):
    st.session_state["cc_current_plot_family"] = None
    labels = feature_labels()
    outcome = context["outcome"]
    categorical = context["outcome_interpretation"] == "categorical"
    metadata = frame.drop_duplicates("sample_id")
    with st.sidebar.expander("Groups and statistics", expanded=False):
        minimum = st.selectbox("Minimum category size", [5, 10, 20], index=1, key="cc_plot_minimum")
        grouping = outcome if categorical else st.selectbox(
            "Group correlations by", [g for g in D.ASSOCIATION_GROUP_LABELS if g != outcome and (g == "__all__" or g in frame)],
            format_func=D.ASSOCIATION_GROUP_LABELS.get, key="cc_plot_grouping",
        )
        available = ordered_levels(metadata, grouping)
        key = "cc_plot_levels_" + grouping
        if key in st.session_state and not set(st.session_state[key]) <= set(available):
            del st.session_state[key]
        levels = st.multiselect("Category levels" if categorical else "Group levels", available,
            default=available, format_func=lambda v: D.association_level_label(grouping, v),
            disabled=grouping == "__all__", key=key)
        if grouping == "__all__":
            levels = ["__all__"]
        method = "Kruskal-Wallis" if categorical else st.radio(
            "Descriptive association method", ["Spearman", "Pearson", "Descriptive OLS"], horizontal=True, key="cc_analysis_method"
        )
        annotations = st.multiselect("Plot annotation fields", ["n", "coefficient", "p", "fdr"],
            default=["n", "coefficient", "p", "fdr"], key="cc_plot_annotations")
        pooled = False if categorical or grouping == "__all__" else st.checkbox(
            "Show pooled association across displayed donors", True, key="cc_plot_pooled")
        group_lines = False if categorical else st.checkbox("Show group-specific OLS trend lines", True, key="cc_plot_group_lines")
        pooled_dash = st.radio("Pooled correlation line style", ["dash", "solid"], horizontal=True,
            format_func=lambda v: "Dashed black" if v == "dash" else "Solid black", key="cc_plot_pooled_dash", disabled=not pooled)
        rule = "none" if categorical else st.selectbox("Trend-line rule", ["all", "p", "fdr", "none"],
            format_func=lambda v: {"all":"All eligible groups", "p":"Nominal p below cutoff", "fdr":"Module-set FDR below cutoff", "none":"No trend lines"}[v], key="cc_plot_line_rule")
        cutoff = st.radio("Association significance cutoff", [0.05, 0.10], horizontal=True, key="cc_plot_cutoff", disabled=rule not in {"p", "fdr"})
    with st.sidebar.expander("Plot appearance", expanded=False):
        color_options = {"__single_color__": "Single color", **{k:v for k,v in D.COLOR_LABELS.items() if k in frame}}
        color_by = st.selectbox("Color points by", list(color_options), index=list(color_options).index("diagnosis_group"),
            format_func=color_options.get, key="cc_plot_color")
        single = st.color_picker("Point color", "#2C7FB8", key="cc_plot_single_color") if color_by == "__single_color__" else None
        palette = st.selectbox("Continuous point-color scale", list(CONTINUOUS_COLOR_SCALES), key="cc_plot_palette")
        reverse = st.checkbox("Reverse color scale", False, key="cc_plot_reverse_palette")
    if not levels:
        st.info("Select at least one group level.")
        return method
    shown = frame if grouping == "__all__" else frame.loc[frame[grouping].isin(levels)].copy()
    if shown.empty:
        st.info("No donors match the selected group levels.")
        return method
    st.caption("Separate figures for each measure; tissue components are panels. Colors and diagnosis marker shapes follow the historical plots. OLS lines are visual guides, not adjusted models.")
    st.caption("Module-set FDR is BH across valid modules only, separately for each component, measure, outcome, donor group and correlation method. Archived q_scan/q_GL_outcome remain separate. Grouped fits use the historical SciPy two-sided Pearson/Spearman tests; matching pooled baseline rows retain their recorded inference method.")
    if baseline is not None and not baseline.empty:
        baseline = baseline.copy()
        baseline.loc[baseline.n.lt(minimum), "p"] = np.nan
        baseline = from_baseline(baseline)
        baseline = baseline.loc[baseline.module.eq(context["module"])]
    from .core import canonical_hash
    stats_levels = tuple([*levels, "__pooled__"] if pooled else levels)
    family_spec = dict(family=context["family"], estimator="calibrated_beta"+str(context["beta"]),
        scope=context["cohort_scope"], mask=tuple(context["mask_spec"]), tissue=context["regional_tissue"],
        transform="raw" if context["transform"] == "native" else context["transform"],
        components=tuple(context["components"]), metrics=tuple(context["metric_families"]),
        outcome=outcome, grouping=grouping, levels=tuple(levels) if categorical else stats_levels,
        minimum_n=minimum, donors=tuple(sorted(shown.sample_id.unique())), categorical=categorical,
        revision=context.get("catalog_revision"),
        use_baseline=baseline is not None and set(shown.sample_id) == set(frame.sample_id),
        inference_version=context.get("baseline_inference_version", "auto"))
    family_id = canonical_hash(family_spec)
    st.session_state["cc_current_plot_family"] = family_spec
    if st.button("Calculate module-set FDR for all displayed groups", key="cc_plot_family_fdr"):
        try:
            with st.spinner("Reading one module at a time; correcting each group/component/outcome separately…"):
                complete = module_statistics(**family_spec)
            st.session_state["cc_plot_family_statistics"] = (family_id, complete)
        except (OSError, ValueError) as error:
            st.warning("Full-family FDR unavailable; no partial family substituted: " + str(error))
    saved_family = st.session_state.get("cc_plot_family_statistics")
    complete = saved_family[1] if saved_family and saved_family[0] == family_id else None
    if complete is None:
        st.caption("Calculate module-set FDR above for grouped, Pearson, transformed or filtered tests. Until then those FDRs are unavailable—not corrected over just this module. Matching native pooled Spearman FDRs are available from the complete catalog.")
    subtitles = enrichment_subtitles(shown, context)
    corr_method = "spearman" if method == "Spearman" else "pearson"
    for metric in context["metric_families"]:
        part = panel_frame(shown.loc[shown.metric_family.eq(metric)], context["components"])
        if part.empty:
            continue
        st.subheader(labels[metric])
        plot_context = dict(context, metric_family=metric, plot_grouping=grouping, plot_levels=list(levels), plot_method=method)
        if categorical:
            statistics = calculate_categorical_associations(part, ["component"], category_column=outcome, min_group_n=minimum)
            if complete is not None:
                selected = complete.loc[complete.module.eq(context["module"]) & complete.metric_family.eq(metric)]
                statistics = selected.assign(categorical_fdr_across_modules=selected.q_across_modules)
            figure = categorical_association_figure(part, statistics, category_variable=outcome,
                category_label=D.ASSOCIATION_OUTCOME_LABELS[outcome], category_levels=levels,
                category_labels={str(v):D.association_level_label(outcome,v) for v in levels},
                feature_label=labels[metric], scale_label=context["transform"], module=context["module"],
                minimum_group_n=minimum, annotation_fields=["effect" if a == "coefficient" else a for a in annotations],
                module_definition=definition_label(context), kegg_subtitles=subtitles, hover_fields=D.HOVER_LABELS, single_point_color=single)
        else:
            base = baseline
            if grouping != "__all__" and set(shown.sample_id) != set(frame.sample_id):
                base = None
            if base is not None:
                base = base.loc[base.metric_family.eq(metric) & base.method.eq("Spearman")]
            narrow = list(dict.fromkeys(["sample_id", "component", "metric_value", outcome] + ([] if grouping == "__all__" else [grouping])))
            statistics = plot_statistics(part[narrow], outcome, grouping, tuple(stats_levels), minimum, base)
            if complete is not None:
                selected = complete.loc[complete.module.eq(context["module"]) & complete.metric_family.eq(metric)]
                for method_name, prefix in (("Spearman", "spearman"), ("Pearson", "pearson")):
                    exact = selected.loc[selected.method.eq(method_name)].set_index(["component", "grouping_level"])
                    for index, stat in statistics.iterrows():
                        key = (stat.component, stat.grouping_level)
                        if key in exact.index:
                            source = exact.loc[key]
                            statistics.loc[index, prefix+"_p"] = source.p
                            statistics.loc[index, prefix+"_fdr_across_modules"] = source.q_across_modules
                            statistics.loc[index, prefix+"_fdr_module_family_n"] = source.module_family_n
                            statistics.loc[index, prefix+"_p_method"] = source[prefix+"_p_method"]
                            statistics.loc[index, "fdr_scope"] = source.module_fdr_scope
            figure = grouped_association_figure(part, statistics,
                phenotype=outcome, phenotype_label=D.ASSOCIATION_OUTCOME_LABELS[outcome], feature_label=labels[metric],
                scale_label=context["transform"], grouping_variable=grouping, grouping_levels=levels,
                grouping_labels={str(v):D.association_level_label(grouping,v) for v in levels}, module=context["module"],
                color_by=color_by, color_label=color_options[color_by], hover_fields=D.HOVER_LABELS,
                correlation_method=corr_method, annotation_fields=annotations, trend_line_rule=rule, significance_cutoff=cutoff,
                minimum_group_n=minimum, show_group_trends=group_lines, show_pooled=pooled,
                continuous_colorscale=palette, reverse_colorscale=reverse, single_point_color=single,
                pooled_line_dash=pooled_dash, categorical_color_fields=D.CATEGORICAL_ONLY_ASSOCIATION_OUTCOMES,
                kegg_subtitles=subtitles, module_definition=definition_label(context))
        draw(figure, render, "cc_association_plot_"+metric, plot_context, preserve_layout=True)
        with st.expander(labels[metric] + " · plotted statistics"):
            displayed = statistics.drop(columns=[c for c in ("q_scan", "q_GL_outcome") if c in statistics and statistics[c].isna().all()])
            table(displayed, "cc_plot_statistics_"+metric, "Plotted association statistics")
    return method


def render_distributions(frame, context, render, draw):
    grouping = st.selectbox("Group distributions by", [g for g in D.ASSOCIATION_GROUP_LABELS if g == "__all__" or g in frame],
        format_func=D.ASSOCIATION_GROUP_LABELS.get, key="cc_distribution_group")
    levels = ordered_levels(frame, grouping)
    levels_key = "cc_distribution_levels_"+grouping
    if levels_key in st.session_state and not set(st.session_state[levels_key]) <= set(levels):
        del st.session_state[levels_key]
    selected = st.multiselect("Distribution group levels", levels, default=levels,
        format_func=lambda v:D.association_level_label(grouping,v), key=levels_key, disabled=grouping == "__all__")
    if grouping == "__all__":
        selected = ["__all__"]
    chart_type = st.radio("Distribution view", ["Histogram", "Violin", "Raincloud", "ECDF"], horizontal=True, key="cc_distribution_view")
    bins = st.slider("Histogram bins", 10, 80, 30, disabled=chart_type != "Histogram", key="cc_distribution_bins")
    shown = frame.copy() if grouping == "__all__" else frame.loc[frame[grouping].isin(selected)].copy()
    if grouping == "__all__":
        shown[grouping] = "__all__"
    shown["distribution_group"] = "All displayed donors" if grouping == "__all__" else shown[grouping].map(lambda v:D.association_level_label(grouping,v))
    groups = [D.association_level_label(grouping,v) for v in selected]
    for metric in context["metric_families"]:
        part = panel_frame(shown.loc[shown.metric_family.eq(metric)], context["components"])
        if part.empty:
            continue
        st.subheader(feature_labels()[metric])
        figure = distribution_figure(part, feature_labels()[metric], context["transform"], groups,
            context["module"], chart_type, bins=bins, module_definition=definition_label(context),
            group_column="distribution_group", group_label=D.ASSOCIATION_GROUP_LABELS[grouping])
        draw(figure, render, "cc_distributions_"+metric,
            dict(context, metric_family=metric, distribution_grouping=grouping, distribution_levels=list(selected)), preserve_layout=True)
    return shown, grouping
