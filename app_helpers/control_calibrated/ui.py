"""Ordinary descriptive app views backed by the shared score adapters.

Every plot uses the application's existing publication/export renderer. This
module is source-aware presentation; calibration remains an offline operation.
"""

from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from app_helpers import data as D
from . import adapter as A
from .registry import APP, ROOT, METRICS, PRIMARY, HELP, ESTIMATORS, feature_labels
from .inputs import atomic_json, atomic_parquet, sha
from .core import canonical_hash
from .analysis import bh
from . import hosted as H

COLORS = {
    "Gain / strengthening": "#167D9A",
    "Loss / weakening": "#C48028",
    "Control": "#2878A1",
    "MCI": "#D5A532",
    "AD": "#C66D42",
    "Unclassified": "#888888",
}


def table(frame, key, label="Results"):
    priority = [
        c
        for c in (
            "module_key",
            "component",
            "metric_family",
            "outcome",
            "n",
            "n_tested",
            "effect",
            "p",
            "q_scan",
            "q_GL_outcome",
            "q_search",
            "method",
            "beta",
            "cohort_scope",
            "unavailable_reason",
        )
        if c in frame
    ]
    frame = frame[priority + [c for c in frame if c not in priority]]
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button(
        "Download " + label.lower(),
        frame.to_csv(index=False).encode(),
        key + ".csv",
        "text/csv",
        key=key + "_download",
    )


def draw(figure, render, key, context, *, annotate=True):
    figure.update_layout(
        template="plotly_white",
        height=max(450, figure.layout.height or 450),
        font=dict(family="Arial", size=13),
        margin=dict(t=120, b=75, l=70, r=35),
        meta=context,
    )
    if annotate:
        figure.add_annotation(
            text=f"beta{context['beta']} · {context['cohort_scope']} · {context['transform']} · {context['mask']}",
            xref="paper",
            yref="paper",
            x=0,
            y=1.13,
            showarrow=False,
            xanchor="left",
            font=dict(size=11, color="#555555"),
        )
    for annotation in figure.layout.annotations:
        if str(annotation.text).startswith("component_label="):
            annotation.text = annotation.text.split("=", 1)[1]
    render(figure, key=key, metadata=context)


@st.cache_data(show_spinner=False, max_entries=1, ttl=600)
def load_module(
    family,
    estimator,
    module,
    scope,
    mask,
    fdr_scope,
    fdr_threshold,
    tissue,
    revision,
    metrics=None,
):
    return A.scores(
        module_set=family,
        estimator=estimator,
        module=module,
        cohort_scope=scope,
        differential_edge_rule=mask,
        differential_fdr_scope=fdr_scope,
        differential_fdr_threshold=fdr_threshold,
        regional_tissue=tissue,
        metric_families=metrics,
    )


def _ols(x, y, frame, covariates):
    cov = frame[covariates].copy()
    observed = cov.notna().all(axis=1).to_numpy()
    numeric = [
        c
        for c in cov
        if pd.api.types.is_numeric_dtype(cov[c])
        and c not in D.CATEGORICAL_OUTCOME_LABELS
    ]
    categorical = [c for c in cov if c not in numeric]
    if categorical:
        cov = pd.get_dummies(cov, columns=categorical, drop_first=True, dtype=float)
    z = np.column_stack(
        (np.ones(len(x)), x, cov.to_numpy(dtype=float, na_value=np.nan))
    )
    valid = np.isfinite(z).all(1) & np.isfinite(y) & observed
    z = z[valid]
    y = np.asarray(y)[valid]
    n = len(y)
    p = np.nan
    coef = np.nan
    se_value = np.nan
    r_squared = np.nan
    reason = ""
    rank = np.linalg.matrix_rank(z) if n else 0
    if rank < z.shape[1]:
        reason = "rank-deficient descriptive model"
    elif n <= z.shape[1] + 1:
        reason = "insufficient complete-case residual degrees of freedom"
    elif np.unique(y).size <= 1:
        reason = "constant outcome"
    else:
        inverse = np.linalg.inv(z.T @ z)
        b = inverse @ z.T @ y
        res = y - z @ b
        h = np.sum((z @ inverse) * z, axis=1)
        coef = b[1]
        r_squared = 1 - np.sum(res**2) / np.sum((y - y.mean()) ** 2)
        if np.any(h >= 1):
            reason = "undefined HC3 variance: unit leverage"
        else:
            meat = z.T @ ((res / (1 - h))[:, None] ** 2 * z)
            se = np.sqrt(np.maximum(np.diag(inverse @ meat @ inverse), 0))
            se_value = se[1]
            if np.isfinite(se_value) and se_value > 0:
                p = 2 * stats.t.sf(abs(coef / se_value), n - z.shape[1])
            else:
                reason = "undefined or zero HC3 standard error"
    return dict(
        n=n,
        effect=coef,
        p=p,
        unavailable_reason=reason,
        effect_name="OLS coefficient per score unit (HC3 SE)",
        standard_error_HC3=se_value,
        model_rank=rank,
        residual_df=n - z.shape[1],
        descriptive_in_sample_R_squared=r_squared,
        expanded_model_terms="Intercept, score"
        + (", " + ", ".join(cov.columns) if len(cov.columns) else ""),
    )


@st.cache_data(show_spinner=False, max_entries=8, ttl=600)
def descriptive_analysis(
    frame, context, outcomes, method="Spearman", covariates=(), minimum_n=3
):
    """Save exact searched context before computing donor-level tests."""
    from app_helpers.correlations import (
        calculate_correlations,
        calculate_categorical_associations,
    )

    spec = dict(
        context,
        outcomes=list(outcomes),
        method=method,
        covariates=list(covariates),
        minimum_n=minimum_n,
        donor_membership_sha256=canonical_hash(
            sorted(frame.sample_id.unique().tolist())
        ),
        tested_cells=sorted(
            frame[["module_key", "component", "metric_family"]]
            .drop_duplicates()
            .astype(str)
            .agg("::".join, axis=1)
            .tolist()
        ),
        correction="q_search BH across all finite tests in this saved selected-module analysis, both displayed G/L when selected; not baseline q_scan",
        role="independent target",
        prediction=False,
        numeric_inference="Spearman: exact tie-aware pairings where tractable, else fixed9999 Monte Carlo (+1), seed611104; Pearson remains a separately labeled parametric analysis",
        component_aliases="Legacy displayed-component family retained; exact aliases are not independent replication",
    )
    analysis_id = canonical_hash(spec)
    dest = ROOT / "associations" / "saved" / analysis_id
    atomic_json(dest.with_suffix(".json"), spec)
    rows = []
    for (key, component, metric), g in frame.groupby(
        ["module_key", "component", "metric_family"], sort=False
    ):
        for outcome in outcomes:
            base = dict(
                module_key=key,
                component=component,
                metric_family=metric,
                outcome=outcome,
                method=method,
                analysis_id=analysis_id,
                **{
                    k: v
                    for k, v in context.items()
                    if k
                    in (
                        "family",
                        "module",
                        "beta",
                        "cohort_scope",
                        "allocation_id",
                        "transform",
                        "mask",
                        "score_version",
                    )
                },
                role="independent_target",
                model=(
                    "descriptive_OLS_HC3"
                    if method == "Descriptive OLS"
                    else "unadjusted_descriptive"
                ),
                covariates=",".join(covariates),
                subgroup_membership_sha256=spec["donor_membership_sha256"],
            )
            if outcome in D.CATEGORICAL_OUTCOME_LABELS or (
                context.get("outcome") == outcome
                and context.get("outcome_interpretation") == "categorical"
            ):
                r = calculate_categorical_associations(
                    g.assign(_group=0),
                    ["_group"],
                    category_column=outcome,
                    min_group_n=5,
                ).iloc[0]
                rows.append(
                    dict(
                        base,
                        n=r["n_tested"],
                        group_counts=r["level_counts"],
                        effect=r.epsilon_squared,
                        effect_name="epsilon_squared",
                        p=r.categorical_p,
                        unavailable_reason=r.unavailable_reason,
                        method="Kruskal-Wallis",
                    )
                )
            elif method == "Descriptive OLS":
                rows.append(
                    dict(
                        base,
                        **_ols(
                            pd.to_numeric(g.metric_value, errors="coerce").to_numpy(
                                dtype=float, na_value=np.nan
                            ),
                            pd.to_numeric(g[outcome], errors="coerce").to_numpy(
                                dtype=float, na_value=np.nan
                            ),
                            g,
                            covariates,
                        ),
                    )
                )
            else:
                r = calculate_correlations(
                    g.assign(_group=0), ["_group"], [outcome], min_group_n=minimum_n
                ).iloc[0]
                prefix = method.lower()
                permutation = None
                if prefix == "spearman":
                    from .validation_inference import (
                        infer_group,
                        finite_numeric,
                        stable_seed,
                    )

                    x = finite_numeric(g.metric_value)
                    y = finite_numeric(g[outcome])
                    valid = np.isfinite(x) & np.isfinite(y)
                    permutation = infer_group(
                        x[valid][None, :],
                        y[valid],
                        g.loc[valid],
                        draws=9999,
                        seed=stable_seed(
                            611104, analysis_id, key, component, metric, outcome
                        ),
                    )
                rows.append(
                    dict(
                        base,
                        n=r["n"],
                        effect=r.spearman_rho if prefix == "spearman" else r.pearson_r,
                        effect_name=(
                            "spearman_rho" if prefix == "spearman" else "pearson_r"
                        ),
                        p=(
                            permutation["p"][0]
                            if permutation is not None
                            else r[prefix + "_p"]
                        ),
                        p_method=(
                            permutation["method"][0]
                            if permutation is not None
                            else "parametric_Pearson_t"
                        ),
                        permutation_draws=(
                            int(permutation["draws"][0])
                            if permutation is not None
                            else 0
                        ),
                        unavailable_reason=r.unavailable_reason,
                    )
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_search"] = bh(result.p)
        result["correction_scope"] = spec["correction"]
    atomic_parquet(dest.with_suffix(".parquet"), result)
    return result


def _filters(frame):
    with st.sidebar.expander("Metadata and subgroup filters"):
        fields = st.multiselect(
            "Filter metadata fields",
            [c for c in D.ASSOCIATION_OUTCOME_LABELS if c in frame],
            default=["diagnosis_group"],
            key="cc_filter_fields",
        )
        for col in fields:
            if (
                col in D.CATEGORICAL_OUTCOME_LABELS
                or col in D.SELECTABLE_ASSOCIATION_OUTCOMES
            ):
                levels = sorted(frame[col].dropna().unique().tolist(), key=str)
                widget = "cc_filter_" + col
                if widget in st.session_state and not set(
                    st.session_state[widget]
                ) <= set(levels):
                    del st.session_state[widget]
                selected = st.multiselect(
                    D.ASSOCIATION_OUTCOME_LABELS[col],
                    levels,
                    default=levels,
                    key=widget,
                )
                frame = frame[frame[col].isin(selected)]
            else:
                values = pd.to_numeric(frame[col], errors="coerce")
                finite = values[np.isfinite(values)]
                if finite.nunique() > 1:
                    bounds = st.slider(
                        D.ASSOCIATION_OUTCOME_LABELS[col],
                        float(finite.min()),
                        float(finite.max()),
                        (float(finite.min()), float(finite.max())),
                        key="cc_range_" + col,
                    )
                    frame = frame[values.between(*bounds)]
    return frame


def full_partition_search(
    context,
    estimator,
    scope,
    metrics,
    components,
    outcome,
    donors,
    mask,
    tissue,
    outcomes=None,
):
    outcomes = [outcome] if outcomes is None else list(outcomes)
    jobs = [
        j
        for j in A.catalog()["modules"]
        if j["family"] == context["family"]
        and (tissue is None or j["regional_tissue"] == tissue)
    ]
    intent = dict(
        context,
        planned_modules=[j["module_key"] for j in jobs],
        metrics=metrics,
        components=components,
        outcome=outcome,
        outcomes=outcomes,
        numeric_inference="Exact tie-aware pairings where tractable, else fixed9999 Monte Carlo (+1), seed611104",
        donor_membership_sha256=canonical_hash(sorted(donors)),
        method=(
            "Registered per-outcome Spearman / Kruskal-Wallis"
            if len(outcomes) > 1
            else (
                "Kruskal-Wallis"
                if outcome in D.CATEGORICAL_OUTCOME_LABELS
                or context.get("outcome_interpretation") == "categorical"
                else "Spearman"
            )
        ),
        correction="BH over every estimable selected metric/component/module/outcome in this full-partition search, before display filtering; q_search, separate from baseline q_scan",
    )
    aid = canonical_hash(intent)
    path = ROOT / "associations/saved" / aid
    if path.with_suffix(".parquet").exists():
        return pd.read_parquet(path.with_suffix(".parquet"))
    atomic_json(path.with_suffix(".json"), intent)
    parts = []
    for j in jobs:
        try:
            frame = A.scores(
                module_set=j["family"],
                module=j["module"],
                estimator=estimator,
                cohort_scope=scope,
                regional_tissue=tissue,
                metric_families=metrics,
                differential_edge_rule=mask["rule"],
                differential_fdr_scope=mask.get("fdr_scope", "global"),
                differential_fdr_threshold=mask.get("threshold", 0.05),
            )
            frame = frame[
                frame.metric_family.isin(metrics)
                & frame.component.isin(components)
                & frame.sample_id.isin(donors)
            ].copy()
            frame["metric_value"] = frame[
                "metric_"
                + ("raw" if context["transform"] == "native" else context["transform"])
            ]
            if frame.empty:
                continue
            result = descriptive_analysis(
                frame,
                dict(context, module=j["module"], search_parent=aid),
                outcomes,
            )
            result["module"] = j["module"]
            result["regional_tissue"] = j["regional_tissue"]
            parts.append(result)
        except (A.ScoreUnavailable, OSError) as error:
            parts.append(
                pd.DataFrame(
                    [
                        dict(
                            module_key=j["module_key"],
                            module=j["module"],
                            regional_tissue=j["regional_tissue"],
                            component=c,
                            metric_family=m,
                            outcome=selected_outcome,
                            method=(
                                "Kruskal-Wallis"
                                if selected_outcome in D.CATEGORICAL_OUTCOME_LABELS
                                or (
                                    selected_outcome == outcome
                                    and context.get("outcome_interpretation")
                                    == "categorical"
                                )
                                else "Spearman"
                            ),
                            effect=np.nan,
                            p=np.nan,
                            n=0,
                            unavailable_reason=str(error),
                        )
                        for c in components
                        for m in metrics
                        for selected_outcome in outcomes
                    ]
                )
            )
    if not parts:
        raise A.ScoreUnavailable("No modules or donor scores match this search.")
    result = pd.concat(parts, ignore_index=True)
    result["q_search"] = bh(result.p)
    result["analysis_id"] = aid
    result["correction_scope"] = intent["correction"]
    atomic_parquet(path.with_suffix(".parquet"), result)
    return result


def correlation_heatmap_figure(
    results, metrics, components, outcomes, all_modules, title, modules=None
):
    """One row per module/measure/component; keep unavailable cells as NA."""
    numeric = results.loc[
        results.method.eq("Spearman")
        & results.metric_family.isin(metrics)
        & results.component.isin(components)
        & results.outcome.isin(outcomes)
    ].copy()
    if numeric.empty and not (all_modules and modules):
        raise A.ScoreUnavailable(
            "No numeric correlation outcomes match these selections."
        )
    numeric["row"] = (
        numeric.metric_family.map(feature_labels()) + " · " + numeric.component
    )
    if all_modules:
        numeric["row"] = (
            "M" + numeric.module.astype(int).astype(str) + " · " + numeric["row"]
        )
    rows = (
        numeric.assign(
            _metric=numeric.metric_family.map({m: i for i, m in enumerate(metrics)}),
            _component=numeric.component.map({c: i for i, c in enumerate(components)}),
        )
        .sort_values((["module"] if all_modules else []) + ["_metric", "_component"])[
            "row"
        ]
        .drop_duplicates()
    )
    if all_modules and modules is not None:
        rows = [
            f"M{module} · {feature_labels()[metric]} · {component}"
            for module in sorted(modules)
            for metric in metrics
            for component in components
        ]
    matrix = numeric.pivot(index="row", columns="outcome", values="effect").reindex(
        index=rows, columns=outcomes
    )
    counts = numeric.pivot(index="row", columns="outcome", values="n").reindex(
        index=rows, columns=outcomes
    )
    matrix = matrix.rename(columns=D.ASSOCIATION_OUTCOME_LABELS)
    figure = px.imshow(
        matrix,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        labels={"color": "Spearman rho"},
        title=title,
    )
    figure.update_traces(
        customdata=counts.to_numpy(),
        hovertemplate="%{y}<br>%{x}<br>Spearman rho=%{z:.3f}<br>Paired donor N=%{customdata}<extra></extra>",
        hoverongaps=False,
    )
    figure.update_layout(height=max(450, 22 * len(matrix) + 220))
    figure.update_xaxes(
        title_text="Outcome", side="top", tickangle=-45, automargin=True
    )
    figure.update_layout(title=dict(y=0.995, yanchor="top", pad=dict(t=8)))
    if figure.layout.height > 900:
        figure.update_layout(
            coloraxis_colorbar=dict(lenmode="pixels", len=300, y=1, yanchor="top")
        )
    figure.update_yaxes(
        title_text=(
            "Module / donor measure / component"
            if all_modules
            else "Donor measure / component"
        ),
        tickmode="array",
        tickvals=matrix.index.tolist(),
        ticktext=matrix.index.tolist(),
        automargin=True,
    )
    return figure


def render_correlation_heatmaps(
    frame, context, estimator, scope, mask, jobs, baseline_compatible, render
):
    all_modules = (
        st.radio(
            "Correlation heatmap scope",
            ["Selected module", "All modules"],
            horizontal=True,
            key="cc_heatmap_scope",
        )
        == "All modules"
    )
    outcomes = st.multiselect(
        "Outcome columns in heatmap and table",
        D.NUMERIC_OUTCOMES,
        default=D.NUMERIC_OUTCOMES,
        format_func=D.ASSOCIATION_OUTCOME_LABELS.get,
        key="cc_heatmap_outcomes",
    )
    if not outcomes:
        st.info("Select at least one numeric outcome.")
        return
    metrics, components = context["metric_families"], context["components"]
    heatmap_context = dict(
        context, heatmap_scope="all_modules" if all_modules else "selected_module"
    )
    if all_modules:
        heatmap_context.update(
            module=None,
            planned_modules=[j["module_key"] for j in jobs],
            donor_membership_sha256=canonical_hash(
                sorted(frame.sample_id.unique().tolist())
            ),
            manifest_sha256=sha(H.ensure(ROOT / "manifest.json")),
        )
        heatmap_context.pop("donor_score_metadata_sha256", None)
    try:
        if baseline_compatible:
            results = A.baseline_associations(
                module_set=context["family"],
                estimator=estimator,
                cohort_scope=scope,
                module=None if all_modules else context["module"],
                regional_tissue=context["regional_tissue"],
                inference_version=context.get("baseline_inference_version", "auto"),
            )
            st.caption(
                "Frozen native, unmasked baseline. Saved q_scan and q_GL_outcome are preserved across module and outcome display filters."
            )
        elif all_modules:
            search_id = canonical_hash(heatmap_context)
            if st.button(
                "Save and calculate all-module heatmap", key="cc_run_heatmap_search"
            ):
                with st.spinner(
                    "Calculating correlations across all modules for the selected donors, mask and score scale…"
                ):
                    result = full_partition_search(
                        heatmap_context,
                        estimator,
                        scope,
                        metrics,
                        components,
                        context["outcome"],
                        frame.sample_id.unique().tolist(),
                        mask,
                        context["regional_tissue"],
                        outcomes=list(D.ASSOCIATION_OUTCOME_LABELS),
                    )
                st.session_state["cc_partition_heatmap"] = (search_id, result)
            saved = st.session_state.get("cc_partition_heatmap")
            if not saved or saved[0] != search_id:
                st.info(
                    "Calculate the all-module heatmap for these donors, measures, mask and score scale."
                )
                return
            results = saved[1]
            st.caption(
                "q_search covers all estimable modules, selected measures/components and registered outcomes in this saved search. Outcome display filters preserve that correction; baseline q_scan is not reused."
            )
        else:
            results = descriptive_analysis(
                frame, context, list(D.ASSOCIATION_OUTCOME_LABELS)
            )
            st.caption(
                "Heatmap recalculated for the selected donors, measures, mask and scale. Its selected-module q_search covers all registered outcomes and selected components; baseline q_scan is not reused."
            )
        results = results.loc[
            results.metric_family.isin(metrics)
            & results.component.isin(components)
            & results.outcome.isin(outcomes)
            & results.method.eq("Spearman")
        ].copy()
        title = context["family"] + (
            f" · All {len(jobs)} modules" if all_modules else f" · M{context['module']}"
        )
        if context["regional_tissue"]:
            title += " · " + context["regional_tissue"]
        figure = correlation_heatmap_figure(
            results,
            metrics,
            components,
            outcomes,
            all_modules,
            title
            + " · Correlation heatmaps"
            + f"<br><sup>beta{context['beta']} · {scope} · {context['transform']} · {context['mask']}</sup>",
            modules=[j["module"] for j in jobs] if all_modules else None,
        )
        displayed_modules = (
            len(jobs) if all_modules else int(results.module_key.nunique())
        )
        heatmap_context.update(
            displayed_outcomes=outcomes,
            displayed_modules=displayed_modules,
        )
        st.caption(
            f"Showing {displayed_modules} module(s), {len(figure.data[0].y)} measure/component rows and {len(outcomes)} numeric outcomes. Blank cells indicate unavailable correlations or undefined physical components; hover shows paired donor N."
        )
        if figure.layout.height > 900:
            st.caption(
                "Scroll within the heatmap to see every module. The statistics download includes every displayed module and outcome."
            )
            with st.container(height=900, border=False):
                draw(figure, render, "cc_heatmap", heatmap_context, annotate=False)
        else:
            draw(figure, render, "cc_heatmap", heatmap_context, annotate=False)
        table(results, "cc_heatmap_table", "Heatmap statistics")
    except (A.ScoreUnavailable, OSError, ValueError) as error:
        st.info(str(error))


def baseline_browser(
    family,
    estimator,
    scope,
    module,
    outcome,
    components,
    regional_tissue=None,
    inference_version="auto",
):
    panel = st.expander(
        "Full-catalog baseline associations · frozen q_scan", expanded=False,
        key='cc_baseline_panel', on_change='rerun',
    )
    if not panel.open:
        return
    with panel:
        try:
            frame = A.baseline_associations(
                module_set=family,
                estimator=estimator,
                cohort_scope=scope,
                phenotype=outcome,
                regional_tissue=regional_tissue,
                inference_version=inference_version,
            )
        except A.ScoreUnavailable as error:
            st.info(str(error))
            return
        st.caption(
            "All independent targets, native units, unmasked edges. q_scan includes all registered partitions, all 20 outcomes, G/L/S/A and both betas in this dataset/cohort/allocation. q_GL_outcome corrects G/L jointly for this outcome and beta. Display filters retain these saved corrections."
        )
        selected = st.radio(
            "Baseline display",
            ["Selected module", "All modules"],
            horizontal=True,
            key="cc_baseline_display",
        )
        top = st.number_input(
            "Maximum displayed baseline rows", 10, 10000, 100, key="cc_baseline_top"
        )
        if selected == "Selected module":
            frame = frame[frame.module.eq(module)]
        if components:
            frame = frame[frame.component.isin(components)]
        table(
            frame.sort_values("q_scan", na_position="last").head(int(top)),
            "cc_baseline",
            "Baseline associations",
        )


def same_module_comparison(frame, context, outcome, render):
    """Use the established paired-donor comparison helper for historical scores."""
    from app_helpers.association_score_comparison import (
        ScoreSeries,
        common_case_scores,
        comparison_statistics,
        score_comparison_figure,
    )

    panel = st.expander('Compare with historical Connectivity on shared donors',
                        key='cc_legacy_comparison_panel', on_change='rerun')
    if not panel.open:
        return
    with panel:
        st.caption(
            "Correlations use the same observed independent-target donor intersection. Historical Connectivity retains its native powers, normalization and reference exposure; it is not a second independent B protocol. This saved comparison has its own correction family."
        )
        method = st.radio(
            "Source-comparison correlation",
            ["spearman", "pearson"],
            horizontal=True,
            key="cc_legacy_comparison_method",
        )
        identity = canonical_hash(
            dict(context, comparison="historical_native", outcome=outcome)
        )
        if st.button(
            "Calculate shared-donor source comparison",
            key="cc_calculate_legacy_comparison",
        ):
            series = []
            columns = {}
            from .masks import options as mask_options

            selected_mask = mask_options(context["family"], context["module"])[
                context["mask"]
            ]
            historical_mask = dict(
                differential_edge_rule=selected_mask["rule"],
                differential_fdr_scope=selected_mask.get("fdr_scope", "global"),
                differential_fdr_threshold=selected_mask.get("threshold", 0.05),
            )
            for (metric, component), part in frame.groupby(
                ["metric_family", "component"], sort=False
            ):
                key = f"calibrated:{metric}:{component}"
                series.append(
                    ScoreSeries(
                        key,
                        metric,
                        component,
                        f"Calibrated beta{context['beta']}",
                        feature_labels()[metric] + " · " + component,
                        "#167D9A" if metric == "gain" else "#C48028",
                    )
                )
                columns[key] = part.set_index("sample_id").metric_value
            scale = "raw" if context["transform"] == "native" else context["transform"]
            for component in frame.component.unique():
                try:
                    legacy = (
                        D.load_resolved_scope(
                            "control_anchored",
                            module=context["module"],
                            metric_family="connectivity",
                            component=component,
                            module_set=context["family"],
                            metric_scale=scale,
                            **historical_mask,
                        )
                        if component not in ("CT", "TS")
                        else D.load_aggregate_scope(
                            "control_anchored",
                            module=context["module"],
                            metric_family="connectivity",
                            module_set=context["family"],
                            metric_scale=scale,
                            **historical_mask,
                        )
                    )
                    if legacy.empty:
                        st.info(
                            "Historical Connectivity has no saved "
                            + component
                            + " comparator."
                        )
                        continue
                    value = (
                        "metric_" + scale
                        if component not in ("CT", "TS")
                        else component + "_" + scale
                    )
                    key = "historical:" + component
                    columns[key] = legacy.set_index("sample_id")[value]
                    series.append(
                        ScoreSeries(
                            key,
                            "connectivity",
                            component,
                            "Historical native Connectivity",
                            "Historical Connectivity · "
                            + component
                            + " · "
                            + context["mask"],
                            "#7C6E9B",
                        )
                    )
                except (ValueError, FileNotFoundError, KeyError) as error:
                    st.info(
                        "Historical comparator unavailable for "
                        + component
                        + ": "
                        + str(error)
                    )
            if not any(s.family == "connectivity" for s in series):
                st.info(
                    "A historical Connectivity comparator is unavailable for this selection."
                )
                return
            common = common_case_scores(
                frame.drop_duplicates("sample_id"),
                columns,
                series,
                [outcome],
                context["module"],
            )
            intent = dict(
                context,
                comparison="same-module historical native Connectivity",
                outcome=outcome,
                series=[s.__dict__ for s in series],
                shared_donor_hash=canonical_hash(
                    sorted(common.sample_id.unique().tolist())
                ),
                methods=["Spearman", "Pearson"],
                correction="q_comparison across all selected score/component sources and both numeric methods, before display changes; not baseline q_scan",
            )
            aid = canonical_hash(intent)
            dest = ROOT / "associations/saved" / aid
            atomic_json(dest.with_suffix(".json"), intent)
            if common.empty:
                st.info(
                    "There are no shared finite donors for these score/component choices."
                )
                return
            result = comparison_statistics(common, outcome, minimum_n=3)
            p = np.r_[result.pearson_p, result.spearman_p]
            q = bh(p)
            result["q_comparison_pearson"] = q[: len(result)]
            result["q_comparison_spearman"] = q[len(result) :]
            result["analysis_id"] = aid
            atomic_parquet(dest.with_suffix(".parquet"), result)
            st.session_state["cc_legacy_comparison"] = (identity, result, series)
        saved = st.session_state.get("cc_legacy_comparison")
        if saved and saved[0] == identity:
            result, series = saved[1:]
            figure = score_comparison_figure(
                result,
                series,
                method=method,
                module=context["module"],
                outcome_label=D.ASSOCIATION_OUTCOME_LABELS[outcome],
                y_title="Correlation on shared independent-target donors",
                subtitle=f"Frozen B beta{context['beta']} versus historical Connectivity; N={result.n.min()} shared donors",
            )
            draw(figure, render, "cc_shared_source_comparison", context)
            table(
                result, "cc_shared_source_statistics", "Shared-donor source comparison"
            )


def render_view(
    view,
    family,
    estimator,
    render=None,
    regional_tissue=None,
    default_scope="complete_450",
):
    if render is None:
        from app_helpers.figure_export import render_exportable_plot

        render = render_exportable_plot
    from .validation_ui import select_validation

    if select_validation(view, family, estimator, render):
        return
    from .validation_data import local_root as local_validation_root

    baseline_inference_version = st.sidebar.selectbox(
        "20-outcome baseline inference",
        (
            ["repaired", "original"]
            if local_validation_root() is not None
            else ["original"]
        ),
        format_func=lambda v: (
            "Repaired tie-aware permutation inference"
            if v == "repaired"
            else "Original asymptotic inference (audit)"
        ),
        key="cc_baseline_inference_version",
    )
    c = A.catalog()
    beta = ESTIMATORS[estimator]
    jobs = [j for j in c["modules"] if j["family"] == family]
    regional = c["families"][family]["regional"]
    with st.sidebar:
        if regional:
            tissues = list(dict.fromkeys(j["regional_tissue"] for j in jobs))
            if (
                "cc_regional_tissue" in st.session_state
                and st.session_state["cc_regional_tissue"] not in tissues
            ):
                del st.session_state["cc_regional_tissue"]
            regional_tissue = st.selectbox(
                "Calibrated regional tissue", tissues, key="cc_regional_tissue"
            )
            jobs = [j for j in jobs if j["regional_tissue"] == regional_tissue]
        modules = [j["module"] for j in jobs]
        if (
            "cc_module" in st.session_state
            and st.session_state["cc_module"] not in modules
        ):
            del st.session_state["cc_module"]
        requested = st.query_params.get("screen_module", "")
        default_module = (
            int(requested)
            if str(requested).isdigit() and int(requested) in modules
            else modules[0]
        )
        module = st.selectbox(
            "Module", modules, index=modules.index(default_module), key="cc_module"
        )
        scopes = ["complete_450", "incomplete_region_validation", "maximum_component"]
        scope_labels = {
            v: f"{label} ({A.metadata(family,v).sample_id.nunique()})"
            for v, label in [
                ("complete_450", "Original independent targets"),
                (
                    "incomplete_region_validation",
                    "Incomplete-tissue independent targets",
                ),
                ("maximum_component", "All aligned independent targets"),
            ]
        }
        default_scope = (
            st.query_params.get("cc_scope", default_scope)
            if st.query_params.get("cc_scope", default_scope) in scopes
            else default_scope
        )
        scope = st.radio(
            "Donor cohort scope",
            scopes,
            index=scopes.index(default_scope),
            format_func=scope_labels.get,
            key="cc_scope",
        )
        labels = feature_labels()
        metrics = st.multiselect(
            "Donor connectivity measures",
            list(METRICS),
            default=["gain", "loss"],
            format_func=labels.get,
            key="cc_metrics",
        )
        scale = st.selectbox(
            "Score scale for plots and new analyses",
            ["raw", "asinh", "rint"],
            format_func=lambda v: {
                "raw": "Native standardized score units (default)",
                "asinh": "Asinh · scale fitted to R training diagnostics",
                "rint": "RINT · knots fitted to R training diagnostics",
            }[v],
            key="cc_scale",
        )
        from .masks import options as mask_options

        masks = mask_options(family, module, regional_tissue)
        mask_id = st.selectbox(
            "Frozen physical-edge mask",
            list(masks),
            format_func=lambda v: masks[v]["label"],
            key="cc_mask",
        )
        mask = masks[mask_id]
        outcome = st.selectbox(
            "Association outcome",
            list(D.ASSOCIATION_OUTCOME_LABELS),
            format_func=D.ASSOCIATION_OUTCOME_LABELS.get,
            key="cc_outcome",
        )
        interpretation = (
            st.radio(
                "Outcome interpretation",
                ["numeric", "categorical"],
                format_func=lambda v: (
                    "Numeric / ordinal" if v == "numeric" else "Categorical comparison"
                ),
                horizontal=True,
                key="cc_outcome_interpretation",
            )
            if outcome in D.SELECTABLE_ASSOCIATION_OUTCOMES
            else "categorical" if outcome in D.CATEGORICAL_OUTCOME_LABELS else "numeric"
        )
    from app_helpers.analysis_context import apply_controls
    applied = apply_controls(dict(module_set=family, estimator=estimator, method='control_calibrated',
        module=module, cohort_scope=scope, metrics=metrics, scale=scale, mask_id=mask_id,
        phenotype=outcome, interpretation=interpretation, regional_tissue=regional_tissue,
        baseline_inference_version=baseline_inference_version), view='calibrated:' + view)
    family = applied['module_set']; estimator = applied['estimator']; module = applied['module']
    scope = applied['cohort_scope']; metrics = applied['metrics']; scale = applied['scale']
    mask_id = applied['mask_id']; outcome = applied['phenotype']; interpretation = applied['interpretation']
    regional_tissue = applied['regional_tissue']; baseline_inference_version = applied['baseline_inference_version']
    beta = ESTIMATORS[estimator]
    jobs = [j for j in c['modules'] if j['family'] == family and (not regional or j['regional_tissue'] == regional_tissue)]
    mask = mask_options(family, module, regional_tissue)[mask_id]
    st.subheader(f"Control-calibrated connectivity · beta{beta}")
    st.caption(
        "In native units: G is magnitude gain, L is magnitude loss, S=G−L is net change, and A=G+L is total magnitude atypicality. G and L share one eligible physical-edge denominator."
    )
    with st.expander("How to read these donor measures"):
        st.write(HELP)
    st.caption(
        "Frozen B seed611010: brain R82/C41; motor R60/C30. R/C donors are excluded from every independent-target score and association here. Partitions retain historical discovery overlap; validation scopes have been examined previously."
    )
    if mask_id != "all":
        st.info(
            "Exploratory reuse of a historical AD–Control discovery mask. It changes the physical-edge aggregation set; per-edge R/C calibration is unchanged. Baseline q_scan does not cover this mask search."
        )
    mask_manifest = ROOT / "masks/manifest.json"
    if H.available(mask_manifest):
        missing = (
            H.read_json(mask_manifest)
            .get("unavailable", {})
            .get(next(j["module_key"] for j in jobs if j["module"] == module), {})
        )
        if missing:
            with st.expander("Unavailable mask mappings and exact reasons"):
                st.json(missing)
    try:
        frame = load_module(
            family,
            estimator,
            module,
            scope,
            mask["rule"],
            mask.get("fdr_scope", "global"),
            mask.get("threshold", 0.05),
            regional_tissue,
            (
                H.ensure(ROOT / "manifest.json").stat().st_mtime_ns,
                sha(Path(A.__file__)),
            ),
            tuple(metrics),
        )
    except (A.ScoreUnavailable, OSError, ValueError) as error:
        st.info(str(error))
        return
    frame = frame[frame.metric_family.isin(metrics)].copy()
    if frame.empty:
        st.info("Select at least one donor measure.")
        return
    frame["metric_value"] = frame["metric_" + scale]
    if scale != "raw":
        frame["native_unavailable_reason"] = frame.unavailable_reason
        unavailable = np.isfinite(frame.metric_raw) & ~np.isfinite(frame.metric_value)
        frame.loc[unavailable, "unavailable_reason"] = (
            "unavailable_frozen_R_diagnostic_transform"
        )
        st.caption(
            "The chosen transform is applied once, separately to each native metric using its frozen R diagnostic fit. The G/L/S/A identities refer to native units; baseline q_scan remains native. A constant R diagnostic distribution makes RINT unavailable."
        )
    frame["measure"] = frame.metric_family.map(labels)
    frame["score_scale"] = scale
    frame["score_source"] = f"Control-calibrated connectivity beta{beta}"
    with st.sidebar:
        available = frame[["component", "component_label"]].drop_duplicates()
        mapping = dict(available.itertuples(index=False, name=None))
        default = (
            ["TS_" + A.component_legacy(regional_tissue)]
            if regional
            else [v for v in ("TS", "CT") if v in mapping]
            or available.component.tolist()
        )
        if "cc_components" in st.session_state and not set(
            st.session_state["cc_components"]
        ) <= set(mapping):
            del st.session_state["cc_components"]
        components = st.multiselect(
            "Tissue / tissue-pair / fixed union",
            available.component.tolist(),
            default=default,
            format_func=mapping.get,
            key="cc_components",
        )
    frame = frame[frame.component.isin(components)].copy()
    frame = _filters(frame)
    if frame.empty:
        st.info("No donors or components match these filters.")
        return
    content_hash = hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
    ).hexdigest()
    context = dict(
        family=family,
        module=module,
        beta=beta,
        cohort_scope=scope,
        allocation_id=frame.allocation_id.iloc[0],
        transform="native" if scale == "raw" else scale,
        mask=mask_id,
        score_version=f"control_calibrated_connectivity.v2.beta{beta}",
        regional_tissue=regional_tissue,
        outcome=outcome,
        outcome_interpretation=interpretation,
        metric_families=metrics,
        components=components,
        donor_score_metadata_sha256=content_hash,
        baseline_inference_version=baseline_inference_version,
    )
    finite = np.isfinite(frame.metric_value)
    all_module_heatmap = (
        view == "Correlation heatmaps"
        and st.session_state.get("cc_heatmap_scope") == "All modules"
    )
    a, b, d = st.columns(3)
    a.metric("Independent target candidates", frame.sample_id.nunique())
    b.metric(
        (
            "Finite targets in sidebar module"
            if all_module_heatmap
            else "Targets with selected finite scores"
        ),
        frame.loc[finite, "sample_id"].nunique(),
    )
    d.metric(
        (
            "Sidebar module cells available"
            if all_module_heatmap
            else "Score / component cells available"
        ),
        f"{finite.sum():,} / {len(frame):,}",
    )
    with st.expander("Coverage, denominators and score provenance"):
        cols = [
            "module_key",
            "component",
            "metric_family",
            "possible_edges",
            "eligible_edges",
            "signed_eligible_edges",
            "unavailable_reason",
        ]
        counts = (
            frame.groupby([x for x in cols if x in frame], dropna=False)
            .size()
            .rename("donors")
            .reset_index()
        )
        table(counts, "cc_coverage", "Score coverage")
        st.json(context)
        imputation = [
            v
            for v in ("sparse_endpoint_cells_imputed", "absent_tissue_endpoint_cells")
            if v in frame
        ]
        if imputation:
            st.dataframe(
                frame.drop_duplicates(["sample_id", "component"])
                .groupby("component")[imputation]
                .agg(["sum", "max"]),
                use_container_width=True,
            )
            st.caption(
                "Counts refer to unique selected physical endpoints. Present-tissue sparse cells use frozen R-only means; absent tissue endpoints remain unavailable."
            )
    title = f"{family} · M{module} · {view}"
    good = frame.loc[finite].copy()
    baseline_compatible = (
        mask_id == "all"
        and scale == "raw"
        and set(metrics) <= set(PRIMARY)
        and set(frame.sample_id.unique()) == set(A.metadata(family, scope).sample_id)
        and not (
            outcome in D.SELECTABLE_ASSOCIATION_OUTCOMES
            and interpretation == "categorical"
        )
    )
    if view in (
        "Edge summaries",
        "Edge volcano",
        "Donor edge explorer",
        "Gene & edge associations",
    ):
        from .inspection_ui import render_inspection

        render_inspection(view, frame, context, render, mask)
        return
    if view in ("KEGG enrichment", "Module details"):
        if regional:
            from app_helpers.regional_modules import (
                load_regional_details,
                load_regional_kegg,
            )

            details = load_regional_details()
            details = details[
                details.partition_source.eq(family)
                & details.tissue.eq(regional_tissue)
                & details.module.eq(module)
            ]
            table(details, "cc_details", "Module details")
            if view == "KEGG enrichment":
                table(
                    load_regional_kegg().query(
                        "partition_source == @family and tissue == @regional_tissue and module == @module"
                    ),
                    "cc_kegg",
                    "KEGG enrichment",
                )
        else:
            table(
                D.load_module_details(module=module, module_set=family),
                "cc_details",
                "Module details",
            )
            if view == "KEGG enrichment":
                table(
                    D.load_kegg(module=module, module_set=family),
                    "cc_kegg",
                    "KEGG enrichment",
                )
        return
    if view == "Methods & data":
        st.markdown((APP / "docs/control_calibrated_connectivity.md").read_text())
        st.json(H.read_json(ROOT / "associations/testing_manifest.json"))
        table(A.coverage(family), "cc_catalog_coverage", "Full catalog coverage")
        return
    if view in ("Statistics", "Module finder", "CT–TS screen"):
        try:
            if baseline_compatible:
                results = A.baseline_associations(
                    module_set=family,
                    estimator=estimator,
                    cohort_scope=scope,
                    phenotype=outcome,
                    regional_tissue=regional_tissue,
                    inference_version=baseline_inference_version,
                )
                results = results[
                    results.metric_family.isin(metrics)
                    & results.component.isin(components)
                ]
                q_column = "q_scan"
                st.caption(
                    "Frozen native, unmasked baseline; q_scan includes both betas and the full scan, before these display filters."
                )
            else:
                if not st.button(
                    "Save and calculate full-partition search for these filters",
                    key="cc_run_partition_search",
                ):
                    st.info(
                        "These donors, metric, mask or transform define an additional search. Calculate it to view matching rankings and q_search."
                    )
                    return
                with st.spinner(
                    "Reading cached donor scores and calculating this declared descriptive search…"
                ):
                    results = full_partition_search(
                        context,
                        estimator,
                        scope,
                        metrics,
                        components,
                        outcome,
                        frame.sample_id.unique().tolist(),
                        mask,
                        regional_tissue,
                    )
                q_column = "q_search"
                st.caption(
                    "q_search covers the complete selected partition/metric/component search before Top K. It is separate from baseline q_scan."
                )
            top = st.number_input("Top rows", 10, 10000, 100, key="cc_top_rows")
            table(
                results.sort_values(q_column).head(int(top)),
                "cc_rankings",
                "Association ranking",
            )
            if view == "CT–TS screen":
                pivot = (
                    results[results.component.isin(["CT", "TS"])]
                    .pivot_table(
                        index=["module", "metric_family", "outcome"],
                        columns="component",
                        values="effect",
                    )
                    .reset_index()
                )
                if {"CT", "TS"} <= set(pivot):
                    draw(
                        px.scatter(
                            pivot,
                            x="TS",
                            y="CT",
                            color="metric_family",
                            hover_data=["module"],
                            title=title,
                            labels={
                                "TS": "TS outcome effect",
                                "CT": "CT outcome effect",
                            },
                        ),
                        render,
                        "cc_ct_ts",
                        context,
                    )
        except A.ScoreUnavailable as error:
            st.info(str(error))
        return
    if view == "Correlation heatmaps":
        render_correlation_heatmaps(
            frame, context, estimator, scope, mask, jobs, baseline_compatible, render
        )
        return
    elif view == "Feature distributions":
        grouping = st.selectbox(
            "Distribution grouping",
            [v for v in D.ASSOCIATION_GROUP_LABELS if v in frame],
            format_func=D.ASSOCIATION_GROUP_LABELS.get,
            key="cc_distribution_group",
        )
        draw(
            px.box(
                good,
                x=grouping,
                y="metric_value",
                color="measure",
                facet_col="component_label",
                points="outliers",
                color_discrete_map=COLORS,
                title=title,
                labels={"metric_value": context["transform"] + " score units"},
            ),
            render,
            "cc_distributions",
            context,
        )
        from app_helpers.distributions import calculate_pairwise_distribution_statistics

        levels = sorted(good[grouping].dropna().unique(), key=str)
        if len(levels) >= 2 and st.checkbox(
            "Calculate descriptive group differences", key="cc_group_differences"
        ):
            minimum = st.number_input(
                "Minimum donors per comparison group", 3, 100, 10, key="cc_group_min"
            )
            contrasts = tuple(
                (a, b) for ai, a in enumerate(levels) for b in levels[ai + 1 :]
            )
            intent = dict(
                context,
                grouping=grouping,
                contrasts=[list(p) for p in contrasts],
                minimum_n=int(minimum),
                donors=canonical_hash(sorted(frame.sample_id.unique())),
                correction="Pairwise q values belong to this selected-module search; not baseline q_scan",
            )
            aid = canonical_hash(intent)
            atomic_json(ROOT / "associations/saved" / f"{aid}.json", intent)
            result = calculate_pairwise_distribution_statistics(
                good,
                ["module", "metric_family", "component", "component_label"],
                category_column=grouping,
                contrasts=contrasts,
                minimum_group_n=int(minimum),
                bootstrap_resamples=1000,
                seed=42,
                include_ks=True,
            )
            atomic_parquet(ROOT / "associations/saved" / f"{aid}.parquet", result)
            table(result, "cc_group_differences_table", "Group differences")
            st.caption(
                "Intervals use 1,000 donor resamples within groups, seed42. Edges are not donor replicates."
            )
    elif view == "Donor explorer":
        donor = st.selectbox(
            "Independent target donor", sorted(frame.sample_id.unique()), key="cc_donor"
        )
        selected = frame[frame.sample_id.eq(donor)]
        draw(
            px.bar(
                selected,
                x="component_label",
                y="metric_value",
                color="measure",
                barmode="group",
                color_discrete_map=COLORS,
                title=title + " · " + donor,
            ),
            render,
            "cc_donor_plot",
            context,
        )
        table(selected, "cc_donor_table", "Donor connectivity and coverage")
    else:
        paired = (
            good.loc[good[outcome].notna()]
            .groupby(["component_label", "measure"])
            .sample_id.nunique()
        )
        st.caption(
            "Observed donor pairs for this outcome: "
            + "; ".join(
                f"{component} · {measure}: N={n}"
                for (component, measure), n in paired.items()
            )
        )
        if interpretation == "categorical":
            draw(
                px.box(
                    good,
                    x=outcome,
                    y="metric_value",
                    color="measure",
                    facet_col="component_label",
                    color_discrete_map=COLORS,
                    title=title,
                    labels={"metric_value": context["transform"] + " score units"},
                ),
                render,
                "cc_association_plot",
                context,
            )
        else:
            draw(
                px.scatter(
                    good,
                    x=outcome,
                    y="metric_value",
                    color="measure",
                    symbol="diagnosis_group",
                    facet_col="component_label",
                    hover_data=["sample_id", "diagnosis_group"],
                    color_discrete_map=COLORS,
                    title=title,
                    labels={
                        "metric_value": context["transform"] + " score units",
                        outcome: D.ASSOCIATION_OUTCOME_LABELS[outcome],
                    },
                ),
                render,
                "cc_association_plot",
                context,
            )
        method = (
            st.radio(
                "Descriptive association method",
                ["Spearman", "Pearson", "Descriptive OLS"],
                horizontal=True,
                key="cc_analysis_method",
            )
            if interpretation != "categorical"
            else "Kruskal-Wallis"
        )
        covariates = []
        if interpretation == "categorical":
            st.caption(
                "Kruskal–Wallis comparison; at least5 donors per included group. Category codes are labels, never an arbitrary numeric ordering."
            )
        if method == "Descriptive OLS":
            covariates = st.multiselect(
                "Explicit model covariates",
                [
                    x
                    for x in D.ASSOCIATION_OUTCOME_LABELS
                    if x in frame and x != outcome
                ],
                default=[],
                key="cc_covariates",
            )
            st.caption(
                "Descriptive complete-case OLS with HC3 standard errors. Supplied cognitive/motor slopes keep their existing adjustment; extra covariates are included only when explicitly selected."
            )
        analysis_view_id = canonical_hash(
            dict(context, method=method, covariates=covariates)
        )
        if st.button(
            "Save and calculate this descriptive analysis", key="cc_calculate_analysis"
        ):
            result = descriptive_analysis(frame, context, [outcome], method, covariates)
            st.session_state["cc_current_analysis"] = (analysis_view_id, result)
        saved = st.session_state.get("cc_current_analysis")
        if saved and saved[0] == analysis_view_id:
            table(saved[1], "cc_selected_analysis", "Saved descriptive association")
        st.caption(
            "A changed donor set, transform, method or mask is a separate saved search. q_search covers its declared selected-module tests; it does not inherit baseline q_scan or cover choosing among saved searches."
        )
    baseline_browser(
        family,
        estimator,
        scope,
        module,
        outcome,
        components,
        regional_tissue,
        baseline_inference_version,
    )
    if (
        view == "Associations"
        and not regional
        and scope == "complete_450"
        and interpretation == "numeric"
    ):
        same_module_comparison(frame, context, outcome, render)
    panel = st.expander('Donor scores and download', key='cc_donor_scores_panel', on_change='rerun')
    if panel.open:
        with panel:
            table(frame, 'cc_scores', 'Donor scores')
