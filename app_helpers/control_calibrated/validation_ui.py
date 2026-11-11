"""Association and sensitivity results, explicitly separate from score units."""

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app_helpers import data as D
from . import validation_data as V
from .registry import ESTIMATORS

COHORT_LABELS = {
    "original": "Original independent targets",
    "incomplete": "Incomplete-tissue independent targets",
}
MODEL_LABELS = {
    "A": "A · all-available unadjusted",
    "B": "B · matched-sample unadjusted",
    "C": "C · covariate-adjusted partial ranks",
    "C_age_spline": "Age-spline sensitivity",
}


def _table(frame, key, label):
    priority = [
        c
        for c in [
            "family",
            "module",
            "component",
            "metric_family",
            "series",
            "outcome",
            "cohort",
            "study_stratum",
            "model",
            "n",
            "effect",
            "ci_low",
            "ci_high",
            "p",
            "q_across_modules",
            "module_family_n",
            "q_scan",
            "q_GL_outcome",
            "q_focused",
            "unavailable_reason",
        ]
        if c in frame
    ]
    frame = frame[priority + [c for c in frame if c not in priority]]
    st.dataframe(frame.head(5000), hide_index=True, width="stretch")
    if len(frame) > 5000:
        st.caption(
            f"Table preview: first 5,000 rows. Download contains all {len(frame):,} filtered rows; correction preceded display filtering."
        )
    st.download_button(
        "Download " + label,
        frame.to_csv(index=False).encode(),
        key + ".csv",
        "text/csv",
        key=key + "_download",
    )


def _plot(frame, key, render, source, label="Rank association"):
    if frame.empty:
        st.info("No results match these selections.")
        return
    f = frame.copy()
    if "module_key" not in f:
        f["module_key"] = f.family + "::M" + f.module.astype(str)
    f["row"] = f.module_key + " · " + f.component + " · " + f.metric_family
    f["cohort_label"] = f.cohort.map(COHORT_LABELS)
    f["display_model"] = f.get(
        "display_model", f.model.map(lambda v: MODEL_LABELS.get(v, v))
    )
    present = list(f.display_model.drop_duplicates())
    preferred = list(MODEL_LABELS.values()) + ["ROS · C adjusted", "MAP · C adjusted"]
    model_order = [m for m in preferred if m in present] + [
        m for m in present if m not in preferred
    ]
    row_order = list(f.row.drop_duplicates())
    f["estimate_status"] = np.where(
        f.get("primary_eligible", f.p.notna()),
        "Eligible estimate",
        "Descriptive / ineligible",
    )
    # Keep NA rows in the accompanying table; a blank interval is not zero.
    figure = px.scatter(
        f,
        x="effect",
        y="row",
        color="cohort_label",
        symbol="estimate_status",
        facet_col="display_model",
        category_orders={"display_model": model_order},
        symbol_map={"Eligible estimate": "circle", "Descriptive / ineligible": "x"},
        hover_data=[
            c
            for c in [
                "n",
                "outcome",
                "beta",
                "cohort",
                "p",
                "q_across_modules",
                "module_family_n",
                "q_scan",
                "q_GL_outcome",
                "q_focused",
                "unavailable_reason",
            ]
            if c in f
        ],
        color_discrete_map={
            "Original independent targets": "#167D9A",
            "Incomplete-tissue independent targets": "#C48028",
        },
        title=f"{label} · beta{source['beta']} · {D.ASSOCIATION_OUTCOME_LABELS[source['outcome']]}<br><sup>Unmasked · exploratory · correction within the declared model/cohort/study family; N in hover</sup>",
        labels={
            "effect": label,
            "row": "Family / module / physical component / measure",
        },
    )
    # Draw endpoints directly: percentile intervals need not contain the point
    # estimate, so clipping negative error lengths would alter the interval.
    for col, model in enumerate(model_order, 1):
        for cohort, color in [("original", "#167D9A"), ("incomplete", "#C48028")]:
            part = f.loc[
                f.display_model.eq(model)
                & f.cohort.eq(cohort)
                & np.isfinite(f.ci_low)
                & np.isfinite(f.ci_high)
            ]
            xs = []
            ys = []
            for _, r in part.iterrows():
                xs.extend([r.ci_low, r.ci_high, None])
                ys.extend([r.row, r.row, None])
            figure.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )
    figure.update_layout(
        template="plotly_white",
        height=max(480, 30 * f.row.nunique() + 230),
        margin=dict(t=150),
        meta=source,
        legend_title_text="",
    )
    # Fix the complete categorical domain explicitly. Shared subplot axes can
    # otherwise inherit a short range from a previous figure in the browser.
    figure.update_yaxes(
        categoryorder="array",
        categoryarray=row_order,
        range=[len(row_order) - 0.5, -0.5],
        autorange=False,
    )
    figure.for_each_annotation(
        lambda a: a.update(text=a.text.replace("display_model=", ""))
    )
    figure.add_vline(x=0, line_color="#777777", line_dash="dot")
    if label == "Rank association":
        figure.update_xaxes(range=[-1, 1])
    render(figure, key=key, metadata=source)


def select_validation(view, family, estimator, render):
    if view not in (
        "Associations",
        "Statistics",
        "Correlation heatmaps",
        "Module finder",
        "CT–TS screen",
    ):
        return False
    choice = st.sidebar.selectbox(
        "Association results",
        [
            "Donor-score views / 20-outcome baseline",
            "Covariate validation / 8-outcome scan",
        ],
        key="cc_validation_mode",
    )
    if not choice.startswith("Covariate"):
        return False
    st.subheader("Control-calibrated connectivity · covariate validation")
    if V.local_root() is None:
        st.info(
            "These validation analyses are available in the local research app after processing completes. They have not been published to the hosted app."
        )
        return True
    show(family, ESTIMATORS[estimator], render)
    return True


def show(family, beta, render):
    spec = V.read_json("analysis_spec.json")
    summary = V.manifest()
    dataset = spec["families"][family]["dataset"]
    panel = st.radio(
        "Validation view",
        [
            "Covariate triplets",
            "ROS / MAP",
            "Focused comparisons",
            "Composition / missingness",
            "Baseline inference repair",
            "Methods / provenance",
        ],
        horizontal=True,
        key="ccv_panel",
    )
    st.caption(
        f"beta{beta} · {'primary' if beta==3 else 'sensitivity'} · native frozen G/L/S/A · unmasked · analysis {spec['analysis_id'][:12]}"
    )
    st.caption(
        "Exploratory ROSMAP associations: these cohorts have been examined before and historical module discovery can overlap original targets. Model adjustment and complete-case selection do not establish causality or resilience."
    )
    if panel == "Methods / provenance":
        st.json(spec)
        st.json(V.read_json("provenance_inventory.json"))
        st.json(summary)
        return
    if panel == "Baseline inference repair":
        repair = V.read_json("repair_complete.json")
        st.json(repair)
        st.caption(
            "The repaired version uses tie-aware permutation Spearman inference and recomputes the complete original 20-outcome correction families. Nominal tests and legacy alias membership are preserved. The old release remains available for audit."
        )
        scope = st.selectbox(
            "Baseline donor scope",
            ["complete_450", "incomplete_region_validation", "maximum_component"],
            key="ccv_repair_scope",
        )
        filters = [
            ("dataset", "=", dataset),
            ("cohort_scope", "=", scope),
            (
                "module_key",
                "in",
                [
                    j["module_key"]
                    for j in __import__(
                        "app_helpers.control_calibrated.adapter", fromlist=["catalog"]
                    ).catalog()["modules"]
                    if j["family"] == family
                ],
            ),
        ]
        changes = V.read_table("repaired_baseline/before_after.parquet", filters)
        if st.checkbox("Only changed p-values", True, key="ccv_only_changed"):
            changes = changes.loc[changes.p_changed]
        if st.checkbox("Only repaired zero p-values", False, key="ccv_only_zero"):
            changes = changes.loc[changes.p_old.eq(0)]
        _table(changes, "ccv_baseline_changes", "complete filtered baseline changes")
        st.caption(
            "Maximum-component donors overlap the two main cohorts; this is descriptive, not a third independent replication."
        )
        return
    if panel == "Composition / missingness":
        _table(
            V.read_table("results/composition.parquet"),
            "ccv_composition",
            "cohort composition",
        )
        _table(
            V.read_table(
                "results/missingness.parquet",
                filters=[("family", "=", family), ("beta", "=", beta)],
            ),
            "ccv_missingness",
            "score and outcome missingness",
        )
        return
    outcomes = list(spec["outcomes"])
    outcome = st.selectbox(
        "Validation clinical endpoint",
        outcomes,
        format_func=D.ASSOCIATION_OUTCOME_LABELS.get,
        key="ccv_outcome",
    )
    if "slope" in outcome:
        st.info(
            "This endpoint is a supplied adjusted slope. A/B describe association with its supplied values; C is an additional demographic/study sensitivity on the tested subset. Stored outcome values are unchanged. Exact original generating-release model details remain partly unresolved; pathology is not added again automatically."
        )
    cohorts = st.multiselect(
        "Independent donor cohorts",
        list(COHORT_LABELS),
        default=list(COHORT_LABELS),
        format_func=COHORT_LABELS.get,
        key="ccv_cohorts",
    )
    source = dict(
        analysis_id=spec["analysis_id"],
        family=family,
        beta=beta,
        outcome=outcome,
        cohorts=cohorts,
        mask="all",
        transform="native",
        role="independent_targets",
        correction_scope="Module-only BH within fixed model, cohort, study stratum, component, measure, outcome and beta. Archived q_scan and focused-family corrections are preserved separately.",
        interval="Study-stratified donor percentile bootstrap; rerank/refit; no edge resampling",
    )
    if panel == "Focused comparisons":
        kind = st.selectbox(
            "Focused result",
            [
                "Triplet / age spline",
                "Score-by-study interaction",
                "Shared-donor Connectivity / eigengenes",
                "Joint G/L beyond comparators",
            ],
            key="ccv_focused_kind",
        )
        if kind == "Triplet / age spline":
            parts = []
            for fam in spec["focused"]["modules"]:
                parts.append(
                    V.results(
                        fam,
                        beta,
                        cohorts,
                        models=["A", "B", "C", "C_age_spline"],
                        strata=["all"],
                        outcome=outcome,
                    ).loc[lambda f: f.focused]
                )
            f = pd.concat(parts, ignore_index=True)
            st.caption(
                "Base A/B/C q values cover the full canonical catalog. The prespecified age-spline sensitivity runs only on these focused candidates and has a separate, within-focused correction. A smaller q here does not select the spline model over the base model."
            )
        else:
            key = {
                "Score-by-study interaction": "interaction",
                "Shared-donor Connectivity / eigengenes": "comparators",
                "Joint G/L beyond comparators": "incremental",
            }[kind]
            f = (
                V.read_table(
                    f"results/{key}.parquet",
                    filters=[
                        ("beta", "=", beta),
                        ("outcome", "=", outcome),
                        ("cohort", "in", cohorts),
                    ],
                )
                if cohorts
                else pd.DataFrame()
            )
            source["correction_scope"] = (
                "Prespecified within-focused model/cohort family across named modules, both components, all eight endpoints and both betas; shared comparator aliases counted once"
            )
            if key == "interaction":
                st.caption(
                    "Interaction uses one pooled standardized rank scale, both main effects and the score-by-study term. Primary p: study-restricted Freedman–Lane; HC3 sensitivity and t intervals are separate. Within-study significance differences do not establish heterogeneity."
                )
                source["interval"] = "HC3 t interval for pooled interaction coefficient"
            if key == "incremental":
                st.caption(
                    "Incremental in-sample rank-outcome R² and a joint two-term G/L test. Both models use identical donors and comparators. This is explanatory association; high G/L collinearity makes individual coefficients unstable."
                )
            if key == "comparators":
                st.caption(
                    "Every series in a context uses the same complete-case donors and covariates. Historical Connectivity uses its existing reference/powers; eigengenes retain the existing 450-donor PC1 fits. Missing incomplete-cohort comparator values are not substituted."
                )
        if not f.empty:
            source.update(
                families=sorted(f.family.unique()),
                components=sorted(f.component.unique()),
                models=sorted(f.model.unique()),
            )
            if f.family.nunique() > 1:
                source["family"] = " / ".join(sorted(f.family.unique()))
            _plot(
                f,
                "ccv_focused_plot",
                render,
                source,
                label=(
                    "Incremental in-sample rank R²"
                    if kind.startswith("Joint")
                    else (
                        "Pooled rank interaction coefficient"
                        if kind.startswith("Score-by")
                        else "Rank association"
                    )
                ),
            )
            _table(f, "ccv_focused_table", "focused results")
        else:
            st.info("Select at least one independent donor cohort.")
        return
    if panel == "ROS / MAP" and dataset != "brain":
        st.info(
            "ROS/MAP sensitivity is not applicable to these single-study MAP motor data. The base adjusted analysis records and removes the constant study term."
        )
        return
    models = ["C"] if panel == "ROS / MAP" else ["A", "B", "C"]
    strata = ["ROS", "MAP"] if panel == "ROS / MAP" else ["all"]
    f = V.results(family, beta, cohorts, models=models, strata=strata, outcome=outcome)
    if f.empty:
        st.info("Select at least one cohort.")
        return
    modules = sorted(f.module.unique().tolist())
    chosen = st.multiselect(
        "Validation modules (empty selects all)", modules, default=[], key="ccv_modules"
    )
    if chosen:
        f = f.loc[f.module.isin(chosen)]
    components = list(f.component.drop_duplicates())
    chosen_components = st.multiselect(
        "Validation physical components",
        components,
        default=[c for c in ("TS_DLPFC", "CT_DLPFC__PCG", "TS_VH") if c in components]
        or components[:1],
        key="ccv_components",
    )
    metrics = st.multiselect(
        "Validation donor measures",
        list(METRICS_UI),
        default=["gain", "loss"],
        format_func=METRICS_UI.get,
        key="ccv_metrics",
    )
    f = f.loc[
        f.component.isin(chosen_components) & f.metric_family.isin(metrics)
    ].copy()
    if panel == "ROS / MAP":
        f["display_model"] = f.study_stratum + " · C adjusted"
    shown_modules = sorted(f.module.unique())[:12]
    shown = f.loc[f.module.isin(shown_modules)]
    source.update(
        models=models,
        study_strata=strata,
        components=chosen_components,
        metrics=metrics,
    )
    _plot(shown, "ccv_triplet_plot", render, source)
    if len(shown_modules) < f.module.nunique():
        st.caption(
            "Plot shows the first 12 selected modules; the complete filtered table/download retains every selected module. Select modules to inspect others. Corrections were calculated before display filtering."
        )
    st.caption(
        "A→B describes complete-case selection. B→C describes adjustment on identical donors. A loss of significance alone is not evidence that effects differ. Adjusted primary eligibility requires N≥30 and residual df≥20; excluded rows retain their reasons."
    )
    st.warning(
        "With 9,999 Monte Carlo draws the minimum empirical p is 0.0001. Broad FDR families can be resolution-limited; floor values and family diagnostics are retained. Non-significance at that resolution is not evidence of no association."
    )
    _table(f, "ccv_triplet_table", "complete validation results")
    if st.checkbox(
        "Show residual-rank diagnostic (association model, not native G/L units)",
        key="ccv_residual_diagnostic",
    ):
        residual_diagnostic(f, source, render)


def residual_diagnostic(frame, source, render):
    from .validation_run import prepared, cohort_mask
    from . import validation_inference as I

    rows = frame.loc[frame.model.eq("C")].reset_index(drop=True)
    if rows.empty:
        st.info("No adjusted context selected.")
        return
    index = st.selectbox(
        "Residual diagnostic context",
        list(rows.index),
        format_func=lambda i: f"{rows.loc[i,'module_key']} · {rows.loc[i,'component']} · {rows.loc[i,'metric_family']} · {rows.loc[i,'cohort']} · {rows.loc[i,'study_stratum']}",
        key="ccv_residual_context",
    )
    row = rows.loc[index]
    root = V.local_root()
    for suffix in (".npy", "_metadata.parquet", "_rows.parquet"):
        V.verified(root, "prepared/" + row.input_variant + suffix)
    matrix, meta, _ = prepared(str(root), row.input_variant)
    x = np.asarray(matrix[int(row.row_id)])
    y = I.finite_numeric(meta[row.outcome])
    use = (
        cohort_mask(meta, row.cohort)
        & I.complete_covariates(meta)
        & np.isfinite(x)
        & np.isfinite(y)
    )
    if row.study_stratum != "all":
        use &= meta.study.eq(row.study_stratum).to_numpy()
    subset = meta.loc[use]
    if I.membership_hash(subset.sample_id) != row.donor_set_hash:
        raise ValueError("Diagnostic donor membership differs from saved inference")
    d = I.nuisance_design(subset)
    if d.reason or len(subset) <= d.rank + 1:
        st.info("Residual diagnostic unavailable: " + str(row.unavailable_reason))
        return
    from scipy.stats import rankdata

    q = np.linalg.qr(d.matrix)[0]
    rx = rankdata(x[use])
    ry = rankdata(y[use])
    points = pd.DataFrame(
        {
            "residual_score_rank": rx - q @ (q.T @ rx),
            "residual_outcome_rank": ry - q @ (q.T @ ry),
        }
    )
    context = dict(
        source,
        model="C",
        module_key=row.module_key,
        component=row.component,
        metric_family=row.metric_family,
        cohort=row.cohort,
        study_stratum=row.study_stratum,
        n=len(points),
        donor_set_hash=row.donor_set_hash,
        plot_units="residualized midranks; not native connectivity units",
    )
    figure = px.scatter(
        points,
        x="residual_score_rank",
        y="residual_outcome_rank",
        title=f"{row.module_key} · {row.component} · {row.metric_family} · beta{row.beta} · {row.cohort} · N={len(points)}",
        labels={
            "residual_score_rank": "Residualized score midrank (not G/L score units)",
            "residual_outcome_rank": "Residualized outcome midrank",
        },
    )
    figure.update_layout(template="plotly_white", meta=context)
    st.caption(
        "Only the association model is residualized. These axes are residualized midranks, not new stored donor scores. No donor identifiers are included in the plot/export data."
    )
    render(figure, key="ccv_residual_plot", metadata=context)


METRICS_UI = {
    "gain": "G · gain",
    "loss": "L · loss",
    "net_change": "S · net change (secondary)",
    "magnitude_atypicality": "A · magnitude atypicality (secondary)",
}
