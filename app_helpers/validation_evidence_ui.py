"""Focused evidence in Compare; only saved aggregates/private local artifacts."""

from __future__ import annotations
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from app_helpers import validation_evidence as E
from app_helpers.analysis_context import apply_controls
from app_helpers.data import ASSOCIATION_OUTCOME_LABELS
from app_helpers.table_controls import filterable_dataframe

SECTIONS = (
    "Overview & cohorts",
    "Control structure",
    "MDC decomposition",
    "Same-edge follow-up",
    "Adjusted evidence (A/B/C)",
)


def _table(frame, key, title):
    if frame.empty:
        st.info(
            "This exact evidence table is unavailable; no other score version or cohort is substituted."
        )
        return
    filterable_dataframe(
        frame, table_key="evidence_" + key, table_name=title, hide_index=True
    )
    st.download_button(
        "Download " + title.lower(),
        frame.to_csv(sep="\t", index=False).encode(),
        file_name="validation_evidence_" + key + ".tsv",
        mime="text/tab-separated-values",
        key="evidence_download_" + key,
    )


def _layout(figure, title, height=520):
    figure.update_layout(
        template="plotly_white",
        title=dict(
            text=title, x=0, xanchor="left", y=0.99, yanchor="top", yref="container"
        ),
        height=height,
        margin=dict(t=150, b=65, l=90, r=35),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
    )
    return figure


def forest(frame, effect="effect", low="ci_low", high="ci_high", title="Association"):
    """Draw exact interval endpoints even when they exclude their point."""
    f = frame.copy()
    if "row_label" not in f:
        f["row_label"] = f.apply(
            lambda r: f"{r.get('family',r.get('partition',''))} M{r.get('module','')} · {r.get('component','')} · {r.get('metric_family',r.get('series',''))}",
            axis=1,
        )
    if "cohort" not in f:
        f["cohort"] = "estimate"
    order = list(f.row_label.drop_duplicates())
    fig = go.Figure()
    for i, (label, g) in enumerate(f.groupby("cohort", sort=False, dropna=False)):
        color = ("#167D9A", "#C48028", "#6B4C9A")[i % 3]
        xs = []
        ys = []
        for _, row in g.iterrows():
            if (
                low in row
                and high in row
                and np.isfinite(row[low])
                and np.isfinite(row[high])
            ):
                xs.extend([row[low], row[high], None])
                ys.extend([row.row_label, row.row_label, None])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        hover = [
            f"N={r.get('n','NA')}<br>p={r.get('p','NA')}<br>q_scan={r.get('q_scan','NA')}<br>q_GL_outcome={r.get('q_GL_outcome','NA')}<br>q_focused={r.get('q_focused','NA')}"
            for _, r in g.iterrows()
        ]
        fig.add_trace(
            go.Scatter(
                x=g[effect],
                y=g.row_label,
                mode="markers",
                name=str(label),
                marker=dict(
                    color=color, size=8, symbol="circle" if i == 0 else "diamond"
                ),
                text=hover,
                hovertemplate="%{y}<br>Estimate=%{x:.4f}<br>%{text}<extra>%{fullData.name}</extra>",
            )
        )
    fig.add_vline(x=0, line_color="#777777", line_dash="dot")
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=order,
        autorange="reversed",
        tickmode="array",
        tickvals=order,
        ticktext=order,
    )
    extent = pd.concat(
        [
            pd.to_numeric(f[column], errors="coerce")
            for column in (effect, low, high)
            if column in f
        ]
    ).to_numpy()
    extent = extent[np.isfinite(extent)]
    if len(extent):
        bottom, top = min(0.0, float(extent.min())), max(0.0, float(extent.max()))
        pad = max((top - bottom) * 0.06, 0.02)
        fig.update_xaxes(range=[bottom - pad, top + pad])
    return _layout(fig, title, max(450, len(order) * 28 + 210))


def _applied(section):
    with st.container(border=True):
        a, b, c = st.columns(3)
        family = a.selectbox(
            "Evidence partition", list(E.FOCUS), key="evidence_partition"
        )
        options = E.FOCUS[family]
        if st.session_state.get("evidence_module") not in options:
            st.session_state.pop("evidence_module", None)
        module = b.selectbox(
            "Evidence module",
            options,
            format_func=lambda m: f"M{m}",
            key="evidence_module",
        )
        component = c.selectbox(
            "Physical component",
            E.COMPONENTS,
            format_func=E.COMPONENT_LABELS.get,
            key="evidence_component",
        )
        beta = 3
        outcomes = ["cogn_global", "gpath"]
        if section == "Adjusted evidence (A/B/C)":
            outcomes = [
                "cogn_global",
                "cogng_demog_slope",
                "cogng_path_slope",
                "motor10_demog_slope",
                "sqrt_parksc_demog_slope",
                "gpath",
                "amyloid",
                "tangles",
            ]
            beta = st.selectbox("Frozen donor-score beta", [3, 1], key="evidence_beta")
        outcome = (
            st.selectbox(
                "Evidence outcome",
                outcomes,
                format_func=ASSOCIATION_OUTCOME_LABELS.get,
                key="evidence_outcome_" + section,
            )
            if section in ("Same-edge follow-up", "Adjusted evidence (A/B/C)")
            else "cogn_global"
        )
        applied = apply_controls(
            dict(
                module_set=family,
                module=module,
                components=(component,),
                phenotype=outcome,
                estimator=f"calibrated_beta{beta}",
                method="control_anchored",
                cohort_scope="fixed_disjoint_evidence",
                source=E.VERSION,
            ),
            view="evidence:" + section,
        )
    family, module, component, outcome = (
        applied["module_set"],
        applied["module"],
        applied["components"][0],
        applied["phenotype"],
    )
    beta = int(applied["estimator"].removeprefix("calibrated_beta"))
    st.caption(
        f"Applied: {family}::M{module} · {E.COMPONENT_LABELS[component]} · all frozen edges · {E.VERSION}"
    )
    with st.expander("Cohort, score and inference context"):
        score_view = section in ("Same-edge follow-up", "Adjusted evidence (A/B/C)")
        protocol = dict(
            partition=family,
            module=module,
            physical_component=component,
            mask="all frozen eligible edges",
            cohorts="Original analysis complete-expression donors versus incomplete-region donors; associated discovery set has 464 complete-expression donors",
            donor_roles="Independent target-only original/incomplete for calibrated scores; all original/validation Controls for group structure; R82/C41 never counted as independent score targets",
            donor_score_beta=(
                beta if score_view else "not applicable to a group network"
            ),
            group_adjacency_power=(
                "TS=3 / CT=2"
                if not score_view
                else "not applicable to donor-level effect test; see group MDC separately"
            ),
            estimator=(
                "Frozen R82 reference, C41 magnitude calibration"
                if score_view
                else "Group Pearson signedAlt adjacency, frozen unique tissue-gene membership"
            ),
            adjustment=(
                "Age/sex/education/study partial midranks for clinical C models; A/B and AD–Control Hedges are unadjusted"
                if score_view
                else "No covariate regression on a group adjacency"
            ),
            uncertainty="Donor bootstrap; structure uses stability ranges, MDC uses percentile intervals; no edge-as-replicate tests",
            inference=(
                "Saved Freedman–Lane / HC3 in adjusted report; selected raw-edge diagnosis uses Welch"
                if score_view
                else "No new permutation p or q; historical sample/gene nulls remain separate"
            ),
            correction=(
                "Original full-search / frozen selected-edge family, or exact adjusted-report q_scan/q_GL_outcome/q_focused; never recomputed after filters"
                if score_view
                else "No new structure/MDC q; saved full-module dual-null q only"
            ),
            artifact=E.VERSION,
        )
        st.json(protocol)
        counts = E.table("cohorts")
        if not counts.empty:
            counts = counts.loc[
                counts.dataset.eq("brain")
                & counts.component.eq(component)
                & counts.population.eq(
                    "independent targets" if score_view else "all applicable roles"
                )
            ]
            _table(
                counts,
                "selected_context_" + section,
                "Component-specific donor and observed-outcome counts",
            )
    return family, module, component, outcome, beta


def _subset(frame, family, module, component):
    f = frame
    for column, value in (
        ("partition", family),
        ("family", family),
        ("module", module),
        ("component", component),
    ):
        if column in f:
            f = f.loc[f[column].eq(value)]
    return f.copy()


def _overview(render):
    st.caption(
        "Retrospective, conditional within-ROSMAP evidence. Technical reproduction, independent donors, structural preservation and association support are different questions."
    )
    rows = [
        dict(
            question="Original structure in new Controls?",
            view="Control structure",
            scope="Four fixed module namespaces; DLPFC and DLPFC–PCG; donor-resampled group networks",
        ),
        dict(
            question="What changed in MDC?",
            view="MDC decomposition",
            scope="Full saved catalog overview + focused AD numerator/Control denominator; historical overlap audited",
        ),
        dict(
            question="Do original edges retain signal?",
            view="Same-edge follow-up",
            scope="Frozen original-only candidates; selected-family inference; detailed reconstruction local-only",
        ),
        dict(
            question="Does adjustment change the association?",
            view="Adjusted evidence (A/B/C)",
            scope="Exact already-public Nov-04 report; all eight endpoints, beta3/beta1; no legacy substitution",
        ),
    ]
    _table(pd.DataFrame(rows), "guide", "Evidence guide")
    links = st.columns(2)
    for i, view in enumerate(SECTIONS[1:]):
        links[i % 2].button(
            "Open " + view,
            on_click=_go_evidence,
            args=(view,),
            key="evidence_link_" + str(i),
        )
    st.subheader("Cohort identities and observed donor counts")
    cohorts = E.table("cohorts")
    if not cohorts.empty:
        coverage = cohorts.loc[
            cohorts.population.eq("all applicable roles")
            & cohorts.dataset.eq("brain")
            & cohorts.component.isin(E.COMPONENTS)
        ]
        coverage = coverage.groupby(
            ["component", "cohort", "diagnosis"], as_index=False, observed=True
        )[["unique_donors", "cogn_global_n", "gpath_n"]].sum()
        fig = go.Figure()
        for diagnosis, g in coverage.groupby("diagnosis", sort=False):
            fig.add_trace(
                go.Bar(
                    x=g.component + " · " + g.cohort,
                    y=g.unique_donors,
                    name=diagnosis,
                    marker_color={
                        "Control": "#377EB8",
                        "MCI": "#DDAA33",
                        "AD": "#EE7733",
                        "Unclassified": "#999999",
                    }.get(diagnosis, "#888888"),
                    customdata=g[["cogn_global_n", "gpath_n"]].to_numpy(),
                    hovertemplate="%{x}<br>Donors=%{y}<br>Cognition N=%{customdata[0]}<br>gpath N=%{customdata[1]}<extra>%{fullData.name}</extra>",
                )
            )
        fig.update_layout(barmode="stack")
        render(
            _layout(fig, "Distinct donors by cohort, physical component and role"),
            key="validation_evidence_cohorts",
        )
    _table(cohorts, "cohorts", "Audited cohort counts")
    st.warning(
        "Legacy Edge volcano “Validation” is the original complete-donor AD/Control held-out split, not the incomplete-region validation cohort. R=reference, C=calibration, T=independent target; T is not the legacy 70/30 test split."
    )
    intersections = E.table("intersections")
    _table(intersections, "intersections", "Membership intersections")
    st.caption(
        "Zero overlap is shown only where inclusion records were compared. Missing historical membership means independence not verified. Component cohorts overlap with each other; do not sum them."
    )
    st.caption(
        "The associated full-cohort discovery eigengene artifact contains 464 complete-expression donors; the historical score-analysis subset contains 450. All 464 are excluded from the incomplete cohort. Fourteen other complete targets are listed separately, not silently added to either validation group."
    )
    st.subheader("Saved MDC catalog: full tested denominators")
    counts = E.table("mdc_overview")
    if not counts.empty:
        source = st.selectbox(
            "MDC overview partition",
            list(counts.partition.drop_duplicates()),
            key="evidence_mdc_overview_partition",
        )
        display = counts.loc[counts.partition.eq(source) & counts["mask"].eq("all")]
        _table(display, "mdc_overview", "All-edge MDC evidence counts")
        with st.expander("Every frozen mask and correction context"):
            _table(
                counts.loc[counts.partition.eq(source)],
                "mdc_masks",
                "Complete MDC evidence counts",
            )
    st.caption(
        "“Original-significant evaluable” is the original q<0.05 subset with finite positive ratios in both cohorts. Nominal/adjusted support additionally requires the same log2(MDC) sign and the saved validation dual-null p/q<0.05. Counts are module–component hypotheses, not independent biological replicates. Unavailable and descriptive-only rows remain counted."
    )
    a, b = st.columns(2)
    a.button(
        "Open legacy association replication",
        on_click=_go_detail,
        args=("Association replication",),
        key="evidence_link_association",
    )
    b.button(
        "Open legacy MDC replication",
        on_click=_go_detail,
        args=("MDC replication",),
        key="evidence_link_mdc",
    )
    if st.button("Open matched-control comparisons", key="evidence_link_matched"):
        st.query_params["view"] = "Matched-control results"
        st.rerun()
    with st.expander("Availability, source scope and sampled numerical checks"):
        _table(
            E.table("availability"),
            "availability",
            "Analysis availability and numerical checks",
        )
    with st.expander("Meeting figures and results note"):
        st.caption(
            "Saved aggregate-only PNG/PDF figures; individual charts also retain the normal figure exporter."
        )
        for name in E.manifest().get("artifacts", {}):
            data = E.public_artifact(name)
            st.download_button(
                "Download " + Path(name).name,
                data,
                file_name=Path(name).name,
                key="evidence_meeting_" + name,
            )


def _go_detail(section):
    st.session_state["validation_review_section"] = section
    st.session_state["validation_calibrated_source"] = "legacy"


def _go_evidence(section):
    st.session_state["validation_evidence_view"] = section


def _local_network_file(family, module, component, suffix):
    name = f"{family}__M{module}__{component}.{suffix}"
    receipt = E.PRIVATE / "manifest.json"
    if not receipt.is_file():
        return None
    m = json.loads(receipt.read_text())
    expected = m.get("local_artifacts", {}).get(name)
    path = E.PRIVATE / name
    if not expected or not path.is_file():
        return None
    if E.sha(path) != expected:
        raise ValueError("Local network artifact checksum mismatch")
    return path


def _structure(render, family, module, component):
    f = _subset(E.table("structure"), family, module, component)
    if f.empty:
        st.info("Focused component has not been computed.")
        return
    _table(f, "structure", "Control-only structure preservation")
    row = f.iloc[0]
    st.caption(
        f"Original Controls N={row.original_control_n}; incomplete Controls N={row.validation_control_n}. Original reference/calibration/target Controls: {row.original_reference_controls}/{row.original_calibration_controls}/{row.original_target_controls}. Original Controls participated in historical discovery; this is conditional preservation of user-selected modules, not a formal WGCNA Zsummary test."
    )
    values = []
    for metric, label in [
        ("signed_r_concordance", "Signed correlation concordance"),
        ("signed_adjacency_concordance", "Signed powered-adjacency concordance"),
        ("magnitude_concordance", "Magnitude concordance"),
        ("hub_DLPFC", "DLPFC absolute node-connectivity concordance"),
        ("hub_PCG", "PCG absolute node-connectivity concordance"),
    ]:
        if metric in row and np.isfinite(row[metric]):
            values.append(
                dict(
                    row_label=label,
                    effect=row[metric],
                    ci_low=row.get(metric + "_resampling_low"),
                    ci_high=row.get(metric + "_resampling_high"),
                    cohort="Observed; interval=donor-resampling stability range",
                )
            )
    if values:
        render(
            forest(
                pd.DataFrame(values),
                title="Controls-only structure: corresponding fixed edges/nodes",
            ),
            key="validation_evidence_structure_concordance",
        )
    st.warning(
        "These concordance intervals are 95% donor-resampling stability ranges, not calibrated population-correlation confidence intervals. Rebuilding noisy networks can attenuate the resampling distribution below the observed point. No edge-as-replicate p-value is reported."
    )
    st.caption(
        f"Equal-Control-N sensitivity uses {int(row.sample_matched_control_n)} donors per cohort without replacement; its median and stability ranges are retained in the table. TS power=3, CT power=2. This group adjacency power is not the donor-score beta."
    )
    path = _local_network_file(family, module, component, "npz")
    if path is None:
        st.info(
            "Detailed paired network heatmaps and edge scatter remain local-only. Their aggregate density/concordance estimates are available above; no private matrix was published."
        )
        return
    with np.load(path, allow_pickle=False) as arrays:
        r, c, t = arrays["rows"], arrays["columns"], arrays["tissues"]
        values = [
            arrays[k] for k in ("original_complete_Control", "incomplete_Control")
        ]
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Original Controls",
            "Incomplete-region Controls",
            "Corresponding edge magnitudes",
        ],
    )
    maximum = max(float(np.max(abs(v))) for v in values)
    for col, v in enumerate(values, 1):
        matrix = np.full((len(t), len(t)), np.nan)
        matrix[r, c] = abs(v)
        matrix[c, r] = abs(v)
        if component.startswith("CT_"):
            matrix = matrix[np.ix_(t == "DLPFC", t == "PCG")]
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                coloraxis="coloraxis",
                hovertemplate="Node row=%{y}<br>Node column=%{x}<br>|Adjacency|=%{z:.5f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
    fig.add_trace(
        go.Scattergl(
            x=abs(values[0]),
            y=abs(values[1]),
            mode="markers",
            marker=dict(size=2, opacity=0.2),
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, maximum],
            y=[0, maximum],
            mode="lines",
            line=dict(color="black", dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.update_layout(
        coloraxis=dict(
            colorscale="YlOrRd",
            cmin=0,
            cmax=maximum,
            colorbar=dict(title="|Adjacency|", x=1.03),
        ),
        showlegend=False,
    )
    render(
        _layout(
            fig,
            f"{family} M{module} · {E.COMPONENT_LABELS[component]} · same node order and scale",
        ),
        key="validation_evidence_paired_networks",
    )
    for suffix in ("png", "pdf"):
        export = _local_network_file(family, module, component, suffix)
        if export:
            st.download_button(
                "Download meeting network figure (" + suffix.upper() + ")",
                export.read_bytes(),
                export.name,
                key="evidence_local_" + suffix,
            )


def _mdc(render, family, module, component):
    f = _subset(E.table("mdc_decomposition"), family, module, component)
    if f.empty:
        st.info("Exact MDC decomposition is unavailable.")
        return
    row = f.iloc[0]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["AD and Control mean absolute adjacency", "MDC = AD / Control"],
    )
    for cohort, label, color in [
        ("original_complete", "Disjoint original complete-expression", "#167D9A"),
        ("incomplete", "Incomplete-region", "#C48028"),
    ]:
        for diagnosis in ("ad", "control"):
            key = cohort + "_mean_abs_" + diagnosis
            x = label + " · " + diagnosis.upper()
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[row[key]],
                    mode="markers",
                    marker=dict(color=color, size=10),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[x, x],
                    y=[row[key + "_ci_low"], row[key + "_ci_high"]],
                    mode="lines",
                    line=dict(color=color),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=[label],
                y=[row[cohort + "_mdc"]],
                mode="markers",
                marker=dict(color=color, size=10),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[label, label],
                y=[row[cohort + "_mdc_ci_low"], row[cohort + "_mdc_ci_high"]],
                mode="lines",
                line=dict(color=color),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.add_hline(y=1, line_dash="dash", line_color="black", row=1, col=2)
    render(
        _layout(
            fig, f"{family} M{module}: separate AD numerator and Control denominator"
        ),
        key="validation_evidence_mdc_decomposition",
    )
    st.caption(
        f"Δlog2(MDC)={row.delta_log2_mdc:.3f} = AD log2-density change {row.log2_ad_density_change:.3f} − Control log2-density change {row.log2_control_density_change:.3f}. These between-cohort changes are descriptive, not causal attribution."
    )
    st.caption(
        "Intervals are 95% study-stratified donor-bootstrap estimates (199 draws), rebuilding numerator and denominator together. They are not permutation p-values or a new FDR test. No epsilon is added to denominators; undefined ratios remain unavailable."
    )
    _table(f, "mdc_decomposition", "Focused MDC decomposition and uncertainty")
    historic = E.table("historical_mdc")
    historic = historic.loc[
        historic.partition.eq(family)
        & historic.module.eq(module)
        & historic.component.map(E.canonical_component).eq(component)
    ]
    st.subheader("Historical MDC — preserved, separate evidence")
    st.warning(
        "Historical tissue-union networks can include incomplete donors. They are not the disjoint comparison plotted above. Full-cohort saved-network membership is not archived here: independence not verified. The original saved p/q and dual-null meaning are unchanged."
    )
    _table(historic, "historical_mdc", "Saved historical MDC and dual-null inference")


def _gene_symbols(frame):
    from app_helpers.gene_symbols import gene_symbol

    f = frame.copy()
    for column in ("gene_a", "gene_b"):
        if column in f:
            f[column] = (
                f[column]
                .map(gene_symbol)
                .map(lambda s: "Unmapped gene" if str(s).startswith("ENSG") else s)
            )
    return f


def _same_edges(render, family, module, component, outcome):
    public = _subset(E.table("edge_summary"), family, module, component)
    _table(public, "edge_summary", "Frozen same-edge follow-up summary")
    st.caption(
        "Frozen union: top 20 original-target |Hedges g| plus separate top 20 original-target adjusted |ρ| lists for global cognition and gpath. Modules and ranks were selected retrospectively. The same R82 reference/C41 calibration and beta3 raw edge readout are used for original and incomplete independent targets. R/C donors are never validation targets."
    )
    st.caption(
        "AD–Control Hedges contrasts are unadjusted. Clinical correlations adjust age, sex, education and ROS/MAP using the existing rank/complete-case contract; these are not causal group effects."
    )
    local = E.private_table("same_edges")
    if local.empty:
        st.info(
            "Exact candidate identities, edge effects and contributions remain local-only. Hosted results show permitted aggregate counts; there is no reconstruction or donor-array download on Streamlit Cloud."
        )
        return
    f = _gene_symbols(_subset(local, family, module, component))
    meaning = st.radio(
        "Edge evidence meaning",
        [
            "AD–Control raw-edge difference",
            "Edge–outcome association",
            "Calibrated magnitude contribution",
        ],
        horizontal=True,
        key="evidence_edge_meaning",
    )
    if meaning == "Calibrated magnitude contribution":
        contribution = _subset(
            E.private_table("edge_contributions"), family, module, component
        )
        identities = _gene_symbols(
            _subset(E.private_table("edge_candidates"), family, module, component)
        )
        contribution = contribution.merge(
            identities[
                [
                    "partition",
                    "module",
                    "component",
                    "edge_code",
                    "tissue_a",
                    "gene_a",
                    "tissue_b",
                    "gene_b",
                ]
            ],
            on=["partition", "module", "component", "edge_code"],
            how="left",
            validate="many_to_one",
        )
        st.caption(
            "a=(|w|−ν)/τ. Gain=max(a,0), loss=max(−a,0), each divided by the full frozen eligible component-edge denominator—not the displayed top-edge count. Strengthening magnitude is not an increase in signed correlation and is not a clinical effect size."
        )
        _table(
            contribution, "edge_contributions", "Local frozen-edge score contributions"
        )
        return
    f = f.loc[
        (
            f.test.eq("AD_Control")
            if meaning.startswith("AD")
            else f.test.eq("edge_outcome") & f.outcome.eq(outcome)
        )
    ]
    rank = st.selectbox(
        "Frozen original rank list",
        [
            "All frozen candidates",
            "AD_Control_raw_w",
            "cognition_raw_w",
            "pathology_raw_w",
        ],
        key="evidence_edge_rank_list",
    )
    if rank != "All frozen candidates":
        f = f.loc[f.selected_by.str.contains(rank, regex=False)]
    if f.empty:
        st.info("No exact fixed candidates match this list.")
        return
    f["row_label"] = f.tissue_a + ":" + f.gene_a + " — " + f.tissue_b + ":" + f.gene_b
    plots = []
    for _, row in f.iterrows():
        plots.extend(
            [
                dict(
                    row_label=row.row_label,
                    effect=row.original_effect,
                    ci_low=row.original_ci_low,
                    ci_high=row.original_ci_high,
                    cohort="Original independent targets",
                    n=row.original_n,
                    p=row.original_p,
                ),
                dict(
                    row_label=row.row_label,
                    effect=row.effect,
                    ci_low=row.ci_low,
                    ci_high=row.ci_high,
                    cohort="Incomplete independent targets",
                    n=row.n,
                    p=row.p,
                ),
            ]
        )
    render(
        forest(pd.DataFrame(plots), title="Fixed original-ranked edges · " + meaning),
        key="validation_evidence_same_edge_forest",
    )
    st.caption(
        "Original_search_q is BH across all valid original edges of this fixed module/component/endpoint, before top-20 selection. Selected_family_q covers the complete frozen union across all four namespaces/two components, separately by endpoint/meaning, before display filters. These are different families; neither is recalculated here. Correlated edges are not independent biological replications."
    )
    _table(f, "same_edges", "Local same-edge original and validation effects")


def _adjusted(render, family, module, component, outcome, beta):
    st.caption(
        "Read-only adapter to the exact already-public report control_calibrated_validation_20261104. Coverage: eight endpoints, all listed fixed candidates, native unmasked G/L/S/A, both betas. No older baseline, outcome panel or score release is substituted."
    )
    view = st.radio(
        "Adjusted evidence view",
        [
            "A/B/C and beta sensitivity",
            "ROS/MAP and interaction",
            "Incremental value",
            "Selection versus adjustment",
        ],
        horizontal=True,
        key="evidence_adjusted_view",
    )
    if view == "Incremental value":
        name = "incremental"
    elif view == "Selection versus adjustment":
        name = "focused_selection_adjustment"
    else:
        name = "focused_all_endpoints"
    full = E.report_table(name)
    f = _subset(full, family, module, component)
    if "outcome" in f:
        f = f.loc[f.outcome.eq(outcome)]
    if f.empty:
        st.info(
            "This exact adjusted-report table/selection is unavailable. No legacy bundle, other outcome panel or score version is substituted."
        )
        return
    if "beta" in f and view != "A/B/C and beta sensitivity":
        f = f.loc[f.beta.eq(beta)]
    if view == "A/B/C and beta sensitivity":
        f = f.loc[
            f.metric_family.isin(["gain", "loss"])
            & f.study_stratum.eq("all")
            & f.model.isin(["A", "B", "C", "C_age_spline"])
        ]
        f["row_label"] = f.apply(
            lambda r: f"{r.metric_family} · beta{r.beta} · model {r.model}", axis=1
        )
        render(
            forest(f, title="A/B/C: all-available → same-donor unadjusted → adjusted"),
            key="validation_evidence_ABC",
        )
        st.caption(
            "B and C use identical complete-case donors. C adjusts score and outcome midranks for age, sex, education and ROS/MAP; the age-spline sensitivity has its own focused family. q_scan/q_GL_outcome/q_focused retain the original report meanings. Score beta1/beta3 is distinct from group-network TS3/CT2 power."
        )
    elif view == "ROS/MAP and interaction":
        f = f.loc[
            f.metric_family.isin(["gain", "loss"])
            & f.model.eq("C")
            & f.study_stratum.isin(["ROS", "MAP"])
        ]
        f["row_label"] = f.metric_family + " · " + f.study_stratum
        render(
            forest(f, title="Within-ROS and within-MAP adjusted estimates"),
            key="validation_evidence_ROS_MAP",
        )
        interaction = _subset(E.report_table("interaction"), family, module, component)
        interaction = interaction.loc[
            interaction.outcome.eq(outcome) & interaction.beta.eq(beta)
        ]
        _table(interaction, "interaction", "Saved study-interaction tests")
        st.caption(
            "Different subgroup significance is not an interaction test. The saved interaction uses pooled standardized ranks with score×study, Freedman–Lane and separate HC3 sensitivity."
        )
    elif view == "Incremental value":
        st.warning(
            "These are in-sample explanatory ΔR² estimates, not held-out predictive improvement. Connectivity/PC1 fits can overlap original targets. Incomplete-cohort rows with no_shared_comparator_donors remain unavailable; no PCA is refitted or comparator substituted."
        )
        f["row_label"] = "G/L beyond historical connectivity + constituent PC1"
        render(
            forest(f, title="Incremental explanatory rank-outcome R²"),
            key="validation_evidence_incremental",
        )
    _table(f, "adjusted_" + name, "Saved adjusted evidence")


def render_view(render):
    slugs = dict(
        zip(
            (
                "overview",
                "control_structure",
                "mdc_decomposition",
                "same_edges",
                "adjusted",
            ),
            SECTIONS,
        )
    )
    requested = st.query_params.get("evidence_section")
    if requested in slugs and st.session_state.get("_last_evidence_link") != requested:
        st.session_state["_last_evidence_link"] = requested
        st.session_state["validation_evidence_view"] = slugs[requested]
    section = st.selectbox("Evidence view", SECTIONS, key="validation_evidence_view")
    m = E.manifest()
    renderer = render

    def render(figure, **kwargs):
        context = st.session_state.get("analysis_context:evidence:" + section, {})
        provenance = dict(
            evidence_version=E.VERSION,
            source_score_release=m.get("source_score_release"),
            source_adjusted_analysis_id=m.get("source_adjusted_analysis_id"),
            applied_context=dict(context),
        )
        if context:
            component = context["components"][0]
            prefix = f"{context['module_set']} M{context['module']} · {E.COMPONENT_LABELS[component]}"
            if section in ("Same-edge follow-up", "Adjusted evidence (A/B/C)"):
                prefix += " · " + ASSOCIATION_OUTCOME_LABELS.get(
                    context["phenotype"], context["phenotype"]
                )
            figure.update_layout(
                title_text=prefix
                + "<br><sup>"
                + str(figure.layout.title.text or "")
                + "</sup>"
            )
            provenance.update(
                module=context["module"],
                module_definition=context["module_set"],
                component=component,
            )
        figure.update_layout(meta={**dict(figure.layout.meta or {}), **provenance})
        return renderer(figure, metadata=provenance, **kwargs)

    st.caption("Evidence version: " + m["version"] + " · " + m["status"])
    if m["status"] == "unavailable":
        st.info(m["reason"])
        return
    if section == "Overview & cohorts":
        _overview(render)
    else:
        family, module, component, outcome, beta = _applied(section)
        if section == "Control structure":
            _structure(render, family, module, component)
        elif section == "MDC decomposition":
            _mdc(render, family, module, component)
        elif section == "Same-edge follow-up":
            _same_edges(render, family, module, component, outcome)
        else:
            _adjusted(render, family, module, component, outcome, beta)
    with st.expander("Artifact provenance, definitions and availability"):
        st.json(m)
