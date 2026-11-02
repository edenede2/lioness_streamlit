"""Selected-donor and descriptive edge analyses from frozen calibration caches."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from .registry import ROOT
from .inspection import iter_edges, ranked_edges
from .core import canonical_hash
from .inputs import atomic_json, atomic_parquet
from .analysis import matrix_test, bh


def render_inspection(view, frame, context, render, mask):
    from .ui import table, draw

    if view == "Edge summaries":
        summary = (
            frame.groupby(
                ["component", "metric_family", "diagnosis_group"], dropna=False
            )
            .metric_value.agg(["count", "mean", "std", "median"])
            .reset_index()
        )
        table(summary, "cc_edge_score_summary", "Donor connectivity summary by group")
    st.caption(
        "Edge weights are reconstructed in bounded batches from the frozen cache. No controls or preprocessing are refitted. Contributions use the selected aggregate’s actual common denominator. These annotations imply no individual-donor abnormality p value."
    )
    components = list(frame.component.unique())
    component = st.selectbox(
        "Inspect edge universe", components, key="cc_edge_component"
    )
    donor = st.selectbox(
        "Independent target donor",
        sorted(frame.sample_id.unique()),
        key="cc_edge_donor",
    )
    rank = st.selectbox(
        "Rank edges by",
        ["gain_contribution", "loss_contribution", "abs_w", "delta_abs"],
        key="cc_edge_rank",
    )
    top = st.number_input("Edges to display", 10, 1000, 50, key="cc_edge_top")
    kwargs = dict(
        module_set=context["family"],
        module=context["module"],
        estimator="calibrated_beta" + str(context["beta"]),
        component=component,
        cohort_scope=context["cohort_scope"],
        regional_tissue=context.get("regional_tissue"),
        mask=mask["rule"],
        fdr_scope=mask.get("fdr_scope", "global"),
        threshold=mask.get("threshold", 0.05),
    )
    cachekey = canonical_hash(dict(kwargs, sample_ids=[donor], rank=rank, top=int(top)))
    cachekey = canonical_hash(dict(cachekey=cachekey, score_context=context))
    if st.button("Inspect selected donor’s contributions", key="cc_inspect_edges"):
        try:
            st.session_state["cc_edge_result"] = (
                cachekey,
                ranked_edges(sample_ids=[donor], rank_by=rank, top=int(top), **kwargs),
            )
        except (ValueError, FileNotFoundError) as error:
            st.info(str(error))
    saved = st.session_state.get("cc_edge_result")
    if saved and saved[0] == cachekey:
        result = saved[1]
        table(result, "cc_edge_detail", "Calibrated edge contributions")
        if not result.empty:
            chart = result.head(25).copy()
            chart["edge"] = (
                chart.tissue_a
                + ":"
                + chart.gene_a
                + " ↔ "
                + chart.tissue_b
                + ":"
                + chart.gene_b
            )
            draw(
                px.bar(
                    chart,
                    x=rank,
                    y="edge",
                    orientation="h",
                    color="sign_class",
                    title="Selected donor · calibrated edge drivers",
                    hover_data=[
                        "a",
                        "nu",
                        "tau",
                        "calibration_n",
                        "aggregate_denominator",
                    ],
                ),
                render,
                "cc_edge_drivers",
                context,
            )
    if view in ("Gene & edge associations", "Edge volcano"):
        from app_helpers.data import (
            ASSOCIATION_OUTCOME_LABELS,
            CATEGORICAL_OUTCOME_LABELS,
        )

        volcano = view == "Edge volcano"
        outcome = (
            "diagnosis_group"
            if volcano
            else st.selectbox(
                "Edge association outcome",
                list(ASSOCIATION_OUTCOME_LABELS),
                format_func=ASSOCIATION_OUTCOME_LABELS.get,
                key="cc_edge_outcome",
            )
        )
        channel = st.selectbox(
            "Edge readout",
            ["a", "abs_w", "delta_abs", "w", "gain_contribution", "loss_contribution"],
            key="cc_edge_channel",
        )
        method = (
            "Kruskal-Wallis"
            if outcome in CATEGORICAL_OUTCOME_LABELS
            else st.radio(
                "Edge correlation method",
                ["Spearman", "Pearson"],
                horizontal=True,
                key="cc_edge_method",
            )
        )
        work = (
            frame[frame.diagnosis_group.isin(["AD", "Control"])] if volcano else frame
        )
        if volcano:
            st.caption(
                "New descriptive AD–Control comparison of calibrated edge readouts in independent B targets. MCI and other diagnoses are excluded from this contrast. Historical discovery p values and masks are not relearned or substituted."
            )
        st.caption(
            "This bounded selected-module search uses independent donors and the full selected physical-edge universe. Correction is over all estimable edges in this declared search, before ranking. It is separate from the donor-score baseline q_scan."
        )
        ids = sorted(work.sample_id.unique())
        meta = work.drop_duplicates("sample_id").set_index("sample_id")
        intent = dict(
            context,
            component=component,
            outcome=outcome,
            channel=channel,
            donor_membership_sha256=canonical_hash(ids),
            method=method,
            contrast="AD-Control" if volcano else "all selected donors",
            correction="BH over all estimable physical edges in this selected-module search; not baseline q_scan",
            minimum_numeric_n=3,
            minimum_nominal_group_n=5,
        )
        aid = canonical_hash(intent)
        dest = ROOT / "associations/saved" / aid
        if st.button(
            "Save and run descriptive edge associations", key="cc_run_edge_association"
        ):
            atomic_json(dest.with_suffix(".json"), intent)
            records = []
            buffer = []
            previous = None

            # iter_edges yields a donor vector for each bounded edge block.
            def finish(parts):
                if not parts:
                    return
                data = pd.concat(parts, ignore_index=True)
                matrix = data.pivot(
                    index="physical_edge_code", columns="sample_id", values=channel
                ).reindex(columns=ids)
                answer = matrix_test(
                    matrix.to_numpy(float),
                    meta.reindex(ids)[outcome],
                    outcome in CATEGORICAL_OUTCOME_LABELS,
                )
                codes = matrix.index.to_numpy()
                annotation = data.drop_duplicates("physical_edge_code").set_index(
                    "physical_edge_code"
                )
                if method == "Pearson":
                    from scipy.stats import pearsonr

                    y = pd.to_numeric(
                        meta.reindex(ids)[outcome], errors="coerce"
                    ).to_numpy(float)
                    for ri, values in enumerate(matrix.to_numpy(float)):
                        finite = np.isfinite(values) & np.isfinite(y)
                        if (
                            finite.sum() >= 3
                            and np.unique(values[finite]).size > 1
                            and np.unique(y[finite]).size > 1
                        ):
                            answer["effect"][ri], answer["p"][ri] = pearsonr(
                                values[finite], y[finite]
                            )
                if volcano:
                    with np.errstate(invalid="ignore"):
                        difference = np.nanmean(
                            matrix.loc[
                                :, meta.reindex(ids).diagnosis_group.eq("AD").to_numpy()
                            ].to_numpy(float),
                            axis=1,
                        ) - np.nanmean(
                            matrix.loc[
                                :,
                                meta.reindex(ids)
                                .diagnosis_group.eq("Control")
                                .to_numpy(),
                            ].to_numpy(float),
                            axis=1,
                        )
                for ri, code in enumerate(codes):
                    records.append(
                        dict(
                            physical_edge_code=int(code),
                            gene_a=annotation.loc[code, "gene_a"],
                            gene_b=annotation.loc[code, "gene_b"],
                            method=method,
                            edge_readout=channel,
                            component=component,
                            beta=context["beta"],
                            cohort_scope=context["cohort_scope"],
                            allocation_id=context["allocation_id"],
                            role="independent_target",
                            mask=context["mask"],
                            **(
                                {"mean_AD_minus_Control": difference[ri]}
                                if volcano
                                else {}
                            ),
                            **{k: v[ri] for k, v in answer.items()},
                        )
                    )

            try:
                for part in iter_edges(sample_ids=ids, **kwargs):
                    start = int(part.physical_edge_code.iloc[0]) if len(part) else None
                    if previous is not None and start != previous:
                        finish(buffer)
                        buffer = []
                    previous = start
                    buffer.append(
                        part[
                            [
                                "sample_id",
                                "physical_edge_code",
                                "gene_a",
                                "gene_b",
                                channel,
                            ]
                        ]
                    )
                finish(buffer)
                if not records:
                    st.info(
                        "No physical edges exist in the selected component and frozen mask."
                    )
                    return
                result = pd.DataFrame(records)
                result["q_search"] = bh(result.p)
                result["analysis_id"] = aid
                atomic_parquet(dest.with_suffix(".parquet"), result)
                st.session_state["cc_edge_association"] = (aid, result)
            except (ValueError, FileNotFoundError) as error:
                st.info(str(error))
        saved = st.session_state.get("cc_edge_association")
        if saved and saved[0] == aid:
            result = saved[1]
            table(
                result.sort_values("q_search").head(1000),
                "cc_edge_assoc",
                "Edge association search",
            )
            if volcano:
                plot = result[result.p.notna()].nsmallest(5000, "q_search").copy()
                plot["minus_log10_p"] = -np.log10(plot.p)
                draw(
                    px.scatter(
                        plot,
                        x="mean_AD_minus_Control",
                        y="minus_log10_p",
                        color="q_search",
                        hover_data=["gene_a", "gene_b", "n_tested", "group_counts"],
                        title="Independent-target AD–Control edge comparison",
                        labels={
                            "mean_AD_minus_Control": "Mean AD − Control · " + channel,
                            "minus_log10_p": "−log10 Kruskal–Wallis p",
                        },
                    ),
                    render,
                    "cc_calibrated_volcano",
                    context,
                )
                st.caption(
                    "Displays up to5,000 edges after correction over the complete declared search; display selection never recalculates q_search."
                )
