"""Metadata-only donor exploration, independent of module and score selectors."""
from __future__ import annotations

import textwrap
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app_helpers import donor_explorer as D
from app_helpers.data import ASSOCIATION_OUTCOME_LABELS, NUMERIC_OUTCOMES, SELECTABLE_ASSOCIATION_OUTCOMES
from app_helpers.table_controls import filterable_dataframe

LABELS = {**ASSOCIATION_OUTCOME_LABELS, "sample_id": "Public sample ID", "available_tissues": "Available regions / tissues",
          "n_available_tissues": "Number of available regions / tissues", "in_original_analysis": "In original analysis cohort",
          "metadata_namespace": "Dataset namespace", "metadata_snapshot": "Metadata snapshot", "ad_control_split": "Saved AD–Control split label"}
CONTINUOUS = [c for c in NUMERIC_OUTCOMES if c not in SELECTABLE_ASSOCIATION_OUTCOMES]
PALETTE = ["#377EB8", "#DDAA33", "#EE7733", "#999999", "#009E73", "#CC79A7", "#56B4E9", "#332288"]
DIAGNOSIS_COLORS = {"Control": "#377EB8", "MCI": "#DDAA33", "AD": "#EE7733", "Unclassified": "#999999", D.MISSING: "#CCCCCC"}


@st.cache_data(max_entries=2, ttl=1800, show_spinner=False)
def cached_catalog(dataset, revision):
    return D.load_catalog(dataset)


def _label(field):
    return LABELS.get(field, field.replace("has_", "Has ").replace("_", " "))


def _plot(figure, render, key):
    figure.update_layout(template="plotly_white", height=figure.layout.height or 500, margin=dict(t=90, l=65, r=30, b=70),
                         legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0, xanchor="left"))
    render(figure, key=key)


def _table(frame, key, title):
    filterable_dataframe(frame, table_key=key, table_name=title, hide_index=True)
    st.download_button("Download " + title.lower(), frame.to_csv(sep="\t", index=False).encode(),
                       key + ".tsv", mime="text/tab-separated-values", key=key + "_download")


def _filters(cohort, context):
    categories = {}; ranges = {}
    with st.expander("Filter donor properties"):
        selected = st.multiselect("Categorical donor filters", [c for c in D.CATEGORICAL_FIELDS if c in cohort],
                                 format_func=_label, default=["diagnosis_group", "study"], key=context + "_category_fields")
        for c in selected:
            levels = sorted(D.category_values(cohort, c).unique().tolist(), key=lambda x: (x == D.MISSING, str(x)))
            categories[c] = st.multiselect(_label(c) + " values", levels, default=levels, key=context + "_levels_" + c)
        continuous = st.multiselect("Numeric donor filters", [c for c in CONTINUOUS if c in cohort],
                                   format_func=_label, key=context + "_range_fields")
        for c in continuous:
            values = pd.to_numeric(cohort[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                st.caption(_label(c) + ": no observed values in this cohort.")
                continue
            low, high = float(values.min()), float(values.max())
            if low != high:
                bounds = st.slider(_label(c) + " range", low, high, (low, high), step=(high-low)/200,
                                   format="%.5f", key=context + "_range_" + c)
            else:
                bounds = (low, high)
                st.caption(_label(c) + f": constant ({low:g}).")
            keep_missing = st.checkbox("Keep missing " + _label(c), value=True, key=context + "_keep_na_" + c)
            ranges[c] = (*bounds, keep_missing)
        observed = st.multiselect("Require observed values for", [c for c in ASSOCIATION_OUTCOME_LABELS if c in cohort],
                                  format_func=_label, key=context + "_required")
        st.caption("Filters apply to all summaries, plots, details and downloads on this page. Missing outcomes are never imputed. Clearing a category's levels selects no donors.")
    return D.filter_donors(cohort, categories, ranges, observed)


def _overview(frame, tissues, render):
    left, right = st.columns(2)
    with left:
        field = st.selectbox("Summarize donors by", [c for c in D.CATEGORICAL_FIELDS if c in frame],
                             format_func=_label, key="donor_explorer_breakdown")
        counts = D.category_values(frame, field).value_counts().rename_axis("Category").reset_index(name="Donors")
        counts["Percent"] = 100 * counts.Donors / len(frame)
        figure = px.bar(counts, x="Category", y="Donors", text="Donors", labels={"Category": _label(field)},
                        color="Category", color_discrete_map=DIAGNOSIS_COLORS if field == "diagnosis_group" else None,
                        color_discrete_sequence=PALETTE)
        figure.update_layout(showlegend=False)
        _plot(figure, render, "donor_explorer_property_counts")
        _table(counts, "donor_explorer_category_summary", "Donor category summary")
    with right:
        patterns = frame.assign(Diagnosis=D.category_values(frame, "diagnosis_group")).groupby(
            ["available_tissues", "Diagnosis"], observed=True).size().reset_index(name="Donors")
        order = patterns.groupby("available_tissues").Donors.sum().sort_values(ascending=False).index.tolist()
        figure = px.bar(patterns, x="available_tissues", y="Donors", color="Diagnosis", barmode="stack",
                        category_orders={"available_tissues": order, "Diagnosis": ["Control", "MCI", "AD", "Unclassified", D.MISSING]}, color_discrete_map=DIAGNOSIS_COLORS,
                        labels={"available_tissues": "Exact coverage pattern"})
        _plot(figure, render, "donor_explorer_coverage_patterns")
        st.caption("Exact coverage patterns are mutually exclusive: each selected donor is counted once in this chart.")
    st.subheader("Region / pair coverage by diagnosis")
    coverage = D.coverage_summary(frame, tissues)
    st.caption("The following region/pair rows overlap: a donor with all three regions appears in every applicable row. Do not sum their totals.")
    _table(coverage, "donor_explorer_coverage_counts", "Region and pair counts")


def _phenotypes(frame, context, render):
    fields = [c for c in CONTINUOUS if c in frame]
    a, b, c = st.columns(3)
    outcome = a.selectbox("Donor phenotype", fields, index=fields.index("cogn_global") if "cogn_global" in fields else 0,
                          format_func=_label, key=context + "_phenotype")
    grouping = b.selectbox("Group donor phenotypes by", ["__all__", *[f for f in D.CATEGORICAL_FIELDS if f in frame]],
                           index=1, format_func=lambda x: "All selected donors" if x == "__all__" else _label(x), key=context + "_phenotype_group")
    view = c.selectbox("Donor phenotype view", ["Box + donor points", "Histogram", "ECDF", "Two-phenotype scatter"], key=context + "_phenotype_view")
    values = pd.to_numeric(frame[outcome], errors="coerce").replace([np.inf, -np.inf], np.nan)
    work = frame.copy()
    work[outcome] = values
    work["Group"] = "All selected donors" if grouping == "__all__" else D.category_values(work, grouping)
    work = work.dropna(subset=[outcome])
    st.caption(f'{len(work):,}/{len(frame):,} selected donors have an observed {_label(outcome)}. Values use the supplied metadata scale; amyloid and tangles are not square-root transformed again.')
    if work.empty:
        st.info("No observed phenotype values in this selection.")
        return
    palette = dict(color_discrete_map=DIAGNOSIS_COLORS) if grouping == "diagnosis_group" else dict(color_discrete_sequence=PALETTE)
    if view == "Box + donor points":
        fig = px.box(work, x="Group", y=outcome, color="Group", points="all", hover_data=["sample_id", "metadata_snapshot"], labels={outcome: _label(outcome)}, **palette)
    elif view == "Histogram":
        fig = px.histogram(work, x=outcome, color="Group", barmode="overlay", opacity=.6,
                           labels={outcome: _label(outcome)}, **palette)
        low, high = float(work[outcome].min()), float(work[outcome].max())
        width = (high - low) / 25 if high > low else 1.
        fig.update_traces(xbins=dict(start=low - width/2, end=high + width/2, size=width), autobinx=False)
    elif view == "ECDF":
        fig = px.ecdf(work, x=outcome, color="Group", labels={outcome: _label(outcome)}, **palette)
    else:
        other = st.selectbox("Second donor phenotype", [f for f in fields if f != outcome], format_func=_label, key=context + "_second_phenotype")
        work[other] = pd.to_numeric(work[other], errors="coerce").replace([np.inf, -np.inf], np.nan)
        work = work.dropna(subset=[other])
        st.caption(f"The scatter includes {len(work):,} donors observed for both phenotypes. It is descriptive; no new association/FDR family is calculated here.")
        if work.empty:
            st.info("No donors have both selected phenotype values.")
            return
        fig = px.scatter(work, x=outcome, y=other, color="Group", hover_data=["sample_id", "metadata_snapshot"], labels={outcome: _label(outcome), other: _label(other)}, **palette)
    _plot(fig, render, "donor_explorer_phenotype_plot")
    summary = work.groupby("Group", observed=True)[outcome].agg(n="count", mean="mean", sd="std", median="median", minimum="min", maximum="max")
    summary["Q1"] = work.groupby("Group", observed=True)[outcome].quantile(.25)
    summary["Q3"] = work.groupby("Group", observed=True)[outcome].quantile(.75)
    _table(summary.reset_index(), "donor_explorer_phenotype_summary", "Phenotype summary")


def _missingness(frame, context, render):
    fields = [c for c in ASSOCIATION_OUTCOME_LABELS if c in frame]
    summary = D.missingness(frame, fields).sort_values("missing_percent", ascending=False)
    plotted = summary.assign(label_wrapped=summary.label.map(lambda value: "<br>".join(textwrap.wrap(value, width=35))))
    figure = px.bar(plotted, x="missing_percent", y="label_wrapped", orientation="h", hover_data={"label": True, "label_wrapped": False, "observed": True, "missing": True, "donors": True},
                    labels={"missing_percent": "Missing (%)", "label_wrapped": "Metadata field"})
    figure.update_yaxes(autorange="reversed")
    figure.update_xaxes(range=[0, 100])
    figure.update_layout(height=max(600, 38 * len(summary) + 180))
    _plot(figure, render, "donor_explorer_missingness")
    _table(summary, "donor_explorer_missingness_summary", "Metadata missingness")
    selected = st.multiselect("Inspect outcome availability together", fields, default=[c for c in ("cogn_global", "cogng_demog_slope", "cogng_path_slope") if c in fields],
                              format_func=_label, key=context + "_availability_fields")
    if selected:
        # This is observed-outcome overlap, not correlations or imputation.
        observed = frame[selected].notna().astype(int)
        overlap = observed.T @ observed
        fig = px.imshow(overlap.to_numpy(), x=[_label(c) for c in selected], y=[_label(c) for c in selected],
                        text_auto=len(selected) <= 8, color_continuous_scale="Blues", labels=dict(color="Donors observed for both"))
        fig.update_layout(height=max(600, 35 * len(selected) + 250))
        _plot(fig, render, "donor_explorer_outcome_overlap")
        st.caption("Diagonal: donors with that outcome. Off-diagonal: donors with both outcomes; not a correlation heatmap.")


def _details(frame, regional, original, dataset, context):
    all_columns = list(frame.columns)
    defaults = [c for c in ("sample_id", "metadata_namespace", "diagnosis_group", "study", "available_tissues", "age_at_death", "sex_code", "clusters", "cogn_global", "gpath", "metadata_snapshot") if c in frame]
    columns = st.multiselect("Donor table columns", all_columns, default=defaults, format_func=_label, key=context + "_table_columns")
    if columns:
        shown = frame[columns]
        filterable_dataframe(shown, table_key="donor_explorer_samples", table_name="Selected donor metadata",
                             column_config={c: _label(c) for c in columns}, hide_index=True)
        exported = frame[list(dict.fromkeys(["metadata_namespace", "metadata_snapshot", *columns]))]
        st.caption("Downloads retain the dataset namespace and metadata snapshot even when those columns are hidden.")
        st.download_button("Download selected donors and columns", exported.to_csv(sep="\t", index=False).encode(),
                           f"donor_explorer_{dataset}_selected.tsv", mime="text/tab-separated-values", key="donor_explorer_download_samples")
    sample = st.selectbox("Inspect one public sample", sorted(frame.sample_id), key=context + "_sample")
    row = frame.loc[frame.sample_id.eq(sample)].iloc[0]
    detail = pd.DataFrame([{"Property": _label(c), "Value": D.MISSING if pd.isna(row[c]) else str(row[c])}
                           for c in all_columns])
    _table(detail, "donor_explorer_sample_detail", "Selected sample properties")
    if dataset == "brain" and sample in set(original.sample_id):
        with st.expander("Compare this sample's original and regional metadata snapshots"):
            a = original.loc[original.sample_id.eq(sample)].iloc[0]
            b = regional.loc[regional.sample_id.eq(sample)].iloc[0]
            rows = []
            for c in sorted((set(original) & set(regional) & D.ALLOWED_FIELDS) - {"sample_id"}):
                av, bv = a[c], b[c]
                same = D.same_metadata_value(av, bv)
                rows.append(dict(field=c, property=_label(c), original=D.MISSING if pd.isna(av) else str(av),
                                 regional=D.MISSING if pd.isna(bv) else str(bv), different=not same))
            _table(pd.DataFrame(rows), "donor_explorer_snapshot_detail", "Sample snapshot comparison")


def render_view(render):
    st.header("Donor explorer")
    st.caption("Explore packaged donor metadata and expression-region coverage. No expression, edge, score or prediction files are loaded.")
    dataset = st.radio("Donor dataset", list(D.DATASET_LABELS), format_func=D.DATASET_LABELS.get, horizontal=True, key="donor_explorer_dataset")
    st.caption("Brain and motor public sample IDs have separate namespaces; identical IDs are not evidence of the same person. This page never links the datasets.")
    try:
        catalog, original = cached_catalog(dataset, D.source_revision(dataset))
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        st.warning("Donor metadata for this dataset are unavailable or failed validation. " + str(error))
        return
    tissues = D.BRAIN_TISSUES if dataset == "brain" else D.MOTOR_TISSUES
    options = D.cohort_options(dataset, catalog, original)
    preset = st.selectbox("Donor cohort / region coverage", list(options), format_func=options.get, key="donor_explorer_cohort_" + dataset)
    required = (); mode = "at_least"
    if preset == "custom":
        a, b = st.columns(2)
        required = a.multiselect("Required brain regions", D.BRAIN_TISSUES, default=["DLPFC"], key="donor_explorer_required_regions")
        mode = b.radio("Coverage rule", ["at_least", "exactly", "any"], format_func=lambda x: {"at_least": "At least these regions", "exactly": "Only these regions", "any": "Any of these regions"}[x], key="donor_explorer_coverage_rule")
    cohort = D.select_cohort(dataset, catalog, original, preset, required, mode)
    context = "donor_explorer_" + dataset + "_" + preset
    if dataset == "motor":
        st.info("The public motor metadata cover the common 357 VH/SMA/QUAD donors, shared by both module partitions. Single-tissue-only or larger raw motor cohorts are not packaged here. 'Motor dataset' does not mean these people lack brain tissue.")
    else:
        st.caption("'DLPFC available' includes donors who also have other regions. 'DLPFC only' excludes AC and PCG. All-three coverage includes 464 donors; the original analyzed cohort is a distinct 450-donor subset. Coverage alone does not imply every score is available.")
        st.caption("Metadata snapshot: " + (cohort.metadata_snapshot.iloc[0] if len(cohort) else "Regional / validation metadata") + ". Diagnosis is the stored group label, not reconstructed from CogDx. Snapshots are never silently combined.")
    frame = _filters(cohort, context)
    a, b, c = st.columns(3)
    a.metric("Selected unique donors", len(frame))
    b.metric("Cohort before property filters", len(cohort))
    c.metric("All three regions / tissues", int(frame.n_available_tissues.eq(3).sum()))
    if frame.empty:
        st.info("No donors match the selected coverage and metadata filters. Missing regions and outcomes are not imputed.")
        return
    section = st.radio("Donor view", ["Overview", "Phenotype distributions", "Missingness", "Donor table & details"], horizontal=True, key="donor_explorer_section")
    def render_in_context(figure, **kwargs):
        title = ("Brain" if dataset == "brain" else "Motor") + " · " + options[preset]
        title = "<br>".join(textwrap.wrap(title, width=58))
        figure.update_layout(title=dict(text=title + f"<br><sup>Selected donors: {len(frame):,}</sup>",
                                        x=.02, xanchor="left", y=.96, yanchor="top", font=dict(size=14)),
                             margin=dict(t=150), legend=dict(y=1.03),
                             meta=dict(dataset=dataset, cohort=preset, selected_donors=len(frame),
                                       metadata_snapshot=frame.metadata_snapshot.iloc[0]))
        render(figure, **kwargs)
    if section == "Overview":
        _overview(frame, tissues, render_in_context)
    elif section == "Phenotype distributions":
        _phenotypes(frame, context, render_in_context)
    elif section == "Missingness":
        _missingness(frame, context, render_in_context)
    else:
        _details(frame, catalog, original, dataset, context)
    with st.expander("Metadata provenance and snapshot differences"):
        st.write("Source files: " + "; ".join(str(p.relative_to(D.DATA_DIR)) for p in D.source_paths(dataset)))
        st.caption("Only existing approved public metadata are used. No donor names, projid, private mapping or new clinical values are added. File/cohort names are not a new claim about assessment timing or release date.")
        st.caption("AD–Control split labels are copied from the selected metadata snapshot; they are not a global train/test role for every analysis or prediction model.")
        if dataset == "brain":
            st.caption("The snapshot audit below compares all matched original-analysis donors, independently of the current page filters.")
            differences = D.snapshot_differences(original, catalog)
            changed = differences.loc[differences[["differing_values", "missing_only_original", "missing_only_regional"]].gt(0).any(axis=1)]
            if not changed.empty:
                st.warning("Some original-450 and regional metadata values differ. Choose the original-analysis cohort to inspect values used by that analysis; other coverage cohorts use the regional snapshot. No value is overwritten or filled from the alternative snapshot.")
                _table(changed, "donor_explorer_snapshot_differences", "Metadata snapshot differences")
            else:
                st.caption("Matched original/regional records agree to numerical precision, with no one-sided missing values. Numeric comparisons use rtol=1e-10, atol=1e-12; stored values are not changed.")
