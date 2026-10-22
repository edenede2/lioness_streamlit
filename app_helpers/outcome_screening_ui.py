"""Compact offline catalogs only: no expression or genome-wide app computation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_helpers.data import OUTCOME_LABELS, MODULE_SET_LABELS
from app_helpers.drive_data import DATA_DIR
from app_helpers.figure_export import render_exportable_plot
from app_helpers.table_controls import filterable_dataframe

ROOT = DATA_DIR/"outcome_screening"


def load_manifest(root=ROOT):
    path = Path(root)/"manifest.json"
    if not path.is_file():
        return {"files": [], "status": "unavailable"}
    record = json.loads(path.read_text())
    if record.get("schema_version") != 1:
        raise ValueError("Unsupported gene/edge association manifest")
    return record


@st.cache_data(max_entries=3, ttl=900, show_spinner=False)
def load_catalog(path, expected_hash, expected_rows, outcome=None, modules=()):
    path = Path(path)
    if path.stat().st_size >= 95*1024**2:
        raise ValueError("Catalog shard is too large for Community Cloud")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024**2), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError("Screening catalog hash mismatch; do not mix releases")
    file = pq.ParquetFile(path)
    if file.metadata.num_rows != expected_rows:
        raise ValueError("Screening catalog row-count mismatch")
    filters = []
    if outcome is not None:
        filters.append(("outcome", "=", outcome))
    if modules:
        filters.append(("module", "in", list(modules)))
    frame = pd.read_parquet(path, filters=filters or None)
    if {"donor", "projid", "sample_id"} & set(frame):
        raise ValueError("Screening aggregate grain/privacy check failed")
    return frame


def render_view(root=ROOT):
    st.subheader("Gene & edge associations")
    st.caption("Hits are ranked by absolute original-cohort correlation, retaining their sign. Validation evaluates the frozen union of Pearson and Spearman hits; its selected-hit FDR is not the original full-search FDR.")
    manifest = load_manifest(root)
    if not manifest.get("files"):
        st.info("Screening catalogs are not available yet. Existing module analyses are unchanged.")
        return
    analysis = st.radio("Screening view", ["genes", "module_edges", "whole_network"], horizontal=True,
        format_func=lambda v:{"genes":"Genes", "module_edges":"Module edges", "whole_network":"Whole-network edges"}[v])
    available = [e for e in manifest["files"] if e["path"].startswith(analysis+"/") and e.get("status") == "complete"]
    if not available:
        st.info("This catalog is unfinished and cannot be selected. Motor full M5 and whole-network searches require their numerical/resource pilot.")
        return
    entry = st.selectbox("Completed catalog", available, format_func=lambda e:e.get("catalog_id",e["path"]).removeprefix(analysis+"/").removesuffix(".parquet").replace("control_anchored", "Control-referenced").replace("_", " "))
    outcomes = [o for o in OUTCOME_LABELS if o in entry.get("outcomes",manifest.get("outcomes",[]))]
    if not outcomes:
        st.info("This completed search has no eligible outcomes.")
        return
    first, second, third = st.columns(3)
    outcome = first.selectbox("Outcome", outcomes, format_func=OUTCOME_LABELS.get)
    correlation = second.selectbox("Correlation", ["spearman", "pearson"], format_func=str.title)
    cohort = third.selectbox("Displayed statistics", ["original", "validation"], format_func=lambda v:"Original full search" if v == "original" else "Frozen-hit validation")
    selected_modules = ()
    if analysis == "module_edges":
        modules = entry.get("modules",[])
        if not modules:
            # Compatibility with the first incremental manifest schema: a
            # single int column, never the whole donor or statistic catalog.
            module_column = pd.read_parquet(Path(root)/entry["path"], columns=["module"], filters=[("outcome", "=", outcome)])
            modules = sorted(module_column.module.dropna().astype(int).unique())
        selected_modules = st.multiselect("Modules", modules, default=modules[:1], format_func=lambda m:f"M{m}")
        if not selected_modules:
            st.info("Select a module to load its compact results.")
            return
    view = load_catalog(str(Path(root)/entry["path"]), entry["sha256"], entry["rows"],outcome,tuple(selected_modules)).copy()
    if view.empty:
        st.info("This completed search has no variable edges/genes with enough observed outcome pairs. No correlations were invented.")
        return
    component_col = "component" if "component" in view else "tissue"
    levels = sorted(view[component_col].unique())
    selected_levels = st.multiselect("Tissues / components", levels, default=levels)
    view = view.loc[view[component_col].isin(selected_levels)]
    selection = st.selectbox("Original selection", ["union", "spearman", "pearson"], format_func=lambda v:"Pearson/Spearman frozen union" if v == "union" else f"Top original {v.title()} hits")
    if selection != "union":
        view = view.loc[view[f"{selection}_rank"].notna()]
    group = st.selectbox("Donor group", ["pooled", "control", "mci", "ad"], format_func=lambda v:"All donors (pooled)" if v == "pooled" else v.upper() if v in {"mci", "ad"} else v.title())
    prefix = ("" if cohort == "original" else "validation_") if group == "pooled" else f"{cohort}_{group}_"
    r_col, p_col = f"{prefix}{correlation}_r", f"{prefix}{correlation}_p"
    if group != "pooled":
        st.warning("Diagnosis-specific checks are descriptive post-selection annotations. No subgroup full-search FDR was calculated.")
        q_col = None
    else:
        q_col = f"{correlation}_fdr" if cohort == "original" else f"validation_{correlation}_selected_fdr"
        if analysis != "genes":
            scope = st.radio("FDR scope", ["within_module", "partition"], horizontal=True, format_func=lambda v:"Original module / selected module hits" if v == "within_module" else "Partition-wide")
            if scope == "partition":
                q_col = f"{correlation}_partition_fdr" if cohort == "original" else f"validation_{correlation}_partition_selected_fdr"
        threshold = st.selectbox("FDR display filter", [None, .05, .10], format_func=lambda v:"No filter" if v is None else f"FDR < {v:.2f}")
        if threshold is not None:
            view = view.loc[view[q_col].lt(threshold)]
    if view.empty:
        st.info("No frozen selected hits satisfy these display filters.")
        return
    view["display_correlation"] = view[r_col]
    view["label"] = view["gene_symbol"] if analysis == "genes" else view.gene_symbol_left+" — "+view.gene_symbol_right+" ("+view[component_col]+")"
    view = view.sort_values("display_correlation", key=lambda x:x.abs(), ascending=False, kind="stable")
    limit = st.number_input("Maximum plotted hits", min_value=1, max_value=200, value=min(30, len(view)))
    plotted = view.loc[view.display_correlation.notna()].head(int(limit))
    if plotted.empty:
        st.info("These hits are unavailable in the selected validation/group cohort; see unavailable reasons in the table.")
    else:
        hover = [component_col, p_col, f"{prefix}n"]
        if q_col:
            hover.append(q_col)
        fig = px.bar(plotted.iloc[::-1], x="display_correlation", y="label", orientation="h", color=component_col,
            hover_data=[c for c in hover if c in plotted], labels={"display_correlation":f"{correlation.title()} correlation", "label":""}, height=max(450, min(1400, 24*len(plotted)+160)))
        fig.update_layout(legend=dict(orientation="h", y=1.1), margin=dict(t=90, l=80))
        fig.update_xaxes(range=[-1, 1], zeroline=True)
        render_exportable_plot(fig, plot_id="screen_ranking")
    if group == "pooled":
        replication = view.dropna(subset=[f"{correlation}_r", f"validation_{correlation}_r"])
        if len(replication):
            fig = px.scatter(replication, x=f"{correlation}_r", y=f"validation_{correlation}_r", color=component_col, hover_name="label",
                labels={f"{correlation}_r":"Original correlation", f"validation_{correlation}_r":"Validation correlation"})
            fig.add_trace(go.Scatter(x=[-1,1], y=[-1,1], mode="lines", line=dict(color="black", dash="dash"), showlegend=False, hoverinfo="skip"))
            fig.update_xaxes(range=[-1,1]); fig.update_yaxes(range=[-1,1])
            fig.update_layout(legend=dict(orientation="h", y=1.1), margin=dict(t=90))
            render_exportable_plot(fig, plot_id="screen_replication")
    st.caption(f"{len(view):,} selected hits shown. FDR display filters do not change the original search or validation families. Raw signed donor-edge weights only; these are associations, not causal effects.")
    filterable_dataframe(view, table_key="screen_table", table_name="Selected-hit statistics")
    st.download_button("Download displayed statistics (TSV)", view.to_csv(sep="\t", index=False).encode(), "gene_edge_associations.tsv", mime="text/tab-separated-values")
    if analysis == "module_edges":
        with st.expander("Open existing module exploration tools"):
            for module in sorted(view.module.unique())[:5]:
                for tool in ("Associations", "Donor edge explorer"):
                    query = urlencode(dict(view=tool, screen_partition=str(view.partition.iloc[0]),
                        screen_method=str(view.network_method.iloc[0]), screen_module=int(module)))
                    st.link_button(f"M{module}: {tool}", "?"+query)
            st.caption("These links open the existing module tools. Their legacy module-score and edge-selection settings are unchanged; no new donor weights are exposed by this catalog.")
    with st.expander("Catalog availability and methods"):
        st.json({k:v for k,v in manifest.items() if k != "files"})
        st.write("Whole-network p-values are sorted offline with disk-backed exact BH. No genome-wide calculations run inside this page. New donor-level values are not included in this aggregate release.")
