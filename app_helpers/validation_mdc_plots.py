"""Display-only resolved MDC adapters for the historical MDC plot helpers.

Never pool unavailable tissues, recalculate a ratio, or correct filtered p-values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from app_helpers.charts import mdc_entropy_figure, mdc_overview_figure


def filter_retained_edges(frame, minimum):
    if int(minimum) < 0:
        raise ValueError("Minimum retained edges must be nonnegative")
    if int(minimum) == 0:
        return frame.copy()
    column = "n_retained_edges" if "n_retained_edges" in frame else "n_edges"
    counts = pd.to_numeric(frame[column], errors="coerce")
    return frame.loc[counts.ge(int(minimum))].copy()


def filter_wide_components(frame, minimum):
    """Hide each under-supported scope independently; keep saved q untouched."""
    result = frame.copy()
    if int(minimum) < 0:
        raise ValueError("Minimum retained edges must be nonnegative")
    for scope in ("total", "ts", "ct"):
        possible = ["n_retained_edges_"+scope, "n_edges" if scope == "total" else "n_"+scope+"_edges"]
        column = next((c for c in possible if c in result), None)
        if column is None and scope == "total" and {"n_ts_edges", "n_ct_edges"} <= set(result):
            counts = pd.to_numeric(result.n_ts_edges, errors="coerce") + pd.to_numeric(result.n_ct_edges, errors="coerce")
        elif column is not None:
            counts = pd.to_numeric(result[column], errors="coerce")
        else:
            counts = None
        if counts is None:
            if int(minimum):
                raise ValueError("MDC retained-edge counts unavailable for "+scope)
            continue
        excluded = counts.lt(int(minimum))
        if int(minimum):
            excluded |= counts.isna()
        result["display_excluded_"+scope] = excluded
        for prefix in ("mdc_", "log2_mdc_"):
            result.loc[excluded, prefix+scope] = np.nan
    if int(minimum) and "ts_minus_ct_log2_mdc" in result:
        result.loc[result.mdc_ts.isna() | result.mdc_ct.isna(), "ts_minus_ct_log2_mdc"] = np.nan
    return result


def component_frame(frame, component, scope, details=None):
    data = frame.loc[frame.component.eq(component)].copy()
    if data.module.duplicated().any():
        raise ValueError("Select one MDC mask and method before plotting")
    for field in ("mdc", "log2_mdc", "directional_fdr"):
        data[field + "_" + scope] = pd.to_numeric(data[field], errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Decode biological direction from the ratio, not historical q_loss/q_gain names.
    data["direction_" + scope] = np.select(
        [data.mdc.gt(1), data.mdc.lt(1), data.mdc.eq(1)],
        ["Higher in AD", "Higher in Control", "Equal"], default="Not available")
    for field in ("n_retained_edges", "n_ad", "n_control"):
        data[field + "_" + scope] = data[field]
    keep = ["module"] + [c for c in data if c.endswith("_" + scope)]
    data = data[keep]
    if details is not None:
        fields = ["module", "module_size", "cluster_type", "tissue_entropy", "tissue_entropy_normalized"]
        data = data.merge(details[fields], on="module", how="left", validate="one_to_one")
    return data


def paired_components(frame, ts, ct):
    data = component_frame(frame, ts, "ts").merge(
        component_frame(frame, ct, "ct"), on="module", how="inner", validate="one_to_one")
    # Required historical adapter fields; deliberately unavailable, NOT pooled.
    data["mdc_total"] = np.nan
    data["directional_fdr_total"] = np.nan
    return data


def comparison_figure(frame, ts, ct, labels, module, threshold, scale, definition):
    data = paired_components(frame, ts, ct)
    figure = mdc_overview_figure(data, module, threshold, definition, scale)
    axis = "log2 MDC" if scale == "log2" else "Raw MDC (AD/Control)"
    figure.update_layout(title=dict(text=f"{labels[ts]} versus {labels[ct]} MDC across modules<br><sup>{definition} · incomplete-region donors</sup>"))
    figure.update_xaxes(title_text=labels[ts]+" · "+axis)
    figure.update_yaxes(title_text=labels[ct]+" · "+axis)
    lookup = data.set_index("module")
    for trace in figure.data:
        if (trace.meta or {}).get("export_role") == "selected_module" and trace.hovertemplate:
            trace.hovertemplate = trace.hovertemplate.replace("TS ", labels[ts]+" ").replace("CT ", labels[ct]+" ")
        if trace.customdata is not None:
            modules = [int(str(row[0]).removeprefix("M")) for row in trace.customdata]
            trace.customdata = np.array([[f"M{m}", *lookup.loc[m, ["mdc_ts", "directional_fdr_ts", "n_retained_edges_ts", "n_ad_ts", "n_control_ts", "mdc_ct", "directional_fdr_ct", "n_retained_edges_ct", "n_ad_ct", "n_control_ct"]].tolist()] for m in modules], dtype=object)
            trace.hovertemplate = (
                "Module: %{customdata[0]}<br>"+labels[ts]+" MDC=%{customdata[1]:.4g}; saved FDR=%{customdata[2]:.3g}<br>"
                "Retained edges=%{customdata[3]:,}; AD n=%{customdata[4]}; Control n=%{customdata[5]}<br>"
                +labels[ct]+" MDC=%{customdata[6]:.4g}; saved FDR=%{customdata[7]:.3g}<br>"
                "Retained edges=%{customdata[8]:,}; AD n=%{customdata[9]}; Control n=%{customdata[10]}<extra></extra>")
    return figure


def entropy_figure(frame, component, scope, labels, details, module, threshold, scale, definition):
    data = component_frame(frame, component, scope, details)
    figure = mdc_entropy_figure(data, scope=scope, selected_module=module,
        threshold=threshold, scale=scale, module_definition=definition)
    label = labels[component]
    figure.layout.title.text = figure.layout.title.text.replace(scope.upper()+" MDC", label+" MDC", 1)
    for trace in figure.data:
        if trace.hovertemplate:
            trace.hovertemplate = trace.hovertemplate.replace(scope.upper()+" ", label+" ")
        if trace.customdata is not None:
            lookup = data.set_index("module")
            rows = []
            for row in trace.customdata:
                m = int(str(row[0]).removeprefix("M"))
                rows.append([*row, *lookup.loc[m, ["n_retained_edges_"+scope, "n_ad_"+scope, "n_control_"+scope]].tolist()])
            trace.customdata = np.array(rows, dtype=object)
            trace.hovertemplate = trace.hovertemplate.replace("<extra></extra>",
                "<br>Retained edges=%{customdata[8]:,}; AD n=%{customdata[9]}; Control n=%{customdata[10]}<extra></extra>")
    return figure
