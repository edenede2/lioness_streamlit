"""Same-module, common-donor score comparisons within Associations.

No networks, eigengenes, transformations or masks are fitted here. Callers supply
the existing cohort/method-aware loaders. Correlations are recomputed because
the complete-case intersection can differ from the stored association cohort.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from html import escape
import textwrap
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app_helpers.charts import aggregate_to_long, resolved_to_long, grouped_association_figure
from app_helpers.correlations import calculate_correlations
from app_helpers.composite_bars import refresh_composite_bars


@dataclass(frozen=True)
class ScoreSeries:
    key: str
    family: str
    component: str
    group: str
    label: str
    color: str


def score_series_catalog(tissues, component_labels, *, pooled=True, eigengenes=True):
    """Only eigengenes from this module definition; no regional-ID matching."""
    colors = ("#0072B2", "#009E73", "#E69F00")
    result = []
    if eigengenes:
        result.extend(ScoreSeries(f"eigengene:{t}", "eigengene", f"TS_{'PCGBA23' if t == 'PCG' else t}",
            "Gene expression", f"{t} eigengene", colors[i % len(colors)])
            for i, t in enumerate(tissues))
    components = {**({"TS": "Pooled TS", "CT": "Pooled CT"} if pooled else {}),
                  **component_labels}
    for c, label in components.items():
        ct = c == "CT" or c.startswith("CT_")
        result.append(ScoreSeries(f"connectivity:{c}", "connectivity", c,
            "Between-region coordination" if ct else "Within-region coordination",
            str(label), "#7B3294" if ct else "#D55E00"))
    return result


def adjacent_series(series):
    """Retain editable order while keeping regional eigengenes adjacent."""
    groups = list(dict.fromkeys(s.group for s in series))
    return [s for group in groups for s in series if s.group == group]


def load_series(module, series, loader: Callable, scale):
    sources, columns = {}, {}
    for spec in series:
        resolved = spec.component not in {"CT", "TS"}
        source_key = (spec.family, resolved)
        if source_key not in sources:
            source = loader(int(module), spec.family, resolved)
            sources[source_key] = (
                pd.DataFrame() if source.empty else
                resolved_to_long(source, scale) if resolved else
                aggregate_to_long(source, scale)
            )
        source = sources[source_key]
        if source.empty:
            columns[spec.key] = pd.Series(dtype=float)
            continue
        if not source["module"].astype(int).eq(int(module)).all():
            raise ValueError("Score loader returned a different module")
        part = source.loc[source.component.eq(spec.component), ["sample_id", "metric_value"]]
        if part.sample_id.duplicated().any():
            raise ValueError(f"Multiple score values per donor for {spec.label}")
        columns[spec.key] = pd.to_numeric(part.set_index("sample_id").metric_value, errors="coerce")
    return columns


def common_case_scores(metadata, columns, series, outcomes, module):
    """Intersect by sample ID, across all displayed scores and outcomes, per module."""
    base = metadata.set_index("sample_id", verify_integrity=True).copy()
    matrix = pd.DataFrame({s.key: columns[s.key].reindex(base.index) for s in series})
    required = matrix.copy()
    for outcome in outcomes:
        required[outcome] = pd.to_numeric(base[outcome], errors="coerce")
    complete = np.isfinite(required.to_numpy(dtype=float)).all(axis=1)
    base = base.loc[complete]
    parts = []
    for spec in series:
        part = base.reset_index().copy()
        part["module"] = int(module)
        part["component"] = spec.key
        part["source_component"] = spec.component
        part["component_label"] = spec.label
        part["metric_family"] = spec.family
        part["score_group"] = spec.group
        part["metric_value"] = matrix.loc[base.index, spec.key].to_numpy()
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def comparison_statistics(frame, outcome, *, minimum_n=10):
    result = calculate_correlations(frame,
        ["module", "component", "component_label", "metric_family", "score_group"],
        [outcome], min_group_n=minimum_n)
    # Never BH-adjust only the displayed bars or carry forward an old donor set's FDR.
    return result.drop(columns=["pearson_fdr_displayed_family", "spearman_fdr_displayed_family"],
                       errors="ignore")


def scatter_statistics(frame, outcome, options):
    variable = options["grouping_variable"]
    pieces = []
    for level in options["grouping_levels"]:
        part = frame if variable == "__all__" else frame.loc[frame[variable].eq(level)]
        if not part.empty:
            stat = comparison_statistics(part, outcome, minimum_n=options["minimum_group_n"])
            stat["grouping_level"] = str(level)
            pieces.append(stat)
    if options["show_pooled"]:
        stat = comparison_statistics(frame, outcome, minimum_n=options["minimum_group_n"])
        stat["grouping_level"] = "__pooled__"
        pieces.append(stat)
    if not pieces:
        return pd.DataFrame(columns=["component", "grouping_level"])
    result = pd.concat(pieces, ignore_index=True)
    result["pearson_fdr_across_modules"] = np.nan
    result["spearman_fdr_across_modules"] = np.nan
    return result


def _wrapped(text, width=76):
    return "<br>".join(escape(line) for line in textwrap.wrap(str(text), width,
        break_long_words=False, break_on_hyphens=False))


def score_comparison_figure(statistics, series, *, method, module, outcome_label,
                            y_title, subtitle="", composite_keys=(),
                            composite_label="Regional eigengenes", show_values=True,
                            show_significance=False, significance_cutoff=.05,
                            absolute_values=False):
    series = adjacent_series(series)
    stats = statistics.set_index("component").reindex([s.key for s in series])
    signed_values = stats["spearman_rho" if method == "spearman" else "pearson_r"]
    values = signed_values.abs() if absolute_values else signed_values
    p_values = stats["spearman_p" if method == "spearman" else "pearson_p"]
    labels = [s.label for s in series]
    if len(labels) != len(set(labels)):
        raise ValueError("Give each displayed series a distinct label")
    composite = set(composite_keys)
    if composite and (len(composite) < 2 or not composite.issubset({s.key for s in series})):
        raise ValueError("Select at least two visible score series for a composite column")
    if composite and (not composite_label.strip() or composite_label in
                      [s.label for s in series if s.key not in composite]):
        raise ValueError("Give the composite column a distinct, nonempty label")
    details = np.array([[s.group, int(row.n), row.pearson_r, row.spearman_rho,
                         row.pearson_p, row.spearman_p, row.unavailable_reason]
                        for s, row in zip(series, stats.itertuples())], dtype=object)
    texts = [(f"{abs(v):.3f}" + ("*" if show_significance and pd.notna(p) and p < significance_cutoff else ""))
             if pd.notna(v) else "Unavailable" for v, p in zip(values, p_values)]
    hover = ("%{x}<br>%{customdata[0]}<br>N=%{customdata[1]}"
             "<br>Pearson r=%{customdata[2]:.4f}; p=%{customdata[4]:.3g}"
             "<br>Spearman ρ=%{customdata[3]:.4f}; p=%{customdata[5]:.3g}"
             "<br>%{customdata[6]}<extra></extra>")
    figure = go.Figure()
    if not composite:
        figure.add_trace(go.Bar(x=labels, y=values, marker_color=[s.color for s in series],
            name="Outcome correlation", showlegend=False, cliponaxis=False,
            text=texts if show_values else [""]*len(texts), textposition="outside",
            customdata=details, hovertemplate=hover))
    else:
        for i, spec in enumerate(series):
            grouped = spec.key in composite
            summary = spec.label + (f" · {texts[i]}" if show_values and grouped else "")
            record = [*details[i], summary, spec.label]
            figure.add_trace(go.Bar(x=[composite_label if grouped else spec.label], y=[values.iloc[i]],
                name=spec.label, legendgroup=spec.label, showlegend=False,
                marker=dict(color=spec.color, line=dict(color="white", width=0 if grouped else 0.7)), base=0,
                width=.62, text=[texts[i]] if show_values and not grouped else [""],
                textposition="outside", cliponaxis=False,
                meta={"score_composite": dict(value=float(values.iloc[i]), order=i, id=spec.key,
                    label=texts[i], show_value=show_values)} if grouped else None,
                customdata=np.array([record], dtype=object),
                hovertemplate=hover.replace("%{x}", "%{customdata[8]}")))
        figure.update_layout(barmode="overlay")
        labels = list(dict.fromkeys(composite_label if s.key in composite else s.label for s in series))
    group_key = []
    for group in dict.fromkeys(s.group for s in series):
        if composite:
            items = []
            for spec in series:
                if spec.group != group:
                    continue
                label = spec.label
                if spec.key in composite and show_values:
                    label += f" · {texts[[s.key for s in series].index(spec.key)]}"
                items.append(f'<span style="color:{spec.color}">■</span> {escape(label)}')
            group_key.append(f"<b>{escape(group)}</b>: " + "<br />".join(items))
        else:
            labels_in_group = [s.label.removesuffix(" eigengene") for s in series if s.group == group]
            group_key.append(f"<b>{escape(group)}</b>: {escape(', '.join(labels_in_group))}")
    figure.add_annotation(name="publication-panel-title-score-groups", x=0.5, y=1,
        xref="x domain", yref="y domain", text="<br />".join(group_key), align="left",
        showarrow=False, font_size=12, yanchor="bottom", yshift=8)
    title = f"Module M{module}: score comparison · {escape(outcome_label)}"
    if subtitle:
        title += f"<br><sup>{_wrapped(subtitle,100)}</sup>"
    figure.update_layout(template="plotly_white", height=720 if composite else 660,
        title=dict(text=title,x=0.01,xanchor="left"),
        margin=dict(l=70, r=80 if composite else 25, t=180+18*(len(series)-3) if composite else 180, b=150), bargap=0.35,
        meta={"figure_export": {"bar_selection": {"category_label": "Score series"}},
              "score_comparison": {"module": int(module), "common_case": True, "subtitle": subtitle,
                  "composite_keys": [s.key for s in series if s.key in composite],
                  "composite_label": composite_label if composite else None,
                  "composite_style": "vertical_deltas" if composite else None,
                  "y_axis_mode": "absolute" if absolute_values else "signed",
                  "value_labels": "absolute correlation; original sign retained in hover/statistics",
                  "significance": f"nominal p < {significance_cutoff:g}" if show_significance else None}})
    ticktext = []
    tick_seen = set()
    for spec in series:
        if spec.key in composite:
            if composite_label not in tick_seen:
                ticktext.append(_wrapped(composite_label,16))
                tick_seen.add(composite_label)
            continue
        tissue = spec.component.removeprefix("TS_").replace("PCGBA23","PCG")
        if spec.family == "eigengene" and spec.label == f"{tissue} eigengene":
            ticktext.append(escape(tissue))
        elif spec.component in {"CT","TS"} and spec.label == f"Pooled {spec.component}":
            ticktext.append(f"Pooled<br>{spec.component}")
        else:
            ticktext.append(_wrapped(spec.label,12))
    figure.update_xaxes(categoryorder="array", categoryarray=labels, tickangle=0, automargin=True,
                        tickmode="array", tickvals=labels, ticktext=ticktext)
    figure.layout.meta["figure_export"]["bar_selection"].update(
        group_label_annotation="publication-panel-title-score-groups", group_customdata_index=0,
        category_display_labels={label:text.replace("<br>"," ") for label,text in zip(labels,ticktext)})
    if composite:
        figure.layout.meta["figure_export"]["bar_selection"].update(
            series_label="Individual score series", group_item_customdata_index=7,
            group_items_multiline=True, group_item_colors=True)
    figure.update_yaxes(title_text=y_title, range=[0, 1.15] if absolute_values else [-1.15, 1.15],
                        tickvals=[0,.25,.5,.75,1] if absolute_values else [-1,-0.5,0,0.5,1],
                        zeroline=True, zerolinecolor="#333333", zerolinewidth=1)
    refresh_composite_bars(figure)
    return figure


def combine_panels(panels, *, title):
    """Compose existing single-panel figures; preserve native axes for the exporter."""
    rows = (len(panels)+1)//2
    result = make_subplots(rows=rows, cols=2, horizontal_spacing=0.12,
        vertical_spacing=0.38 if rows > 1 else 0.0)
    export_meta = None
    components, records = [], []
    for i, (heading, source) in enumerate(panels):
        row, col = i//2+1, i%2+1
        suffix = str(i+1) if i else ""
        source = go.Figure(source)
        for trace in source.data:
            # Existing one-component association figures have an unused second cell.
            if getattr(trace, "xaxis", None) not in {None, "x"}:
                continue
            if i > 1 and trace.legendgroup:
                trace.showlegend = False
            result.add_trace(trace, row=row, col=col)
        for axis in ("x", "y"):
            style = source.layout[axis+"axis"].to_plotly_json()
            for name in ("domain", "anchor", "matches"):
                style.pop(name, None)
            result.layout[axis+"axis"+suffix].update(style)
        target_x = result.layout["xaxis"+suffix].domain
        target_y = result.layout["yaxis"+suffix].domain
        old_x = source.layout.xaxis.domain or (0, 1)
        old_y = source.layout.yaxis.domain or (0, 1)
        for annotation in source.layout.annotations:
            a = annotation.to_plotly_json()
            for axis, old, target in (("x",old_x,target_x),("y",old_y,target_y)):
                ref = a.get(axis+"ref", "paper")
                if ref == "paper":
                    fraction = (a.get(axis,0)-old[0])/(old[1]-old[0])
                    a[axis] = target[0]+fraction*(target[1]-target[0])
                elif ref in {axis, axis+" domain"}:
                    a[axis+"ref"] = axis+suffix+(" domain" if ref.endswith(" domain") else "")
            if a.get("name", "").startswith("publication-statistics-"):
                a["name"] = f"publication-statistics-{i}"
            result.add_annotation(**a)
        meta = (source.layout.meta or {}).get("figure_export", {})
        component = f"comparison-panel-{i}"
        if meta.get("correlation_records"):
            export_meta = dict(meta)
            records.extend({**r, "component": component} for r in meta["correlation_records"])
        components.append(component)
        if source.layout.coloraxis:
            result.update_layout(coloraxis=source.layout.coloraxis)
    if export_meta:
        export_meta.update(components=components, correlation_records=records)
    subtitles = [(f.layout.meta or {}).get("score_comparison",{}).get("subtitle","")
                 for _,f in panels]
    subtitle = next((s for s in subtitles if s), "")
    title = _wrapped(title,100)+(f"<br><sup>{_wrapped(subtitle,115)}</sup>" if subtitle else "")
    result.update_layout(template="plotly_white", title=dict(text=title,x=0.01,xanchor="left"),
        width=1400, height=820 if rows == 1 else 1450,
        margin=dict(l=75,r=55,t=270,b=150),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.28, yanchor="bottom",
                    groupclick="togglegroup"),
        meta={"figure_export": export_meta or {}})
    if any(f.layout.barmode == "overlay" for _,f in panels):
        result.update_layout(barmode="overlay")
    refresh_composite_bars(result)
    return result


def render_score_comparison(*, modules, selected_module, tissues, component_labels,
        cohort_scope, eigengenes_available, metadata, load_scores, outcome, outcome_labels,
        scale, scale_label, correlation_method, minimum_n, scatter_options,
        enrichment, render_plot, table_renderer, module_definition, default_regions=None,
        analysis_context=None, default_network_components=("TS","CT")):
    """Thin additive UI using the app's bounded loaders and publication exporter."""
    import streamlit as st
    st.subheader("Same-module score comparison")
    selected = st.multiselect("Comparison modules", modules, default=[selected_module],
        format_func=lambda m:f"M{m}", max_selections=6, key="score_comparison_modules")
    regions = st.multiselect("Same-module eigengene regions", list(tissues),
        default=list(default_regions if default_regions is not None else tissues) if eigengenes_available else [],
        disabled=not eigengenes_available)
    if not eigengenes_available:
        st.info("Same-module eigengenes are unavailable for this cohort/definition. No original-cohort or independent-region eigengenes are substituted.")
    catalog = score_series_catalog(regions, component_labels,
        pooled=cohort_scope == "complete_450", eigengenes=eigengenes_available)
    lookup = {s.key:s for s in catalog}
    defaults = [s.key for s in catalog if s.family == "eigengene" or s.component in default_network_components]
    if not defaults:
        # Incomplete-region donors cannot have all six components at once.
        first_ct = next((s for s in catalog if s.component.startswith("CT_")), None)
        first_ts = next((s for s in catalog if s.component.startswith("TS_") and
                         (first_ct is None or s.component[3:] in first_ct.component[3:].split("__"))), None)
        defaults = [s.key for s in (first_ts,first_ct) if s is not None]
    visible = st.multiselect("Visible score series (display order)", list(lookup), default=defaults,
        format_func=lambda key:lookup[key].label,
        help="Remove/add series to change order. Same-family bars stay adjacent; individual correlations are never added together.")
    series = adjacent_series([lookup[key] for key in visible])
    arrangement = st.radio("Bar arrangement", ["Separate bars", "Vertical composite column"],
        help="Equal-width segments end at each correlation. Their heights are successive differences, not summed correlations.")
    absolute_values = st.checkbox("Show absolute correlations on Y-axis", value=False,
        help="Plot |r| or |ρ| above zero. The original signs, p-values and sample counts remain in hover text and tables; scatter panels are unchanged.")
    show_values = st.checkbox("Show correlation values", value=True)
    show_significance = st.checkbox("Show significance asterisks (nominal p < 0.05)", value=False)
    composite_keys, composite_label = (), "Regional eigengenes"
    if arrangement == "Vertical composite column":
        default_composite = [s.key for s in series if s.family == "eigengene"]
        if len(default_composite) < 2:
            default_composite = [s.key for s in series[:2]]
        composite_keys = st.multiselect("Series in composite column", [s.key for s in series],
            default=default_composite, format_func=lambda key:lookup[key].label)
        composite_label = st.text_input("Composite column label", "Regional eigengenes"
            if composite_keys and all(lookup[key].family == "eigengene" for key in composite_keys) else "Selected scores")
        st.caption("Equal-width vertical segments: the column ends at the strongest correlation, not their sum. Each segment is the difference from the preceding value; tied values share a boundary. " +
            ("Absolute mode combines both signs above zero." if absolute_values else "Opposite signs remain on opposite sides of zero."))
    st.caption("Value labels show correlation magnitude only, without signs or N. " +
        ("Absolute mode hides direction on the axis. " if absolute_values else "Bar position retains direction. ") +
        "Signed correlations, p-values and N remain in hover text and the statistics table. Optional * means recalculated nominal p < 0.05, not module-set FDR.")
    with st.expander("Score labels and colors"):
        edited = []
        for spec in series:
            left, right = st.columns([3,1])
            label = left.text_input(f"Label · {spec.label}", spec.label, key=f"comparison_label::{spec.key}")
            color = right.color_picker(f"Color · {spec.label}", spec.color, key=f"comparison_color::{spec.key}")
            edited.append(replace(spec, label=label, color=color))
        series = edited
    signed_title = "Correlation to clinical cognition" if outcome == "cogn_global" else f"Correlation to {outcome_labels[outcome]}"
    absolute_title = "Absolute " + signed_title[0].lower() + signed_title[1:]
    label_key = f"comparison_y_label::{outcome}"
    default_title = absolute_title if absolute_values else signed_title
    if label_key not in st.session_state or st.session_state[label_key] in {signed_title, absolute_title}:
        st.session_state[label_key] = default_title
    y_title = st.text_input("Correlation axis label", key=label_key)
    layout = st.radio("Score comparison layout", ["Comparison bars", "CT scatter beside comparison bars"])
    combined = layout != "Comparison bars"
    ct_options = [s for s in catalog if s.component == "CT" or s.component.startswith("CT_")]
    ct_key = st.selectbox("Companion CT component", [s.key for s in ct_options],
        format_func=lambda key:lookup[key].label, disabled=not combined) if ct_options else None
    adjusted = st.checkbox("Include pathology-adjusted cognitive-slope scatter", value=False,
        disabled=not combined or "cogng_path_slope" not in metadata)
    overview = st.checkbox("Show score-scatter overview above", value=False)
    separate = st.checkbox("Show separate panels too", value=False, disabled=not combined)
    st.caption("Each bar shows a " + ("correlation magnitude" if absolute_values else "signed correlation") + " across the same complete-case donors for that module, after the active cohort and donor filters. Eigengenes belong to this module definition and are not edge-filtered. Network bars use the active method and edge rule. Nominal p-values are recalculated; FDRs from a different donor set are not reused.")
    if not selected or not series:
        st.info("Select at least one module and score series.")
        return
    if arrangement == "Vertical composite column" and len(composite_keys) < 2:
        st.info("Select at least two visible score series for a composite column.")
        return
    if composite_keys and (not composite_label.strip() or composite_label in
                          [s.label for s in series if s.key not in composite_keys]):
        st.info("Give the composite column a distinct, nonempty label.")
        return
    if (combined or overview) and scatter_options["trend_line_rule"] == "fdr":
        st.info("FDR-based trend filtering is unavailable after the common-donor recalculation, so no trend lines are shown. Choose All eligible groups or the nominal-p line rule to display OLS guides.")
    for module in selected:
        needed = list(series)
        if combined and ct_key and ct_key not in {s.key for s in needed}:
            needed.append(lookup[ct_key])
        outcomes = list(dict.fromkeys([outcome]+(["cogng_path_slope"] if adjusted else [])))
        try:
            columns = load_series(module, needed, load_scores, scale)
            common = common_case_scores(metadata, columns, needed, outcomes, module)
        except (ValueError, KeyError, FileNotFoundError) as error:
            st.warning(f"M{module}: score comparison unavailable: {error}")
            continue
        if common.empty:
            missing = [s.label for s in needed if columns[s.key].dropna().empty]
            st.warning(f"M{module}: no common complete-case donors. Unavailable scores: {', '.join(missing) or 'no overlap across selected scores/outcomes'}. Choose fewer series or regions.")
            continue
        statistics = comparison_statistics(common, outcome, minimum_n=minimum_n)
        unavailable = statistics.loc[statistics.component.isin(visible) & ~statistics.eligible]
        if not unavailable.empty:
            st.info("Unavailable correlations: " + "; ".join(
                f"{r.component_label}: {r.unavailable_reason} (N={r.n})"
                for r in unavailable.itertuples()))
        subtitles = enrichment(int(module), [s.component for s in needed])
        subtitle = subtitles.get("CT") or next(iter(subtitles.values()), "KEGG enrichment: unavailable")
        bars = score_comparison_figure(statistics, series, method=correlation_method,
            module=module, outcome_label=outcome_labels[outcome], y_title=y_title, subtitle=subtitle,
            composite_keys=composite_keys, composite_label=composite_label,
            show_values=show_values, show_significance=show_significance, absolute_values=absolute_values)
        def scatter(frame, target):
            options = dict(scatter_options)
            options["annotation_fields"] = [v for v in options["annotation_fields"] if v != "fdr"]
            # An old FDR-based line rule cannot be applied to this new donor family.
            if options["trend_line_rule"] == "fdr":
                options["trend_line_rule"] = "none"
            kwargs = dict(phenotype=target, phenotype_label=outcome_labels[target],
                feature_label="Connectivity" if frame.metric_family.eq("connectivity").all() else "Module score",
                scale_label=scale_label, module=module,
                module_definition=module_definition,
                kegg_subtitles={s.key:subtitles.get(s.component, "") for s in needed})
            return grouped_association_figure(frame, scatter_statistics(frame,target,options), **kwargs, **options)
        context = dict(**(analysis_context or {}), module=f"M{module}", outcome=outcome, module_definition=module_definition,
                       scale=scale, common_case_n=common.sample_id.nunique(),
                       bar_arrangement=arrangement, composite_score_series=list(composite_keys),
                       bar_y_axis="absolute" if absolute_values else "signed")
        def show(fig, suffix):
            render_plot(fig, plot_id=f"associations.score_comparison.M{module}.{suffix}",
                metadata=context, use_container_width=True, config={"displaylogo":False})
        if overview:
            show(scatter(common.loc[common.component.isin(visible)], outcome), "overview")
        panels = []
        if combined and ct_key:
            ct = common.loc[common.component.eq(ct_key)]
            panels.append(("Between-region coordination", scatter(ct, outcome)))
        panels.append(("Score comparison", bars))
        if adjusted and combined and ct_key:
            st.caption("The companion uses the existing demographics- and pathology-adjusted cognitive slope; no additional adjustment model is fitted. Its missing values also enter the shared donor intersection.")
            panels.append(("Pathology-adjusted cognitive slope", scatter(ct, "cogng_path_slope")))
        if combined and len(panels)>1:
            show(combine_panels(panels,title=f"Module M{module}: {outcome_labels[outcome]} · {module_definition}"),"combined")
            if separate:
                for i,(_,panel) in enumerate(panels):
                    show(panel,f"panel_{i+1}")
        else:
            show(bars,"bars")
        display = statistics.loc[statistics.component.isin(visible)].copy()
        display.insert(0,"module_definition",module_definition)
        for key,value in (analysis_context or {}).items():
            display[key] = value
        with st.expander(f"M{module} common-donor correlations and data"):
            table_renderer(display, table_key=f"comparison_stats_M{module}", table_name="Common-donor correlations",
                            hide_index=True, use_container_width=True)
            st.download_button("Download comparison statistics (TSV)", display.to_csv(sep="\t",index=False).encode(),
                file_name=f"{module_definition}_M{module}_{outcome}_score_comparison.tsv", key=f"comparison_download_M{module}")
            public = common[["sample_id","module","component","source_component","metric_family","metric_value",*outcomes]]
            public = public.copy()
            for key,value in (analysis_context or {}).items():
                public[key] = value
            public["score_edge_filter"] = np.where(public.metric_family.eq("eigengene"),
                "not applicable (expression)", (analysis_context or {}).get("differential_edge_rule","all"))
            st.download_button("Download common-donor plot data (TSV)", public.to_csv(sep="\t",index=False).encode(),
                file_name=f"M{module}_{outcome}_score_comparison_data.tsv", key=f"comparison_points_M{module}")
