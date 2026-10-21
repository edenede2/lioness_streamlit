"""Standalone, explicitly triggered module-level comparison view."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import uuid

import streamlit as st

from app_helpers import data as D
from app_helpers import regional_modules as R
from app_helpers.figure_export import render_exportable_plot
from app_helpers.score_comparison import (
    COMBINED_COMPONENT, COMPARISON_COLORS, COMPONENT_LABELS, CORRELATION_LABELS, MAX_BOXES, SCALE_ORDER, ComparisonSeriesSpec,
    availability_reason, calculate_series, comparison_figure, display_results,
    effect_settings, source_registry,
)
from app_helpers.table_controls import filterable_dataframe


BOX_STATE = "module_comparison_boxes"
RESULT_STATE = "module_comparison_result"


def _new_box(label: str, spec: ComparisonSeriesSpec) -> dict:
    return {"uid": uuid.uuid4().hex[:12], "label": label, **asdict(spec)}


def default_boxes(registry: dict) -> list[dict]:
    rows = []
    regional = "single_region_complete_case_l3"
    if regional in registry:
        for tissue in R.REGION_ORDER:
            rows.append(_new_box(f"Regional eigengene: {tissue}", ComparisonSeriesSpec(
                source=regional, feature="eigengene", method="not_applicable",
                component="regional", tissues=(tissue,), cohort_scope="common_450",
            )))
    if "control_derived" in registry:
        for component in ("CT", "TS"):
            rows.append(_new_box(f"Control-derived brain: {component}", ComparisonSeriesSpec(
                source="control_derived", component=component,
            )))
    if not rows and registry:
        source = next(iter(registry))
        if registry[source]["regional"]:
            row = registry[source]["combinations"][0]
            rows.append(_new_box("Comparison 1", ComparisonSeriesSpec(
                source=source, feature=row["score_variant"], method=row["network_method"],
                component="regional", cohort_scope=row["cohort_scope"], tissues=(row["tissue"],),
            )))
        else:
            rows.append(_new_box("Comparison 1", ComparisonSeriesSpec(
                source=source, method=registry[source]["methods"][0],
            )))
    return rows


def edit_boxes(action: str, uid: str = "") -> None:
    """Widget callback; the completed previous render supplies each row's config."""
    rows = [dict(row) for row in st.session_state[BOX_STATE]]
    index = next((i for i, row in enumerate(rows) if row["uid"] == uid), None)
    if action in {"add", "duplicate"} and len(rows) < MAX_BOXES:
        template = rows[index] if index is not None else rows[-1]
        copy = dict(template, uid=uuid.uuid4().hex[:12], label=f"{template['label'][:93]} (copy)")
        rows.insert(index + 1 if index is not None else len(rows), copy)
    elif action == "remove" and index is not None and len(rows) > 1:
        rows.pop(index)
    elif action == "up" and index is not None and index > 0:
        rows[index - 1], rows[index] = rows[index], rows[index - 1]
    elif action == "down" and index is not None and index < len(rows) - 1:
        rows[index + 1], rows[index] = rows[index], rows[index + 1]
    st.session_state[BOX_STATE] = rows


def _select(label, options, default, key, **kwargs):
    options = list(options)
    if not options:
        raise ValueError(f"No available options for {label}")
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]
    return st.selectbox(label, options, index=options.index(default) if default in options else 0, key=key, **kwargs)


def _cohort_label(spec: ComparisonSeriesSpec, entry: dict) -> str:
    if spec.cohort_scope == "incomplete_region_validation":
        return "Non-discovery validation donors (component-specific N)"
    if spec.is_regional:
        return R.COHORT_LABELS[spec.cohort_scope]
    donors = entry.get("donors", 450)
    if isinstance(donors, dict):
        donors = donors.get(spec.method, next(iter(donors.values())))
    return f"Source complete-tissue cohort ({donors} donors)"


def _edge_controls(row: dict, source: str, method: str, feature: str, prefix: str, cohort_scope="complete_450") -> dict:
    settings = effect_settings(source)
    completed_networks = settings.get("completed_networks", [])
    has_effect = settings.get("status") == "complete" and (
        not completed_networks or f"lioness/{method}" in completed_networks
    )
    choices = ["all"]
    if cohort_scope == "complete_450" and D.differential_data_available(source):
        choices.append("fdr")
    if has_effect:
        choices.append("effect")
    original = row.get("differential_edge_rule", "all")
    default = "effect" if D.is_effect_rule(original) else "fdr" if original != "all" else "all"
    if feature == "eigengene":
        choices, default = ["all"], "all"
    subset = _select("Edge subset", choices, default, prefix + "edges",
                     format_func=lambda value: {"all": "All edges", "fdr": "FDR-filtered edges", "effect": "Effect-size-filtered edges"}[value],
                     disabled=feature == "eigengene")
    values = dict(differential_edge_rule="all", differential_fdr_scope="per_module",
                  differential_fdr_threshold=.05, score_normalization="standard_pruned")
    if subset == "fdr":
        values["differential_edge_rule"] = "ad_control_discovery_fdr05"
        values["differential_fdr_scope"] = _select(
            "Edge FDR scope", D.FDR_SCOPE_LABELS, row.get("differential_fdr_scope", "per_module"),
            prefix + "fdr_scope", format_func=D.FDR_SCOPE_LABELS.get,
        )
        values["differential_fdr_threshold"] = _select(
            "Edge FDR threshold", D.FDR_THRESHOLD_LABELS, row.get("differential_fdr_threshold", .05),
            prefix + "fdr_threshold", format_func=D.FDR_THRESHOLD_LABELS.get,
        )
    elif subset == "effect":
        # Old, fully complete brain manifests predate explicit availability lists.
        parsed = {}
        if D.is_effect_rule(original):
            statistic_key, cutoff_key, direction_key = original[len(D.EFFECT_RULE_PREFIX):].split("__")
            fixed = cutoff_key.startswith("fixed_")
            parsed = dict(
                statistic=statistic_key, direction=direction_key,
                cutoff_mode="fixed" if fixed else "module_top_fraction",
                cutoff_value=float(cutoff_key.removeprefix("fixed_")) if fixed else int(cutoff_key.removeprefix("module_top_").removesuffix("pct")) / 100,
            )
        statistic = _select("Effect statistic", settings.get("available_effect_statistics", D.EFFECT_STATISTIC_LABELS),
                            parsed.get("statistic", "hedges_g"), prefix + "effect_statistic",
                            format_func=D.EFFECT_STATISTIC_LABELS.get)
        modes = settings.get("available_cutoff_modes", {}).get(
            statistic, ["fixed", "module_top_fraction"] if statistic == "hedges_g" else ["module_top_fraction"])
        mode = _select("Effect threshold rule", modes, parsed.get("cutoff_mode", "fixed"), prefix + "effect_mode",
                       format_func=lambda value: {"fixed": "Absolute Hedges’ g threshold", "module_top_fraction": "Top fraction within each module"}[value])
        thresholds = settings.get("available_fixed_hedges_g", D.HEDGES_FIXED_THRESHOLDS) if mode == "fixed" else settings.get("available_module_top_fractions", D.MODULE_TOP_FRACTIONS)
        cutoff = _select("Effect threshold", thresholds, parsed.get("cutoff_value", .4 if mode == "fixed" else .05), prefix + "effect_threshold",
                         format_func=lambda value: f"|g| ≥ {value:g}" if mode == "fixed" else f"Top {value:.0%}")
        direction = _select("Effect direction", settings.get("available_directions", D.EFFECT_DIRECTION_LABELS),
                            parsed.get("direction", "either"), prefix + "direction", format_func=D.EFFECT_DIRECTION_LABELS.get)
        values["differential_edge_rule"] = D.effect_rule_key(statistic, mode, float(cutoff), direction)
    if subset != "all":
        normalizations = settings.get("available_score_normalizations", D.SCORE_NORMALIZATION_LABELS) if subset == "effect" else D.SCORE_NORMALIZATION_LABELS
        values["score_normalization"] = _select(
            "Score handling", normalizations, row.get("score_normalization", "standard_pruned"),
            prefix + "normalization", format_func=D.SCORE_NORMALIZATION_LABELS.get,
        )
    return values


def _box_editor(row: dict, registry: dict, position: int, count: int) -> tuple[dict, ComparisonSeriesSpec | None]:
    prefix = f"module_comparison__{row['uid']}__"
    with st.expander(f"Box {position + 1}: {row['label']}", expanded=False):
        label = st.text_input("Box label", value=row["label"], max_chars=100, key=prefix + "label")
        tick_label = st.text_area(
            "X-axis tick label", value=row.get("x_tick_label", ""), max_chars=200,
            height=80, key=prefix + "x_tick_label",
            help="Leave blank to follow the Box label. Enter a custom label below this box only; line breaks are supported. The box name in tables and downloads is unchanged.",
        )
        show_module_count = st.checkbox(
            "Show module count in X-axis tick", value=row.get("show_module_count", True),
            key=prefix + "show_module_count",
            help="Include the module count and, for CT + TS, score-observation count below this box's tick label. Counts always remain available in the tables.",
        )
        source = _select("Partition / score source", registry, row["source"], prefix + "source",
                         format_func=lambda value: registry[value]["label"])
        regional = registry[source]["regional"]
        features = ["eigengene", "connectivity"] if regional else list(D.FEATURE_LABELS)
        if not regional and D.descriptive_eigengene_data_available(source):
            features.append("eigengene")
        feature = _select("Module score", features, row["feature"], prefix + "feature",
                          format_func=lambda value: {**D.FEATURE_LABELS, **D.EIGENGENE_FEATURE_LABELS}[value])
        if regional:
            completed = registry[source]["combinations"]
            cohorts = [scope for scope in R.COHORT_LABELS if any(record["cohort_scope"] == scope and record["score_variant"] == feature for record in completed)]
            cohort = _select("Source cohort", cohorts, row.get("cohort_scope", "common_450"), prefix + "cohort", format_func=R.COHORT_LABELS.get)
            methods = list(dict.fromkeys(record["network_method"] for record in completed if record["cohort_scope"] == cohort and record["score_variant"] == feature))
            method = _select("Network method", methods, row["method"], prefix + "method",
                             format_func=lambda value: "Not applicable (eigengene)" if value == "not_applicable" else D.METHOD_LABELS[value], disabled=feature == "eigengene")
            tissues = [tissue for tissue in R.regional_tissues(source) if any(record["tissue"] == tissue and record["cohort_scope"] == cohort and record["score_variant"] == feature and record["network_method"] == method for record in completed)]
            tissue_key = prefix + "tissues"
            if tissue_key in st.session_state and not set(st.session_state[tissue_key]).issubset(tissues):
                del st.session_state[tissue_key]
            default_tissues = [t for t in row.get("tissues", ()) if t in tissues]
            if source != row["source"] and not default_tissues and tissues:
                default_tissues = [tissues[0]]
            all_tissues = R.regional_tissues(source)
            all_ready = set(tissues) == set(all_tissues)
            all_key = prefix + "all_regions"
            if not all_ready and st.session_state.get(all_key):
                st.session_state[all_key] = False
            use_all = st.checkbox(
                "All regions in one box", value=all_ready and set(default_tissues) == set(all_tissues),
                disabled=not all_ready, key=all_key,
                help="Pool all distinct regional modules in one distribution; module correlations are not averaged. FDR remains separate within each region.",
            )
            if use_all:
                # Streamlit removes widget state while a control is hidden.
                # Preserve the custom subset outside the widget's own key.
                if tissue_key in st.session_state:
                    st.session_state[prefix + "custom_tissues"] = list(st.session_state[tissue_key])
                selected_tissues = all_tissues
                st.caption(f"Includes every regional module from {', '.join(all_tissues)}. Each module is one observation.")
            else:
                remembered = st.session_state.get(prefix + "custom_tissues", default_tissues)
                if not set(remembered).issubset(tissues):
                    remembered = default_tissues
                selected_tissues = st.multiselect(
                    "Regions in this box", tissues, default=remembered, key=tissue_key,
                    help="Regional modules are distinct observations. FDR is corrected separately within each tissue partition.",
                )
                st.session_state[prefix + "custom_tissues"] = list(selected_tissues)
            if not all_ready:
                st.caption("All regions becomes available when this score/method/cohort is complete for every region.")
            component = "regional"
            edges = dict(differential_edge_rule="all", differential_fdr_scope="per_module", differential_fdr_threshold=.05, score_normalization="standard_pruned")
            st.caption("Regional scores are all-edge only; eigengenes are never edge-filtered.")
        else:
            cohort = "complete_450"  # Legacy key names this cohort; muscle has 357 donors.
            method = _select("Network method", ["not_applicable"] if feature == "eigengene" else registry[source]["methods"], row["method"], prefix + "method",
                             format_func=lambda value: "Not applicable (eigengene)" if value == "not_applicable" else D.METHOD_LABELS[value], disabled=feature == "eigengene")
            from app_helpers import camel_validation as V
            if source == "control_derived" and feature != "eigengene" and method == "control_anchored" and V.available():
                cohort=_select("Source cohort",["complete_450","incomplete_region_validation"],row.get("cohort_scope","complete_450"),prefix+"cohort",format_func=D.COHORT_SCOPE_LABELS.get)
            components = list(D.module_set_components(source))
            if feature == "eigengene":
                components = [value for value in components if value.startswith("TS_")]
            else:
                components = components if cohort == "incomplete_region_validation" else ["CT", "TS", COMBINED_COMPONENT, *components]
            component = _select("Score component", components, row["component"], prefix + "component", format_func=COMPONENT_LABELS.get)
            if component == COMBINED_COMPONENT:
                st.caption("CT + TS pools two correlations per module, one for each component; donor scores are not summed or averaged. BH remains separate for CT and TS. Top K ranks modules by their strongest eligible component and includes both qualifying scores.")
            selected_tissues = ()
            edges = _edge_controls(row, source, method, feature, prefix, cohort)
            if cohort == "incomplete_region_validation" and edges["differential_edge_rule"] == "all":
                edges["score_normalization"]=_select("Score handling",D.SCORE_NORMALIZATION_LABELS,row.get("score_normalization","standard_pruned"),prefix+"normalization",format_func=D.SCORE_NORMALIZATION_LABELS.get)
            st.caption("This partition's own donor cohort is used; it is not intersected with other boxes.")
        scale = _select("Score transformation", SCALE_ORDER, row["scale"], prefix + "scale",
                        format_func=lambda value: {"raw": "Raw", "asinh": "asinh", "rint": "RINT"}[value])
        color_columns = st.columns(2)
        with color_columns[0]:
            box_color = st.color_picker(
                "Box color", value=row.get("box_color", COMPARISON_COLORS[position % len(COMPARISON_COLORS)]),
                key=prefix + "box_color",
            )
        with color_columns[1]:
            dots_match_box = st.checkbox(
                "Match dots to box color", value=row.get("dots_match_box", True),
                key=prefix + "dots_match_box",
            )
            # Keep an independent custom color in row state while its widget is
            # hidden; Streamlit discards the state of unrendered widgets.
            dot_color = row.get("dot_color")
            if not dots_match_box:
                dot_color = st.color_picker(
                    "Dot color", value=dot_color or box_color, key=prefix + "dot_color",
                )
            else:
                st.caption("Dots follow the box color.")
        buttons = st.columns(4)
        buttons[0].button("Move up", key=prefix + "up", disabled=position == 0, on_click=edit_boxes, args=("up", row["uid"]))
        buttons[1].button("Move down", key=prefix + "down", disabled=position == count - 1, on_click=edit_boxes, args=("down", row["uid"]))
        buttons[2].button("Duplicate", key=prefix + "duplicate", disabled=count >= MAX_BOXES, on_click=edit_boxes, args=("duplicate", row["uid"]))
        buttons[3].button("Remove", key=prefix + "remove", disabled=count == 1, on_click=edit_boxes, args=("remove", row["uid"]))
        config = dict(source=source, feature=feature, method=method, component=component,
                      scale=scale, cohort_scope=cohort, tissues=tuple(selected_tissues), **edges)
        updated = dict(
            uid=row["uid"], label=label.strip() or f"Box {position + 1}",
            box_color=box_color, dot_color=dot_color, dots_match_box=dots_match_box, **config,
            x_tick_label=tick_label, show_module_count=show_module_count,
        )
        if regional and not selected_tissues:
            st.warning("Select at least one completed region for this box.")
            return updated, None
        spec = ComparisonSeriesSpec(**config)
        reason = availability_reason(spec, registry)
        if reason:
            st.warning(reason)
            return updated, None
        return updated, spec


def render_view(data_manifest: dict) -> None:
    st.header("Module–outcome comparison")
    st.caption("Each observation is one module, not an individual donor: absolute score–outcome correlation for numeric outcomes, or Kruskal epsilon-squared for ROS/MAP study membership. All regions pools distinct regional modules; CT + TS includes separate component effects per module. Comparisons are descriptive and use each source's own cohort.")
    registry = source_registry(data_manifest)
    if not registry:
        st.info("No completed score sources are available.")
        return
    if BOX_STATE not in st.session_state:
        st.session_state[BOX_STATE] = default_boxes(registry)
    with st.sidebar:
        st.header("Comparison controls")
        outcome = st.selectbox("Comparison outcome", [*D.OUTCOME_LABELS, "study"], format_func=D.ASSOCIATION_OUTCOME_LABELS.get, key="module_comparison_outcome")
        method = st.selectbox("Correlation statistic", list(CORRELATION_LABELS), format_func=CORRELATION_LABELS.get, key="module_comparison_correlation", disabled=outcome == "study")
        st.caption("ROS/MAP is nominal: boxes contain module Kruskal epsilon-squared effects, not correlations of study codes. Single-study sources have unavailable tests." if outcome == "study" else "Spearman usually agrees for Raw and asinh because asinh preserves ranks. Pearson can change. RINT ties or missingness can affect agreement.")
        subset_labels = dict(D.ANALYSIS_SUBSET_LABELS, all_donors="All source donors (pooled)")
        subset = st.selectbox("Evaluation subset", list(subset_labels), format_func=subset_labels.get, key="module_comparison_subset")
        min_n = st.selectbox("Minimum donor N per module", [5, 10, 20], index=1, key="module_comparison_min_n")
        selection = st.radio("Modules shown", ["All eligible modules", "Top K"], key="module_comparison_selection")
        k = st.number_input("K per box", min_value=1, value=20, step=1, disabled=selection != "Top K", key="module_comparison_k")
        fdr = st.selectbox("Module-association FDR filter", [None, .05, .10], format_func=lambda value: "None" if value is None else f"FDR < {value:.2f}" + (" (exploratory)" if value == .10 else ""), key="module_comparison_fdr")
        points = st.checkbox("Show module points", value=True, key="module_comparison_points")
    rows = st.session_state[BOX_STATE]
    boxes, updated = [], []
    for position, row in enumerate(rows):
        new_row, spec = _box_editor(row, registry, position, len(rows))
        updated.append(new_row)
        if spec is not None:
            boxes.append((new_row["uid"], new_row["label"], spec))
    st.session_state[BOX_STATE] = updated
    st.button("Add box", disabled=len(rows) >= MAX_BOXES, on_click=edit_boxes, args=("add",), key="module_comparison_add")
    st.caption(f"{len(rows)}/{MAX_BOXES} boxes. Expand a box to change its source, score, component, transformation, edge subset, colors or X-axis tick label.")
    valid = len(boxes) == len(rows)
    # Manifest revision invalidates compact caches when an incremental release changes data.
    regional_manifest = R.load_regional_manifest() if any(entry["regional"] for entry in registry.values()) else {}
    revision = hashlib.sha256(json.dumps([data_manifest, regional_manifest], sort_keys=True).encode()).hexdigest()
    context = (outcome, subset, min_n, revision)
    if st.button("Calculate comparison", type="primary", disabled=not valid, key="module_comparison_calculate"):
        catalogs = {}
        try:
            with st.spinner("Calculating pooled correlations from narrow, serialized score batches…"):
                for _, _, spec in boxes:
                    if spec.key not in catalogs:
                        table = calculate_series(spec, outcome, subset, min_n, revision)
                        if spec.is_regional:
                            expected = {
                                record["tissue"]: int(record["modules"])
                                for record in registry[spec.source]["combinations"]
                                if record["cohort_scope"] == spec.cohort_scope
                                and record["score_variant"] == spec.feature
                                and record["network_method"] == spec.method
                                and record["tissue"] in spec.tissues
                            }
                            observed = table.groupby("tissue")["module_key"].nunique().to_dict()
                            if observed != expected:
                                raise ValueError("Regional module counts do not reconcile with the completion manifest")
                        else:
                            expected_n = registry[spec.source]["modules"]
                            expected_components = ("CT", "TS") if spec.combines_ct_ts else (spec.component,)
                            observed = table.groupby("component")["module_key"].nunique().to_dict()
                            expected = {component: expected_n for component in expected_components}
                            if observed != expected or table.duplicated(["module_key", "component"]).any():
                                raise ValueError("Module/component counts do not reconcile with the selected partition manifest")
                        catalogs[spec.key] = table
            # Atomic replacement: do not expose partial BH families on an I/O error.
            st.session_state[RESULT_STATE] = {"context": context, "catalogs": catalogs}
        except (ValueError, FileNotFoundError, OSError, RuntimeError) as error:
            st.error(f"Comparison was not replaced because a score source could not be read or validated: {error}")
            return
    result = st.session_state.get(RESULT_STATE)
    if not valid or result is None or result["context"] != context or any(spec.key not in result["catalogs"] for _, _, spec in boxes):
        st.info("Choose completed score combinations and click Calculate comparison. Colors, labels, order, correlation method, K and display filters reuse the calculated compact statistics.")
        return
    displayed, complete, summary = display_results(result["catalogs"], boxes, method, int(k) if selection == "Top K" else None, fdr)
    for table in (displayed, complete, summary):
        table["source_label"] = table["source"].map(lambda value: registry[value]["label"])
    summary["cohort_label"] = [_cohort_label(spec, registry[spec.source]) for _, _, spec in boxes]
    if selection == "Top K":
        st.warning("Top K is selected using this outcome. These are outcome-selected descriptive distributions, not independent evidence of predictive superiority.")
    if any(spec.differential_edge_rule != "all" for _, _, spec in boxes):
        st.warning("Edge masks were selected using discovery AD–Control differences. Discovery/all-donor associations can be selection-biased; held-out validation is an exploratory check, not external validation.")
    st.caption("BH FDR uses all valid modules before display filtering, separately per component and regional tissue partition. Pooling regions or CT + TS does not combine FDR families. No between-box significance tests are performed.")
    if displayed.empty:
        st.info("No modules meet the current eligibility and FDR display filters. Summary counts and complete calculated results remain available.")
    else:
        box_colors = {row["uid"]: row["box_color"] for row in updated}
        dot_colors = {
            row["uid"]: row["box_color"] if row["dots_match_box"] else row["dot_color"]
            for row in updated
        }
        figure = comparison_figure(
            displayed, summary, outcome, method, points,
            box_colors=box_colors, dot_colors=dot_colors,
            tick_labels={row["uid"]: row["x_tick_label"].strip() for row in updated if row["x_tick_label"].strip()},
            show_module_counts={row["uid"]: row["show_module_count"] for row in updated},
        )
        render_exportable_plot(figure, key="module_outcome_comparison_plot")
    filterable_dataframe(summary, table_key="module_comparison_summary", table_name="Box summaries", hide_index=True)
    choice = st.radio("Module table", ["Displayed modules", "Complete calculated results"], horizontal=True, key="module_comparison_table_choice")
    table = displayed if choice == "Displayed modules" else complete
    filterable_dataframe(table, table_key="module_comparison_modules", table_name="Module correlation results", hide_index=True)
    downloads = st.columns(2)
    downloads[0].download_button("Download displayed modules (TSV)", D.dataframe_to_tsv_bytes(displayed), file_name="module_comparison_displayed.tsv", mime="text/tab-separated-values", key="module_comparison_download_displayed")
    downloads[1].download_button("Download complete calculated results (TSV)", D.dataframe_to_tsv_bytes(complete), file_name="module_comparison_complete.tsv", mime="text/tab-separated-values", key="module_comparison_download_complete")
