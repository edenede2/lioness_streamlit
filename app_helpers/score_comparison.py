"""Descriptive score–outcome comparisons with module/component observations.

Only compact correlation tables are cached. Donor scores are projected to one
scale and scanned in bounded batches under the app's shared analysis lock. Box
labels, ordering, Top K and display filters are deliberately not model identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import html
import json
from pathlib import Path
import textwrap
from typing import Iterator, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import streamlit as st

from app_helpers import data as D
from app_helpers.correlations import (
    add_across_module_fdr, calculate_correlations,
    add_categorical_across_module_fdr, calculate_categorical_associations,
)
from app_helpers.drive_data import data_path_available, ensure_data_path
from app_helpers import regional_modules as R
from app_helpers.streaming_associations import _serialized_heavy_read


MAX_BOXES = 12
COMPARISON_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
COMBINED_COMPONENT = "CT_TS"
COMPONENT_LABELS = {**D.EDGE_SCOPE_LABELS, COMBINED_COMPONENT: "CT + TS together"}
SCALE_ORDER = ("raw", "asinh", "rint")
CORRELATION_LABELS = {"spearman": "Spearman", "pearson": "Pearson"}
SPLIT_VALUES = {
    "discovery_ad_control": "Discovery",
    "validation_ad_control": "Validation",
    "mci_external": "MCI_external",
}


@dataclass(frozen=True)
class ComparisonSeriesSpec:
    source: str
    feature: str = "connectivity"
    method: str = "control_anchored"
    component: str = "CT"
    scale: str = "raw"
    cohort_scope: str = "complete_450"
    tissues: tuple[str, ...] = ()
    differential_edge_rule: str = "all"
    differential_fdr_scope: str = "per_module"
    differential_fdr_threshold: float = 0.05
    score_normalization: str = "standard_pruned"

    def __post_init__(self) -> None:
        if self.scale not in SCALE_ORDER:
            raise ValueError("Unsupported score scale")
        if self.feature not in {*D.FEATURE_LABELS, "eigengene"}:
            raise ValueError("Unsupported comparison feature")
        if self.feature == "eigengene" and self.differential_edge_rule != "all":
            raise ValueError("Eigengenes cannot be edge-filtered")
        if self.is_regional:
            if self.feature not in {"connectivity", "eigengene"}:
                raise ValueError("Regional sources provide connectivity or eigengenes")
            if not self.tissues or len(set(self.tissues)) != len(self.tissues):
                raise ValueError("Select distinct regional tissues")
            if not set(self.tissues).issubset(R.regional_tissues(self.source)):
                raise ValueError("Unknown regional tissue")
            if self.differential_edge_rule != "all":
                raise ValueError("Regional scores are all-edge only")
        elif self.source not in D.MODULE_SET_LABELS:
            raise ValueError("Unknown module definition")
        elif self.component not in {"CT", "TS", COMBINED_COMPONENT, *D.module_set_components(self.source)}:
            raise ValueError("Component does not belong to this module definition")
        if self.feature == "eigengene" and not self.is_regional and not self.component.startswith("TS_"):
            raise ValueError("Saved multi-tissue eigengenes are tissue-specific")

    @property
    def is_regional(self) -> bool:
        return self.source in R.REGIONAL_SOURCE_ORDER

    @property
    def combines_ct_ts(self) -> bool:
        return not self.is_regional and self.component == COMBINED_COMPONENT

    @property
    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:20]


def source_registry(data_manifest: dict, regional_manifest: dict | None = None) -> dict[str, dict]:
    """Only registered, completed sources; availability checks do not load scores."""
    result = {}
    for source in ("full_cohort", "control_derived", *D.MUSCLE_MODULE_SETS):
        entry = data_manifest.get("module_sets", {}).get(source)
        if entry is None or entry.get("status", "complete") != "complete":
            continue
        if not data_path_available(D.module_set_path("module_details.tsv", source)):
            continue
        donors = entry.get("donors_per_method", entry.get("donors", 450))
        result[source] = {
            "label": D.MODULE_SET_LABELS[source], "regional": False,
            "methods": tuple(entry.get("methods", D.MODULE_SET_METHODS[source])),
            "modules": int(entry["modules"]),
            "donors": donors,
        }
    if regional_manifest is None and data_path_available(R.REGIONAL_MANIFEST):
        regional_manifest = R.load_regional_manifest()
    if regional_manifest:
        completed = R.completed_combinations(regional_manifest)
        for source in R.REGIONAL_SOURCE_ORDER:
            if completed.empty:
                continue
            rows = completed.loc[completed["partition_source"].eq(source)]
            if rows.empty:
                continue
            result[source] = {
                "label": regional_manifest.get("source_labels", {}).get(source, source),
                "regional": True, "combinations": rows.to_dict("records"),
            }
    return result


@st.cache_data(show_spinner=False, max_entries=4, ttl=1800)
def effect_settings(source: str) -> dict:
    """Cache small settings, not the much larger driver-file manifest."""
    manifest = D.load_effect_size_manifest(source)
    return {key: manifest[key] for key in (
        "status", "completed_networks", "available_effect_statistics",
        "available_cutoff_modes", "available_fixed_hedges_g",
        "available_module_top_fractions", "available_directions",
        "available_score_normalizations",
    ) if key in manifest}


def score_path(spec: ComparisonSeriesSpec):
    if spec.is_regional:
        raise ValueError("A regional series has one score path per tissue")
    if spec.feature == "eigengene":
        return D.module_set_path("module_eigengenes.parquet", spec.source)
    filename = "aggregate_plot_data.parquet" if spec.component in {"CT", "TS", COMBINED_COMPONENT} else "resolved_plot_data.parquet"
    return D.differential_estimator_path(
        filename, spec.source, "lioness", spec.method, "all",
        spec.differential_edge_rule, spec.differential_fdr_scope,
        spec.differential_fdr_threshold, spec.score_normalization,
    )


def availability_reason(spec: ComparisonSeriesSpec, registry: dict) -> str:
    entry = registry.get(spec.source)
    if entry is None:
        return "This partition has no completed public catalog."
    if spec.is_regional:
        for tissue in spec.tissues:
            matches = [row for row in entry["combinations"] if (
                row["cohort_scope"] == spec.cohort_scope
                and row["score_variant"] == spec.feature
                and row["network_method"] == spec.method
                and row["tissue"] == tissue
            )]
            if not matches or not data_path_available(R.regional_score_path(
                spec.source, spec.cohort_scope, spec.feature, spec.method, tissue
            )):
                return f"{tissue}: this source/method/cohort combination is unfinished or unavailable."
    else:
        if spec.feature != "eigengene" and spec.method not in entry["methods"]:
            return "This network method is unavailable for the selected partition."
        if spec.cohort_scope == "incomplete_region_validation":
            from app_helpers import camel_validation as V
            try:
                V.validate_identity(spec.source,"lioness",spec.method,spec.feature,spec.differential_edge_rule)
            except ValueError as error:
                return str(error)
            if spec.component in {"CT","TS",COMBINED_COMPONENT}:
                return "Validation donors do not have pooled three-region scores. Select one resolved component."
            return ""
        if spec.cohort_scope != "complete_450":
            return "No expanded cohort is registered for this multi-tissue source."
        if not data_path_available(score_path(spec)):
            return "The exact score/edge-filter combination is unavailable; no substitute will be used."
    return ""


def _metadata(spec: ComparisonSeriesSpec, outcome: str, analysis_subset: str, tissue: str = "") -> pd.DataFrame:
    if spec.is_regional and spec.cohort_scope == "maximum_tissue":
        path = R.REGIONAL_ROOT / "metadata" / spec.cohort_scope / f"{tissue}.parquet"
        columns = ["sample_id", outcome]
        if analysis_subset != "all_donors":
            columns.append("ad_control_split")
        result = (R.regional_metadata(spec.source, spec.cohort_scope, tissue)[columns].copy()
                  if outcome == "study" else pq.read_table(ensure_data_path(path), columns=columns).to_pandas())
    else:
        source = (
            "muscle_control_l4" if spec.source == R.MOTOR_SOURCE
            else "full_cohort" if spec.is_regional else spec.source
        )
        # This metadata table is small (357–450 rows); never load expression here.
        result = (D.load_sample_metadata(source,cohort_scope=spec.cohort_scope)
                  if spec.cohort_scope == "incomplete_region_validation" else D.load_sample_metadata(source))
        columns = ["sample_id", outcome]
        if analysis_subset != "all_donors":
            columns.append("ad_control_split")
        missing = set(columns).difference(result.columns)
        if missing:
            raise ValueError(f"Required outcome/subset metadata are unavailable: {sorted(missing)}")
        result = result[columns].copy()
    if result["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs in score-comparison metadata")
    if analysis_subset != "all_donors":
        result = result.loc[result["ad_control_split"].eq(SPLIT_VALUES[analysis_subset])]
    return result[["sample_id", outcome]]


def _row_group_matches(parquet: pq.ParquetFile, group: int, filters: list[tuple]) -> bool:
    names = parquet.schema_arrow.names
    for column, operator, value in filters:
        if operator != "=":
            raise ValueError("Comparison scanner supports equality predicates only")
        stats = parquet.metadata.row_group(group).column(names.index(column)).statistics
        if stats is not None and stats.has_min_max and not (stats.min <= value <= stats.max):
            return False
    return True


def _scan_score_vectors(path, filters: list[tuple], value: str | tuple[str, ...], modules: list[int]) -> Iterator[tuple[int, pd.DataFrame]]:
    """Scan a single family once, without whole-table decoding or threaded arenas.

    Only the selected scalar vectors survive each 8,192-row batch. They are small
    (at most CT + TS × one feature × scale), uncached, and released as
    each module's statistics are calculated. This avoids repeatedly decoding the
    muscle catalogs that have all modules in a single large Parquet row group.
    """
    materialized = ensure_data_path(path, filters)
    paths = materialized if isinstance(materialized, list) else [materialized]
    files = []
    for item in paths:
        candidate = Path(item)
        files.extend(sorted(candidate.rglob("*.parquet")) if candidate.is_dir() else [candidate])
    if not files:
        raise FileNotFoundError("No score Parquet files are available for this configuration")
    values = [value] if isinstance(value, str) else list(value)
    columns = list(dict.fromkeys(["sample_id", "module", *values, *(column for column, _, _ in filters)]))
    pieces: dict[int, list[pd.DataFrame]] = {}
    for file in files:
        with pq.ParquetFile(file) as parquet:
            row_groups = [group for group in range(parquet.num_row_groups) if _row_group_matches(parquet, group, filters)]
            for batch in parquet.iter_batches(batch_size=8192, row_groups=row_groups, columns=columns, use_threads=False):
                table = pa.Table.from_batches([batch])
                mask = pc.is_in(table["module"], value_set=pa.array(modules, type=table["module"].type))
                for column, _, expected in filters:
                    mask = pc.and_(mask, pc.equal(table[column], expected))
                selected = table.filter(mask).select(["sample_id", "module", *values]).to_pandas()
                for module, part in selected.groupby("module", observed=True, sort=False):
                    vector = part[["sample_id", *values]]
                    if isinstance(value, str):
                        vector = vector.rename(columns={value: "metric_value"})
                    pieces.setdefault(int(module), []).append(vector)
    for module in modules:
        parts = pieces.pop(int(module), [])
        yield int(module), pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _module_frames(spec: ComparisonSeriesSpec) -> Iterator[tuple[int, str, pd.DataFrame]]:
    """Yield module, regional tissue/combined component, and narrow score vector."""
    if spec.cohort_scope == "incomplete_region_validation":
        from app_helpers import camel_validation as V
        source=V.for_dataset(spec.source,spec.method)
        for module in sorted(source.manifest()["completed_module_ids"]):
            frame=source.scores(module,spec.feature,spec.differential_edge_rule,spec.score_normalization,
                           spec.component,spec.scale,False)
            frame=frame[["sample_id",f"metric_{spec.scale}",*[c for c in
                ("node_convention","n_features","n_source_assignment_rows","cross_filter_node_convention_warning") if c in frame]]].rename(columns={f"metric_{spec.scale}":"metric_value"})
            yield int(module),"",frame
        return
    if spec.is_regional:
        details = R.load_regional_details()
        for tissue in spec.tissues:
            modules = sorted(details.loc[
                details["partition_source"].eq(spec.source) & details["tissue"].eq(tissue),
                "module",
            ].astype(int).unique())
            path = R.regional_score_path(spec.source, spec.cohort_scope, spec.feature, spec.method, tissue)
            for module, frame in _scan_score_vectors(path, [], f"metric_{spec.scale}", list(modules)):
                yield int(module), tissue, frame
        return
    modules = sorted(D.load_module_details(module_set=spec.source)["module"].astype(int).unique())
    aggregate = spec.component in {"CT", "TS", COMBINED_COMPONENT}
    value = (
        (f"CT_{spec.scale}", f"TS_{spec.scale}") if spec.combines_ct_ts
        else f"{spec.component}_{spec.scale}" if aggregate else f"metric_{spec.scale}"
    )
    filters = [("metric_family", "=", spec.feature)]
    if spec.feature != "eigengene":
        filters.append(("lioness_method", "=", spec.method))
    if not aggregate:
        filters.append(("component", "=", spec.component))
    for module, frame in _scan_score_vectors(score_path(spec), filters, value, list(modules)):
        if spec.combines_ct_ts:
            for component in ("CT", "TS"):
                vector = frame if frame.empty else frame[["sample_id", f"{component}_{spec.scale}"]].rename(
                    columns={f"{component}_{spec.scale}": "metric_value"},
                )
                yield int(module), component, vector
        else:
            yield int(module), "", frame


def _module_statistic(frame: pd.DataFrame, metadata: pd.DataFrame, outcome: str, min_n: int) -> dict:
    if outcome == "study":
        if "sample_id" in frame and frame["sample_id"].duplicated().any():
            raise ValueError("More than one score per donor/module/component")
        joined = frame.merge(metadata, on="sample_id", how="inner", validate="one_to_one") if not frame.empty else pd.DataFrame(columns=["metric_value", outcome])
        joined["_test"] = "pooled"
        if joined.empty:
            row = dict(n=0, eligible=False, unavailable_reason="no saved score / observed study", epsilon_squared=np.nan, categorical_p=np.nan)
        else:
            row = calculate_categorical_associations(joined, ["_test"], category_column=outcome, min_group_n=min_n).iloc[0].drop(labels=["_test", "outcome"]).to_dict()
        return {**row, "minimum_group_n": min_n, "pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan}
    if frame.empty:
        return dict(n=0, eligible=False, minimum_group_n=min_n,
                    unavailable_reason="no saved score / structurally unavailable",
                    pearson_r=np.nan, pearson_p=np.nan, spearman_rho=np.nan, spearman_p=np.nan)
    if frame["sample_id"].duplicated().any():
        raise ValueError("More than one score per donor/module/component")
    joined = frame.merge(metadata, on="sample_id", how="inner", validate="one_to_one")
    if joined.empty:
        return dict(n=0, eligible=False, minimum_group_n=min_n,
                    unavailable_reason="no donors with this outcome/evaluation subset",
                    pearson_r=np.nan, pearson_p=np.nan, spearman_rho=np.nan, spearman_p=np.nan)
    joined["_test"] = "pooled"
    row = calculate_correlations(joined, ["_test"], [outcome], min_group_n=min_n).iloc[0]
    return row.drop(labels=["_test", "outcome", "pearson_fdr_displayed_family", "spearman_fdr_displayed_family"]).to_dict()


def statistics_from_module_frames(
    spec: ComparisonSeriesSpec, frames: Iterator[tuple[int, str, pd.DataFrame]],
    metadata_by_tissue: dict[str, pd.DataFrame], outcome: str, min_n: int,
    analysis_subset: str = "all_donors",
) -> pd.DataFrame:
    """Normalize rows then correct each complete valid module family, never Top K."""
    rows = []
    for module, scope, frame in frames:
        tissue = scope if spec.is_regional else ""
        component = scope if spec.is_regional or spec.combines_ct_ts else spec.component
        if spec.combines_ct_ts and component not in {"CT", "TS"}:
            raise ValueError("Combined CT + TS frames must identify their component")
        module_key = f"{spec.source}:{tissue + ':' if tissue else ''}M{module}"
        rows.append({
            "series_key": spec.key, "source": spec.source, "module": module,
            "module_key": module_key, "tissue": tissue, "component": component,
            "observation_key": f"{module_key}:{component}",
            "component_label": tissue if spec.is_regional else D.EDGE_SCOPE_LABELS.get(component, component),
            "feature": spec.feature, "method": spec.method, "scale": spec.scale,
            "cohort_scope": spec.cohort_scope, "outcome": outcome,
            "analysis_subset": analysis_subset,
            "differential_edge_rule": spec.differential_edge_rule,
            "differential_fdr_scope": spec.differential_fdr_scope,
            "differential_fdr_threshold": spec.differential_fdr_threshold,
            "score_normalization": spec.score_normalization,
            **_module_statistic(frame, metadata_by_tissue[tissue], outcome, min_n),
        })
        for field in ("node_convention","n_features","n_source_assignment_rows","cross_filter_node_convention_warning"):
            if field in frame and not frame.empty:
                rows[-1][field]=frame[field].iloc[0]
        if spec.source=="full_cohort" and spec.feature!="eigengene":
            from app_helpers.validation_datasets import LEGACY_DUPLICATE_BRAIN_MODULES
            rows[-1]["cross_filter_node_convention_warning"]=int(module) in LEGACY_DUPLICATE_BRAIN_MODULES
            rows[-1]["node_convention"]=("legacy_unfiltered_unique_tissue_gene" if spec.differential_edge_rule=="all"
                                          else "legacy_effect_positional_assignment_rows")
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No module assignments are available for this series")
    result = add_across_module_fdr(
        result, family_columns=["series_key", "tissue", "component"], module_column="module_key",
    )
    if outcome == "study":
        result = add_categorical_across_module_fdr(
            result, family_columns=["series_key", "tissue", "component"], module_column="module_key",
        )
    return result


@st.cache_data(show_spinner=False, max_entries=24, ttl=1800)
@_serialized_heavy_read
def calculate_series(
    spec: ComparisonSeriesSpec, outcome: str, analysis_subset: str,
    min_n: int, data_revision: str,
) -> pd.DataFrame:
    """Cache compact module/component rows. Failed/partial families never cache."""
    if outcome not in {*D.OUTCOME_LABELS, "study"}:
        raise ValueError("Nominal category codes cannot be correlated")
    if analysis_subset not in D.ANALYSIS_SUBSET_LABELS:
        raise ValueError("Unknown evaluation subset")
    if min_n not in (5, 10, 20):
        raise ValueError("Minimum N must be 5, 10 or 20")
    tissues = spec.tissues if spec.is_regional else ("",)
    metadata = {tissue: _metadata(spec, outcome, analysis_subset, tissue) for tissue in tissues}
    return statistics_from_module_frames(spec, _module_frames(spec), metadata, outcome, min_n, analysis_subset)


def display_results(
    catalogs: dict[str, pd.DataFrame], boxes: list[tuple[str, str, ComparisonSeriesSpec]],
    correlation_method: str = "spearman", top_k: int | None = None,
    fdr_cutoff: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pure display operation: no reads, recomputation or FDR re-adjustment."""
    if correlation_method not in CORRELATION_LABELS or (top_k is not None and top_k <= 0):
        raise ValueError("Invalid correlation method or Top K")
    displayed, complete, summaries = [], [], []
    for position, (box_id, label, spec) in enumerate(boxes):
        frame = catalogs[spec.key].copy()
        nominal = frame["outcome"].eq("study").all()
        coefficient = "epsilon_squared" if nominal else ("spearman_rho" if correlation_method == "spearman" else "pearson_r")
        p_column = "categorical_p" if nominal else f"{correlation_method}_p"
        fdr_column = "categorical_fdr_across_modules" if nominal else f"{correlation_method}_fdr_across_modules"
        if frame.duplicated(["module_key", "component"]).any():
            raise ValueError("Duplicate module/component observations in comparison box")
        # Also normalize compact results retained in a session from the previous
        # single-component schema; display edits must not trigger donor reads.
        frame["observation_key"] = frame["module_key"] + ":" + frame["component"]
        frame["box_id"], frame["box_label"], frame["box_order"] = box_id, label, position
        frame["signed_correlation"] = np.nan if nominal else frame[coefficient]
        frame["absolute_correlation"] = np.nan if nominal else frame[coefficient].abs()
        frame["association_effect"] = frame[coefficient] if nominal else frame[coefficient].abs()
        frame["association_measure"] = "Kruskal epsilon-squared" if nominal else f"Absolute {correlation_method} correlation"
        frame["selected_p"], frame["selected_fdr"] = frame[p_column], frame[fdr_column]
        frame["correlation_method"] = "not_applicable" if nominal else correlation_method
        eligible = frame.loc[frame["eligible"] & np.isfinite(frame[coefficient])].copy()
        qualified = eligible if fdr_cutoff is None else eligible.loc[eligible[fdr_column].lt(fdr_cutoff)]
        selected = qualified.sort_values(
            ["association_effect", "module_key", "component"], ascending=[False, True, True], kind="stable",
        )
        if top_k is not None:
            # K always counts biological modules. In a CT + TS box, rank each
            # module by its strongest eligible component and retain both scores
            # when both pass the optional display FDR filter.
            top_modules = selected["module_key"].drop_duplicates().head(top_k)
            selected = selected.loc[selected["module_key"].isin(top_modules)]
        selected = selected.copy()
        selected["display_rank"] = np.arange(1, len(selected) + 1)
        frame["displayed"] = frame["observation_key"].isin(selected["observation_key"])
        complete.append(frame)
        displayed.append(selected)
        summaries.append({
            "box_id": box_id, "box_label": label, "source": spec.source,
            "feature": spec.feature, "method": spec.method, "scale": spec.scale,
            "cohort_scope": spec.cohort_scope, "differential_edge_rule": spec.differential_edge_rule,
            "differential_fdr_scope": spec.differential_fdr_scope,
            "differential_fdr_threshold": spec.differential_fdr_threshold,
            "score_normalization": spec.score_normalization,
            "component": ", ".join(spec.tissues) if spec.is_regional else spec.component,
            "total_modules": frame["module_key"].nunique(), "eligible_modules": eligible["module_key"].nunique(),
            "invalid_modules": frame["module_key"].nunique() - eligible["module_key"].nunique(),
            "fdr_qualified_modules": qualified["module_key"].nunique(),
            "plotted_modules": selected["module_key"].nunique(),
            "excluded_from_plot": frame["module_key"].nunique() - selected["module_key"].nunique(),
            "total_score_observations": len(frame), "eligible_score_observations": len(eligible),
            "plotted_score_observations": len(selected), "excluded_score_observations": len(frame) - len(selected),
            "eligible_donor_n_min": eligible["n"].min(), "eligible_donor_n_max": eligible["n"].max(),
            "plotted_donor_n_min": selected["n"].min(), "plotted_donor_n_max": selected["n"].max(),
            "median_absolute_correlation": np.nan if nominal else selected["absolute_correlation"].median(),
            "median_association_effect": selected["association_effect"].median(),
            "q1_absolute_correlation": np.nan if nominal else selected["absolute_correlation"].quantile(.25),
            "q3_absolute_correlation": np.nan if nominal else selected["absolute_correlation"].quantile(.75),
            "q1_association_effect": selected["association_effect"].quantile(.25),
            "q3_association_effect": selected["association_effect"].quantile(.75),
            "requested_top_k": top_k, "association_fdr_cutoff": fdr_cutoff,
        })
    if not boxes:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return pd.concat(displayed, ignore_index=True), pd.concat(complete, ignore_index=True), pd.DataFrame(summaries)


def comparison_figure(
    displayed: pd.DataFrame, summary: pd.DataFrame, outcome: str, method: str,
    show_points: bool = True, *, box_colors: Mapping[str, str] | None = None,
    dot_colors: Mapping[str, str] | None = None,
    tick_labels: Mapping[str, str] | None = None,
    show_module_counts: Mapping[str, bool] | None = None,
) -> go.Figure:
    """Boxplot values are module/component correlations, never donor scores."""
    box_colors, dot_colors = box_colors or {}, dot_colors or {}
    tick_labels, show_module_counts = tick_labels or {}, show_module_counts or {}
    figure = go.Figure()
    tick_values, tick_text = [], []
    nominal = outcome == "study"
    effect_column = "association_effect" if "association_effect" in displayed else "absolute_correlation"
    for index, row in summary.reset_index(drop=True).iterrows():
        box_id = row["box_id"]
        tick_values.append(index)
        # Explicit line breaks keep labels legible in narrow layouts and exports.
        wrap_width = 10 if len(summary) > 6 else 22
        count_label = f"{row['plotted_modules']}<br>modules" if len(summary) > 6 else f"{row['plotted_modules']} modules"
        if row["component"] == COMBINED_COMPONENT:
            suffix = "CT/TS<br>scores" if len(summary) > 6 else "CT/TS scores"
            count_label += f"<br>{row['plotted_score_observations']} {suffix}"
        label = str(tick_labels.get(box_id, row["box_label"]))
        # Accept plain text, including explicit line breaks; never reinterpret
        # custom labels as Plotly HTML. Box names/identities and data stay intact.
        lines = [html.escape(chunk) for line in label.split("\n")
                 for chunk in (textwrap.wrap(line, wrap_width) or [""])]
        tick_text.append("<br>".join(lines) + (
            f"<br>{count_label}" if show_module_counts.get(box_id, True) else ""
        ))
        part = displayed.loc[displayed["box_id"].eq(box_id)]
        color = box_colors.get(box_id, COMPARISON_COLORS[index % len(COMPARISON_COLORS)])
        figure.add_trace(go.Box(
            x=[index] * len(part), y=part[effect_column], name=row["box_label"], width=.5,
            marker_color=color, line_color=color, boxpoints=False, showlegend=False,
            hoverinfo="y", quartilemethod="linear",
        ))
        if show_points and not part.empty:
            # Use deterministic jitter; labels/order changes cannot change observations.
            jitter = np.array([int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / (2**32 - 1) for key in part["observation_key"]])
            source_column = "source_label" if "source_label" in part else "source"
            custom = part[[source_column, "module_key", "component_label", effect_column if nominal else "signed_correlation", "n", "selected_p", "selected_fdr"]].to_numpy()
            figure.add_trace(go.Scatter(
                x=index + (jitter - .5) * .36, y=part[effect_column], mode="markers",
                marker=dict(color=dot_colors.get(box_id, color), size=6, opacity=.65), showlegend=False,
                customdata=custom, name=row["box_label"],
                hovertemplate=("%{customdata[1]}<br>Partition: %{customdata[0]}<br>Component: %{customdata[2]}"
                               + ("<br>Kruskal epsilon-squared: %{y:.4f}" if nominal else "<br>Signed correlation: %{customdata[3]:.4f}<br>|correlation|: %{y:.4f}") +
                               "<br>Donors: %{customdata[4]}<br>p=%{customdata[5]:.3g}"
                               "<br>Module-set FDR=%{customdata[6]:.3g}<extra></extra>"),
            ))
    figure.update_layout(
        template="plotly_white", title=f"Module associations with {D.ASSOCIATION_OUTCOME_LABELS.get(outcome, outcome)}",
        yaxis=dict(title="Kruskal epsilon-squared (ROS vs MAP)" if nominal else "|Spearman ρ|" if method == "spearman" else "|Pearson r|", range=[0, 1]),
        xaxis=dict(title="Score / partition combination", type="linear", range=[-.6, len(summary) - .4],
                   tickmode="array", tickvals=tick_values, ticktext=tick_text, automargin=True,
                   tickfont=dict(size=12 if len(summary) > 6 else 14)),
        height=740 if len(summary) > 6 else 640, font=dict(size=15),
        margin=dict(t=75, b=210 if len(summary) > 6 else 150, l=65, r=25),
    )
    return figure
