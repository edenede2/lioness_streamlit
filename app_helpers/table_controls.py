"""Compact, type-aware row filters for Streamlit dataframes."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import pandas as pd
import streamlit as st

from app_helpers.streamlit_compat import normalize_width_kwargs, stretch_width_kwargs


def _column_label(column: object, column_config: Mapping[object, object]) -> str:
    configured = column_config.get(column)
    if isinstance(configured, str):
        return configured
    if isinstance(configured, Mapping):
        label = configured.get("label")
        if label:
            return str(label)
    return str(column)


def filterable_dataframe(
    frame: pd.DataFrame,
    *,
    table_key: str,
    table_name: str,
    column_config: Mapping[object, object] | None = None,
    **dataframe_kwargs: Any,
) -> object | None:
    """Render a dataframe with compact, type-aware filters on column values."""

    columns = list(frame.columns)
    if not columns:
        st.info(f"{table_name} has no columns to display.")
        return None
    config = dict(column_config or {})
    schema = "\x1f".join(map(str, columns)).encode("utf-8")
    schema_digest = hashlib.sha1(schema, usedforsecurity=False).hexdigest()[:12]
    widget_prefix = f"table_filters__{table_key}__{schema_digest}"
    columns_key = f"{widget_prefix}__columns"
    reset_key = f"{widget_prefix}__reset"

    filtered = frame.copy()
    with st.popover("Filter table rows", icon=":material/filter_alt:"):
        st.caption(table_name)
        if st.button(
            "Reset row filters",
            key=reset_key,
            **stretch_width_kwargs(),
        ):
            for state_key in list(st.session_state):
                if str(state_key).startswith(widget_prefix) and state_key != reset_key:
                    del st.session_state[state_key]
        filter_columns = st.multiselect(
            f"{table_name} filter columns",
            options=columns,
            format_func=lambda column: _column_label(column, config),
            key=columns_key,
            help="Choose one or more columns, then set their value filters below.",
        )
        for column in filter_columns:
            label = _column_label(column, config)
            series = frame[column]
            column_digest = hashlib.sha1(
                str(column).encode("utf-8"), usedforsecurity=False
            ).hexdigest()[:10]
            column_prefix = f"{widget_prefix}__{column_digest}"
            st.markdown(f"**{label}**")
            missing_policy = "Keep missing"
            if series.isna().any():
                missing_policy = st.selectbox(
                    f"{label} missing values",
                    options=["Keep missing", "Exclude missing", "Only missing"],
                    key=f"{column_prefix}__missing",
                )

            if missing_policy == "Only missing":
                column_mask = series.isna()
            elif pd.api.types.is_bool_dtype(series) or isinstance(
                series.dtype, pd.CategoricalDtype
            ):
                options = sorted(series.dropna().unique().tolist(), key=str)
                selected_values = st.multiselect(
                    f"{label} values",
                    options=options,
                    key=f"{column_prefix}__values",
                    help="Leave empty to keep every non-missing value.",
                )
                column_mask = (
                    series.isin(selected_values)
                    if selected_values
                    else pd.Series(True, index=series.index)
                )
            elif pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce").replace(
                    [float("inf"), float("-inf")], pd.NA
                )
                valid = numeric.dropna()
                if valid.empty:
                    st.caption("No finite numeric values are available.")
                    column_mask = pd.Series(True, index=series.index)
                else:
                    minimum = valid.min()
                    maximum = valid.max()
                    if minimum == maximum:
                        st.caption(f"Only one numeric value: {minimum:g}")
                        column_mask = numeric.eq(minimum)
                    else:
                        integer_values = pd.api.types.is_integer_dtype(series)
                        lower_default = int(minimum) if integer_values else float(minimum)
                        upper_default = int(maximum) if integer_values else float(maximum)
                        range_columns = st.columns(2)
                        lower = range_columns[0].number_input(
                            f"{label} minimum",
                            value=lower_default,
                            step=1 if integer_values else None,
                            key=f"{column_prefix}__minimum",
                        )
                        upper = range_columns[1].number_input(
                            f"{label} maximum",
                            value=upper_default,
                            step=1 if integer_values else None,
                            key=f"{column_prefix}__maximum",
                        )
                        if lower > upper:
                            st.warning(f"{label}: minimum exceeds maximum.")
                            column_mask = pd.Series(False, index=series.index)
                        else:
                            column_mask = numeric.between(lower, upper, inclusive="both")
            elif pd.api.types.is_datetime64_any_dtype(series):
                datetimes = pd.to_datetime(series, errors="coerce")
                valid = datetimes.dropna()
                if valid.empty:
                    st.caption("No valid dates are available.")
                    column_mask = pd.Series(True, index=series.index)
                else:
                    chosen_dates = st.date_input(
                        f"{label} date range",
                        value=(valid.min().date(), valid.max().date()),
                        min_value=valid.min().date(),
                        max_value=valid.max().date(),
                        key=f"{column_prefix}__dates",
                    )
                    if isinstance(chosen_dates, (tuple, list)) and len(chosen_dates) == 2:
                        start, end = map(pd.Timestamp, chosen_dates)
                        column_mask = datetimes.between(start, end, inclusive="both")
                    else:
                        column_mask = pd.Series(True, index=series.index)
            else:
                nonmissing = series.dropna().astype(str)
                unique_values = sorted(nonmissing.unique().tolist(), key=str)
                if len(unique_values) <= 100:
                    selected_values = st.multiselect(
                        f"{label} values",
                        options=unique_values,
                        key=f"{column_prefix}__values",
                        help="Leave empty to keep every non-missing value.",
                    )
                    column_mask = (
                        series.astype("string").isin(selected_values)
                        if selected_values
                        else pd.Series(True, index=series.index)
                    )
                else:
                    contains = st.text_input(
                        f"{label} contains",
                        key=f"{column_prefix}__contains",
                        help="Case-insensitive literal text match.",
                    )
                    column_mask = (
                        series.astype("string").str.contains(
                            contains, case=False, regex=False, na=False
                        )
                        if contains
                        else pd.Series(True, index=series.index)
                    )

            if missing_policy == "Keep missing":
                column_mask = column_mask.fillna(False) | series.isna()
            elif missing_policy == "Exclude missing":
                column_mask = column_mask.fillna(False) & series.notna()
            filtered = filtered.loc[column_mask.reindex(filtered.index, fill_value=False)]

        st.caption(f"Showing {len(filtered):,} of {len(frame):,} rows")
        st.caption("Filters affect the visible table; downloads retain their full row set.")

    return st.dataframe(
        filtered,
        column_config=config or None,
        **normalize_width_kwargs(dataframe_kwargs),
    )
