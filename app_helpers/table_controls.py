"""Compact, consistent column controls for Streamlit dataframes."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd
import streamlit as st


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
    default_columns: Iterable[object] | None = None,
    column_config: Mapping[object, object] | None = None,
    **dataframe_kwargs: Any,
) -> object | None:
    """Render a dataframe with a compact per-table visible-column picker."""

    columns = list(frame.columns)
    if not columns:
        st.info(f"{table_name} has no columns to display.")
        return None
    requested_defaults = list(default_columns) if default_columns is not None else columns
    defaults = [column for column in requested_defaults if column in columns]
    if not defaults:
        defaults = columns
    config = dict(column_config or {})
    schema = "\x1f".join(map(str, columns)).encode("utf-8")
    schema_digest = hashlib.sha1(schema, usedforsecurity=False).hexdigest()[:12]
    widget_key = f"table_columns__{table_key}__{schema_digest}"

    with st.popover("Choose table columns", icon=":material/view_column:"):
        st.caption(table_name)
        if st.button(
            "Reset to all columns",
            key=f"{widget_key}__reset",
            use_container_width=True,
        ):
            st.session_state[widget_key] = columns
        multiselect_arguments: dict[str, Any] = {
            "options": columns,
            "format_func": lambda column: _column_label(column, config),
            "key": widget_key,
            "help": "Search, add, or remove the columns shown below.",
        }
        if widget_key not in st.session_state:
            multiselect_arguments["default"] = defaults
        selected = st.multiselect(
            f"{table_name} columns",
            **multiselect_arguments,
        )
        st.caption(f"Showing {len(selected)} of {len(columns)} columns")
        st.caption("Column choices affect the visible table; downloads retain their full schema.")

    if not selected:
        st.warning(f"Select at least one column for {table_name}.")
        return None
    selected_config = {
        column: configuration
        for column, configuration in config.items()
        if column in selected
    }
    return st.dataframe(
        frame.loc[:, selected],
        column_config=selected_config or None,
        **dataframe_kwargs,
    )
