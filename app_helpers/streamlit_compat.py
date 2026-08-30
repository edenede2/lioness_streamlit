"""Small compatibility adapters for Streamlit's container-width API transition."""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any

import streamlit as st


@lru_cache(maxsize=1)
def uses_modern_width_api() -> bool:
    """Return whether charts use ``width=`` instead of ``use_container_width``."""

    return "width" in inspect.signature(st.plotly_chart).parameters


def stretch_width_kwargs(stretch: bool = True) -> dict[str, object]:
    """Return width arguments accepted by the active Streamlit runtime."""

    if uses_modern_width_api():
        return {"width": "stretch" if stretch else "content"}
    return {"use_container_width": bool(stretch)}


def normalize_width_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate legacy/current width options without emitting deprecation warnings."""

    result = dict(kwargs)
    if uses_modern_width_api():
        legacy = result.pop("use_container_width", None)
        if legacy is not None and "width" not in result:
            result["width"] = "stretch" if legacy else "content"
    elif "width" in result and isinstance(result["width"], str):
        result["use_container_width"] = result.pop("width") == "stretch"
    return result


def plotly_chart(
    figure: object,
    *,
    use_container_width: bool | None = None,
    width: str | int | None = None,
    **kwargs: Any,
) -> object:
    """Render Plotly using the non-deprecated width argument for this runtime."""

    width_kwargs: dict[str, Any] = {}
    if width is not None:
        width_kwargs["width"] = width
    elif use_container_width is not None:
        width_kwargs["use_container_width"] = use_container_width
    return st.plotly_chart(
        figure,
        **normalize_width_kwargs(width_kwargs),
        **kwargs,
    )
