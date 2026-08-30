from __future__ import annotations

from app_helpers import streamlit_compat


def test_plotly_width_adapter_uses_legacy_argument_on_old_streamlit(monkeypatch) -> None:
    calls = []

    def legacy_plotly_chart(figure, use_container_width=True, **kwargs):
        calls.append((figure, use_container_width, kwargs))
        return "legacy"

    monkeypatch.setattr(streamlit_compat.st, "plotly_chart", legacy_plotly_chart)
    streamlit_compat.uses_modern_width_api.cache_clear()
    assert streamlit_compat.plotly_chart("figure", use_container_width=True) == "legacy"
    assert calls == [("figure", True, {})]
    streamlit_compat.uses_modern_width_api.cache_clear()


def test_plotly_width_adapter_uses_stretch_on_current_streamlit(monkeypatch) -> None:
    calls = []

    def current_plotly_chart(figure, width="content", **kwargs):
        calls.append((figure, width, kwargs))
        return "current"

    monkeypatch.setattr(streamlit_compat.st, "plotly_chart", current_plotly_chart)
    streamlit_compat.uses_modern_width_api.cache_clear()
    assert streamlit_compat.plotly_chart("figure", use_container_width=True) == "current"
    assert calls == [("figure", "stretch", {})]
    assert streamlit_compat.normalize_width_kwargs(
        {"use_container_width": False, "height": 200}
    ) == {"width": "content", "height": 200}
    streamlit_compat.uses_modern_width_api.cache_clear()
