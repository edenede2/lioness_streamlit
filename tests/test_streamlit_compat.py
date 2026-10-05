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


def test_stateful_tabs_enable_active_tab_reruns(monkeypatch) -> None:
    calls = []
    tabs = (object(), object())

    def current_tabs(labels, **kwargs):
        calls.append((labels, kwargs))
        return tabs

    monkeypatch.setattr(streamlit_compat.st, "tabs", current_tabs)
    assert streamlit_compat.stateful_tabs(["One", "Two"], key="test_tabs") == tabs
    assert calls == [
        (["One", "Two"], {"key": "test_tabs", "on_change": "rerun"})
    ]


def test_stateful_tabs_fall_back_on_old_streamlit(monkeypatch) -> None:
    calls = []
    tabs = (object(),)

    def legacy_tabs(labels, **kwargs):
        calls.append((labels, kwargs))
        if kwargs:
            raise TypeError("unexpected keyword argument")
        return tabs

    monkeypatch.setattr(streamlit_compat.st, "tabs", legacy_tabs)
    assert streamlit_compat.stateful_tabs(["One"], key="test_tabs") == tabs
    assert calls == [
        (["One"], {"key": "test_tabs", "on_change": "rerun"}),
        (["One"], {}),
    ]


def test_tab_is_open_defaults_true_for_legacy_container() -> None:
    class CurrentTab:
        open = False

    assert streamlit_compat.tab_is_open(object()) is True
    assert streamlit_compat.tab_is_open(CurrentTab()) is False
