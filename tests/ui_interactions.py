"""Legacy regression playback explicitly submits the new Apply action.

Dedicated workspace tests cover unapplied state. This driver lets older
scientific assertions continue to exercise the same intentional selections.
"""


def apply_driver(app):
    original = app._run

    def run(widget_state=None, timeout=None, _submitted=False):
        # AppTest models expanders as blocks, not widgets. Real browsers send
        # their bool states with child-button events; preserve that state here.
        if widget_state is not None:
            from streamlit.runtime.state.common import user_key_from_element_id

            for panel in app.expander:
                identifier = panel.proto.id
                if not identifier:
                    continue
                key = user_key_from_element_id(identifier)
                try:
                    value = app.session_state[key]
                except KeyError:
                    value = panel.proto.expanded
                state = next(
                    (item for item in widget_state.widgets if item.id == identifier),
                    None,
                )
                if state is None:
                    state = widget_state.widgets.add(id=identifier)
                state.bool_value = bool(value)
        result = original(widget_state, timeout=timeout)
        pending = any(
            "Unapplied settings" in item.value or "Apply the new dataset" in item.value
            for item in result.info
        )
        if pending and not result.exception and not _submitted:
            button = next(
                (value for value in result.button if value.label == "Apply analysis"),
                None,
            )
            if button is not None:
                button.click()
                result = run(
                    result._tree.get_widget_states(), timeout=timeout, _submitted=True
                )
        return result

    app._run = run
    return app
