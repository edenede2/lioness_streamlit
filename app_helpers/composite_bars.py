"""Equal-width delta segments for correlation composites, including exports.

Segment endpoints are original signed correlations, not additive correlations.
Explicit label identities let exports remove/recolor series without stale labels
or gaps left behind by an omitted segment.
"""
from __future__ import annotations

import numpy as np

LABEL_PREFIX = "score-composite-value-"


def composite_settings(trace):
    meta = trace.meta if isinstance(trace.meta, dict) else {}
    return meta.get("score_composite", {})


def _value(trace):
    value = composite_settings(trace)["value"]
    return np.nan if value is None else float(value)  # JSON snapshots encode NaN as null.


def refresh_composite_bars(figure, *, canvas_height=None, canvas_width=None, label_font=None):
    groups = {}
    for trace in figure.data:
        settings = composite_settings(trace)
        if trace.type == "bar" and settings:
            groups.setdefault((trace.xaxis or "x", trace.yaxis or "y", trace.x[0]), []).append(trace)
    figure.layout.annotations = tuple(a for a in figure.layout.annotations
        if not (a.name or "").startswith(LABEL_PREFIX))
    if not groups:
        return
    for (xaxis, yaxis, category), traces in groups.items():
        previous = {True: 0., False: 0.}
        ordered = sorted(traces, key=lambda t: (
            abs(_value(t)) if np.isfinite(_value(t)) else np.inf,
            composite_settings(t)["order"]))
        labels = []
        for trace in ordered:
            settings = composite_settings(trace)
            value = _value(trace)
            valid = np.isfinite(value)
            base = previous[value >= 0] if valid else 0.
            trace.base = base
            trace.y = [value - base if valid else np.nan]
            trace.width = 0.62
            trace.text = [""]
            if valid:
                previous[value >= 0] = value
                if settings.get("show_value", True):
                    labels.append((value, trace, settings))
        # A single value column with leader lines also handles ties and tiny
        # deltas; labels never imply that a segment's delta is the correlation.
        labels.sort(key=lambda item: (item[0], item[2]["order"]))
        if not labels:
            continue
        font = label_font or dict(size=12, color="#202124")
        height = canvas_height or figure.layout.height or 720
        margins = figure.layout.margin
        ylayout = figure.layout[yaxis[0] + "axis" + yaxis[1:]]
        domain = ylayout.domain or (0, 1)
        limits = ylayout.range or (-1.15, 1.15)
        panel_height = max(50, (height - (margins.t or 0) - (margins.b or 0)) * (domain[1] - domain[0]))
        desired_gap = (float(font["size"]) * 1.5 + 3) * abs(limits[1] - limits[0]) / panel_height
        gap = min(max(.12, desired_gap), 1.9 / max(1, len(labels) - 1))
        positions = np.array([np.clip(value, -1.05, 1.05) for value, _, _ in labels])
        for i in range(1, len(positions)):
            positions[i] = max(positions[i], positions[i - 1] + gap)
        positions[-1] = min(positions[-1], 1.05)
        for i in range(len(positions) - 2, -1, -1):
            positions[i] = min(positions[i], positions[i + 1] - gap)
        axis = figure.layout[xaxis[0] + "axis" + xaxis[1:]]
        categories = list(axis.categoryarray or dict.fromkeys(t.x[0] for t in figure.data if t.type == "bar"))
        column = categories.index(category)
        # Keep the default first-column labels away from the neighboring TS
        # column. Reserve actual space inside the axis, not over its title.
        side = -1 if column == 0 else 1
        if column in (0, len(categories)-1):
            xdomain = axis.domain or (0, 1)
            panel_width = max(60, ((canvas_width or figure.layout.width or 900) -
                (margins.l or 0) - (margins.r or 0)) * (xdomain[1]-xdomain[0]))
            label_width = max(len(settings["label"]) for _,_,settings in labels) * float(font["size"]) * .66 + 8
            padding = label_width * (len(categories)+1) / panel_width + .12
            axis.update(range=[-.5-padding if side<0 else -.5,
                               len(categories)-.5+padding if side>0 else len(categories)-.5],autorange=False)
        for label_y, (value, trace, settings) in zip(positions, labels):
            color = trace.marker.color
            if not isinstance(color, str):
                color = color[0]
            figure.add_annotation(name=f"{LABEL_PREFIX}{xaxis}-{settings['id']}",
                x=column + side*.31, y=value, xref=xaxis, yref=yaxis,
                ax=column + side*.40, ay=float(label_y), axref=xaxis, ayref=yaxis,
                text=settings["label"], font=font,
                xanchor="left" if side>0 else "right", yanchor="middle", showarrow=True, arrowhead=0,
                arrowwidth=.8, arrowcolor=color, borderpad=1, bgcolor="rgba(255,255,255,0.9)")
