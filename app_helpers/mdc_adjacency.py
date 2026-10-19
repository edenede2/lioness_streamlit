"""Configurable frame for published split-triangle MDC images.

The original raster and its AD-derived ordering are preserved byte for byte.
Only its viewport and the surrounding Plotly labels change. Numeric color limits
come from the matching release manifest, never from reverse-mapping pixel colors.
"""

from __future__ import annotations

import base64
import hashlib
import html
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

from app_helpers.drive_data import ensure_data_path

COLORSCALE = [[0, "#FFFF00"], [0.5, "#FFA500"], [1, "#FF0000"]]
REGION_COLORS = {"AC": "#4C78A8", "DLPFC": "#E69F00", "PCG": "#009E73"}


def _longest_run(indices: np.ndarray) -> tuple[int, int]:
    if not len(indices):
        raise ValueError("The published heatmap layout could not be identified.")
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    run = max(runs, key=len)
    return int(run[0]), int(run[-1])


@lru_cache(maxsize=8)
def _image_geometry(path: str, size: int, mtime_ns: int) -> dict:
    """Cache small bounds only, not decoded pixels or images (about 40 MB each)."""
    with Image.open(path) as image:
        width, height = image.size
        if width * height > 25_000_000:
            raise ValueError("This image exceeds the configurable heatmap size limit.")
        pixels = np.asarray(image.convert("RGB"))
    # Locate the broad sequential-palette rectangle; thin annotation strips and
    # the color bar form separate, much smaller runs. No edge weights are inferred.
    mask = (pixels[:, :, 0] >= 200) & (pixels[:, :, 2] < 40)
    counts = mask.sum(axis=1)
    top, bottom = _longest_run(np.flatnonzero(counts >= counts.max() * 0.65))
    columns = mask[top : bottom + 1].sum(axis=0)
    left, right = _longest_run(np.flatnonzero(columns >= (bottom - top + 1) * 0.8))
    core_width, core_height = right - left + 1, bottom - top + 1
    if (
        min(core_width, core_height) < 100
        or core_width * core_height < width * height * 0.25
        or not 0.85 <= core_width / core_height <= 1.15
    ):
        raise ValueError("The published heatmap uses an unsupported image layout.")
    core = [left - 0.5, top - 0.5, right + 0.5, bottom + 0.5]
    # Brain images have a fixed, source-builder-defined 3200px template. Only
    # expose its tissue strips when the observed core agrees with that template.
    brain_template = (
        width == height == 3200
        and abs(left - 416) <= 3 and abs(top - 672) <= 3
        and abs(right - 2591) <= 3 and abs(bottom - 2815) <= 3
    )
    annotated = [361.5, 620.5, core[2], core[3]] if brain_template else core
    return dict(width=width, height=height, core_bounds=core, annotation_bounds=annotated,
                region_annotations_available=brain_template)


def load_heatmap_manifest(path: Path, module: int, module_set: str) -> dict:
    """Read only the selected image's compact, matching release manifest."""
    manifest_path = ensure_data_path(path.parent / "manifest.tsv")
    if isinstance(manifest_path, list):
        raise ValueError("Expected one matching MDC image manifest.")
    # Historical motor manifests use commas despite their .tsv extension.
    frame = pd.read_csv(manifest_path, sep=None, engine="python")
    selected = frame.loc[pd.to_numeric(frame["module"], errors="coerce").eq(int(module))]
    if len(selected) != 1:
        raise ValueError("The matching image manifest must identify the module exactly once.")
    record = selected.iloc[0].to_dict()
    if "module_set" in record and record["module_set"] != module_set:
        raise ValueError("The heatmap manifest belongs to a different module definition.")
    if record.get("png", record.get("filename")) != path.name:
        raise ValueError("The heatmap filename does not match its manifest.")
    if record.get("upper_triangle") != "AD" or record.get("lower_triangle") != "Control":
        raise ValueError("Unexpected diagnosis ordering in the heatmap manifest.")
    if record.get("color_palette") != "yellow_orange_red":
        raise ValueError("The configurable frame requires the published yellow–red palette.")
    cap = float(record.get("display_vmax", record.get("display_q99", np.nan)))
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("The heatmap manifest has no valid numeric color limit.")
    record["color_limit"] = cap
    return record


def configure_heatmap_view(figure: go.Figure, show_regions: bool) -> None:
    """Change the viewport only; fixed ranges prevent revealing baked-in text."""
    metadata = (figure.layout.meta or {}).get("figure_export", {}).get("split_mdc_heatmap")
    if not metadata:
        return
    show_regions = bool(show_regions and metadata["region_annotations_available"])
    left, top, right, bottom = metadata[
        "annotation_bounds" if show_regions else "core_bounds"
    ]
    figure.update_xaxes(range=[left, right], fixedrange=True, autorange=False)
    figure.update_yaxes(range=[bottom, top], fixedrange=True, autorange=False)
    metadata["region_annotations_shown"] = show_regions


def split_heatmap_figure(
    path: Path, module: int, module_set: str, module_set_label: str, *,
    region_counts: dict[str, int] | None = None,
) -> go.Figure:
    record = load_heatmap_manifest(path, module, module_set)
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    expected_hash = record.get("png_sha256")
    if isinstance(expected_hash, str) and expected_hash != digest:
        raise ValueError("The heatmap image checksum does not match its release manifest.")
    stat = path.stat()
    geometry = _image_geometry(str(path), stat.st_size, stat.st_mtime_ns)
    for dimension in ("width", "height"):
        if dimension in record and int(record[dimension]) != geometry[dimension]:
            raise ValueError("The heatmap dimensions do not match its release manifest.")
    # Region-strip geometry is specific to the published brain renderer, not
    # inferred for a different module definition sharing a numeric module ID.
    if module_set not in {"full_cohort", "control_derived"}:
        geometry = dict(geometry, region_annotations_available=False)
    figure = go.Figure(go.Image(
        source="data:image/png;base64," + base64.b64encode(blob).decode("ascii"),
        x0=0, y0=0, dx=1, dy=1, hoverinfo="skip",
        meta={"export_role": "source_image"},
    ))
    if geometry["region_annotations_available"]:
        for region, count in (region_counts or {}).items():
            if region in REGION_COLORS and count > 0:
                figure.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers", name=f"{region} ({int(count):,})",
                    marker=dict(color=REGION_COLORS[region], symbol="square", size=12),
                    hoverinfo="skip", meta={"export_role": "legend"}, showlegend=True,
                ))
    cap = record["color_limit"]
    figure.add_trace(go.Scatter(
        x=[None, None], y=[None, None], mode="markers", showlegend=False,
        marker=dict(color=[0, cap], cmin=0, cmax=cap, colorscale=COLORSCALE,
                    showscale=True, colorbar=dict(title=dict(text="Absolute edge weight", side="right"), x=1.02,
                                                 thickness=18, len=0.9)),
        hoverinfo="skip", meta={"export_role": "colorbar"},
    ))
    statistics = []
    for scope in ("total", "ts", "ct"):
        value = record.get(f"mdc_{scope}")
        if value is not None and pd.notna(value):
            statistics.append(f"{scope.upper()} MDC={float(value):.3g}")
    if statistics:
        figure.add_annotation(
            text=" | ".join(statistics), name="publication-statistics-mdc",
            xref="paper", yref="paper", x=0, y=1.02,
            xanchor="left", yanchor="bottom", showarrow=False,
        )
    n_nodes = int(record["n_nodes"])
    figure.update_layout(
        template="plotly_white", height=800, width=900,
        title=dict(text=f"{html.escape(module_set_label)} · Module M{int(module)}"
                        f"<br><sup>Upper triangle: AD | Lower triangle: Control | {n_nodes:,} tissue–gene nodes</sup>",
                   x=0.02, xanchor="left", font=dict(size=20)),
        margin=dict(l=75, r=145, t=150, b=75),
        legend=dict(orientation="h", x=0, y=-0.10, xanchor="left", yanchor="top",
                    title="Tissue–gene nodes"),
        showlegend=geometry["region_annotations_available"],
        xaxis=dict(title="Genes", showticklabels=False, showgrid=False,
                   zeroline=False, constrain="domain"),
        yaxis=dict(title="Genes", showticklabels=False, showgrid=False,
                   zeroline=False, scaleanchor="x", scaleratio=1, constrain="domain"),
        meta={"figure_export": {"split_mdc_heatmap": {
            **geometry, "source_sha256": digest, "module_set": module_set,
            "module": int(module), "color_limit": cap, "palette_locked": True,
            "region_annotations_shown": geometry["region_annotations_available"],
            "provenance": "Unchanged published PNG; editable vector frame; no edge-value reconstruction",
        }}},
    )
    configure_heatmap_view(figure, True)
    return figure
