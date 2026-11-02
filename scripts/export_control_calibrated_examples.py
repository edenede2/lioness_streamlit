#!/usr/bin/env python3
"""Publication exports from reviewed aggregate results, without donor rows."""
from pathlib import Path
import sys, json
from dataclasses import replace

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import pandas as pd
import plotly.express as px
from app_helpers import data as D
from app_helpers.control_calibrated import adapter as A
from app_helpers.control_calibrated.inputs import sha, atomic_json
from app_helpers.figure_export import PlotExportSpec, PlotContentConfig, preset_config
from app_helpers.figure_export.adapters import render_figure


def main():
    out = APP / "artifacts/control_calibrated_implementation_20261102/exports"
    out.mkdir(exist_ok=True)
    checks = []
    for family, module, estimator in [
        ("full_cohort", 188, "calibrated_beta3"),
        ("muscle_control_l4", 312, "calibrated_beta1"),
    ]:
        result = A.baseline_associations(
            module_set=family, module=module, estimator=estimator
        )
        result = result[
            result.metric_family.isin(["gain", "loss"])
            & result.component.isin(["TS", "CT"])
            & result.method.eq("Spearman")
        ].copy()
        result["measure"] = (
            result.metric_family.map({"gain": "Gain G", "loss": "Loss L"})
            + " · "
            + result.component
        )
        matrix = result.pivot(
            index="measure", columns="outcome", values="effect"
        ).rename(columns=D.ASSOCIATION_OUTCOME_LABELS)
        # Wrap long supplied outcome labels while preserving the original source table.
        import textwrap

        matrix.columns = ["<br>".join(textwrap.wrap(v, 24)) for v in matrix.columns]
        beta = 3 if estimator.endswith("3") else 1
        figure = px.imshow(
            matrix,
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            labels={
                "color": "Spearman rho",
                "x": "Outcome",
                "y": "Measure / component",
            },
            title=f"{family} · M{module} · beta{beta}",
        )
        figure.update_layout(
            width=1500,
            height=700,
            title=f"{family} · M{module} · beta{beta}<br><sup>Independent B targets · observed paired N {int(result.n.min())}–{int(result.n.max())} · unmasked · native units</sup>",
            meta=dict(
                score_version=f"control_calibrated_connectivity.v2.beta{beta}",
                cohort_scope="complete_450",
                transform="native",
                testing_manifest_id=result.analysis_id.iloc[0],
            ),
        )
        figure.update_xaxes(tickangle=-60)
        spec = PlotExportSpec.from_figure(
            f"{family}_M{module}_beta{beta}_heatmap",
            figure,
            dict(
                score_version=f"control_calibrated_connectivity.v2.beta{beta}",
                cohort_scope="complete_450",
                transform="native",
            ),
        )
        result.to_csv(
            out / f"{family}_M{module}_beta{beta}_aggregate_statistics.csv", index=False
        )
        for fmt in ("png", "svg"):
            config = replace(
                preset_config("Grant figure", spec.aspect_ratio),
                width=300,
                height=155,
                dpi=200,
                format=fmt,
            )
            raw, artifact = render_figure(spec, config, PlotContentConfig())
            path = out / f"{family}_M{module}_beta{beta}_heatmap.{fmt}"
            path.write_bytes(raw)
            checks.append(
                dict(
                    path=path.name,
                    sha256=sha(path),
                    bytes=len(raw),
                    format=fmt,
                    source_rows=len(result),
                    paired_N_range=[int(result.n.min()), int(result.n.max())],
                    privacy="Aggregate association statistics only; no donor identifiers or donor rows",
                    preset="Grant figure",
                    width_mm=300,
                    height_mm=155,
                    dpi=200,
                )
            )
            print(json.dumps(checks[-1]), flush=True)
    atomic_json(out / "export_checks.json", checks)


if __name__ == "__main__":
    main()
