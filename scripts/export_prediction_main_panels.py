#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from app_helpers.charts import prediction_curve_figure, prediction_performance_figure
from app_helpers.data import PREDICTION_BLOCK_LABELS, PREDICTION_MODEL_LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_prediction_figures"
OUT.mkdir(exist_ok=True)


def paper_style(fig, *, width: int, height: int):
    fig.update_layout(
        width=width,
        height=height,
        font={"family": "Arial, Helvetica, sans-serif", "size": 17, "color": "#111111"},
        title_font={"family": "Arial, Helvetica, sans-serif", "size": 20, "color": "#111111"},
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 75, "r": 30, "t": 90, "b": 90},
    )
    fig.update_xaxes(title_font={"size": 18}, tickfont={"size": 15})
    fig.update_yaxes(title_font={"size": 18}, tickfont={"size": 15})
    if fig.layout.legend:
        fig.update_layout(legend_font={"size": 14})
    return fig


def save(fig, stem: str, *, width: int, height: int):
    fig = paper_style(fig, width=width, height=height)
    fig.write_image(OUT / f"{stem}.pdf", format="pdf", width=width, height=height)
    fig.write_image(OUT / f"{stem}.svg", format="svg", width=width, height=height)
    fig.write_image(OUT / f"{stem}.png", format="png", width=width, height=height, scale=2)


# -----------------------------------------------------------------------------
# Panel A: ROC curves from the donor-averaged OOF predictions used by the app.
# -----------------------------------------------------------------------------
oof_path = (
    ROOT / "data/prediction_targeted/targeted_oof_predictions.parquet/"
    "raw__primary__control_derived__control_anchored__not_applicable.parquet"
)
oof = pd.read_parquet(oof_path)

common = (
    oof["panel_strategy"].eq("tissue_neutral_ad")
    & oof["model_outcome"].eq("diagnosis_binary")
    & oof["edge_mask"].eq("all")
    & oof["score_normalization"].eq("standard_pruned")
    & oof["score_transform"].eq("raw")
)
oof = oof.loc[common].copy()

configs = [
    ("Covariates", "CT_pooled", "covariates", 0.6838761501387469),
    ("TS + covariates", "TS_pooled", "covariates_plus_network", 0.679604),
    ("CT + covariates", "CT_pooled", "covariates_plus_network", 0.8455527968453337),
]

curve_rows = []
for label, block, variant, expected_auc in configs:
    selected = oof.loc[
        oof["predictor_block"].eq(block)
        & oof["model_variant"].eq(variant)
    ].copy()
    if selected.empty:
        raise RuntimeError(f"No OOF rows for {label}: {block} / {variant}")
    selected = selected.drop_duplicates("sample_id")
    target_text = selected["target"].astype(str)
    if target_text.isin(["AD", "Control"]).all():
        y = target_text.eq("AD").astype(int).to_numpy()
    else:
        y = pd.to_numeric(selected["target"], errors="raise").astype(int).to_numpy()
    probability = pd.to_numeric(selected["probability_AD"], errors="raise").to_numpy()
    auc = roc_auc_score(y, probability)
    if abs(auc - expected_auc) > 0.003:
        raise RuntimeError(
            f"Unexpected AUC for {label}: observed {auc:.6f}, expected ~{expected_auc:.6f}"
        )
    fpr, tpr, _ = roc_curve(y, probability)
    curve_rows.extend(
        {"curve": "ROC", "class_label": f"{label} (AUC={auc:.3f})", "x": x, "y": yy}
        for x, yy in zip(fpr, tpr, strict=True)
    )

roc_frame = pd.DataFrame(curve_rows)
roc_fig = prediction_curve_figure(
    roc_frame,
    curve="ROC",
    title="AD vs Control: donor-averaged out-of-fold ROC curves",
)
roc_fig.update_layout(legend={"orientation": "h", "y": 1.16, "x": 0.0, "xanchor": "left"})
save(roc_fig, "Figure_prediction_ROC_CT_TS", width=980, height=650)


# -----------------------------------------------------------------------------
# Panel B: same selected modules, decomposed into regions and region pairs.
# Uses the app's prediction_performance_figure.
# -----------------------------------------------------------------------------
perf = pd.read_parquet(ROOT / "data/prediction_targeted/targeted_oof_performance.parquet")
blocks = [
    "TS_AC",
    "TS_DLPFC",
    "TS_PCGBA23",
    "CT_AC__DLPFC",
    "CT_AC__PCGBA23",
    "CT_DLPFC__PCGBA23",
]
resolved = perf.loc[
    perf["module_definition"].eq("control_derived")
    & perf["network_method"].eq("control_anchored")
    & perf["panel_strategy"].eq("tissue_neutral_ad")
    & perf["model_outcome"].eq("diagnosis_binary")
    & perf["edge_mask"].eq("all")
    & perf["score_normalization"].eq("standard_pruned")
    & perf["score_transform"].eq("raw")
    & perf["model_variant"].eq("covariates_plus_network")
    & perf["metric"].eq("roc_auc")
    & perf["predictor_block"].isin(blocks)
].copy()
if "eigengene_source" in resolved:
    resolved = resolved.loc[
        resolved["eigengene_source"].astype(str).isin(["not_applicable", "nan", "None"])
    ]
resolved = resolved.sort_values("predictor_block").drop_duplicates("predictor_block")
if set(resolved["predictor_block"].astype(str)) != set(blocks):
    missing = sorted(set(blocks) - set(resolved["predictor_block"].astype(str)))
    raise RuntimeError(f"Missing resolved predictor blocks: {missing}")

expected = {
    "TS_AC": 0.662224,
    "TS_DLPFC": 0.684132,
    "TS_PCGBA23": 0.718818,
    "CT_AC__DLPFC": 0.786804,
    "CT_AC__PCGBA23": 0.774244,
    "CT_DLPFC__PCGBA23": 0.790784,
}
for row in resolved.itertuples(index=False):
    if abs(float(row.value) - expected[str(row.predictor_block)]) > 0.004:
        raise RuntimeError(
            f"Unexpected resolved AUC for {row.predictor_block}: {row.value}"
        )

block_labels = dict(PREDICTION_BLOCK_LABELS)
block_labels.update({
    "TS_AC": "AC (TS)",
    "TS_DLPFC": "DLPFC (TS)",
    "TS_PCGBA23": "PCG (TS)",
    "CT_AC__DLPFC": "AC–DLPFC (CT)",
    "CT_AC__PCGBA23": "AC–PCG (CT)",
    "CT_DLPFC__PCGBA23": "DLPFC–PCG (CT)",
})
resolved_fig = prediction_performance_figure(
    resolved,
    metric="roc_auc",
    block_labels=block_labels,
    block_order=blocks,
    model_labels=PREDICTION_MODEL_LABELS,
    title="AD vs Control: prediction by region and region pair",
)
resolved_fig.update_yaxes(title_text="ROC-AUC", range=[0.60, 0.82])
resolved_fig.update_layout(showlegend=False, margin={"l": 80, "r": 30, "t": 90, "b": 125})
save(resolved_fig, "Figure_prediction_resolved_components", width=1100, height=620)

summary = resolved[["predictor_block", "value", "n_oof"]].copy()
summary.to_csv(OUT / "resolved_prediction_values.tsv", sep="\t", index=False)
print(summary.to_string(index=False))
