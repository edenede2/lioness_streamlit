#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from app_helpers.charts import prediction_curve_figure, prediction_performance_figure
from app_helpers.data import PREDICTION_BLOCK_LABELS, PREDICTION_MODEL_LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_prediction_figures"
OUT.mkdir(exist_ok=True)


def paper_style(fig, *, width: int, height: int, top_margin: int = 55):
    fig.update_layout(
        width=width,
        height=height,
        font={"family": "Arial, Helvetica, sans-serif", "size": 18, "color": "#111111"},
        title={"text": ""},
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 80, "r": 30, "t": top_margin, "b": 90},
    )
    fig.update_xaxes(title_font={"size": 19}, tickfont={"size": 16})
    fig.update_yaxes(title_font={"size": 19}, tickfont={"size": 16})
    if fig.layout.legend:
        fig.update_layout(legend_font={"size": 15})
    return fig


def save(fig, stem: str, *, width: int, height: int, top_margin: int = 55):
    fig = paper_style(fig, width=width, height=height, top_margin=top_margin)
    fig.write_image(OUT / f"{stem}.pdf", format="pdf", width=width, height=height)
    fig.write_image(OUT / f"{stem}.svg", format="svg", width=width, height=height)
    fig.write_image(OUT / f"{stem}.png", format="png", width=width, height=height, scale=2)


# Panel A: donor-averaged OOF ROC curves for the primary AD-vs-Control comparison.
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
        oof["predictor_block"].eq(block) & oof["model_variant"].eq(variant)
    ].copy()
    selected = selected.drop_duplicates("sample_id")
    target_text = selected["target"].astype(str)
    y = (
        target_text.eq("AD").astype(int).to_numpy()
        if target_text.isin(["AD", "Control"]).all()
        else pd.to_numeric(selected["target"], errors="raise").astype(int).to_numpy()
    )
    probability = pd.to_numeric(selected["probability_AD"], errors="raise").to_numpy()
    auc = roc_auc_score(y, probability)
    if abs(auc - expected_auc) > 0.003:
        raise RuntimeError(f"Unexpected AUC for {label}: {auc:.6f}")
    fpr, tpr, _ = roc_curve(y, probability)
    curve_rows.extend(
        {"curve": "ROC", "class_label": f"{label} (AUC={auc:.3f})", "x": x, "y": yy}
        for x, yy in zip(fpr, tpr, strict=True)
    )
roc_frame = pd.DataFrame(curve_rows)
roc_fig = prediction_curve_figure(roc_frame, curve="ROC", title="")
roc_fig.update_layout(
    legend={"orientation": "h", "y": 1.035, "x": 0.5, "xanchor": "center", "yanchor": "bottom"}
)
save(roc_fig, "Figure_prediction_ROC_CT_TS", width=980, height=610, top_margin=72)


# Panel B: same targeted module identities, decomposed into individual regions and region pairs.
perf = pd.read_parquet(ROOT / "data/prediction_targeted/targeted_oof_performance.parquet")
blocks = ["AC", "DLPFC", "PCG", "AC_DLPFC", "AC_PCG", "DLPFC_PCG"]
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
resolved = resolved.sort_values(["predictor_block", "evidence_tier"], kind="stable").drop_duplicates("predictor_block")
expected = {
    "AC": 0.662224,
    "DLPFC": 0.684132,
    "PCG": 0.718818,
    "AC_DLPFC": 0.786804,
    "AC_PCG": 0.774244,
    "DLPFC_PCG": 0.790784,
}
if set(resolved["predictor_block"].astype(str)) != set(blocks):
    raise RuntimeError("Resolved predictor blocks are incomplete")
for row in resolved.itertuples(index=False):
    if abs(float(row.value) - expected[str(row.predictor_block)]) > 0.004:
        raise RuntimeError(f"Unexpected resolved AUC for {row.predictor_block}: {row.value}")
block_labels = dict(PREDICTION_BLOCK_LABELS)
block_labels.update({
    "AC": "AC (TS)",
    "DLPFC": "DLPFC (TS)",
    "PCG": "PCG (TS)",
    "AC_DLPFC": "AC–DLPFC (CT)",
    "AC_PCG": "AC–PCG (CT)",
    "DLPFC_PCG": "DLPFC–PCG (CT)",
})
resolved_fig = prediction_performance_figure(
    resolved,
    metric="roc_auc",
    block_labels=block_labels,
    block_order=blocks,
    model_labels=PREDICTION_MODEL_LABELS,
    title="",
)
resolved_fig.update_yaxes(title_text="ROC-AUC", range=[0.60, 0.81])
resolved_fig.update_layout(showlegend=False, margin={"l": 80, "r": 30, "t": 35, "b": 120})
save(resolved_fig, "Figure_prediction_resolved_components", width=1100, height=560, top_margin=35)

resolved[["predictor_block", "value", "n_oof"]].to_csv(
    OUT / "resolved_prediction_values.tsv", sep="\t", index=False
)
