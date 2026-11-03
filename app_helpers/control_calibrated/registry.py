"""Versioned donor-score capabilities, independent of prediction registries."""

from pathlib import Path
import json

SCHEMA_VERSION = 2
SOURCE_LABEL = "Control-calibrated connectivity"
ESTIMATORS = {"calibrated_beta3": 3, "calibrated_beta1": 1}
METRICS = {
    "gain": ("G", "Gain / strengthening"),
    "loss": ("L", "Loss / weakening"),
    "net_change": ("S", "Net magnitude change"),
    "magnitude_atypicality": ("A", "Magnitude atypicality"),
    "mean_abs_w": ("mean_abs_w", "Mean raw magnitude"),
    "mean_delta_abs": ("mean_delta_abs", "Mean raw magnitude change"),
    "signed_distance": ("D", "Signed-profile squared distance D"),
    "signed_abs_z": ("mean_abs_z", "Mean absolute signed z"),
}
PRIMARY = ("gain", "loss", "net_change", "magnitude_atypicality")
CAPABILITIES = {
    "donor_scores": True,
    "associations": True,
    "regression_associations": True,
    "distributions": True,
    "comparisons": True,
    "edge_inspection": True,
    "prediction": False,
    "predictive_regression": False,
    "feature_selection": False,
}
APP = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = APP / "data/control_calibrated_v2"


def runtime_root():
    from app_helpers.drive_data import DATA_DIR, LOCAL_DATA_DIR

    bootstrap = APP / "control_calibrated_release.json"
    if (
        DATA_DIR == LOCAL_DATA_DIR.resolve()
        and (RESEARCH_ROOT / "manifest.json").exists()
    ):
        return RESEARCH_ROOT
    if bootstrap.exists():
        release = json.loads(bootstrap.read_text())
        prefix = Path(release["data_prefix"])
        if prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError("Invalid calibrated release prefix")
        return DATA_DIR / prefix
    return DATA_DIR / "control_calibrated_v2"


ROOT = runtime_root()
HELP = "Deviations in LIONESS contribution magnitude relative to disjoint calibration Controls. These are not observed within-person correlations, Gaussian tail probabilities, or fold changes. Gain and loss share one physical-edge denominator. A=G+L; signed distance D is separate. Reversal is a sign annotation. Per-edge loss is bounded by nu/tau; lower loss variation does not establish better biology."


def is_calibrated(estimator):
    return estimator in ESTIMATORS


def beta_for(estimator):
    return ESTIMATORS[estimator]


def feature_labels():
    return {k: v[1] for k, v in METRICS.items()}


def source_spec(estimator):
    if not is_calibrated(estimator):
        raise ValueError("Unknown calibrated source")
    capabilities = dict(CAPABILITIES)
    if manifest().get("capabilities", {}).get("edge_reconstruction") is False:
        capabilities["edge_inspection"] = False
    return dict(
        id=f"control_calibrated_connectivity.v2.beta{beta_for(estimator)}",
        label=SOURCE_LABEL,
        beta=beta_for(estimator),
        capabilities=capabilities,
        default_metric="gain",
    )


def require_capability(estimator, capability):
    if is_calibrated(estimator) and not CAPABILITIES.get(capability, False):
        raise ValueError(f"{SOURCE_LABEL} is excluded from {capability}")


def manifest(root=ROOT):
    from .hosted import available, read_json

    path = Path(root) / "manifest.json"
    return (
        read_json(path)
        if available(path)
        else {
            "status": "unavailable",
            "reason": "Control-calibrated catalog has not been generated",
            "families": {},
        }
    )


def ready(family=None, root=ROOT):
    m = manifest(root)
    return m.get("status") == "complete" and (
        family is None or family in m.get("families", {})
    )
