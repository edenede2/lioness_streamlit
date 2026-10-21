"""Explicit, immutable identities for non-discovery-donor validation.

No score may fall back to another partition, network method or cohort. The old
Control-derived CAMEL directory remains a supported on-disk schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app_helpers.camel import COMPONENT_TISSUES, COMPONENT_LABELS, score_framework

MOTOR_COMPONENTS = {
    "TS_VH": ("VH",), "TS_SMA": ("SMA",), "TS_QUAD": ("QUAD",),
    "CT_VH__SMA": ("VH", "SMA"), "CT_VH__QUAD": ("VH", "QUAD"),
    "CT_SMA__QUAD": ("SMA", "QUAD"),
}
MOTOR_LABELS = {c: "–".join(ts) for c, ts in MOTOR_COMPONENTS.items()}


@dataclass(frozen=True)
class ValidationDataset:
    module_set: str
    modules: int
    assignments: int
    methods: tuple[str, ...]
    original_donors: int
    excluded_complete: int
    validation_donors: int
    controls: int
    counts: tuple[int, ...]
    motor: bool = False

    @property
    def components(self):
        return MOTOR_COMPONENTS if self.motor else COMPONENT_TISSUES

    @property
    def labels(self):
        return MOTOR_LABELS if self.motor else COMPONENT_LABELS

    @property
    def expected_counts(self):
        return dict(zip(self.components, self.counts, strict=True))

    @property
    def rule_count(self):
        return 4 if self.motor else 31

    def framework(self, method):
        self.check_method(method)
        return score_framework(self.module_set, method)

    def check_method(self, method):
        if method not in self.methods:
            raise ValueError(f"No validation method {method!r} for {self.module_set}")

    def relative_root(self, method):
        self.check_method(method)
        if self.module_set == "control_derived":
            return Path("control_derived/camel_validation")
        return Path("validation") / self.module_set / method

    def expected_rows(self):
        return self.modules * sum(self.counts) * self.rule_count


DATASETS = {
    "control_derived": ValidationDataset("control_derived", 186, 49392,
        ("control_anchored",), 450, 464, 793, 164, (266, 752, 195, 230, 14, 176)),
    "full_cohort": ValidationDataset("full_cohort", 154, 46898,
        ("standard", "control_anchored"), 450, 464, 793, 164, (266, 752, 195, 230, 14, 176)),
    "muscle_control_l4": ValidationDataset("muscle_control_l4", 155, 33491,
        ("control_anchored",), 357, 357, 142, 120, (110, 88, 67, 58, 40, 25), True),
    "muscle_all_donors_l4": ValidationDataset("muscle_all_donors_l4", 173, 31948,
        ("control_anchored",), 357, 357, 142, 120, (110, 88, 67, 58, 40, 25), True),
}


def dataset(module_set="control_derived", method="control_anchored"):
    try:
        result = DATASETS[module_set]
    except KeyError as exc:
        raise ValueError("Unknown non-discovery validation partition") from exc
    result.check_method(method)
    return result


def manifest_matches(manifest, module_set, method):
    """Legacy manifests are accepted only in the original legacy namespace."""
    return (manifest.get("module_set", "control_derived") == module_set
            and manifest.get("network_method", "control_anchored") == method)
