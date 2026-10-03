from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app_helpers import protocol_v11


def test_protocol_public_bundle_is_aggregate_only() -> None:
    manifest = protocol_v11.load_protocol_manifest()
    assert manifest["protocol_version"] == "1.1"
    assert manifest["complete_expression_n"] == 464
    assert manifest["privacy"]["donor_identifiers_present"] is False
    for name in (
        "outcome_dictionary.tsv",
        "outcome_exclusions.tsv",
        "source_inventory.tsv",
        "expression_inventory.tsv",
        "split_registry_summary.tsv",
    ):
        frame = protocol_v11.load_protocol_table(name)
        assert not frame.empty
        assert {"donor", "projid", "donor_id"}.isdisjoint(frame.columns)


def test_protocol_outcomes_preserve_verified_and_blocked_definitions() -> None:
    outcomes = protocol_v11.load_protocol_table("outcome_dictionary.tsv")
    status = outcomes.set_index("key")["audit_status"].to_dict()
    assert status["diagnosis_binary"] == "verified"
    assert status["cogng_path_slope"] == "blocked_definition"
    assert outcomes.set_index("key").loc["amyloid", "actual_n"] == 456


def test_protocol_results_loader_handles_not_yet_published_tables() -> None:
    assert protocol_v11.load_protocol_table("performance").empty
    assert protocol_v11.load_protocol_table("coefficients").empty

