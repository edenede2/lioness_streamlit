"""The documented all command must reach a reviewable, local-only bundle."""

import sys
from scripts import run_control_calibrated_validation as CLI
from app_helpers.control_calibrated import (
    validation_finish as F,
    validation_focused as C,
    validation_report as P,
)


def test_all_runs_benchmark_report_and_local_activation_then_is_immutable(
    monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(sys, "argv", ["validation", "all", "--workers", "2"])
    monkeypatch.setattr(
        CLI.S, "freeze", lambda: (tmp_path, {"analysis_id": "synthetic"})
    )

    def action(name):
        def run(*args, **kwargs):
            calls.append(name)
            if name == "activate":
                (tmp_path / "complete.json").write_text("{}")
            return {"status": "complete"}

        return run

    monkeypatch.setattr(CLI.R, "prepare", action("prepare"))
    monkeypatch.setattr(CLI, "benchmark", action("benchmark"))
    monkeypatch.setattr(CLI.R, "run_tasks", lambda r, s, stage, w: calls.append(stage))
    for name in ["assemble_repair", "assemble_scan"]:
        monkeypatch.setattr(CLI.R, name, action(name))
    monkeypatch.setattr(C, "run_focused", action("focused"))
    for name in ["summarize", "verify", "activate"]:
        monkeypatch.setattr(F, name, action(name))
    monkeypatch.setattr(P, "build", action("report"))
    CLI.main()
    assert calls == [
        "prepare",
        "benchmark",
        "repair",
        "assemble_repair",
        "scan",
        "assemble_scan",
        "focused",
        "summarize",
        "verify",
        "report",
        "activate",
    ]
    before = list(calls)
    CLI.main()
    assert calls == before
