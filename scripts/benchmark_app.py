"""Read-only, payload-free UI benchmark; run with the pinned app environment.

Example: python scripts/benchmark_app.py --output /tmp/lioness-ui-baseline.json
No source data, donor values, credentials, or figure payloads enter this report.
"""

from __future__ import annotations

import argparse
from collections import Counter
from functools import wraps
import json
from pathlib import Path
import resource
import sys
import time
from unittest.mock import patch


def measure(app_path: Path, view: str, repeats: int) -> dict:
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from app_helpers import data, drive_data
    import pandas as pd
    import pyarrow.parquet as pq

    counts = Counter()
    timings = Counter()

    def instrument(name, function):
        @wraps(function)
        def measured(*args, **kwargs):
            start = time.perf_counter()
            counts[name] += 1
            try:
                return function(*args, **kwargs)
            finally:
                timings[name] += time.perf_counter() - start

        return measured

    from contextlib import ExitStack

    with ExitStack() as stack:
        for module, names in (
            (
                data,
                (
                    "load_aggregate_scope",
                    "load_resolved_scope",
                    "load_kegg",
                    "load_mdc_summary",
                    "load_mdc_resolved",
                    "load_sample_metadata",
                    "_read_filtered",
                    "_read_nested_parquet_bounded",
                ),
            ),
            (drive_data, ("ensure_data_path",)),
            (pd, ("read_parquet",)),
            (pq, ("read_table",)),
        ):
            for name in names:
                original = getattr(module, name)
                stack.enter_context(
                    patch.object(module, name, instrument(name, original))
                )
        st.cache_data.clear()
        app = AppTest.from_file(str(app_path), default_timeout=240)
        app.query_params["view"] = view
        runs = []
        for index in range(repeats):
            counts.clear()
            timings.clear()
            start = time.perf_counter()
            app.run()
            runs.append(
                {
                    "state": "cold" if index == 0 else "warm",
                    "seconds": round(time.perf_counter() - start, 4),
                    "peak_rss_mib": round(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2
                    ),
                    "load_calls": dict(counts),
                    "load_seconds": {
                        key: round(value, 4) for key, value in timings.items()
                    },
                    "exception_types": [str(value.type) for value in app.exception],
                    "charts": len(app.get("plotly_chart")),
                    "tables": len(app.dataframe),
                }
            )
    return {"view": view, "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--app",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "streamlit_app.py",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=[
            "Module details",
            "KEGG enrichment",
            "MDC",
            "Module finder",
            "Associations",
        ],
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    sys.path.insert(0, str(args.app.resolve().parent))
    results = [measure(args.app, view, args.repeats) for view in args.views]
    report = {
        "schema_version": 1,
        "measurement": "local AppTest; peak_rss is cumulative process high-water, not per-view attribution",
        "two_session_browser_check": "not_measured_by_this_script",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.output),
                "views": len(results),
                "exceptions": sum(
                    len(run["exception_types"])
                    for result in results
                    for run in result["runs"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
