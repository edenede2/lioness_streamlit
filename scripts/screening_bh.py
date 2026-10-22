"""Private disk-spilling BH. DuckDB is deliberately not an app dependency."""
from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd

DUCKDB_VERSION = "1.4.3"


def external_bh(paths, p_column, selected_ids, *, temporary_directory, memory_gib=4, batch_size=65536):
    """Exact reverse BH over ALL finite tests, return only selected q-values.

    Files have a globally unique int64 test_id and float64 p column. Temporary
    files and complete p-value batches belong upstream, never in the app bundle.
    The Arrow result is consumed batchwise; no pandas materialization of sort.
    """
    import duckdb
    if duckdb.__version__ != DUCKDB_VERSION:
        raise RuntimeError(f"Private analysis requires duckdb=={DUCKDB_VERSION}")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", p_column):
        raise ValueError("Unsafe p-value column")
    if not 1 <= memory_gib <= 16:
        raise ValueError("BH sort RAM must stay within the shared analysis budget")
    root = Path(temporary_directory); root.mkdir(parents=True, exist_ok=True)
    wanted = set(map(int, selected_ids)); found = []
    connection = duckdb.connect()
    try:
        connection.execute(f"SET memory_limit='{memory_gib}GB'")
        connection.execute("SET threads=1")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute("SET temp_directory=?", [str(root)])
        relation = connection.read_parquet([str(Path(p)) for p in paths]).project(f"test_id, {p_column} AS p")
        relation.create_view("tests")
        invalid = connection.execute("SELECT count(*) FROM tests WHERE isfinite(p) AND (p<0 OR p>1)").fetchone()[0]
        if invalid:
            raise ValueError("Out-of-range original p-values")
        total = int(connection.execute("SELECT count(*) FROM tests WHERE isfinite(p)").fetchone()[0])
        # Descending p, descending identity gives ascending-rank positions
        # total,...,1. Reverse cumulative min is exact even for tied p-values.
        reader = connection.execute("SELECT test_id,p FROM tests WHERE isfinite(p) ORDER BY p DESC,test_id DESC").fetch_record_batch(batch_size)
        rank, running = total, 1.
        for batch in reader:
            ids = batch.column(0).to_numpy(); p = batch.column(1).to_numpy()
            ranks = np.arange(rank, rank-len(p), -1, dtype=float)
            q = np.minimum.accumulate(np.minimum(running, p*total/ranks))
            running = float(q[-1]); rank -= len(p)
            keep = np.fromiter((int(i) in wanted for i in ids), bool, len(ids))
            found.extend(zip(ids[keep].tolist(), q[keep].tolist()))
        if rank != 0:
            raise RuntimeError("External BH count/stream mismatch")
        return pd.Series(dict(found), dtype=float), total
    finally:
        connection.close()
