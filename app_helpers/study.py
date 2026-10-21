"""Public, nominal ROS/MAP study labels; never infer study from sample IDs."""
from __future__ import annotations

import pandas as pd
from functools import lru_cache
from pathlib import Path

STUDY_LABEL = "Study / project (ROS / MAP)"
STUDY_LEVELS = ("ROS", "MAP")
STUDY_SCHEMA_VERSION = 1
STUDY_CACHE = Path(__file__).resolve().parents[1] / "data/study/sample_study.parquet"


def normalize_study(values: pd.Series) -> pd.Series:
    result = values.astype("string").str.strip().str.upper().replace("", pd.NA)
    if not set(result.dropna()).issubset(STUDY_LEVELS):
        raise ValueError("Unexpected study label; expected ROS, MAP or missing.")
    return result


@lru_cache(maxsize=1)
def _study_cache(path: str, mtime_ns: int, size: int) -> pd.DataFrame:
    cache = pd.read_parquet(path, columns=["metadata_namespace", "sample_id", "study"])
    if cache.duplicated(["metadata_namespace", "sample_id"]).any():
        raise ValueError("Duplicate study-label cache keys")
    cache["study"] = normalize_study(cache.study)
    return cache


def with_study_metadata(frame: pd.DataFrame, namespace: str | None = None) -> pd.DataFrame:
    """Old bundles remain readable, with explicitly missing study membership."""
    result = frame.copy()
    result["study"] = normalize_study(
        result["study"] if "study" in result else pd.Series(pd.NA, index=result.index, dtype="string")
    )
    # The small approved repository overlay also supports older Drive metadata.
    # Namespace isolation is required: brain/motor public pseudonyms can overlap.
    if namespace is not None and STUDY_CACHE.is_file():
        stat = STUDY_CACHE.stat()
        cache = _study_cache(str(STUDY_CACHE), stat.st_mtime_ns, stat.st_size)
        lookup = cache.loc[cache.metadata_namespace.eq(namespace)].set_index("sample_id").study
        observed = result.sample_id.map(lookup).astype("string")
        conflict = result.study.notna() & observed.notna() & result.study.ne(observed)
        if conflict.any():
            raise ValueError("Metadata and source-backed study cache disagree")
        result["study"] = result.study.fillna(observed)
    return result
