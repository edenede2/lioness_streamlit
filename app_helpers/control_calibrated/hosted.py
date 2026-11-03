"""Lossless, target-only module bundles for the hosted read-only data backend.

Scopes are metadata subsets of the frozen maximum-component scores. Identical
physical unions share numeric rows inside a module; their context/provenance
remain separate. No expression, reference arrays, fits or identity crosswalks
are stored here. Per-module downloads use the application's Drive verifier.
"""

from pathlib import Path
from functools import lru_cache
import base64, hashlib, json, os, threading, struct
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from app_helpers import drive_data as D
from .inputs import sha
from .inputs import atomic_json

LAYOUT = "calibrated_module_bundles_v1"
CONTEXT = (
    "component",
    "component_class",
    "module",
    "module_key",
    "family",
    "beta",
    "allocation_id",
    "edge_universe_sha256",
    "artifact_hash",
    "cohort_scope",
    "mask",
    "mask_provenance",
)
_READ_LOCK = threading.RLock()


def ensure(path):
    value = D.ensure_data_path(Path(path))
    if isinstance(value, list):
        raise ValueError(
            "A calibrated artifact must resolve exactly one immutable file"
        )
    try:
        key = value.resolve().relative_to(D.DATA_DIR).as_posix()
    except ValueError:
        key = None
    entry = D.load_drive_index().get("files", {}).get(key, {})
    if entry.get("sha256"):
        stat = value.stat()
        _verified(str(value), stat.st_mtime_ns, stat.st_size, entry["sha256"])
    return value


def available(path):
    try:
        return D.data_path_available(Path(path))
    except ValueError:
        return Path(path).exists()


def read_json(path):
    path = ensure(path)
    stat = path.stat()
    return _json_content(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=32)
def _json_content(path, mtime, size):
    return json.loads(Path(path).read_text())


def is_hosted(root):
    p = Path(root) / "manifest.json"
    if not available(p):
        return False
    return read_json(p).get("storage_layout") == LAYOUT


def module_path(root, module_key):
    name = hashlib.sha256(module_key.encode()).hexdigest()[:20]
    return Path(root) / "modules" / (name + ".parquet")


def pack_module(destination, job, sources, metadata):
    """Copy exact numbers; deduplicate only byte-hashed identical edge unions.

    sources holds (beta, mask_key, path, expected_sha256). The caller chooses
    maximum-component source files only. Every source file is content-verified.
    """
    targets = metadata.loc[metadata.role.eq("target"), "sample_id"].tolist()
    lookup = {v: i for i, v in enumerate(targets)}
    entries = {}
    seen = {}
    writer = None
    schema = None
    original = None
    schemas = {}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.parquet")
    source_files = []
    rows = 0
    try:
        for beta, mask, path, expected in sources:
            path = Path(path)
            if sha(path) != expected:
                raise ValueError("Score source content changed: " + str(path))
            frame = pd.read_parquet(path)
            if (
                not frame.role.eq("target").all()
                or not frame.beta.eq(beta).all()
                or not frame.module_key.eq(job["module_key"]).all()
            ):
                raise ValueError("Source role/beta/module identity mismatch")
            if not frame.cohort_scope.eq("maximum_component").all():
                raise ValueError(
                    "Host packing must use the frozen maximum-component scope"
                )
            source_schema = pa.Schema.from_pandas(frame, preserve_index=False)
            schemas.setdefault(str(beta), {})[mask] = base64.b64encode(
                source_schema.serialize().to_pybytes()
            ).decode()
            context_fields = [k for k in CONTEXT if k in frame]
            source_files.append(dict(beta=beta, mask=mask, sha256=expected))
            for component, g in frame.groupby("component", sort=False):
                if g.sample_id.duplicated().any():
                    raise ValueError("More than one row per donor/physical component")
                context = {}
                for key in context_fields:
                    if g[key].nunique(dropna=False) != 1:
                        raise ValueError("Nonconstant physical context: " + key)
                    value = g[key].iloc[0]
                    context[key] = (
                        value.item() if isinstance(value, np.generic) else value
                    )
                data = g.drop(columns=[*context_fields, "sample_id"]).reset_index(
                    drop=True
                )
                target_row = g.sample_id.map(lookup)
                if target_row.isna().any():
                    raise ValueError("Non-target donor in score bundle")
                data.insert(0, "target_row", target_row.to_numpy(np.int32))
                data = data.reindex(columns=sorted(data.columns))
                digest = hashlib.sha256()
                digest.update(str(beta).encode())
                digest.update(context["edge_universe_sha256"].encode())
                digest.update(
                    pd.util.hash_pandas_object(data, index=False).to_numpy().tobytes()
                )
                aggregation = digest.hexdigest()
                entries.setdefault(str(beta), {}).setdefault(mask, {})[component] = (
                    dict(aggregation=aggregation, context=context)
                )
                if aggregation not in seen:
                    data.insert(0, "aggregation_id", aggregation)
                    table = pa.Table.from_pandas(data, preserve_index=False)
                    if writer is None:
                        schema = table.schema
                        writer = pq.ParquetWriter(
                            temporary,
                            schema,
                            compression="zstd",
                            compression_level=9,
                            use_dictionary=True,
                        )
                    writer.write_table(
                        table.cast(schema), row_group_size=max(1, len(data))
                    )
                    rows += len(data)
                    seen[aggregation] = True
        if writer is None:
            raise ValueError("Module has no planned scores")
        # Metadata is attached to the footer after all entries are known.
        header = dict(
            layout=LAYOUT,
            module_key=job["module_key"],
            input_variant=job["input_variant"],
            target_membership_sha256=hashlib.sha256(
                json.dumps(targets, separators=(",", ":")).encode()
            ).hexdigest(),
            entries=entries,
            source_files=source_files,
            schemas=schemas,
            rows=rows,
            unique_aggregations=len(seen),
        )
        writer.add_key_value_metadata(
            {"calibrated_header": json.dumps(header, separators=(",", ":"))}
        )
        writer.close()
        writer = None
        temporary.replace(destination)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
    return dict(
        path=destination.name,
        sha256=sha(destination),
        bytes=destination.stat().st_size,
        source_files=len(sources),
        numeric_rows=rows,
        unique_aggregations=len(seen),
    )


def range_receipt(path):
    """Seal each compressed row group and the footer for selective downloads."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(-8, 2)
        tail = stream.read(8)
        if tail[4:] != b"PAR1":
            raise ValueError("Invalid module Parquet footer")
        start = size - 8 - struct.unpack("<I", tail[:4])[0]

        def block(offset, length):
            stream.seek(offset)
            value = stream.read(length)
            if len(value) != length:
                raise ValueError("Incomplete module range")
            return dict(
                offset=offset, length=length, sha256=hashlib.sha256(value).hexdigest()
            )

        result = dict(
            schema_version=1,
            size=size,
            module_sha256=sha(path),
            footer=block(start, size - start),
            groups={},
        )
        with pq.ParquetFile(path) as parquet:
            for i in range(parquet.num_row_groups):
                row = parquet.metadata.row_group(i)
                columns = [row.column(j) for j in range(row.num_columns)]
                starts = [
                    (
                        c.dictionary_page_offset
                        if c.has_dictionary_page
                        else c.data_page_offset
                    )
                    for c in columns
                ]
                offset = min(starts)
                end = max(a + c.total_compressed_size for a, c in zip(starts, columns))
                stat = columns[0].statistics
                if stat.min != stat.max:
                    raise ValueError("Mixed aggregation row group")
                result["groups"][stat.min] = block(offset, end - offset)
        return result


def _range_bytes(entry, block):
    start = block["offset"] + entry.get("module_offset", 0)
    length = block["length"]
    response = D._authorized_session().get(
        D.DRIVE_MEDIA_URL.format(file_id=entry["id"]),
        params={"alt": "media", "supportsAllDrives": "true"},
        headers={"Range": f"bytes={start}-{start+length-1}"},
        stream=True,
        timeout=(30, 120),
    )
    try:
        response.raise_for_status()
        if response.status_code != 206:
            raise IOError("The app backend did not honor a bounded module range")
        chunks = []
        received = 0
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            received += len(chunk)
            if received > length:
                raise IOError("Hosted module response exceeded its bounded range")
            chunks.append(chunk)
        data = b"".join(chunks)
    finally:
        response.close()
    if len(data) != length or (
        block.get("sha256") and hashlib.sha256(data).hexdigest() != block["sha256"]
    ):
        raise IOError("Module range content checksum mismatch")
    return data


def _sparse_module(root, job, entry, receipt, beta, mask, component=None):
    if receipt.get("container_path"):
        sidecar_path = (
            Path(root)
            / "range_receipts"
            / module_path(root, job["module_key"]).with_suffix(".json").name
        )
        if not sidecar_path.exists() or sha(sidecar_path) != receipt["range_sha256"]:
            data = _range_bytes(
                entry,
                dict(
                    offset=receipt["range_offset"],
                    length=receipt["range_length"],
                    sha256=receipt["range_sha256"],
                ),
            )
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = sidecar_path.with_suffix(".partial")
            temporary.write_bytes(data)
            temporary.replace(sidecar_path)
        sidecar = read_json(sidecar_path)
        entry = dict(entry, module_offset=receipt["module_offset"])
    else:
        sidecar = read_json(Path(root) / receipt["range_path"])
    if sidecar["module_sha256"] != receipt["sha256"]:
        raise ValueError("Module range receipt belongs to another artifact")
    cache = Path(root) / "range_cache" / module_path(root, job["module_key"]).name
    state_path = cache.with_suffix(".loaded.json")
    state = read_json(state_path) if state_path.exists() else {}
    valid_footer = False
    if (
        cache.exists()
        and cache.stat().st_size == sidecar["size"]
        and state.get("footer_sha256") == sidecar["footer"]["sha256"]
    ):
        with cache.open("rb") as f:
            f.seek(sidecar["footer"]["offset"])
            valid_footer = (
                hashlib.sha256(f.read(sidecar["footer"]["length"])).hexdigest()
                == sidecar["footer"]["sha256"]
            )
    if not valid_footer:
        cache.parent.mkdir(parents=True, exist_ok=True)
        footer = _range_bytes(entry, sidecar["footer"])
        with cache.open("wb") as f:
            f.write(b"PAR1")
            f.truncate(sidecar["size"])
            f.seek(sidecar["footer"]["offset"])
            f.write(footer)
        state = dict(footer_sha256=sidecar["footer"]["sha256"], groups=[])
        atomic_json(state_path, state)
    with pq.ParquetFile(cache) as f:
        header = json.loads(f.metadata.metadata[b"calibrated_header"])
    chosen = header["entries"].get(str(beta), {}).get(mask)
    if chosen is None:
        raise ValueError("The selected beta/mask has no hosted artifact")
    if component is not None:
        chosen = {k: v for k, v in chosen.items() if k == component}
    selected = list(dict.fromkeys(v["aggregation"] for v in chosen.values()))
    pending = []
    with cache.open("r+b") as f:
        for aggregation in selected:
            block = sidecar["groups"][aggregation]
            if aggregation in state["groups"]:
                f.seek(block["offset"])
                cached = f.read(block["length"])
                if hashlib.sha256(cached).hexdigest() == block["sha256"]:
                    continue
            pending.append((aggregation, block))
        clusters = []
        for item in sorted(pending, key=lambda x: x[1]["offset"]):
            if clusters:
                first = clusters[-1][0][1]["offset"]
                last = clusters[-1][-1][1]
                end = last["offset"] + last["length"]
                if (
                    item[1]["offset"] - end <= 65536
                    and item[1]["offset"] + item[1]["length"] - first <= 8 * 1024**2
                ):
                    clusters[-1].append(item)
                    continue
            clusters.append([item])
        for cluster in clusters:
            start = cluster[0][1]["offset"]
            last = cluster[-1][1]
            end = last["offset"] + last["length"]
            data = _range_bytes(entry, dict(offset=start, length=end - start))
            for aggregation, block in cluster:
                value = data[
                    block["offset"] - start : block["offset"] - start + block["length"]
                ]
                if hashlib.sha256(value).hexdigest() != block["sha256"]:
                    raise IOError("Module range content checksum mismatch")
                f.seek(block["offset"])
                f.write(value)
                if aggregation not in state["groups"]:
                    state["groups"].append(aggregation)
            f.flush()
    atomic_json(state_path, state)
    return cache


@lru_cache(maxsize=128)
def _verified(path, mtime, size, expected):
    if sha(path) != expected:
        raise ValueError("Hosted module content checksum mismatch")
    return True


def _prune_modules(root, current):
    """Bound new module disk cache; callers hold the read lock until decoding."""
    if not Path(root).resolve().is_relative_to(D.DATA_DIR):
        return
    paths = [
        *list((Path(root) / "modules").glob("*.parquet")),
        *list((Path(root) / "range_cache").glob("*.parquet")),
    ]
    total = sum(p.stat().st_size for p in paths)
    limit = int(os.environ.get("LIONESS_CALIBRATED_CACHE_MIB", "512")) * 1024**2
    for p in sorted(paths, key=lambda x: x.stat().st_mtime_ns):
        if total <= limit:
            break
        if p != current:
            size = p.stat().st_size
            p.unlink()
            total -= size
            p.with_suffix(".loaded.json").unlink(missing_ok=True)


def read_scores(root, job, beta, mask, metadata, component=None):
    """Read only chosen row groups, restore the exact maximum-scope schema."""
    root = Path(root)
    receipt = read_json(root / "release_manifest.json")["modules"][job["module_key"]]
    with _READ_LOCK:
        original_path = module_path(root, job["module_key"])
        transport_path = (
            root / receipt["container_path"]
            if receipt.get("container_path")
            else original_path
        )
        try:
            key = transport_path.resolve().relative_to(D.DATA_DIR).as_posix()
        except ValueError:
            key = None
        entry = D.load_drive_index().get("files", {}).get(key, {})
        if entry.get("id") and receipt.get("range_path"):
            path = _sparse_module(root, job, entry, receipt, beta, mask, component)
        else:
            path = ensure(original_path)
            stat = path.stat()
            _verified(str(path), stat.st_mtime_ns, stat.st_size, receipt["sha256"])
        with pq.ParquetFile(path) as f:
            header = json.loads(f.metadata.metadata[b"calibrated_header"])
        if header["layout"] != LAYOUT or header["module_key"] != job["module_key"]:
            raise ValueError("Hosted module identity mismatch")
        targets = metadata.loc[metadata.role.eq("target"), "sample_id"].tolist()
        membership = hashlib.sha256(
            json.dumps(targets, separators=(",", ":")).encode()
        ).hexdigest()
        if membership != header["target_membership_sha256"]:
            raise ValueError("Hosted target membership differs from score bundle")
        chosen = header["entries"].get(str(beta), {}).get(mask)
        if chosen is None:
            raise ValueError("The selected beta/mask has no verified hosted artifact")
        if component is not None:
            chosen = {k: v for k, v in chosen.items() if k == component}
        if not chosen:
            schema_bytes = (
                header["schemas"][str(beta)][mask]
                if "schemas" in header
                else header["schema"]
            )
            schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(schema_bytes)))
            return pa.Table.from_batches([], schema=schema).to_pandas()
        ids = list(dict.fromkeys(v["aggregation"] for v in chosen.values()))
        values = pq.read_table(
            path, filters=[("aggregation_id", "in", ids)]
        ).to_pandas()
        encoded_schema = (
            header["schemas"][str(beta)][mask]
            if "schemas" in header
            else header["schema"]
        )
        schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(encoded_schema)))
        pieces = []
        for component, record in chosen.items():
            frame = (
                values.loc[values.aggregation_id.eq(record["aggregation"])]
                .drop(columns="aggregation_id")
                .copy()
            )
            frame["sample_id"] = np.asarray(targets, dtype=object)[
                frame.pop("target_row").to_numpy(int)
            ]
            for key, value in record["context"].items():
                frame[key] = value
            pieces.append(
                pa.Table.from_pandas(
                    frame[schema.names], schema=schema, preserve_index=False
                ).to_pandas()
            )
        result = pd.concat(pieces, ignore_index=True)
        _prune_modules(root, path)
        return result
