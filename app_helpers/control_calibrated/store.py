"""Explicit private artifact persistence, separate from pure numerical code."""

from pathlib import Path
from dataclasses import asdict
import json
import hashlib
import numpy as np
from .core import FittedComponent, Settings


def save_fit(path, fit):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "rows": fit.rows,
        "columns": fit.columns,
        "powers": fit.powers,
        **{"ref_" + k: v for k, v in fit.reference.items()},
        **{"cal_" + k: v for k, v in fit.calibration.items()},
    }
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.chmod(0o600)
    temporary.replace(path.with_suffix(".npz"))
    digest = hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest()
    meta = dict(
        nodes=fit.nodes,
        settings=asdict(fit.settings),
        protocol=fit.protocol,
        R_ids=fit.R_ids,
        C_ids=fit.C_ids,
        edge_keys=fit.edge_keys,
        protocol_hash=fit.protocol_hash,
        array_sha256=digest,
        privacy="Contains private reference expression; excluded from shared review bundle",
    )
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(meta, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path.with_suffix(".json"))


def load_fit(path):
    path = Path(path)
    m = json.loads(path.with_suffix(".json").read_text())
    if (
        m.get("array_sha256")
        != hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest()
    ):
        raise ValueError(
            "Calibration array content hash mismatch; regenerate stale artifact"
        )
    if m["protocol"].get("version") != "control_calibrated_connectivity_v2":
        raise ValueError("Old calibration affected by missingness bugs; regenerate")
    a = np.load(path.with_suffix(".npz"))
    s = m["settings"]
    s["sign_seeds"] = tuple(s["sign_seeds"])
    s["sign_percentiles"] = tuple(s["sign_percentiles"])
    fit = FittedComponent(
        list(map(tuple, m["nodes"])),
        a["rows"],
        a["columns"],
        a["powers"],
        Settings(**s),
        {k[4:]: a[k] for k in a.files if k.startswith("ref_")},
        {k[4:]: a[k] for k in a.files if k.startswith("cal_")},
        m["protocol"],
        tuple(m["R_ids"]),
        tuple(m["C_ids"]),
        tuple(m["edge_keys"]),
    )
    if fit.protocol_hash != m["protocol_hash"]:
        raise ValueError("Fit artifact protocol hash mismatch")
    return fit
