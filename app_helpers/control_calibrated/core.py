"""Pure control-calibrated magnitude scoring. No data lookup, UI, outcome or file I/O.

Calibrations are edge-local. Arrays carry private expression only in memory;
the caller controls persistence. All score/coverage policies are explicit.
"""

from dataclasses import dataclass, asdict
import hashlib, json
import numpy as np

VERSION = "control_calibrated_connectivity_v2"
CLASSES = ("zero", "near_zero", "uncertain", "preserved", "reversed")
FAMILIES = (
    "control_derived",
    "full_cohort",
    "muscle_control_l4",
    "muscle_all_donors_l4",
)


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def member_hash(ids):
    return canonical_hash(sorted(map(str, ids)))


@dataclass(frozen=True)
class Settings:
    expected_R: int
    expected_C: int
    numerical_min_C: int = 3
    relative_multiplier: float = 64.0
    sign_agreement: float = 0.8
    near_zero_fraction: float = 0.01
    sign_seeds: tuple = tuple(range(711010, 711110))
    sign_percentiles: tuple = (2.5, 97.5)
    sparse_cell_policy: str = "R_mean"
    edge_batch: int = 256
    target_batch: int = 64

    def validate(self):
        if self.expected_R < 3 or self.expected_C < self.numerical_min_C:
            raise ValueError("Declared protocol has too few controls")
        if self.numerical_min_C < 3 or self.relative_multiplier < 0:
            raise ValueError("Invalid numerical guard")
        if not 0 <= self.sign_agreement <= 1 or self.near_zero_fraction < 0:
            raise ValueError("Invalid annotation setting")
        if self.edge_batch < 1 or self.target_batch < 1:
            raise ValueError("Batch sizes must be positive")
        if self.sparse_cell_policy not in ("R_mean", "unavailable"):
            raise ValueError("Unknown sparse-cell policy")


def physical_nodes(nodes):
    """Deduplicate exact tissue–gene identities and retain source-row mapping."""
    unique = []
    lookup = {}
    positions = []
    for node in nodes:
        key = tuple(map(str, node))
        if len(key) != 2:
            raise ValueError("Expected tissue/gene endpoints")
        if key not in lookup:
            lookup[key] = len(unique)
            unique.append(key)
        positions.append(lookup[key])
    return unique, np.asarray(positions, int)


def component_edges(nodes, component):
    if len(set(map(tuple, nodes))) != len(nodes):
        raise ValueError("Resolve duplicate physical nodes before edges")
    if not component.startswith(("TS_", "CT_")):
        raise ValueError("Unknown component")
    required = component[3:].split("__")
    t = np.array([n[0] for n in nodes])
    r, c = np.tril_indices(len(nodes), -1)
    keep = (
        (t[r] == required[0]) & (t[c] == required[0])
        if len(required) == 1
        else np.isin(t[r], required) & np.isin(t[c], required) & (t[r] != t[c])
    )
    return r[keep], c[keep]


def native_power(family, component, all_motor_powers=None):
    if family not in FAMILIES:
        raise ValueError("Unsupported deployed family")
    if family in ("control_derived", "full_cohort"):
        return 3.0 if component.startswith("TS_") else 2.0
    ts = {"VH": 5, "SMA": 7, "QUAD": 13}
    ct = {"VH-SMA": 8, "VH-QUAD": 10, "SMA-QUAD": 10}
    if family == "muscle_all_donors_l4":
        if all_motor_powers is None:
            raise ValueError(
                "All-donor motor native powers require verified provenance"
            )
        ts, ct = all_motor_powers["beta_ts"], all_motor_powers["beta_ct"]
    return float(
        ts[component[3:]]
        if component.startswith("TS_")
        else ct[component[3:].replace("__", "-")]
    )


def edge_cache_key(endpoints, power, protocol):
    # Module membership is deliberately absent. Protocol includes reference,
    # calibration, preprocessing, engine/calibration/guard versions and rotation.
    pair = sorted([list(map(str, e)) for e in endpoints])
    return canonical_hash(
        dict(endpoints=pair, power=float(power), protocol=protocol, version=VERSION)
    )


def calibrate(weights, settings):
    settings.validate()
    w = np.ascontiguousarray(weights, dtype=np.float64)
    if w.ndim != 2 or len(w) != settings.expected_C:
        raise ValueError("Calibration row count differs from declared C; no fallback")
    n = np.isfinite(w).sum(0)
    allfinite = n == settings.expected_C
    safe = np.where(np.isfinite(w), w, 0.0)
    scale = np.max(abs(safe), axis=0) if w.shape[1] else np.zeros(0)
    v = np.divide(safe, scale, out=np.zeros_like(safe), where=scale > 0)
    mu = v.mean(0)
    nu = abs(v).mean(0)
    d = v - mu
    ad = abs(v) - nu
    sd = np.sqrt(np.sum(d * d, axis=0) / (len(w) - 1))
    tau = np.sqrt(np.sum(ad * ad, axis=0) / (len(w) - 1))
    floor = settings.relative_multiplier * np.finfo(np.float64).eps * np.sqrt(len(w))
    count_ok = allfinite & (n >= settings.numerical_min_C)
    numeric = count_ok & (scale > 0) & np.isfinite(tau) & (tau > 0)
    valid = numeric & (tau > floor)
    signed = count_ok & (scale > 0) & np.isfinite(sd) & (sd > floor)
    reasons = np.full(w.shape[1], "eligible", dtype="<U48")
    reasons[~count_ok] = "incomplete_declared_calibration_count"
    reasons[count_ok & ~numeric] = "constant_or_invalid_magnitude"
    reasons[numeric & ~valid] = "relative_roundoff_guard"
    den = np.sum(ad * ad, axis=0)
    influence = (
        np.divide(np.max(ad * ad, axis=0), den, out=np.zeros_like(den), where=den > 0)
        if w.shape[1]
        else np.zeros(0)
    )
    agreement = (np.sign(v) == np.sign(mu)).mean(0)
    if settings.sign_seeds and w.shape[1]:
        means = np.stack(
            [
                v[np.random.default_rng(seed).integers(len(w), size=len(w))].mean(0)
                for seed in settings.sign_seeds
            ]
        )
        lo, hi = np.percentile(means, settings.sign_percentiles, axis=0)
        supported = (lo * hi > 0) & (agreement >= settings.sign_agreement) & count_ok
    else:
        lo = np.full(w.shape[1], np.nan)
        hi = lo.copy()
        supported = np.zeros(w.shape[1], bool)
    return dict(
        n=n,
        scale=scale,
        mu_scaled=mu,
        nu_scaled=nu,
        sd_scaled=sd,
        tau_scaled=tau,
        mu=scale * mu,
        nu=scale * nu,
        sigma=scale * sd,
        tau=scale * tau,
        numeric_valid=numeric,
        valid=valid,
        signed_valid=signed,
        reasons=reasons,
        influence_share=influence,
        influence_flag=influence > 0.5,
        mean_sign_supported=supported,
        mean_sign_interval_low=scale * lo,
        mean_sign_interval_high=scale * hi,
        sign_agreement=agreement,
    )


def edge_attributes(weights, cal):
    w = np.asarray(weights, float)
    v = np.divide(w, cal["scale"], out=np.zeros_like(w), where=cal["scale"] > 0)
    a = np.divide(
        abs(v) - cal["nu_scaled"],
        cal["tau_scaled"],
        out=np.full_like(v, np.nan),
        where=cal["valid"],
    )
    z = np.divide(
        v - cal["mu_scaled"],
        cal["sd_scaled"],
        out=np.full_like(v, np.nan),
        where=cal["signed_valid"],
    )
    return a, z


class Accumulator:
    """Merge edge blocks without changing either G/L denominator."""

    def __init__(self, n):
        self.count = np.zeros(n, int)
        self.zcount = np.zeros(n, int)
        self.G = np.zeros(n)
        self.L = np.zeros(n)
        self.raw = np.zeros(n)
        self.delta = np.zeros(n)
        self.D = np.zeros(n)
        self.absz = np.zeros(n)
        self.loss_bound = np.zeros(n)
        self.partsG = np.zeros((n, len(CLASSES)))
        self.partsL = self.partsG.copy()

    def add(self, w, cal, settings, mask=None, signed_mask=None):
        mask = cal["valid"].copy() if mask is None else cal["valid"] & mask
        sm = (
            cal["signed_valid"].copy()
            if signed_mask is None
            else cal["signed_valid"] & signed_mask
        )
        a, z = edge_attributes(w, cal)
        good = np.isfinite(a) & mask
        zg = np.isfinite(z) & sm
        pos = np.where(good, np.maximum(a, 0), 0)
        neg = np.where(good, np.maximum(-a, 0), 0)
        self.count += good.sum(1)
        self.zcount += zg.sum(1)
        self.G += pos.sum(1)
        self.L += neg.sum(1)
        self.raw += np.where(good, abs(w), 0).sum(1)
        self.delta += np.where(good, abs(w) - cal["nu"], 0).sum(1)
        self.D += np.where(zg, z * z, 0).sum(1)
        self.absz += np.where(zg, abs(z), 0).sum(1)
        bound = np.divide(
            cal["nu_scaled"],
            cal["tau_scaled"],
            out=np.zeros_like(cal["nu_scaled"]),
            where=cal["valid"],
        )
        self.loss_bound += np.where(good, bound, 0).sum(1)
        zero = w == 0
        near = (abs(w) <= settings.near_zero_fraction * cal["nu"]) & ~zero
        uncertain = ~cal["mean_sign_supported"][None, :] & ~zero & ~near
        rev = (
            cal["mean_sign_supported"][None, :]
            & (np.sign(w) != np.sign(cal["mu_scaled"]))
            & ~zero
            & ~near
        )
        pres = ~zero & ~near & ~uncertain & ~rev
        classes = [zero, near, uncertain, pres, rev]
        for k, cls in enumerate(classes):
            self.partsG[:, k] += np.where(cls, pos, 0).sum(1)
            self.partsL[:, k] += np.where(cls, neg, 0).sum(1)

    def finish(
        self, expected_edges, possible_edges, available=None, expected_signed=None
    ):
        n = len(self.G)
        available = (
            np.ones(n, bool) if available is None else np.asarray(available, bool)
        )
        valid = (self.count == expected_edges) & (self.count > 0) & available
        sdvalid = (self.zcount > 0) & available
        if expected_signed is not None:
            sdvalid &= self.zcount == expected_signed
        divide = lambda x: np.divide(
            x, self.count, out=np.full_like(x, np.nan, dtype=float), where=valid
        )
        out = dict(
            G=divide(self.G),
            L=divide(self.L),
            mean_abs_w=divide(self.raw),
            mean_delta_abs=divide(self.delta),
            loss_bound=divide(self.loss_bound),
            D=np.divide(self.D, self.zcount, out=np.full(n, np.nan), where=sdvalid),
            mean_abs_z=np.divide(
                self.absz, self.zcount, out=np.full(n, np.nan), where=sdvalid
            ),
            eligible_edges=self.count,
            calibration_eligible_edges=np.full(n, expected_edges),
            possible_edges=np.full(n, possible_edges),
            signed_eligible_edges=self.zcount,
        )
        out["S"] = out["G"] - out["L"]
        out["A"] = out["G"] + out["L"]
        reason = np.full(n, "ok", dtype="<U64")
        reason[~available] = "missing_component_tissue"
        reason[available & (expected_edges == 0)] = "no_eligible_magnitude_edges"
        reason[available & (expected_edges > 0) & ~valid] = (
            "incomplete_target_edge_coverage"
        )
        out["NA_reason"] = reason
        for k, cls in enumerate(CLASSES):
            out["G_" + cls] = divide(self.partsG[:, k])
            out["L_" + cls] = divide(self.partsL[:, k])
        return out


def aggregate_weights(w, cal, settings, mask=None, available=None, signed_mask=None):
    physical = (
        np.ones(len(cal["valid"]), bool) if mask is None else np.asarray(mask, bool)
    )
    m = cal["valid"] & physical
    sm = cal["signed_valid"] & physical
    if signed_mask is not None:
        sm &= np.asarray(signed_mask, bool)
    acc = Accumulator(len(w))
    acc.add(w, cal, settings, m, sm)
    return acc.finish(int(m.sum()), int(physical.sum()), available, int(sm.sum()))


def fit_reference(X, settings, available=None):
    X = np.asarray(X, float)
    settings.validate()
    if X.ndim != 2 or len(X) != settings.expected_R:
        raise ValueError("Reference count differs from declared R")
    present = (
        np.ones(X.shape, bool)
        if available is None
        else (
            np.broadcast_to(np.asarray(available, bool)[:, None], X.shape)
            if np.asarray(available).ndim == 1
            else np.asarray(available, bool)
        )
    )
    finite = np.isfinite(X) & present
    count = finite.sum(0)
    mean = np.divide(
        np.where(finite, X, 0).sum(0), count, out=np.zeros(X.shape[1]), where=count > 0
    )
    ref = (
        np.where(finite, X, np.where(present, mean, np.nan))
        if settings.sparse_cell_policy == "R_mean"
        else np.where(present, X, np.nan)
    )
    endpoint_valid = np.isfinite(ref).all(0) & present.all(0) & (count > 0)
    ref = np.where(np.isfinite(ref), ref, 0.0)
    centered = ref - ref.mean(0)
    ss = (centered * centered).sum(0)
    endpoint_valid &= ss > 0
    return dict(
        values=ref,
        mean=ref.mean(0),
        impute_mean=mean,
        ss=ss,
        endpoint_valid=endpoint_valid,
        imputed_cells=np.array(
            int((~finite).sum()) if settings.sparse_cell_policy == "R_mean" else 0
        ),
    )


def component_available(values, nodes, rows, columns, declared=None):
    """Resolve authoritative tissue presence; never infer it from module genes.

    ``declared`` is a dataset-wide tissue→donor-mask mapping, node-presence
    matrix, or explicit whole-component donor mask. With missing cells a
    manifest is mandatory. Explicit presence is not overridden by gene values.
    """
    values = np.asarray(values, float)
    if values.ndim != 2 or values.shape[1] != len(nodes):
        raise ValueError("Endpoint input shape mismatch")
    if declared is None:
        if not np.isfinite(values).all():
            raise ValueError(
                "Authoritative tissue-presence manifest required for missing input"
            )
        return np.ones(values.shape, bool)
    if isinstance(declared, dict):
        return np.column_stack([np.asarray(declared[node[0]], bool) for node in nodes])
    available = np.asarray(declared, bool)
    if available.shape == (len(values),):
        return np.broadcast_to(available[:, None], values.shape).copy()
    if available.shape != values.shape:
        raise ValueError("Tissue-presence shape mismatch")
    return available.copy()


def generate_weights(
    reference, values, rows, columns, powers, settings, engine, available=None
):
    values = np.asarray(values, float)
    n = len(values)
    available = (
        np.ones(values.shape, bool)
        if available is None
        else (
            np.broadcast_to(np.asarray(available, bool)[:, None], values.shape)
            if np.asarray(available).ndim == 1
            else np.asarray(available, bool)
        )
    )
    result = np.full((n, len(rows)), np.nan)
    required = np.unique(np.r_[rows, columns])
    local = values[:, required]
    present = available[:, required]
    finite = np.isfinite(local) & present
    imputed = (
        np.where(
            finite, local, np.where(present, reference["impute_mean"][required], np.nan)
        )
        if settings.sparse_cell_policy == "R_mean"
        else np.where(present, local, np.nan)
    )
    # Missingness is edge-local, even when unrelated edges share a batch.
    rr_all = np.searchsorted(required, rows)
    cc_all = np.searchsorted(required, columns)
    edge_valid = (
        present[:, rr_all]
        & present[:, cc_all]
        & np.isfinite(imputed[:, rr_all])
        & np.isfinite(imputed[:, cc_all])
    )
    safe = np.where(np.isfinite(imputed), imputed, 0.0)
    goodedge = reference["endpoint_valid"][rows] & reference["endpoint_valid"][columns]
    if n and goodedge.any():
        rr = np.searchsorted(required, rows[goodedge])
        cc = np.searchsorted(required, columns[goodedge])
        ref = reference["values"][:, required]
        mean = reference["mean"][required]
        centered = ref - mean
        cross = (centered[:, rr] * centered[:, cc]).sum(0)
        weights = engine(
            ref,
            safe,
            rr,
            cc,
            powers[goodedge],
            reference_statistics=(mean, reference["ss"][required], cross),
        )
        weights[~edge_valid[:, goodedge]] = np.nan
        result[:, goodedge] = weights
    return result


@dataclass
class FittedComponent:
    nodes: list
    rows: np.ndarray
    columns: np.ndarray
    powers: np.ndarray
    settings: Settings
    reference: dict
    calibration: dict
    protocol: dict
    R_ids: tuple
    C_ids: tuple
    edge_keys: tuple

    @property
    def protocol_hash(self):
        return canonical_hash(
            dict(protocol=self.protocol, physical_edge_keys=sorted(self.edge_keys))
        )

    def score(
        self,
        X,
        target_ids,
        engine,
        available=None,
        edge_mask=None,
        signed_mask=None,
        role="target",
    ):
        X = np.asarray(X, float)
        target_ids = tuple(map(str, target_ids))
        available = component_available(
            X, self.nodes, self.rows, self.columns, available
        )
        physical = (
            np.ones(len(self.rows), bool)
            if edge_mask is None
            else np.asarray(edge_mask, bool)
        )
        if physical.shape != (len(self.rows),):
            raise ValueError("Physical edge-mask shape mismatch")
        union_available = available[
            :, np.unique(np.r_[self.rows[physical], self.columns[physical]])
        ].all(1)
        if len(X) != len(target_ids) or len(set(target_ids)) != len(target_ids):
            raise ValueError("Target identity alignment/uniqueness failure")
        if role == "target" and (set(target_ids) & (set(self.R_ids) | set(self.C_ids))):
            raise ValueError("Training-role donor cannot be an independent target")
        if role == "calibration_diagnostic" and not set(target_ids) <= set(self.C_ids):
            raise ValueError("Calibration diagnostic requires declared C donors")
        if role not in ("target", "calibration_diagnostic"):
            raise ValueError("Unknown scoring role")
        output = []
        E = len(self.rows)
        mask = self.calibration["valid"] & physical
        sm = self.calibration["signed_valid"] & physical
        if signed_mask is not None:
            sm &= np.asarray(signed_mask, bool)
        for start in range(0, len(X), self.settings.target_batch):
            end = min(len(X), start + self.settings.target_batch)
            acc = Accumulator(end - start)
            for lo in range(0, E, self.settings.edge_batch):
                hi = min(E, lo + self.settings.edge_batch)
                sl = slice(lo, hi)
                cal = {k: v[sl] for k, v in self.calibration.items()}
                w = generate_weights(
                    self.reference,
                    X[start:end],
                    self.rows[sl],
                    self.columns[sl],
                    self.powers[sl],
                    self.settings,
                    engine,
                    available[start:end],
                )
                acc.add(w, cal, self.settings, mask[sl], sm[sl])
            output.append(
                acc.finish(
                    int(mask.sum()),
                    int(physical.sum()),
                    union_available[start:end],
                    int(sm.sum()),
                )
            )
        if not output:
            return {}
        return {k: np.concatenate([o[k] for o in output]) for k in output[0]}


def fit_component(
    X_R,
    X_C,
    nodes,
    rows,
    columns,
    powers,
    R_ids,
    C_ids,
    settings,
    provenance,
    engine,
    R_available=None,
    C_available=None,
):
    settings.validate()
    R_ids = tuple(map(str, R_ids))
    C_ids = tuple(map(str, C_ids))
    rows = np.asarray(rows, int)
    columns = np.asarray(columns, int)
    powers = np.broadcast_to(np.asarray(powers, float), (len(rows),)).copy()
    if (
        len(R_ids) != settings.expected_R
        or len(C_ids) != settings.expected_C
        or len(X_C) != settings.expected_C
    ):
        raise ValueError("Declared protocol count mismatch")
    if (
        len(set(R_ids)) != len(R_ids)
        or len(set(C_ids)) != len(C_ids)
        or set(R_ids) & set(C_ids)
    ):
        raise ValueError("R/C role overlap or duplicate donor")
    if len(set(map(tuple, nodes))) != len(nodes):
        raise ValueError("Duplicate physical nodes unresolved")
    edges = [
        tuple(sorted((tuple(nodes[r]), tuple(nodes[c])))) for r, c in zip(rows, columns)
    ]
    if any(r == c for r, c in zip(rows, columns)) or len(set(edges)) != len(edges):
        raise ValueError("Duplicate/self physical edge")
    if not np.isfinite(powers).all() or (powers <= 0).any():
        raise ValueError("Invalid functional power")
    if np.asarray(X_R).shape[1] != len(nodes) or np.asarray(X_C).shape[1] != len(nodes):
        raise ValueError("Endpoint alignment failure")
    R_available = component_available(X_R, nodes, rows, columns, R_available)
    C_available = component_available(X_C, nodes, rows, columns, C_available)
    # Bootstrap draws index positions. Canonical donor order makes finite
    # Monte Carlo sign annotations invariant to caller row ordering as well.
    ro = np.argsort(np.asarray(R_ids), kind="stable")
    co = np.argsort(np.asarray(C_ids), kind="stable")
    X_R = np.asarray(X_R)[ro]
    X_C = np.asarray(X_C)[co]
    R_available = R_available[ro]
    C_available = C_available[co]
    R_ids = tuple(R_ids[i] for i in ro)
    C_ids = tuple(C_ids[i] for i in co)
    reference = fit_reference(X_R, settings, R_available)
    parts = []
    for lo in range(0, len(rows), settings.edge_batch):
        sl = slice(lo, min(len(rows), lo + settings.edge_batch))
        wc = generate_weights(
            reference,
            X_C,
            rows[sl],
            columns[sl],
            powers[sl],
            settings,
            engine,
            C_available,
        )
        parts.append(calibrate(wc, settings))
    calibration = (
        {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        if parts
        else calibrate(np.empty((settings.expected_C, 0)), settings)
    )
    protocol = dict(provenance)
    protocol.update(
        version=VERSION,
        R_membership=member_hash(R_ids),
        C_membership=member_hash(C_ids),
        R_count=len(R_ids),
        C_count=len(C_ids),
        row_order="canonical identity string; preserves frozen membership, stabilizes bootstrap indexing",
        calibration="edge-specific absolute sample SD ddof1",
        settings={
            k: v
            for k, v in asdict(settings).items()
            if k not in ("edge_batch", "target_batch")
        },
        engine="signed Pearson powered add-one LIONESS; no M5/MDC normalization",
    )
    keys = []
    for r, c, pair, power in zip(rows, columns, edges, powers):
        endpoints = sorted((int(r), int(c)), key=lambda i: tuple(nodes[i]))
        source = hashlib.sha256(
            np.ascontiguousarray(
                np.r_[X_R[:, endpoints], X_C[:, endpoints]], dtype=np.float64
            ).tobytes()
            + np.packbits(
                np.r_[R_available[:, endpoints], C_available[:, endpoints]]
            ).tobytes()
        ).hexdigest()
        keys.append(
            edge_cache_key(
                pair, power, {**protocol, "endpoint_input_and_presence_sha256": source}
            )
        )
    keys = tuple(keys)
    return FittedComponent(
        list(map(tuple, nodes)),
        rows,
        columns,
        powers,
        settings,
        reference,
        calibration,
        protocol,
        R_ids,
        C_ids,
        keys,
    )
