"""Narrow readers for the disjoint incomplete-region CAMEL validation cohort.

Only compact manifests are cached. A read materializes one module and one mask;
the RINT/asinh maps always come from original donors, never displayed donors.
"""
from __future__ import annotations
import json
import os
from functools import partial
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
from app_helpers.camel import COMPONENT_LABELS, FEATURES, FrozenRINT, feature_values
from app_helpers.drive_data import DATA_DIR, data_path_available, ensure_data_path
from app_helpers.validation_datasets import DATASETS, dataset, manifest_matches


def root(module_set="control_derived", method="control_anchored") -> Path:
    # Explicit offline preview only. Production uses the indexed Drive backend.
    config=dataset(module_set,method)
    override=os.environ.get("PARTITION_VALIDATION_DATA_ROOT")
    if (module_set,method)==("control_derived","control_anchored"):
        override=os.environ.get("CAMEL_VALIDATION_DATA_ROOT",override)
    return Path(override) if override else DATA_DIR/config.relative_root(method)


def manifest(module_set="control_derived", method="control_anchored") -> dict:
    p=root(module_set,method)/"manifest.json"
    result=json.loads(ensure_data_path(p).read_text()) if data_path_available(p) else {}
    if result and not manifest_matches(result,module_set,method):
        raise ValueError("Validation manifest belongs to a different partition or method")
    return result


def available(rule: str = "all", module_set="control_derived", method="control_anchored") -> bool:
    m=manifest(module_set,method)
    return (m.get("status")=="complete" and rule in m.get("completed_rules",[])
            and (m.get("backend_verified",False) or bool(os.environ.get("CAMEL_VALIDATION_DATA_ROOT"))
                 or bool(os.environ.get("PARTITION_VALIDATION_DATA_ROOT"))))


def completed_effect_options(module_set="control_derived", method="control_anchored") -> list[dict]:
    result=[]
    for rule in manifest(module_set,method).get("completed_rules",[]):
        if not rule.startswith("ad_control_discovery_effect__"): continue
        _,statistic,threshold,direction=rule.split("__")
        if threshold.startswith("fixed_"):
            mode="fixed"; value=float(threshold.removeprefix("fixed_"))
        elif threshold.startswith("module_top_") and threshold.endswith("pct"):
            mode="module_top_fraction"; value=float(threshold.removeprefix("module_top_").removesuffix("pct"))/100
        else: raise ValueError("Unknown completed validation effect mask")
        result.append(dict(rule=rule,statistic=statistic,mode=mode,value=value,direction=direction))
    return result


def validate_identity(module_set, estimator, method, feature=None, rule="all"):
    dataset(module_set,method)
    if (module_set,method)!=("control_derived","control_anchored") and not manifest(module_set,method):
        raise ValueError("This partition/method validation catalog is not available")
    if estimator!="lioness":
        raise ValueError("Incomplete-region validation is available only for LIONESS/CAMEL")
    if feature is not None and feature not in FEATURES:
        raise ValueError("No eigengene/BONOBO catalog exists for this validation cohort")
    if rule!="all" and not str(rule).startswith("ad_control_discovery_effect__"):
        raise ValueError("FDR edge filtering is not part of the CAMEL validation extension")
    ready=available(rule) if (module_set,method)==("control_derived","control_anchored") else available(rule,module_set,method)
    if not ready:
        raise ValueError("This validation score/mask catalog has not completed validation")


def metadata(module_set="control_derived", method="control_anchored") -> pd.DataFrame:
    result=pd.read_parquet(ensure_data_path(root(module_set,method)/"sample_metadata.parquet"))
    if len(result)!=dataset(module_set,method).validation_donors or result.sample_id.duplicated().any() or {"donor","projid"}&set(result):
        raise ValueError("Validation metadata failed its privacy/grain checks")
    return result


def summaries(module: int, rule="all", component=None, *, module_set="control_derived", method="control_anchored") -> pd.DataFrame:
    filters=[("differential_edge_rule","=",rule)]
    if component is not None: filters.append(("component","=",component))
    path=ensure_data_path(root(module_set,method)/f"modules/M{int(module)}.parquet",filters)
    result=pd.read_parquet(path,filters=filters)
    result["scope"]=result.component
    result["scope_label"]=result.component.map(dataset(module_set,method).labels)
    result["weight_mean"]=np.divide(result.signed_weight_sum,result.n_retained_edges,
        out=np.zeros(len(result)),where=result.n_retained_edges.to_numpy()>0)
    result["cancellation_index"]=1-np.divide(np.abs(result.signed_weight_sum),result.absolute_weight_sum,
        out=np.ones(len(result)),where=result.absolute_weight_sum.to_numpy()>0)
    result["estimator"]="lioness"; result["network_method"]=method; result["edge_rule"]="all"
    return result


def scores(module: int, feature: str | None="connectivity", rule="all", normalization="standard_pruned",
           component: str | None=None, scale: str | None=None, embedded=True, *,
           module_set="control_derived", method="control_anchored") -> pd.DataFrame:
    if module is None: raise ValueError("Read validation scores one module at a time")
    source=summaries(module,rule,component,module_set=module_set,method=method)
    filters=[("differential_edge_rule","=",rule),("score_normalization","=",normalization)]
    if feature is not None: filters.append(("metric_family","=",feature))
    if component is not None: filters.append(("component","=",component))
    if scale=="raw":
        # Raw requires no transformation parameters or curve-file download.
        maps=pd.DataFrame([dict(component=c,metric_family=f,asinh_scale=np.nan)
            for c in source.component.unique() for f in ([feature] if feature else FEATURES)])
    else:
        scales=root(module_set,method)/"scales.parquet"
        if scale=="asinh" and data_path_available(scales):
            filters.append(("module","=",int(module)))
            path=ensure_data_path(scales,filters)
        else:
            path=ensure_data_path(root(module_set,method)/f"rint/M{int(module)}.parquet",filters)
        columns=["component","metric_family","asinh_scale"] if scale=="asinh" else None
        maps=pd.read_parquet(path,filters=filters,columns=columns)
    if "asinh_scale" not in maps: raise ValueError("Incomplete frozen transformation maps; no validation-fitted substitute is allowed")
    pieces=[]
    for (comp,family),mapping in maps.groupby(["component","metric_family"],sort=False):
        if len(mapping)!=1: raise ValueError("Ambiguous frozen transformation identity")
        record=mapping.iloc[0]
        part=source.loc[source.component.eq(comp)].copy()
        raw=feature_values(part,family,normalization)
        part=part[["sample_id","module","component","n_possible_edges","n_retained_edges"]]
        part["metric_family"]=family
        if scale in (None,"raw"): part["metric_raw"]=raw
        if scale in (None,"asinh"): part["metric_asinh"]=np.arcsinh(raw/float(record.asinh_scale))
        if scale in (None,"rint"): part["metric_rint"]=FrozenRINT(np.asarray(record.x,float),np.asarray(record.z,float)).transform(raw)
        part["metric_scale"]=float(record.asinh_scale)
        part["component_class"]="CT" if comp.startswith("CT_") else "TS"
        part["component_label"]=dataset(module_set,method).labels[comp]
        part["lioness_method"]=method; part["score_framework"]=dataset(module_set,method).framework(method)
        part["score_normalization"]=normalization; part["differential_edge_rule"]=rule
        pieces.append(part)
    result=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame()
    if embedded and not result.empty:
        result=result.merge(metadata(module_set,method),on="sample_id",how="left",validate="many_to_one")
    return result


def statistics(rule="all", normalization="standard_pruned", *, module_set="control_derived", method="control_anchored", **selections) -> pd.DataFrame:
    path=root(module_set,method)/"statistics"/rule/f"{normalization}.parquet"
    if not data_path_available(path):
        raise ValueError("Validated robustness statistics are not available for this validation mask yet")
    filters=[(name,"=",value) for name,value in selections.items() if value is not None]
    columns=None
    if selections.get("module") is None:
        columns=["module","metric_family","component","component_label","diagnosis_group","phenotype","n","lioness_method",
                 "r_raw","p_raw","r_asinh","p_asinh","r_rint","p_rint","rho","p_spearman",
                 "rho_raw","p_spearman_raw","rho_asinh","p_spearman_asinh","rho_rint","p_spearman_rint"]
    return pd.read_parquet(ensure_data_path(path,filters),filters=filters,columns=columns)


def mdc(rule="all", *, module_set="control_derived", method="control_anchored") -> pd.DataFrame:
    path=root(module_set,method)/"mdc/resolved.parquet"
    if not data_path_available(path): return pd.DataFrame()
    return pd.read_parquet(ensure_data_path(path,[("differential_edge_rule","=",rule)]),
        filters=[("differential_edge_rule","=",rule)])


def for_dataset(module_set="control_derived", method="control_anchored"):
    """A bound reader, not mutable process/session-global dataset selection."""
    config=dataset(module_set,method)
    return SimpleNamespace(config=config, module_set=module_set, method=method,
        **{name:partial(globals()[name],module_set=module_set,method=method)
           for name in ("root","manifest","available","completed_effect_options","metadata","summaries","scores","statistics","mdc")})


def available_datasets():
    result=[]
    for key,config in DATASETS.items():
        for method in config.methods:
            source=for_dataset(key,method)
            try:
                if source.manifest(): result.append((key,method))
            except ValueError:
                # A run-specific offline override belongs to one identity.
                continue
    return result
