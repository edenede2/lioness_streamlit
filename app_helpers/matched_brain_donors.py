"""Approved, module-bounded donor reads for the 247-module matched partition."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from app_helpers.camel import FEATURES,FrozenRINT,feature_values
from app_helpers.correlations import (calculate_correlations,calculate_categorical_associations,
    add_across_module_fdr,add_categorical_across_module_fdr)
from app_helpers.streaming_associations import _serialized_heavy_read
from app_helpers import matched_brain_results as R

ROOT=Path(__file__).resolve().parents[1]/"data/matched_brain_donors"
BALANCED="Balanced incomplete-region validation (AD–Control)"
PROFILE="Discovery-demography-matched incomplete-region validation (AD–Control)"
KEYS=["module","component","component_label"]


def manifest():
    p=ROOT/"manifest.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def available():
    m=manifest()
    return m.get("status")=="complete" and m.get("module_set")=="matched_control_brain_l4"


def _entry(kind,module=None):
    if module is not None and int(module) not in manifest().get("modules",[]):raise ValueError("Unknown module")
    return next(e for e in manifest()["files"] if e["kind"]==kind and (module is None or e.get("module")==int(module)))


@st.cache_data(show_spinner=False,max_entries=8,ttl=1800)
def _read(path,version,predicates=()):
    p=Path(path)
    if p.is_absolute() or ".." in p.parts:raise ValueError("Invalid donor shard")
    return pd.read_parquet(ROOT/p,filters=list(predicates) or None,use_threads=False)


def _load(kind,module=None,filters=()):
    e=_entry(kind,module)
    return _read(e["path"],e["sha256"],tuple(filters))


def metadata():
    return _load("metadata")


def load_scores(module,*,cohort,components,feature="connectivity",rule="all",normalization="standard_pruned",scale="raw"):
    """Read at most one module and reconstruct scores using frozen mappings."""
    if feature not in (*FEATURES,"eigengene"):raise ValueError("Unknown feature")
    if scale not in {"raw","asinh","rint"}:raise ValueError("Unknown transformation")
    if rule not in R.RULES:raise ValueError("Unknown edge mask")
    if feature!="eigengene" and normalization not in {"standard_pruned","retained_edge"}:raise ValueError("Unknown normalization")
    allowed=set(manifest()["cohort_roles"])|{BALANCED,PROFILE}
    if cohort not in allowed:raise ValueError("Unknown cohort")
    if not components or set(components)-set(R.COMPONENTS):raise ValueError("Select valid components")
    incomplete=cohort in {"Incomplete-region validation",BALANCED,PROFILE}
    if incomplete and set(components)&{"all","TS","CT"}:
        raise ValueError("Incomplete-region donors have no three-tissue pooled scores")
    if feature=="eigengene" and any(not c.startswith("TS_") for c in components):
        raise ValueError("Same-module eigengenes are available per tissue only")
    role="Incomplete-region validation" if cohort in {BALANCED,PROFILE} else cohort
    predicates=(("cohort_scope","=",role),("component","in",list(components)))
    if feature=="eigengene":
        f=_load("eigengenes",module,predicates).copy()
        f["metric_value"]=f["metric_"+scale]
    else:
        f=_load("scores",module,(*predicates,("differential_edge_rule","=",rule))).copy()
        if f.empty:return f
        f["metric_value"]=np.nan
        mappings=_load("transformations",module,(("component","in",list(components)),
            ("differential_edge_rule","=",rule),("metric_family","=",feature),("score_normalization","=",normalization))) if scale!="raw" else None
        for component,index in f.groupby("component",observed=True).groups.items():
            raw=feature_values(f.loc[index],feature,normalization)
            if scale=="raw":values=raw
            else:
                mapping=mappings.loc[mappings.component.eq(component)]
                if len(mapping)!=1:raise ValueError("Frozen transformation absent or ambiguous")
                row=mapping.iloc[0]
                if row.fit_cohort!="Discovery":raise ValueError("Unfrozen transformation")
                values=np.arcsinh(raw/row.asinh_scale) if scale=="asinh" else FrozenRINT(np.asarray(row.rint_x),np.asarray(row.rint_z)).transform(raw)
            f.loc[index,"metric_value"]=values
        f["metric_family"]=feature;f["score_normalization"]=normalization
    if cohort in {BALANCED,PROFILE}:
        members=_load("membership",filters=(("cohort_scope","=",cohort),("component","in",list(components))))
        f=f.merge(members[["sample_id","component"]],on=["sample_id","component"],validate="one_to_one")
        f["cohort_scope"]=cohort
    if f.empty:return f
    if f.duplicated(["sample_id","component"]).any():raise ValueError("Duplicated plotted donors")
    f=f.merge(metadata(),on="sample_id",validate="many_to_one")
    f["component_label"]=f.component.map(R.COMPONENTS)
    f["scale"]=scale
    return f.sort_values(["component","sample_id"],kind="stable").reset_index(drop=True)


def filter_donors(frame,diagnoses,grouping,levels):
    f=frame.loc[frame.diagnosis_group.isin(diagnoses)].copy()
    if grouping!="__all__":f=f.loc[f[grouping].isin(levels)]
    return f


def numeric_statistics(frame,outcome,grouping,levels,min_n=10):
    if frame.empty:return pd.DataFrame()
    from app_helpers.data import NUMERIC_OUTCOMES
    if outcome not in NUMERIC_OUTCOMES:raise ValueError("Nominal category codes cannot enter a correlation")
    pieces=[]
    for level in (["__all__"] if grouping=="__all__" else levels):
        part=frame.copy() if grouping=="__all__" else frame.loc[frame[grouping].eq(level)].copy()
        part["grouping_level"]=str(level);pieces.append(part)
    if grouping!="__all__":
        pooled=frame.copy();pooled["grouping_level"]="__pooled__";pieces.append(pooled)
    if not pieces:return pd.DataFrame()
    result=calculate_correlations(pd.concat(pieces,ignore_index=True),[*KEYS,"grouping_level"],[outcome],min_group_n=min_n)
    # Never mistake the helper's displayed-test family for module-set BH.
    result=result.drop(columns=[c for c in result if c.endswith("fdr_displayed_family")])
    result["grouping_variable"]=grouping
    return result


def selected_statistics(frame,*,outcome,grouping,levels,min_n,categorical=False):
    if categorical:
        return calculate_categorical_associations(frame,KEYS,category_column=outcome,min_group_n=min_n)
    return numeric_statistics(frame,outcome,grouping,levels,min_n)


def family_statistics(*,cohort,components,feature,rule,normalization,scale,diagnoses,grouping,levels,outcome,min_n,categorical=False):
    """An explicit, serialized all-module action; cache only compact tests."""
    # The bundle hashes invalidate cached statistics after any data replacement.
    version=tuple(e["sha256"] for e in manifest()["files"])
    return _family_statistics(version,cohort,tuple(components),feature,rule,normalization,scale,
        tuple(diagnoses),grouping,tuple(levels),outcome,int(min_n),bool(categorical))


@st.cache_data(show_spinner=False,max_entries=3,ttl=1800)
@_serialized_heavy_read
def _family_statistics(version,cohort,components,feature,rule,normalization,scale,diagnoses,grouping,levels,outcome,min_n,categorical):
    rows=[]
    for module in manifest()["modules"]:
        f=load_scores(module,cohort=cohort,components=components,feature=feature,rule=rule,normalization=normalization,scale=scale)
        if f.empty:continue
        f=filter_donors(f,diagnoses,grouping,levels)
        if not f.empty:rows.append(selected_statistics(f,outcome=outcome,grouping=grouping,levels=levels,min_n=min_n,categorical=categorical))
    if not rows:return pd.DataFrame()
    result=pd.concat(rows,ignore_index=True)
    if categorical:return add_categorical_across_module_fdr(result,family_columns=["component","outcome"])
    return add_across_module_fdr(result,family_columns=["component","outcome","grouping_variable","grouping_level"])
