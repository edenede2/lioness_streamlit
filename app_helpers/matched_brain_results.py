"""Narrow, local-first reads of complete precomputed matched-Control results."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]/"data/matched_brain_results"
COMPONENTS={"TS_DLPFC":"DLPFC","CT_AC__DLPFC":"AC–DLPFC","CT_DLPFC__PCGBA23":"DLPFC–PCG",
    "TS_AC":"AC","TS_PCGBA23":"PCG","CT_AC__PCGBA23":"AC–PCG","TS":"TS pooled","CT":"CT pooled","all":"All edges"}
COHORTS={
    "Held-out validation":"Held-out matched donors",
    "Discovery-demography-matched incomplete-region validation (AD–Control)":"Incomplete: matched to discovery demographics",
    "Balanced incomplete-region validation (AD–Control)":"Incomplete: AD–Control matched",
    "Incomplete-region validation":"Incomplete: all eligible donors"}
RULES={"all":"All edges","ad_higher":"Hedges’ g ≥ 0.40: AD higher",
    "control_higher":"Hedges’ g ≤ −0.40: Control higher","either":"|Hedges’ g| ≥ 0.40: either direction"}
TESTS={"diagnosis_label":"Unrestricted diagnosis-label permutation",
    "matched_pair_label":"Demographic matched-pair label swaps",
    "score_diagnosis_label":"Unrestricted diagnosis-label permutation",
    "score_matched_pair_label":"Demographic matched-pair label swaps",
    "covariate_adjusted_association":"Demographic-adjusted association"}
FILTER_FIELDS={"test_type","metric_family","score_normalization","differential_edge_rule","scale","phenotype","correlation_method","module"}


def manifest():
    p=ROOT/"manifest.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def available():
    m=manifest()
    return m.get("status")=="complete" and m.get("module_set")=="matched_control_brain_l4"


def _path(relative):
    p=Path(relative)
    if p.is_absolute() or ".." in p.parts: raise ValueError("Invalid result path")
    result=ROOT/p
    if not result.resolve().is_relative_to(ROOT.resolve()): raise ValueError("Result path escapes bundle")
    return result


def entries():
    return [x for x in manifest().get("files",[]) if x["kind"]=="camel"]


def options(field,*,test=None,feature=None):
    values=set()
    for e in entries():
        if test and test not in e["choices"]["test_type"]: continue
        if feature and feature not in e["choices"]["metric_family"]: continue
        values.update(e["choices"].get(field,[]))
    return sorted(values)


@st.cache_data(show_spinner=False,max_entries=12,ttl=1800)
def _read(path_key,version,selection):
    filters=list(selection)
    return pd.read_parquet(_path(path_key),filters=filters,use_threads=False)


def camel(cohort,component,**filters):
    if not filters or not {"test_type","metric_family","score_normalization","differential_edge_rule","scale","phenotype","correlation_method"}<=set(filters):
        raise ValueError("A full narrow statistical context is required")
    if set(filters)-FILTER_FIELDS: raise ValueError("Unknown CAMEL predicate")
    e=next((e for e in entries() if e["cohort_scope"]==cohort and e["component"]==component),None)
    if e is None: return pd.DataFrame()
    result=_read(e["path"],e["sha256"],tuple((k,"=",v) for k,v in sorted(filters.items())))
    if len(result)>manifest()["module_count"]: raise ValueError("Unexpected duplicate/non-narrow test selection")
    return result


@st.cache_data(show_spinner=False,max_entries=3,ttl=1800)
def _table(path_key,version):
    return pd.read_parquet(_path(path_key))


def table(kind):
    e=next((e for e in manifest().get("files",[]) if e["kind"]==kind),None)
    return _table(e["path"],e["sha256"]) if e else pd.DataFrame()


def kegg(rule="all",regional=False):
    name=f"kegg_{rule}__{'regional' if regional else 'tissue_expanded'}.parquet"
    e=next((e for e in manifest().get("files",[]) if e["path"]==name),None)
    return _table(e["path"],e["sha256"]) if e else pd.DataFrame()


def audits():
    return json.loads((ROOT/"audits.json").read_text()) if available() else {}


def selected_modules(frame,k=20,significance=None):
    """Display filtering only; retain full-family p/FDR and family counts."""
    f=frame.copy()
    if significance is not None: f=f.loc[f.q_permutation.lt(significance)]
    f["absolute_effect"]=f.effect.abs()
    f=f.sort_values(["absolute_effect","module"],ascending=[False,True],kind="stable")
    return f.head(k) if k else f
