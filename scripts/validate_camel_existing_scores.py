#!/usr/bin/env python3
"""Compare sampled original donor scores with the unchanged historical R catalog."""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
from pathlib import Path
import sys
APP=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(APP))
import numpy as np
import pandas as pd
from app_helpers.camel import frozen_edge_weights, FEATURES
from run_camel_validation import RUN, ORIGINAL, load_inputs, module_source, atomic_json


def main():
    x=load_inputs(); report=[]
    old_path=next(ORIGINAL.parent.glob('*tissue_resolved_component_data.parquet'))
    selected=x["original"].groupby("diagnosis_group",sort=False).head(3)
    donors=selected.donor.tolist()
    positions=np.array([x["controls"].index(d) if d in x["controls"] else -1 for d in donors])
    for module in (436,356,121,771):
        a,full,ri,ci,_,_=module_source(module)
        tissues=a.public_tissue.to_numpy()
        weights=frozen_edge_weights(full.loc[x["controls"]].to_numpy(float),full.loc[donors].to_numpy(float),
            ri,ci,np.where(tissues[ri]==tissues[ci],3,2),reference_positions=positions)
        historical=pd.read_parquet(old_path,filters=[("module","=",module),("donor","in",donors)],
            columns=["donor","component","metric_family","metric_raw"])
        errors=[]
        for row in historical.itertuples(index=False):
            component=row.component.replace("MFBA9BA46","DLPFC")
            if component.startswith("TS_"):
                t=component[3:]; mask=(tissues[ri]==t)&(tissues[ci]==t)
            else:
                left,right=component[3:].split("__")
                mask=((tissues[ri]==left)&(tissues[ci]==right))|((tissues[ri]==right)&(tissues[ci]==left))
            w=weights[donors.index(row.donor),mask]
            pos=w[w>0]; neg=w[w<0]
            v=dict(connectivity=2*w.sum()/len(a),positive_density=pos.mean() if len(pos) else 0.,
                negative_density=neg.mean() if len(neg) else 0.,abs_sum=np.abs(w).sum(),
                positive_abs_sum=pos.sum(),negative_abs_sum=-neg.sum())
            expected=v[row.metric_family] if len(w) else np.nan
            if np.isnan(expected) and pd.isna(row.metric_raw): continue
            if not np.isfinite(expected) or not np.isfinite(row.metric_raw):
                raise ValueError("Historical structural availability does not agree")
            error=abs(expected-row.metric_raw)/max(1.,abs(row.metric_raw))
            errors.append(error)
        maximum=float(max(errors,default=0.))
        if maximum>1e-8: raise ValueError(f"M{module}: historical R score reproduction failed ({maximum})")
        report.append(dict(module=module,donors=len(donors),tested_values=len(errors),max_relative_error=maximum))
    atomic_json(dict(passed=True,source=str(old_path.relative_to(APP.parents[1])),checks=report),RUN/'existing_score_reproduction.json')
    print({'passed':True,'modules':len(report),'maximum_relative_error':max(q['max_relative_error'] for q in report)})


if __name__=='__main__': main()
