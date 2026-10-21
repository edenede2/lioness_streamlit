"""Vectorized, pairwise-complete robustness statistics for saved score blocks.

Vectorizing outcomes changes neither the tests nor their module-only BH
families. No cross-donor imputation or fitted transformation occurs here.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as student_t


def paired(x,y):
    a=np.broadcast_to(np.asarray(x,float),y.shape)
    good=np.isfinite(a)&np.isfinite(y); n=good.sum(axis=0)
    a=np.where(good,a,0.); b=np.where(good,y,0.)
    am=np.divide(a.sum(axis=0),n,out=np.zeros(y.shape[1]),where=n>0)
    bm=np.divide(b.sum(axis=0),n,out=np.zeros(y.shape[1]),where=n>0)
    ac=np.where(good,a-am,0.); bc=np.where(good,b-bm,0.)
    av=(ac*ac).sum(axis=0); bv=(bc*bc).sum(axis=0); covariance=(ac*bc).sum(axis=0)
    amin=np.min(a,axis=0,where=good,initial=np.inf); amax=np.max(a,axis=0,where=good,initial=-np.inf)
    bmin=np.min(b,axis=0,where=good,initial=np.inf); bmax=np.max(b,axis=0,where=good,initial=-np.inf)
    eligible=(n>=10)&(amax>amin)&(bmax>bmin)&(av>0)&(bv>0)
    denominator=np.sqrt(av*bv)
    r=np.divide(covariance,denominator,out=np.full(y.shape[1],np.nan),where=eligible)
    r=np.clip(r,-1,1); residual=np.maximum(0.,1-r*r)
    statistic=np.divide(np.abs(r)*np.sqrt(np.maximum(n-2,0)),np.sqrt(residual),
        out=np.full_like(r,np.inf),where=residual>np.finfo(float).eps)
    p=np.where(eligible,2*student_t.sf(statistic,np.maximum(n-2,1)),np.nan)
    return dict(a=a,b=b,good=good,n=n,eligible=eligible,r=r,p=p,av=av,bv=bv,cov=covariance,
        am=am,bm=bm,amin=amin,amax=amax,ac=ac,bc=bc)


def leave_one_out(pair):
    a,b,n=pair["a"],pair["b"],pair["n"]
    sx,sy=a.sum(axis=0),b.sum(axis=0)
    sx2,sy2=(a*a).sum(axis=0),(b*b).sum(axis=0); sxy=(a*b).sum(axis=0)
    denominator=np.sqrt(np.maximum(sx2-sx*sx/np.maximum(n,1),0)*np.maximum(sy2-sy*sy/np.maximum(n,1),0))
    full=np.divide(sxy-sx*sy/np.maximum(n,1),denominator,out=np.full_like(sx,np.nan),where=denominator>0)
    nm=np.maximum(n-1,1)
    cov=sxy-a*b-(sx-a)*(sy-b)/nm
    vx=sx2-a*a-(sx-a)**2/nm; vy=sy2-b*b-(sy-b)**2/nm
    den=np.sqrt(np.maximum(vx,0)*np.maximum(vy,0))
    valid=pair["good"]&(den>0)&pair["eligible"]
    estimates=np.divide(cov,den,out=np.full_like(cov,np.nan),where=valid)
    count=np.isfinite(estimates).sum(axis=0); ok=(count>0)&np.isfinite(full)&pair["eligible"]
    delta=np.max(np.abs(estimates-full),axis=0,where=np.isfinite(estimates),initial=-np.inf)
    minimum=np.min(np.abs(estimates),axis=0,where=np.isfinite(estimates),initial=np.inf)
    flips=np.any((np.sign(estimates)!=np.sign(full))&np.isfinite(estimates),axis=0).astype(float)
    return tuple(np.where(ok,v,np.nan) for v in (delta,minimum,flips))


def robust_rows(base, frame, outcomes):
    y=np.column_stack([pd.to_numeric(frame[o],errors="coerce").to_numpy(float) for o in outcomes])
    raw=paired(frame.metric_raw.to_numpy(float)[:,None],y)
    rows=[{**base,"phenotype":o,"n":int(raw["n"][j]),"eligible":bool(raw["eligible"][j]),
        "unavailable_reason":"","scale":float(frame.metric_scale.iloc[0]),"analysis_subset":"all_donors",
        "metric_zero_variance":bool(raw["amax"][j]<=raw["amin"][j])} for j,o in enumerate(outcomes)]
    for j,row in enumerate(rows):
        if row["n"]<10: row["unavailable_reason"]="n < 10"
        elif row["metric_zero_variance"]: row["unavailable_reason"]="constant network score"
        elif not row["eligible"]: row["unavailable_reason"]="constant outcome"
    for scale in ("raw","asinh","rint"):
        values=frame[f"metric_{scale}"].to_numpy(float)[:,None]
        # Use the same raw-outcome pairwise subset as the historical helper.
        pair=paired(np.where(raw["good"],values,np.nan),y)
        pair["eligible"] &= raw["eligible"]
        pair["r"]=np.where(pair["eligible"],pair["r"],np.nan)
        pair["p"]=np.where(pair["eligible"],pair["p"],np.nan)
        xa=np.where(pair["good"],pair["a"],np.nan)
        ya=np.where(pair["good"],pair["b"],np.nan)
        rho=paired(rankdata(xa,axis=0,nan_policy="omit"),rankdata(ya,axis=0,nan_policy="omit"))
        for field in ("r","p"): rho[field]=np.where(pair["eligible"],rho[field],np.nan)
        sd=np.sqrt(np.divide(pair["av"],pair["n"]-1,out=np.full(len(outcomes),np.nan),where=pair["n"]>1))
        third=np.divide((pair["ac"]**3).sum(axis=0),pair["n"],out=np.full(len(outcomes),np.nan),where=pair["n"]>0)
        skew=np.divide(third,sd**3,out=np.full(len(outcomes),np.nan),where=(sd>0)&(pair["n"]>=3))
        for j,row in enumerate(rows):
            reason=("n < 10" if pair["n"][j]<10 else "constant transformed score" if pair["amax"][j]<=pair["amin"][j]
                else "constant outcome" if not pair["eligible"][j] else "")
            row.update({f"r_{scale}":pair["r"][j],f"p_{scale}":pair["p"][j],
                f"rho_{scale}":rho["r"][j],f"p_spearman_{scale}":rho["p"][j],f"skew_{scale}":skew[j],
                f"eligible_{scale}":bool(pair["eligible"][j]),f"unavailable_reason_{scale}":reason})
        if scale in ("asinh","rint"):
            loo=leave_one_out(pair)
            for j,row in enumerate(rows):
                for suffix,array in zip(("max_delta","min_abs","sign_flip"),loo): row[f"loo_{scale}_{suffix}"]=array[j]
        if scale=="asinh":
            # quantile must ignore outcomes missing from this exact pair.
            limits=np.full((2,len(outcomes)),np.nan)
            valid_columns=np.flatnonzero(np.isfinite(xa).any(axis=0))
            if len(valid_columns):
                limits[:,valid_columns]=np.nanquantile(xa[:,valid_columns],[.025,.975],axis=0,method="median_unbiased")
            clipped=np.clip(xa,limits[0],limits[1]); winsor=paired(clipped,y)
            for j,row in enumerate(rows):
                row["r_winsor"]=winsor["r"][j] if raw["eligible"][j] else np.nan
                row["p_winsor"]=winsor["p"][j] if raw["eligible"][j] else np.nan
        if scale=="rint":
            slopes=np.divide(pair["cov"],pair["av"],out=np.full(len(outcomes),np.nan),where=pair["eligible"])
            for j,row in enumerate(rows):
                row["slope_rint"]=slopes[j]
                row["intercept_rint"]=pair["bm"][j]-slopes[j]*pair["am"][j]
                row["x_min_rint"]=pair["amin"][j] if pair["eligible"][j] else np.nan
                row["x_max_rint"]=pair["amax"][j] if pair["eligible"][j] else np.nan
    for row in rows:
        row["rho"]=row["rho_raw"]; row["p_spearman"]=row["p_spearman_raw"]
    return rows
