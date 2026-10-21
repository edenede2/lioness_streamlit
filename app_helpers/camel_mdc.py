"""Pure group-level signedAlt MDC kernels (not donor CAMEL scores)."""
from __future__ import annotations
import numpy as np


def standardized(values):
    x=np.asarray(values,dtype=float)
    if not np.isfinite(x).all(): raise ValueError("MDC requires finite observed-region expression")
    z=x-x.mean(axis=0)
    scale=np.sqrt(np.sum(z*z,axis=0))
    return np.divide(z,scale,out=np.zeros_like(z),where=scale>0)


def adjacency_edges(z, rows, columns, power):
    corr=np.clip((z.T@z)[rows,columns],-1,1)
    return np.sign(corr)*np.abs(corr)**power


def mdc_from_edges(ad, control, masks):
    n=masks.sum(axis=1)
    numerator=np.divide(masks@np.abs(ad),n,out=np.full(len(n),np.nan),where=n>0)
    denominator=np.divide(masks@np.abs(control),n,out=np.full(len(n),np.nan),where=n>0)
    ratio=np.divide(numerator,denominator,out=np.full(len(n),np.nan),where=denominator>1e-10)
    return ratio,numerator,denominator


def empirical(observed, draws, upper):
    valid=np.isfinite(draws)
    n=valid.sum(axis=1)
    exceed=((draws>=observed[:,None]) if upper else (draws<=observed[:,None]))&valid
    return np.divide(1+exceed.sum(axis=1),1+n,out=np.full(len(n),np.nan,dtype=float),where=(n>0)&np.isfinite(observed))
