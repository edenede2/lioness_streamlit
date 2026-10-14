#!/usr/bin/env python3
"""High-resolution paired AD/Control MDC heatmaps for both brain module sets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image

APP=Path(__file__).resolve().parents[1]
REPO=APP.parents[1]
DEFAULT_OUTPUT=REPO/'out/brain_mdc_split_triangle_heatmaps/20261014'
CONFIGS={
    'full_cohort':dict(label='Full-cohort L4 modules',count=154,data=APP/'data'),
    'control_derived':dict(label='Control-derived L4 modules',count=186,data=APP/'data/control_derived'),
}
REGIONS=('AC','DLPFC','PCG')
REGION_COLORS=('#4C78A8','#E69F00','#009E73')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def paired_matrix(ad, control):
    """Sort once, then assign AD to displayed upper-right, Control to lower-left."""
    ad,control=np.asarray(ad),np.asarray(control)
    if ad.ndim!=2 or ad.shape[0]!=ad.shape[1] or ad.shape!=control.shape:
        raise ValueError('Both networks must have identical square dimensions')
    if not np.isfinite(ad).all() or not np.isfinite(control).all():
        raise ValueError('Undefined adjacency entries must be resolved before rendering')
    if not np.allclose(ad,ad.T,atol=1e-10,rtol=0) or not np.allclose(control,control.T,atol=1e-10,rtol=0):
        raise ValueError('Adjacencies are not symmetric')
    order=np.argsort(-np.abs(ad).sum(axis=1),kind='stable')
    a=np.abs(ad[np.ix_(order,order)])
    c=np.abs(control[np.ix_(order,order)])
    upper=np.triu_indices(len(a),1);lower=np.tril_indices(len(a),-1)
    combined=np.full(a.shape,np.nan)
    combined[upper]=a[upper];combined[lower]=c[lower]
    return combined,order


def ratio(ad,control,tissues):
    n=len(ad);i,j=np.triu_indices(n,1);same=tissues[i]==tissues[j]
    result={}
    for key,mask in [('total',np.ones(len(i),bool)),('ts',same),('ct',~same)]:
        den=np.abs(control[i[mask],j[mask]]).mean() if mask.any() else np.nan
        result[key]=np.abs(ad[i[mask],j[mask]]).mean()/den if den>=1e-10 else np.nan
    return result


def value_label(value):
    return f'{value:.3g}' if np.isfinite(value) else 'NA'


def figure(combined,tissues,order,label,module,mdc,color_cap='q99'):
    n=len(combined)
    finite=combined[np.isfinite(combined)]
    cap=float(np.quantile(finite,.99)) if color_cap=='q99' else 1.
    if cap<=0:cap=1.
    fig=plt.figure(figsize=(32/3,32/3),dpi=300,facecolor='white')
    ax=fig.add_axes([.13,.12,.68,.67])
    cmap=plt.colormaps['hot'].copy();cmap.set_bad('white')
    im=ax.imshow(combined,cmap=cmap,vmin=0,vmax=cap,origin='upper',
                 interpolation='nearest',rasterized=True,aspect='auto')
    ax.plot([-.5,n-.5],[-.5,n-.5],color='white',lw=.65)
    ax.set_xticks([]);ax.set_yticks([])
    ax.set_xlabel('Module genes (same ordering in both triangles)',fontsize=15,labelpad=10)
    ax.set_ylabel('Module genes',fontsize=15,labelpad=10)
    for spine in ax.spines.values():spine.set_color('#333333');spine.set_linewidth(.7)
    codes=np.asarray(tissues)[order]
    color=ListedColormap(REGION_COLORS)
    top=fig.add_axes([.13,.795,.68,.011]);left=fig.add_axes([.113,.12,.011,.67])
    top.imshow(codes[None,:],cmap=color,vmin=-.5,vmax=2.5,aspect='auto',interpolation='nearest')
    left.imshow(codes[:,None],cmap=color,vmin=-.5,vmax=2.5,aspect='auto',interpolation='nearest',origin='upper')
    top.axis('off');left.axis('off')
    handles=[Patch(facecolor=REGION_COLORS[k],label=f'{region} ({np.sum(codes==k):,} genes)')
             for k,region in enumerate(REGIONS) if np.any(codes==k)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.47,.86),
               ncol=len(handles),frameon=False,fontsize=12)
    fig.text(.47,.98,f'{label} · Module M{module}',ha='center',va='top',fontsize=19,fontweight='bold')
    fig.text(.47,.942,'MDC (AD/Control): '+ ' | '.join(f'{s.upper() if s!="total" else "All"}={value_label(mdc[s])}' for s in ['total','ts','ct']),
             ha='center',va='top',fontsize=14)
    fig.text(.47,.906,f'Upper triangle: AD | Lower triangle: Control | {n:,} tissue–gene nodes',
             ha='center',va='top',fontsize=14)
    cax=fig.add_axes([.86,.12,.021,.67])
    bar=fig.colorbar(im,cax=cax,extend='max' if color_cap=='q99' else 'neither')
    bar.ax.tick_params(labelsize=12)
    bar.set_label('Absolute signedAlt edge weight',fontsize=14,labelpad=12)
    note='Shared AD/Control color scale; pooled 99th-percentile display cap' if color_cap=='q99' else 'Shared 0–1 scale across both diagnoses and all modules'
    fig.text(.47,.035,note+'\nBrain adjacency: βTS=3, βCT=2; diagonal excluded',ha='center',va='bottom',fontsize=11,color='#444444')
    return fig,cap


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--reuse-adjacencies',action='store_true')
    p.add_argument('--module-set',choices=['all',*CONFIGS],default='all')
    p.add_argument('--module',type=int,action='append')
    p.add_argument('--color-cap',choices=['q99','full'],default='q99')
    args=p.parse_args();output=args.output.resolve();cache=output/'_private_adjacencies'
    output.mkdir(parents=True,exist_ok=True)
    if not args.reuse_adjacencies:
        subprocess.run(['Rscript',str(APP/'scripts/export_brain_mdc_adjacencies.R'),str(REPO),str(cache)],check=True,cwd=REPO)
    keys=list(CONFIGS) if args.module_set=='all' else [args.module_set]
    overall={}
    for key in keys:
        config=CONFIGS[key]
        sources=pd.read_csv(cache/key/'manifest.tsv',sep='\t')
        stats=pd.read_csv(config['data']/'mdc_ad_vs_control_summary.tsv',sep='\t').set_index('module')
        if len(sources)!=config['count'] or set(sources.module)!=set(stats.index):
            raise ValueError('Source module coverage differs from the selected MDC catalog')
        rows=sources if not args.module else sources[sources.module.isin(args.module)]
        if args.module and set(args.module)-set(rows.module):raise ValueError('Unknown module for selected definition')
        destination=output/key;destination.mkdir(parents=True,exist_ok=True)
        pdf=destination/'AD_Control_MDC_split_triangles.pdf'
        records=[]
        with PdfPages(pdf) as book:
            for position,row in enumerate(rows.itertuples(index=False),1):
                module=int(row.module);n=int(row.n_nodes)
                ad=np.memmap(cache/key/f'M{module}__AD.f64',dtype='<f8',mode='r',shape=(n,n),order='F')
                control=np.memmap(cache/key/f'M{module}__Control.f64',dtype='<f8',mode='r',shape=(n,n),order='F')
                tissues=np.fromfile(cache/key/f'M{module}__tissues.i32',dtype='<i4')
                if len(tissues)!=n or not set(tissues).issubset({0,1,2}):raise ValueError('Tissue code mapping failed')
                calculated=ratio(ad,control,tissues)
                stored={s:float(stats.loc[module,f'mdc_{s}']) for s in calculated}
                np.testing.assert_allclose(list(calculated.values()),list(stored.values()),rtol=1e-5,atol=1e-5,equal_nan=True)
                combined,order=paired_matrix(ad,control)
                fig,cap=figure(combined,tissues,order,config['label'],module,stored,args.color_cap)
                png=destination/f'M{module}.png'
                fig.savefig(png,dpi=300,facecolor='white',metadata={'Description':f'{key}: upper=AD, lower=Control; identical paired ordering; absolute signedAlt weights'})
                book.savefig(fig,dpi=200)
                plt.close(fig)
                with Image.open(png) as im:
                    if im.size!=(3200,3200):raise ValueError(f'PNG resolution mismatch: {im.size}')
                records.append(dict(module_set=key,module=module,n_nodes=n,png=png.name,
                    upper_triangle='AD',lower_triangle='Control',ordering='AD absolute weighted degree; stable ties',
                    mdc_total=stored['total'],mdc_ts=stored['ts'],mdc_ct=stored['ct'],
                    directional_fdr_total=stats.loc[module,'directional_fdr_total'],
                    directional_fdr_ts=stats.loc[module,'directional_fdr_ts'],directional_fdr_ct=stats.loc[module,'directional_fdr_ct'],
                    color_cap=args.color_cap,display_vmax=cap,width=3200,height=3200,dpi=300,
                    png_sha256=sha(png),mdc_reproduction_error=row.max_mdc_reproduction_error))
                if position%20==0 or position==len(rows):print(f'{key}: {position}/{len(rows)} PNG/PDF pages',flush=True)
                del ad,control,combined,fig
        manifest=pd.DataFrame(records)
        manifest.to_csv(destination/'manifest.tsv',sep='\t',index=False)
        overall[key]=dict(modules=len(records),expected_modules=config['count'],pdf=pdf.name,
                          pdf_sha256=sha(pdf),manifest_sha256=sha(destination/'manifest.tsv'),
                          source_statistics_sha256=sha(config['data']/'mdc_ad_vs_control_summary.tsv'))
    result=dict(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),datasets=overall,
                upper_triangle='AD',lower_triangle='Control',edge_display='Absolute signedAlt adjacency',
                beta_ts=3,beta_ct=2,old_results_overwritten=False,
                display_cap_note='Display saturation only; adjacency values, MDC and existing FDRs are unchanged.',
                privacy='No donor/projid identifiers or Ensembl labels are included in figures or public manifests.')
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
