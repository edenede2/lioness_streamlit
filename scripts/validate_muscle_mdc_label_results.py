#!/usr/bin/env python3
"""Independent reconciliation of completed MDC p-values, BH and balanced ranges."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

APP=Path(__file__).resolve().parents[1]
PUBLIC=APP/'data/muscle_control_l4/mdc_label_validation'
PRIVATE=APP.parents[1]/'out/mdc_muscle_control_l4/20261013_label_balance_ct_ts'


def main():
    manifest=json.loads((PUBLIC/'manifest.json').read_text())
    assert manifest['status']=='complete'
    assert manifest['completed_label_permutations']==manifest['label_permutations']==200
    assert manifest['completed_balanced_resamples']==manifest['balanced_resamples']==100
    result=pd.read_parquet(PUBLIC/'results.parquet')
    null=pd.read_parquet(PUBLIC/'label_replicates.parquet')
    balanced=pd.read_parquet(PUBLIC/'balanced_replicates.parquet')
    keys=['normalization_variant','module','scope']
    ln={k:g.statistic.dropna().to_numpy() for k,g in null.groupby(keys)}
    bn={k:g.statistic.dropna().to_numpy() for k,g in balanced.groupby(keys)}
    for row in result.itertuples():
        key=(row.normalization_variant,row.module,row.scope)
        values=ln[key]
        assert row.null_finite_n==len(values)
        if row.test_eligible:
            assert len(values)>=180 and np.ptp(values)>0
            p=min(1.,2*min((1+np.sum(values<=row.statistic))/(1+len(values)),
                          (1+np.sum(values>=row.statistic))/(1+len(values))))
            np.testing.assert_allclose(p,row.label_p,rtol=0,atol=0)
        else:assert pd.isna(row.label_p) and pd.isna(row.label_fdr)
        b=bn[key]
        assert row.balanced_finite_n==len(b)
        if len(b):np.testing.assert_allclose([row.balanced_low,row.balanced_median,row.balanced_high],np.quantile(b,[.025,.5,.975]),rtol=1e-14)
    for _,family in result.groupby(['normalization_variant','scope']):
        valid=family[family.label_p.notna()]
        assert family.tested_module_count.eq(len(valid)).all()
        if len(valid):
            order=np.argsort(valid.label_p.to_numpy(),kind='stable')
            p=valid.label_p.to_numpy()[order]
            q=np.minimum(1.,np.minimum.accumulate((p*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
            np.testing.assert_allclose(q,valid.label_fdr.to_numpy()[order],rtol=1e-12,atol=1e-14)
    observed=result[result.scope!='CT_minus_TS']
    for _,part in observed.groupby(['normalization_variant','module']):
        g=part.set_index('scope')
        for column in ['mean_abs_ad','mean_abs_control']:
            for whole,pieces in [('total',['CT','TS']),('CT',['CT_VH__SMA','CT_VH__QUAD','CT_SMA__QUAD']),('TS',['TS_VH','TS_SMA','TS_QUAD'])]:
                assert g.loc[whole,'n_edges']==g.loc[pieces,'n_edges'].sum()
                target=float(g.loc[whole,column]*g.loc[whole,'n_edges'])
                source=(g.loc[pieces,column]*g.loc[pieces,'n_edges']).sum()
                np.testing.assert_allclose(target,source,rtol=2e-6,atol=1e-12)
    report={'status':'passed','label_permutations':200,'balanced_resamples':100,
            'scope_rows':len(result),'tested_scope_rows':int(result.label_p.notna().sum()),
            'fixed_control_module_caveat_retained':True}
    (PRIVATE/'independent_qa.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
