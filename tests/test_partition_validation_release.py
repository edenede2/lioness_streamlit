"""Partition-scoped publication guards and aggregate-only review paths."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(APP/"scripts"))
from run_camel_validation import atomic_json
from publish_partition_validation import approval_gate
from app_helpers.validation_datasets import LEGACY_DUPLICATE_BRAIN_MODULES


def test_publication_approval_cannot_transfer_to_other_method_or_grain(tmp_path):
    approval=dict(approved=True,module_set="full_cohort",methods=["control_anchored"],score_rows=7795942,
        score_files=154,transformation_files=154,transformation_rows=343728,validation_samples=793)
    report=dict(status="passed",modules=154,score_rows=7795942,transform_rows=343728,samples=793,completed_rules=["all"])
    manifest=dict(module_set="full_cohort",network_method="control_anchored",completed_rules=["all"])
    atomic_json(approval,tmp_path/"publication_approval.json")
    atomic_json(report,tmp_path/"public_bundle/release_manifest.json")
    atomic_json(manifest,tmp_path/"public_bundle/manifest.json")
    assert approval_gate(tmp_path,"full_cohort","control_anchored").modules==154
    with pytest.raises(ValueError,match="Explicit approval"):
        approval_gate(tmp_path,"full_cohort","standard")
    atomic_json({**report,"samples":794},tmp_path/"public_bundle/release_manifest.json")
    with pytest.raises(ValueError,match="sample grain"):
        approval_gate(tmp_path,"full_cohort","control_anchored")
    atomic_json({**report,"status":"scores_passed"},tmp_path/"public_bundle/release_manifest.json")
    with pytest.raises(ValueError,match="Release audit"):
        approval_gate(tmp_path,"full_cohort","control_anchored")


def test_publication_waiter_requires_completed_release_qa_and_skips_intermediate_masks(tmp_path):
    from publish_ready_brain_validation import candidate
    root=tmp_path/"full_cohort/control_anchored"
    first=["all",*["ad_control_discovery_effect__hedges_g__fixed_0.40__"+d for d in
        ("ad_higher","control_higher","either")]]
    rules=[*first,"later_mask_a","later_mask_b"]
    approval=dict(approved=True,module_set="full_cohort",methods=["control_anchored"],score_rows=7795942,
        score_files=154,transformation_files=154,transformation_rows=343728,validation_samples=793)
    report=dict(status="scores_passed",modules=154,score_rows=7795942,transform_rows=343728,samples=793,
        completed_rules=first,mdc_status="unavailable")
    manifest=dict(module_set="full_cohort",network_method="control_anchored",completed_rules=first,rules=rules)
    atomic_json(approval,root/"publication_approval.json")
    atomic_json(report,root/"public_bundle/release_manifest.json")
    atomic_json(manifest,root/"public_bundle/manifest.json")
    assert candidate(tmp_path,"control_anchored",{}) is None
    report.update(status="passed",mdc_status="complete")
    atomic_json(report,root/"public_bundle/release_manifest.json")
    ready=candidate(tmp_path,"control_anchored",{})
    assert ready["stage"]=="first_release"
    assert candidate(tmp_path,"control_anchored",{"control_anchored":ready}) is None
    atomic_json({**manifest,"completed_rules":[*first,"later_mask_a"]},root/"public_bundle/manifest.json")
    assert candidate(tmp_path,"control_anchored",{}) is None  # audit now stale
    atomic_json({**report,"completed_rules":[*first,"later_mask_a"]},root/"public_bundle/release_manifest.json")
    assert candidate(tmp_path,"control_anchored",{}) is None  # deliberate no per-mask uploads
    atomic_json({**manifest,"completed_rules":rules},root/"public_bundle/manifest.json")
    atomic_json({**report,"completed_rules":rules},root/"public_bundle/release_manifest.json")
    assert candidate(tmp_path,"control_anchored",{})["stage"]=="final"


def test_brain_queue_finishes_reviewed_brain_before_any_motor_or_publication():
    from run_partition_validation_pipeline import stages
    jobs=stages()
    assert {j[0] for j in jobs}=={"control_anchored","standard"}
    assert all(not any(s in j[2] for s in ("publish","muscle","motor")) for j in jobs)
    for method in ("control_anchored","standard"):
        names=[j[1] for j in jobs if j[0]==method]
        assert names.index("mdc_pilot")<names.index("mdc_run")<names.index("first_release_qa")
        assert names.index("first_release_qa")<names.index("first_aggregate_install")
        assert names.index("remaining_associations")<names.index("final_qa")<names.index("final_aggregate_install")


def test_mdc_preserves_legacy_unique_mask_and_resume(tmp_path,monkeypatch):
    import run_partition_validation_mdc as runner
    import run_camel_validation_mdc as kernel
    from app_helpers.camel_mdc import standardized
    source=tmp_path/"edge-source.tsv";source.write_text("fixed test edge source\n")
    atomic_json(dict(status="passed"),tmp_path/"audit.json")
    genes=["A","B","C","B"]
    assignment=pd.DataFrame({"module":188,"Tissue":"MF","public_tissue":"DLPFC",
        "Gene Symbol":genes,"feature_id":["DLPFC_"+g for g in genes]})
    ri,ci=np.tril_indices(4,-1);unique=(ri!=3)&(ci!=3)
    masks={"all":unique,"effect":np.ones(len(ri),bool)}
    state=dict(assignment=assignment,method="control_anchored",
        config=SimpleNamespace(module_set="full_cohort",components={"TS_DLPFC":("DLPFC",)}))
    monkeypatch.setattr(runner.P,"_STATE",state)
    monkeypatch.setattr(kernel,"module_source",lambda module:(assignment,None,ri,ci,masks,source))
    monkeypatch.setattr(kernel,"COMPONENT_TISSUES",{"TS_DLPFC":("DLPFC",)})
    monkeypatch.setattr(kernel,"COMPONENT_LABELS",{"TS_DLPFC":"DLPFC"})
    rng=np.random.default_rng(42)
    az,cz=standardized(rng.normal(size=(12,3))),standardized(rng.normal(size=(11,3)))
    background=(az,cz,{"DLPFC_"+g:i for i,g in enumerate(["A","B","C"])},{"DLPFC":np.arange(3)})
    monkeypatch.setattr(kernel,"component_background",lambda component:background)
    # No gene null can draw four distinct genes from a three-gene synthetic
    # background. The real expression background contains thousands of genes.
    background=(standardized(rng.normal(size=(12,8))),standardized(rng.normal(size=(11,8))),
        background[2],{"DLPFC":np.arange(8)})
    path=runner.run_module(188,"TS_DLPFC",tmp_path)
    first=pd.read_parquet(path)
    assert first.n_possible_edges.tolist()==[3,6] and first.n_features.tolist()==[3,4]
    assert first.cross_filter_node_convention_warning.all()
    assert first.sample_null_valid_draws.eq(200).all() and first.gene_null_valid_draws.eq(200).all()
    assert first.sample_p_resolution.eq(1/201).all()
    runner.run_module(188,"TS_DLPFC",tmp_path)
    pd.testing.assert_frame_equal(first,pd.read_parquet(path))
    with np.load(tmp_path/"public_bundle/mdc/adjacency/TS_DLPFC/M188.npz") as arrays:
        assert arrays["canonical_nodes"].tolist()==[True,True,True,False]
    assert json.loads((tmp_path/"mdc_parts/TS_DLPFC/M188.json").read_text())["direct_adjacency_error"]<1e-10


def test_full_cohort_pooled_review_uses_aggregate_shards_only(monkeypatch):
    from app_helpers import camel_validation as V,validation_review as R
    root=APP.parents[1]/"out/partition_validation/20261021_frozen_references_legacy_conventions/full_cohort/control_anchored/public_bundle"
    if not (root/"review/manifest.json").is_file():
        pytest.skip("Private review not staged; never download tests")
    monkeypatch.setenv("PARTITION_VALIDATION_DATA_ROOT",str(root))
    monkeypatch.delenv("CAMEL_VALIDATION_DATA_ROOT",raising=False)
    def forbidden(*args,**kwargs):
        raise AssertionError("Review must not read donor scores or metadata")
    for name in ("scores","summaries","metadata"):
        monkeypatch.setattr(V,name,forbidden)
    reader=R.for_dataset("full_cohort","control_anchored")
    frame=reader.association_table("all","standard_pruned","connectivity","cogn_global")
    pooled=frame.loc[frame.diagnosis_group.eq("All donors")]
    assert len(pooled)==154*6 and pooled.original_q_spearman.notna().any()
    assert not {"sample_id","donor","projid"}&set(frame)
    assert set(frame.loc[frame.cross_filter_node_convention_warning,"module"])==set(LEGACY_DUPLICATE_BRAIN_MODULES)
    app=AppTest.from_string('''
from app_helpers.validation_review_ui import render_view
from app_helpers.figure_export import render_exportable_plot
render_view(render_exportable_plot)
''',default_timeout=120).run()
    selector=next((e for e in app.selectbox if e.label=="Validation dataset"),None)
    if selector is not None:
        selector.set_value(("full_cohort","control_anchored"));app.run()
    assert not app.exception
    next(e for e in app.radio if e.label=="Validation section").set_value("Association replication");app.run()
    assert not app.exception and app.get("plotly_chart")
    assert next(e for e in app.selectbox if e.label=="Validation donor group").value=="All donors"
    displayed=next(e.value for e in app.dataframe if "original_fdr" in e.value)
    assert displayed.diagnosis_group.eq("All donors").all() and displayed.original_fdr.notna().any()
    assert displayed.original_family_n.le(154).all() and displayed.validation_family_n.le(154).all()


@pytest.mark.parametrize("method",["control_anchored","standard"])
def test_full_cohort_donor_views_render_from_private_staging_without_drive(method,monkeypatch):
    from app_helpers import camel_validation as V
    run=APP.parents[1]/"out/partition_validation/20261021_frozen_references_legacy_conventions/full_cohort"
    bundle=run/method/"public_bundle"
    if not (bundle/"manifest.json").is_file() or "all" not in json.loads((bundle/"manifest.json").read_text())["completed_rules"]:
        pytest.skip("Method's first association context is not staged")
    monkeypatch.setenv("PARTITION_VALIDATION_DATA_ROOT",str(bundle))
    # Keep other definitions at their real namespaced paths, not this override.
    from app_helpers.drive_data import DATA_DIR
    from app_helpers.validation_datasets import dataset
    target_method=method
    monkeypatch.setattr(V,"root",lambda module_set="control_derived",method="control_anchored":
        bundle if (module_set,method)==("full_cohort",target_method) else DATA_DIR/dataset(module_set,method).relative_root(method))
    app=AppTest.from_file(APP/"streamlit_app.py",default_timeout=180).run()
    def widget(elements,label):return next(e for e in elements if e.label==label)
    widget(app.selectbox,"Module definition").set_value("full_cohort");app.run()
    widget(app.radio,"Network method").set_value(method);app.run()
    assert not app.exception
    widget(app.radio,"Donor cohort scope").set_value("incomplete_region_validation");app.run()
    assert not app.exception and app.get("plotly_chart")
    for view in ("Feature distributions","Correlation heatmaps","Statistics","Edge summaries","MDC"):
        widget(app.pills,"Analysis view").set_value(view);app.run()
        assert not app.exception,view
