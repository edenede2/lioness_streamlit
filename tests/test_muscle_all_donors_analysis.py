from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from app_helpers import data as D
from scripts import run_muscle_all_donors_analysis as A


def test_source_is_biological_l4_not_table_4():
    config = A.configuration()
    assert A.ASSIGNMENTS.name.endswith("_table_3.csv")
    assert config["modules"] == 173 and config["assignments"] == 31948
    assert config["biological_level"] == 4 and config["zero_indexed_table"] == 3
    assert config["beta_ts"] == {"VH": 6., "SMA": 6., "QUAD": 10.}
    assert config["beta_ct"] == {"VH-SMA": 6., "VH-QUAD": 6., "SMA-QUAD": 7.}


def test_public_parquet_validation_covers_shards_not_dataset_directories(tmp_path):
    flat = tmp_path / "flat.parquet"
    flat.touch()
    dataset = tmp_path / "scores.parquet"
    dataset.mkdir()
    shard = dataset / "part-0.parquet"
    shard.touch()
    assert A.public_parquet_files(tmp_path) == [flat, shard]


def test_publication_keeps_large_statistics_on_drive_not_in_repository():
    from scripts.publish_muscle_all_donors import repository_hot_file
    assert repository_hot_file('mdc_sensitivity/summary.parquet', 200_000)
    assert repository_hot_file('aggregate_statistics.parquet', 1_000_000)
    assert not repository_hot_file('effect_size/resolved_statistics.parquet', 60*1024**2)
    assert not repository_hot_file('resolved_plot_data.parquet', 1_000_000)


def test_qa_resume_rejects_changed_numerical_inputs_or_incomplete_stages():
    from copy import deepcopy
    config = {"sources": {"assignments": "unchanged"}, "code_hashes": {
        "run_muscle_all_donors_analysis.py": "new-qa", "run_muscle_lioness.py": "same"}}
    old = deepcopy(config)
    old["code_hashes"]["run_muscle_all_donors_analysis.py"] = "old-qa"
    manifest = {"configuration": old, "active_stage": "validate", "status": "needs_review",
                "stages": {name: {"status": "complete"} for name in A.STAGES if name != "validate"}}
    assert A.public_validation_resume_allowed(manifest, config)
    changed = deepcopy(config)
    changed["sources"]["assignments"] = "different"
    assert not A.public_validation_resume_allowed(manifest, changed)
    changed = deepcopy(config)
    changed["code_hashes"]["run_muscle_lioness.py"] = "different"
    assert not A.public_validation_resume_allowed(manifest, changed)
    del manifest["stages"]["effect_public"]
    assert not A.public_validation_resume_allowed(manifest, config)


def test_new_module_definition_cannot_mix_brain_or_control_muscle_data():
    assert D.module_set_data_dir(A.KEY) != D.module_set_data_dir("muscle_control_l4")
    assert D.module_set_tissues(A.KEY) == ("VH", "SMA", "QUAD")
    assert D.module_set_components(A.KEY) == D.module_set_components("muscle_control_l4")
    assert A.KEY not in D.PREDICTION_MODULE_SETS
    assert "prediction" not in A.STAGES
    assert "bonobo" not in A.STAGES


def test_all_engines_bind_to_new_assignments_betas_and_separate_outputs():
    # Isolate configured backend globals from old-dataset unit tests.
    code = '''
import json
from scripts import run_muscle_all_donors_analysis as A
L,M,B,S,R,V,E,P=A.engines(A.configuration())
assert L.DEFAULT_ASSIGNMENTS==M.DEFAULT_ASSIGNMENTS==B.ASSIGNMENTS==E.DEFAULT_ASSIGNMENTS==P.ASSIGNMENTS==A.ASSIGNMENTS
assert L.BETAS_TS==M.BETAS_TS==B.BETAS_TS=={'VH':6.,'SMA':6.,'QUAD':10.}
assert M.M5_PARAMETERS['beta_ct']==L.BETAS_CT==B.BETAS_CT
assert E.edge_metadata.__globals__['BETAS_TS']==L.BETAS_TS
assert B.MODULE_COUNT==V.MODULE_COUNT==P.MODULE_COUNT==173
assert R.MODULE_SET==V.MODULE_SET==E.MODULE_DEFINITION==P.MODULE_SET==A.KEY
assert V.PUBLIC==A.PUBLIC/'mdc_label_validation'
assert R.PUBLIC_ROOT==A.PUBLIC
assert B.PRIVATE_MAP.name=='private_public_sample_mapping.tsv'
assert B.METHODS==P.METHODS==A.METHODS==('control_anchored',)
assert M.DEFAULT_OUTPUT==A.MDC and R.DEFAULT_OUTPUT==A.SENSITIVITY
print(json.dumps({'status':'passed'}))
'''
    output = subprocess.check_output([sys.executable, "-c", code], cwd=A.APP, text=True)
    assert json.loads(output)["status"] == "passed"


def test_sampled_formula_qa_has_both_small_large_modules_and_all_donor_roles():
    source = A.ROOT / "input_qa.json"
    if not source.exists():
        return
    result = json.loads(source.read_text())
    assert result["status"] == "passed" and result["donors"] == 357
    assert result["diagnosis_counts"] == {"Control":120, "MCI":82, "AD":155}
    checks = result["direct_formula_checks"]
    assert len({r["module"] for r in checks}) == 2
    assert {r["diagnosis"] for r in checks} == {"Control", "MCI", "AD"}
    assert max(r["max_edge_error"] for r in checks) < 1e-8


def test_incremental_drive_index_preserves_existing_datasets_and_rejects_scope_escape():
    import pytest
    from scripts.update_muscle_drive_index import merge_index, PREFIX
    existing={'root_folder_id':'root','files':{'muscle_control_l4/old.parquet':{'size':1,'id':'old'},f'{PREFIX}/stale.parquet':{'size':2,'id':'stale'}}}
    updated=merge_index(existing,{f'{PREFIX}/new.parquet':{'size':3,'id':'new'}},'root',2)
    assert updated['files']['muscle_control_l4/old.parquet']==existing['files']['muscle_control_l4/old.parquet']
    assert f'{PREFIX}/stale.parquet' not in updated['files']
    assert updated['file_count']==2 and updated['total_bytes']==4
    with pytest.raises(ValueError):merge_index(existing,{'other/new.parquet':{'size':3}},'root',2)
    with pytest.raises(ValueError):merge_index(existing,{},'different-root',2)
