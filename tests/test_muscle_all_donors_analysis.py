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
