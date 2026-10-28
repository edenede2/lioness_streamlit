"""The matched partition must not expose incomplete or legacy data."""
import json
from app_helpers import data as D
from app_helpers.camel import score_framework

KEY="matched_control_brain_l4"


def test_separate_control_reference_namespace():
    assert D.MODULE_SET_METHODS[KEY]==("control_anchored",)
    assert D.MODULE_SET_DIRS[KEY]!=D.MODULE_SET_DIRS["control_derived"]
    assert D.module_set_tissues(KEY)==("AC","DLPFC","PCG")
    assert len(D.module_set_components(KEY))==6
    assert KEY not in D.PREDICTION_MODULE_SETS
    assert score_framework(KEY)=="CAMEL"


def test_incomplete_registry_is_not_selectable(monkeypatch,tmp_path):
    manifest=tmp_path/"manifest.json"
    monkeypatch.setattr(D,"ensure_data_path",lambda path:manifest)
    monkeypatch.setattr(D,"data_path_available",lambda path:True)
    manifest.write_text(json.dumps({"module_sets":{KEY:{"status":"running"}}}))
    assert KEY not in D.available_module_sets()
    manifest.write_text(json.dumps({"module_sets":{KEY:{"status":"complete"}}}))
    assert KEY in D.available_module_sets()
    monkeypatch.setattr(D,"data_path_available",lambda path:False)
    assert KEY not in D.available_module_sets()
