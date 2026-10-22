import json
from pathlib import Path
import sys
import pytest
from streamlit.testing.v1 import AppTest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_helpers.outcome_screening_ui import ROOT


def element(elements, label):
    return next(e for e in elements if e.label == label)


def test_empty_catalog_explicit_unavailable(tmp_path):
    app = AppTest.from_string(f"from app_helpers.outcome_screening_ui import render_view\nrender_view({str(tmp_path)!r})").run()
    assert not app.exception
    assert app.info and not app.get('plotly_chart')


def test_gene_ranking_validation_and_descriptive_groups_render():
    if not (ROOT/'manifest.json').is_file():
        pytest.skip('A staged aggregate catalog is required')
    app = AppTest.from_string('from app_helpers.outcome_screening_ui import render_view\nrender_view()', default_timeout=30).run()
    assert not app.exception
    assert len(app.get('plotly_chart')) == 2
    element(app.selectbox, 'Displayed statistics').select('validation').run()
    assert not app.exception
    element(app.selectbox, 'Donor group').select('ad').run()
    assert not app.exception
    assert len(app.get('plotly_chart')) == 1
    assert any('descriptive' in e.value for e in app.warning)
    element(app.radio, 'Screening view').set_value('whole_network').run()
    assert not app.exception
    assert any('unfinished' in e.value for e in app.info)


def test_staging_denies_donor_rows_and_preserves_atomic_manifest(tmp_path):
    import pandas as pd
    from scripts.run_gene_edge_screen import atomic_json
    from scripts.screening_sources import sha
    from scripts.stage_outcome_screening import stage
    source = tmp_path/'source/public_bundle'; source.mkdir(parents=True)
    data = source/'bad.parquet'
    pd.DataFrame({'sample_id':['S-fixture'], 'weight':[1.]}).to_parquet(data)
    atomic_json(dict(schema_version=1, files=[dict(path='bad.parquet', status='complete', rows=1, sha256=sha(data))]), source/'manifest.json')
    atomic_json(dict(status='passed'), tmp_path/'source/qa/independent_direct_checks.json')
    with pytest.raises(ValueError, match='Aggregate-only'):
        stage(tmp_path/'source', tmp_path/'deploy')
    assert not (tmp_path/'deploy/manifest.json').exists()
