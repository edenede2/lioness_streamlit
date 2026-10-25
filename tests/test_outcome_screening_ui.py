import json
from pathlib import Path
import sys
import pytest
from streamlit.testing.v1 import AppTest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_helpers.outcome_screening_ui import ROOT, completed_catalogs, pending_catalog_message


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
    assert any('not been run yet' in e.value and 'not a plotting error' in e.value for e in app.info)
    assert 'Whole-network edges — not yet available' in element(app.radio, 'Screening view').options
    assert not app.get('plotly_chart')
    element(app.radio, 'Screening view').set_value('genes').run()
    assert not app.exception and len(app.get('plotly_chart')) >= 1


def test_pending_motor_does_not_disable_completed_brain_catalogs():
    manifest = {'motor_module_edges_status': 'unavailable_pending_seed_stability_review', 'files': [
        {'path': 'module_edges/control_derived/ready.parquet', 'status': 'complete'},
        {'path': 'module_edges/motor/pending.parquet', 'status': 'running'},
    ]}
    assert completed_catalogs(manifest, 'module_edges') == [manifest['files'][0]]
    assert completed_catalogs(manifest, 'whole_network') == []
    assert 'have not been run yet' in pending_catalog_message('whole_network')
    assert 'motor search does not invalidate' in pending_catalog_message('module_edges')


def test_incomplete_deployment_reports_missing_file_not_unfinished_analysis(tmp_path):
    manifest = {'schema_version': 1, 'files': [
        {'path': 'module_edges/control_derived/ready.parquet', 'status': 'complete'},
    ]}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    app = AppTest.from_string(f"from app_helpers.outcome_screening_ui import render_view\nrender_view({str(tmp_path)!r})").run()
    element(app.radio, 'Screening view').set_value('module_edges').run()
    assert not app.exception and not app.get('plotly_chart')
    assert any('data file is missing' in item.value for item in app.error)
    assert 'Module edges' in element(app.radio, 'Screening view').options


def test_gene_only_release_has_view_specific_pending_message(tmp_path):
    manifest = {'schema_version': 1, 'files': [
        {'path': 'genes/brain/AC.parquet', 'status': 'complete'},
    ]}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    app = AppTest.from_string(f"from app_helpers.outcome_screening_ui import render_view\nrender_view({str(tmp_path)!r})").run()
    element(app.radio, 'Screening view').set_value('module_edges').run()
    assert not app.exception
    assert any('within-module edge catalog' in item.value for item in app.info)
    assert 'Module edges — not yet available' in element(app.radio, 'Screening view').options


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


def test_completed_brain_catalogs_render_both_methods_and_fdr_scopes():
    if not (ROOT/'manifest.json').is_file():
        pytest.skip('A staged aggregate catalog is required')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    entries=[e for e in manifest['files'] if e['path'].startswith('module_edges/')]
    if len(entries)!=3:
        pytest.skip('All three brain catalogs must be complete')
    app=AppTest.from_string('from app_helpers.outcome_screening_ui import render_view\nrender_view()',default_timeout=30).run()
    element(app.radio,'Screening view').set_value('module_edges').run()
    for entry in entries:
        element(app.selectbox,'Completed catalog').set_value(entry).run()
        assert not app.exception
        for cohort in ('original','validation'):
            element(app.selectbox,'Displayed statistics').select(cohort).run()
            for scope in ('within_module','partition'):
                element(app.radio,'FDR scope').set_value(scope).run()
                assert not app.exception
                assert len(app.get('plotly_chart'))==2
        # A module query loads only that module/outcome, never the full catalog.
        from app_helpers.outcome_screening_ui import load_catalog
        module=int(entry['modules'][0])
        narrow=load_catalog(str(ROOT/entry['path']),entry['sha256'],entry['rows'],'cogn_global',(module,))
        assert set(narrow.module)=={module} and set(narrow.outcome)=={'cogn_global'}
        assert len(narrow)<=20


def test_pilot_review_contains_no_donor_rows_and_does_not_authorize_execution():
    if not (ROOT/'manifest.json').is_file():
        pytest.skip('A staged catalog is required')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    if 'pilot_review_file' not in manifest:
        pytest.skip('Pilot review is not staged yet')
    review=json.loads((ROOT/manifest['pilot_review_file']['path']).read_text())
    assert review['pilot_configs']==8 and review['pilot_measurements']==48
    assert review['formulas_passed'] and review['validation_controls_add_one_passed']
    assert review['genome_launch_authorized'] is False and review['donor_values_public'] is False
    coefficients=[r['pearson'] for r in review['seed_comparisons'] if r['pearson'] is not None]
    assert review['seed_edge_pearson_min']==min(coefficients)
    assert all('sample_id' not in r and 'donor' not in r for r in review['measured_configs'])


def test_main_app_module_edges_and_pending_whole_network_navigation():
    if not (ROOT/'manifest.json').is_file() or not completed_catalogs(json.loads((ROOT/'manifest.json').read_text()), 'module_edges'):
        pytest.skip('A completed brain module catalog is required')
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'streamlit_app.py'), default_timeout=45)
    app.query_params['view'] = 'Gene & edge associations'
    app.run()
    element(app.radio, 'Screening view').set_value('module_edges').run()
    assert not app.exception and len(app.get('plotly_chart')) == 2
    assert len(element(app.selectbox, 'Completed catalog').options) == 3
    element(app.radio, 'Screening view').set_value('whole_network').run()
    assert not app.exception and not app.get('plotly_chart')
    assert any('not a plotting error' in item.value for item in app.info)
    element(app.radio, 'Screening view').set_value('module_edges').run()
    assert not app.exception and len(app.get('plotly_chart')) == 2
