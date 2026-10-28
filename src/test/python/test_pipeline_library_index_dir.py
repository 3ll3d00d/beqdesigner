'''
`run.index_dir`: the discovery index kept apart from the work directory, in a folder where file locks work. Every route
to the index -- the profile, ScanSettings, the CLI, the service -- must agree on where it is.
'''
import json
import os

import yaml

from pipeline.library import cli
from pipeline.library.index import INDEX_FILE_NAME, index_path
from pipeline.library.profile import profile_from_config
from pipeline.library.status import ScanSettings
from pipeline.service.context import load_context


def _profile_file(tmp_path, **run):
    media = tmp_path / 'films'
    media.mkdir(exist_ok=True)
    (media / 'a.mkv').write_bytes(b'x')
    config = tmp_path / 'p.yaml'
    config.write_text(yaml.safe_dump({
        'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': [str(media)]}],
        'run': {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(tmp_path / 'q'), **run}}))
    return config


def test_the_index_is_in_the_work_directory_unless_an_index_folder_is_set():
    assert index_path('/work') == os.path.join('/work', INDEX_FILE_NAME)
    assert index_path('/work', '') == os.path.join('/work', INDEX_FILE_NAME)
    assert index_path('/work', '/locks') == os.path.join('/locks', INDEX_FILE_NAME)


def test_the_profile_and_scan_settings_carry_the_index_folder_and_write_it_under_run():
    from dataclasses import replace
    profile = profile_from_config({'sync': {'index_dir': '/from-sync'}, 'run': {'work_dir': '/w'}})
    assert profile.index_dir == '/from-sync' and ScanSettings.from_profile(profile).index_dir == '/from-sync'
    assert profile_from_config({'sync': {'index_dir': '/s'}, 'run': {'index_dir': '/r'}}).index_dir == '/r'

    unset = profile_from_config({'run': {'work_dir': '/w'}})
    assert unset.index_dir == '' and 'index_dir' not in unset.to_config()['run']
    assert replace(unset, index_dir='/locks').to_config()['run']['index_dir'] == '/locks'
    assert 'index_dir' not in replace(profile, index_dir='').to_config().get('sync', {})


def test_scan_status_and_accept_use_the_index_folder_and_leave_none_in_the_work_directory(tmp_path, capsys):
    locks = tmp_path / 'locks'
    config = _profile_file(tmp_path, index_dir=str(locks))

    assert cli.main(['scan', '--profile', str(config)]) == 0
    assert json.loads(capsys.readouterr().out)['titles'] == 1
    assert os.path.isfile(locks / INDEX_FILE_NAME)
    assert not os.path.exists(tmp_path / 'work' / INDEX_FILE_NAME)

    assert cli.main(['status', '--profile', str(config), '--json']) == 0
    assert json.loads(capsys.readouterr().out)['titles'] == 1
    assert cli.main(['accept', '--profile', str(config), '--dry-run']) == 0


def test_the_index_dir_flag_overrides_the_file(tmp_path, capsys):
    config = _profile_file(tmp_path)
    flagged = tmp_path / 'flagged'

    assert cli.main(['scan', '--profile', str(config), '--index-dir', str(flagged)]) == 0
    assert os.path.isfile(flagged / INDEX_FILE_NAME) and not os.path.exists(tmp_path / 'work' / INDEX_FILE_NAME)
    capsys.readouterr()
    assert cli.main(['status', '--profile', str(config), '--index-dir', str(flagged), '--json']) == 0
    assert json.loads(capsys.readouterr().out)['titles'] == 1


def test_the_service_finds_the_index_in_the_index_folder(tmp_path):
    config = _profile_file(tmp_path, index_dir=str(tmp_path / 'locks'))
    assert load_context(str(config), {}).index_file == os.path.join(str(tmp_path / 'locks'), INDEX_FILE_NAME)
    plain = _profile_file(tmp_path)
    assert load_context(str(plain), {}).index_file == os.path.join(str(tmp_path / 'work'), INDEX_FILE_NAME)


def test_a_service_scan_writes_the_index_in_the_index_folder(tmp_path):
    from pipeline.service import work
    from pipeline.service.executions import ScanRequest
    config = _profile_file(tmp_path, index_dir=str(tmp_path / 'locks'))
    (tmp_path / 'work').mkdir()

    result = work.scan(load_context(str(config), {}), ScanRequest())

    assert result.titles == 1
    assert os.path.isfile(tmp_path / 'locks' / INDEX_FILE_NAME) and not os.path.exists(tmp_path / 'work' / INDEX_FILE_NAME)
