'''
model/worklist_edit.py: what the settings drawer decides without a widget -- editing a profile's config, checking folders
and repositories, the ignore preview over index rows, and what makes the index out of date. `import ui.beq` first (the
shared fixture needs the app's modules): see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import json
import os
import subprocess
from dataclasses import replace

import pytest

from model.worklist_edit import LEVEL_ERROR, LEVEL_INFO, LEVEL_OK, check_directory, check_relative_dir, check_repository, \
    config_value, discovery_changed, folder_of, move_item, parse_external_ids, preview_rules, remote_owner_and_name, \
    repository_location, unique_name, with_config
from model.worklist_profile import WorkListSetup
from pipeline.library.ignore import rule_from_config
from pipeline.library.index import LibraryIndex, index_path
from pipeline.library.profile import Profile, SourceSpec, profile_from_config
from pipeline.library.status import ScanSettings
from gui.worklist_fixture import make_index, title_row


def test_with_config_sets_and_removes_a_key_and_leaves_the_rest_alone():
    profile = profile_from_config({'run': {'work_dir': '/w', 'audio_types': ['a']}, 'custom': {'x': 1}})

    edited = with_config(profile, 'run', 'designer', 'rolloff')
    assert edited.config['run'] == {'work_dir': '/w', 'audio_types': ['a'], 'designer': 'rolloff'}
    assert edited.config['custom'] == {'x': 1} and profile.config['run'].get('designer') is None   # the original is not changed

    assert with_config(edited, 'run', 'designer', '').config['run'] == {'work_dir': '/w', 'audio_types': ['a']}
    assert 'sync' not in with_config(profile, 'sync', 'image_owner', None).config      # nothing to remove: no empty section
    assert config_value(edited, 'run', 'designer') == 'rolloff' and config_value(edited, 'run', 'nope', 7) == 7
    assert 'sync' not in with_config(with_config(profile, 'sync', 'image_owner', 'me'), 'sync', 'image_owner', '').config


def test_move_item_and_unique_name():
    assert move_item(['a', 'b', 'c'], 2, 0) == ['c', 'a', 'b']
    assert move_item(['a', 'b', 'c'], 0, 2) == ['b', 'c', 'a']
    assert unique_name('jriver', []) == 'jriver' and unique_name('jriver', ['jriver', 'jriver-2']) == 'jriver-3'


def test_a_directory_is_ok_creatable_or_refused(tmp_path):
    assert check_directory('').level == 'empty'
    assert check_directory(str(tmp_path)).level == LEVEL_OK
    assert check_directory(str(tmp_path / 'a' / 'b')).level == LEVEL_INFO         # its nearest existing parent is writable
    (tmp_path / 'f').write_text('x')
    assert check_directory(str(tmp_path / 'f')).level == LEVEL_ERROR
    assert check_directory(str(tmp_path / 'f' / 'sub')).level == LEVEL_ERROR


def test_repository_location_finds_a_containing_git_root_and_relative_folder(tmp_path):
    root = tmp_path / 'catalogue'
    root.mkdir()
    subprocess.run(['git', '-C', str(root), 'init'], check=True, capture_output=True)
    nested = root / 'records' / 'films'
    nested.mkdir(parents=True)

    assert repository_location(str(nested))[:2] == (str(root), 'records/films')
    assert repository_location(str(root))[:2] == (str(root), '')
    assert repository_location(str(tmp_path / 'outside'))[2].level == LEVEL_ERROR


def test_repository_location_ignores_a_directory_merely_named_dot_git(tmp_path):
    (tmp_path / '.git').mkdir()
    location = tmp_path / 'records'
    location.mkdir()

    assert repository_location(str(location))[2].level == LEVEL_ERROR


@pytest.mark.skipif(os.name == 'nt' or os.geteuid() == 0, reason='permissions are not enforced for root or on Windows')
def test_a_directory_that_cannot_be_written_is_refused(tmp_path):
    locked = tmp_path / 'locked'
    locked.mkdir()
    locked.chmod(0o500)
    try:
        assert check_directory(str(locked)).level == LEVEL_ERROR and 'not writable' in check_directory(str(locked)).message
        assert check_directory(str(locked / 'child')).level == LEVEL_ERROR
    finally:
        locked.chmod(0o700)


def test_a_repository_must_be_a_git_working_tree(tmp_path):
    plain = tmp_path / 'plain'
    plain.mkdir()
    repo = tmp_path / 'repo'
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)

    assert check_repository('').level == 'empty'
    assert 'does not exist' in check_repository(str(tmp_path / 'nope')).message
    assert 'not a git repository' in check_repository(str(plain)).message
    assert check_repository(str(repo)).level == LEVEL_OK
    (repo / 'sub').mkdir()
    assert check_repository(str(repo / 'sub')).level == LEVEL_OK        # inside a working tree counts (as publish sees it)


@pytest.mark.parametrize('text,level', [('', 'empty'), ('beq/xml', LEVEL_OK), ('img\\x', LEVEL_OK), ('/abs', LEVEL_ERROR),
                                        ('C:\\x', LEVEL_ERROR), ('../up', LEVEL_ERROR), ('a/../b', LEVEL_ERROR)])
def test_a_folder_inside_a_repository_is_relative_and_stays_inside(text, level):
    assert check_relative_dir(text).level == level


def test_the_owner_and_name_come_from_a_plain_github_remote_only(tmp_path):
    repo = tmp_path / 'r'
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    assert remote_owner_and_name(str(repo)) is None                     # no remote
    subprocess.run(['git', '-C', str(repo), 'remote', 'add', 'origin', 'ssh://gitea.lan/me/r.git'], check=True)
    assert remote_owner_and_name(str(repo)) is None
    subprocess.run(['git', '-C', str(repo), 'remote', 'set-url', 'origin', 'https://github.com/me/r.git'], check=True)
    assert remote_owner_and_name(str(repo)) == ('me', 'r')
    assert remote_owner_and_name(str(tmp_path / 'nowhere')) is None


def test_external_ids_are_parsed_and_a_bad_pair_is_explained():
    assert parse_external_ids('imdb=tt0113277, tmdb: 603\nfoo=bar baz') == {'imdb': 'tt0113277', 'tmdb': '603', 'foo': 'bar baz'}
    assert parse_external_ids('') == {}
    with pytest.raises(ValueError, match='name=value'):
        parse_external_ids('imdb')


def test_the_folder_of_a_path_keeps_its_separators():
    assert folder_of('/films/Kids/a.mkv') == '/films/Kids'
    assert folder_of('D:\\Films\\Kids\\a.mkv') == 'D:\\Films\\Kids'
    assert folder_of('a.mkv') == 'a.mkv'


def _real_rows(tmp_path, rows):
    make_index(tmp_path, rows)
    with LibraryIndex(index_path(str(tmp_path))) as index:
        return index.titles()


def test_the_preview_evaluates_a_rule_exactly_as_discovery_does_over_a_row(tmp_path):
    rows = _real_rows(tmp_path, [
        title_row('a', 'Alpha', path='/films/Kids/a.mkv', kind='movie', year='2001', source='films'),
        title_row('b', 'Beta', path='/FILMS/kids/sub/b.mkv', kind='movie', year='1959', source='disk'),
        title_row('c', 'Gamma', path='/films/Other/c.mkv', kind='tv', year='1999', source='films',
                  external_ids=json.dumps({'imdb': 'tt1'}))])
    rules = [rule_from_config(e) for e in ({'path': '/films/kids'}, {'year': '<1960'}, {'source': 'films', 'kind': 'tv'},
                                          {'external_ids': {'imdb': 'tt1'}})]

    preview = preview_rules(rows, rules, {}, ['films', 'disk'])

    assert preview.per_rule == (2, 1, 1, 1)          # case and separators do not matter; a source rule reads the row's source
    assert (preview.total, preview.by_rules, preview.by_id) == (3, 3, 0)
    assert preview.notes == ('Rule 4 uses external ids: 2 titles have no external ids in the index, so it cannot match them.',)


def test_a_title_ignored_by_id_that_no_rule_matches_is_counted_once_as_by_id(tmp_path):
    rows = _real_rows(tmp_path, [title_row('a', 'Alpha', kind='tv'), title_row('b', 'Beta', kind='tv'),
                                 title_row('c', 'Gamma')])

    preview = preview_rules(rows, [rule_from_config({'kind': 'tv'})], {'a': 'x', 'c': 'y', 'zzz': 'not a title'})

    assert (preview.total, preview.by_rules, preview.by_id, preview.matched) == (3, 2, 1, 3)


def test_with_no_rows_nothing_can_be_said(tmp_path):
    preview = preview_rules([], [rule_from_config({'kind': 'tv'})], {})
    assert preview.scanned is False and preview.per_rule == (0,) and 'Rescan' in preview.text()


def _setup(profile, **settings):
    return WorkListSetup(profile, ScanSettings.from_profile(profile) if profile else None, 'file')


def test_discovery_changes_when_a_source_a_rule_an_ignore_or_a_scan_setting_changes():
    base = profile_from_config({'sources': [{'name': 'a', 'kind': 'filesystem', 'globs': ['/a']}],
                                'run': {'work_dir': '/w', 'queue_dir': '/q', 'designer': 'd'}})
    same = profile_from_config(base.to_config())
    assert not discovery_changed(_setup(base), _setup(same))

    def changed(profile):
        return discovery_changed(_setup(base), _setup(profile))

    assert changed(replace(base, sources=base.sources + (SourceSpec('b', 'filesystem', {'globs': ['/b']}),)))
    assert changed(replace(base, sources=(SourceSpec('a', 'filesystem', {'globs': ['/other']}),)))
    assert changed(replace(base, ignore=(rule_from_config({'kind': 'tv'}),)))
    assert changed(replace(base, ignored_titles={'x': ''}))
    assert changed(with_config(base, 'run', 'tv_mode', 'season'))
    assert changed(replace(base, work_dir='/elsewhere'))
    assert not changed(with_config(base, 'sync', 'commit_message', 'hello'))   # nothing a scan reads
    assert discovery_changed(_setup(None), _setup(base)) and not discovery_changed(_setup(None), _setup(None))
