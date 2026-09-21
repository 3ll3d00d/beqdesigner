'''
model/worklist_profile.py: where the work list's profile comes from until chunk 26c (a settings editor) exists -- a
profile file if the LIBRARY_PROFILE_PATH preference names one, else one built from the saved library preferences.
Pure functions over a real temp-file Preferences. `import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import json

import pytest
from qtpy.QtCore import QSettings

from model.jriver.connections import SavedConnection, save_connections
from model.library_sources import JRiverSourcePage
from model.preferences import DESIGNER_DEFAULT, DESIGNER_QUEUE_DIR, LIBRARY_FILESYSTEM_GLOBS, LIBRARY_IMAGES_REPO, \
    LIBRARY_JRIVER_BROWSE_NODE, LIBRARY_JRIVER_CONNECTION, LIBRARY_PROFILE_PATH, LIBRARY_SOURCE_DEFAULT, \
    LIBRARY_TV_MODE, LIBRARY_WORK_DIR, LIBRARY_XML_REPO, Preferences
from model.worklist_profile import ORIGIN_FILE, ORIGIN_PREFERENCES, bootstrap_profile, default_designer, load_setup
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.library.filesystem import FilesystemLibrarySource
from pipeline.library.jriver import JRiverLibrarySource
from pipeline.library.pathmap import PathMapping
from pipeline.library.profile import build_source
from pipeline.library.status import ScanSettings

DESIGNER = 'test.profile'


@pytest.fixture(autouse=True)
def _designer():
    register_designer(DESIGNER, lambda request: None)
    yield
    unregister_designer(DESIGNER)


@pytest.fixture
def prefs(tmp_path):
    (tmp_path / 'work').mkdir()
    p = Preferences(QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat))
    p.set(LIBRARY_WORK_DIR, str(tmp_path / 'work'))
    p.set(DESIGNER_QUEUE_DIR, str(tmp_path / 'queue'))
    p.set(DESIGNER_DEFAULT, DESIGNER)
    return p


def test_a_filesystem_profile_is_built_from_the_preferences(prefs, tmp_path):
    prefs.set(LIBRARY_SOURCE_DEFAULT, 'filesystem')
    prefs.set(LIBRARY_FILESYSTEM_GLOBS, ['/films/**/*.mkv', '/tv'])
    prefs.set(LIBRARY_XML_REPO, '/repos/xml')
    prefs.set(LIBRARY_IMAGES_REPO, '/repos/images')
    prefs.set(LIBRARY_TV_MODE, 'season')

    profile = bootstrap_profile(prefs, DESIGNER)

    assert [(s.name, s.kind, s.settings) for s in profile.sources] == \
           [('filesystem', 'filesystem', {'globs': ['/films/**/*.mkv', '/tv']})]
    assert profile.ignore == () and profile.ignored_titles == {}  # no rules
    assert (profile.work_dir, profile.queue_dir) == (str(tmp_path / 'work'), str(tmp_path / 'queue'))
    assert (profile.xml_repo, profile.images_repo) == ('/repos/xml', '/repos/images')
    settings = ScanSettings.from_profile(profile)
    assert settings.designer == DESIGNER and settings.tv_mode == 'season'
    assert isinstance(build_source(profile.sources[0].kind, profile.sources[0].settings), FilesystemLibrarySource)


def test_a_jriver_profile_carries_the_saved_server_node_and_mappings_and_builds_what_the_page_builds(qtbot, prefs):
    save_connections(prefs, [SavedConnection('media.local:52199', 'user', 'pass', True,
                                             path_mappings=(PathMapping('W:\\Films', '/mnt/films'),),
                                             field_mappings={'movie': {'imdb': ['My IMDb']}})])
    prefs.set(LIBRARY_SOURCE_DEFAULT, 'jriver')
    prefs.set(LIBRARY_JRIVER_CONNECTION, 'media.local:52199')
    prefs.set(LIBRARY_JRIVER_BROWSE_NODE, 1007)

    profile = bootstrap_profile(prefs, DESIGNER)

    (spec,) = profile.sources
    assert (spec.name, spec.kind) == ('jriver', 'jriver')
    built = build_source(spec.kind, spec.settings)
    page = JRiverSourcePage()
    qtbot.addWidget(page)
    page.load(prefs)
    expected = page.build_source()
    assert isinstance(built, JRiverLibrarySource)
    for attribute in ('host', 'port', 'browse_node_id', 'username', 'password', 'ssl', 'path_mappings',
                      'external_id_fields'):
        assert getattr(built, attribute) == getattr(expected, attribute), attribute
    json.dumps(profile.config)  # plain data, as a profile file is


def test_a_source_that_is_not_set_up_gives_no_source_and_a_problem_not_an_error(prefs):
    prefs.set(LIBRARY_SOURCE_DEFAULT, 'filesystem')  # no globs
    assert bootstrap_profile(prefs, DESIGNER).sources == ()
    prefs.set(LIBRARY_SOURCE_DEFAULT, 'jriver')      # no server saved
    assert bootstrap_profile(prefs, DESIGNER).sources == ()
    prefs.set(LIBRARY_SOURCE_DEFAULT, 'jriver')
    prefs.set(LIBRARY_JRIVER_CONNECTION, 'gone:1')   # a server that was deleted
    assert bootstrap_profile(prefs, DESIGNER).sources == ()

    setup = load_setup(prefs)

    assert not setup.ready
    assert any('No library source' in p for p in setup.problems)


def test_an_empty_preference_set_bootstraps_a_profile_with_no_work_dir_and_says_what_is_missing(tmp_path):
    empty = Preferences(QSettings(str(tmp_path / 'empty.ini'), QSettings.Format.IniFormat))

    setup = load_setup(empty)

    assert setup.origin == ORIGIN_PREFERENCES and setup.profile is not None
    assert setup.profile.work_dir == '' and setup.index_file is None
    assert not setup.ready
    assert any('work directory' in p for p in setup.problems)
    assert any('review queue' in p for p in setup.problems)


def test_a_complete_setup_is_ready_and_names_its_index(prefs, tmp_path):
    prefs.set(LIBRARY_FILESYSTEM_GLOBS, ['/films'])

    setup = load_setup(prefs)

    assert setup.ready and setup.problems == () and setup.origin == ORIGIN_PREFERENCES
    assert setup.index_file == str(tmp_path / 'work' / 'library-index.sqlite')
    assert setup.settings.designer == DESIGNER and setup.source_names == ['filesystem']


def test_a_missing_work_directory_is_a_problem_and_has_no_index(prefs, tmp_path):
    prefs.set(LIBRARY_FILESYSTEM_GLOBS, ['/films'])
    prefs.set(LIBRARY_WORK_DIR, str(tmp_path / 'absent'))

    setup = load_setup(prefs)

    assert not setup.ready and setup.index_file is None
    assert any('does not exist' in p for p in setup.problems)


def test_the_designer_falls_back_to_a_registered_one(prefs):
    from pipeline.designer.registry import registered_designers
    assert default_designer(prefs) == DESIGNER
    prefs.set(DESIGNER_DEFAULT, 'no.such.designer')
    assert default_designer(prefs) == registered_designers()[0]
    prefs.set(DESIGNER_DEFAULT, None)
    assert default_designer(prefs) == registered_designers()[0]


def test_a_profile_file_wins_over_the_preferences(prefs, tmp_path):
    prefs.set(LIBRARY_FILESYSTEM_GLOBS, ['/from/preferences'])
    (tmp_path / 'other-work').mkdir()
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps({
        'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': ['/from/file']},
                    {'name': 'second', 'kind': 'filesystem', 'globs': ['/second']}],
        'ignore': [{'kind': 'tv'}],
        'run': {'work_dir': str(tmp_path / 'other-work'), 'queue_dir': str(tmp_path / 'q2'), 'designer': DESIGNER}}))
    prefs.set(LIBRARY_PROFILE_PATH, str(path))

    setup = load_setup(prefs)

    assert setup.origin == ORIGIN_FILE and setup.path == str(path) and setup.ready
    assert setup.source_names == ['disk', 'second']
    assert len(setup.profile.ignore) == 1
    assert setup.settings.work_dir == str(tmp_path / 'other-work')


def test_a_profile_file_without_a_designer_uses_the_default_designer(prefs, tmp_path):
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps({'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': ['/f']}],
                                'run': {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(tmp_path / 'q')}}))
    prefs.set(LIBRARY_PROFILE_PATH, str(path))

    assert load_setup(prefs).settings.designer == DESIGNER


@pytest.mark.parametrize('content', ['{not json', '[1, 2]', '{"sources": [{"name": "x"}]}',
                                     '{"sources": [{"name": "a", "kind": "nope"}]}'])
def test_a_profile_file_that_cannot_be_read_is_reported_not_raised(prefs, tmp_path, content):
    path = tmp_path / 'bad.json'
    path.write_text(content)
    prefs.set(LIBRARY_PROFILE_PATH, str(path))

    setup = load_setup(prefs)

    assert setup.profile is None and setup.error and not setup.ready and setup.origin == ORIGIN_FILE


def test_a_missing_profile_file_returns_to_preferences_and_clears_its_stale_path(prefs, tmp_path):
    prefs.set(LIBRARY_PROFILE_PATH, str(tmp_path / 'nope.yaml'))

    setup = load_setup(prefs)

    assert setup.profile is not None and setup.origin == ORIGIN_PREFERENCES and not setup.error
    assert prefs.get(LIBRARY_PROFILE_PATH) == ''
