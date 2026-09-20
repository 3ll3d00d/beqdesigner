'''The catalogue profile (design/library-sync/workflow-rework §12.4): the ordered sources, the old shape, round trips.'''
import json

import pytest
import yaml

from pipeline.library import profile as profile_module
from pipeline.library.filesystem import FilesystemLibrarySource
from pipeline.library.ignore import rule_from_config
from pipeline.library.profile import Profile, SourceSpec, build_source, load_profile, profile_from_config, \
    save_profile

_PROFILE = {
    'sources': [
        {'name': 'films', 'kind': 'jriver', 'host': 'media.local', 'port': 52199, 'browse_node_id': 1007,
         'path_mappings': [{'from': 'W:\\', 'to': '/media/films'}]},
        {'name': 'disk', 'kind': 'filesystem', 'globs': ['/mnt/extra/**/*.mkv']},
    ],
    'ignore': [{'path': '/media/films/Kids/**'}, {'kind': 'tv', 'reason': 'not doing TV'}],
    'ignore_titles': {'jriver-3fa9c2-1234': 'rip is broken'},
    'run': {'work_dir': '/work', 'queue_dir': '/queue', 'designer': 'rolloff', 'tv_mode': 'season'},
    'sync': {'xml_repo': '/xml', 'xml_dir': 'filters', 'images_repo': '/img', 'image_dir': 'images',
             'meta_defaults': {'source': 'Disc'}},
    'designers': {'rolloff': 'http://designer.local/design'},
}


def test_an_ordered_list_of_sources_keeps_its_order_and_settings():
    profile = profile_from_config(_PROFILE)

    assert [(s.name, s.kind) for s in profile.sources] == [('films', 'jriver'), ('disk', 'filesystem')]
    assert profile.sources[0].settings['browse_node_id'] == 1007
    assert profile.sources[1].settings == {'globs': ['/mnt/extra/**/*.mkv']}
    assert profile.source('disk').kind == 'filesystem'


def test_the_rest_of_the_profile_is_read_from_the_existing_sections():
    profile = profile_from_config(_PROFILE)

    assert (profile.work_dir, profile.queue_dir) == ('/work', '/queue')
    assert (profile.xml_repo, profile.xml_dir, profile.images_repo, profile.image_dir) == \
        ('/xml', 'filters', '/img', 'images')
    assert profile.ignore == (rule_from_config({'path': '/media/films/Kids/**'}),
                              rule_from_config({'kind': 'tv', 'reason': 'not doing TV'}))
    assert profile.ignored_titles == {'jriver-3fa9c2-1234': 'rip is broken'}


def test_work_and_queue_dirs_fall_back_to_the_sync_section():
    profile = profile_from_config({'sync': {'work_dir': '/w', 'queue_dir': '/q'}})

    assert (profile.work_dir, profile.queue_dir) == ('/w', '/q')
    assert profile_from_config({'run': {'work_dir': '/run'}, 'sync': {'work_dir': '/sync'}}).work_dir == '/run'


def test_ignored_titles_may_be_ids_or_id_and_reason_pairs():
    ids = profile_from_config({'ignore_titles': ['a', {'id': 'b', 'reason': 'why'}, {'id': 'c'}]}).ignored_titles

    assert ids == {'a': '', 'b': 'why', 'c': ''}
    with pytest.raises(ValueError, match='ignore_titles entries'):
        profile_from_config({'ignore_titles': [42]})


def test_the_older_config_shape_loads_as_a_profile_of_the_one_source_in_use():
    old = {'run': {'source': 'jriver', 'work_dir': '/work', 'queue_dir': '/queue'},
           'sources': {'jriver': {'host': 'media.local', 'port': 52199, 'browse_node_id': 1}, 'other': {'x': 1}},
           'sync': {'xml_repo': '/xml'}}

    profile = profile_from_config(old)

    assert profile.sources == (SourceSpec('jriver', 'jriver', {'host': 'media.local', 'port': 52199,
                                                               'browse_node_id': 1}),)
    assert profile.xml_repo == '/xml' and profile.ignore == () and profile.ignored_titles == {}


def test_an_older_filesystem_config_takes_its_globs_from_run():
    profile = profile_from_config({'run': {'source': 'filesystem', 'globs': ['/films']}})

    assert profile.sources == (SourceSpec('filesystem', 'filesystem', {'globs': ['/films']}),)


def test_a_config_with_no_sources_is_an_empty_profile_not_an_error():
    assert profile_from_config({}).sources == ()
    assert profile_from_config({'run': {'work_dir': '/w'}}).sources == ()


@pytest.mark.parametrize('config, message', [
    ({'sources': [{'kind': 'jriver'}]}, 'name and a kind'),
    ({'sources': [{'name': 'x'}]}, 'name and a kind'),
    ({'sources': ['jriver']}, 'name and a kind'),
    ({'sources': [{'name': 'x', 'kind': 'plex'}]}, 'kind must be one of'),
    ({'sources': [{'name': 'x', 'kind': 'jriver'}, {'name': 'x', 'kind': 'filesystem'}]}, 'must be unique'),
    ({'sources': 'jriver'}, 'must be a list'),
    ({'ignore': [{'kind': 'music'}]}, 'kind must be one of'),
])
def test_a_malformed_profile_is_refused_up_front(config, message):
    with pytest.raises(ValueError, match=message):
        profile_from_config(config)


def test_a_profile_round_trips_and_keeps_what_it_does_not_manage():
    profile = profile_from_config(_PROFILE)

    written = profile.to_config()

    assert profile_from_config(written) == profile
    assert written['designers'] == _PROFILE['designers']                       # not managed, kept
    assert written['run']['designer'] == 'rolloff' and written['sync']['meta_defaults'] == {'source': 'Disc'}
    assert written['sources'] == _PROFILE['sources']
    assert written['ignore'] == _PROFILE['ignore']


def test_editing_a_profile_rewrites_only_the_parts_it_manages():
    profile = profile_from_config(_PROFILE)
    edited = Profile(sources=tuple(reversed(profile.sources)), ignore=profile.ignore[:1], ignored_titles={},
                     work_dir='/elsewhere', queue_dir=profile.queue_dir, xml_repo=profile.xml_repo,
                     xml_dir=profile.xml_dir, images_repo='', image_dir='', config=profile.config)

    written = edited.to_config()

    assert [s['name'] for s in written['sources']] == ['disk', 'films']
    assert written['ignore'] == [{'path': '/media/films/Kids/**'}]
    assert 'ignore_titles' not in written
    assert written['run']['work_dir'] == '/elsewhere' and written['run']['designer'] == 'rolloff'
    assert 'images_repo' not in written['sync'] and written['sync']['xml_repo'] == '/xml'
    assert profile_from_config(written) == edited


def test_an_empty_profile_writes_no_empty_sections():
    assert Profile().to_config() == {'sources': []}


@pytest.mark.parametrize('suffix', ['.yaml', '.json'])
def test_a_profile_saves_and_loads_as_yaml_or_json(tmp_path, suffix):
    path = str(tmp_path / f'catalogue{suffix}')
    profile = profile_from_config(_PROFILE)

    save_profile(profile, path)

    assert load_profile(path) == profile
    text = (tmp_path / f'catalogue{suffix}').read_text()
    assert (yaml.safe_load(text) if suffix == '.yaml' else json.loads(text))['sources'][0]['name'] == 'films'


def test_a_config_whose_root_is_not_a_mapping_is_refused(tmp_path):
    (tmp_path / 'c.json').write_text('[1, 2]')

    with pytest.raises(ValueError, match='root must be an object'):
        load_profile(str(tmp_path / 'c.json'))


def test_build_source_builds_each_kind_and_says_what_is_missing(monkeypatch):
    seen = {}

    class Source:
        def __init__(self, host, port, browse_node_id, **kwargs):
            seen.update(host=host, port=port, node=browse_node_id, **kwargs)

    monkeypatch.setattr(profile_module, 'JRiverLibrarySource', Source)

    assert isinstance(build_source('filesystem', {'globs': ['/films']}), FilesystemLibrarySource)
    build_source('jriver', {'host': 'h', 'port': '52199', 'browse_node_id': 3, 'ssl': True,
                            'path_mappings': [{'from': 'W:\\', 'to': '/w'}]})
    assert (seen['host'], seen['port'], seen['node'], seen['ssl']) == ('h', 52199, 3, True)
    assert [(m.source, m.target) for m in seen['path_mappings']] == [('W:\\', '/w')]
    with pytest.raises(ValueError, match='glob is required'):
        build_source('filesystem', {})
    with pytest.raises(ValueError, match='host is required'):
        build_source('jriver', {'port': 1, 'browse_node_id': 1})
    with pytest.raises(ValueError, match='unsupported source'):
        build_source('plex', {})


# --- review fixes -------------------------------------------------------------------------------------------------------

def test_saving_a_profile_keeps_a_sync_work_dir_that_differs_from_the_run_one():
    config = {'sources': [], 'run': {'work_dir': '/w1'}, 'sync': {'work_dir': '/w2', 'queue_dir': '/q'}}

    written = profile_from_config(config).to_config()

    assert written['run']['work_dir'] == '/w1' and written['sync']['work_dir'] == '/w2'


def test_a_changed_work_dir_is_written_where_the_file_had_it():
    profile = profile_from_config({'sync': {'work_dir': '/w', 'xml_repo': '/x'}})

    written = Profile(work_dir='/new', xml_repo='/x', config=profile.config).to_config()

    assert written['sync']['work_dir'] == '/new' and 'run' not in written


def test_a_file_in_the_older_mapping_shape_keeps_the_settings_of_the_sources_it_does_not_use():
    old = {'run': {'source': 'jriver'},
           'sources': {'jriver': {'host': 'h', 'port': 1, 'browse_node_id': 2}, 'filesystem': {'globs': ['/films']}}}
    profile = profile_from_config(old)

    written = profile.to_config()

    assert written['sources']['filesystem'] == {'globs': ['/films']}          # not lost
    assert written['sources']['jriver'] == {'host': 'h', 'port': 1, 'browse_node_id': 2}
    assert profile_from_config(written) == profile


def test_converting_an_older_file_to_a_list_that_would_drop_a_source_is_refused_not_done_silently():
    old = {'run': {'source': 'jriver'},
           'sources': {'jriver': {'host': 'h', 'port': 1, 'browse_node_id': 2}, 'filesystem': {'globs': ['/films']}}}
    profile = profile_from_config(old)
    two = Profile(sources=(profile.sources[0], SourceSpec('disk', 'filesystem', {'globs': ['/x']})), config=profile.config)

    with pytest.raises(ValueError, match='cannot hold the unused source.*filesystem'):
        two.to_config()


def test_a_yaml_date_in_the_config_is_written_back_as_text_not_a_traceback(tmp_path):
    path = tmp_path / 'p.yaml'
    path.write_text('sync:\n  meta_defaults: {released: 2024-01-05}\nsources: []\n')

    profile = load_profile(str(path))

    assert profile.to_config()['sync']['meta_defaults'] == {'released': '2024-01-05'}


def test_a_value_that_is_not_plain_data_is_a_clear_value_error():
    with pytest.raises(ValueError, match='cannot be saved'):
        profile_module._plain({'a': {1, 2}})


def test_none_for_the_lists_a_profile_holds_is_no_entries():
    profile = Profile(sources=None, ignore=None, ignored_titles=None)

    assert (profile.sources, profile.ignore, profile.ignored_titles) == ((), (), {})
    assert profile.to_config() == {'sources': []}
    assert profile_from_config({'ignore_titles': None, 'ignore': None}).ignored_titles == {}


def test_a_malformed_yaml_file_is_a_value_error(tmp_path):
    path = tmp_path / 'bad.yaml'
    path.write_text('sources: [unclosed\n')
    with pytest.raises(ValueError, match='not valid YAML'):
        load_profile(str(path))
