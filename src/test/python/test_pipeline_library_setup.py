'''
pipeline/library/setup.py: from a profile file (and options over it) to a library run. The command line and the pipeline
service both build their runs here, so the same file is the same run from either.
'''
import pytest

from pipeline.designer.registry import registered_designers, unregister_designer
from pipeline.library.setup import configured_values, effective_profile, index_settings, run_config_from_values, \
    run_profile, scan_values, stage_settings


def _config(**sections):
    config = {'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': ['/media/films']}]}
    config.update(sections)
    return config


@pytest.fixture
def declared_designer():
    yield 'setup-test'
    for name in ('setup-test', 'http://designer.local/design'):
        if name in registered_designers():
            unregister_designer(name)


def test_an_override_wins_and_a_missing_one_leaves_the_file_alone():
    config = _config(run={'work_dir': '/w', 'designer': 'a'})

    assert configured_values(config, 'run', {'designer': 'b', 'queue_dir': None}) == \
        {'work_dir': '/w', 'designer': 'b', 'xml_repo': None, 'xml_dir': None}


def test_the_current_repository_names_are_read_into_the_legacy_ones_and_a_disagreement_is_refused():
    assert configured_values(_config(sync={'filter_repo': '/f'}), 'sync')['xml_repo'] == '/f'
    with pytest.raises(ValueError, match='sync.filter_repo and legacy sync.xml_repo disagree'):
        configured_values(_config(sync={'filter_repo': '/f', 'xml_repo': '/x'}), 'sync')


def test_a_scan_takes_run_over_sync():
    values = scan_values(_config(run={'work_dir': '/run'}, sync={'work_dir': '/sync', 'images_repo': '/img'}))

    assert values['work_dir'] == '/run' and values['images_repo'] == '/img'


def test_the_run_profile_takes_the_values_directories_and_lends_its_own():
    config = _config(run={'queue_dir': '/q'}, sync={'work_dir': '/from-sync'})
    values = configured_values(config, 'run', {'queue_dir': '/flag-q'})

    profile = run_profile(config, values)

    assert values['work_dir'] == '/from-sync'                         # the profile keeps it under sync:
    assert profile.work_dir == '/from-sync' and profile.queue_dir == '/flag-q'   # the value wins over the file


def test_a_profile_with_no_sources_is_refused():
    with pytest.raises(ValueError, match='lists no sources'):
        run_profile({'run': {}}, {})


def test_effective_profile_only_changes_what_is_given():
    profile = run_profile(_config(run={'work_dir': '/w', 'queue_dir': '/q'}), {})

    assert effective_profile(profile, {}) is profile
    assert effective_profile(profile, {'xml_repo': '/x'}).xml_repo == '/x'


def test_the_run_config_registers_the_files_designers_and_refuses_an_unknown_one(declared_designer):
    config = _config(designers={declared_designer: 'http://designer.local/design'})
    values = {'work_dir': '/w', 'queue_dir': '/q', 'designer': declared_designer, 'parallelism': {'extract': 2},
              'tv_mode': 'season', 'target_fs': 500}

    run_config = run_config_from_values(values, config)

    assert declared_designer in registered_designers()
    assert (run_config.designer, run_config.extract_parallelism, run_config.design_parallelism) == (declared_designer, 2, 1)
    assert run_config.tv_mode == 'season' and run_config.config.target_fs == 500
    with pytest.raises(ValueError, match="designer 'nobody' is not registered"):
        run_config_from_values({**values, 'designer': 'nobody'}, config)
    with pytest.raises(ValueError, match='work-dir is required'):
        run_config_from_values({**values, 'work_dir': ''}, config)


def test_a_designer_given_as_a_url_is_registered_under_it(declared_designer):
    url = 'http://designer.local/design'

    assert run_config_from_values({'work_dir': '/w', 'queue_dir': '/q', 'designer': url}, _config()).designer == url


@pytest.mark.parametrize('through, publishes', [('extract', False), ('design', False), ('publish', True), ('commit', True)])
def test_stage_settings_publish_only_when_the_run_goes_that_far(through, publishes):
    config = _config(sync={'filter_repo': '/filters', 'images_repo': '/images', 'push': False})
    profile = run_profile(config, {'work_dir': '/w', 'queue_dir': '/q'})

    settings, publish = stage_settings(profile, config, {'work_dir': '/w', 'queue_dir': '/q', 'designer': 'd'}, through)

    assert settings.xml_repo == '/filters' and settings.images_repo == '/images' and settings.designer == 'd'
    assert (publish is not None) is publishes
    if publish is not None:
        assert publish.xml_repo.local_path == '/filters' and publish.push is False


def test_the_index_is_judged_with_the_profiles_directories_when_the_values_have_none():
    profile = run_profile(_config(run={'work_dir': '/w', 'queue_dir': '/q'}), {})

    assert (index_settings(profile, {}).work_dir, index_settings(profile, {'queue_dir': '/other'}).queue_dir) == \
        ('/w', '/other')
