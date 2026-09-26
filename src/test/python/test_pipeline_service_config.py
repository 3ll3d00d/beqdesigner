'''
pipeline/service/config.py (design/pipeline-service.md §7) and pipeline/service/context.py: the service's own settings, a
job's profile read again for every job, and the secrets the environment puts in it.
'''
import json

import pytest

from pipeline.service.config import ServiceConfig, load_service_config, secret, service_config_from_values
from pipeline.service.context import apply_secrets, env_name, load_context


# --- service.yaml -------------------------------------------------------------------------------------------------------

def test_no_file_is_the_defaults():
    config = load_service_config(None, '/config/profile.yaml', env={})

    assert config == ServiceConfig(profile_path='/config/profile.yaml')
    assert (config.host, config.port, config.allow_repository_writes, config.history_limit) == ('0.0.0.0', 8080, False, 200)


def test_a_file_sets_what_it_names(tmp_path):
    path = tmp_path / 'service.yaml'
    path.write_text('listen: {host: 127.0.0.1, port: 9000}\nallow_repository_writes: true\nhistory_limit: 5\n'
                    'shutdown_grace_seconds: 1.5\nstate_dir: /state\nschedule: {enabled: true}\nnotify: [{name: phone}]\n')

    config = load_service_config(str(path), 'p.yaml', env={})

    assert (config.host, config.port, config.allow_repository_writes, config.history_limit) == ('127.0.0.1', 9000, True, 5)
    assert config.shutdown_grace_seconds == 1.5 and config.state_dir == '/state'
    assert config.schedule == {'enabled': True} and config.notify == ({'name': 'phone'},)


@pytest.mark.parametrize('values, message', [
    ({'lisen': {}}, 'unknown key lisen'),
    ({'listen': {'port': 0}}, 'listen.port must be 1-65535'),
    ({'listen': {'port': '8080'}}, 'listen.port must be 1-65535'),
    ({'listen': {'address': 'x'}}, 'listen takes host and port'),
    ({'allow_repository_writes': 'yes'}, 'allow_repository_writes must be true or false'),
    ({'history_limit': 0}, 'history_limit must be 1 or more'),
    ({'shutdown_grace_seconds': -1}, 'shutdown_grace_seconds must be 0 or more'),
    ({'notify': {'name': 'x'}}, 'notify must be a list of targets'),
])
def test_a_typo_or_a_wrong_value_is_refused(values, message):
    with pytest.raises(ValueError, match=message):
        service_config_from_values(values, 'p.yaml', env={})


def test_the_token_comes_from_the_environment_or_a_secret_file_and_is_never_shown(tmp_path):
    token_file = tmp_path / 'token'
    token_file.write_text('from-file\n')

    assert service_config_from_values({}, 'p', env={'BEQ_SERVICE_TOKEN': 'plain'}).token == 'plain'
    config = service_config_from_values({}, 'p', env={'BEQ_SERVICE_TOKEN_FILE': str(token_file)})
    assert config.token == 'from-file' and 'from-file' not in repr(config)
    assert service_config_from_values({}, 'p', env={'BEQ_SERVICE_TOKEN': '  '}).token is None


def test_a_secret_file_that_cannot_be_read_is_an_error(tmp_path):
    with pytest.raises(ValueError, match='X_FILE names .*cannot be read'):
        secret({'X_FILE': str(tmp_path / 'missing')}, 'X')


# --- a job's profile ----------------------------------------------------------------------------------------------------

def _profile(tmp_path, **sections):
    config = {'sources': [{'name': 'films', 'kind': 'jriver', 'host': 'mc', 'port': 52199, 'browse_node_id': 1},
                          {'name': 'Other Disk', 'kind': 'jriver', 'host': 'mc2', 'port': 52199, 'browse_node_id': 2},
                          {'name': 'disk', 'kind': 'filesystem', 'globs': [str(tmp_path / 'media')]}],
              'run': {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(tmp_path / 'queue'), 'designer': 'd'}}
    config.update(sections)
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(config))
    return path, config


def test_env_names_are_upper_case_with_underscores():
    assert env_name('Other Disk') == 'OTHER_DISK' and env_name('rolloff-2') == 'ROLLOFF_2'


def test_the_environment_puts_the_secrets_in_the_profile_and_leaves_the_file_alone(tmp_path):
    _, config = _profile(tmp_path, designers={'rolloff': 'http://d/design', 'private': {'url': 'http://p', 'headers': {'A': '1'}}})
    env = {'TMDB_API_KEY': 'tmdb', 'JRIVER_PASSWORD_OTHER_DISK': 'two', 'JRIVER_PASSWORD': 'all',
           'BEQ_DESIGNER_HEADERS_ROLLOFF': '{"Authorization": "Bearer t"}', 'BEQ_DESIGNER_HEADERS_PRIVATE': '{"B": "2"}'}

    applied = apply_secrets(config, env)

    assert applied['run']['tmdb_api_key'] == 'tmdb'
    assert [s.get('password') for s in applied['sources']] == ['all', 'two', None]   # a named one wins; not filesystem
    assert applied['designers'] == {'rolloff': {'url': 'http://d/design', 'headers': {'Authorization': 'Bearer t'}},
                                    'private': {'url': 'http://p', 'headers': {'A': '1', 'B': '2'}}}
    assert 'password' not in config['sources'][0] and 'tmdb_api_key' not in config['run']   # a copy


def test_the_older_sources_mapping_takes_the_password_too():
    applied = apply_secrets({'sources': {'jriver': {'host': 'mc'}, 'filesystem': {'globs': []}}},
                            {'JRIVER_PASSWORD_JRIVER': 'pw'})

    assert applied['sources'] == {'jriver': {'host': 'mc', 'password': 'pw'}, 'filesystem': {'globs': []}}


@pytest.mark.parametrize('headers, message', [('not json', 'is not JSON'), ('["a"]', 'must be a JSON object')])
def test_designer_headers_must_be_a_json_object(headers, message):
    with pytest.raises(ValueError, match=f'BEQ_DESIGNER_HEADERS_D {message}'):
        apply_secrets({'designers': {'d': 'http://d'}}, {'BEQ_DESIGNER_HEADERS_D': headers})


def test_each_job_reads_the_profile_again(tmp_path):
    path, config = _profile(tmp_path)

    first = load_context(str(path), env={})
    config['run']['designer'] = 'other'
    path.write_text(json.dumps(config))
    second = load_context(str(path), env={})

    assert first.values['designer'] == 'd' and second.values['designer'] == 'other'
    assert second.work_dir == str(tmp_path / 'work') and [s.name for s in second.profile.sources] == \
        ['films', 'Other Disk', 'disk']


def test_the_profiles_directories_under_sync_are_used(tmp_path):
    path, _ = _profile(tmp_path, run={'designer': 'd'}, sync={'work_dir': str(tmp_path / 'w'), 'queue_dir': str(tmp_path / 'q')})

    context = load_context(str(path), env={})

    assert context.work_dir == str(tmp_path / 'w') and context.scan_settings().queue_dir == str(tmp_path / 'q')
