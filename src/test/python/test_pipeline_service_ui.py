'''
The browser app served by the pipeline service (design/web-review.md §4): its files at /ui, the page for any path the app
routes itself, nothing outside the build, no token for the page, and what /ui says when the app is not installed.
'''
import json

import pytest
from fastapi.testclient import TestClient

from pipeline.service.__main__ import build
from pipeline.service.api import create_app
from pipeline.service.config import ServiceConfig
from pipeline.service.jobs import JobManager

PAGE = '<!doctype html><div id="root"></div>'


@pytest.fixture
def built_app(tmp_path):
    ''' What `npm run build` leaves: the page, hashed assets, and a file beside the page. '''
    dist = tmp_path / 'dist'
    (dist / 'assets').mkdir(parents=True)
    (dist / 'index.html').write_text(PAGE)
    (dist / 'assets' / 'index-abc123.js').write_text('console.log(1)')
    (dist / 'favicon.svg').write_text('<svg/>')
    (tmp_path / 'secret.txt').write_text('not for the web')
    return dist


@pytest.fixture
def serve(tmp_path):
    managers = []

    def make(ui_dir):
        manager = JobManager(lambda job, control: None)
        managers.append(manager)
        app = create_app(manager, ServiceConfig(profile_path=str(tmp_path / 'profile.json'), token='t'), env={},
                         ui_dir=str(ui_dir) if ui_dir else None, checks=lambda: [])
        return TestClient(app)
    yield make
    for manager in managers:
        manager.stop(grace_seconds=5)


def test_the_app_is_served_at_ui_and_the_service_root_sends_a_person_there(serve, built_app):
    client = serve(built_app)

    assert client.get('/', follow_redirects=False).headers['location'] == '/ui/'
    assert client.get('/ui', follow_redirects=False).headers['location'] == '/ui/'
    page = client.get('/ui/')
    assert (page.status_code, page.text, page.headers['cache-control']) == (200, PAGE, 'no-cache')
    assert page.headers['content-type'].startswith('text/html')


@pytest.mark.parametrize('path', ['/ui/titles', '/ui/titles/fs-alien', '/ui/jobs/123?x=1', '/ui/signin'])
def test_any_path_the_app_routes_itself_is_the_page(serve, built_app, path):
    response = serve(built_app).get(path)
    assert (response.status_code, response.text) == (200, PAGE)


def test_hashed_assets_are_cached_for_good_and_other_files_are_served(serve, built_app):
    client = serve(built_app)

    asset = client.get('/ui/assets/index-abc123.js')
    assert (asset.status_code, asset.text) == (200, 'console.log(1)')
    assert asset.headers['cache-control'] == 'public, max-age=31536000, immutable'
    assert client.get('/ui/favicon.svg').headers['cache-control'] == 'no-cache'


@pytest.mark.parametrize('path', ['/ui/assets/index-gone.js', '/ui/missing.css'])
def test_a_missing_file_is_a_404_not_the_page(serve, built_app, path):
    response = serve(built_app).get(path)
    assert response.status_code == 404 and response.headers['content-type'] == 'application/problem+json'


@pytest.mark.parametrize('path', ['/ui/%2e%2e/secret.txt', '/ui/..%2Fsecret.txt', '/ui/assets/%2e%2e/%2e%2e/secret.txt',
                                  '/ui/../secret.txt'])
def test_nothing_outside_the_build_is_served(serve, built_app, path):
    response = serve(built_app).get(path)
    assert response.status_code == 404 and 'not for the web' not in response.text


def test_the_page_needs_no_token_but_the_api_still_does(serve, built_app):
    client = serve(built_app)
    assert client.get('/ui/').status_code == 200
    assert client.get('/v1/status').status_code == 401


def test_without_the_app_ui_says_how_to_build_it_and_the_root_goes_to_the_api_page(serve):
    client = serve(None)

    response = client.get('/ui/')
    assert response.status_code == 404 and response.json()['title'] == 'Browser app not installed'
    assert 'npm run build' in response.json()['detail']
    assert client.get('/', follow_redirects=False).headers['location'] == '/docs'


def test_a_folder_that_is_not_a_built_app_is_refused(serve, tmp_path):
    with pytest.raises(ValueError, match='has no index.html'):
        serve(tmp_path)


def test_the_command_line_takes_the_app_from_ui_dir_and_refuses_a_folder_that_is_not_one(tmp_path, built_app, capsys,
                                                                                        monkeypatch):
    (tmp_path / 'media').mkdir()
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': [str(tmp_path / 'media')]}],
                                   'run': {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(tmp_path / 'queue'),
                                           'designer': 'manual'}}))
    env = {'BEQ_SERVICE_TOKEN': 't'}
    app, manager, _ = build(['--profile', str(profile), '--ui-dir', str(built_app)], env)
    try:
        with TestClient(app) as client:
            assert client.get('/ui/').text == PAGE
    finally:
        manager.stop(grace_seconds=5)

    monkeypatch.setenv('BEQ_SERVICE_UI', str(tmp_path / 'media'))
    with pytest.raises(SystemExit):
        build(['--profile', str(profile)], env)
    assert 'is not a built browser app' in capsys.readouterr().err
