'''
The release version has one source: the tag, which CI writes into src/main/python/VERSION for the app and the service.
pyproject.toml keeps no number of its own (the app is never built as a package), so the two cannot disagree.
'''
import pathlib
import tomllib

ROOT = pathlib.Path(__file__).parents[3]


def test_pyproject_keeps_no_version_of_its_own():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    assert 'version' not in project and 'version' in project.get('dynamic', [])


def test_every_release_build_writes_the_tag_into_the_version_file():
    for workflow in ('create-app.yaml', 'create-image.yaml'):
        text = (ROOT / '.github' / 'workflows' / workflow).read_text(encoding='utf-8')
        assert 'src/main/python/VERSION' in text, workflow


def test_the_release_build_tests_as_ci_does_without_ffmpeg():
    ''' The 2.2.0-alpha.1 app builds failed: no Qt platform or libpulse, and only they ran without ffmpeg. '''
    release = (ROOT / '.github' / 'workflows' / 'create-app.yaml').read_text(encoding='utf-8')
    tests = (ROOT / '.github' / 'workflows' / 'test.yaml').read_text(encoding='utf-8')
    command = 'uv run pytest src/test/python -n auto --cov=./src/main/python -o faulthandler_timeout=300'
    assert command in release and command in tests
    assert 'QT_QPA_PLATFORM: offscreen' in release and 'libpulse0' in release
    assert 'install -y ffmpeg' not in release and 'choco install ffmpeg' not in release   # the optional-ffmpeg run
