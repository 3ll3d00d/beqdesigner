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
