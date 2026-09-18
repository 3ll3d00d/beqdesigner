'''
design/library-sync-pipeline-plan.md Appendix C (chunk 4): the `LibrarySource` abstraction --
`pipeline.library.source`'s `LibraryItem`/`LibrarySource` shapes and `pipeline.library.registry`'s
in-process binding, mirroring `pipeline.designer.registry`'s exact pattern. Nothing else in the codebase
exercises this yet (chunks 5-7 will), so this chunk's own tests use a small fake source to prove the shape
works, not a real implementation.
'''
import pytest

from pipeline.library.registry import get_source, register_source, registered_sources, unregister_source
from pipeline.library.source import LibraryItem


class _FakeSource:
    def __init__(self, items):
        self._items = items

    def list_items(self, **query):
        return [i for i in self._items if not query.get('title_contains') or query['title_contains'] in i.title]


def test_library_item_defaults():
    item = LibraryItem(id='1', source_path='/media/x.mkv', display_name='X')
    assert item.id == '1'
    assert item.source_path == '/media/x.mkv'
    assert item.display_name == 'X'
    assert item.title is None
    assert item.year is None
    assert item.kind == 'movie'
    assert item.external_ids == {}
    assert item.audio_stream == 0
    assert item.playlist_name is None
    assert item.art_path is None
    assert item.meta == {}
    assert item.fingerprint == ''


def test_register_and_get_source_round_trips():
    items = [LibraryItem(id='1', source_path='/media/x.mkv', display_name='X', title='X')]
    fake = _FakeSource(items)
    register_source('fake', fake)
    try:
        assert get_source('fake') is fake
        assert list(get_source('fake').list_items()) == items
    finally:
        unregister_source('fake')


def test_get_source_unregistered_raises_keyerror():
    with pytest.raises(KeyError) as exc_info:
        get_source('does-not-exist')
    assert 'does-not-exist' in str(exc_info.value)
    assert 'registered' in str(exc_info.value)


def test_unregister_source_removes_it():
    register_source('fake', _FakeSource([]))
    unregister_source('fake')
    with pytest.raises(KeyError):
        get_source('fake')


def test_registered_sources_is_sorted():
    register_source('zebra', _FakeSource([]))
    register_source('alpha', _FakeSource([]))
    register_source('mike', _FakeSource([]))
    try:
        assert registered_sources() == ['alpha', 'mike', 'zebra']
    finally:
        unregister_source('zebra')
        unregister_source('alpha')
        unregister_source('mike')


def test_list_items_accepts_source_specific_query_kwargs():
    items = [
        LibraryItem(id='1', source_path='/media/a.mkv', display_name='A', title='Player One'),
        LibraryItem(id='2', source_path='/media/b.mkv', display_name='B', title='Something Else'),
    ]
    fake = _FakeSource(items)
    assert fake.list_items(title_contains='Player') == [items[0]]
    assert fake.list_items() == items


def test_pipeline_library_source_module_has_no_qtpy_import():
    import ast
    import pathlib
    source = (pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python' / 'pipeline' / 'library' /
             'source.py').resolve()
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith('qtpy') for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith('qtpy')


def test_pipeline_library_registry_module_has_no_qtpy_import():
    import ast
    import pathlib
    source = (pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python' / 'pipeline' / 'library' /
             'registry.py').resolve()
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith('qtpy') for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith('qtpy')
