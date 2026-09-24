from pathlib import Path

from pipeline.library.source import LibraryItem
from pipeline.library.union import reconstruct_claims
from pipeline.library.workdir import entry_directory, item_directory, work_ids
from pipeline.review import project_paths


def _item(identifier, title, stream=0):
    return LibraryItem(id=identifier, source_path=f'/films/{title}.mkv', display_name=title,
                       audio_stream=stream)


def test_extraction_folder_names_the_track_and_selected_stream(tmp_path):
    root = str(tmp_path)
    item = _item('jriver-123456789abc-42', 'Film: Director Cut', stream=2)

    folder = item_directory(root, item, create=True)

    assert Path(folder).name == 'Film_ Director Cut - audio 3'
    assert item_directory(root, item, create=True) == folder
    assert entry_directory(root, item.id) == folder
    assert work_ids(root) == {item.id}
    assert reconstruct_claims(root, None).ids == {item.id}
    assert project_paths(root, item.id)[0] == folder


def test_same_named_tracks_get_separate_readable_folders_without_losing_identity(tmp_path):
    root = str(tmp_path)
    first = _item('jriver-123456789abc-42', 'Film')
    second = _item('jriver-123456789abc-43', 'Film')

    assert Path(item_directory(root, first, create=True)).name == 'Film - audio 1'
    assert Path(item_directory(root, second, create=True)).name == 'Film - audio 1 (2)'
    assert work_ids(root) == {first.id, second.id}


def test_existing_id_folder_is_reused_for_cached_extraction(tmp_path):
    item = _item('jriver-123456789abc-42', 'Film', stream=1)
    old = tmp_path / item.id
    old.mkdir()

    assert item_directory(str(tmp_path), item, create=True) == str(old)
    assert entry_directory(str(tmp_path), item.id) == str(old)
    assert work_ids(str(tmp_path)) == {item.id}


def test_long_track_name_keeps_its_selected_stream_in_folder_name(tmp_path):
    item = _item('jriver-123456789abc-42', 'A' * 120, stream=4)

    assert Path(item_directory(str(tmp_path), item, create=True)).name.endswith(' - audio 5')
