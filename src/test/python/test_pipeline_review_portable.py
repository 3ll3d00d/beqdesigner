'''
TODO R6: a work directory written by the container is reviewed from a desktop that sees it under another root. An entry's
poster is kept in the title's own folder, and found there whatever the root, so the title page shows it and the publish
digest is unchanged (nothing looks out of date because the reviewer's paths differ).
'''
import os
import shutil

from PIL import Image

from pipeline.library.artwork import resolve_art
from pipeline.library.source import LibraryItem
from pipeline.review import current_publish_digest, entry_art_path, read_entry, update_entry
from test_pipeline_library_commit import repos  # noqa: F401 (a fixture)
from test_pipeline_library_index import env  # noqa: F401
from test_pipeline_library_stages import _accepted


def _poster(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new('RGB', (20, 30), (200, 10, 10)).save(path)
    return path


def test_library_artwork_is_kept_in_the_titles_own_folder(tmp_path):
    media_art = _poster(str(tmp_path / 'media' / 'Heat' / 'folder.jpg'))
    item = LibraryItem(id='heat', source_path=str(tmp_path / 'media' / 'Heat' / 'Heat.mkv'), display_name='Heat',
                       art_candidates=(media_art,))
    title_dir = str(tmp_path / 'work' / 'Heat - audio 1')

    found = resolve_art(item, {}, title_dir)

    assert found == os.path.join(title_dir, 'poster.jpg') and os.path.isfile(found)
    assert open(found, 'rb').read() == open(media_art, 'rb').read()
    assert resolve_art(item, {}, None) == media_art   # nowhere to keep it: as found


def test_an_entry_written_under_one_root_opens_under_another_with_the_same_digest(env, repos, tmp_path):
    (a,), settings = _accepted(env, repos, 'a')
    folder = os.path.join(env.work, a.id)
    update_entry(env.queue, a.id, art_path=_poster(os.path.join(folder, 'poster.jpg')))
    entry = read_entry(env.queue, a.id)
    designed = current_publish_digest(entry, work_dir=env.work, has_image=True)

    desktop = str(tmp_path / 'nas' / 'beq' / 'work')   # the same folder, mounted somewhere else
    shutil.copytree(env.work, desktop)
    shutil.rmtree(folder)                               # nothing of the old root is left to find

    assert entry_art_path(entry, desktop) == os.path.join(desktop, a.id, 'poster.jpg')
    assert current_publish_digest(entry, work_dir=desktop, has_image=True) == designed


def test_a_poster_that_is_nowhere_stays_as_recorded(tmp_path):
    from pipeline.review import QueueEntry
    entry = QueueEntry(id='x', fs=1000, meta={}, curve={}, art_path='/work/x/poster.jpg')
    assert entry_art_path(entry, str(tmp_path)) == '/work/x/poster.jpg'
    assert entry_art_path(entry, None) == '/work/x/poster.jpg'
