'''
TODO R6: the title page's Metadata tab shows a poster designed on another machine (the container) as this machine sees it,
in the title's folder under this machine's work directory.

`import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import os

from PIL import Image

from model.worklist_metadata import MetadataPanel
from pipeline.review import QueueEntry
from test_worklist_title import _prefs


def test_a_poster_recorded_under_the_containers_root_is_shown_from_this_machines(qtbot, tmp_path):
    folder = tmp_path / 'desktop-work' / 'Heat - audio 1'
    folder.mkdir(parents=True)
    (folder / '.beq-title-id').write_text('jriver-1-2')
    Image.new('RGB', (20, 30), (10, 200, 10)).save(str(folder / 'poster.jpg'))
    panel = MetadataPanel(None, _prefs(tmp_path), lambda: str(tmp_path / 'queue'), lambda: {}, lambda: None,
                          work_dir=lambda: str(tmp_path / 'desktop-work'))
    qtbot.addWidget(panel)
    entry = QueueEntry(id='jriver-1-2', fs=1000, meta={'title': 'Heat'}, curve={},
                       art_path='/work/Heat - audio 1/poster.jpg')

    panel.show_entry('jriver-1-2', entry)

    assert panel.artPathField.text() == os.path.join(str(folder), 'poster.jpg')
    assert panel.artPreviewLabel.pixmap() is not None and not panel.artPreviewLabel.pixmap().isNull()
