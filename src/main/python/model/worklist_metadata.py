'''
The metadata and artwork of a title, on the title page -- design/library-sync/workflow-rework/design.md §12.10, chunk 27b.

`MetadataPanel` is the form: **Essentials** (title, year, audio types, edition, season and episodes, note, warning, TMDB id
and *Reload*) and a collapsed **More** (the other `BeqMetadata` fields), the artwork (browse, download, clear, a preview)
and one status line. It edits `QueueEntry.meta` and `art_path`/`art_overridden` of the title the page is showing, and
nothing else. `model.worklist_title.TitlePage` composes it (the header badge, which decisions are offered, moving between
titles) and tells the window when an edit changed what the index would say.

**An edit is never lost, and never silently applied to something it was not made against.**

* What is typed is *dirty*, field by field (`textEdited`: a person typing, not a `setText`). It is saved when a field is left
  (`editingFinished`, which Enter also emits), and `flush()` saves it before the page moves to another title, decides one or
  is left. A failed save -- unparseable episodes, a write error -- says why on the status line and keeps the edit dirty;
  `flush()` then returns False and the page does not move.
* The write **reads the entry again and merges only the fields the person touched into the fresh entry's `meta`**
  (`save_meta_changes`), never the on-screen copy: a run that resolved another field meanwhile keeps it. A title whose entry
  has vanished is reported, not recreated. The status of the entry does not matter to a metadata edit -- pending, skipped,
  accepted, published and rejected titles are all editable -- but a title a run is working on is read-only (a design in
  flight writes a new entry over whatever is there).
* Only what is invalid *to parse* is refused (episodes that are not numbers and ranges). Everything else is saved as typed and
  `problems()` says what `pipeline.metadata.validate()` would refuse to publish, by the test the index uses
  (`pipeline.library.status.metadata_problems`, over the profile's `meta_defaults`). For a pending, accepted or published
  title the header badge and the work list's "metadata incomplete" therefore agree; the index judges no other status, so a
  skipped or rejected title's gaps are listed on the page without the alarm (`badge_alarms`).
* A blank field means "unset": its key is *removed* from `meta` -- the title, year and audio types too -- so a default
  (`meta_defaults`, `BeqMetadata`'s own) applies again instead of a `''`/`[]` shadowing it. Nothing reads those three keys
  by index (`publication_meta`, `metadata_problems`, the index rows, the commit message and the old dialog all default a
  missing one), and a run that redesigns the title fills what is missing again. Episodes are the exception: a blank box is
  saved as `[]`, "no episodes" being a real answer that a redesign must not refill.
* Changing the season, or the TMDB id, drops the TMDB season id and episode count a run looked up (they belong to the old
  season of the old series; the catalogue then gets the plain season text instead of another show's structured season).

TMDB *Reload* and an artwork download run on the thread pool (`QRunnable` + a signals object) and never raise: a missing key or
a network error is a sentence on the status line, with the API key taken out of it (`redact`: `requests` puts the whole URL,
`?api_key=...`, in the text of its errors). A lookup that returns after the page has moved to another title *or been left*
(Back, closing) is dropped (`leave()`), and the request itself times out (`pipeline.metadata.TMDB_TIMEOUT_SECONDS`), so Reload
cannot stay disabled behind a hung connection. A download is different on purpose: it lands on the title that asked, even
after the page moved on -- unless a run is working on that title by then (the design in flight would overwrite it): it is
refused with a message (on the status line, or the window's status bar if the title is not shown).
Reload fills only what TMDB knows (it has no audio types) and leaves the form dirty; it is saved like any other edit.
'''
import logging
import os
import traceback
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

import requests
from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QFileDialog, QLineEdit, QPushButton, QWidget

from model.preferences import TMDB_API_KEY
from model.worklist_artwork import ArtworkError, DownloadJob, check_local_image
from model.worklist_model import is_dark_palette, warning_colour
from pipeline.library.status import metadata_problems
from pipeline.metadata import BeqMetadata, format_episodes, parse_episodes, redact, tmdb_details_by_id, tmdb_lookup  # noqa: F401 (redact: the page's own tests and callers)
from pipeline.review import QueueEntry, read_entry, update_entry
from ui.worklistmetadata import Ui_metadataPanel

logger = logging.getLogger('worklist')

REMOVE = object()   # a change that unsets a key of `meta`

# the meta key of each text box, in the order they are shown
_FIELDS = {'title': 'titleField', 'year': 'yearField', 'audio_types': 'audioTypesField', 'edition': 'editionField',
           'season': 'seasonField', 'episodes': 'episodesField', 'note': 'noteField', 'warning': 'warningField',
           'the_movie_db': 'movieDbIdField', 'alt_title': 'altTitleField', 'sort_title': 'sortTitleField',
           'language': 'languageField', 'source': 'sourceField', 'rating': 'ratingField', 'author': 'authorField',
           'avs': 'avsField', 'runtime': 'runtimeField', 'gain': 'gainField'}
INDEXED_STATUSES = ('pending', 'accepted', 'published')   # the index says "metadata incomplete" of these, and of no other
NO_ENTRY = 'This title has no queue entry yet: there is nothing to edit until it is designed.'
UNREADABLE = 'The queue entry of this title could not be read (see above): there is nothing to edit until that is fixed.'
RUNNING = 'A run is working on this title now: its metadata cannot be edited until it finishes.'
_STANDING = (NO_ENTRY, UNREADABLE, RUNNING)   # sentences that say what the panel *is*, so they go when it stops being so
_TMDB_EXTRAS = ('overview', 'genres', 'collection')   # from a TMDB reload: kept, not shown in a box of their own
_FILL_FROM_TMDB = ('title', 'alt_title', 'year', 'rating', 'runtime', 'the_movie_db')
_PROBLEM_FIELD = (('title', 'titleField'), ('year', 'yearField'), ('audio type', 'audioTypesField'),
                  ('episodes', 'episodesField'))


# --- what is said and what is written (no widgets) ---------------------------------------------------------------------------

def field_texts(meta: Mapping[str, Any]) -> Dict[str, str]:
    ''' What each text box shows for `meta`: audio types comma separated, episodes as ranges ("1-3, 5"). '''
    texts = {}
    for key in _FIELDS:
        value = meta.get(key)
        if key == 'audio_types':
            texts[key] = ', '.join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value or '')
        elif key == 'episodes':
            try:
                texts[key] = format_episodes(value or [])
            except TypeError:   # not a list of numbers: show it as it is rather than lose it
                texts[key] = str(value)
        else:
            texts[key] = '' if value is None else str(value)
    return texts


def parse_audio_types(text: str) -> List[str]:
    return [t.strip() for t in text.split(',') if t.strip()]


def apply_changes(meta: Mapping[str, Any], changes: Mapping[str, Any]) -> dict:
    '''
    :return: a copy of `meta` with `changes` applied: a value replaces the key's, `REMOVE` unsets it. A change of the season
        *or of the TMDB id* also drops `season_id` and `season_episode_count`: they are the old season's, of the old series.
    '''
    merged = dict(meta)
    for key, value in changes.items():
        if value is REMOVE:
            merged.pop(key, None)
        else:
            merged[key] = value
    if any(_differs(meta, changes, key) for key in ('season', 'the_movie_db')):
        merged.pop('season_id', None)
        merged.pop('season_episode_count', None)
    return merged


def _differs(meta: Mapping[str, Any], changes: Mapping[str, Any], key: str) -> bool:
    ''' Whether `changes` sets `key` to something other than what `meta` has (unset and blank are the same). '''
    if key not in changes:
        return False
    new = '' if changes[key] is REMOVE else str(changes[key] or '')
    return new != str(meta.get(key) or '')


def save_meta_changes(queue_dir: str, entry_id: str, changes: Mapping[str, Any]) -> Tuple[QueueEntry, bool]:
    '''
    Applies `changes` to the entry **as it is on disk now** (not as it was when the page showed it).
    :return: (the entry, whether anything was written): an edit that leaves the metadata as it is writes nothing.
    :raises FileNotFoundError: if the title has no queue entry.
    '''
    fresh = read_entry(queue_dir, entry_id)
    merged = apply_changes(fresh.meta, changes)
    if merged == fresh.meta:
        return fresh, False
    return update_entry(queue_dir, entry_id, meta=merged), True


def badge_alarms(problems: List[str], status: str) -> bool:
    '''
    Whether the badge should warn. The index judges the metadata of a pending, accepted or published title only; a skipped or
    rejected one is not headed for the catalogue, so what is missing is said there without the alarm.
    '''
    return bool(problems) and status in INDEXED_STATUSES


def badge_text(problems: List[str], status: str, unsaved: bool) -> str:
    '''
    The header line: what is missing, or that the metadata is complete. "Ready to publish" is only said of a title that has
    been accepted; before that the metadata being complete is all that is known. A skipped or rejected title is not "not
    ready to publish" -- it is not going to be published as it is -- so its gaps are only listed (`badge_alarms`).
    '''
    if problems:
        text = ('Not ready to publish: ' if badge_alarms(problems, status) else 'Metadata incomplete: ') + '; '.join(problems)
    else:
        text = 'Ready to publish' if status in ('accepted', 'published') else 'Metadata complete'
    return text + (' (unsaved edits)' if unsaved else '')


def ok_colour() -> str:
    return '#7ee787' if is_dark_palette() else '#1a7f37'


# --- TMDB, off the UI thread ------------------------------------------------------------------------------------------------

class _TmdbSignals(QObject):
    finished = Signal(object)       # a `BeqMetadata`
    failed = Signal(str)


class _TmdbJob(QRunnable):
    def __init__(self, api_key: str, tmdb_id: str, title: str, year: str, kind: str):
        super().__init__()
        self.signals = _TmdbSignals()
        self._args = (api_key, tmdb_id, title, year, kind)

    def run(self):
        api_key, tmdb_id, title, year, kind = self._args
        try:
            found = tmdb_details_by_id(tmdb_id, api_key, kind=kind) if tmdb_id else \
                tmdb_lookup(title, year, api_key, kind=kind)
        except requests.RequestException as error:      # HTTPError, ConnectionError, Timeout...
            self.signals.failed.emit(redact(f'TMDB lookup failed: {error}', api_key))
        except Exception as error:                      # an answer that is not what TMDB says it is
            # the traceback is redacted too: an exception's text may carry the URL, key and all
            logger.error('TMDB lookup for %s failed: %s', tmdb_id or title,
                         redact(''.join(traceback.format_exception(error)), api_key))
            self.signals.failed.emit(redact(f'TMDB lookup failed: {type(error).__name__}: {error}', api_key))
        else:
            self.signals.finished.emit(found)


# --- the panel --------------------------------------------------------------------------------------------------------------

class MetadataPanel(QWidget, Ui_metadataPanel):
    '''
    :param preferences: the TMDB key.
    :param queue_dir: where the queue entries are (asked each time).
    :param running: the titles a run is working on now, by id (asked each time): they are read-only.
    :param meta_defaults: what publish fills in where a title's metadata is silent (asked each time).
    :param choose_file: asks the person for an image file; '' for none. The file dialog unless given.
    '''
    saved = Signal(str)             # a title's id: its metadata or artwork was written
    edited = Signal()               # something was typed, or the form was reloaded: the validity may have changed
    tmdb_finished = Signal(str)     # the message shown once a lookup ended, whatever the outcome ('' if it was dropped)
    art_finished = Signal(str)      # the same for a download
    elsewhere = Signal(str)         # something to tell about a title that is not on the page (a download refused for it)

    def __init__(self, parent, preferences, queue_dir: Callable[[], str], running: Callable[[], Mapping[str, str]],
                 meta_defaults: Callable[[], Optional[dict]], choose_file: Optional[Callable[[], str]] = None):
        super().__init__(parent)
        self.setupUi(self)
        self._preferences, self._queue_dir, self._running, self._defaults = preferences, queue_dir, running, meta_defaults
        self.choose_file = choose_file or self._ask_for_file
        self._title_id = ''
        self._entry: Optional[QueueEntry] = None
        self._kind = 'movie'
        self._dirty: Set[str] = set()
        self._extras: Dict[str, Any] = {}
        self._generation = 0            # bumped per title shown: a lookup that comes back to another one is dropped
        self._tmdb_busy = self._art_busy = False
        self._error = ''                # why the entry could not be read, if that is why there is none
        self._highlighted: Optional[QLineEdit] = None
        self._boxes = {key: getattr(self, attr) for key, attr in _FIELDS.items()}
        # A box that is being destroyed while it has the keyboard says `editingFinished`: by then the panel is half gone and
        # saving into it (or writing its status line) crashes the process. Once Qt has begun deleting the panel nothing is saved.
        self._going = going = [False]
        self.destroyed.connect(lambda *_: going.__setitem__(0, True))
        for key, box in self._boxes.items():
            box.textEdited.connect(lambda _text, key=key: None if going[0] else self._on_typed(key))
            box.editingFinished.connect(lambda: None if going[0] else self.flush())
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)     # Enter in a field must never click a button
        self.metadataScroll.viewport().setAutoFillBackground(False)   # the form sits on the tab, not on a panel of its own
        self.metadataFormHost.setAutoFillBackground(False)
        self.moreButton.toggled.connect(self._on_more)
        self.saveMetadataButton.clicked.connect(lambda: self.flush())
        self.revertButton.clicked.connect(lambda: self.discard())
        self.reloadTmdbButton.clicked.connect(lambda: self.reload_tmdb())
        self.browseArtButton.clicked.connect(lambda: self.browse_art())
        self.downloadArtButton.clicked.connect(lambda: self.download_art())
        self.clearArtButton.clicked.connect(lambda: self.clear_art())
        self.artUrlField.returnPressed.connect(lambda: self.download_art())
        self._refresh_enabled()

    # --- showing an entry ---------------------------------------------------------------------------------------------------

    @property
    def title_id(self) -> str:
        return self._title_id

    @property
    def dirty(self) -> bool:
        return bool(self._dirty)

    def show_entry(self, title_id: str, entry: Optional[QueueEntry], kind: str = 'movie', keep_edits: bool = False,
                   error: str = '') -> None:
        '''
        Shows `entry`. What is typed is kept (`keep_edits`, for the same title only: a re-read after a refresh) -- otherwise the
        caller has flushed or discarded it. `error` is why there is no entry, if it is there but could not be read.
        '''
        self.clear_highlight()
        self._error = error
        keep = keep_edits and title_id == self._title_id
        if not keep:
            self._dirty, self._extras = set(), {}
            self.metadataStatusLabel.setText('')
        if title_id != self._title_id:
            self._generation += 1
            self._tmdb_busy = self._art_busy = False
        self._title_id, self._entry, self._kind = title_id, entry, kind
        meta = entry.meta if entry is not None else {}
        for key, text in field_texts(meta).items():
            if key not in self._dirty and self._boxes[key].text() != text:
                self._boxes[key].setText(text)
        genres = meta.get('genres') or []
        self.genresLabel.setText(', '.join(g.get('name', '') for g in genres if isinstance(g, dict)))
        self._render_art()
        self._refresh_enabled()

    def refresh_enabled(self) -> None:
        ''' A run started or finished with this title. '''
        self._refresh_enabled()

    def _editable(self) -> bool:
        return self._entry is not None and self._title_id not in self._running()

    def _refresh_enabled(self) -> None:
        editable = self._editable()
        for box in self._boxes.values():
            box.setReadOnly(not editable)
        for widget in (self.artUrlField, self.browseArtButton, self.clearArtButton):
            widget.setEnabled(editable)
        self.reloadTmdbButton.setEnabled(editable and not self._tmdb_busy)
        self.downloadArtButton.setEnabled(editable and not self._art_busy)
        self.saveMetadataButton.setEnabled(editable and bool(self._dirty))
        self.revertButton.setEnabled(bool(self._dirty))
        standing = (UNREADABLE if self._error else NO_ENTRY) if self._entry is None else '' if editable else RUNNING
        shown = self.metadataStatusLabel.text()
        if standing and (self._entry is None or not self._dirty):
            self._say(standing)
        elif not standing and shown in _STANDING:
            self._say('')       # the run that held it read-only has ended: the sentence about it must not stay

    def leave(self) -> None:
        '''
        The page is being left (the caller has saved or discarded what was typed): what a lookup or a download still on its way
        would do to it is no longer wanted. A TMDB answer that arrives later is dropped -- it would fill the form of a page nobody
        is looking at, and be saved to this title when another one is opened. (A download still lands on the title that asked.)
        '''
        self._generation += 1
        self._tmdb_busy = self._art_busy = False
        self.clear_highlight()
        self._refresh_enabled()

    def _on_more(self, shown: bool) -> None:
        self.moreBox.setVisible(shown)
        self.moreButton.setArrowType(Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow)

    def _say(self, text: str, problem: bool = False) -> None:
        self.metadataStatusLabel.setText(text)
        self.metadataStatusLabel.setStyleSheet(f'color: {warning_colour().name()}' if problem else '')

    # --- what is typed ------------------------------------------------------------------------------------------------------

    def _on_typed(self, key: str) -> None:
        self.clear_highlight()
        self._dirty.add(key)
        self._say('Unsaved edits: saved when you leave the field or move on.')
        self._refresh_enabled()
        self.edited.emit()

    def changes(self) -> Dict[str, Any]:
        '''
        :return: what the person has changed, by `meta` key (see `apply_changes`).
        :raises ValueError: (with a sentence for the person) if the episodes are not numbers and ranges.
        '''
        changes: Dict[str, Any] = {}
        for key in self._dirty:
            if key in _TMDB_EXTRAS:
                changes[key] = self._extras[key]
                continue
            text = self._boxes[key].text().strip()
            if key == 'episodes':
                try:
                    changes[key] = parse_episodes(text)
                except ValueError as error:
                    raise ValueError(f'Episodes: {error} (use numbers and ranges such as 1-3, 5)') from None
            elif key == 'audio_types':
                changes[key] = parse_audio_types(text) or REMOVE
            else:
                changes[key] = text or REMOVE
        return changes

    def problems(self) -> List[str]:
        '''
        What `validate()` says of the metadata as it would be saved now (saved metadata with what is typed applied), by the
        same test the index uses, over the profile's defaults; or why the typed edit cannot be saved.
        '''
        if self._entry is None:
            return []
        try:
            meta = apply_changes(self._entry.meta, self.changes())
        except ValueError as error:
            return [str(error)]
        return list(metadata_problems(meta, self._defaults()))

    def highlight_problem(self) -> None:
        '''
        Marks the first box that a problem is about (a red border) **without moving the keyboard there**. It is what a refused
        Accept shows: the key that was refused must not leave the person one keystroke from typing into a field (the next A, or
        a held one, would be an "a" in the year). The mark goes when anything is typed or another title is shown.
        '''
        self.clear_highlight()
        problems = ' '.join(self.problems()).lower()
        target = next((attr for word, attr in _PROBLEM_FIELD if word in problems), 'titleField')
        box = getattr(self, target)
        box.setStyleSheet(f'QLineEdit {{ border: 2px solid {warning_colour().name()}; }}')
        self._highlighted = box

    def clear_highlight(self) -> None:
        if self._highlighted is not None:
            self._highlighted.setStyleSheet('')
            self._highlighted = None

    # --- saving -------------------------------------------------------------------------------------------------------------

    def flush(self) -> bool:
        '''
        Saves what is dirty. Returns True if nothing is left unsaved; False -- the reason is on the status line and the edit is
        kept -- if it could not be: episodes that do not parse, the title's entry gone, a run working on it, a write error.
        '''
        if not self._dirty:
            return True
        title_id, queue_dir = self._title_id, self._queue_dir()
        if self._entry is None or not queue_dir:
            self._say('Not saved: this title has no queue entry.', True)
            return False
        if title_id in self._running():
            self._say('Not saved: a run is working on this title now. Try again when it finishes.', True)
            return False
        try:
            changes = self.changes()
        except ValueError as error:
            self._say(f'Not saved: {error}', True)
            return False
        try:
            _, wrote = save_meta_changes(queue_dir, title_id, changes)
        except FileNotFoundError:
            self._say('Not saved: this title has no queue entry any more.', True)
            return False
        except Exception as error:   # a full disk, a permission, a file damaged since it was read
            logger.exception('Could not save the metadata of %s', title_id)
            self._say(f'Not saved: {type(error).__name__}: {error}', True)
            return False
        self._dirty, self._extras = set(), {}
        self._say('Saved.' if wrote else '')
        self._refresh_enabled()
        if wrote:
            self.saved.emit(title_id)
        return True

    def discard(self) -> None:
        ''' Throws away what is typed (Revert, or leaving after a save that failed): the boxes show what is saved. '''
        self._dirty, self._extras = set(), {}
        self.show_entry(self._title_id, self._entry, self._kind)
        self.edited.emit()

    # --- TMDB ---------------------------------------------------------------------------------------------------------------

    def reload_tmdb(self) -> bool:
        '''
        Looks the title up on TMDB by its id if it has one, else by title and year (a series if there is a season), on a
        worker. The result fills the boxes and leaves them dirty. Never raises; the outcome is on the status line.
        :return: True if a lookup was started.
        '''
        if not self._editable() or self._tmdb_busy:
            return False
        api_key = self._preferences.get(TMDB_API_KEY)
        tmdb_id = self.movieDbIdField.text().strip()
        title, year = self.titleField.text().strip(), self.yearField.text().strip()
        if not api_key:
            self._say('No TMDB API key is set (Preferences).', True)
            return False
        if not tmdb_id and not title:
            self._say('Enter a title (and year), or a TMDB id, to look up.', True)
            return False
        kind = 'tv' if self.seasonField.text().strip() or self._kind == 'tv' else 'movie'
        job = _TmdbJob(api_key, tmdb_id, title, year, kind)
        generation = self._generation
        job.signals.finished.connect(lambda found: self._on_tmdb(generation, found, title, year))
        job.signals.failed.connect(lambda message: self._on_tmdb_failed(generation, message))
        self._tmdb_busy = True
        self._refresh_enabled()
        self._say('Looking it up on TMDB...')
        QThreadPool.globalInstance().start(job)
        return True

    def _tmdb_done(self, generation: int) -> bool:
        ''' :return: True if the lookup is for the title now shown. '''
        current = generation == self._generation
        if current:
            self._tmdb_busy = False
            self._refresh_enabled()
        return current

    def _on_tmdb(self, generation: int, found: BeqMetadata, title: str, year: str) -> None:
        if not self._tmdb_done(generation):
            self.tmdb_finished.emit('')
            return
        if not found.the_movie_db:
            message = f'TMDB has no match for "{title}"' + (f' ({year})' if year else '') + '. Nothing was changed.'
            self._say(message, True)
            self.tmdb_finished.emit(message)
            return
        for key in _FILL_FROM_TMDB:
            value = getattr(found, key)
            if value:
                self._boxes[key].setText(value)
                self._dirty.add(key)
        for key in _TMDB_EXTRAS:
            value = getattr(found, key)
            if value:
                self._extras[key] = value
                self._dirty.add(key)
        self.genresLabel.setText(', '.join(g.get('name', '') for g in found.genres if isinstance(g, dict)))
        message = 'Filled in from TMDB. It is saved when you leave a field or move on (Revert undoes it).'
        self._say(message)
        self._refresh_enabled()
        self.edited.emit()
        self.tmdb_finished.emit(message)

    def _on_tmdb_failed(self, generation: int, message: str) -> None:
        if self._tmdb_done(generation):
            self._say(message, True)
            self.tmdb_finished.emit(message)
        else:
            self.tmdb_finished.emit('')

    # --- artwork ------------------------------------------------------------------------------------------------------------

    def _ask_for_file(self) -> str:
        path, _ = QFileDialog.getOpenFileName(self, 'Choose artwork', filter='Images (*.png *.jpg *.jpeg)')
        return path

    def _render_art(self) -> None:
        path = self._entry.art_path if self._entry is not None else None
        self.artPathField.setText(path or '')
        self.artPathField.setToolTip(path or '')
        pixmap = QPixmap(path) if path and os.path.isfile(path) else QPixmap()
        if not pixmap.isNull():
            self.artPreviewLabel.setPixmap(pixmap.scaled(
                self.artPreviewLabel.minimumWidth(), self.artPreviewLabel.minimumHeight(),
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.artPreviewLabel.clear()
            self.artPreviewLabel.setText('The chosen file is missing' if path else 'No artwork')

    def _write_art(self, title_id: str, **fields) -> bool:
        ''' Writes the artwork fields of `title_id`'s entry; the outcome is on the status line if it is still shown. '''
        try:
            update_entry(self._queue_dir(), title_id, **fields)
        except FileNotFoundError:
            self._art_message(title_id, 'Not saved: this title has no queue entry any more.', True)
            return False
        except Exception as error:
            logger.exception('Could not save the artwork of %s', title_id)
            self._art_message(title_id, f'Not saved: {type(error).__name__}: {error}', True)
            return False
        self.saved.emit(title_id)
        return True

    def _art_message(self, title_id: str, text: str, problem: bool = False) -> None:
        if title_id == self._title_id:
            self._say(text, problem)

    def browse_art(self) -> bool:
        ''' Uses an image file from disk as it is (the entry points at it: it is not copied). '''
        if not self._editable():
            return False
        path = self.choose_file()
        if not path:
            return False
        try:
            check_local_image(path)
        except ArtworkError as error:
            self._say(f'Not used: {error}.', True)
            return False
        return self._set_art(self._title_id, path)

    def _set_art(self, title_id: str, path: Optional[str]) -> bool:
        ''' `art_overridden` is True once a person has set or cleared it: automatic artwork never overwrites that. '''
        if not self._write_art(title_id, art_path=path, art_overridden=path is not None):
            return False
        self._entry = read_entry(self._queue_dir(), title_id) if title_id == self._title_id else self._entry
        self._render_art()
        self._art_message(title_id, 'Artwork set.' if path else 'Artwork cleared.')
        return True

    def clear_art(self) -> bool:
        if not self._editable():
            return False
        return self._set_art(self._title_id, None)

    def download_art(self) -> bool:
        ''' Downloads the image at the URL box into the queue's art cache, on a worker, and uses it. '''
        url = self.artUrlField.text().strip()
        if not self._editable() or self._art_busy or not url:
            return False
        queue_dir = self._queue_dir()
        if not queue_dir:
            return False
        title_id, generation = self._title_id, self._generation
        job = DownloadJob(url, queue_dir, title_id)
        job.signals.finished.connect(lambda path: self._on_downloaded(title_id, generation, path))
        job.signals.failed.connect(lambda message: self._on_download_failed(title_id, generation, message))
        self._art_busy = True
        self._refresh_enabled()
        self._say('Downloading...')
        QThreadPool.globalInstance().start(job)
        return True

    def _download_done(self, generation: int) -> None:
        if generation == self._generation:
            self._art_busy = False
            self._refresh_enabled()

    def _on_downloaded(self, title_id: str, generation: int, path: str) -> None:
        '''
        The artwork is for the title that asked, even if the page has moved on since; only the message is not shown then. Not
        if a run is working on it now: a design in flight writes a new entry over whatever is put there, so it would be lost
        (the file stays in the cache; download it again when the run has finished).
        '''
        self._download_done(generation)
        if title_id in self._running():
            message = (f'Not used: a run is working on {title_id} now, so the downloaded artwork was not set. '
                       f'Download it again when the run has finished.')
            self._art_message(title_id, message, True)
            self.art_finished.emit(message if title_id == self._title_id else '')
            if title_id != self._title_id:
                self.elsewhere.emit(message)
            return
        if self._set_art(title_id, path) and title_id == self._title_id:
            self.artUrlField.clear()
        self.art_finished.emit('' if title_id != self._title_id else self.metadataStatusLabel.text())

    def _on_download_failed(self, title_id: str, generation: int, message: str) -> None:
        self._download_done(generation)
        self._art_message(title_id, f'Not used: {message}.', True)
        self.art_finished.emit('' if title_id != self._title_id else self.metadataStatusLabel.text())
