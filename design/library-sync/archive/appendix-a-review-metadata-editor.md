# Library sync plan -- archived handoff spec

> Part of the library sync plan -- **start at the index**: [`../../library-sync-pipeline-plan.md`](../../library-sync-pipeline-plan.md).
> Contains Appendix A (chunk 1). Section numbers are global across the plan; the index maps every `§` to its file.
> Status of what this file describes: **built and shipped** -- kept as the record of the original spec; read only if you need the detail

## Appendix A -- Chunk 1 handoff spec: review queue metadata + artwork editor

Self-contained enough to implement without reading the rest of this
document (though §3.1.2/§3.1.3 above have the full rationale). Touches
only existing files -- no new modules, no JRiver/library-source
dependency.

### A.1 Why

Two things are true of the review queue *today*, independent of any
library-source work:

1. `pipeline.review.publish_reviewed_queue()` requires a valid
   `BeqMetadata` (title/year/audio_types at minimum --
   `pipeline.metadata.validate()`), but `model/batch.py`'s batch-design
   flow never populates `QueueEntry.meta` at all (`design_and_queue()`
   is called with no `meta` argument) -- so a human currently cannot
   get a batch-designed title through review and publish without some
   other, unbuilt path supplying metadata.
2. `publish_reviewed_queue()` calls `session.report(...)` without ever
   passing `poster_path` -- `pipeline.publish.art.fetch_poster()` and
   `pipeline.publish.report.compose_with_poster()` both already exist
   and are unit-tested standalone, but nothing wires them together, so
   every report published through the queue today renders chart-only.

`ReviewQueueDialog` (`model/review.py`/`ui/review.py`) has no editing
capability at all today -- it only ever displays
`entry.meta.get('title', entry.id)`, read-only. This chunk fixes both
gaps by adding an editing UI there, reusing patterns that already exist
elsewhere in this codebase rather than inventing new ones:
`CreateAVSPostDialog.load_tmdb_info()`/`__apply_tmdb_metadata()`
(`model/postbuilder.py`) for the "paste an id or search by title/year"
TMDB flow, and `SaveReportDialog.choose_image()`/`load_image_from_url()`
(`model/report.py`) for the artwork browse/download flow.

### A.2 Scope

**In scope:**
- Two new additive fields on `pipeline.review.QueueEntry`:
  `art_path: Optional[str] = None`, `art_overridden: bool = False`.
- A "Metadata" tab in `ReviewQueueDialog`'s detail pane with editable
  `BeqMetadata` fields, a "Reload from TMDB" action, and a "Save
  Metadata" action that writes to `entry.meta` via
  `pipeline.review.update_entry()`.
- An "Artwork" section (can live in the same tab, below the metadata
  form) to browse a local image file or download one from a pasted
  URL, writing to `entry.art_path`/`entry.art_overridden`.
- Wiring `pipeline.review.publish_reviewed_queue()` to pass
  `poster_path=entry.art_path` into `session.report(...)`.
- Locking edits once an entry is `accepted`/`published` (read-only,
  not hidden).
- Tests: `pytest-qt` coverage in `gui/test_review_dialog.py`, plain
  `pytest` coverage in `test_pipeline_review.py`.

**Explicitly out of scope** (deferred, either to a later chunk or
indefinitely -- do not build these now):
- Anything JRiver/library-source (chunks 3-10) -- this chunk only adds
  the *editing* surface; nothing here auto-populates `meta`/`art_path`
  from a library yet. Auto-resolution lands in chunk 8.
- TV (`kind='tv'`) support in the reload-from-TMDB control -- default
  to `kind='movie'` only, same trim `pipeline.metadata.tmdb_lookup()`'s
  `kind` param would need a UI control for; add later if wanted.
- `genres`/`collection` as editable fields -- these are TMDB-shaped
  structures (`[{'id':.., 'name':..}]` / a dict), not sensibly hand-edited
  as text. Show `genres` read-only (comma-joined names) after a TMDB
  reload; leave `collection` invisible (still round-trips through
  `entry.meta` if TMDB set it, just not surfaced in the form).
- Combo-box pickers for `language`/`source` (postbuilder has fixed
  lists for these) -- plain text fields for v1, upgradeable later.
- Per-field autosave -- edits batch behind an explicit "Save Metadata"
  button (single-shot actions -- browse/download artwork, TMDB reload
  -- still take effect immediately on their own button, matching how
  accept/skip/reject already write immediately).
- Copying a browsed local artwork file into a managed cache directory
  -- `art_path` may point anywhere on disk the user chose; if they
  later move/delete it, publish fails loudly at that point (acceptable
  for v1, same risk `SaveReportDialog.choose_image()` already has).
  Only a *downloaded* URL gets written into a durable location (see
  A.5) since there is no original local file to reference.

### A.3 Data model changes

`pipeline/review.py`:

```python
@dataclass
class QueueEntry:
    id: str
    fs: int
    meta: dict
    curve: dict
    candidates: List[CandidateSummary] = field(default_factory=list)
    decline_reason: Optional[str] = None
    decline_message: Optional[str] = None
    status: str = 'pending'
    chosen_candidate_index: Optional[int] = None
    reviewer_note: Optional[str] = None
    art_path: Optional[str] = None       # NEW -- local image file used as this entry's report poster
    art_overridden: bool = False         # NEW -- True once a human has explicitly set/cleared art_path;
                                          # future auto-resolution (chunk 8) must never overwrite it
```

Add both fields after `reviewer_note`, at the end -- `_entry_from_dict()`
(`QueueEntry(**d)`) already tolerates missing keys via the dataclass
defaults, so existing on-disk entries written before this chunk load
fine with `art_path=None, art_overridden=False`. No migration needed.

Update `docs/schema/review_queue.schema.json` to document them (the
schema has `"additionalProperties": true` so this isn't required for
anything to keep working, but the schema is a published reference and
should stay accurate):

```json
"art_path": {
  "oneOf": [{ "type": "string" }, { "type": "null" }],
  "description": "local image file path used as this entry's report poster, if any -- set by a human via the review dialog's Artwork section, or (later) auto-resolved from a library source or TMDB. null if none."
},
"art_overridden": {
  "type": "boolean",
  "description": "true once a human has explicitly set or cleared art_path -- auto-resolution must never silently overwrite it."
}
```
(add both to the `properties` object; `required` stays unchanged --
these are optional.)

### A.4 `pipeline/review.py` -- publish wiring

In `publish_reviewed_queue()`, the existing call:

```python
image_png = session.report([unfiltered, filtered], complete_filter, meta=meta, spec=report_spec,
                           mv_offset=chosen.mv_adjust_db)
```

becomes:

```python
image_png = session.report([unfiltered, filtered], complete_filter, meta=meta, poster_path=entry.art_path,
                           spec=report_spec, mv_offset=chosen.mv_adjust_db)
```

That's the entire pipeline-layer change -- `Session.report()` already
forwards `poster_path` to `render_report()`, which already forwards it
to `compose_with_poster()`. `entry.art_path` is `None` for any entry
that never had artwork set, and `render_report(poster_path=None)`
already renders chart-only, so this is backward compatible for every
existing entry.

### A.5 `review.ui` changes

Wrap the existing detail-pane content in a `QTabWidget` (`detailTabs`)
with two tabs, so the metadata form doesn't have to compete for space
with the candidate list/chart. The accept/skip/reject action buttons
stay **outside** the tab widget (in `actionButtonsLayout`, unchanged
position) since they act on the entry regardless of which tab is open.

```
detailPane (QWidget, unchanged)
  detailLayout (QVBoxLayout, unchanged)
    titleLabel                          -- unchanged, stays above the tabs
    declineReasonLabel                  -- unchanged
    detailTabs (QTabWidget)             -- NEW, wraps everything below
      "Candidates" tab (existing widgets, moved in as-is, unchanged behaviour):
        candidateList
        commentaryTable
        previewChart
      "Metadata" tab (metadataTab, QWidget)  -- NEW
        metadataForm (QFormLayout):
          "Title:"        titleField        (QLineEdit)
          "Alt title:"     altTitleField     (QLineEdit)
          "Sort title:"    sortTitleField    (QLineEdit)
          "Year:"          yearField         (QLineEdit)
          "Audio types:"   audioTypesField   (QLineEdit)   -- comma-separated, e.g. "Atmos, TrueHD 7.1"
          "Edition:"       editionField      (QLineEdit)
          "Season:"        seasonField       (QLineEdit)
          "Note:"          noteField         (QLineEdit)
          "Warning:"       warningField      (QLineEdit)
          "Language:"      languageField     (QLineEdit)
          "Source:"        sourceField       (QLineEdit)
          "Rating:"        ratingField       (QLineEdit)
          "Author:"        authorField       (QLineEdit)
          "AVS post URL:"  avsField          (QLineEdit)
          "Runtime:"       runtimeField      (QLineEdit)
          "Gain override:" gainField         (QLineEdit)   -- blank = use the chosen candidate's mv_adjust_db (today's default)
          "Genres:"        genresLabel       (QLabel, read-only -- set only by Reload from TMDB)
          "TMDB id:"       movieDbIdField    (QLineEdit)
                           reloadTmdbButton  (QPushButton, "Reload from TMDB") -- same row as movieDbIdField
        saveMetadataButton (QPushButton, "Save Metadata")
        metadataStatusLabel (QLabel)         -- transient feedback: "Saved" / a validation error
        --- Artwork ---
        artworkGroupLabel (QLabel, "Artwork")
        artPathField      (QLineEdit, read-only -- shows the current art_path, or empty)
        browseArtButton   (QPushButton, "Browse...")
        artUrlField       (QLineEdit, placeholder "Paste an image URL")
        downloadArtButton (QPushButton, "Download")
        clearArtButton    (QPushButton, "Clear")
        artPreviewLabel   (QLabel -- shows a small QPixmap thumbnail of art_path if it exists and is readable; blank otherwise)
    (candidateList/commentaryTable/previewChart end -- moved into the Candidates tab, not duplicated)
    actionButtonsLayout                  -- unchanged: acceptButton, skipButton, rejectButton
```

Regenerate with `cd src/main/python/ui && ../../../../.venv/bin/pyuic6 review.ui -o review.py` (or run
`./convert.sh` which does the same for every `.ui` in that directory).

### A.6 `model/review.py` changes

**Loading a selected entry** -- extend the existing `__on_row_selected`
to also populate the new tab:

```python
def __on_row_selected(self, *_):
    entry = self.__current_entry()
    ... # existing candidate/decline/chart logic, unchanged
    self.__load_metadata_form(entry)
    self.__load_artwork_section(entry)

def __load_metadata_form(self, entry):
    meta = entry.meta if entry is not None else {}
    self.titleField.setText(meta.get('title', ''))
    self.altTitleField.setText(meta.get('alt_title', ''))
    self.sortTitleField.setText(meta.get('sort_title', ''))
    self.yearField.setText(meta.get('year', ''))
    self.audioTypesField.setText(', '.join(meta.get('audio_types', [])))
    self.editionField.setText(meta.get('edition', ''))
    self.seasonField.setText(meta.get('season', ''))
    self.noteField.setText(meta.get('note', ''))
    self.warningField.setText(meta.get('warning', ''))
    self.languageField.setText(meta.get('language', ''))
    self.sourceField.setText(meta.get('source', ''))
    self.ratingField.setText(meta.get('rating', ''))
    self.authorField.setText(meta.get('author', ''))
    self.avsField.setText(meta.get('avs', ''))
    self.runtimeField.setText(meta.get('runtime', ''))
    self.gainField.setText(meta.get('gain') or '')
    self.movieDbIdField.setText(meta.get('the_movie_db', ''))
    genres = meta.get('genres') or []
    self.genresLabel.setText(', '.join(g.get('name', '') for g in genres))
    self.metadataStatusLabel.setText('')
    editable = entry is not None and entry.status in ('pending', 'skipped')
    self.metadataTab.setEnabled(editable)  # whole tab greyed out once accepted/published
```

`__load_artwork_section`:

```python
def __load_artwork_section(self, entry):
    art_path = entry.art_path if entry is not None else None
    self.artPathField.setText(art_path or '')
    if art_path and os.path.isfile(art_path):
        self.artPreviewLabel.setPixmap(QPixmap(art_path).scaledToWidth(160, Qt.TransformationMode.SmoothTransformation))
    else:
        self.artPreviewLabel.clear()
```

(`from qtpy.QtGui import QPixmap` added to the existing import block.)

**Saving metadata** -- a new method, wired to `saveMetadataButton.clicked`:

```python
def __save_metadata(self):
    entry = self.__current_entry()
    if entry is None:
        return
    fields = {
        'title': self.titleField.text().strip(),
        'year': self.yearField.text().strip(),
        'audio_types': [t.strip() for t in self.audioTypesField.text().split(',') if t.strip()],
    }
    optional = {
        'alt_title': self.altTitleField.text().strip(),
        'sort_title': self.sortTitleField.text().strip(),
        'edition': self.editionField.text().strip(),
        'season': self.seasonField.text().strip(),
        'note': self.noteField.text().strip(),
        'warning': self.warningField.text().strip(),
        'language': self.languageField.text().strip(),
        'source': self.sourceField.text().strip(),
        'rating': self.ratingField.text().strip(),
        'author': self.authorField.text().strip(),
        'avs': self.avsField.text().strip(),
        'runtime': self.runtimeField.text().strip(),
        'gain': self.gainField.text().strip(),
        'the_movie_db': self.movieDbIdField.text().strip(),
    }
    fields.update({k: v for k, v in optional.items() if v})  # blank = leave unset, don't stomp a BeqMetadata default
    merged = {**entry.meta, **fields}
    update_entry(self.__queue_dir, entry.id, meta=merged)
    self.metadataStatusLabel.setText('Saved')
    self.__reload_queue()
```

Note the blank-means-absent rule: an empty `languageField`/`sourceField`
must not write `''` into `entry.meta` (that would override
`BeqMetadata`'s `'English'`/`'Disc'` defaults at publish time with an
empty string) -- only non-empty optional fields are merged in.
`title`/`year`/`audio_types` are always written (including empty/`[]`)
since they're required by `pipeline.metadata.validate()` and a reviewer
clearing them back out is a legitimate (if unusual) action.

**Reload from TMDB** -- wired to `reloadTmdbButton.clicked`, mirrors
`CreateAVSPostDialog.load_tmdb_info()` exactly:

```python
def __reload_tmdb(self):
    from pipeline.metadata import tmdb_details_by_id, tmdb_lookup
    from model.preferences import TMDB_API_KEY
    api_key = self.__preferences.get(TMDB_API_KEY)
    tmdb_id = self.movieDbIdField.text().strip()
    try:
        if tmdb_id:
            meta = tmdb_details_by_id(tmdb_id, api_key, kind='movie')
        else:
            meta = tmdb_lookup(self.titleField.text().strip(), self.yearField.text().strip(), api_key, kind='movie')
        self.titleField.setText(meta.title)
        self.altTitleField.setText(meta.alt_title)
        self.yearField.setText(meta.year)
        self.ratingField.setText(meta.rating)
        self.runtimeField.setText(meta.runtime)
        self.movieDbIdField.setText(meta.the_movie_db)
        self.genresLabel.setText(', '.join(g.get('name', '') for g in meta.genres))
        self.__pending_tmdb_extras = {'poster': meta.poster, 'overview': meta.overview,
                                      'genres': meta.genres, 'collection': meta.collection}
    except requests.HTTPError as e:
        QMessageBox.critical(self, 'TMDB lookup failed', str(e))
```

`meta.poster`/`meta.overview`/`meta.genres`/`meta.collection` aren't
backed by a visible text field (poster goes through the Artwork
section instead, per A.2's scope trim on genres/collection) --
stash them on `self.__pending_tmdb_extras` and fold them into the
`optional` dict in `__save_metadata()` (`optional['overview'] =
self.__pending_tmdb_extras.get('overview', '')`, plus `genres`/
`collection` merged in directly, not through the blank-means-absent
filter since they're not strings) so a TMDB reload's fuller result
still reaches `entry.meta` on the next Save. Reset
`self.__pending_tmdb_extras = {}` in `__load_metadata_form()` so
switching entries doesn't leak one title's TMDB extras onto another's
save.

**Artwork actions**, immediate-write (no separate Save step, matching
accept/skip/reject's immediacy):

```python
def __browse_art(self):
    entry = self.__current_entry()
    if entry is None:
        return
    path, _ = QFileDialog.getOpenFileName(self, 'Choose artwork', filter='Images (*.png *.jpg *.jpeg)')
    if path:
        update_entry(self.__queue_dir, entry.id, art_path=path, art_overridden=True)
        self.__reload_queue()

def __download_art(self):
    entry = self.__current_entry()
    url = self.artUrlField.text().strip()
    if entry is None or not url:
        return
    try:
        resp = requests.get(url)
        resp.raise_for_status()
    except Exception as e:
        QMessageBox.critical(self, 'Download failed', str(e))
        return
    cache_dir = os.path.join(self.__queue_dir, '_art_cache')
    os.makedirs(cache_dir, exist_ok=True)
    ext = os.path.splitext(url)[1] or '.jpg'
    dest = os.path.join(cache_dir, f"{entry.id}{ext}")
    with open(dest, 'wb') as f:
        f.write(resp.content)
    update_entry(self.__queue_dir, entry.id, art_path=dest, art_overridden=True)
    self.artUrlField.clear()
    self.__reload_queue()

def __clear_art(self):
    entry = self.__current_entry()
    if entry is None:
        return
    update_entry(self.__queue_dir, entry.id, art_path=None, art_overridden=False)
    self.__reload_queue()
```

`_art_cache/` under the queue directory (not a tempfile, unlike
`SaveReportDialog.__download_image()`) -- a downloaded URL has no
original local file to reference, and `art_path` must remain valid
across app restarts and future `sync_library()` runs, so it needs a
durable location. Naming by `entry.id` means a second download for the
same entry simply overwrites the first (fine -- there's only ever one
current artwork per entry).

`requests` needs a module-level `import requests` added to
`model/review.py` (not currently imported there).

**Constructor wiring** -- add to `ReviewQueueDialog.__init__`, alongside
the existing button connections:

```python
self.saveMetadataButton.clicked.connect(self.__save_metadata)
self.reloadTmdbButton.clicked.connect(self.__reload_tmdb)
self.browseArtButton.clicked.connect(self.__browse_art)
self.downloadArtButton.clicked.connect(self.__download_art)
self.clearArtButton.clicked.connect(self.__clear_art)
self.__pending_tmdb_extras = {}
```

### A.7 Tests to add

`src/test/python/gui/test_review_dialog.py` (extend the existing
`_write_entry` helper to accept `meta=None`/`art_path=None` overrides
rather than hardcoding `meta={'title': entry_id}`, then add):

- `test_selecting_a_row_populates_the_metadata_form` -- write an entry
  with a full `meta` dict, select it, assert `dialog.titleField.text()`
  etc. match.
- `test_save_metadata_persists_edited_fields` -- select an entry, set
  `dialog.editionField.setText('Extended Cut')`, call
  `dialog._ReviewQueueDialog__save_metadata()`, assert
  `read_entry(queue_dir, entry_id).meta['edition'] == 'Extended Cut'`.
- `test_save_metadata_does_not_overwrite_defaults_with_blank_fields` --
  leave `languageField`/`sourceField` blank, save, assert `'language'
  not in read_entry(...).meta` (or, equivalently, that
  `BeqMetadata(**read_entry(...).meta).language == 'English'`).
- `test_metadata_tab_is_read_only_once_accepted` -- write an entry with
  `status='accepted'`, select it, assert
  `dialog.metadataTab.isEnabled() is False`.
- `test_browse_art_sets_art_path_and_marks_overridden` -- monkeypatch
  `QFileDialog.getOpenFileName` to return a fixed path (standard
  pytest-qt pattern for file dialogs -- check how other dialogs in this
  suite already do this, e.g. `test_batch_*`/`test_extract_*`, for the
  exact monkeypatch target), call `dialog._ReviewQueueDialog__browse_art()`,
  assert `read_entry(...).art_path == path` and `art_overridden is True`.
- `test_clear_art_resets_override` -- set `art_path` via `update_entry`
  directly, call `__clear_art()`, assert `art_path is None` and
  `art_overridden is False`.
- `test_reload_tmdb_by_id_populates_form_without_saving` -- monkeypatch
  `pipeline.metadata.tmdb_details_by_id` (imported inside the method, so
  patch `model.review.tmdb_details_by_id` after the `from ... import`
  runs, or patch at the `pipeline.metadata` source and call through --
  confirm the exact monkeypatch target once the import style in A.6 is
  finalised) to return a known `BeqMetadata`, call
  `dialog._ReviewQueueDialog__reload_tmdb()`, assert the form fields
  updated **and** `read_entry(...).meta` is unchanged (reload doesn't
  auto-save).

`src/test/python/test_pipeline_review.py` (pure pipeline layer, no Qt):

- `test_queue_entry_round_trips_art_path_and_overridden` -- write an
  entry with `art_path='/tmp/x.jpg', art_overridden=True`, read it back,
  assert both fields survive.
- `test_reading_a_pre_existing_entry_without_art_fields_defaults_them`
  -- hand-write a JSON file *without* `art_path`/`art_overridden` keys
  (simulating an entry written before this chunk), read it via
  `read_entry()`, assert `art_path is None` and `art_overridden is False`
  (backward compatibility).
- Extend `test_publish_reviewed_queue_with_image` (or add a sibling
  test) -- set `art_path` on the entry (via `update_entry`) to a real
  small JPEG written with PIL (same pattern
  `test_pipeline_publish_report.py` already uses:
  `Image.new('RGB', (300, 450), color=(10, 20, 30)).save(poster_path,
  format='JPEG')`), publish with `images_repo` given, fetch the pushed
  PNG, and assert its height is greater than a chart-only baseline (or
  decode it and check the top-left pixel matches the poster's fill
  colour) -- proof `poster_path` actually reached `render_report()`.

### A.8 Manual verification

Since this touches a real dialog: run the app
(`PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen` is for tests
only -- run normally for manual verification), open **Tools -> Review
Batch Designs...**, point it at a queue directory with at least one
pending entry (or create one via `model/batch.py`'s Batch Extract &
Design dialog first), and confirm: the Metadata tab shows blank fields
for an entry with empty `meta`, typing a title/year and clicking Save
persists across a Refresh, Reload from TMDB (with a real or test TMDB
key) populates the form without saving, Browse/Download artwork shows
a thumbnail and persists `art_path`, and the tab greys out once the
entry is Accepted.

### A.9 Acceptance checklist

- [x] `QueueEntry.art_path`/`art_overridden` added, schema doc updated.
- [x] `publish_reviewed_queue()` passes `poster_path=entry.art_path`.
- [x] `review.ui` has the new `detailTabs`/`metadataTab` structure,
      recompiled to `review.py`.
- [x] `ReviewQueueDialog` wires save/reload/browse/download/clear and
      locks the tab on `accepted`/`published`.
- [x] All new tests pass; full suite still green
      (`PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen uv run pytest src/test/python -q`).
- [ ] Manual verification (A.8) done in the real app. (not verifiable from code/tests -- unconfirmed)
