# Library sync plan -- archived handoff spec

> Part of the library sync plan -- **start at the index**: [`../../library-sync-pipeline-plan.md`](../../library-sync-pipeline-plan.md).
> Contains Appendix B (chunk 2). Section numbers are global across the plan; the index maps every `§` to its file.
> Status of what this file describes: **built and shipped** -- kept as the record of the original spec; read only if you need the detail

## Appendix B -- Chunk 2 handoff spec: `.beq` project files as the published source

Self-contained enough to implement without reading the rest of this
document, though §3.3/§3.3.1 above have the full rationale and this
spec assumes you've read them once. Entirely within `pipeline/`
(Qt-free) plus `pipeline/review.py`'s two public entry points -- no
Qt/GUI work in this chunk. Independent of Chunk 1 (Appendix A) and of
every JRiver/library-source chunk.

### B.1 Why

Output 1 (§3.3) is a mono `.beq` project file per title, and
(optionally) a linked multichannel one. §3.3.1 established the
non-obvious part: once that project exists, a human editing it in the
interactive app is what must reach beqcatalogue, not
`QueueEntry.candidates[...]`'s frozen designer output -- and the
pipeline needs a way to tell "still exactly what the designer produced"
apart from "a human changed this" so it never silently clobbers an
edit, and so `publish_reviewed_queue()` knows which of (up to two)
project files to actually read from. This chunk builds all of that:
the project-writing itself, the edit-detection hash, the two-sided
resolution (mono vs. multichannel) including conflict detection, and
wiring both into the two existing `pipeline.review` entry points that
need it.

### B.2 Scope

**In scope:**
- `Session.load_channel_signals()` (`pipeline/orchestrate.py`) -- exact
  code already given in §3.3 above.
- New module `pipeline/publish/project.py` -- all functions in B.3.
- `pipeline.review.design_and_queue()` gains three new optional
  parameters (`multichannel_wav_path`, `channel_layout_name`,
  `project_dir`) and writes output 1's project files when
  `project_dir` is given and the outcome was `Applied`.
- `pipeline.review.publish_reviewed_queue()` gains one new optional
  parameter (`work_dir`) and, when given, regenerates (hash-gated) and
  then reads the published filter from the project file(s) rather than
  unconditionally from `apply_reviewed_entry()`.
- The extract-cache manifest convention (§4.1, not yet built in this
  chunk) gets one more field reserved for it: `channel_layout_name`,
  read back by `publish_reviewed_queue()`'s regeneration step. This
  chunk doesn't write the manifest (that's chunk 5) -- it only needs to
  agree on the field name now so chunk 5 doesn't have to revisit this
  chunk later. See B.5's note on this.

**Explicitly out of scope:**
- Anything that decides *how* a conflict (§3.3.1's "both edited and
  disagree" case) is presented to a human -- `publish_reviewed_queue()`
  raising/reporting it is in scope; a GUI dialog or CLI message
  explaining it is not (deferred to chunks 9/10).
- The extract cache itself (chunk 5) and `design_if_needed()`'s
  fingerprint-skip logic (chunk 6) -- this chunk only defines the
  manifest field name they'll need to also write.
- Detecting an edit to an individually-`free()`'d slave channel (§9's
  open question) -- out of scope, not solved by this chunk's hash
  mechanism, which only ever inspects the master/mono signal.

### B.3 New module: `pipeline/publish/project.py`

```python
def _filter_hash(filters: CompleteFilter) -> str:
    ''' sha256 of the filter's canonical JSON -- what pipeline_filter_hash stores/compares against. '''
    import hashlib, json
    return hashlib.sha256(json.dumps(filters.to_json(), sort_keys=True).encode('utf-8')).hexdigest()


def write_project(path: str, signals: Sequence[SingleChannelSignalData], filter_hash: Optional[str] = None) -> None:
    '''
    Writes a .beq project -- the same gzip+JSON shape app.py's exportProject()/importProject() use
    (model.codec.signaldata_to_json() per signal, no BassManagedSignalData wrapper). If filter_hash is
    given, stamps it as an extra 'pipeline_filter_hash' key on signals[0]'s dict -- an additive key
    signaldata_from_json()/signalmodel_from_json() simply don't look for, so it round-trips fine through
    the interactive app's own load path but is never re-emitted by a human's re-export (app.py's
    exportProject() only ever calls the generic signaldata_to_json(), which has no concept of this key) --
    that asymmetry is the edit-detection mechanism (see read_project_filter()).
    '''
    import gzip, json
    from model.codec import signaldata_to_json
    output = [signaldata_to_json(s) for s in signals]
    if filter_hash is not None and output:
        output[0]['pipeline_filter_hash'] = filter_hash
    with gzip.open(path, 'wb') as f:
        f.write(json.dumps(output).encode('utf-8'))


def write_mono_project(session: Session, mono_wav_path: str, filters: CompleteFilter, out_path: str) -> None:
    sig = session.load(mono_wav_path)
    session.set_filters(sig, filters)
    write_project(out_path, [sig], filter_hash=_filter_hash(filters))


def write_multichannel_project(session: Session, multichannel_wav_path: str, filters: CompleteFilter,
                               channel_layout_name: str, out_path: str) -> None:
    ''' Loads every channel (Session.load_channel_signals()), applies `filters` to the first (the master),
    and enslave()s every other channel -- including LFE, which needs no special handling beyond already
    being named correctly by load_channel_signals() -- to it. '''
    channels = session.load_channel_signals(multichannel_wav_path, channel_layout_name=channel_layout_name)
    master = channels[0]
    session.set_filters(master, filters)
    for slave in channels[1:]:
        master.enslave(slave)
    write_project(out_path, channels, filter_hash=_filter_hash(filters))


def _is_safe_to_overwrite(path: Optional[str]) -> bool:
    ''' True if path doesn't exist yet, or exists and is still pipeline-pure (no human edit to lose). '''
    if path is None or not os.path.isfile(path):
        return True
    _, is_pure = read_project_filter(path)
    return is_pure


def write_title_projects_if_safe(session: Session, mono_wav_path: str, filters: CompleteFilter, mono_out_path: str,
                                 multichannel_wav_path: Optional[str] = None, channel_layout_name: str = 'unknown',
                                 multichannel_out_path: Optional[str] = None) -> dict:
    '''
    The one entry point both design_and_queue() and publish_reviewed_queue() call. Writes each project
    only if it's safe to (§3.3.1's hash gate) -- an existing human-edited file is left untouched, the other
    one (if any) still gets written/updated normally. Both targets are independent; one being unsafe never
    blocks the other.
    :return: {'mono': True|False, 'multichannel': True|False|None} -- True=written, False=skipped (existing
        file was human-edited), None=not applicable (no multichannel_wav_path given).
    '''
    result = {'mono': False, 'multichannel': None}
    if _is_safe_to_overwrite(mono_out_path):
        write_mono_project(session, mono_wav_path, filters, mono_out_path)
        result['mono'] = True
    if multichannel_wav_path is not None:
        if _is_safe_to_overwrite(multichannel_out_path):
            write_multichannel_project(session, multichannel_wav_path, filters, channel_layout_name, multichannel_out_path)
            result['multichannel'] = True
        else:
            result['multichannel'] = False
    return result


def read_project_filter(path: str) -> Tuple[CompleteFilter, bool]:
    '''
    :return: (the master/mono signal's current filter, True if pipeline_filter_hash matches a fresh hash
        of it -- i.e. still exactly what the pipeline last wrote -- False if a human has changed it since,
        or the file predates this mechanism and never had a hash).
    '''
    import gzip, json
    from model.codec import filter_from_json
    with gzip.open(path, 'rb') as f:
        data = json.loads(f.read().decode('utf-8'))
    master = data[0]
    filt = filter_from_json(master['filter_presets'][master['active_filter_preset']])
    stored = master.get('pipeline_filter_hash')
    return filt, stored is not None and stored == _filter_hash(filt)


class ProjectFilterConflict(Exception):
    ''' Raised by resolve_published_filter() when both the mono and multichannel projects were
    independently edited and now disagree -- this plan deliberately does not guess which one wins. '''
    def __init__(self, mono_filter: CompleteFilter, multichannel_filter: CompleteFilter):
        super().__init__('mono and multichannel projects have independently edited, conflicting filters')
        self.mono_filter = mono_filter
        self.multichannel_filter = multichannel_filter


def resolve_published_filter(mono_path: str, multichannel_path: Optional[str] = None) -> Tuple[CompleteFilter, bool]:
    '''
    §3.3.1's three-way resolution. :return: (the filter to publish, True if it came from a human edit on
    either side, False if both projects are still pipeline-pure).
    :raises ProjectFilterConflict: if both projects were independently edited and disagree.
    '''
    mono_filter, mono_pure = read_project_filter(mono_path)
    if multichannel_path is None or not os.path.isfile(multichannel_path):
        return mono_filter, not mono_pure
    mc_filter, mc_pure = read_project_filter(multichannel_path)
    if not mono_pure and not mc_pure:
        if mono_filter.to_json() != mc_filter.to_json():
            raise ProjectFilterConflict(mono_filter, mc_filter)
        return mono_filter, True
    if not mc_pure:
        return mc_filter, True
    return mono_filter, not mono_pure
```

### B.4 `Session.load_channel_signals()`

Exact code already given in §3.3 above (the `AutoWavLoader.prepare()`/
`get_signal()`-direct version, not the `auto_load()`/
`BassManagedSignalData` one) -- add it to `pipeline/orchestrate.py::Session`
next to `load_channels()`, with `from model.ffmpeg import get_channel_name`
imported inside the method (matching how `load_channels()` already does
this same lazy import).

### B.5 `pipeline/review.py` changes

`design_and_queue()` -- three new optional parameters, and a call to
`write_title_projects_if_safe()` right after `write_queue_entry()`,
only when `project_dir` is given and the outcome was `Applied` (a
`Declined` outcome has no filter to write, matching "candidates empty
on decline"):

```python
def design_and_queue(session, entry_id, wav_path, designer, queue_dir, meta=None, coverage='complete_programme',
                     bass_management=None, channels=None,
                     multichannel_wav_path: Optional[str] = None, channel_layout_name: str = 'unknown',
                     project_dir: Optional[str] = None) -> QueueEntry:
    ...  # unchanged up to and including write_queue_entry(queue_dir, entry)
    if project_dir is not None and isinstance(outcome, Applied):
        from pipeline.publish.project import write_title_projects_if_safe
        mono_out = os.path.join(project_dir, f"{entry_id}.mono.beq")
        mc_out = os.path.join(project_dir, f"{entry_id}.multichannel.beq") if multichannel_wav_path else None
        write_title_projects_if_safe(session, wav_path, outcome.filters, mono_out,
                                     multichannel_wav_path=multichannel_wav_path,
                                     channel_layout_name=channel_layout_name, multichannel_out_path=mc_out)
    return entry
```

Note `wav_path` is already documented as "an already-extracted (mono)
wav file" -- it *is* the mono project's source, no new parameter needed
for that half.

`publish_reviewed_queue()` -- one new optional parameter. When given,
regenerate (hash-gated, via the same `write_title_projects_if_safe()`)
before reading, then read the filter to actually publish from the
project file(s) instead of `apply_reviewed_entry()`'s raw candidate --
`apply_reviewed_entry()`/`chosen` stay exactly as they are today (still
needed for `meta.gain`'s default), just no longer the source of the
*filter itself* when a project directory is available:

```python
def publish_reviewed_queue(queue_dir, xml_repo, meta_defaults=None, images_repo=None, image_owner=None,
                           image_repo_name=None, xml_dir='', image_dir='', report_spec=ReportSpec(),
                           config=AnalysisConfig(), work_dir: Optional[str] = None) -> List[dict]:
    from pipeline.publish.project import write_title_projects_if_safe, resolve_published_filter, ProjectFilterConflict
    ...
    for entry in read_queue(queue_dir):
        if entry.status != 'accepted':
            continue
        chosen = _chosen_candidate(entry)
        complete_filter = apply_reviewed_entry(entry)  # unchanged -- still drives meta.gain's default below
        meta = BeqMetadata(**{**(meta_defaults or {}), **entry.meta})
        if meta.gain is None:
            meta.gain = f"{chosen.mv_adjust_db:+g}"

        if work_dir is not None:
            project_dir = os.path.join(work_dir, entry.id)
            mono_path = os.path.join(project_dir, f"{entry.id}.mono.beq")
            mc_wav = os.path.join(project_dir, 'multichannel.wav')
            mc_path = os.path.join(project_dir, f"{entry.id}.multichannel.beq") if os.path.isfile(mc_wav) else None
            layout = _read_channel_layout_name(project_dir)  # manifest.json's channel_layout_name, 'unknown' if absent/missing
            write_title_projects_if_safe(session, os.path.join(project_dir, 'mono.wav'), complete_filter, mono_path,
                                         multichannel_wav_path=mc_wav if mc_path else None,
                                         channel_layout_name=layout, multichannel_out_path=mc_path)
            try:
                complete_filter, _ = resolve_published_filter(mono_path, mc_path)
            except ProjectFilterConflict:
                results.append({'id': entry.id, 'error': 'project_conflict'})
                continue

        image_png = None
        # ...unchanged from here: image_png = session.report(..., complete_filter, ...), xml = self.to_beq_xml(complete_filter, meta), etc.
```

`_read_channel_layout_name(project_dir)` -- a small helper, this
chunk's only piece of forward-looking coupling to chunk 5's (not yet
built) extract-cache manifest: reads
`os.path.join(project_dir, 'manifest.json')`'s `channel_layout_name`
key if the file and key exist, else returns `'unknown'`
(`get_channel_name()`'s own fallback already handles an unknown layout
sanely by channel count). Chunk 5 needs to actually write that key when
it builds the manifest -- noted here so it doesn't have to revisit this
chunk later; this chunk's own tests (B.6) can just hand-write a minimal
`manifest.json` with that key to exercise the read side.

An entry that hits `ProjectFilterConflict` is recorded in `results`
with an `'error'` key and **not** marked `'published'` -- a rerun will
retry it, same "only accepted-and-not-yet-published entries do
anything" idempotency the rest of this function already has.

### B.6 Tests to add

All in `src/test/python/` (pipeline layer, no Qt) -- a new
`test_pipeline_publish_project.py` for the new module, plus additions
to the existing `test_pipeline_review.py`:

**`test_pipeline_publish_project.py`:**
- `test_write_and_read_project_round_trips_the_filter` -- write a mono
  project for a known `CompleteFilter`, `read_project_filter()` it
  back, assert the filter matches and `is_pure is True`.
- `test_read_project_filter_detects_a_human_edit` -- write a project,
  then hand-edit the on-disk JSON's `filter_presets`/`active_filter_preset`
  to a different filter (simulating a resave through the interactive
  app, which wouldn't preserve `pipeline_filter_hash`), read it back,
  assert `is_pure is False` and the filter returned is the *edited* one.
- `test_write_multichannel_project_enslaves_every_channel_to_the_master`
  -- write a multichannel project from a real multichannel wav fixture
  (or a small synthetic one, matching how other pipeline tests build
  fixtures), read it back via `model.codec.signalmodel_from_json`
  directly (not `read_project_filter()`, which only looks at the
  master), assert every non-master signal's `master_name` in the raw
  JSON points at the master and the master's `slave_names` lists all of
  them, including the one named with the `_LFE` suffix.
- `test_write_title_projects_if_safe_skips_an_edited_target` -- write a
  project, hand-edit it (as above), call
  `write_title_projects_if_safe()` again with a different filter,
  assert the result dict says `'mono': False` and the file's filter is
  still the hand-edited one, not the new one.
- `test_write_title_projects_if_safe_writes_the_other_target_independently`
  -- edit only the mono project, call with both mono+multichannel
  targets, assert `{'mono': False, 'multichannel': True}` and the
  multichannel file did get the new filter.
- `test_resolve_published_filter_prefers_the_edited_side` -- mono
  edited, multichannel still pure (or absent) -> resolves to the mono
  edit; and the mirror case (multichannel edited, mono pure) ->
  resolves to the multichannel edit.
- `test_resolve_published_filter_raises_on_disagreeing_edits` -- edit
  both projects to *different* filters, assert
  `resolve_published_filter()` raises `ProjectFilterConflict` carrying
  both filters.

**`test_pipeline_review.py` additions:**
- `test_design_and_queue_writes_a_mono_project_when_project_dir_given`
  -- call with `project_dir=str(tmp_path / 'projects')`, assert the
  `<entry_id>.mono.beq` file exists and its filter matches
  `entry.candidates[0].filters`.
- `test_design_and_queue_writes_a_multichannel_project_when_given_one`
  -- same, plus `multichannel_wav_path`, assert the `.multichannel.beq`
  file exists too.
- `test_design_and_queue_does_not_write_projects_without_project_dir`
  -- omitting `project_dir` (today's call shape, unchanged) writes no
  `.beq` files -- backward compatibility.
- `test_publish_reviewed_queue_reads_the_edited_mono_project_when_work_dir_given`
  -- set up an accepted entry with a `work_dir`/project layout, hand-edit
  the mono project's filter after design, call `publish_reviewed_queue(...,
  work_dir=...)`, assert the published XML reflects the *edited* filter,
  not `entry.candidates[0]`'s.
- `test_publish_reviewed_queue_without_work_dir_uses_apply_reviewed_entry_as_before`
  -- omitting `work_dir` (today's call shape) publishes exactly what it
  does today -- backward compatibility, protects every existing
  `test_publish_reviewed_queue_*` test in this file from regressing.
- `test_publish_reviewed_queue_reports_a_project_conflict_without_publishing`
  -- both projects edited to disagree, assert the entry is *not* marked
  `published`, and the result list contains an `'error'` entry for it.

### B.7 Acceptance checklist

- [x] `pipeline/publish/project.py` created with all of B.3's functions.
- [x] `Session.load_channel_signals()` added (B.4).
- [x] `design_and_queue()`/`publish_reviewed_queue()` updated per B.5,
      both fully backward compatible when their new parameters are omitted.
- [x] All new tests (B.6) pass; full suite still green
      (`PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen uv run pytest src/test/python -q`).
- [x] `pipeline_qt_boundary` test (this package's AST scan for stray
      `qtpy` imports) still passes -- `pipeline/publish/project.py` must
      not import qtpy, same rule as every other module in `pipeline/`.
