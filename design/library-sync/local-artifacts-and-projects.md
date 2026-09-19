# Library sync plan -- local artifacts and `.beq` projects

> Part of the library sync plan -- **start at the index**: [`../library-sync-pipeline-plan.md`](../library-sync-pipeline-plan.md).
> Contains §3.3, §3.3.1. Section numbers are global across the plan; the index maps every `§` to its file.
> Status of what this file describes: **built** (see `status-and-open-items.md` §10 for the exceptions)

### 3.3 Output 1 -- local artifacts: extracted audio, designer output, and `.beq` project files

The user's clarification (2026-09-18): the extracted wav(s) and the
designer's raw filter output aren't the end of output 1 -- a beqdesigner
project file (`.beq`) needs producing too, so a human can open the
result directly in the interactive app. Two projects per title, not
one: **a mono project** (the signal design actually ran against) and,
whenever the kept extraction is multichannel, **a multichannel
project** with the same designed filter linked across every channel
and the LFE channel correctly identified as LFE -- this was not covered
by anything written so far in this plan and needed adding.

**This is independent of the library-source work**, same shape as
Chunk 1 (Appendix A) -- `model/batch.py`'s *existing* manual
batch-design flow already produces exactly the mono-wav-plus-optional-
multichannel-wav pair per candidate (pipeline/README.md's "Batch design
+ review" section: "a candidate whose kept file is multichannel gets a
second, mono-only extraction made just for the design step... That same
multichannel kept file is also decomposed... and sent alongside as
DesignRequest.channels"), so this belongs in `pipeline.review`
(Qt-free, already shipped) rather than being new library-specific code
-- both the existing manual flow and this plan's new library-driven
flow benefit equally, exactly as Chunk 1's metadata/artwork editor
does.

**What already exists, reused as-is:**
- The `.beq` file format itself -- gzip'd JSON, `app.py.exportProject()`/
  `importProject()`, built from `model.codec.signaldata_to_json()` per
  signal (no raw audio embedded -- only the analysed avg/peak/median
  curves, the filter, and master/slave names, so the project stays
  self-contained even if the source wav later moves/is deleted).
- Filter linking -- `SingleChannelSignalData.enslave(other)` (pure
  Python, no Qt): appends to `.slaves`, sets `other.master = self`, and
  immediately applies `self`'s current filter to `other`
  (`signal.on_filter_change`). `model.codec.signaldata_to_json()`
  already round-trips `master_name`/`slave_names`, and
  `signalmodel_from_json()` already reconstructs the links on load --
  nothing new needed on the serialisation side.
- LFE identification -- `model.ffmpeg.get_channel_name(prefix, idx,
  count, layout)` already produces names like `"<prefix>_LFE"`, the
  exact suffix `model/signal.py`'s bass-management code already keys
  off (`signal.name.endswith('_LFE')`). `Session.load_channels()`
  already uses this same function for the same labelling today (for
  `DesignRequest.channels`), just returning raw arrays instead of full
  signal objects -- see below for why this plan adds a second,
  parallel loader rather than reusing that one.

**New, small addition to `pipeline/orchestrate.py::Session`** --
`load_channels()` stays as-is (`DesignRequest.channels`, dict-of-arrays,
a different consumer/shape, design/designer-interface.md §2). A second
method returns full signal objects instead:

```python
def load_channel_signals(self, path: str, name: Optional[str] = None,
                         channel_layout_name: str = 'unknown',
                         decimate: bool = True) -> List[SingleChannelSignalData]:
    '''
    Loads every channel of a (possibly multichannel) wav as its own SingleChannelSignalData, named
    "<name>_<channel-label>" (model.ffmpeg.get_channel_name -- same labelling load_channels() uses) --
    ready for set_filters()/enslave(), unlike load_channels()'s raw decomposed arrays. A single-element
    list if path is actually mono.
    '''
```

Implementation calls `model.signal.AutoWavLoader.prepare()`/`get_signal()`
directly, once per channel, rather than going through `auto_load()`
(what `Session.load()` uses for the mono case) -- `auto_load()` wraps a
multichannel result in a `BassManagedSignalData`, which exists for
bass-management headroom calculations this method has no use for, and
which would need `BASS_MANAGEMENT_LPF_FS`/`_POSITION` out of the
`_ConfigPreferences` stand-in (`pipeline/orchestrate.py`, not real
QSettings) for a purpose it's not actually being used for here.
Calling `prepare()`/`get_signal()` directly -- the same two calls
`auto_load()` itself makes internally, per channel, in a loop -- avoids
that dependency entirely rather than working around it:

```python
def load_channel_signals(self, path, name=None, channel_layout_name='unknown', decimate=True):
    from model.ffmpeg import get_channel_name
    default_name = name or os.path.splitext(os.path.basename(path))[0]
    loader = AutoWavLoader(self.__preferences)
    loader.load(path)
    channel_count = loader.info.channels
    signals = []
    for idx in range(channel_count):
        channel_name = get_channel_name(default_name, idx, channel_count, channel_layout_name=channel_layout_name)
        loader.prepare(channel=idx + 1, name=channel_name, channel_count=channel_count, decimate=decimate)
        signals.append(loader.get_signal(idx + 1, channel_name))
    return signals
```

(`get_channel_name(text, idx, count, layout)` returns `text` unchanged
when `count == 1`, so a mono `path` naturally comes back as a single
unsuffixed-name signal -- no special-casing needed.) This removes what
was previously flagged as an implementation risk by construction, not
by verifying it away.

**New module, `pipeline/publish/project.py`:**

```python
def write_project(path: str, signals: Sequence[SingleChannelSignalData]) -> None:
    ''' Same gzip+JSON shape as app.py's exportProject() -- model.codec.signaldata_to_json() per signal,
    no BassManagedSignalData wrapper (this is a designed-filter project, not a bass-management one). '''

def write_title_projects(session: Session, mono_wav_path: str, filters: CompleteFilter,
                         multichannel_wav_path: Optional[str], channel_layout_name: str,
                         mono_out_path: str, multichannel_out_path: Optional[str]) -> None:
    # NB: as built this is `write_title_projects_if_safe()` (hash-gated, returns a per-target
    # written/skipped dict) plus `write_mono_project()`/`write_multichannel_project()` -- see Appendix B.3.
    '''
    Writes the mono project unconditionally (session.load(mono_wav_path), set_filters(), write_project()).
    If multichannel_wav_path is given, also loads every channel via load_channel_signals(),
    applies `filters` to the first channel (the master) and enslave()s every other channel -- including the
    LFE one, which needs no special handling here beyond already being named correctly -- to it, then writes
    that project too.
    '''
```

**On-disk convention** (no new `QueueEntry` fields needed for this --
everything is derivable from `work_dir` + `item.id`, keeping this
additive-free):

```
<work_dir>/<item.id>/
    mono.wav                    -- Session.extract(mono_mix=True), used for design + the mono project
    multichannel.wav            -- the kept extraction, only present when one was requested and the
                                    source is actually multichannel (mirrors model/batch.py's existing
                                    "Mix to Mono?" toggle -- see §4.1's extract-cache note)
    manifest.json                -- extract cache's fingerprint record (§4.1)
    <item.id>.mono.beq           -- always written once a candidate is designed
    <item.id>.multichannel.beq   -- only written when multichannel.wav exists
```

**When produced, and idempotency**: at design time
(`design_if_needed()`, §4.2), from the top-pick candidate
(`QueueEntry.candidates[0]`) -- cheap to redo (no audio decode, just
filter application + JSON), so it rides the same design-cache
fingerprint with no separate cache of its own: whenever
`design_if_needed()` actually (re)designs, it also (re)writes both
project files; when it skips (fingerprint match), the existing project
files are left alone since nothing about them would differ.

**Regenerating for a reviewer's actual pick**: the project files above
are built from the designer's top pick, but a reviewer may `accept` a
different (`chosen_candidate_index != 0`) candidate -- at that point
the on-disk projects would mismatch the human's actual decision unless
regenerated. See §3.3.1 immediately below for exactly when that
regeneration is (and, critically, is *not*) allowed to happen.

### 3.3.1 The `.beq` project is what gets published, not the raw designer output

The user's key correction (2026-09-18): once output 1's mono project
file exists, a human can open it in the full interactive app and change
the filter directly -- add a manual biquad, nudge a Q, whatever the
auto-designer didn't quite get right. **That edited filter is what must
end up in beqcatalogue**, not `QueueEntry.candidates[chosen_candidate_index].filters`
(the designer's untouched raw output) as originally drafted above. This
inverts a piece of the data flow this plan had assumed throughout: the
project file stops being a one-way, disposable *derivative* of the
queue entry and becomes the actual editable source of truth for the
published filter, with the queue entry demoted to "the designer's
original proposal, plus review status" -- still useful (audit trail,
what the designer actually thought), but no longer what
`publish_reviewed_queue()` reads to build the XML.

This immediately raises the follow-up problem the user identified next:
**how does the pipeline tell "still the designer's raw output" apart
from "a human opened this and changed it"?** -- since the answer
controls two different things that must not be conflated: (a) whether
`publish_reviewed_queue()` should trust the project file's filter over
the queue entry's, and (b) whether design-time/accept-time regeneration
(the paragraph above) is safe to run at all, or would silently clobber
a human's edit.

**Fix: a content hash, stored inside the project file itself, that only
the pipeline ever writes.** `write_project()`/`write_title_projects()`
compute `sha256(json.dumps(complete_filter.to_json(), sort_keys=True))`
for whichever filter they just applied, and stash it as an extra key --
`pipeline_filter_hash` -- on the *master* signal's dict, alongside
whatever `model.codec.signaldata_to_json()` already produces (an
additive key; `signaldata_from_json()`/`signalmodel_from_json()` only
ever reads the keys they know about, so this is silently ignored by the
interactive app's own load path -- verified against `model/codec.py`,
not assumed). The tell is what happens on the way back out: if a human
opens the project in the app, edits the filter, and re-exports via
`app.py`'s `exportProject()`, that path calls the *generic*
`signaldata_to_json()`, which has no concept of `pipeline_filter_hash`
and will not re-emit it -- so a human-resaved project always comes back
with either no hash at all, or (if a first save happened to preserve an
unrelated stale one) a hash that no longer matches the filter actually
stored. Either way, recomputing the hash from the project's *current*
`filter_presets[active_filter_preset]` and comparing:

```python
def read_project_filter(path: str) -> Tuple[CompleteFilter, bool]:
    ''' :return: (the master signal's current filter, True if pipeline_filter_hash matches a fresh hash of
    it -- i.e. this project is still exactly what the pipeline last wrote, False if a human has changed
    it since (or the file predates this mechanism and never had a hash at all). '''
```

- **`publish_reviewed_queue()`** reads the mono project's filter via
  `read_project_filter()` and publishes *that* filter -- regardless of
  whether it matches (a matching hash just means "the human's edit is
  the designer's own output," the common case, so this is a strict
  superset of today's behaviour, not a divergent path). `entry.candidates`/
  `apply_reviewed_entry()` are unchanged and still readable (still the
  audit trail of what the designer originally proposed), but stop being
  what `publish_reviewed_queue()` actually reads the filter from. If no
  project file exists yet (`work_dir` wasn't threaded through, or an
  older entry predates chunk 2), it falls back to today's
  `apply_reviewed_entry()` path -- backward compatible.
- **Design-time/accept-time regeneration** (the paragraph above this
  subsection) is now gated on the hash: before overwriting an *existing*
  project file, check whether its current filter's hash still matches
  its stored `pipeline_filter_hash`. A match means it's safe (still
  pure pipeline output, nothing to lose) -- overwrite as planned. A
  mismatch means a human edited it -- **do not overwrite**; the human's
  edit now outranks the designer's top pick and even a reviewer's
  candidate switch. **Reporting (as built):**
  `design_and_queue(on_projects=...)` hands `write_title_projects_if_safe()`'s
  result to its caller; `design_if_needed()` keeps it as
  `DesignCacheResult.projects`, and `run_library()` lists the item in
  `LibraryRunReport.project_edit_preserved` when any project was skipped
  (`project_edit_preserved` on the result). The Library Sync status line
  says how many were kept. At publish the equivalent is the result's
  `edited_project` key (below), since a skipped regeneration there is the
  normal case, not news. The both-edited *conflict* case is reported as
  before (`{'id', 'error': 'project_conflict'}`).

**Both projects are legitimate places to design/edit a filter; only
mono is ever what gets *published*.** The user's correction
(2026-09-18): mono vs. multichannel is a distinction about what
*analysis* a filter was designed against -- a human (or, per the
already-shipped designer contract's per-channel `channels`/
`channel_scope` diagnostic input, a designer itself) may reasonably do
real design/refinement work in the multichannel project rather than the
mono one, e.g. checking a candidate against the LFE channel
specifically. It is not merely a personal-use copy of whatever mono
produced, so publish cannot simply always prefer mono's file and ignore
multichannel edits -- it needs to notice which one actually carries the
human's edit, whichever that is. What *is* fixed, regardless of which
signal a filter was analysed against, is the shape of the thing that
gets published: a single `CompleteFilter` written into beqcatalogue's
XML exactly as today -- this pipeline has never produced a genuinely
per-channel filter, so "publishing a multichannel-derived edit" needs
no conversion, just reading the right file.

`read_project_filter()` (above) therefore checks **both** project
files' hashes, not just mono's:
- Neither shows a human edit -> both still equal the last
  pipeline-designed filter; publish either (they're identical).
- Exactly one shows a human edit -> that project's filter (the master
  channel's, for multichannel) is authoritative; publish it, and write
  it back into the *other* project too (same mechanism as the
  accept-time regeneration above) so both stay consistent with what
  was actually published rather than one silently going stale.
  **As built:** `resolve_published_projects()` returns a `PublishedFilter`
  (the filter plus `edited_side`: `None`/`'mono'`/`'multichannel'`/`'both'`;
  `resolve_published_filter()` remains as a thin wrapper), and
  `align_projects()` rewrites the *pure* sibling with that filter -- it can
  never overwrite an edit, because a sibling of an edited side is pure by
  construction. `publish_reviewed_queue()` adds `edited_project` (the side)
  and, when a sibling was rewritten, `projects_aligned` to that entry's
  result; both keys are absent when nothing was edited. The rewritten
  sibling is pipeline-written again (hash matches), so a later regeneration
  from the candidate would overwrite it -- harmless, since an entry is
  published once and the edited side wins on the next resolve regardless.
- **Both show edits, and they differ** -- a genuine conflict this plan
  cannot silently resolve by picking one. Publish refuses and surfaces
  it (the same `project_edit_preserved`-style reporting as the
  clobber-prevention case above), rather than guessing which of two
  independent human edits should win. Exactly how a human resolves that
  conflict (which project to treat as authoritative, or hand-merge) is
  left to UI/CLI-error-message design in the relevant chunk, not
  decided here.
