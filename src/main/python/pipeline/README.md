# Headless BEQ pipeline

`pipeline/` runs BEQ filter creation and publishing without a GUI — extract
audio, get a filter from a pluggable designer, simulate headroom, and publish
beqcatalogue-format XML + a report image to git — so it can be driven by a
script, a batch job, or (via `pipeline/designer/http_binding.py`) a designer
implementation in another process or language entirely.

This package, `designer-interface.md`'s contract, and everything built on
top of it (HTTP binding, batch design, review queue) shipped in full — see
"What shipped" below. The design docs in `design/` now record *why* things
are shaped the way they are and what remains genuinely open, not a build
plan; start here for the current state, and follow the links below into
`design/` for the reasoning behind a specific decision.

## Architecture rules

1. **`pipeline/` never imports `qtpy`.** Enforced by
   `test_pipeline_qt_boundary.py` (an AST scan of the package, plus a
   runtime check that `QApplication.instance()` stays `None` through a full
   pipeline run).
2. **Dependencies point inward.** The GUI (`model/`, `ui/`) calls into
   `pipeline/`; `pipeline/` never calls back into Qt-bound code. The one
   deliberate exception is `Session.load()` reusing
   `model.signal.AutoWavLoader` — that module imports `qtpy` at the top for
   dialogs `AutoWavLoader` never touches, but constructs no `QApplication`
   and needs no display; see `orchestrate.py`'s module docstring.

## Package layout

```
pipeline/
    config.py              # AnalysisConfig -- explicit workflow config, no QSettings
    filters.py              # FilterSpec, create_filter()
    stats.py                # signal_stats() -- peak/rms/crest/headroom
    metadata.py              # BeqMetadata, tmdb_lookup(), validation
    review.py                # QueueEntry, batch_design(), apply/publish a reviewed queue
    orchestrate.py            # Session facade -- extract/load/design/stats/publish
    designer/
        contract.py           # DesignRequest, DesignResponse, BiquadSpec, DesignCandidate
        registry.py           # register_designer()/get_designer(), in-process callable registry
        convert.py             # validates a DesignResponse, converts BiquadSpec -> CompleteFilter
        http_binding.py         # http_designer(url) -- an HTTP-backed DesignerCallable
    publish/
        xml.py                 # to_beq_xml() -- wraps the existing HDXmlParser path
        art.py                  # TMDB poster fetch
        report.py                # headless report renderer (Agg canvas, no Qt)
        git.py                    # write / commit exactly named paths / push, and repo_state() read back from git
        catalogue.py              # where a title lives in the repos, and the digest that says it is out of date
        project.py                 # per-title .beq project files; a hand edit of one is what gets published
    library/                       # library-scale runs -- see "Library sync (CLI)" below
        source.py, registry.py      # LibraryItem / LibrarySource, an in-process registry of sources
        jriver.py, filesystem.py     # the JRiver (MCWS browse node) and plain-filesystem sources
        pathmap.py                    # translate a server's (Windows) paths to local ones
        extract_cache.py, design_cache.py   # idempotent extract and design
        season.py                      # TV: join a season's episodes into one track
        profile.py, ignore.py, union.py   # the catalogue profile, its ignore rules, and merging several sources
        run.py, sync.py, commit.py, revise.py, cli.py   # run_library(), publish/commit/sync_library(), revise_entry(), the command line
        state.py, status.py, index.py, catalogue_scan.py   # discovery: what each title needs next, and the SQLite index of it
        selection.py, stages.py, bulk.py   # doing work for a selection: Selection, run_stages(--through), accept_top_pick()

model/preferences.py          # GUI: durable list of configured HTTP designer endpoints + the review queue
                              #   directory default, both on the Preferences dialog's "Designers" page
model/batch.py                # GUI: Batch Extract & Design dialog -- search/extract many files, with an
                              #   optional per-candidate design_and_queue() step and an embedded Review tab
model/review.py               # GUI: ReviewQueueDialog, embedded as model/batch.py's Review tab and also
                              #   reachable standalone via Tools > Review Batch Designs...
ui/preferences.py, ui/batch.py, ui/review.py   # the corresponding dialogs
```

## Workflow stages

1. **Extract + load** — `Session.extract()`/`.load()` wrap `model.ffmpeg.Executor`
   (`run_sync()`, a public accessor added for this) and
   `model.signal.AutoWavLoader`. Mono downmix with LFE at +10 dB, optionally
   decimated to `AnalysisConfig.target_fs` (1 kHz by default).
2. **Design** — `Session.design(sig, designer='name')` builds a
   `DesignRequest`, invokes a registered `DesignerCallable`
   (`pipeline.designer.registry`), validates the `DesignResponse`
   (`pipeline.designer.convert`), and returns `Applied(...)` or
   `Declined(reason)` — see "The designer contract" below.
3. **Simulate** — `Session.stats(sig)` → `signal_stats()`: peak, RMS, crest,
   headroom, and the fs it was measured at (so a published `beq_gain` is
   never ambiguous about what it measured).
4. **Metadata + XML** — `Session.tmdb(title, year, kind)` → `BeqMetadata`;
   `Session.to_beq_xml(sig, meta)` wraps the existing `HDXmlParser`/
   `flat24hd.xml` path. Validates filter types/budget and fails loudly
   rather than writing an unpublishable entry.
5. **Art + report** — `pipeline/publish/art.py` fetches the TMDB poster;
   `pipeline/publish/report.py` renders the same "pixel perfect" layout the
   interactive `SaveReportDialog` produces, headlessly (an Agg canvas, no
   Qt-embedded figure), at a fixed size/DPI passed in rather than measured
   from a window.
6. **Publish** — `pipeline/publish/git.py` commits the XML to one repo and
   the report image to a second, builds the image's GitHub raw-content URL,
   and sets it as `BeqMetadata.spectrum_url`/`.pva_url` *before* the XML is
   written (the XML needs that URL). Runs as the invoking user's own git
   identity — no credential of its own.

`Session` (`pipeline/orchestrate.py`) composes all of the above; see its
module docstring for `Applied`/`Declined` handling and how a designer's
provenance (confidence, method, residual, alternatives) travels as far as
the report but never into the published XML.

## The designer contract

The full, implementable-without-this-repo contract is
[`../../../../design/designer-interface.md`](../../../../design/designer-interface.md)
(`DesignRequest`/`DesignResponse`/`BiquadSpec`, validation rules, the RBJ
gain convention, worked examples). In short: a designer is a pure function
`DesignRequest -> DesignResponse`, returning one or more ranked filter
candidates or a decline — never an exception for a normal "nothing to
correct" outcome.

`DesignRequest.mono_mix` is the primary/required signal; `.channels` is an
optional per-channel diagnostic (channel_scope, bass-management
reconstruction) `Session.design()` forwards but never derives on its own --
a caller that wants it supplies it explicitly via `Session.load_channels()`
(pure decomposition of a multichannel wav into named arrays, no
mixing/gain-staging -- `mono_mix` itself is still only ever produced by the
ffmpeg downmix path, `Session.extract(mono_mix=True)`, so it stays
bit-consistent with every other mono downmix in the app). Both `model/
batch.py` and `model/extract.py`'s design steps supply `channels` whenever
their kept extraction is multichannel, at no extra ffmpeg cost -- decomposed
straight from the file already on disk.

Two bindings exist:

- **In-process Python callable** — `pipeline.designer.registry.register_designer(name, callable)`.
  The cheapest path; what `Session.design(designer='name')` looks up.
- **HTTP** — `pipeline.designer.http_binding.http_designer(url)` produces a
  `DesignerCallable` that POSTs a `DesignRequest` (arrays as base64 float64,
  per `designer-interface.md §7.1`) and parses a `DesignResponse` back.
  Register it the same way as any in-process designer. Wire format is
  published as JSON Schema at `docs/schema/http_designer_request.schema.json`
  / `http_designer_response.schema.json`. A non-2xx/timeout/malformed body
  raises `HttpDesignerError`; a well-formed-but-invalid `DesignResponse`
  goes through the same validation as any other binding.

In the GUI, **Preferences → Designers** (`model/preferences.py`) maintains
a durable list of `{name, url, headers}` HTTP endpoints, registered under a
`http:`-prefixed name on every app startup. The same page also holds the
`DESIGNER_QUEUE_DIR` default -- the review queue directory `model/batch.py`'s
Run tab and `model/review.py`'s Review tab both remember and default to --
and `DESIGNER_DEFAULT`, the designer name `model/batch.py`'s Run tab
preselects in its designer combo.

## Batch design + review

Design can run unattended over many titles, writing one `QueueEntry` JSON
file per title (`pipeline.review`) to a queue directory — `pending` either
way, with `candidates` populated from an `Applied` outcome (top pick +
alternatives) or empty with a decline reason. Two Qt-free entry points do
the load+design+write step, `pipeline.review.design_and_queue()` (an
already-extracted wav file in hand) and `pipeline.review.batch_design()`
(extracts first, via `Session.extract()`, then calls the former per item) —
either can be driven by a script/cron job with no GUI involved, writing
into the same queue directory a GUI batch run uses (no manifest, no
locking — see `write_queue_entry()`'s docstring). A human then works
through the queue via one dialog, `model/batch.py`'s `BatchExtractDialog`
("Batch Extract & Design"), which has two tabs:

- **Run** — pick a search filter (or add BD-folder candidates), extract
  many files with the same manual per-candidate stream/channel/LFE
  override this dialog has always had, and optionally check "Design
  filters?" to additionally run `design_and_queue()` on each candidate as
  its extraction completes, against a chosen designer (preselected from
  the `DESIGNER_DEFAULT` preference, itself set on the Preferences
  dialog's Designers page) and queue directory. "Mix to Mono?" is
  independent of this -- it only governs the *kept* extraction; design
  always needs a mono signal (`Session.design()`'s `mono_mix`), so a
  candidate whose kept file is multichannel gets a second, mono-only
  extraction made just for the design step rather than either forcing
  mono onto the kept file or blocking design on it (we commonly want both
  a multichannel file to keep and a mono one to design). That same
  multichannel kept file is also decomposed (`Session.load_channels()`,
  no extra ffmpeg run) and sent alongside as `DesignRequest.channels`, so
  a candidate's per-channel picture isn't just thrown away by only ever
  designing from a downmix. Finishing a design run switches to the Review
  tab with the new entries loaded.
- `model/extract.py`'s `ExtractAudioDialog` (Tools → Extract Audio) has
  the same optional "Design filters?" step for the single-file case --
  same `DesignJob` (reused directly, not reimplemented), same mono-
  downmix/`channels` behaviour, same `DESIGNER_DEFAULT`/`DESIGNER_QUEUE_DIR`
  preferences. Not offered in Remux mode, which applies a filter someone
  already designed/reviewed rather than designing a new one.
- **Review** (`model/review.py`/`ui/review.py`'s `ReviewQueueDialog`,
  embedded as a tab — `setWindowFlags(Qt.WindowType.Widget)` on an
  otherwise-unmodified `QDialog` instance) — one row per queue entry, a
  detail pane with every candidate's confidence/method/commentary and a
  **live** filtered-curve chart (redraws as the reviewer changes the
  pick), and keyboard-first triage: Enter/A accepts and advances, digit
  keys change the pick, S skips, R rejects (permanent, unlike skip).
  "Publish accepted" drives `pipeline.review.publish_reviewed_queue()` —
  XML-only from the dialog today; the underlying function also supports an
  images repo. `reject()` (QDialog's Escape-key handler) is overridden to
  a no-op -- the default behaviour hides the dialog, which would blank
  this tab when embedded rather than closing anything.

**Tools → Batch Extract / Design…** opens on the Run tab; **Tools → Review
Batch Designs…** opens the same dialog straight on the Review tab (e.g. to
review a queue directory a headless job populated, with no extraction of
its own to run).

`QueueEntry`'s format is published at `docs/schema/review_queue.schema.json`.
Each candidate's filters are stored as `CompleteFilter.to_json()` — the same
already-published `docs/schema/filter.schema.json` shape — so applying a
reviewer's pick is a plain `filter_from_json()`, not a second conversion
path.

## Library sync (CLI)

`pipeline.library.cli` runs the whole workflow over a library with no GUI, so it can be scheduled:

```
PYTHONPATH=src/main/python python -m pipeline.library.cli [--config FILE] run  [options]
PYTHONPATH=src/main/python python -m pipeline.library.cli [--config FILE] publish [options]
PYTHONPATH=src/main/python python -m pipeline.library.cli [--config FILE] commit  [options]
PYTHONPATH=src/main/python python -m pipeline.library.cli [--config FILE] sync    [options]   # publish, then commit
PYTHONPATH=src/main/python python -m pipeline.library.cli [--config FILE] revise  [options]   # send titles back
PYTHONPATH=src/main/python python -m pipeline.library.cli [--config FILE] scan    [options]   # what does each title need?
PYTHONPATH=src/main/python python -m pipeline.library.cli [--config FILE] status  [options]   # counts, from the last scan
PYTHONPATH=src/main/python python -m pipeline.library.cli [--config FILE] accept  [options]   # bulk accept confident titles
```

`--config` goes *before* the command. `-h` after a command lists every option with its meaning.

- **`run`** reads a library, extracts the audio of each title that is new or changed (idempotent -- a title whose
  source and settings are unchanged is skipped), designs it, and writes an entry to the **review queue**. It never
  publishes, so an unattended `run` has nothing to auto-publish. A title whose extraction or design **failed** is not
  tried again while its source and the settings are unchanged (`--retry-failed` tries it again), so a nightly job does
  not repeat a failure every night (a transient failure -- a NAS offline, a designer down -- stays skipped until then, so `run`
  prints `warning: N titles skipped: failed earlier ... use --retry-failed` on stderr whenever it skipped any; the exit status
  is unchanged). With a *selector* it does much more -- see "Doing the work for a selection" below.
- A person then reviews the queue in the app (Tools > Library Sync, or Review Batch Designs) and accepts entries.
- **`publish`** writes only the *accepted* entries into the catalogue repositories' working trees: the XML in one
  and, optionally, a report image in another. Each entry is marked published (meaning *written*), so a re-run only
  publishes what is still accepted. It commits and pushes nothing, so the result can be looked at first. An entry
  whose metadata is incomplete is **refused on its own** (`invalid_metadata`, listing the problems) and the rest are
  still published. `--republish` also writes again each *published* entry whose catalogue copy is **out of date** --
  a metadata typo fixed, a new poster, an edited project, or an XML missing from the repository -- at the same path,
  keeping it published, without a second review; `commit` then commits it as a revision. `--id` restricts it to
  named entries. The filter is published from each title's `.beq` project, so a hand edit is what ships, when the work
  directory is known: `--work-dir`, else `work_dir` in the `sync:` section, else in the `run:` section; without one the
  designer's own pick is published, which would write over a hand-edited filter. A changed report style (`ReportSpec`)
  or `--xml-dir` is not the same: the style makes titles out of date, a new `xml_dir` is a new *location* -- the file
  at the old one is left behind for a person to remove (and the image's GitHub owner/repo is not in the digest either).
  One bad entry never stops the batch: anything that goes wrong for it (incomplete or unbuildable metadata -- a missing
  or null title, an unknown field -- a project conflict, git refusing, a poster file that is gone) is that entry's
  `{"id", "error", ...}` result and it keeps its status, so a rerun retries it.
- **`commit`** commits what `publish` wrote and pushes it: **one commit per repository** containing exactly those
  files (anything else staged in the clone is left alone), then one push per repository, **images first** so a pushed
  XML never points at an image that is not there. Whether a file is committed or pushed is read from git, not
  remembered, so running it again does only what is left (a rejected push is retried; an unchanged file is not an
  error) and a commit you made by hand is respected. `--no-push` commits locally only. Paths within a repository
  are always `/`-separated (git's spelling, on Windows too), and the clone may be a subdirectory of a repository. A
  published file that is still not committed afterwards (a `.gitignore` rule matches it) is reported as
  `not_committed` and is an error, never silently skipped; an XML naming a report image, committed without an
  `--images-repo`, is warned about (`warnings`, also on stderr) because the image is not committed with it.
- **`sync`** is `publish` then `commit`. It runs as the invoking user's own git/SSH configuration.
- **`revise`** sends titles back: `--to review` reopens them for another pick (from accepted, skipped, rejected or
  published); `--to design` also drops the protection an accepted or published entry has, so the next `run`
  designs it again (metadata, artwork, the reviewer note and any hand-edited `.beq` project carry over);
  `--to extract` also forgets the recorded extraction, so the next `run` runs ffmpeg again. It changes state only;
  the work happens in the next `run` and `publish`/`commit`. A title whose files were **written but never
  committed** has them put back as git has them (deleted, or restored to the previous revision); one already
  **committed** keeps them, becomes a *revision* (`revision` on the entry counts these) and is rewritten at the same
  path when published again. Each records a line in the entry's reviewer note (`--reason` adds why). Both repositories
  are checked to be git repositories *before* anything changes (a `ValueError` otherwise), the images repository is put
  back before the XML repository, and the entry is written last, so a failure part-way leaves it `published` and
  running it again finishes the job. **Revision rule:** `revision` goes up when a committed XML that is unchanged in
  the working tree is superseded -- by a reopen, or by a `--republish` that rewrites it -- and not when it is
  already rewritten and uncommitted (that revision was counted); a never-committed file has nothing to count.
- **`accept`** is *bulk accept*: it accepts the designer's top pick for the titles waiting for review whose top pick is
  at least `--threshold` confident (default 0.90), and **leaves out, and reports,** any that a person should still look
  at -- incomplete metadata, a designer decline, or a `.beq` project someone has edited since it was designed. Each
  accepted title gets the reviewer note "bulk accepted, confidence >= 0.90". `--dry-run` shows what it would do.
  It works from the last `scan` and does not publish. Each entry is re-read as it is written: one a person changed since the
  plan (edited, accepted or skipped in the app) is left alone and reported as "changed while accepting".

- **`scan`** is *discovery*: it lists each source of a profile, merges them, reads the outputs (extract manifests, the
  review queue, `.beq` projects, both repositories) and records what every title **needs next**, without extracting,
  designing or publishing anything and without reading a media file (a JRiver listing is one request; a filesystem
  source costs one `stat` per file). The result is a disposable SQLite index in the work directory
  (`library-index.sqlite`, schema in `design/library-sync/workflow-rework/design.md` §12.5); deleting it costs a rescan
  and nothing else. A source that cannot be listed keeps the titles it had and is reported (exit status 1), so one
  library being down never makes its titles vanish -- and neither does one that lists *nothing* when it listed some at the last
  scan (an unmounted share): its previous listing is kept and it is reported, unless `--allow-empty`. `--source NAME` rescans just
  one. `--from-outputs` rebuilds a lost (or live) index from the outputs alone; the generation goes back to 0, so the next selector
  `run` scans first. Changing `--tv-mode` keeps a row that has a review pending, marked `superseded by <id>` in its detail, rather
  than calling it gone. An index file this did not write (a SQLite file with none of its tables) is refused, never overwritten.
- **`status`** prints, from that index, how many titles need each thing -- `attention`, `review`, `extract`, `design`,
  `publish`, `commit` -- and how many are `done`, plus what is new since the previous scan, the flags and each source's
  last scan. It never lists a source, so it is instant and suits a scheduled job that reports what is waiting for a
  person. `--json` is the machine-readable form. It opens the index read-only: it never creates, migrates or drops one.

**What a title needs** (`pipeline/library/state.py`, the first row that applies): *attention* if extract or design
failed (remembered against the source fingerprint and settings it failed with, and retried only when either changes),
the mono and multichannel projects disagree, or the source changed after the title was accepted; *review* if its
design is current and it is waiting for a person (a designer's decline included) or an accepted title's metadata is
incomplete; *extract*, *design*, *publish* (accepted and not written, or written and out of date -- a metadata typo
reaches the catalogue without a second review) and *commit* (written but not committed, or committed but not pushed)
are machine work; everything else -- pushed, skipped, rejected, and the flags **Ignored**, **Shadowed**, **Gone** -- is
*done*. **Possible duplicate** and **Already in catalogue** (its TMDB id is in the local XML repo under a file this
profile did not publish) are labels only. A title decided by a person is never made stale by a settings change, only by
its source.

**Libraries.** `--source jriver` reads the files under one Media Center *browse node*; `--source filesystem` reads
folders or globs. A DVD or Blu-ray rip folder is one title. TV can be run as a filter per episode (the default) or a
whole season joined into one track (`--tv-mode season`); the published metadata lists the episodes covered.

**Needs.** `ffmpeg`/`ffprobe` (DVDs also need a build with the `dvdvideo` demuxer), a designer (below), `pyyaml` to
read a `.yaml` config, and a TMDB API key if you want TMDB metadata (optional).

### Doing the work for a selection

`run` normally lists the library and works through everything. Give it any *selector* and it instead works from the
**index** (`scan` first; `run` takes one itself if there has never been one) on just the titles you select, and runs
every stage up to `--through` that each still needs:

```
--needs {attention,extract,design,review,publish,commit,done}   what the title needs next (repeatable)
--source NAME     --match TEXT     --id ID     --new-since-scan
--through {extract,design,publish,commit}                       how far to go (default design)
--retry-failed                                                   also try titles that failed before
```

The same vocabulary is what the work list's strip chips (**Attention, New, Extract, Design, Review, Publish, Commit,
Done**) and action button mean in the app; every selector given must hold. `--source NAME` is the name of one of the
*profile's* sources (with `--profile`), unlike `run --source jriver|filesystem` without one, which says which kind of
library to read.

- **`--through extract`** extracts; **`design`** extracts first if the title has not been, then designs; the user never
  picks prerequisites. Both are machine work.
- **`--through publish`** additionally writes the titles a person has **accepted**, and *published* titles whose
  catalogue copy is out of date (`--needs publish` selects them), into the repositories' working trees; **`commit`**
  also commits and pushes what was published (one commit and one push per repository, images first). Both need
  `--xml-repo` (from the `sync:` section as usual). A title waiting for **review** is never taken past design: review
  is a person's.
- A title that needs something the chosen `--through` does not reach, or cannot be helped by a run (waiting for review, a
  project conflict, a source changed since it was accepted, done), is **skipped and reported with the reason**.
- A failed title is remembered against its source and the settings and skipped until either changes, or
  `--retry-failed`. A TV episode that fails inside a season is remembered the same way.
- Work is done one title at a time, so a cancelled run (from Python: `run_stages(..., should_cancel=)`) leaves only
  whole titles done; the report says what was `attempted` and what was `not_run`. The index is refreshed at the end.

A scheduled job is `scan`, then `run --through design`; it never accepts, publishes or commits. Publishing and committing
what a person accepted is `run --needs publish --through commit` (or `sync`).

From Python the same is `pipeline.library.selection.Selection`, `selection_from_chip()`, `plan_stages()` (what would run, and what
would be skipped and why), `pipeline.library.stages.run_stages(profile, selection, through, run_config=, index=,
publish=, retry_failed=, should_cancel=, on_progress=)` (`Progress(done, total, title, stage)`), and
`pipeline.library.bulk.plan_accept()`/`accept_top_pick()`.

### Designers

`run --designer NAME` needs the designer registered in the process. The GUI does this from Preferences; the CLI
declares them itself, in the config file's `designers:` or with `--designer-url NAME=URL` (repeatable, wins over
the file), or you can pass an `http(s)://` URL as `--designer`. A designer is an HTTP endpoint speaking the
contract in `design/designer-interface.md`; the config form also takes a `timeout` (seconds, default 300) and
`headers` (e.g. a bearer token). An unknown name stops the run before anything is extracted.

### Config file

JSON or YAML. Every command-line option can be given in the file, under the same name with underscores, in the
section for its command; **a flag overrides the file**. Repeatable flags replace the file's list.

```yaml
sources:                         # a source's own settings; the run section may also hold them
  jriver:
    host: media.local
    port: 52199
    browse_node_id: 1007         # -1 is the root of the browse tree
    username: me                 # optional
    password: secret
    ssl: false
    timeout: 5
    path_mappings:               # JRiver reports the *server's* paths (Windows form); map them to local ones
      - {from: 'W:\', to: /media/films}
    external_id_fields:          # config only: which JRiver fields hold each id; omitted ones keep the defaults
      movie: {imdb: [IMDb ID], tmdb: [TheMovieDB Movie ID, TMDb ID]}   # first field with a value wins
      tv:    {imdb: [IMDb Series ID], tmdb: [TheMovieDB Series ID]}    # [] switches an id off
  filesystem:
    globs: [/media/films, '/media/tv/**/*.mkv']

designers:                       # config only for timeout/headers; see Designers above
  rolloff: http://designer.local:8080/design
  private: {url: 'https://designer.example/design', timeout: 600, headers: {Authorization: 'Bearer TOKEN'}}

run:
  source: jriver                 # or filesystem
  work_dir: /var/lib/beq/work    # extracted audio, caches, .beq projects
  queue_dir: /var/lib/beq/queue  # the review queue
  designer: rolloff
  tmdb_api_key: XXXX             # optional
  tv_mode: season                # episode (default) | season
  keep_multichannel: false
  audio_types: [DTS-HD MA 5.1]
  analysis: {target_fs: 1000, resolution: 1.0}   # analysis settings nest under run/sync, not top level

sync:
  queue_dir: /var/lib/beq/queue
  work_dir: /var/lib/beq/work    # publish from each title's .beq project, so a hand edit is what ships
  xml_repo: /home/me/beq-filters # a local clone
  xml_dir: filters
  images_repo: /home/me/beq-images
  image_dir: images
  meta_defaults: {source: Disc, author: me}   # config only: BeqMetadata fields for anything an entry lacks
```

Settings with **no flag**: `sources.jriver.external_id_fields`, `designers` timeouts/headers, and
`sync.meta_defaults`. Everything else has one.

### One catalogue from several libraries (a profile)

`run --profile FILE` reads a **catalogue profile** instead of `--config`: the same file format, plus an *ordered* list of
sources, ignore rules and per-title ignores. `--source`, `--glob` and the other source options are not used with it (every
other option still comes from its `run:` section, and flags still override).

```yaml
sources:                       # earlier wins when two sources have the same file
  - {name: films, kind: jriver, host: media.local, port: 52199, browse_node_id: 1007,
     path_mappings: [{from: 'W:\\', to: /media/films}]}
  - {name: disk, kind: filesystem, globs: [/mnt/extra/**/*.mkv]}
ignore:                        # a rule lists the fields it constrains; all must match
  - {path: /media/films/Kids/**}                  # a folder prefix or a glob
  - {kind: tv, reason: not doing TV}
  - {year: "<1960"}                               # 1960, <1960, >=1999 or 1990-1999
  - {title: "^Trailer"}                           # a regular expression
  - {source: disk, external_ids: {imdb: tt0113277}}
ignore_titles:                 # one title, by id
  jriver-3fa9c2-1234: the rip is broken
run: {work_dir: /var/lib/beq/work, queue_dir: /var/lib/beq/queue, designer: rolloff}
```

- **The same file in two sources is one title.** "Same file" means the same path after the source's own `path_mappings`,
  ignoring case and `\` versus `/`, with a disc rip's clips folded into the disc folder. The first source wins; the other is
  *shadowed* (no separate title, no second XML). Nothing is read from disk to decide this. Two items with the *same id* are one
  title (first wins); a shadowed copy that already has outputs of its own is flagged on the owner and says so; an item with no
  path never clashes on it; and a JRiver Blu-ray *playlist* entry (`BDMV\PLAYLIST\index.bluray;N`) is its own title, not the disc.
- **The same title in different files is not merged**, since it may be a real second entry (an edition, another audio track):
  both stay titles and each is flagged as a possible duplicate (same TMDB id, else IMDb id, else title and year).
- **A title keeps the id it already has**, so reordering `sources:` never orphans an extraction or publishes a second XML:
  the source whose item already has a queue entry or work directory keeps the file whatever the order. If that item leaves its
  source, the other one takes over *under its own id* (its old outputs are left behind, not reused). A TV season keeps its
  id too, if the series' title is corrected in the library.
- **Ignored titles stay in the list, labelled with the rule**, and are not run. Deleting the rule brings them back. A `path` rule
  matches an item whose path, or any folder above it, matches, so a glob that names a folder (`/films/*/extras`, `/films/Kids*`)
  ignores everything under it. `*`, `**` and `?` are globs; **`[` and `]` are literal** (a folder called `Movie [1080p]`), not a
  character class. A `title` rule is matched against the first 300 characters of the title; a pattern that backtracks
  catastrophically (`(a+)+$`) is the rule author's responsibility. `ignore_titles` also works on a TV *season* id in `tv_mode: season`.
- **Directories from flags count.** The sticky claims are read from the work and queue directories the run actually uses, whether
  the file or `--work-dir`/`--queue-dir` gave them.
- **Saving a profile** leaves alone a `run:`/`sync:` directory that differs and was not changed, keeps a file in the older
  `sources:` mapping shape in that shape (and refuses to convert it to a list if that would drop an unused source), and writes an
  unquoted YAML date back as text. A missing or malformed `--profile`/`--config` file is exit status 2.
  A malformed rule (an unknown key, a bad regular expression) is an error, so a typo cannot silently ignore nothing.
- A source that cannot be read (the media server is down) fails the run: it is never treated as an empty library.

The old shape (`sources:` as a mapping, `run.source` naming the one in use) still loads; `pipeline.library.profile` reads
either into one `Profile`.

### Options

Not repeated here: `python -m pipeline.library.cli run -h` and `sync -h` are the reference (a test keeps every
option documented). In outline, `run` takes the library source (`--source --glob --host --port --browse-node-id
--username --password --ssl --timeout --path-map`), where things go (`--work-dir --queue-dir`), design
(`--designer --designer-url --coverage --keep-multichannel --tv-mode`), redoing work (`--force-extract
--force-design`), which titles (`--needs --match --id --new-since-scan --through --retry-failed`), the repositories and `--push`
(only for `--through publish` or `commit`), metadata (`--tmdb-api-key --audio-type`) and analysis (`--target-fs --resolution --avg-window
--peak-window`); `publish` takes what to publish (`--queue-dir --work-dir --id --republish`), the repositories (`--xml-repo --xml-dir
--images-repo --image-dir --image-owner --image-repo-name`) and the same analysis options; `commit` takes `--queue-dir`, the same
repositories (without the image-URL options), `--id` and `--push`/`--no-push`; `sync` takes everything `publish` does plus `--push`; `revise` takes `--queue-dir --id --to --reason --work-dir` and the repositories. `publish`,
`commit`, `sync` and `revise` read the one `sync:` section of the config file. `scan` takes `--profile --source
--from-outputs --allow-empty`, where things are (`--work-dir --queue-dir`), the settings that decide what is up to date (`--designer
--coverage --keep-multichannel --tv-mode`, the repositories and the analysis options: give the values `run` and `publish`
get, including `--image-owner`/`--image-repo-name`, and `sync.report_spec` from the file: all three are in the published digest) and
reads both `run:` and `sync:`; `status` takes `--profile --work-dir --json`; `accept` takes `--profile --source
--match --id --new-since-scan --threshold --dry-run`, where things are (`--work-dir --queue-dir`) and the same settings as `scan`. The boolean flags come in
pairs (`--keep-multichannel` / `--no-keep-multichannel`) so a flag can turn something off that the file turned on.

### Output and exit status

The commands print JSON to stdout (`status` prints text unless given `--json`).

- `run` prints the report: `extracted` and `cached` (item ids whose audio was, or was not, re-extracted),
  `designed` and `design_cached`, `failed` (`[id, "ErrorType: message"]` -- one bad title never stops the rest),
  `meta_unresolved` (designed without TMDB metadata because TMDB failed), `project_edit_preserved` (a hand-edited
  `.beq` project was left alone), `failed_earlier` (`[id, message]`: not tried, because it failed before with the same
  source and settings -- `--retry-failed`) and, with `--tv-mode season`, `seasons` (season id -> its episodes). Exit status 1
  if anything is in `failed`. With a selector it prints `{through, selected, run, published, publish_errors, committed,
  commit_error, skipped, cancelled, attempted, not_run, counts}`: `run` is the report above, `published` and `committed` are
  what `publish` and `commit` print, `skipped` is `{id, title, reason}` per title left out, and `counts` is titles per needs
  afterwards. Exit status 1 if anything failed, was refused or could not be committed.
- `accept` prints `{threshold, note, accepted, excluded, below_threshold, not_for_review}` -- `excluded` is `{id, title,
  reason}` for a confident title left for a person, `below_threshold` counts the titles waiting for review whose top pick is
  less confident, and `not_for_review` those in the selection that were not waiting for review. `--dry-run` prints the
  same as `{eligible, ...}` and changes nothing.
- `publish` and `sync` print one object per entry they published (`id` plus the publish result -- `image_url`,
  and `republished`; not the XML itself, which is in the repository; `edited_project` and `projects_aligned` say a hand edit
  was what shipped) or refused (`id` and `error`, e.g. `project_conflict` when the mono and multichannel projects were
  edited to disagree, `invalid_metadata` with the `problems`, `git_failed` or `publish_failed` with a `message`). Exit
  status 1 if any entry was refused, 3 if git refused (see below). With `sync` each published entry also carries the
  batch's `xml_commit` (and `image_commit`) sha, where a commit was made -- also when a later push failed.
- `revise` prints one object per `--id`: `id`, `status`, `revision`, `reverted` (catalogue files put back as git has them) and
  `extract_invalidated`, or `id` and `error` (no such entry, already pending, published with no `--xml-repo`). Exit status 1 if any failed.
- `commit` prints `{"xml": {...}, "images": {...}, "missing": [...], "not_committed": [...], "warnings": [...]}`; each
  repository reports its `paths` handled, the new `commit` sha (null if everything was already committed) and whether it
  was `pushed`. `missing` lists published entries with no file in their repository (run `publish` again; exit status 1);
  `not_committed` lists published files git will not commit (exit status 3). If git refuses (a rejected push), it prints
  what was done before the failure with an `error` key, and git's message on stderr in one line.
- `scan` prints `{generation, titles, new, gone, dropped, errors, counts}` (`counts` is titles per needs; `errors` maps a
  source that could not be listed to why). Exit status 1 if any source could not be listed.
- `status` prints text by default or, with `--json`, `{generation, last_scan_at, titles, counts, new, flags, sources}`.
  Exit status 1 if there is no scanned index.
- Exit status 2 is a bad option or config file (a message on stderr).
- Exit status 3 is git: `publish`, `commit`, `sync` and `run --through commit` exit 3 when git refused (a rejected or
  failed push, an images repository that is not a git repository) or will not commit a published file. It is distinct
  from 1 ("an entry could not be published, or a published entry has no file"), so a wrapper can tell them apart.

A typical schedule: `run` nightly from cron, review in the app, `sync` (or `publish`, a look at the clones, then `commit`)
when the queue has accepted entries.

## Design decisions (resolved)

Kept here as a short index; each was worked through in more detail in
`design/api-headless-pipeline.md §14` before being implemented.

| | Decision | Resolution |
|---|---|---|
| D1 | Headroom measured at which sample rate? | The decimated analysis fs; `signal_stats()`/`Stats` carries that fs alongside the numbers so a published `beq_gain` is never ambiguous. |
| D2 | How is required attenuation expressed? | Signal offset, matching existing behaviour. The dead `__find_gain` (would-be `Gain`-filter path) was deleted; a `Gain` filter reaching `to_beq_xml()` fails loudly instead of silently omitting `beq_gain`. |
| D3 | beqcatalogue repo conventions | No fixed layout — `extract_from_repo()` globs `**/*.xml`. Two repos (XML, images); images referenced by GitHub raw-content URL; plain commit + push, no PR, triggered by `repository_dispatch`. |
| D4 | Does the report need to be pixel-identical to the GUI's? | No — a spec-driven render (fixed size/layout) of the existing "pixel perfect" mode, not a port of the dialog's layout code. |
| D5 | CLI vs service (whose git credentials?) | Runs as the invoking user's own git/SSH config — see `pipeline/publish/git.py`'s module docstring. |
| D6 | Catalogue-as-input (`CatalogueEntry.iir_filters()` → apply an existing published BEQ) | **Still open** — smaller, separate job, not built; worth confirming whether it's wanted. |
| D7 | Does designer provenance reach the report? | Yes, as far as the report (a human can see *why* a filter was accepted); never into the published XML, which has no field for it. |
| D8 | How does a designer bind to the pipeline? | In-process callable first, HTTP binding added later (`design/http-designer-binding-plan.md`) once cross-process/cross-language use actually needed it. |

## Testing

See `AGENTS.md`'s "Testing" section for the Qt-boundary subprocess gotcha
and the pytest-qt conventions the GUI dialogs (`gui/test_preferences_designers.py`,
`gui/test_batch_extract_design.py`, `gui/test_extract_design.py`,
`gui/test_review_dialog.py`) follow.

## Open / not built

- **D6** (above) — fetching an already-published BEQ and applying it,
  bypassing extract/design/metadata entirely. Confirmed smaller than this
  pipeline; not started.
