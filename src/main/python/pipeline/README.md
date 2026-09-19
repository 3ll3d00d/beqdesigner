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
        git.py                    # commit + push XML/images to their target repos
        project.py                 # per-title .beq project files; a hand edit of one is what gets published
    library/                       # library-scale runs -- see "Library sync (CLI)" below
        source.py, registry.py      # LibraryItem / LibrarySource, an in-process registry of sources
        jriver.py, filesystem.py     # the JRiver (MCWS browse node) and plain-filesystem sources
        pathmap.py                    # translate a server's (Windows) paths to local ones
        extract_cache.py, design_cache.py   # idempotent extract and design
        season.py                      # TV: join a season's episodes into one track
        run.py, sync.py, cli.py         # run_library(), sync_library(), the command line

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
PYTHONPATH=src/main/python python -m pipeline.library.cli [--config FILE] sync [options]
```

`--config` goes *before* the command. `-h` after a command lists every option with its meaning.

- **`run`** reads a library, extracts the audio of each title that is new or changed (idempotent -- a title whose
  source and settings are unchanged is skipped), designs it, and writes an entry to the **review queue**. It never
  publishes, so an unattended `run` has nothing to auto-publish.
- A person then reviews the queue in the app (Tools > Library Sync, or Review Batch Designs) and accepts entries.
- **`sync`** publishes only the *accepted* entries: XML to one repository and, optionally, a report image to
  another. Each is marked published, so a re-run only retries what is still accepted. It runs as the invoking user's
  own git/SSH configuration.

**Libraries.** `--source jriver` reads the files under one Media Center *browse node*; `--source filesystem` reads
folders or globs. A DVD or Blu-ray rip folder is one title. TV can be run as a filter per episode (the default) or a
whole season joined into one track (`--tv-mode season`); the published metadata lists the episodes covered.

**Needs.** `ffmpeg`/`ffprobe` (DVDs also need a build with the `dvdvideo` demuxer), a designer (below), `pyyaml` to
read a `.yaml` config, and a TMDB API key if you want TMDB metadata (optional).

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

### Options

Not repeated here: `python -m pipeline.library.cli run -h` and `sync -h` are the reference (a test keeps every
option documented). In outline, `run` takes the library source (`--source --glob --host --port --browse-node-id
--username --password --ssl --timeout --path-map`), where things go (`--work-dir --queue-dir`), design
(`--designer --designer-url --coverage --keep-multichannel --tv-mode`), redoing work (`--force-extract
--force-design`), metadata (`--tmdb-api-key --audio-type`) and analysis (`--target-fs --resolution --avg-window
--peak-window`); `sync` takes what to publish (`--queue-dir --work-dir`), the repositories (`--xml-repo --xml-dir
--images-repo --image-dir --image-owner --image-repo-name`) and the same analysis options. The boolean flags come in
pairs (`--keep-multichannel` / `--no-keep-multichannel`) so a flag can turn something off that the file turned on.

### Output and exit status

Both commands print JSON to stdout.

- `run` prints the report: `extracted` and `cached` (item ids whose audio was, or was not, re-extracted),
  `designed` and `design_cached`, `failed` (`[id, "ErrorType: message"]` -- one bad title never stops the rest),
  `meta_unresolved` (designed without TMDB metadata because TMDB failed), `project_edit_preserved` (a hand-edited
  `.beq` project was left alone) and, with `--tv-mode season`, `seasons` (season id -> its episodes). Exit status 1
  if anything is in `failed`.
- `sync` prints one object per entry it published (`id` plus the publish result; `edited_project` and
  `projects_aligned` say a hand edit was what shipped) or refused (`id` and `error`, e.g. `project_conflict` when the
  mono and multichannel projects were edited to disagree). Exit status 1 if any entry was refused.
- Exit status 2 is a bad option or config file (a message on stderr).

A typical schedule: `run` nightly from cron, review in the app, `sync` when the queue has accepted entries.

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
