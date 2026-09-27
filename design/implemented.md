# Implemented design

This is the consolidated account of the features delivered by the headless
pipeline, candidate review, and library work list. It describes the code in
this repository, including the pipeline service delivered after the 2026-09-25
design sweep. The authoritative API
and user instructions are in [`pipeline/README.md`](../src/main/python/pipeline/README.md)
and [`docs/library/`](../docs/library/). The designer's external protocol
remains in [`designer-interface.md`](designer-interface.md); its conformance
checklist remains in [`designer-conformance-tests.md`](designer-conformance-tests.md).
Only unfinished work is listed in [`outstanding.md`](outstanding.md).

## Headless pipeline

`src/main/python/pipeline/` runs extraction, filter design, assessment,
review, and publication without importing Qt or constructing a
`QApplication`. Its reused model modules have Qt-free cores and separate
dialog/model modules. `AnalysisConfig` supplies explicit settings for
scripted runs.

`Session.extract_with_layout()` uses the existing ffmpeg `Executor` to select
an audio stream, resolve Blu-ray/DVD inputs, decimate to the analysis sample
rate, and report channel layout and count. `Session.load()` produces the mono
signal; `load_channels()` supplies aligned per-channel diagnostic arrays.
`Session.design()` creates and validates a versioned `DesignRequest`, invokes
an in-process or HTTP designer, and returns either `Applied` candidates or a
normal `Declined` result. Biquad conversion is in `pipeline/designer/convert.py`.
The HTTP request uses base64 float64 arrays and the versioned JSON response
schema in `docs/schema/`; configured endpoints live in Preferences > Designers.

`pipeline/stats.py` measures signal peak, RMS, crest factor, and headroom at
the analysis rate. `pipeline/metadata.py` validates publication metadata and
resolves TMDB identities. The publishing modules fetch artwork, render a
report without a GUI, resolve catalogue paths, write only named files, and
commit/push through the invoking user's git configuration. A `Gain` filter
cannot silently disappear during export. The original headless acceptance
example and the Qt boundary are covered by `test_pipeline_acceptance.py` and
`test_pipeline_qt_boundary.py`.

The published filter format is now a BEQCatalogue version-1 JSON record per
title plus a derived `database.json` in the filter repository. The publisher,
path naming, aggregate generation, local catalogue identity scan, and the
former XML-specific review/commit/index assertions have moved to this format.
Saved profiles use `sync.filter_repo` and `sync.filter_dir`; old `xml_*` keys
load and are rewritten on save, and conflicting values fail. The CLI exposes
`--filter-repo` and `--filter-dir` while accepting hidden old aliases. Library
GUI copy and the user guide describe JSON records. A fixture exercises the
BEQDesigner record through BEQCatalogue page/database generation and back
through `CatalogueEntry` (`a72c673`, BEQCatalogue `dc56c20d7`). The internal
`xml_*` names remain compatibility plumbing. BEQCatalogue now fetches
`3ll3d00d/beqfilters` into `.input/3ll3d00d/beqfilters` and includes it as
the `3ll3d00d` JSON record source (`f398ffd38`). Its image URLs point to the
separate `3ll3d00d/beqimgs` repository; the catalogue need not clone it.
Publication can place film records and images under `movies/` and TV records
and images under `tv/` in their respective repositories. The optional
`sync.category_folders` setting is exposed in Library Work List Settings and
as `--category-folders` in the CLI. Each filter directory has its own
`database.json`, written once per publish batch. Category folders are **on by default**
(`sync.category_folders: false` keeps the flat layout) and their names are `sync.movies_dir` / `sync.tv_dir`
(`--movies-dir`, `--tv-dir`). Files are named `Title (Year) (Edition) Audio` (`catalogue_stem()`), recorded on the
queue entry as `published_stem` so a metadata edit never moves them; entries published earlier stay at their id.
Accepting a title in the work list writes and commits it locally (`model/worklist_autopublish.py`); only the push is
a separate action. A heatmap (`pipeline/publish/heatmap.py`) is published beside the report image as the record's
second image.
Publish, commit, index status and revision use the same metadata classification.
The configured-source fixture reaches the public `database.json` and returns
to `CatalogueEntry` (`01d9ec5`). The producer repository presently contains only its
licence, so this is a fixture-backed integration check, not a claim that a
live filter has already been published.

## Pipeline service

`pipeline/service/` exposes the library workflow as a long-running HTTP
service over the same profile, index and stages as the CLI. Its typed FastAPI
interface serves OpenAPI 3.1 at `/openapi.json` and local interactive docs in
the Docker image; `docs/schema/service.openapi.json` is checked against the
generated document. Bearer authentication protects `/v1`, while `/health`
and `/ready` are public. Jobs run one at a time, persist bounded history,
stream redacted events and hold a work-directory lease while active.

The optional schedule scans and runs titles still needing extract or design,
never publish or commit, and waits from a scheduled job's finish before its
next tick. A busy tick is skipped. An optional notifier sends completed-job
events to explicitly configured webhook URLs. JSON carries job, designed
title, failure and review-count details; text, Slack and Discord carry a
summary. Redirects are refused and status shows delivery outcomes without
URLs or headers. The Qt-free Docker image has ffmpeg, git and SSH; CI builds
and smoke-tests it on pushes and before publishing amd64/arm64 release tags.
The image smoke has not yet been observed in CI from this worktree.

## Design and review

`pipeline/review.py` stores one JSON `QueueEntry` per title. Rerunning a batch
preserves decisions and reviewer edits where applicable. Candidates contain
filters and human-readable diagnostics; Skip leaves an entry pending and Reject
excludes it. A decline carries one flat candidate (no filters, no confidence;
entries written before are given it when read), so a person may accept it and
publish the title as a record with no filters and the note "Does not require
BEQ" unless the reviewer wrote one; bulk accept never takes it
(worklist-feedback F3). The interactive title page
shows candidates, average/peak before-and-after curves using the main chart's
measure colours and before/after line styles, commentary (wrapping text, a
heading per key, `;`-separated notes as a list; for a declined title, the
decline reason and message laid out the same way, over its measured curves),
metadata,
and artwork. It lets a reviewer Accept & next, Skip, Reject, reopen or revise,
or open the title's project in the main app. Batch Extract & Design can queue
its results, and Review Folder presents the same page over a queue directory.

Every extracted library title has mono and, with a kept multichannel
extraction, diagnostic multichannel `.beq` project files: the run writes any
that are missing, flat, as soon as the title is extracted, and design replaces
them unless a person saved one in between (worklist-feedback F2). The mono signal is the designer's primary input; per-channel arrays are
diagnostics. New multichannel projects contain a `BassManagedSignalData`
composite with the app's LPF settings; older flat projects remain readable.
The project filter hash protects a human edit from automatic regeneration.
Publishing reads the saved authoritative project filter, aligns the other
project when safe, and refuses a conflict instead of guessing. Reviewer
metadata, artwork, and notes survive a pending redesign; the candidate
selection resets because the candidate list has changed.

## Library discovery and sources

`pipeline/library/` defines `LibraryItem` and `LibrarySource`; the production
sources are the filesystem scanner and JRiver MCWS browse adapter. The source
registry is an in-process extension seam for future source types; configured
sources are constructed from the profile rather than registered globally.
The profile orders multiple sources, combines them with sticky ownership and
hard/soft conflict rules, and applies ignore rules. The Settings drawer edits
the profile with validation and atomic writes, including source ordering,
location checks, ignore previews, and a first-save path preference.

JRiver connections are managed in Preferences and shared with the filter
manager. Source setup includes a browse-node picker, local path mappings,
configurable external-ID fields, per-stream codec/channel requests that fill
the selected track's automatic audio type, and optional TVDB lookup after direct TMDB
and IMDb resolution. Unmapped Windows paths on non-Windows hosts produce a
safe, actionable diagnostic. The MCWS zone picker loads on a `QRunnable` and
discards a response that arrives after close or connection change. DVD roots
and Blu-ray roots can be resolved for extraction; a JRiver disc reported as a
pseudo-file (`index.bluray;N`, `VIDEO_TS.dvd;N`) or as the disc's own
`BDMV\index.bdmv` is listed as its disc folder, and extraction also accepts an
`index.bdmv` path as its disc; precise JRiver disc title
mapping and live server evidence remain open. TV can be processed by episode
or as a season whose mono episode tracks are joined in order.

A fresh JRiver scan updates an existing single-title queue entry's audio type
when it is missing or still matches the previous automatic value. A different
value entered during review is retained. This also covers entries made before
codec fields were requested when the discovery index has been rebuilt; the
updated entry is evaluated in the same scan.

`pipeline/library/index.py` is a disposable SQLite discovery index. A scan
lists sources, reads existing work and repository outputs, then stores each
title's next need and reason. The needs state comes from `state.derive_needs()`;
the UI reads index rows instead of recomputing them. `extract_cache.py` uses
source fingerprints and extraction parameters; `design_cache.py` uses the
design fingerprint and protects reviewed/published entries. Both mono and
diagnostic channel audio use the analysis sample rate. Keeping multichannel
audio is optional and off by default. When a multichannel copy is requested,
the current library path extracts it first and builds the
mono mix from its samples and the extractor's pan coefficients. A source
already known to be mono uses the direct mono extraction path. Older cached
multichannel files without recorded coefficients fall back to a direct mono
extraction once. The selected stream, channel count, and fingerprint govern
whether cached audio can be reused.

New JRiver and filesystem extractions use a readable folder named after the
track and selected audio stream. A hidden marker records the stable library ID
so discovery, review projects and publishing still find the same outputs after
restart; older ID-named folders remain usable as cached extractions.

`pipeline/library/stages.py` runs selected titles through extract and design
with separate bounded capacities and cooperative cancellation. A run can
continue from a chosen stage, retry failures, and preserve per-title failure
memory. A failed extraction of a title still in play stays `extract` work:
a run a person starts tries it again, an unattended one (the service's
schedule, `run --unattended`) skips it until the source or settings change,
and Revise is not offered for it. A failed design needs attention and is
retried only on request (worklist-feedback F4). The CLI exposes scan, status, run, revise, accept, publish, commit,
and sync. Publish writes accepted output into repository working trees;
Commit commits and pushes the named published files separately. Revision
reopens a title or invalidates extraction/design as requested, while an edited
project remains protected. Bulk accept requires a threshold and confirmation.
The work list shows a settings-drift banner when accepted/published entries
were designed under different settings.

## Library work list and runtime behavior

The Tools menu opens the Library Work List, a top-level window over the index.
It has pipeline counts, searchable/sortable rows, source and tier filters,
Rescan, selected/all-visible actions, settings, failures, and Last run. A
title page replaces the table in the same window and preserves selection and
scroll position when closed. The Metadata tab validates required fields,
autosaves edits, reloads TMDB data on a worker, and handles poster downloads.
Publish and Commit ask for the target repositories and interlock with running
work. Review Folder uses the same title and publish/commit behavior over a
chosen queue directory. The old Library Sync and Review Queue dialogs were
retired.

Extraction and design workers have separate limits. Publish/commit remain
serialized. Work list rows show bounded, run-scoped progress; Details shows
per-title execution events and ffmpeg commands with selection and Copy all.
The bounded, redacted Details text of the last run is stored in the work
directory and restored with its button when the work list reopens. Starting a
new run clears that history; a completed or failed run replaces it.
The list's run bar counts completed titles; while a title page is open, that
bar follows the viewed track's measured progress and changes when the page
moves to another track.
New runs clear transient row state and limit outcome/progress updates to their
planned titles. The UI reports aggregate progress, cancellation, and the
result of titles that were omitted after a bulk confirmation. Some retry,
failure-text, and stream-stage presentation work remains under **W1/W2**.

## Deliberate boundaries

- A library source path must be readable by the host running ffmpeg; MCWS
  access alone does not transfer the media.
- The source registry has no production caller until a second kind needs it.
- Redesigning pending work resets its selected candidate, while preserving
  reviewer metadata, art, and note.
- A multichannel project hash observes the linked master filter. An edit made
  only to a freed slave channel is not currently detected as a publication
  edit; conflicting authoritative projects are refused.
- Season mode joins mono episodes without level matching and does not keep a
  season-wide multichannel project. Whether this needs product work depends
  on the evidence in **L3**.
- Catalogue-as-input, which would apply an existing published BEQ without
  extracting or designing, was never part of the delivered pipeline. It is
  an optional future feature in **O1**.
