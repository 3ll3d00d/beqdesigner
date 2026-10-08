# Implemented design

**Document type:** Architecture reference — delivered behavior.

This is the account of the headless pipeline, candidate review, the library
work list and the pipeline service as they are built: it describes the code at
`HEAD`, not the history of how it got there. The authoritative API
and user instructions are in [`pipeline/README.md`](../src/main/python/pipeline/README.md)
and [`docs/library/`](../docs/library/). The designer's external protocol
remains in [`designer-interface.md`](designer-interface.md); its conformance
checklist remains in [`designer-conformance-tests.md`](designer-conformance-tests.md).
Only unfinished work is listed in [`TODO.md`](TODO.md).

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
through `CatalogueEntry`. The internal `xml_*` names remain compatibility
plumbing. BEQCatalogue fetches `3ll3d00d/beqfilters` into
`.input/3ll3d00d/beqfilters` and includes it as the `3ll3d00d` JSON record
source. Its image URLs point to the
separate `3ll3d00d/beqimgs` repository; the catalogue need not clone it.
Publication can place film records and images under `movies/` and TV records
and images under `tv/` in their respective repositories. The optional
`sync.category_folders` setting is exposed in Library Work List Settings and
as `--category-folders` in the CLI. Each filter directory has its own
`database.json`, written once per publish batch. Category folders are **on by default**
(`sync.category_folders: false` keeps the flat layout) and their names are `sync.movies_dir` / `sync.tv_dir`
(`--movies-dir`, `--tv-dir`). Files are named `Title (Year) (Edition) Audio` (`catalogue_stem()`), recorded on the
queue entry as `published_stem` so a metadata edit never moves them; entries published earlier stay at their id. Since
letter folders, the recorded stem carries a folder named by its first letter (`H/Heat (1995) Atmos`, `lettered_stem()`:
upper case, unaccented, `0-9` for a digit, `#` otherwise), so a title's record and images sit in `<category>/<letter>/`;
a stem recorded before then has no folder and stays where it is. `database.json` stays in the category folder
(`record_folder()`), aggregating the letter folders beneath it.
Accepting a title in the work list writes and commits it locally (`model/worklist_autopublish.py`); only the push is
a separate action. A heatmap (`pipeline/publish/heatmap.py`) is published beside the report image as the record's
second image.
Publish, commit, index status and revision use the same metadata classification.
The configured-source fixture reaches the public `database.json` and returns
to `CatalogueEntry`. The producer repository presently contains only its
licence, so this is a fixture-backed integration check, not a claim that a
live filter has already been published.

## Pipeline service

`pipeline/service/` exposes the library workflow as a long-running HTTP
service over the same profile, index and stages as the CLI; its full design
is [`pipeline-service.md`](pipeline-service.md). Its typed FastAPI interface
serves OpenAPI 3.1 at `/openapi.json` and local interactive docs in the Docker
image; `docs/schema/service.openapi.json` is checked against the generated
document. Bearer authentication protects `/v1`, while `/health` and `/ready`
are public. Jobs run one at a time, persist bounded history, stream redacted
events and hold the work-directory lease while active; a run job submitted
while a run job is extracting or designing joins it.

The optional schedule scans and runs titles still needing extract or design,
never publish or commit, as an unattended run, and waits from a scheduled
job's finish before its next tick. A busy tick is skipped. Before a tick (and a
run job) through design, `DesignerProbe` asks the designer's `/health`
(`http_binding.check_designer`: a by-reference designer must answer 1.2 with a
shared root; any other is down only for no answer or 502/503/504). A refused
tick is skipped with `last_skip: designer unavailable`, retried within five
minutes, and notified as `failed` once per outage. `/ready` reports the
designer's reachability and whether a TMDB key is set as checks with
`required: false`, so neither makes the service unready; `/v1/status` reports
both, and a running job's progress carries `per_hour`, `remaining_seconds` and
`estimated_finish`. An optional
notifier sends completed-job events to explicitly configured webhook URLs.
JSON carries job, designed title, failure and review-count details; text,
Slack and Discord carry a summary. Redirects are refused and status shows
delivery outcomes without URLs or headers. The Qt-free Docker image has
ffmpeg, git and SSH; CI builds and smoke-tests it on pushes and before
publishing amd64/arm64 release tags. The amd64 image build and smoke run are
verified; the arm64 build and GHCR publish have not yet run
([C1](TODO.md#c1--arm64-image-and-ghcr-publish)). `docker/compose.example.yaml`
runs beqforge's designer image (`ghcr.io/3ll3d00d/beqforge-designer`, pinned to
0.2.0) beside the pipeline, both mounting the work directory so audio goes by
reference; `docker/smoke.py --designer-image` designs a title through it and
checks the designer's log that it did. The guide's supported desktop-review
layout (`docs/library/service.md`) keeps the folders on the container host,
shared to a desktop with its own profile, reviewing through Review Folder while
a run goes; a queue entry's poster is found in the title's folder whatever the
root (`review.entry_art_path`).

## Design and review

`pipeline/review.py` stores one JSON `QueueEntry` per title. Rerunning a batch
preserves decisions and reviewer edits where applicable. Candidates contain
filters and human-readable diagnostics; Reject excludes an entry (the
`skipped` status is still read: the title page's Skip used to write it). A decline carries one flat candidate (no filters, no confidence;
entries written before are given it when read), so a person may accept it and
publish the title as a record with no filters and the note "Does not require
BEQ" unless the reviewer wrote one; bulk accept never takes it. The interactive title page
shows candidates, average/peak before-and-after curves using the main chart's
measure colours and before/after line styles, commentary (wrapping text, a
heading per key, `;`-separated notes as a list; for a declined title, the
decline reason and message laid out the same way, over its measured curves --
the only place the decline is told; the notice and state lines do not repeat it),
metadata,
and artwork. Its top row combines navigation and decisions: review sessions
show Skip (movement only), Reject and Accept & next, with revision and stream
choices under More. Browsing other title states also offers Previous/Next.
Neither decision is offered for a title whose last extraction or design failed,
where Retry is what it needs. Open project is one dropdown for mono or
multichannel in the same bar; its "opened" note clears when the page changes.
In a narrow metadata
column every label goes above its field. Batch Extract & Design can queue
its results, and Review Folder presents the same page over a queue directory.

Designs the designer built and judged unfit to publish (contract 1.1's
`rejected`, with `rejection_reasons`) are validated as candidates are and
carried as `.rejected` on `Applied` and `Declined`, never applied. The queue
entry keeps them (`QueueEntry.rejected`), for a decline as for a success;
`chosen_candidate_index` counts through `offered` -- the candidates, then the
rejected designs -- so a pick past the candidates is a person's override of the
designer, recorded by the entry itself (`overrides_rejection`), and `chosen` is
what apply, publish and the digest use. Nothing picks one automatically: bulk
accept takes index 0. The title page lists them after the candidates under a
heading that cannot be picked, numbered on from them (so the digit keys reach
them), in italics with their reasons as a tooltip; picking one shows its
reasons first in the commentary (one bullet each, never split: a reason may
itself contain `; `, as beqforge's do) and its filter on the chart, and the Accept
button reads *Override & accept...*. Accepting one asks first, with the reasons
(`confirm_override`); Cancel writes nothing. An accepted override is said on the
state and decision lines. A redesign that changes only the rejected designs is
still a changed design, and is not accepted blind.

A designer declared `by_reference: true` in the profile (contract 1.2, §7.1;
[`archive/designer-by-reference.md`](archive/designer-by-reference.md)) is sent each array
whose WAV is under the run's `work_dir` as a relative path, a column and the
SHA-256 of the samples, rather than the samples themselves. `design_and_queue`
names the mono WAV and the kept multichannel WAV it loaded
(`pipeline.designer.sources`), and the binding sends an array inline when the
WAV's header shows it cannot be that array unchanged. Before the first request
by reference, the binding checks the designer's `/health` once. A `422` fails
the design with the designer's reason and is never retried inline. Every
other designer, and every array without a usable source, is sent inline as
in 1.1, and such a request still says `"1.1"`.

A design sends the playback chain: a profile's `run.bass_management`
(`pipeline/library/bass.py`, the contract's five keys) on the library path, the
Preferences crossover in Batch Design. The queue entry records it with the
designer's name and the build it reported (`designer_build()`: a build key in its
commentary, or a decline's closing bracket), and the title page's commentary
ends with both. A declared designer's timeout is multiplied by
`run.parallelism.design`, since a one-at-a-time designer queues the rest.

Every extracted library title has mono and, with a kept multichannel
extraction, diagnostic multichannel `.beq` project files: the run writes any
that are missing, flat, as soon as the title is extracted, and design replaces
them unless a person saved one in between; an existing project is never
replaced at extraction. The mono signal is the designer's primary input; per-channel arrays are
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
`BDMV\index.bdmv` is listed as its disc folder, and an entry that is a playlist
file (`BDMV\PLAYLIST\00305.mpls`) as its disc with that playlist. A Blu-ray's
title (`model.bdmv.resolve_main_title`) is the playlist named (JRiver's
`BlurayPlaylist`), else the one as long as the source's `Duration` (two that
close told apart by the first audio codec), else the longest; a named playlist
missing a clip falls back to one of that length, and a rip whose feature-length
playlists all miss one fails as incomplete. Any source's title leaves out a
clip under two minutes at either end whose audio differs from its longest
clip's (a studio logo), since ffmpeg takes a joined input's streams from its
first clip. DVD title choice is still the longest title (**E3**). TV can be
processed by episode or as a season whose mono episode tracks are joined in
order.

The JRiver adapter is checked against a sanitised capture of a real MC 36
server (`src/test/python/fixtures/jriver/`, with its README): field aliases
(`Year` answers as `Date (year)`), absent unset fields, drive-letter case,
artwork beside the media or in MC's cover-art folder (found through a path
mapping of it), and `Browse/Children`, which is read from its XML in order so
two nodes of one name keep their ids. Each audio stream's codec, channels,
sample rate, bitrate, language and title are requested and kept per stream;
`pipeline/library/streams.py` says them in words for the title page's choice.
JRiver's Playback Info `Streams` (video, audio, subtitle, by ffprobe's global
index) is kept as `selected_streams` and resolved against a probe of the file
before extraction (`run.resolve_selected_stream`, trusted only when it names a
video then an audio stream); a reviewer's choice (`audio_stream_source:
manual`) and a resolved one survive a rescan (`index.carried_choice`). A source
that lists no streams has them read from the file when a person chooses one.

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
retried only on request. A failure because something the title depends on was
unavailable (`pipeline/library/failure.py`: a connection error, timeout or 5xx,
a remote-filesystem errno, a missing file under an empty mount point or an
absent drive) is reported as `unavailable` and not remembered, so the next run
tries it again; `run.stop_after_unavailable` (3) of them in a row stop the run,
and below `run.min_free_gb` (10) free an extraction stops it at once. A title
that only needs design, with its audio still current
(`run.cached_unit_work`), goes straight to design without an extract stage.
Once a title is published, its kept `multichannel.wav` is compressed losslessly
to FLAC (`pipeline/library/retention.py`) and restored by whatever next reads
it; the extract cache and publish digest treat it as unchanged. The CLI exposes scan, status, run, revise, accept, publish, commit,
and sync. Publish writes accepted output into repository working trees;
Commit commits and pushes the named published files separately. Revision
reopens a title or invalidates extraction/design as requested, while an edited
project remains protected. Bulk accept requires a threshold and confirmation.
The work list shows a settings-drift banner when accepted/published entries
were designed under different settings.

## Runs, the lease and joining

Every run -- a work-list run, a CLI `run`, a service job -- holds the
work-directory lease (`<work_dir>/service/lease.json`, heart-beaten, taken over
when stale), so two runs never write one index, queue and repositories side by
side. A run's machine phase (extract and design) takes more titles while it
lasts: `run_stages(join=...)` polls a `JoinQueue`, and extract/design work asked
for from another process goes through the work directory's join inbox, which
the run holding the lease claims from. What the run did not take in time is in
`report.not_joined` and is run by whoever asked once the lease is free. The CLI
hands its titles to the run in progress and waits for it, then reports them
from the index. Publish and Commit never join: they wait for the run (in the
same window, the CLI or the service) or refuse (the work list, while another
process holds the lease). The mechanics are in
[`pipeline-service.md` §5.1](pipeline-service.md#51-the-work-directory-lease-and-joining-a-run).

## Library work list and runtime behavior

The Tools menu opens the Library Work List, a top-level window over the index.
It has pipeline counts (with a *Working* chip for the titles the window's run
has queued or in hand, from its run state rather than the index; a title moves
from Extract to Design as soon as its extraction ends), searchable/sortable
rows, source and tier filters, run actions, Rescan, selected/all-visible
actions and settings. There is no separate failures or Last run panel: a failed
title is a row (the *Attention* chip lists them, the row says why), *Retry N
failed* beside the action button runs the selected failed titles again (else
every failed title listed), each title's last outcome is on its row, and what
belongs to no title (a repository that could not be committed or pushed) is on
the run status line, whole in its tooltip. While its run goes, the action
button and *Retry failed* add extract/design work to it; Publish, Commit and a
bulk accept or revise wait and start in order when it ends, and Cancel drops
them (see *Runs, the lease and joining* below). A
title page replaces the table in the same window and preserves selection and
scroll position when closed. Its header places navigation, the title and
notices/action messages and decision/workflow controls left to right in one
compact row; title and notice text
wraps when space is tight. The duplicate title breadcrumb is hidden, and source,
status and metadata readiness live in Metadata rather than above the charts.
Opening a review title scopes navigation and the position count to the review
titles in the current filtered list, held fixed for that session. Review
sessions show Accept & next, Skip and Reject instead of Previous/Next, with
revision and stream choices under More; browsing other states retains Previous/Next.
Open project is a single top-bar dropdown with mono/multichannel choices, enabled
per readable file and main-window availability. Its tooltip carries project-edit
status; unavailable choices explain why in their tooltips. The candidate pane
always shows the selected proposed filters in a read-only table (type, frequency,
Q, S, gain and biquad count), including rejected candidates; switching right-hand
tabs leaves it visible. Missing designs clear the table.
The Metadata tab validates required fields,
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
result of titles that were omitted after a bulk confirmation. A title's extract
stage says which stream it takes and why ("as the library plays it", "your
choice"), whether multichannel is kept and how many channels it found; a run
that stops on an unavailable dependency says so and lists the rest as *Not
run*. Opening the window rescans when the index was never scanned or the last
scan is more than 12 hours old.

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
  on the evidence in **E5**.
- `run.bass_management` is not in the design fingerprint: changing it does not
  mark designs stale (Revise does). Nor is the Blu-ray logo-clip rule in the
  extraction key: titles extracted before it need a re-extract.
- A missing source file under an empty folder is taken for an unmounted share
  and retried, never remembered; a file missing from a populated folder is the
  title's own failure.
- The discovery index is a database file in the work folder and is not written
  from two machines at once: a desktop reviewing a container's work uses
  Review Folder while a run goes.
- Catalogue-as-input, which would apply an existing published BEQ without
  extracting or designing, was never part of the delivered pipeline. It is
  an optional future feature in **O1**.

Review Folder checks the library profile’s work-directory lease before and
after Publish/Commit confirmation and holds it while its worker writes. A
fresh holder is named on refusal; stale leases do not block the operation.

Failed extraction and design have stage-specific Retry labels in the work list
and title page. A title’s Failures tab shows the full redacted indexed failure,
with selection and Copy failure, separately from the current attempt. During a
retry, the previous indexed failure remains on that tab and in the table’s
Detail tooltip until the run’s refreshed result is read; intermediate extraction
refreshes do not drop it. Success removes it, a new failure replaces it, and
cancellation leaves the index’s failure available.

The title page’s Run Details button and the table’s Details cell open the same
copyable dialog. Persisted failure text is separate from one-run event history,
so a failed title remains inspectable after a disjoint run expires its events.
Cache hits explicitly say a cached result was reused and no command ran. The
old separate Failures/Last run panels are not restored: failures are listed in
the main table, inspected on the title page, and outcomes appear on the rows
and status line.

The title page precomputes the Spectrum comparison image as soon as a title or
candidate is selected, on a `QRunnable` even while another tab is visible. It
uses the profile’s analysis config and the same
`heatmap_for`/`spec_from_preferences` functions (filtered left, unfiltered right,
40 Hz on both panes). `preview_published_projects` gives saved edits the same
precedence as publication, without writing projects. The selected candidate,
settings and audio/project file stamps identify cached images. A page retains
up to four PNGs in least-recently-used order for revisited titles/candidates;
Refresh bypasses that cache. Generation duration is recorded in the app logs.
Changed requests coalesce behind one worker; results for a title left behind
are discarded. Missing audio and project conflicts are shown on the tab.

The title page’s Published BEQs tab uses `worklist_catalogue` to match the
aggregate catalogue by TMDB identity (movie/TV separately), with title/year
fallback, then audio and known edition/language/source and TV episode metadata.
Audio codecs match when their normalized sets intersect, accepting lists or
comma/slash/semicolon/pipe-separated strings and preserving DD+. An explicit
checkbox broadens to other tracks/editions of the same title.
`worklist_published` shares Browse Catalogue’s `database.json` cache, performs
bounded downloads and atomic replacements on a worker, and keeps cached entries
available after refresh failures. A resizable horizontal split places all
controls, selectors and track details in the left column, with one
aspect-preserving chart/heatmap filling the right column. “Show our filter
alongside” adds a resizable third pane containing the current design’s average
and peak curves before and after filtering, without a legend. Both library
charts always use Speclab average/peak colours irrespective of master
preferences; candidate changes update the side-by-side chart and
leaving the title clears it. Images have validated caches
keyed by complete URL; pending requests coalesce and stale image results are
ignored after title/selection changes or leaving. This is read-only browsing,
with an explicit browser link, not an audio-identity check or filter import.
