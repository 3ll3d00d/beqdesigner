# Library sync plan -- orchestration, CLI and the GUI as built

> Part of the library sync plan -- **start at the index**: [`../library-sync-pipeline-plan.md`](../library-sync-pipeline-plan.md).
> Contains §5, §6, §7. Section numbers are global across the plan; the index maps every `§` to its file.
> Status of what this file describes: **built** (see `status-and-open-items.md` §10 for the exceptions); **§7 (the GUI) is superseded by `workflow-rework/design.md` §12.10** and its code was deleted in chunk 27c (`LibrarySyncDialog`, `ReviewQueueDialog`); the text below is kept as history

## 5. Orchestration

New `pipeline/library/run.py`:

```python
@dataclass(frozen=True)
class LibraryRunConfig:
    work_dir: str
    queue_dir: str
    designer: str
    config: AnalysisConfig = field(default_factory=AnalysisConfig)
    coverage: Coverage = 'complete_programme'
    keep_multichannel: bool = False  # mirrors model/batch.py's "Mix to Mono?" -- see §4.1
    force_extract: bool = False
    force_design: bool = False
    tmdb_api_key: Optional[str] = None   # added: resolve_meta() (§3.1.1) only runs when set
    audio_types: Sequence[str] = ()      # added: forwarded to resolve_meta() as BeqMetadata.audio_types

@dataclass(frozen=True)
class LibraryRunReport:
    extracted: List[str]   # item ids that ran ffmpeg this run (either extraction)
    cached: List[str]      # item ids whose extraction(s) were all cached
    designed: List[str]    # item ids that ran a designer this run
    design_cached: List[str]  # also holds accepted/published entries skipped as protected -- not distinguished
    failed: List[Tuple[str, str]]  # (item id, "ExcType: message") -- one item's failure never aborts the rest
    meta_unresolved: List[Tuple[str, str]]  # designed with item.meta only because TMDB failed (§3.1.1)
    project_edit_preserved: List[str]  # item ids designed while a human-edited .beq project was kept (§3.3.1)
    # (all seven default to empty lists)

def run_library(source: LibrarySource, run_config: LibraryRunConfig,
                on_item_done: Optional[Callable[[str], None]] = None,
                **source_query) -> LibraryRunReport:
    ''' source.list_items(**source_query) -> extract_if_needed -> design_if_needed, per item.
    Never calls publish -- see sync_library(). '''
```

Per-item failures (bad file, ffmpeg error, designer exception) are
caught and recorded in `LibraryRunReport.failed`, not raised -- a
library-scale run over hundreds of titles must not abort on the first
bad one (matches `Applied`/`Declined`'s existing "expected outcomes
are values, not exceptions" convention).

## 6. CLI entry point

New `pipeline/library/cli.py` (still Qt-free -- lives inside
`pipeline/`, same import-direction rule as everything else in the
package) with two subcommands:

```
python -m pipeline.library.cli run  --source jriver --work-dir ... --queue-dir ... --designer ...
python -m pipeline.library.cli sync --queue-dir ... --work-dir ... --xml-repo ... --images-repo ...
```

`sync`'s `--work-dir` is what lets `publish_reviewed_queue()` regenerate
output 1's project files for a reviewer's actual pick when it differs
from the top pick (§3.3) -- optional; omitting it just skips that
regeneration (backward compatible).

Source connection details (JRiver ip/auth) and repo targets come from
a config file (YAML/JSON, a plain dict-to-dataclass load, no
QSettings/Preferences dependency -- consistent with `AnalysisConfig`'s
"explicit input, not a process-wide singleton" rule) or CLI flags (flags
override the file). This is what makes the workflow cron-able.

As built: the config file has `run:`, `sync:`, `sources.<name>:` and
`designers:` sections (`analysis:` is a key *inside* `run:`/`sync:`, not a top-level
section, and `designers` is new -- see below) (`pyyaml` added as a dependency); `run` prints
`LibraryRunReport` as JSON and exits 1 if any item failed; `sync` prints
the publish results and exits 1 if any carries an `'error'`; only
`--source jriver` is accepted; `--tmdb-api-key`/`--audio-type` exist for
metadata resolution. `sync` has no way to pass `meta_defaults` except via
the config file's `sync.meta_defaults`.

**Documentation.** Only 3 of ~40 options had help text and nothing but this
plan described the CLI. Now every option has `--help` text (grouped by purpose,
with an epilog saying how the config file maps to flags, the exit status and
the output), `pipeline/README.md` has a "Library sync (CLI)" section (workflow,
designers, a full annotated config file, the settings that have no flag, the
JSON output and exit codes), and `test_pipeline_library_cli_docs.py` fails if
an option lacks help or is missing from the README, and checks that the
README's example config actually drives the CLI.

**Designers (gap found and fixed).** The CLI originally had no way to
register a designer, so `run --designer X` could never resolve `X` outside the
GUI (which registers its Preferences endpoints at startup): every item failed
with `KeyError: No designer registered`. Designers are now declared in the
config file's `designers:` mapping (`name: URL` or `name: {url, timeout,
headers}`), with `--designer-url NAME=URL` (repeatable; wins over the file), or
by passing an `http(s)://` URL as `--designer`. `run` checks the name is
registered *before* extracting anything and stops with a message saying how to
declare it.

## 7. GUI integration

> **Superseded (2026-09-19; deleted in chunk 27c, 2026-09-21):** the Run/Review-tab dialog described here is what
> was built first. §12.10 replaced it with a single-window work list and title page
> (`model/worklist*.py`); this section is kept as history.

New dialog, `model/library_sync.py` / `ui/library_sync.py`
(`LibrarySyncDialog`, reachable via a new Tools menu entry), following
`BatchExtractDialog`'s existing shape rather than inventing a new one:

- A source picker (registered `LibrarySource`s; JRiver connection
  reuses the existing `JRIVER_MCWS_CONNECTIONS` preference, same one
  the DSP-push feature already maintains) and a query/filter field.
  **As built (chunk 13, §11.3):** a **Source** combo over a per-kind
  settings page (Filesystem, JRiver); the JRiver page picks a server from
  the shared list. There is still no query field (neither built-in source
  takes a query); the browse node is chosen with a tree picker (chunk 14) or
  typed. The old hard-wired server/port/credentials fields are gone.
- **Run** tab: calls `run_library()` on a background `QRunnable` (same
  `QThreadPool` pattern as `ProbeJob`/`DesignJob`), streams
  `on_item_done` progress into the UI, switches to the **Review** tab
  (embedded `ReviewQueueDialog`, exactly as `BatchExtractDialog` does
  today, now carrying 3.1.2's metadata editor -- shared, not
  duplicated, so both this flow and the existing manual batch-design
  flow get it) when done.
- **Sync** action/button: calls `sync_library()` explicitly -- never
  auto-triggered after Run, per the review-gate decision. Also the
  natural place to finally wire `images_repo` into the GUI publish
  path, which `ReviewQueueDialog` doesn't do today (README: "XML-only
  from the dialog today").
- New preferences: xml repo / images repo local paths + owner/repo
  names (nothing persists these today -- `ReviewQueueDialog` asks via
  a file picker each time) and a `LIBRARY_SOURCE_DEFAULT` combo,
  following the existing `DESIGNER_QUEUE_DIR`/`DESIGNER_DEFAULT`
  pattern in `model/preferences.py`. **As built:** `LIBRARY_WORK_DIR`,
  `LIBRARY_XML_REPO`, `LIBRARY_IMAGES_REPO`, `LIBRARY_JRIVER_BROWSE_NODE`
  are defined and used; there is no `LIBRARY_SOURCE_DEFAULT`; the queue dir
  reuses `DESIGNER_QUEUE_DIR`. The image `owner`/`repo_name` prefs drafted
  here were **dropped**: both are optional on `sync_library()` and
  `push_image()` parses them from the images repo's git remote, so nothing
  needs them unless a remote is non-GitHub or a fork (then add them back).
  Sync results are split by `pipeline.review.split_publish_results()`:
  `'error'` entries (today only `project_conflict`) are excluded from the
  "Published N" count, listed in a warning dialog with a reviewer-facing
  explanation (`describe_publish_error()`), and the embedded review queue is
  refreshed. The standalone `ReviewQueueDialog` publish path reports the same
  way. Note that path still publishes XML-only and without `work_dir`, i.e.
  it does not read the `.beq` projects -- only the Library Sync dialog's
  Sync button does.
- **Filtering the library view.** The user's note (2026-09-18): once
  `source.list_items()` can return results at library scale (hundreds
  of titles), the Run tab needs a way to filter/narrow that list before
  running anything against it -- at minimum by idempotency status
  (e.g. "stale" -- a cached extraction/design whose source fingerprint
  or config no longer matches, per §4.1/§4.2 -- vs. "new"/never-processed
  vs. "up to date"), and by ordinary library metadata (name, year,
  content type, and whatever else turns out to matter). The exact field
  list and how "stale" gets computed/labelled for display is explicitly
  **deferred to UI design time** (chunk 10) rather than fixed here --
  this bullet exists so the requirement itself isn't lost, not to
  pre-specify the filter bar.
