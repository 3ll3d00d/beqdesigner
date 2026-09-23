# Work-list feedback remediation

> **Partly built.** This is §17 of the library sync plan. Chunk 45a has a
> partial implementation in `8abfa83`; 45b is not started; 45c is implemented in
> `0a0ee37`. It consolidates
> the reported work-list, stream-selection and project findings into three
> independently reviewable changes. §16/chunk 44 landed first.

## 17.1 Findings and scope

| Report | What the code currently does | Smallest correction |
|---|---|---|
| Retry is hard to understand | `retry_failed()` starts a design-through run; the title page calls its button “Retry failed extraction” for *either* failure stage. The indexed failure remains visible until the run finishes, while a new runtime row says queued/extract/design. The starting label says “Extract & design.” | Label retry by the failed stage and identify the old failure as previous while queued/running. Show the new attempt's queued/current/completed/failed result on both the row and open title page. Clear the prior failure presentation only after the index confirms success. Reuse the runtime state; do not add a second retry state store. |
| Design error is hard to read or copy | The Failures tab uses a table cell and tooltip; ordinary failed-result text is not copied into the selectable `resultDetails` pane. The title page shows short labels. `RunDetailsDialog` is already selectable and has **Copy all**, but only holds the current/most recent run. | Show the full persisted failure message, with preserved line breaks, in a read-only selectable view with **Copy** on the title page and Failures tab. Keep the one-line row summary. Reuse the existing Run Details view for live events rather than copying its event buffer into failure storage. |
| Decline reason appears twice | `state_text()` appends the index's “designer declined” detail to “Pending / waiting for decision”; `notice_text()` prints the decline again. The candidate commentary table is empty for a declined entry. | Keep the index detail on the work-list row, but display the decline once on the title page in a selectable **Designer commentary** area, with reason and message on separate lines. Keep the pending decision state concise. This is a presentation change; queue schema and decision state stay as they are. |
| Design reports Extracting | `plan_stages()` can plan design only, but `run_stages()` sends every machine title through `run_unit(..., through='extract')`; `_run_item()` emits “Extracting audio” even for a valid cache hit. | For a design-only plan, validate and load the existing extraction as `UnitWork` without announcing an extraction or occupying an extract slot. If the artifact is stale/missing, replan extraction explicitly and show why. A cache check is not an ffmpeg extraction. Preserve failure-memory, retry and season-group behavior. |
| JRiver stream choice asks for a rescan that cannot help | `JRiverLibrarySource._audio_stream_details()` parses `Audio Streams`, `Audio Codec`, and `Audio Channels`, but `FIELDS` requests none of them. `Audio Format` is an aggregate fallback, not a reliable choice list. `_audio_types()` already recognizes `TrueHD Atmos` when given the codec. `_choose_audio_stream()` raises before probing and logs an opaque id. | Request the JRiver per-stream fields, including language/title for readable choices. When a source still omits a usable list, lazily ffprobe that title's mapped local file on the worker pool and persist normalized audio-list ordinals through the existing index selection path. Show the title/path and actionable probe error in UI/logs. Keep the separate chunk 40 Playback Info selected-stream resolver distinct. |
| Extracted audio has one channel | `keep_multichannel` defaults to false. The pipeline intentionally always writes a mono design mix; it makes a second per-channel extraction only when that setting is on and the probed source has multiple channels. This report alone does not establish which setting or source stream was used. | Make the selected source stream, its channel count, and whether a multichannel copy will be kept visible before running and in Run Details. Verify the selected-stream path with a fixture. Fix a channel-count/extraction defect only if that test reproduces one; keep mono-only output when the setting is off. |
| No visible ffmpeg command history | `Executor.probe_file()` and `run_sync()` already emit command, output and exit events. The per-title `RunDetailsDialog` already shows them with selection and **Copy all**. Opening a title hides the list/footer/Details controls, making the view undiscoverable there. | Add an **Extraction / run details** action on the title page that opens the same dialog for that title. Show “cache hit; no ffmpeg command” where appropriate. Retain the current bounded, one-run event lifetime and credential redaction; do not build another command log. |
| Multichannel project lacks BM sum | Before 45c, `write_multichannel_project()` wrote flat `SingleChannelSignalData` entries and `read_project_filter()` assumed the first JSON item was a flat channel. | Done in `0a0ee37`: new projects use the app's bass-managed wrapper and LPF settings. The nested master holds the filter/hash; old flat projects remain readable. |

## 17.2 Chunks

### 45a — Retry, failure and title-page visibility

`8abfa83` shows the current attempt in the work-list Detail column until the
index refresh completes, rejects progress from titles outside the run, logs
extract/design exceptions with tracebacks, and emits an event when ffmpeg
command preparation fails. The following work remains:

- Make retry wording reflect extract versus design failure and show the prior failure separately from the active attempt on both table and title page. Use the existing run generation and index refresh; align with chunk 44's expiry of old runtime rows.
- Put a full, selectable, copyable failure view in the Failures tab/title page. Format the stage, title and multiline message without truncating the persisted reason or relying on a tooltip. Keep event output in the existing Run Details dialog and expose that dialog from the open title page, during and after the run. Say when a cache hit invoked no ffmpeg command.
- Render decline reason/message once as designer commentary; keep “Pending / waiting for decision” free of the repeated index detail.
- Cover retry after extract and design failures, a second failure, success, cancellation and disjoint subsequent runs in real-widget tests. Cover selection/copy of multiline failure text, title-page access to redacted ffmpeg argv/stdout/stderr, and a declined entry with no candidates. Update `docs/library/work.md` for the controls and history lifetime.

### 45b — Stream evidence and truthful stages

- Request JRiver's `Audio Streams`, `Audio Codec`, `Audio Channels`, `Audio Sample Rate`, `Audio Bitrate`, `Audio Language` and `Audio Title` fields; normalize aligned values into per-stream choices. The supplied six-stream example must list first-stream TrueHD Atmos/8 channels and preserve audio-list ordinals. If per-stream data is absent or incomplete, probe the mapped local source *on demand* without blocking the UI. Do not launch ffprobe for every library title at scan time. If no playable local source exists, explain that with the title/path instead of suggesting another ineffective rescan.
- Keep selection, cache invalidation and automatic audio-type updates in the existing index/queue path. Show the chosen stream and actual probed channel count alongside the `Keep multichannel` setting, and record extraction/cache steps in per-title events. Handle stale/missing design-only artifacts by explicitly switching to extraction; emit “Designing” without an “Extracting” phase on a true cache hit.
- Add JRiver mapping and fallback-probe tests, a GUI choice/failure test, and a short synthetic multistream/multichannel library extraction test. Check that the selected eight-channel stream yields a mono mix and, with `keep_multichannel=true`, eight per-channel arrays with the **same analysis sample rate and frame count**. Check mono-only behavior when the setting is off and status/event text for design-only cache hit versus fallback extraction. The fixture must prove a defect before changing ffmpeg channel mapping.

### 45c — Bass-managed multichannel project

**Implemented in `0a0ee37`.** New multichannel projects serialize one
`BassManagedSignalData` with linked channels and the app's LPF defaults or
the interactive caller's saved LPF settings. `read_project_filter()` reads
the nested master or an old flat master. A saved human edit is not overwritten;
publishing uses that edit and aligns the other project. The real main window
loads the composite and exposes its summed signal.

- Serialize a `BassManagedSignalData` composite for newly generated multichannel projects, using the same LPF frequency/position and channel naming conventions as the interactive loader. Keep the linked channel filters, including LFE, and make the composite sum visible when opened in the app.
- Update project filter/hash lookup for nested master channels and legacy flat files, including the existing “human edited since design” and publish conflict behavior. Do not rewrite human-edited projects as part of this migration.
- Add project round-trip tests through `signalmodel_from_json()`, filter-link and summed-track checks, pure/edited hash tests, and a publish test showing that a saved human edit still wins. Include a small real-widget open-project assertion if the model round trip alone does not establish chart/table visibility.

## 17.3 Completion rule

Each chunk needs its focused automated tests and the broader relevant suite when practical, with exact commands/results recorded. Update the plan index with its commit hash after each implementation commit. Do not mark a report fixed solely from a changed label: the audio fixture, retry lifecycle and project round trip above are the acceptance evidence.

45c verification (`UV_CACHE_DIR=/tmp/beqdesigner-uv-cache`,
`PYTHONPATH=./src/main/python`, `QT_QPA_PLATFORM=offscreen` for all commands):

- `uv run pytest src/test/python/test_pipeline_publish_project.py::test_write_multichannel_project_round_trips_the_sum_links_and_filter src/test/python/test_pipeline_publish_project.py::test_read_project_filter_still_reads_legacy_flat_multichannel_projects src/test/python/test_pipeline_publish_project.py::test_app_resave_of_bass_managed_project_keeps_the_edit_and_blocks_overwrite src/test/python/test_pipeline_review.py::test_publish_reviewed_queue_uses_a_saved_bass_managed_project_edit src/test/python/gui/test_worklist_projects.py::test_the_real_main_window_opens_a_bass_managed_multichannel_project -q` — 5 passed.
- `uv run pytest src/test/python/test_pipeline_publish_project.py src/test/python/test_pipeline_review.py src/test/python/gui/test_worklist_projects.py src/test/python/test_pipeline_library_stages.py -q -k 'not test_design_and_queue_threads_channels_to_the_request and not test_publish_reviewed_queue_xml_only and not test_publish_reviewed_queue_defaults_gain_from_chosen_candidates_mv_adjust_db and not test_publish_reviewed_queue_without_work_dir_uses_apply_reviewed_entry_as_before and not test_publish_reviewed_queue_reads_the_edited_mono_project_when_work_dir_given'` — 109 passed, 5 existing failures from the JSON-output migration or mismatched channel fixture deselected.
- `uv run pytest src/test/python/gui/test_batch_extract_design.py::test_design_session_uses_the_current_bass_management_settings src/test/python/gui/test_extract_design.py::test_design_session_uses_the_current_bass_management_settings -q` — 2 passed.
- `uv run pytest src/test/python/test_pipeline_orchestrate.py -q` — 17 passed; one existing XML-output assertion failed (`test_publish_without_an_image_only_pushes_xml`). A broader GUI extraction run could not complete: `AudioExtractor` and `Executor.run_sync()` both bind ffmpeg progress port 12001, causing `Address already in use` and a timeout in `test_extract_with_design_disabled_does_not_require_a_queue_dir`, even with local socket access. This path is outside 45c.
