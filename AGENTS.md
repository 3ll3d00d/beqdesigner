# BEQDesigner — codebase orientation

A PyQt6 desktop app for designing, analysing and applying **Bass EQ (BEQ)**
filters for movie soundtracks. BEQ recovers low frequency content that is 
discernably present in the track but rolled off; BEQDesigner is where those 
filters get designed, verified against the actual audio, and pushed out to 
whatever DSP platform will apply them.

This file is the map. Read it before making a change that crosses module
boundaries — the app is a single process with a lot of shared mutable model
state wired together by Qt signals, and the seams are not always where you'd
expect.

- Developer setup, build and Qt Designer workflow: `readme.md`.
- End-user documentation (what BEQ is, how each dialog is meant to be used):
  `docs/`, published at <https://beqdesigner.readthedocs.io/>.
- The JRiver Media Center integration has its own, much deeper guide:
  **`src/main/python/model/jriver/AGENTS.md`** — read that before touching
  anything under `model/jriver`.

## The mental model: signals and filters

Two domain objects account for most of the app.

- A **signal** is a mono audio track — one channel of a multichannel movie
  soundtrack, or a bass-managed mixdown of all of them. It is analysed into
  frequency-domain curves (average, peak, median).
- A **filter** is a change in frequency response, expressed as IIR biquads of
  the types real DSP hardware supports.

Everything else is either a way of *getting* signals (extraction from movie
files), a way of *seeing* what a filter does to a signal (the charts,
spectrum, waveform), or a way of *getting the filter out* to a device
(minidsp XML, JRiver `.dsp`, HTP-1 JSON, Equalizer APO, raw biquads).

The main window is a live overlay of both: signal curves and filter response
on one magnitude chart, so you can see the corrected response as you design.

## Layout and layering

Source root is **`src/main/python`** — not the repo root. Run everything with
`PYTHONPATH=./src/main/python`.

| Path | Contents |
|---|---|
| `app.py` | `BeqDesigner` main window: menu wiring, the main magnitude chart, signal/filter tables, presets, and every "open tool X" entry point. Also `SaveChartDialog` and `ExportBiquadDialog`. |
| `model/` | All non-trivial logic. Despite the name it contains dialogs too (see below). |
| `model/jriver/` | Self-contained JRiver MC DSP integration; has its own `AGENTS.md`. |
| `acoustics/` | Vendored DSP helpers — fractional-octave band definitions (IEC 61260) and nominal centre frequencies (IEC 61672). Third-party, own LICENSE; don't restyle it. |
| `ui/` | Qt Designer `.ui` files **and their generated `.py`**. Generated files are checked in and must never be hand-edited. `ui/delegates.py` and `ui/drop.py` are hand-written helpers that happen to live here. |
| `mpl.py`, `svg.py`, `style/mpl/` | Matplotlib canvas widget embedded in Qt, clickable SVG view (used for the JRiver filter-pipeline graph), and the `beq_dark` mpl themes. |
| `src/test/python/` | pytest suite. `scratch/` and `mds/` are exploratory scripts, not tests. |
| `src/main/xml/` | Bundled defaults — per-MC-version JRiver configs, a flat 24HD minidsp config. |

**`model/` is not a pure domain layer.** Most feature dialogs live in
`model/*.py` alongside their logic (`model/extract.py` holds
`ExtractAudioDialog`, `model/merge.py` holds `MergeFiltersDialog`, and so
on), while `ui/*.py` holds only the generated form classes they inherit
from. So `model.report.SaveReportDialog` is the report *feature*; the
`ui.report.Ui_saveReportDialog` it mixes in is just the widget tree. The
genuinely UI-free modules are `iir.py`, `xy.py`, `codec.py`, `minidsp.py`,
`limits.py` (mostly), `acoustics/`, and most of `model/jriver/`.

## Core objects

### Filters — `model/iir.py`

The filter hierarchy, all built on `SOS` → `Biquad`. Coefficients follow the
RBJ Audio EQ Cookbook. Everything is computed in float64 regardless of what
the target hardware does.

- Atomic: `PeakingEQ`, `LowShelf`/`HighShelf`, `FirstOrder_`/`SecondOrder_`
  `LowPass`/`HighPass`, `AllPass`, `Gain`, `LinkwitzTransform`,
  `Passthrough`.
- `ComplexFilter` — a filter made of other filters; `CompoundPassFilter`
  (`ComplexLowPass`/`ComplexHighPass`) synthesises Butterworth/Linkwitz-Riley
  pass filters of arbitrary order out of cascaded biquads.
- `CompleteFilter` — the "filter set applied to one signal"; this is what a
  signal owns and what gets serialised.
- Shelf filters carry both Q and S conventions; `q_to_s`/`s_to_q` convert
  between them and several export targets care which one they get.
- `as_equalizer_apo`/`from_equalizer_apo` are the Equalizer APO text format.

### Signals — `model/signal.py`

- `Signal` — the raw samples plus lazily computed `avg`, `peak`, `median`
  curves (Welch / spectrogram-max, per `docs/concepts.md`). Long signals are
  sliced into segments internally.
- `SignalData` → `SingleChannelSignalData` (one channel + its
  `CompleteFilter` + cached magnitude curves per smoothing level) and
  `BassManagedSignalData` (a composite of channels summed with LFE gain and
  a bass-management LPF).
- `SignalModel` — the collection the main window binds to; also owns
  **master/slave linking** (`model/link.py`), where one signal owns a filter
  and others reuse it.
- Loaders: `AutoWavLoader`/`DialogWavLoaderBridge` (wav/flac),
  `FrdLoader` (text frequency-response files), `PulseLoader` (synthetic).

### Charts — `model/magnitude.py`, `model/limits.py`, `model/xy.py`

`MagnitudeModel` is the shared chart abstraction, used by the main window and
by nearly every dialog with a preview. You give it a matplotlib canvas and a
**data provider callable** returning a list of `MagnitudeData`; it manages
curves, an optional secondary (twinned) y-axis, the legend, normalisation
against a reference curve, and delegates axis ranges to `Limits`. If you are
adding a chart, use it rather than driving matplotlib directly.

`model/xy.py` holds the value objects (`MagnitudeData`, `ComplexData`) and
the smoothing implementations (fractional-octave via `acoustics/`, and
Savitzky-Golay).

## What the app does

### Main window (`app.py`)

Signal table, filter table, magnitude chart, and a linked waveform view.
Beyond add/edit/delete of signals and filters:

- **Visibility controls** — which signal curves (avg/peak/median, filtered
  and/or unfiltered) and which filters (individual/combined/none) are drawn.
- **Smoothing** — 1/1 to 1/48 octave, applied per signal and cached.
- **Normalisation** — pick any curve as a reference and plot everything
  relative to it; plus a "tilt" toggle.
- **Filter presets** — three slots persisted in `QSettings`
  (`FILTERS_PRESET_x`) for stashing and reapplying a filter set.
- **Projects** — the whole signal model (signals, filters, links, cached
  curves) round-tripped through `model/codec.py` as gzipped JSON, saved as
  `.beq`. Signals (`.signal`) and filters (`.filter`) can also be saved
  individually.
- **Waveform** (`model/waveform.py`) — time-domain view under the chart,
  linked to the spectrum, with per-region stats.

### Getting audio in — ffmpeg (`model/ffmpeg.py`, `model/extract.py`, `model/batch.py`)

`ffmpeg`/`ffprobe` are **external binaries located on `PATH` or via
Preferences**; they are not bundled. `find_missing_ffmpeg_tools` and
`describe_missing_binary` produce the user-facing "can't find ffmpeg" error,
and `BeqDesigner.__check_ffmpeg_available` gates the tools that need them.

`Executor` builds the ffmpeg command line (channel splitting, mono mixdown
with LFE offset, decimation to a low `fs` for analysis, optional bass
management, compression format). `AudioExtractor` runs it off the UI thread
and `FfmpegProgressBridge` reads ffmpeg's progress reports over a local UDP
socket to drive the progress bar.

Three entry points share this machinery:

- **Extract Audio** — pull channels out of one movie file as wavs to analyse.
  Also has an optional "Design filters?" step (reuses `model/batch.py`'s
  `DesignJob`), for going straight from one file to a queued design without
  the batch dialog.
- **Remux Audio** (`ExtractAudioDialog(is_remux=True)`) — same dialog, but
  applies the designed filters and writes a new video file with the BEQ'ed
  audio track, optionally alongside the original. No design step here --
  remux applies a filter someone already designed/reviewed.
- **Batch Extract / Design** (`model/batch.py`) — glob for files, probe them
  in the thread pool, extract en masse; optionally also designs each one
  (`pipeline.review.design_and_queue()`) and reviews the results on an
  embedded tab -- see `pipeline/README.md`'s "Batch design + review".

### Library work list (`model/worklist.py`, `model/worklist_model.py`, `model/worklist_profile.py`)

Tools > **Library Work List**: a top-level `QMainWindow` over the library-sync discovery index
(`pipeline/library/index.py`) -- the pipeline strip with a count per kind of work, a searchable table of every title and
what it needs next, a Rescan that lists the sources again on a `QRunnable`. It is the **intended replacement for
`LibrarySyncDialog`** (`model/library_sync.py`, still there and working until chunk 27c) and is **read-only for now**:
it runs, accepts, publishes and commits nothing until chunk 26b. Everything it shows is read from the index rows, not
derived again. Until the settings editor (26c) it builds its profile from the Library Sync preferences
(`worklist_profile.load_setup`), or reads the file named by `LIBRARY_PROFILE_PATH`. The plan is
`design/library-sync-pipeline-plan.md` (chunk table, §8) and `design/library-sync/workflow-rework/`. Tests:
`src/test/python/gui/test_worklist_*.py`, over a fixture index (`gui/worklist_fixture.py`).

### Analysis (`model/analysis.py`)

Standalone spectrum/waveform analyser over a wav: `MaxSpectrumByTime`
(where the heavy hits are, in time × frequency — point, ellipse or
spectrogram rendering) and `Waveform`.

### Getting filters out

| Target | Where | Format |
|---|---|---|
| minidsp 2x4 / 2x4HD / 10x10 / SHD / 88BM / HTx | `model/minidsp.py` (`XmlParser` subclasses) | device XML |
| minidsp, live | `model/minidsp.py` `FilterPublisher` | shells out to `minidsp-rs` |
| Monoprice HTP-1 | `model/sync.py` (`SyncHTP1Dialog`, `HTP1Parser`) | JSON over a WebSocket to the device |
| JRiver Media Center | `model/jriver/` | `.dsp` file or MCWS HTTP push |
| Equalizer APO | `model/iir.py` | config text |
| Raw biquads | `app.py` `ExportBiquadDialog` | float or hex (32-bit IEEE / Q5.23), per-device slot limits |
| FRD / wav | `model/export.py` | text response / audio |
| Report image | `model/report.py` | composed chart + filter table + artwork |

`model/merge.py` batch-converts a directory of BEQ filter files into device
configs; its `DspType` enum is the single source of truth for each device's
sample rate, biquad slot count, channel names and fixed-point-ness.

### BEQ catalogue (`model/catalogue.py`)

Downloads and caches the community BEQ database (`database.json`), presents
a searchable browser with artwork and metadata, and can load an entry's
filters straight into the current signal or push them to a connected minidsp.
`CatalogueEntry` tolerates messy upstream data by design — bad `year`/
`runtime` values are logged and defaulted, not raised.

### AVS post builder (`model/postbuilder.py`)

Generates the forum post + XML attachment used to share a BEQ on avsforum.

### Version checking (`model/checker.py`)

`VersionChecker` polls GitHub releases in the background on startup (opt-in
via preferences, with a separate beta channel) and `ReleaseNotesDialog`
renders the notes.

## Cross-cutting conventions

- **Qt via `qtpy`.** `app.py` sets `QT_API=pyqt6` before any Qt import.
  Import from `qtpy.*`, never `PyQt6.*`.
- **Lazy imports are deliberate.** Roughly two thirds of `app.py`'s
  `model`/`ui` imports sit inside the method that needs them, to keep cold
  start acceptable. Don't hoist them to the top of the file.
- **Background work is `QRunnable` on `QThreadPool.globalInstance()`**, with
  a paired `QObject` signals class (`FileSearchSignals`, `ProbeJobSignals`,
  `JobSignals`, `VersionSignals`, …) to marshal results back. That pairing
  is the pattern — follow it rather than inventing new threading.
- **Preferences** are a thin typed wrapper over `QSettings`
  (`model/preferences.py`). Every key is a module-level `GROUP/name`
  constant; add new settings as constants there with a default, and to
  `PreferencesDialog` if the user should see them. Window geometry is
  persisted the same way.
- **Logging** goes to an in-memory `RollingLogger` (`model/log.py`) exposed
  via *Help → Logs*. There is no log file — run from a terminal to see
  tracebacks.
- **Generated UI.** `.ui` → `.py` with `pyuic6` (`ui/convert.sh`). Edit the
  `.ui`, regenerate, commit both.
- **Version** comes from `src/main/python/VERSION`, written by CI; from
  source it falls back to `0.0.0-alpha.1`.
- Optional external tools: `ffmpeg`/`ffprobe` (extraction, remux, analysis),
  `graphviz` (JRiver filter pipeline rendering), `minidsp-rs` (pushing to a
  connected minidsp). All are degraded-gracefully, not hard dependencies.

## Working from a plan

Multi-commit work is tracked in a plan under `design/` (for example
`design/library-sync-pipeline-plan.md`). A large plan is an **index plus topic
files**: the index (that file) holds the goal, the chunk-status table and a map
from each `§` to the file that holds it, and the design lives in topic files
under a same-named folder (`design/library-sync/`). Read the index first, then
only the one topic file you need; keep every file under ~500 lines and split
rather than append. Keep the plan accurate as you go:

- **After every commit, update the active plan with the new status** --
  before starting the next piece of work. Mark the chunk or item done with
  the commit hash, and record anything the commit changed about the design
  (a deviation, a newly found gap, a follow-up). If the plan lists the item
  as an open gap, remove or reword it.
- Put the plan update **in the same commit** when you can (amend before
  pushing); otherwise make it a separate commit straight after. Never leave
  a commit whose effect the plan still describes as unbuilt.
- The plan is descriptive of the code at `HEAD`, not aspirational: when the
  code and the plan disagree, verify against the code and correct the plan.
  Agreed forward design is allowed, but it lives in its own file, is labelled
  "design, not built", and is marked "Not started" in the chunk table, so the
  status of built code stays trustworthy.
- If a task isn't covered by a plan, there is nothing to update -- don't
  create one just for a small fix.

## Testing

```sh
PYTHONPATH=./src/main/python uv run pytest --cov=./src/main/python
```

Coverage is concentrated where the formats are: `test_iir.py` (filter
maths), `test_minidsp.py` (device XML round trips, with real
`MiniDSP-*.xml` fixtures and `expected_output_*.xml` goldens),
`test_codec.py` (project JSON), `test_ffmpeg.py` (command construction),
`test_link.py`, and the JRiver suite (`test_dsp_roundtrip.py`,
`test_xo.py`, `test_mcws.py`). The dialogs were largely untested — if you
change behaviour, prefer pushing it down into a testable model class over
leaving it in the dialog, but a real dialog can now be tested directly too
(see below).

`src/main/python/pipeline/` (tested by `src/test/python/test_pipeline_*.py`)
is a separate, Qt-free headless pipeline mirroring parts of the app for
API/scripted use — see `src/main/python/pipeline/README.md` for the
architecture, and `design/api-headless-pipeline.md`/
`design/pipeline-implementation-plan.md`/`design/designer-interface.md` for
the design rationale and contract behind it.

**Testing real dialogs — `pytest-qt`.** `src/test/python/gui/` constructs
actual `QDialog`/`QWidget` subclasses under `pytest-qt`'s `qtbot` fixture
and drives them like a user would (`qtbot.mouseClick`, `.setText(...)`,
etc.) — see `test_postbuilder_dialog.py`, `test_filter_dialog.py`,
`test_report_dialog.py`, `test_ffmpeg_execute.py` (that last one drives
the `QThreadPool`-backed async path via `qtbot.waitUntil` on a progress
signal, not a widget). This needs a real Qt platform plugin; set
`QT_QPA_PLATFORM=offscreen` to run without any display server (CI,
containers) — no Xvfb needed. CI's pytest step already sets it
(`.github/workflows/test.yaml`); locally:

```sh
PYTHONPATH=./src/main/python QT_QPA_PLATFORM=offscreen uv run pytest src/test/python
```

Three gotchas hit writing these, all worth knowing before adding more:

1. `qtbot` constructs a real, process-wide `QApplication` that then
   persists for the rest of the test process. Any test elsewhere in the
   suite that asserts `QApplication.instance() is None` (the pipeline's
   Qt-boundary tests) must run in a subprocess, not in-process, or it
   breaks depending on test order/selection — see
   `test_pipeline_qt_boundary.py`'s docstring for why, and
   `test_pipeline_publish_report.py`/`test_pipeline_acceptance.py`'s
   `*_constructs_no_qapplication` tests for the pattern.
2. Use a real `Preferences` backed by a temp-file `QSettings(path,
   QSettings.Format.IniFormat)` in these tests, never the app's real
   `QSettings("3ll3d00d", "beqdesigner")` — see
   `test_postbuilder_dialog.py::_make_preferences`.
3. `ui/beq.py` and `app.py` have a genuine, pre-existing mid-file circular
   import (`ui/beq.py` does `from app import PlotWidgetWithDateAxis` at
   module top level partway through the generated file; `app.py` does
   `from ui.beq import Ui_MainWindow` near its own top). It only resolves
   if `ui.beq`'s own `Ui_MainWindow` — defined earlier in that same file —
   is already cached in `sys.modules` by the time `app.py`'s import runs,
   which is true when the real app launches `app.py` as `__main__` first,
   but *not* true if a test does `from model.filter import ...` or
   `from model.report import ...` cold. Fix: `import ui.beq` (not
   `import app`) as the first import in any test file that transitively
   needs `model.filter` or `model.report` — see `test_filter_dialog.py`/
   `test_report_dialog.py`'s top-of-file comment.

CI (`.github/workflows/test.yaml`) runs the suite on Linux, macOS and
Windows and builds the PyInstaller bundle;
`create-app.yaml` produces release binaries.
