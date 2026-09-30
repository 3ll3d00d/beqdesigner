# Pipeline service — Docker image and the Qt-free extraction path

> **Historical archive:** original path `design/pipeline-service/docker.md` before the 2026-09-27 sweep, with the S0/S5 "As built" notes. Not the current status; see [docker](../../pipeline-service.md#10-docker-image).

**Status: S5 implemented in `80cab0d`; CI image smoke has not run yet.** Part of the [pipeline service design](plan.md);
section numbers continue that file's.

## 10. Docker image (chunk S5)

- **Base:** `python:3.13-slim` (the project's `requires-python`), with
  `ffmpeg` from Debian (it includes the `dvdvideo` demuxer the README asks
  for DVDs; the build asserts `ffmpeg -demuxers` lists it) and `git`,
  `openssh-client`. `graphviz` is not needed.
- **No Qt:** chunk S0 (§10.1 below) comes first, so the image installs neither
  PyQt6 nor its X/GL libraries.
- **Install:** `uv sync --frozen --no-dev --group service` into
  `/opt/venv`; the source is copied to `/app/src/main/python` and
  `PYTHONPATH` set to it, as elsewhere. Multi-stage: build with uv, run
  without it.
- **User:** a non-root `beq` user; `PUID`/`PGID` build args (or `user:` in
  compose) so files written to the mounts belong to the host user.
- **Volumes:** `/config` (ro: the two YAML files), `/work` (work directory,
  index, service state), `/queue` (review queue), the media mounts at the
  paths the profile names (ro), and, only when repository writes are
  enabled, the repository clones and `/home/beq/.ssh` (ro: key and
  `known_hosts`; D5's "the invoking user's git credentials" becomes "the
  credentials mounted for the container user"). Git identity comes from
  `GIT_AUTHOR_NAME`/`GIT_AUTHOR_EMAIL`/`GIT_COMMITTER_*`.
- **Entrypoint:** `python -m pipeline.service --profile /config/profile.yaml
  --service-config /config/service.yaml`; `HEALTHCHECK` curls `/health`.
- **Files:** `docker/Dockerfile`, `docker/compose.example.yaml`,
  `docker/service.example.yaml`, `.dockerignore`.
- **CI smoke test:** a job in `.github/workflows/test.yaml` builds the image
  (Linux only), starts it against a fixture profile with a filesystem source
  over a short synthetic multichannel file and a stub HTTP designer container,
  waits for `/ready`, submits a run filtered to that title through `design`,
  polls the job to `succeeded`, and asserts a queue entry exists. It runs on
  every push, so a broken image is found before a tag.
- **Publishing (decided, main file §12):** built and pushed to GHCR by GitHub Actions
  when a tag is pushed -- the trigger `create-app.yaml` already uses for the
  desktop release. A new job (in `create-app.yaml`, or a sibling
  `create-image.yaml` with the same `on: push: tags`) logs in with
  `GITHUB_TOKEN` (`packages: write`), builds with `docker/build-push-action`
  for `linux/amd64` and `linux/arm64` (a NAS is often ARM), and tags
  `ghcr.io/3ll3d00d/beqdesigner-pipeline:<tag>`; `latest` only for a tag that
  is not a pre-release (the desktop releases are created as pre-releases,
  so the rule is written against the tag name, e.g. no `-alpha`/`-beta`/`-rc`
  suffix). The image carries OCI labels (source, revision, version) and
  `src/main/python/VERSION` is written before the build, as for the app, so
  `/health` reports the release. The smoke test runs against the built image
  before it is pushed.

**As built (S5).** `docker/Dockerfile` uses a pinned uv binary and pinned npm
Swagger UI/ReDoc packages; the runtime image has ffmpeg, git, SSH and the
service dependency group, without Qt. The desktop Qt packages moved to a
default `desktop` group, so normal `uv sync` retains the app. The image runs
as a non-root user; `docker/compose.example.yaml` shows the mounts and token
secret. `docker/smoke.py` exercises a built image with a six-channel WAV,
stub HTTP designer on the host and a review queue assertion in both push CI
and the tag workflow. The tag workflow publishes amd64 and arm64 to GHCR
after the smoke run, with `latest` only for non-prerelease tags.
`uv lock --check --offline`, the focused S5 tests (34) and the full suite
(2310) passed. Docker is not installed in the local environment, so the
image build and smoke run await the first CI execution.
The identical six-channel fixture was also run through the real local service,
ffmpeg and HTTP designer in `622d197`; its job succeeded and produced a queue
entry. The focused test and full suite (2331 tests) passed. This verifies the
fixture and pipeline behavior, while the image build itself still awaits CI.

### 10.1 Chunk S0: a Qt-free extraction path

`pipeline/` never imports `qtpy`, but it reaches `model.ffmpeg.Executor` and
`model.signal.AutoWavLoader`, whose modules import `qtpy.QtWidgets` at the top
for dialogs and table models the pipeline never uses (README "Architecture
rules" 2, the one deliberate exception). S0 removes the exception:

- Move `Executor` and the command construction it needs (and anything it
  imports from `model.preferences`, which is also Qt-bound) into a Qt-free
  module, e.g. `model/ffmpeg_core.py`; `model/ffmpeg.py` re-exports it so the
  app's imports and `test_ffmpeg.py` are unchanged.
- Likewise move `AutoWavLoader`'s wav/flac reading and `SingleChannelSignalData`'s
  non-Qt core out of `model/signal.py` (whose `SignalModel` is a
  `QAbstractTableModel` and stays), or give the pipeline its own loader over
  `soundfile` that produces the same `Signal`; the choice is made by what
  `model/signal.py`'s Qt uses actually entangle, found in the chunk.
- Check the remaining `model.*` imports of `pipeline/` (`model.bdmv`,
  `model.dvd`, `model.iir`, `model.xy`, `model.codec`, `model.merge`,
  `model.minidsp`, `model.execution_events`, `model.magnitude` via
  `model.signal`) for transitive Qt and fix the same way.
- **Tests:** extend `test_pipeline_qt_boundary.py` with a subprocess test
  that blocks `qtpy` and `PyQt6` (a `sys.meta_path` finder that raises
  `ImportError`) and then imports `pipeline.orchestrate`,
  `pipeline.library.stages` and runs a short synthetic extract and design; the
  existing boundary and ffmpeg/signal tests keep passing unchanged, and the
  library extraction/design parity regression test (AGENTS.md) still passes.
- Afterwards README rule 2 loses its exception, and `pyqt6`/`qtpy` stay out
  of the image's dependency set (S5 installs the `service` group plus a
  `pipeline` subset; the split of `pyproject.toml`'s runtime list into what
  the pipeline needs and what only the app needs is part of S5).

**As built (S0 done).** Qt halves split out, each re-importing the Qt-free
module: `model/preferences_dialog.py`, `model/limits_dialog.py`,
`model/dsp_type.py` (a plain enum, re-exported by `model/merge.py`),
`model/minidsp_qt.py`, `model/ffmpeg_qt.py` (`Executor.execute()` imports
`AudioExtractor` there, the GUI's path; `run_sync()` needs no Qt) and
`model/signal_qt.py` (table models, dialogs, their loaders, the smoother).
`model.magnitude` imports its dialogs where it shows them. The whole of
`pipeline/` imports, and a whole `Session` run completes, with Qt blocked
(`test_qt_free_modules.py`). One behavior found by that run:
`model.xy.interp()` reads the desktop's smooth-graphs preference when called
without `smooth=`; with no Qt it now uses the preference's default, so a
container run matches a desktop whose setting is the default (in the app and
the CLI the preference is still read, as before). The dependency split in
`pyproject.toml` is left to S5. Found and fixed on the way (own commits):
the ffmpeg progress port counted up from 12000 in every process, so two
processes collided (`a217bec`).
