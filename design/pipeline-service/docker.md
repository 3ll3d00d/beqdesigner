# Pipeline service — Docker image and the Qt-free extraction path

**Status: design, not built.** Part of the [pipeline service design](../pipeline-service.md);
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
