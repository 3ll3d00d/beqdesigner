# Pipeline service — Docker image and the Qt-free boundary

Part of the [pipeline service design](../pipeline-service.md); section numbers
continue that file's. This describes the image as built. The amd64 build and
smoke run pass; the arm64 build and GHCR publish have not yet run
([C1](../outstanding.md#c1--arm64-image-and-ghcr-publish)).

## 10. Docker image

- **Files:** `docker/Dockerfile`, `docker/compose.example.yaml`,
  `docker/service.example.yaml`, `docker/smoke.py`, `.dockerignore`.
- **Base:** `python:3.13-slim` (the project's `requires-python`), with `ffmpeg`
  from Debian (including the `dvdvideo` demuxer the README asks for DVDs; the
  build asserts `ffmpeg -demuxers` lists it), `git` and `openssh-client`. No
  graphviz and no Qt or its X/GL libraries (§10.1).
- **Install:** multi-stage. A pinned uv binary runs
  `uv sync --frozen --no-dev --group service` into `/opt/venv`; the source is
  copied to `/app/src/main/python` with `PYTHONPATH` set to it. Swagger UI and
  ReDoc are fetched as pinned npm packages and served locally
  (`BEQ_SERVICE_STATIC`), so the try-it-out page works without internet access.
  The runtime stage has no uv.
- **Dependency groups:** the desktop's Qt packages are the default `desktop`
  group in `pyproject.toml`, so a normal `uv sync` installs the app, while the
  image installs only the runtime list and the `service` group.
- **User:** a non-root `beq` user; `PUID`/`PGID` build args (or `user:` in
  compose) so files written to the mounts belong to the host user.
- **Volumes:** `/config` (ro: `profile.yaml` and `service.yaml`), `/work` (work
  directory, index, service state), `/queue` (review queue), the media mounts at
  the paths the profile names (ro), and, only when repository writes are
  enabled, the repository clones and `/home/beq/.ssh` (ro: key and
  `known_hosts`) -- git uses the credentials mounted for the container user.
  Git identity comes from `GIT_AUTHOR_NAME`/`GIT_AUTHOR_EMAIL`/`GIT_COMMITTER_*`.
  The compose example shows the mounts and the token as a Docker secret.
- **Entrypoint:** `python -m pipeline.service --profile /config/profile.yaml
  --service-config /config/service.yaml`; `HEALTHCHECK` requests `/health`.
- **Smoke test:** `docker/smoke.py` starts the built image against a fixture
  profile with a filesystem source over a short synthetic six-channel WAV and
  a stub HTTP designer on the host, waits for `/ready`, submits a run of that
  title through `design`, polls the job to `succeeded` and asserts a queue
  entry exists. It runs in `.github/workflows/test.yaml` on every push (Linux
  only) and in `create-image.yaml` before publishing.
  `test_pipeline_service_docker.py` loads `docker/smoke.py` by path and runs
  the same fixture through the real local service, ffmpeg and HTTP designer
  without Docker.
- **Publishing:** on a pushed tag (the trigger `create-app.yaml` uses for the
  desktop release), `create-image.yaml` builds `linux/amd64` and `linux/arm64`,
  runs the smoke test, and pushes `ghcr.io/3ll3d00d/beqdesigner-pipeline:<tag>`
  with `GITHUB_TOKEN` (`packages: write`). `latest` is applied only to a tag
  without a pre-release suffix (`-alpha`/`-beta`/`-rc`). The image carries OCI
  labels (source, revision, version), and `src/main/python/VERSION` is written
  before the build so `/health` reports the release.

## 10.1 The Qt-free boundary

Nothing under `pipeline/` reaches `qtpy`, `PyQt6`, `qtawesome`, `pyqtgraph` or
`ui.*`, even indirectly. The `model/` modules the pipeline uses keep a Qt-free
core, with their Qt halves in sibling modules that import it:

| Qt-free core | Qt half |
|---|---|
| `model/preferences.py` | `model/preferences_dialog.py` |
| `model/limits.py` | `model/limits_dialog.py` |
| `model/minidsp.py` | `model/minidsp_qt.py` |
| `model/ffmpeg.py` (`Executor`, `run_sync()`) | `model/ffmpeg_qt.py` (`AudioExtractor`, which `Executor.execute()` imports for the GUI) |
| `model/signal.py` (`Signal`, `SignalData`, `AutoWavLoader`) | `model/signal_qt.py` (table models, dialogs, their loaders, the smoother) |
| `model/dsp_type.py` (a plain enum) | re-exported by `model/merge.py` |

`model.magnitude` imports its dialogs only where it shows them.
`model.xy.interp()` called without `smooth=` reads the desktop's smooth-graphs
preference; with no Qt available it uses that preference's default, so a
container run matches a desktop left at the default.

The ffmpeg progress bridge's UDP port is picked by the operating system, so two processes
extracting at once do not collide.

**Tests:** `test_qt_free_modules.py` imports every `pipeline/` module, and the
`model/` modules it uses, in a fresh interpreter with Qt blocked by a
`sys.meta_path` finder, and completes a whole `Session` run there;
`test_pipeline_qt_boundary.py` scans imports and checks no `QApplication` is
constructed. A module the pipeline starts to use is added to
`test_qt_free_modules.py`'s list.
