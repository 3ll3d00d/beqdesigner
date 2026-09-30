# Pipeline service — completed delivery record

Archived on 2026-09-30. This is historical delivery and test evidence.
This supplements the original plan with later delivery evidence, including
run joining (F5) and the local image smoke run; it is not a second design.

Current work is in [TODO](../../TODO.md); current architecture is in
[the service design](../../pipeline-service.md).

## 11. Status

| Chunk | Content | Status |
|---|---|---|
| S0 | Qt-free extraction path (docker.md §10.1) | Built: `55c3425`, `5d136d3`, `579f542`, `4929e93`, `f9d66e5`; port fix `a217bec` |
| S1 | `Selection.kind`/`year`, shared `year.py`, index SQL, CLI `--kind`/`--year` (§3) | Built: `ceabb76` |
| S2 | Config, per-job profile context, `JobManager`, history, lease (§4, §5) | Built: `8f6ff97`, `bc7afd3`, `ca43161`, `cb9f371` |
| S3 | FastAPI app, models, routes, auth, SSE, committed OpenAPI document, user page (§6) | Built: `99d2071` |
| S4 | Auto scheduler and `/v1/schedule` (§8) | Built: `df93b2a` |
| S5 | Docker image, compose example, CI smoke, GHCR publish on tag (docker.md §10) | Built: `80cab0d`, local smoke fixture `622d197`; image build and smoke verified locally at `7db992d`; arm64 and GHCR publish open as [C1](../../TODO.md#c1--arm64-image-and-ghcr-publish) |
| S6 | Notifications (§9) | Built: `af261fc` |
| S7 | README and implemented-design entries | Built: `faafbba`, `73ccfde` |
| F5 | Every run holds the lease; joining a run in progress (§5.1) | Built: `3d38f76`, `ae1a086`, `dc527f5`, `12d875e` |
| -- | Review Folder's Publish/Commit honour the lease | Open: [W3](../worklist-w3.md) |

Tests: `test_pipeline_library_selection.py`, `test_pipeline_library_year.py`,
`test_pipeline_service_*.py`, `test_pipeline_library_inbox.py`, the joining
tests in `test_pipeline_library_stages.py`/`test_pipeline_library_cli.py`, and
the lease tests in `gui/test_worklist_actions.py`.

