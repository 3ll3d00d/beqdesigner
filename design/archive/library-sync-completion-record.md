# Historical consolidated status record

Archived on 2026-09-30. Status rows and maintenance instructions below are
a snapshot, not current work tracking. Use [TODO](../TODO.md) for all open work
and the [design index](../README.md) for current design references.

# Library sync pipeline — current index

The design is kept as one account of what is built and one list of what is
not:

| Need | Source |
|---|---|
| Built architecture and decisions: headless pipeline, review, library sources and index, runs, work list | [Implemented design](../implemented.md) |
| Built pipeline service: HTTP/OpenAPI interface, jobs, the work-directory lease and joining a run, auto mode, notifications | [Pipeline service](../pipeline-service.md) |
| Built Docker image and the Qt-free boundary | [Docker image](../pipeline-service.md#10-docker-image) |
| Everything unbuilt or unverified, with dependencies and completion criteria | [Outstanding design](../TODO.md) |
| Designer request/response and HTTP wire protocol (external contract) | [Designer interface](../designer-interface.md), [conformance tests](../designer-conformance-tests.md) |
| Library CLI and runtime use | [Pipeline README](../../src/main/python/pipeline/README.md) |
| Library user workflow and service user guide | [`docs/library/`](../../docs/library/) |
| Index schema | `SCHEMA` and `SCHEMA_VERSION` in `src/main/python/pipeline/library/index.py` |
| Former plans, topic files and their status notes | [`archive/`](../archive/) (historical, not current status) |

## Status

| Area (former IDs) | Status | Commits | Where to read |
|---|---|---|---|
| Chunks 1-2, 4-29, 34-36, 38, 41-44, 45c | Built | consolidated in `e256b18` (per-chunk hashes in the [archived plan](../archive/library-sync-pipeline-plan.md)) | [Implemented design](../implemented.md) |
| 39: BEQCatalogue JSON records | Built | `a72c673`, `01d9ec5`; BEQCatalogue `dc56c20d7`, `f398ffd38` | [Implemented design](../implemented.md#headless-pipeline) |
| Extraction folders | Built | `ffdcf1a` | [Implemented design](../implemented.md#library-discovery-and-sources) |
| Run Details persistence | Built | `48c2fc9` | [Implemented design](../implemented.md#library-work-list-and-runtime-behavior) |
| Title-page progress | Built | `92d8881` | [Implemented design](../implemented.md#library-work-list-and-runtime-behavior) |
| Work-list feedback F1-F5 | Built | F1 `de45afe`, F2 `7420a54`, F3 `49bbf34`, F4 `c58a4b4`, F5 `3d38f76` `ae1a086` `dc527f5` `12d875e` | [Implemented design](../implemented.md); agreed design in [archive](../archive/library-sync/worklist-feedback.md) |
| Pipeline service S0-S7 | Built | per chunk in [§11](../pipeline-service.md#11-status) | [Pipeline service](../pipeline-service.md) |
| Rejected designs (designer contract 1.1) | Built | contract `8818b10`, queue `13cda02`, title page `3497c43` | [Implemented design](../implemented.md#design-and-review) |
| Review Folder's lease check (service §5.1 gap) | Open | -- | [W3](worklist-w3.md) |
| S5 image: amd64 build and smoke | Verified | image `80cab0d`, smoke run at `7db992d` | [Docker image](../pipeline-service.md#10-docker-image) |
| S5 image: arm64 build and GHCR publish | Not yet run | -- | [C1](../TODO.md#c1--arm64-image-and-ghcr-publish) |
| 3/30, 31 | Waiting for live JRiver and acceptance evidence | -- | [E1-E2](../TODO.md#e1--jriver-response-fixture) |
| 32-33, 37 | Not started; evidence dependent | -- | [E3-E5](../TODO.md#e1--jriver-response-fixture) |
| 40 | Not started beyond a first-stream stub | -- | [J2](../TODO.md#j2--resolve-jrivers-selected-audio-stream) |
| 45a | Partial: decline shown as designer commentary built | `063c922` | [W1](worklist-w1.md) |
| 45b | Partial: codec/channel requests, automatic audio type and rescan backfill built | `62270b4`, `cb9456f` | [W2](../TODO.md#w2--stream-evidence-and-truthful-stages) |
| Designer requests by reference (beqforge R2, contract 1.2) | Built | D1.1 `bfff8e5`, D1.2 `4053773`, D1.3 `a303d20`, D1.4 `8f026ff` | [Implemented design](../implemented.md#design-and-review), [design](designer-by-reference.md) |
| Catalogue as input (D6), reviewer policy (former §9) | Product decisions | -- | [O1, O2](../TODO.md#o1--catalogue-as-input) |
| Intermittent test failures | Watching | -- | [T1, T2](../TODO.md#t1--intermittent-hang-in-the-parallel-suite) |

Chunk 3 became the live-fixture work in chunk 30. The old T1-T17 identifiers
of the archived plan: T2-T8 and T15 remain open as E1-E5; T17's metadata and
manual override are built, its automatic initial selection depends on J2; T1,
T9-T14 and T16 are built or accepted boundaries. (The current T1/T2 in
`TODO.md` are unrelated test-health items.)

## Identifiers cited by source comments

- **§ numbers of the original plan:** §3 sources, metadata and projects; §4
  the extract/design caches; §§5-7 orchestration and CLI; §11 JRiver, DVD and
  seasons; §12 discovery, revision and the work-list UI; §§14-16 parallel
  runs, progress and event lifetime. All now live in `implemented.md`.
- **`pipeline-service.md` §n:** that file keeps its section numbers.
- **`pipeline-service/docker.md` §10.1:** the Qt-free boundary.
- **`worklist-feedback.md` F1-F5:** the archived design; F1 the *Working*
  chip, F2 projects at extraction, F3 a decline as a flat candidate, F4 a
  failed extraction stays extract work, F5 joining the run in progress.

## Keeping this current

After each commit, update the item it touches in `TODO.md` and the row
above, in the same commit or straight after. When an item is complete, move
its lasting behavior into the implemented design as a statement of how the
code works, remove it from `TODO.md`, and mark its row here Built with
the commit hash (AGENTS.md). Hashes live in the status tables, not in the
prose describing the design.
