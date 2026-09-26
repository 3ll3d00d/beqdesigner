# Library sync pipeline — current index

The original multi-chunk plan has been swept into a single account of
[implemented behavior](implemented.md) and one [deduplicated backlog](outstanding.md).
The original topic files are in [`archive/`](archive/) for historical
references. Their old status notes are no longer active. The designer
protocol remains a separate, active
[external contract](designer-interface.md).

## Status at the 2026-09-24 sweep

The consolidation and the library extraction change landed in `e256b18`.
With Keep multichannel enabled, the library now derives the mono design WAV
from the analysis-rate multichannel extraction using the extractor's pan
coefficients; the focused library/orchestration suite passed 175 tests.

| Former chunks | Current status | Where to read |
|---|---|---|
| 1-2, 4-29, 34-36, 38, 41-44, 45c | Implemented | [Implemented design](implemented.md) |
| 3/30, 31 | Waiting for live JRiver and acceptance evidence | [E1-E2](outstanding.md#external-evidence-and-disc-behavior) |
| 32-33, 37 | Unfinished; evidence dependent | [E3-E5](outstanding.md#external-evidence-and-disc-behavior) |
| 39 | Implemented: BEQDesigner terminology in `a72c673`, configured-source bridge in `01d9ec5`; BEQCatalogue JSON reader in `dc56c20d7` and producer repository onboarding in `f398ffd38` | [Implemented design](implemented.md#headless-pipeline) |
| 40 | Selected-stream resolver in progress | [J2](outstanding.md#j2-resolve-jrivers-selected-audio-stream) |
| 45a | Partial | [W1](outstanding.md#w1-retry-failures-and-title-page-details) |
| 45b | Partially built in `62270b4` and `cb9456f`: codec/channel requests, automatic first-stream audio type and existing-entry backfill on rescan | [W2](outstanding.md#w2-stream-evidence-and-truthful-stages) |
| Extraction folders | Implemented in `ffdcf1a`: readable track and stream names with stable ID lookup and legacy cache support | [Implemented design](implemented.md) |
| Run Details persistence | Implemented in `48c2fc9`: redacted last-run details restored after restart | [Implemented design](implemented.md) |
| Title-page progress | Implemented in `92d8881`: viewed track progress on its page, completed-title count on the list | [Implemented design](implemented.md) |
| Pipeline service (S0-S7) | Implemented: schedule `df93b2a`, Docker image/CI `80cab0d`, webhooks `af261fc`, documentation `faafbba`/`73ccfde`. Image build/smoke awaits first CI run. | [Pipeline service](pipeline-service.md) |

Chunk 3 became the live-fixture work in chunk 30. The remaining optional
catalogue-as-input idea is [O1](outstanding.md#o1-catalogue-as-input);
reviewer-policy confirmations are [O2](outstanding.md#o2-reviewer-and-project-policy-confirmation).
The old T1-T17 identifiers are resolved or mapped in the two current docs:
T2-T8 and T15 remain open as E1-E5. T17's metadata and manual override
are delivered; its automatic initial selection still depends on J2.
T1, T9-T14 and T16 are delivered or accepted boundaries in `implemented.md`.

## Current documentation map

| Need | Source |
|---|---|
| Delivered architecture and decisions | [Implemented design](implemented.md) |
| Work remaining, dependencies and completion criteria | [Outstanding design](outstanding.md) |
| Library CLI and runtime use | [Pipeline README](../src/main/python/pipeline/README.md) |
| Library user workflow | [`docs/library/`](../docs/library/) |
| Designer request/response and HTTP wire protocol | [Designer interface](designer-interface.md) |
| Pipeline service: HTTP/OpenAPI interface, auto mode and Docker image (built; CI image smoke pending) | [Pipeline service](pipeline-service.md) |
| Index schema | `SCHEMA` and `SCHEMA_VERSION` in `src/main/python/pipeline/library/index.py` |

Older source comments cite the original section numbers. Their subjects now
live in `implemented.md`: §3 covers sources, metadata and projects; §4 covers
the extract/design caches; §§5-7 cover orchestration and CLI; §11 covers JRiver,
DVD and seasons; §12 covers discovery, revision and the work-list UI; §§14-16
cover parallel runs, progress and event lifetime. Open portions of those
subjects are listed once in `outstanding.md` by item ID.

Future implementation commits should update the relevant item in
`outstanding.md` and the status row above in the same commit, or immediately
in a follow-up commit. Once an item is complete, move its lasting behavior
to `implemented.md` and remove the open item rather than preserving a second
status narrative.
