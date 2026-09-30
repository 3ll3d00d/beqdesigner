# Historical design archive

## Status

These documents are historical plans, implementation and acceptance records,
and superseded proposals. Their open rows and questions record what was known
at the time; they are **not current TODO lists**. All current unfinished work
is in [TODO](../TODO.md). Current architecture and contracts are listed in the
[design index](../README.md).

The original plans retain section numbers and commit hashes for source comments
and older discussions. Their completed behavior is described in the current
architecture references rather than duplicated as active plans.

| Record | Contents |
|---|---|
| [Original library plan](library-sync-pipeline-plan.md) and `library-sync/` | Historical chunks, topic designs, remediation and workflow plans |
| [Consolidated library completion record](library-sync-completion-record.md) | Final consolidated status snapshot before the single TODO reorganization |
| [Service plan](pipeline-service/plan.md) and [completion record](pipeline-service/completion-record.md) | S0–S7 implementation history, delivery commits and verification |
| [Designer requests by reference](designer-by-reference.md) | Completed D1.1–D1.4, live acceptance evidence and original proposals; remaining follow-ons are D2–D4 in TODO |
| [W3 lease guard](worklist-w3.md), [W1 retry/failure details](worklist-w1.md) | Completed work-list fixes and regression evidence |
| Other files | Original headless pipeline, HTTP binding and candidate review plans |

Historical proposals may differ from delivered code. The current designer
contract takes precedence over historical wire proposals. Preserve historical
evidence when archiving a completed TODO, and label its completion date and
implementation commit where available.
