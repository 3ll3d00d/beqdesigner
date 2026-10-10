# Design references and work tracking

## Status

Reorganized on **2026-09-30**. This directory contains current descriptions of
architecture and external contracts, one [prioritized TODO](TODO.md), and an
[archive](archive/README.md) of completed plans and historical decisions.
Design references describe delivered behavior; they are not implementation
plans or additional backlogs. Keeping the name `design/` preserves the stable
contract paths used by schemas, source comments and external implementers.

| Document | Kind and scope |
|---|---|
| [Implemented design](implemented.md) | Architecture reference: delivered headless pipeline, review, library discovery, runs and desktop work list |
| [Pipeline service](pipeline-service.md) | Architecture reference: HTTP control plane, jobs, lease, scheduling, notifications, Docker image and Qt-free boundary |
| [Designer interface](designer-interface.md) | Normative external design contract, v1.2 |
| [Designer conformance specification](designer-conformance-tests.md) | Reference test specification for contract implementers, not pending tasks |
| [Review over HTTP](review-over-http.md) | Architecture reference: the service's review routes, their rules and the in-flight check |
| [Browser app](web-app.md) | Architecture reference: the service's browser app at `/ui`, its build and delivery, and the decision rules it shares with the title page |
| [TODO](TODO.md) | Sole current backlog, with status, dependencies and acceptance criteria in priority order |
| [E2 runbook](e2-runbook.md) | Procedure for the release acceptance run (E2) and its record template |
| [Archive](archive/README.md) | Completed implementation plans, commit/status records, acceptance evidence and superseded proposals |

Runtime instructions are in the [pipeline README](../src/main/python/pipeline/README.md)
and [library user guide](../docs/library/). Historical chunk and section IDs
remain in the archive so older source comments and discussions can be traced.
Update current design references when behavior changes. Record unfinished work
only in TODO; archive completed work and its evidence rather than maintaining
completion tables here.
