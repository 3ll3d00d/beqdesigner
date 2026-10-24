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
| [Designer interface](designer-interface.md) | Normative external design contract, v1.5 (1.5's adjustments built on the caller's side, not yet beqforge's) |
| [Device profiles](device-profiles.md) | The filter on a device's rate and coefficient precision, and optimising it for devices |
| [Designer conformance specification](designer-conformance-tests.md) | Reference test specification for contract implementers, not pending tasks |
| [Review over HTTP](review-over-http.md) | Architecture reference: the service's review routes, their rules and the in-flight check |
| [Browser app](web-app.md) | Architecture reference: the service's browser app at `/ui`, its build and delivery, and the decision rules it shares with the title page |
| [Design version](design-version.md) | Contract 1.4's design identity and the caller's *out of date* selection: why, rules and what is built (TODO L3) |
| [Redesign scope](redesign-scope.md) | Design, not built: contract 1.6's run record, *reviewed at* and `/impact` on the caller's side (TODO RS) |
| [Web app parity](web-parity.md) | Plan, not built: browser editing, visualisation, BEQ comparison and adjusted redesign with variants (TODO WP) |
| [Library run gaps](library-run-gaps.md) | Plan, not built: the first full library run's results and the work left (TODO L1-L6) |
| [TODO](TODO.md) | Sole current backlog, with status, dependencies and acceptance criteria in priority order |
| [E2 runbook](e2-runbook.md) | Procedure for the release acceptance run (E2) and its record template |
| [Archive](archive/README.md) | Completed implementation plans, commit/status records, acceptance evidence and superseded proposals |

Runtime instructions are in the [pipeline README](../src/main/python/pipeline/README.md)
and [library user guide](../docs/library/). Historical chunk and section IDs
remain in the archive so older source comments and discussions can be traced.
Update current design references when behavior changes. Record unfinished work
only in TODO; archive completed work and its evidence rather than maintaining
completion tables here.
