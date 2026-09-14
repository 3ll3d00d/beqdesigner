# Implementation plan — HTTP designer binding

**Status: done.** All four phases below shipped. For the current wire
format, GUI workflow, and how it fits the rest of the pipeline, see
[`src/main/python/pipeline/README.md`](../src/main/python/pipeline/README.md#the-designer-contract)
and `designer-interface.md §7.1` (the formal wire spec). This file stays
only as a phase index for code comments that cite it by phase number.

| Phase | Shipped |
|---|---|
| 1 | `pipeline/designer/http_binding.py` — `http_designer(url)`, wire format (base64 float64 arrays request-side, plain JSON `DesignResponse` response-side), `HttpDesignerError` on transport/shape failures |
| 2 | `designer-interface.md §7` formalised + `docs/schema/http_designer_{request,response}.schema.json` |
| 3 | A durable `DESIGNER_HTTP_ENDPOINTS` preference, registered on every app startup. Originally its own Tools → Designers… dialog; later folded into `model/preferences.py`'s Preferences dialog as a "Designers" page alongside the review queue directory default (see `pipeline/README.md`'s "The designer contract"). |
| 4 | Tools → Batch Design… dialog, driving `pipeline.review.batch_design()` on a background thread with progress. Later folded into `model/batch.py`'s Batch Extract & Design dialog as an optional per-candidate step (see `pipeline/README.md`'s "Batch design + review") rather than a standalone dialog. |

Nothing is left open from this plan.
