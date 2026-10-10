# Review over HTTP — the pipeline service's review routes

**Document type:** Architecture reference — delivered behavior (W2 of [web-review.md](web-review.md), 2026-10-10). It is
§6.8 of [pipeline-service.md](pipeline-service.md), split out to keep that file readable; the routes are in its §6.1 and
the user guide is "Deciding a title" in [`docs/library/service.md`](../docs/library/service.md).

The review routes decide one title with the same rules and write as the app's title page, through the Qt-free
`pipeline/library/decide.py` (`decide()`, `decision_blocked()`, `offered_digest()`) and `review_chart.py`
(`chart_curves()`); `model/worklist_title_decide.py` calls the same functions.

- **`Review`** carries the entry's designs as `CandidateView`s (`index` into `offered`: candidates, then the designs the
  designer rejected, `rejected: true`), `chosen_index`, `declined`, the metadata and `metadata_problems`
  (`metadata_problems()` with the profile's `meta_defaults`), `blocked` (`{accept, reject}`: the decision's status rule,
  then `decision_blocked()` with `run_hint` naming `POST /v1/jobs/run` and `metadata_hint` the app), `in_flight`, `playback`, the designer and its
  build, and the row's `needs`/`detail`. Title and year come from the row, then the entry.
- **`digest`** is `offered_digest()`: a hash of the entry's `fs` and its offered designs. A `Decision` sends it back;
  `decide()` re-reads the entry and refuses (`changed`) if the status no longer allows the decision or the digest differs.
  Refusals map to 409 with `title` *Changed since it was read*, *Not offered now* (`blocked`), *Metadata incomplete* or
  *The designer rejected this design* (`override_rejection` not given); an invalid pick is a 422.
- **In flight:** a title is in flight while a running job (or one joined to it) has it in hand -- from its `queued`,
  `stage_queued` or `stage_started` execution event or a `Progress` naming it, until its `title_completed`, `failed`,
  `skipped` or `cancelled` event, or the job's end (`JobManager.titles_in_hand()`). A design that fails without an
  exception sends no terminal event, so its title stays in hand until the job ends: the refusal errs on the safe side.
  A fresh lease held by something that is not one of this service's jobs (the work list, the CLI) puts every title that
  needs extract or design in flight. A title in flight refuses both decisions, and `next` passes it over.
- **Authority:** only the bearer token. A decision is a person's, made on the review, and writes only the queue entry;
  bulk accept, publish and commit stay behind `allow_repository_writes` (§6.4). `ServiceStatus` says up front whether
  those can be offered: `repository_writes` and `repositories_configured` (the profile names `xml_repo`, which
  `run_stages` needs for both).
- **The index** learns of a decision by `LibraryIndex.refresh()`. `IndexRefresher` does it on a thread of its own after
  the response, folding requests made meanwhile into one, and only when no job of the service is queued or running and no
  fresh lease is held; otherwise the run's own refreshes (after each title, and at its end) take the decision in.
- **Next:** `next_waiting_id()` over the filter's rows after `after`, wrapping round; waiting means the row's
  `review_state` is pending and `needs` is review, the title is not in flight, and the entry, read again, is pending.

Tests: `test_pipeline_service_review.py` (routes, in-flight, refresher) and `test_pipeline_library_decide.py`.
