The **pipeline service** runs the machine stages of the workflow -- discovery, extract and design -- as a long-running process you drive over HTTP, instead of a command you schedule. It sits idle until asked, runs one job at a time, and lets you follow each job as it goes. Review stays in the app: the service fills the review queue, and you review in the [work list](work.md) as usual.

It uses **the same profile file** as the work list and the [command line](unattended.md), and a run from the service is exactly the run `python -m pipeline.library.cli run --profile FILE` makes with that file.

### Starting it

From a checkout of BEQDesigner (see the [readme](https://github.com/3ll3d00d/beqdesigner/blob/main/readme.md)), with `uv sync` done:

```
export BEQ_SERVICE_TOKEN=a-long-random-string
PYTHONPATH=src/main/python uv run python -m pipeline.service --profile /path/to/library-profile.yaml
```

For a container, copy [`docker/compose.example.yaml`](https://github.com/3ll3d00d/beqdesigner/blob/main/docker/compose.example.yaml)
and [`docker/service.example.yaml`](https://github.com/3ll3d00d/beqdesigner/blob/main/docker/service.example.yaml).
Put `profile.yaml`, `service.yaml` and a private `service-token` in the
compose example's `config/` folder, and mount your media at the exact paths
the profile names. Run `docker compose up -d`. The image is published for
tagged releases as `ghcr.io/3ll3d00d/beqdesigner-pipeline:<tag>` for amd64
and arm64; use a fixed tag when you need repeatable deployments. The compose
example maps the port to loopback for a local TLS reverse proxy. Its work and
queue mounts should be writable by the configured UID/GID.

It listens on port 8080 of every address. Open `http://HOST:8080/docs`: the whole interface, with **Try it out** on every call. Press **Authorize**, paste the token, and the calls you try from the page are made for real.

| Option | Meaning |
|---|---|
| `--profile FILE` | the catalogue profile (or `$BEQ_PROFILE`) |
| `--service-config FILE` | the service's own settings, below (or `$BEQ_SERVICE_CONFIG`) |
| `--host`, `--port` | where to listen, over the settings file |
| `--no-auth` | no token; only allowed with `--host 127.0.0.1` |

Every call under `/v1` needs the token as `Authorization: Bearer <token>`. `/health`, `/ready` (is the profile readable, the work directory writable, ffmpeg found and the designer declared?) and the documentation pages do not. `/ready` also reports whether the designer answers (`designer_reachable`), but a designer that is down does not make the service unready: it is not the service's fault, and restarting the service would not help. There is no TLS: put the service behind a reverse proxy to reach it from outside your network.

Stopping it (Ctrl-C, or `SIGTERM`) lets the title in hand finish, then stops.

### What you can ask it

| Call | Does |
|---|---|
| `POST /v1/jobs/scan` | list the sources again (the work list's *Rescan*) |
| `POST /v1/jobs/run` | extract and design the titles a filter selects |
| `GET /v1/titles`, `/v1/titles/{id}` | the titles a filter selects and what each needs, or one title |
| `POST /v1/plan` | what a run would do, and what it would skip and why -- changing nothing |
| `GET /v1/jobs`, `/v1/jobs/{id}` | the jobs, and one job with its result |
| `GET /v1/jobs/{id}/events` | follow a job as it runs (Server-Sent Events) |
| `GET /v1/jobs/{id}/log` | what a job has reported so far, as a list |
| `POST /v1/jobs/{id}/cancel` | cancel: a waiting job is dropped, a running one stops after the title in hand |
| `GET /v1/status` | the counts the work list's strip shows, the job running now, the schedule and whether the designer answers |
| `GET /v1/schedule`, `PUT /v1/schedule` | read or save the automatic extract/design schedule |
| `POST /v1/schedule/trigger` | run one scheduled tick now; returns 409 while a job is queued or running |
| `POST /v1/notify/test` | send a sample of each configured event to one notification target |
| `POST /v1/jobs/accept` | bulk accept (see below) |
| `GET /health`, `/ready` | is it up; is it able to work |

A run selects titles with the same filters as the work list, plus two it has no box for:

```
curl -X POST http://nas:8080/v1/jobs/run \
     -H "Authorization: Bearer $BEQ_SERVICE_TOKEN" -H 'Content-Type: application/json' \
     -d '{"filter": {"kind": "movie", "year": "2026"}, "through": "design"}'
```

| Field | Selects |
|---|---|
| `needs` | what the title needs next: `attention`, `extract`, `design`, `review`, `publish`, `commit`, `done` (a list) |
| `new_since_scan` | titles the latest scan found for the first time (*New*) |
| `source`, `match`, `ids` | a source of the profile, text in the title or path, named titles |
| `kind` | `movie` or `tv` |
| `year` | `2026`, `<1960`, `>=2020` or `1990-1999`: the language of an [ignore rule](setup.md)'s year. A title with no year never matches -- a filesystem source knows a title's year only once TMDB has named it |

`through` is `extract` or `design`, as the work list's action button. `publish` and `commit`, and bulk accept (`POST /v1/jobs/accept`), are refused unless the settings file allows them (`allow_repository_writes: true`); a dry run of bulk accept is always allowed. A misspelt field is an error, not a wider selection.

The interface is published as an OpenAPI document, [`docs/schema/service.openapi.json`](https://github.com/3ll3d00d/beqdesigner/blob/main/docs/schema/service.openapi.json), and at `/openapi.json`, so a client can be generated from it.

### Settings

`--service-config` names a YAML (or JSON) file; with none, the defaults apply.

```yaml
listen: {host: 0.0.0.0, port: 8080}
allow_repository_writes: false   # publish, commit and bulk accept over HTTP
history_limit: 200               # finished jobs remembered, in <work dir>/service/jobs.json
shutdown_grace_seconds: 120
schedule:
  enabled: false
  interval_minutes: 60           # at least 5; measured from a scheduled job's finish
  through: design                # extract or design only
  filter: {kind: movie}          # optional; needs is always extract or design
  retry_failed: false
notify:
  - name: phone
    url: https://ntfy.example/beq
    format: text                # json, text, slack or discord
    events: [review_waiting, failed]
    origins: [schedule]
```

When enabled, a tick scans the sources and works on titles still needing extract or design. It leaves designs in the
review queue. A tick due while another job is active is skipped, so scheduled jobs never accumulate. Before a tick
through design the service asks the designer's `/health`. If nothing answers (or a proxy in front of it answers 502, 503
or 504) the tick is skipped, `last_skip` says why, it is tried again within 5 minutes, and `failed` is notified once for
the outage, not at every skipped tick. A designer that answers anything else, including a designer too old to serve
`/health`, counts as up. A run job through design that you submit while the designer is down fails at once, saying so,
before anything is extracted. Saving the schedule
through `PUT /v1/schedule` writes `<work dir>/service/schedule.json`; that file takes precedence over `service.yaml` on
restart. Set `enabled: false` to pause it without losing its other settings.

Notifications are sent after a job finishes. `review_waiting` fires when a run
designed titles, including a designer decline; `failed` fires for a new title,
source, publish or job failure; `job_finished` can be enabled for every job.
Earlier failures skipped by the pipeline do not trigger another `failed`
notification. Targets default to scheduled jobs and the first two events;
add `api` to `origins` or `job_finished` to `events` when needed. JSON sends
job details, designed title names, years and confidence, failure messages and
the review count. Text, Slack and Discord send a one-line count summary.
`GET /v1/status` shows each target's last delivery outcome without its URL or
headers. A failed webhook never changes the pipeline job's result.

Secrets come from the environment rather than a file, each also as `NAME_FILE` naming a file that holds it (a Docker secret): `BEQ_SERVICE_TOKEN`; `TMDB_API_KEY`; `JRIVER_PASSWORD` or `JRIVER_PASSWORD_<SOURCE>` (the source's name in capitals, `_` for anything else); and `BEQ_DESIGNER_HEADERS_<NAME>`, a JSON object of HTTP headers for that designer.
For a notification target `phone`, `BEQ_NOTIFY_URL_PHONE` overrides its URL
and `BEQ_NOTIFY_HEADERS_PHONE` supplies a JSON object of request headers.
Those values also accept the `_FILE` form and are kept out of service status
and delivery errors. Configure targets only for destinations you intend to
receive the job and title details.

### The service and the work list together

While a job of the service's runs, it holds the work directory. **Extract & design** in the work list (and a command-line `run`) hands its titles to the job, which queues them behind its own: the work list shows them under *Working* and follows the job until it ends, and runs them itself if the job ended without taking them. Publish, Commit and publishing on accept say *the pipeline service on HOST is running a job* and do not start. Looking, reviewing and deciding go on as normal. The work list and the command line hold the work directory for their own runs too: a service job that starts while one of them is running hands its extract and design titles to that run, and ends when it does, with what became of them; a scan, a publish or a bulk accept waits for it to end.

A run job through extract or design that you submit while another run job is extracting or designing does not wait in the queue: it **joins** that run, which queues its titles behind its own. The joined job is *running* at once, says which job it joined in `joined_to`, and ends when that run ends, with its result. If the run had already moved past extracting and designing, the job waits its turn as usual.
