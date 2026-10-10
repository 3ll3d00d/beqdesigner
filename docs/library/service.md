The **pipeline service** runs the machine stages of the workflow -- discovery, extract and design -- as a long-running process you drive over HTTP, instead of a command you schedule. It sits idle until asked, runs one job at a time, and lets you follow each job as it goes. Review stays in the app: the service fills the review queue, and you review in the [work list](work.md) as usual.

It uses **the same profile file** as the work list and the [command line](unattended.md), and a run from the service is exactly the run `python -m pipeline.library.cli run --profile FILE` makes with that file.

### Starting it

From a checkout of BEQDesigner (see the [readme](https://github.com/3ll3d00d/beqdesigner/blob/main/readme.md)), with `uv sync` done:

```
export BEQ_SERVICE_TOKEN=a-long-random-string
PYTHONPATH=src/main/python uv run python -m pipeline.service --profile /path/to/library-profile.yaml
```

It also serves a browser app at `http://HOST:PORT/ui/` (signing in with the token): at present only whether it is
reachable and what it is doing in brief. Running from a checkout, build it first (`cd src/main/web && npm ci && npm run
build`) and add `--ui-dir src/main/web/dist`; the image has it built in.

For a container, copy [`docker/compose.example.yaml`](https://github.com/3ll3d00d/beqdesigner/blob/main/docker/compose.example.yaml)
and [`docker/service.example.yaml`](https://github.com/3ll3d00d/beqdesigner/blob/main/docker/service.example.yaml).
Put `profile.yaml`, `service.yaml` and a private `service-token` in the
compose example's `config/` folder, and mount your media at the exact paths
the profile names. Run `docker compose up -d`. The image is published for
tagged releases as `ghcr.io/3ll3d00d/beqdesigner-pipeline:<tag>` for amd64
and arm64; use a fixed tag when you need repeatable deployments. The compose
example maps the port to loopback for a local TLS reverse proxy. Its work and
queue mounts should be writable by the configured UID/GID.

### The designer

The compose example also runs the designer, `ghcr.io/3ll3d00d/beqforge-designer`, as a second
service called `designer`. Both containers mount the same `work` folder: the designer reads it
at `/work`, its shared root, so the pipeline sends it the names of the extracted audio files
instead of the audio. Keep its version pinned (the example names one): a catalogue designed
over several days should come from one designer build, and pulling `latest` in the middle of a
run would mix two. Its `designer-cache` volume keeps the designer's stage cache across
restarts. Both run as the same user, which must be able to write `work` and the cache. The
[beqforge readme](https://github.com/3ll3d00d/beqforge#running-the-designer-in-a-container)
lists the designer's own settings.

**Change the profile for the container.** The container does not read the app's
Preferences, so a profile whose designer is one you added in *Preferences > Designers* (the
work list shows it as `http:NAME`) does not work there. Declare the designer in the profile
instead, by a name both the container and the work list use, and choose it:

```yaml
designers:
  beqforge: {url: 'http://designer:8420/design', by_reference: true}
run:
  designer: beqforge
  work_dir: /work
  queue_dir: /queue
```

Use the same name on the desktop. The designer's name is one of the settings a design is
judged by, so a title designed by `beqforge` in the container would otherwise look out of
date to a work list that calls the designer something else. `designer` is its name inside the
compose network; a desktop that also calls it needs the address it is published at.

The designer works on one title at a time and queues the rest, and a two-hour,
eight-channel title takes it about two minutes the first time. More than one design at a
time (`run.parallelism.design`) gains nothing against one designer; if you raise it, each
declared designer's timeout is multiplied by it, so the titles waiting in the designer's queue
do not time out (see [the profile file](setup.md#the-profile-file)).

Extraction is the other way round: it mostly reads, so on a pool of separate disks (an Unraid
host, say) raise `run.parallelism.extract` and set `run.disks` so that each disk is read by one
extraction at a time (see [the profile file](setup.md#the-profile-file)). The films must be
mounted from the pool itself (`/mnt/user/...` on Unraid) for the container to see which disk
each title is on.

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
| `GET /v1/titles/{id}/review` | what a person decides a title on: its designs, metadata problems, and whether Accept and Reject are offered |
| `GET /v1/titles/{id}/chart` | the title's measured curves, and the same after one design (`?candidate=`) |
| `POST /v1/titles/{id}/decision` | accept a design, or reject the title (see "Deciding a title" below) |
| `GET /v1/review/next` | the next title waiting for a decision after one (`?after=`), in a filtered list |
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

### Deciding a title

A person can accept or reject a title's design over HTTP, with the same rules as the app's title page. Read the review,
then send the decision with the review's `digest`:

```
curl http://nas:8080/v1/titles/fs-alien/review -H "Authorization: Bearer $BEQ_SERVICE_TOKEN"
curl -X POST http://nas:8080/v1/titles/fs-alien/decision \
     -H "Authorization: Bearer $BEQ_SERVICE_TOKEN" -H 'Content-Type: application/json' \
     -d '{"decision": "accept", "candidate": 0, "digest": "<the review's digest>"}'
```

* `candidate` counts the designs offered: the candidates, best first, then the designs the designer rejected. Accepting a
  rejected design needs `"override_rejection": true`, and is recorded as your override.
* A decision is refused (409, saying why) if the title was decided or redesigned since you read the review (the `digest`
  no longer matches), if a run is working on it, if its last extraction or design failed, or -- for Accept -- if it needs
  extract or design first or its metadata is incomplete. The review's `blocked` says the same before you try.
  Metadata is edited in the app.
* Only the token is needed: a decision writes the title's review queue entry, nothing in the catalogue repositories.
  Publishing and committing accepted titles stay behind `allow_repository_writes` (a run job through `publish` or
  `commit`).
* The counts in `GET /v1/status` take a decision in shortly afterwards, or, while a run is going, when it next
  updates the index.

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

### Reviewing from another machine

The service fills the review queue; you review it in the app on your desktop, or decide titles over HTTP ("Deciding a
title" above). For the app, one layout is supported:

* **The work and queue folders live on the machine that runs the container**, on its own disk (for example a NAS
  running Docker), as the compose example mounts them (`./work`, `./queue`). The desktop reaches them over the network
  (an SMB or NFS share of those two folders), read and write.
* **The desktop has its own profile.** The container's profile names the folders as the container sees them (`/work`,
  `/queue`) and the media as `/media/...`; the desktop's names the same folders by the desktop's paths (`work_dir:
  /mnt/nas/beq/work`, or `W:\beq\work` on Windows), and its sources and JRiver path mappings by the desktop's view of the
  media. Keep everything else the same in both: the sources, the designer's **name** under `designers:` (its URL may
  differ), `run.keep_multichannel` and the analysis settings, since a design is judged by them. Nothing pairs the two
  files; keep them side by side and change both.
* **While a run is going, review with *Tools > Review Folder...*** on the shared queue folder. It works on the queue
  entries only. The work list also keeps the discovery index (a database file in the work folder) up to date as you
  decide, and a database file is not safe to write from two machines at once over a network share. Use the work list
  from the desktop when the service is idle (`GET /v1/status` shows `current_job`), and start runs through the service
  rather than from the desktop's work list.

A title's poster is kept in its own work folder, so it shows on the desktop whatever path the container recorded, and a
title published from the desktop is not made out of date by the difference. Project files carry their own audio and open
from the share.

### A first run over a whole catalogue

At about two minutes a title, a catalogue of a thousand titles is days of work. Start it in steps you can check:

1. **Check it is ready.** `GET /ready` must say `ready`. It also reports, without failing, whether the designer answers
   (`designer_reachable`) and whether TMDB is looked up (`tmdb`). Without `TMDB_API_KEY` the service still designs, with
   only the metadata the library has (no TMDB title, year, artwork or ids), which you then fill in on each title page;
   set the key (see [Starting it](#starting-it)) unless that is what you want.
2. **Scan first:** `POST /v1/jobs/scan`. `GET /v1/titles?needs=extract` then lists what a run would work on, and
   `POST /v1/plan` shows it for a filter without running anything.
3. **Run a small batch**, for example one year or one source:
   `POST /v1/jobs/run` with `{"filter": {"year": "2019", "needs": ["extract", "design"]}, "through": "design"}`. Look at
   the first results in the work list or Review Folder before going on: a wrong path mapping, stream choice or designer
   setting shows in the first few titles, not after a thousand.
4. **Then enable the schedule** (`PUT /v1/schedule` with `"enabled": true`). Each tick runs whatever still needs
   extract or design, so it carries the catalogue on from where the batches left it.

While a job runs, `GET /v1/status`'s `current_job.progress` says how far it has got (`done` of `total` title-stages), how
many it finishes an hour (`per_hour`), and at that rate how long the rest will take (`remaining_seconds`) and when it
should end (`estimated_finish`). The first titles of a run are the least reliable guide: a cold designer takes longer.

**A restart mid-run** (the container stopped, or the machine) loses nothing that was finished. The job is recorded as
`interrupted` and is not resumed: the title in hand at the time is done again from the start by the next run or tick,
and every title already designed stays designed. A run only ever does what is still needed.
