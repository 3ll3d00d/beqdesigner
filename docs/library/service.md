The **pipeline service** runs the machine stages of the workflow -- discovery, extract and design -- as a long-running process you drive over HTTP, instead of a command you schedule. It sits idle until asked, runs one job at a time, and lets you follow each job as it goes. Review stays in the app: the service fills the review queue, and you review in the [work list](work.md) as usual.

It uses **the same profile file** as the work list and the [command line](unattended.md), and a run from the service is exactly the run `python -m pipeline.library.cli run --profile FILE` makes with that file.

### Starting it

From a checkout of BEQDesigner (see the [readme](https://github.com/3ll3d00d/beqdesigner/blob/main/readme.md)), with `uv sync` done:

```
export BEQ_SERVICE_TOKEN=a-long-random-string
PYTHONPATH=src/main/python uv run python -m pipeline.service --profile /path/to/library-profile.yaml
```

It listens on port 8080 of every address. Open `http://HOST:8080/docs`: the whole interface, with **Try it out** on every call. Press **Authorize**, paste the token, and the calls you try from the page are made for real.

| Option | Meaning |
|---|---|
| `--profile FILE` | the catalogue profile (or `$BEQ_PROFILE`) |
| `--service-config FILE` | the service's own settings, below (or `$BEQ_SERVICE_CONFIG`) |
| `--host`, `--port` | where to listen, over the settings file |
| `--no-auth` | no token; only allowed with `--host 127.0.0.1` |

Every call under `/v1` needs the token as `Authorization: Bearer <token>`. `/health`, `/ready` (is the profile readable, the work directory writable, ffmpeg found and the designer declared?) and the documentation pages do not. There is no TLS: put the service behind a reverse proxy to reach it from outside your network.

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
| `GET /v1/status` | the counts the work list's strip shows, and the job running now |
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
```

Secrets come from the environment rather than a file, each also as `NAME_FILE` naming a file that holds it (a Docker secret): `BEQ_SERVICE_TOKEN`; `TMDB_API_KEY`; `JRIVER_PASSWORD` or `JRIVER_PASSWORD_<SOURCE>` (the source's name in capitals, `_` for anything else); and `BEQ_DESIGNER_HEADERS_<NAME>`, a JSON object of HTTP headers for that designer.

### The service and the work list together

While a job of the service's runs, it holds the work directory: the work list's Run, Publish and Commit, and publishing on accept, say *the pipeline service on HOST is running a job* and do not start. Looking, reviewing and deciding go on as normal. The work list does not yet hold the work directory for its own runs, so do not start a service job while the work list is running one.
