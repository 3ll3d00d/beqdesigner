# E2 — acceptance runbook

**Document type:** procedure for [E2](TODO.md#e2--end-to-end-acceptance-record). Version 1, 2026-10-08.

**Under test:** `ghcr.io/3ll3d00d/beqdesigner-pipeline:2.2.0-alpha.1` (tag `2.2.0-alpha.1`, commit
`3ec6b5d`), `ghcr.io/3ll3d00d/beqforge-designer:0.2.0`, and the desktop app from `main` (or the
`2.2.0-alpha.1` build). Fill in the record template at the end as you go and save it as
`design/archive/e2-record-<date>.md`. Anything that does not behave as the **Expect** line says is
a finding: note it, carry on where you can, and it becomes a focused fix.

There is no `docker compose` here, so the two services run as separate containers on one Docker
network, set up exactly as `docker/compose.example.yaml` describes.

## 0. Before you start

Pick these and keep them for the whole run. The commands below use them as shell variables.

| Variable | Meaning | Example |
|---|---|---|
| `E2` | a folder on the **Docker host** holding everything for this run | `/srv/beq-e2` |
| `FILMS` | the films share as the Docker host sees it (JRiver's `W:\`) | `/media/films` |
| `COVERS` | MC's cover-art folder (`X:\JRiver Cover Art`) as the host sees it, if it is shared | `/media/jriver-covers` |
| `HOST` | the Docker host's address, as the desktop reaches it | `nas.local` |
| `SHARE` | where the desktop mounts `$E2` from the host (SMB or NFS) | `/mnt/beq-e2` |
| `TOKEN` | a long random string for the service | `openssl rand -hex 24` |

Run the containers as your user (uid and gid 1000), so both can write the work folder and the
desktop reads what they write.

**The repositories must be disposable.** The desktop profile today publishes into the real
`beqfilters` and `beqimgs`. E2 uses copies whose "remote" is a local bare repository, so a push goes
nowhere public:

```bash
mkdir -p $E2/remotes $E2/repos
for r in beqfilters beqimgs; do
  git clone --bare git@github.com:3ll3d00d/$r.git $E2/remotes/$r.git
  git clone $E2/remotes/$r.git $E2/repos/$r
done
```

**Record the versions:** Docker (`docker version --format '{{.Server.Version}}'`), the host OS,
the desktop OS, MC (Help > About), the desktop app's version (Help > About) and ffmpeg's
(`ffmpeg -version | head -1`).

## 1. The sample

About 50 titles that cover every form in the catalogue. From the E1 capture, include at least:

| Kind | Titles to include (from the capture where possible) |
|---|---|
| Plain files | a few `.mkv` (one AC-3 only, one DTS-HD MA, one TrueHD Atmos), one `.mp4` |
| Blu-ray `index.bdmv` and `index.bluray;1` | several of each |
| JRiver plays a non-first audio stream (J2) | Pacific Rim, The Godfather, Spider-Man 2, Arrietty |
| A named Blu-ray playlist (E4) | Alice in Wonderland, Harry Potter and the Deathly Hallows Part 2, Blade Runner 2049 |
| Duration decides the playlist (E4) | Wall-E, Glory, Michael Clayton |
| A logo clip is dropped (E4) | A Star Is Born, Arrietty, Bande A Part |
| A playlist-file entry (E4) | Frozen Fever (on the Cinderella 2015 disc) |
| An incomplete rip (should fail, saying so) | RoboCop |
| Cover art in MC's folder | TRON, TRON: Legacy, Lone Survivor, Das Boot |
| A DVD rip, if the catalogue has one (E3) | any |
| A TV season, in season mode (E5) | one short season |
| Long titles, for timings (R4) | the three longest in the catalogue |

## 2. Configuration files

Make the folders, owned by your user:

```bash
mkdir -p $E2/config $E2/work $E2/queue $E2/designer-cache
printf '%s' "$TOKEN" > $E2/config/service-token && chmod 600 $E2/config/service-token
```

`$E2/config/profile.yaml`, **the container's profile.** It is the desktop's with container paths:
the designer is declared, not taken from Preferences, and the media is `/media/...`.

```yaml
sources:
- name: jriver
  kind: jriver
  host: 10.151.167.11
  port: 63412
  ssl: false
  username: <your MCWS user>          # the password comes from JRIVER_PASSWORD below
  path_mappings:
  - {from: 'W:\', to: /media/films}
  - {from: 'X:\JRiver Cover Art', to: /media/covers}   # leave out if COVERS is not shared
  external_id_fields: {movie: {tmdb: [TheMovieDB Movie ID]}}
  browse_node_id: 1004
  browse_path: Video > Movies
designers:
  beqforge: {url: 'http://designer:8420/design', by_reference: true}
run:
  designer: beqforge
  tv_mode: season
  keep_multichannel: true
  work_dir: /work
  queue_dir: /queue
  parallelism: {extract: 2, design: 1}    # one designer works on one title at a time
  min_free_gb: 10
  stop_after_unavailable: 3
  bass_management: {lpf_fs: 80, lpf_position: Before, headroom_type: WCS}
```

`$E2/config/service.yaml`. The schedule starts **off**; step 4 turns it on.

```yaml
listen: {host: 0.0.0.0, port: 8080}
allow_repository_writes: false
history_limit: 500
shutdown_grace_seconds: 120
schedule: {enabled: false, interval_minutes: 30, through: design, filter: {}, retry_failed: false}
```

**The desktop's profile** (Library Work List > Settings, or the file): the same sources, designer
name and run settings, with the desktop's paths. Change its designer from `http:beqforge` to a
declared one, **with the same name** (a design is judged by the designer's name):

```yaml
designers:
  beqforge: {url: 'http://<HOST>:8420/design'}   # publish port 8420 on the designer (step 3) to use it from here
run:
  designer: beqforge
  work_dir: <SHARE>/work
  queue_dir: <SHARE>/queue
  # tv_mode, keep_multichannel, parallelism, bass_management: as the container's
sync:
  filter_repo: <E2 repos>/beqfilters        # the disposable copies, as the desktop sees them
  images_repo: <E2 repos>/beqimgs
  image_owner: 3ll3d00d                      # the local remotes have no GitHub name to read
  image_repo_name: beqimgs
```

Set Preferences > JRiver path mappings on the desktop to include `X:\JRiver Cover Art` too.

## 3. Start the two containers

```bash
docker network create beq-e2

docker run -d --name designer --network beq-e2 --restart unless-stopped \
  --user 1000:1000 -p 127.0.0.1:8420:8420 \
  -v $E2/work:/work -v $E2/designer-cache:/cache \
  ghcr.io/3ll3d00d/beqforge-designer:0.2.0

docker run -d --name pipeline --network beq-e2 --restart unless-stopped \
  --user 1000:1000 -p 8080:8080 --stop-timeout 120 \
  -e BEQ_SERVICE_TOKEN_FILE=/config/service-token \
  -e JRIVER_PASSWORD='<your MCWS password>' \
  -e TMDB_API_KEY='<your TMDB key>' \
  -v $E2/config:/config:ro -v $E2/work:/work -v $E2/queue:/queue \
  -v $FILMS:/media/films:ro -v $COVERS:/media/covers:ro \
  ghcr.io/3ll3d00d/beqdesigner-pipeline:2.2.0-alpha.1
```

(Publish 8420 beyond `127.0.0.1` only if the desktop should reach the designer; the pipeline reaches
it on the network as `designer`.)

```bash
API=http://$HOST:8080; AUTH="Authorization: Bearer $TOKEN"
curl -s $API/ready | jq
curl -s -H "$AUTH" $API/v1/status | jq '{version, designer, tmdb}'
```

**Expect:** `ready: true`; checks `designer_reachable` and `tmdb` both `ok: true`; status says
version `2.2.0-alpha.1` and `designer.reachable: true` at `http://designer:8420/design`.

## 4. The container run (milestone part)

**4.1 Scan and choose the sample.**

```bash
curl -s -H "$AUTH" -X POST $API/v1/jobs/scan -H 'Content-Type: application/json' -d '{}' | jq .id
curl -s -H "$AUTH" "$API/v1/titles?limit=1000" | jq -r '.titles[] | [.id, .title, .year, .needs] | @tsv' > $E2/titles.tsv
```

Pick the sample's ids from `titles.tsv` into `$E2/sample.json` as `{"ids": ["jriver-...", ...]}`.

**4.2 Preview, then a first small batch** (five titles) checked by hand before the rest:

```bash
curl -s -H "$AUTH" -X POST $API/v1/plan -H 'Content-Type: application/json' \
  -d "{\"filter\": $(jq '{ids: .ids[:5]}' $E2/sample.json), \"through\": \"design\"}" | jq
curl -s -H "$AUTH" -X POST $API/v1/jobs/run -H 'Content-Type: application/json' \
  -d "{\"filter\": $(jq '{ids: .ids[:5]}' $E2/sample.json), \"through\": \"design\"}" | jq .id
```

Follow it: `curl -sN -H "$AUTH" $API/v1/jobs/<id>/events`. **Expect:** each title's events say
"Extracting audio stream N: ... (as the library plays it | ...); multichannel kept", then
"Extraction complete: C channels (layout)", then design; the job ends `succeeded`; five queue
entries in `$E2/queue`. Open them in the desktop's **Review Folder** on `$SHARE/queue` and check the
title page: candidates, commentary ending with *Designed by beqforge, <build>* and *Bass management
sent: 80 Hz crossover ...*, artwork.

**4.3 Schedule the rest.** Put the sample's filter in the schedule and enable it:

```bash
curl -s -H "$AUTH" -X PUT $API/v1/schedule -H 'Content-Type: application/json' \
  -d "{\"enabled\": true, \"interval_minutes\": 30, \"through\": \"design\", \"retry_failed\": false, \"filter\": $(jq '{ids}' $E2/sample.json)}" | jq
```

The first tick is `interval_minutes` after the `PUT`; to start now,
`curl -s -H "$AUTH" -X POST $API/v1/schedule/trigger | jq`. Once a run leaves none of the sample to do, the schedule
turns itself off: `GET /v1/schedule` shows `enabled: false` and `ended` says why.

While it runs, `GET /v1/status` shows `current_job.progress` with `per_hour`, `remaining_seconds`
and `estimated_finish`. Note them after the first hour and at the end.

**4.4 Break the designer.** During a run: `docker stop designer`, wait 10 minutes,
`docker start designer`. **Expect:** titles designing at the time are `unavailable`, not failed;
after three in a row the job stops with "stopped after 3 titles in a row ..."; `/ready` stays
`ready: true` with `designer_reachable: false`; the next tick is skipped
(`schedule.last_skip: designer unavailable: ...`) and one `failed` notification is sent if
notifications are set up; within 5 minutes of the designer's return the schedule resumes and those
titles are designed. Nothing is listed as *failed* in the work list afterwards.

**4.5 Drop the media.** During an extraction, take the share away from the container's view: on
the host, `sudo umount -l $FILMS` (remount it after 10 minutes). **Expect:** the title in hand is
`unavailable` ("the media storage is not available (... is empty: is it mounted?)"), the run stops
after three, nothing is remembered as failed, and the next tick after remounting carries on. If the
mount point is not empty when unmounted, note what happened instead.

**4.6 Review from the desktop while it runs** (R6). On the desktop, with `$SHARE` mounted:
- **Review Folder** on `$SHARE/queue` during a run: open, accept and reject a few titles.
  **Expect:** posters show (found in each title's folder), projects open.
- **The work list** only once `GET /v1/status` shows no `current_job`: open it, check the counts,
  open a title. **Expect:** it reads the container's index; a title designed by the container is
  not shown as needing design again.
- Note whether reading the index over the share while the container wrote it ever failed.

**4.7 Measurements**, once the sample is designed:

```bash
# per-title extract and design times from a run job's log
curl -s -H "$AUTH" "$API/v1/jobs/<id>/log" | jq -r '
  [.[] | select(.type=="event" and (.kind=="stage_started" or .kind=="stage_completed"))]
  | group_by(.title_id + .stage)[] | {id: .[0].title_id, stage: .[0].stage,
    seconds: ((.[-1].at | sub("\\.[0-9]+";"") | fromdateiso8601) - (.[0].at | sub("\\.[0-9]+";"") | fromdateiso8601))}'
du -sh $E2/work; du -s $E2/work/* | sort -n | tail -5
```

Record the five longest designs (R4), the per-title disk use with Keep multichannel (R5), and the
estimate the status gave against the real finish (R9).

## 5. The desktop and command line (formerly T2–T4)

With the service **stopped** (`docker stop pipeline`), so nothing else holds the work folder:

| Step | Expect |
|---|---|
| Work List > Settings: sources, browse node picker (*Browse...*), ignore rule with its live count | the picker lists MC's views; an ignore rule says how many titles it would ignore |
| Rescan | counts per need; *last scan* updates |
| Open the work list again after 12 hours (or set the clock forward) | it rescans by itself |
| Title page > Revise > Choose audio stream on a J2 title | the list describes every stream (codec, layout, language, rate, bitrate); the current one is JRiver's |
| Choose another stream, run Extract & design | Run Details: "Extracting audio stream N: ... (your choice)"; a rescan keeps the choice |
| Metadata tab: TMDB reload, artwork download | fields fill; the poster shows |
| Projects: open mono and multichannel | the main window opens them |
| Accept a few, *Publish N*, then *Commit N* | records and images in `$E2/repos/*`, commits pushed to `$E2/remotes/*.git` only |
| After publish: `ls $E2/work/<title folder>` | `multichannel.flac`, no `multichannel.wav`; reopening the multichannel project still works |
| Revise > Re-extract: A Star Is Born, Arrietty, Bande A Part | Details names the feature's codec (DTS-HD MA for A Star Is Born), not the logo's AC-3 |
| RoboCop | fails: "The main title of ... cannot be read, an incomplete rip?" |
| A TV season in season mode | one entry for the season, episodes listed |

Then the command line, from a checkout (`PYTHONPATH=src/main/python uv run python -m pipeline.library.cli ...`)
with the desktop profile (`--profile <file>`): `scan`, `status`, `run --needs extract --needs design
--through design --year <one year>`, `publish`, `commit` (or `sync`). Record each command and its
exit code. **Expect:** exit 0, the same outcomes as the work list; `run` exits 1 and says so if a
title failed or was unavailable.

## 6. Teardown

```bash
docker rm -f pipeline designer && docker network rm beq-e2
```

Keep `$E2` until the record is written. Remove the disposable repositories afterwards; nothing in
them was pushed anywhere public.

## Record template

```markdown
# E2 record, <date>

Versions: pipeline 2.2.0-alpha.1 (3ec6b5d), designer 0.2.0, desktop <version>, MC <version>,
Docker <version>, host <OS>, desktop <OS>, ffmpeg <version>.
Sample: <n> titles (<n> files, <n> Blu-ray, <n> DVD, <n> TV seasons).

| Step | Result | Notes |
|---|---|---|
| 3 ready/status | pass/fail | |
| 4.2 first batch | | |
| 4.3 schedule, estimate vs actual | | first-hour estimate <t>, finished <t> |
| 4.4 designer restart | | |
| 4.5 media dropped | | |
| 4.6 desktop review during the run | | index over <SMB/NFS>: <ok/errors> |
| 4.7 longest designs | | <title: seconds> x5 |
| 4.7 disk use | | <MB per title>, FLAC saving <%> |
| 5 (one row per step) | | |
| CLI commands | | <command> -> exit <n> |

Findings (each a TODO item or a fix): ...
Stream choices (J2) and disc resolutions (E4) that were wrong: ...
E3 decision (DVD fallback right?): ...
E5 evidence (season levels, layouts): ...
```
