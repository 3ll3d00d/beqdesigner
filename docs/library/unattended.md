The machine stages of the workflow (discovery, extract and design) can run without anyone watching, from cron or a scheduler, so that when you sit down to review the titles are already waiting. Review, publish and commit cannot: they are the [review gate](concepts.md#the-review-gate).

This is done with a command line tool that ships with the source, `pipeline.library.cli`. It is run with Python from a checkout of BEQDesigner (see the [readme](https://github.com/3ll3d00d/beqdesigner/blob/main/readme.md) for setting one up); the release downloads do not include a command for it. It is described in full, with every option, in the [pipeline README](https://github.com/3ll3d00d/beqdesigner/blob/main/src/main/python/pipeline/README.md#library-sync-cli); `-h` after any command lists its options. This page only says how it relates to the work list.

### A scheduled job

A nightly job is three commands:

```
python -m pipeline.library.cli scan   --profile /path/to/library-profile.yaml
python -m pipeline.library.cli run    --profile /path/to/library-profile.yaml --needs extract --needs design --through design
python -m pipeline.library.cli status --profile /path/to/library-profile.yaml
```

Run them with `PYTHONPATH=src/main/python` set, from the root of the checkout, as `uv run` does.

| Command | What it does |
|---|---|
| **`scan`** | the same as the *Rescan* button: list the sources and work out what each title needs. `--source NAME` rescans just one source, which the window cannot do |
| **`run --needs extract --needs design --through design`** | the same as the *Extract & design* button over the machine tier: extract and design what needs it, and stop |
| **`status`** | prints how many titles need each thing (`--json` for a machine-readable form), and what is new. It reads the index and never lists a source, so it is instant. Use it to send yourself *37 titles are waiting for review* |

The profile is **the same file** the work list uses. The work directory, the review queue and the sources come from it, so the job and the window agree about everything, and what the job does shows up in the window: open the work list (or `Tools > Library Work List` again) and it shows what the last scan found.

Things to check:

* The command line does not read Preferences, so **declare the designer in the profile**: add a `designers:` section (`designers: {rolloff: http://designer.local:8080/design}`) and choose that name as the designer. The window reads that section too, so the same profile works in both.
* A JRiver source in the profile carries its own server login (see [the profile file](setup.md#the-profile-file)), so it works from cron too.
* A title whose extraction or design **failed** is not tried again by a nightly job until its source or the settings change, so a failure does not repeat every night. `run` prints *warning: N titles skipped: failed earlier ... use --retry-failed* on stderr when that happens (the exit status is unchanged), and `--retry-failed` tries them again. This is the same as the work list's *Retry failed*.
* **Do not run the command line and the work list against the same work directory at the same time.** They share the index and the review entries.
* `scan` exits with status 1 if a source could not be listed, and `run` with 1 if anything failed, so a wrapper can tell.

### The same words in both places

The command line and the window select titles in the same way. The strip and the action button are the options of the command.

| In the work list | On the command line |
|---|---|
| the strip: *Attention*, *Extract*, *Design*, *Review*, *Publish*, *Commit*, *Done* | `--needs attention`, `extract`, `design`, `review`, `publish`, `commit`, `done` (repeat the option for more than one) |
| *New* | `--new-since-scan` |
| the source drop-down | `--source NAME` |
| the search box | `--match TEXT` |
| the rows you selected | `--id ID` (repeat for more than one) |
| *Extract & design* | `--through design` |
| *Retry failed* | `--retry-failed` |

Every option you give must hold at once. As in the window, `--through design` does everything up to and including design that the title still needs, and a title that is waiting for review is never taken past design.

### What to leave to a person

**Do not schedule `accept`, `publish`, `commit` or `sync`.** They exist for a person at a terminal, for example to bulk accept confident titles (`accept --dry-run` shows what it would do) or to publish and commit what you accepted in the window, and they are the same steps as *Accept top pick*, *Publish* and *Commit*. Each of them is a decision that a person is meant to make, and the unattended part of the workflow ends when it has designed the titles.
