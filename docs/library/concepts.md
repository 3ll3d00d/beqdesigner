The work list uses a small number of ideas. Once you know them, the buttons make sense.

### Titles

A **title** is one unit of work in the catalogue: a film, or a TV episode, or (if you choose it) a whole season. Each title has a row in the work list.

* Films are one title each.
* TV shows can be handled either as **one filter per episode** (the default, each episode is a title) or as a **whole season as a single track**, where the episodes are joined into one track and the season is one title. This is the *TV shows* option in [Settings > Locations](setup.md#locations). Changing it later is safe: a row that already has a review entry but whose episodes are now grouped differently is kept and labelled *superseded* (see [Discover](discover.md#flags)); one with no entry is simply replaced by the new row.
* A disc rip (a folder containing `BDMV` for Blu-ray or `VIDEO_TS` for DVD) is one title, and its main feature (the longest playlist or title) is what is used.

Every title has a stable **id** such as `jriver-3fa9c2-1234` or `fs-8d1c04a2b9e7f310`. The id names the title's work folder, its review entry and its file in the catalogue (`<id>.xml`). It does not change when you reorder sources, rename the title or fix its metadata, so nothing is ever orphaned or published twice. You will see it in a row's tooltip and in the search box, which matches ids.

### Stages

Every title moves through five stages. Extract and design are done by the machine, review is done by you, and publish and commit put the result into your repositories.

| Stage | What it means | States |
|---|---|---|
| Extract | the audio has been extracted from the movie file (a mono mixdown, plus the multichannel audio if you keep it) | none, current, stale (the file or the settings changed), failed |
| Design | the designer has proposed candidate filters | none, current, stale, failed, protected (accepted or published: never redesigned by a run) |
| Review | a person has decided | pending, accepted, skipped, rejected |
| Publish | the accepted filter has been written into the local XML (and images) repository | not written, written, out of date |
| Commit | the written files have been committed, and pushed | uncommitted, committed, pushed |

You do not normally think about states. The work list turns them into a single answer per title, called **needs**.

### Needs and tiers

**Needs** is the one next thing a title requires, and it is what the strip of buttons at the top of the work list counts and filters by. The needs fall into tiers, and the work list sorts by tier first (attention at the top) and then by how long the title has been waiting, oldest first.

| Tier | Needs | Meaning |
|---|---|---|
| attention | **Attention** | something went wrong that the machine cannot get past: extraction or design failed, the [mono and multichannel projects disagree](review.md#projects), or the source file changed after you accepted or published the title |
| human | **Review** | designed, and waiting for you (this includes a title the designer declined, and an accepted title whose metadata is incomplete) |
| machine | **Extract** | the audio has to be extracted (new, or the source or settings changed) |
| machine | **Design** | extracted, and a filter has to be designed |
| machine | **Publish** | accepted, and not yet written to the repository, or written and now out of date |
| machine | **Commit** | written, and not yet committed and pushed |
| done | **Done** | pushed, skipped, rejected, ignored, shadowed by another source, or gone from its source |

Two extra buttons on the strip are not needs. **All** is everything except Done (the default view, so finished work never crowds out what is left to do). **New** is the titles that the most recent scan saw for the first time.

!!! info
    A *Done* title comes back into the list by itself only if its **source file changes** (it then needs attention, because the filter you accepted was designed on the old file). A change to your *settings*, for example another designer, does **not** put finished titles back: that would put the whole library in the list at once. Instead the work list shows a single banner, [described in Revise](revise.md#the-settings-changed-banner), that lets you decide.

### Flags

A few labels appear next to a title's detail. They are extra information, not needs. See [Discover](discover.md#flags) for what each one means: *Ignored*, *Shadowed*, *Gone*, *Superseded*, *Possible duplicate* and *Already in catalogue*.

### The review gate

Extraction and design can run unattended. Review cannot. **A title is never published because a machine decided it was good enough**:

* the *Extract & design* button and the [unattended commands](unattended.md) stop at design. They never accept, publish or commit.
* only an *accepted* title is published, and only a *published* title is committed.
* [bulk accept](review.md#bulk-accept) exists, but it is you choosing to accept a whole set on a confidence threshold that you set, after a confirmation that lists what will and will not be accepted.

### Where things are kept

| What | Where | Notes |
|---|---|---|
| the **profile** | a YAML file, [described here](setup.md#the-profile-file) | the durable record of what makes up the catalogue |
| extracted audio and `.beq` projects | the **work directory**, one folder per title | the audio can be large, but the folder also holds the `.beq` projects that carry [any hand edits you made to a filter](review.md#projects), so do not clear it casually |
| the discovery index | `library-index.sqlite` in the work directory | a cache of the last scan. Deleting it costs a [rescan](discover.md) and nothing else |
| designed titles waiting for a decision (and their decisions) | the **review queue directory**, one `<id>.json` per title | this is where your accept, skip and reject decisions and metadata edits live |
| published files | your local clones of the XML and images repositories | `<id>.xml` and `<id>.png` |

The discovery index is disposable. The profile, the review queue and the repositories are not: back those up.
