The **Library Work List** is for anyone who wants to design BEQ filters for a lot of films, not one at a time. It keeps track of every title in a movie library, shows what each one needs next (extract its audio, design a filter, be reviewed by you, be published) and does the machine work for you in the background. It is the way to keep a catalogue of BEQ filters up to date, for example the filter repositories that feed [BEQCatalogue](https://beqcatalogue.readthedocs.io/en/latest/) and [ezbeq](https://ezbeq.readthedocs.io/).

Open it with `Tools > Library Work List`.

![Work list](../img/library_worklist.png)

If you only want to look at one film, use [Extract Audio](../ui/extract_audio.md) and the [main window](../ui/main_window.md) instead. If you have a folder of files and want a filter designed for each of them without setting up a library, use [Batch Extract](../ui/batch_extract.md) and then [Review Folder](review.md#review-folder). Everything in this section is about the library-scale workflow.

!!! info
    The library workflow needs three things that the rest of BEQDesigner does not:

    * a **designer**: an HTTP service that proposes filters for an audio track, added in [Preferences > Designers](../ui/preferences.md#designers). BEQDesigner does not design filters by itself in this workflow, it asks the designer.
    * [ffmpeg](../ui/preferences.md#binaries), to extract the audio.
    * for publishing, a local clone of the git repository that holds your BEQ filters (see [Set up a catalogue](setup.md#the-two-repositories)).

### The workflow at a glance

Each title goes through the same stages, and each stage is done by the machine or by you.

| Stage | Who | What happens | Page |
|---|---|---|---|
| Set up | you, once | choose where the library is, where files go, which repositories to publish to | [Set up a catalogue](setup.md) |
| Discover | machine | list the library and work out what each title needs | [Discover](discover.md) |
| Extract and design | machine | extract the audio with ffmpeg and ask the designer for candidate filters | [Do the work](work.md) |
| Review | **you** | look at the candidates, pick one (or skip or reject the title), complete the metadata | [Review](review.md) |
| Publish | machine | write the accepted filters into your local repositories | [Publish and commit](publish.md) |
| Commit | machine | commit and push those repositories | [Publish and commit](publish.md) |
| Revise | you, when needed | send a finished title back for another look | [Revise](revise.md) |

The rule that shapes everything is that **nothing is published without a person looking at it first**. The machine stages can run unattended (see [Unattended runs](unattended.md)) but they stop at review. Accepting a filter is always your decision, whether you make it title by title or in bulk with a confidence threshold you set.

### Where things are

| To do this | Use |
|---|---|
| see what needs doing, run the machine work, publish, commit | `Tools > Library Work List` |
| change where things go, the library sources or what to ignore | *Settings...* in the work list |
| review one title in detail | double click a title in the work list |
| review the results of a Batch Extract run | `Tools > Review Folder...` |
| add a designer | `Settings > Preferences`, [Designers](../ui/preferences.md#designers) |
| add a JRiver Media Center server | `Settings > Preferences`, [JRiver](../ui/preferences.md#jriver) |
| run it from a script or cron | [Unattended runs](unattended.md) |

### Getting started

1. Add a designer in [Preferences > Designers](../ui/preferences.md#designers). The work list will not scan without one.
2. Open `Tools > Library Work List`, press *Settings...* and choose the [work directory, the review queue directory and where your library is](setup.md). The first change you make asks where to keep the [profile file](setup.md#the-profile-file).
3. Press *Rescan*. The library is listed and every title appears, with what it needs. See [Discover](discover.md).
4. Press *Extract & design*. The machine work runs for the titles that need it. See [Do the work](work.md).
5. Open a title with *Open* (or double click) and [review it](review.md): choose a candidate and press *Accept & next*.
6. Press *Publish* and then *Commit*. See [Publish and commit](publish.md).

### What is not covered here

This section explains how to use the work list. It does not explain how a designer works or how to judge a candidate filter, which are covered by [the concepts page](../concepts.md) and [A Worked Example](../workflow/beq.md). The command line and the exact file formats are documented for developers in the [pipeline README](https://github.com/3ll3d00d/beqdesigner/blob/main/src/main/python/pipeline/README.md).
