# L2: extraction failures in the first full library run (done 2026-10-19)

From [the library run plan](../library-run-gaps.md). Nine titles failed extraction on every scheduled run. Each was
reproduced against the real media on the films share before the fix, and checked again after it.

| Titles | Failure | Cause | Now |
|---|---|---|---|
| Elf | `KeyError: 'codec_type'` | the disc's ninth stream comes from ffprobe with no `codec_type`, and `Executor.probe_file` indexed it | streams are read with `.get`; it probes (AC-3 5.1, TrueHD 5.1, ...) |
| Le Samourai, Parasite, House Party (D1), Lost Highway | `ZeroDivisionError` | on these UHD discs ffprobe's default 5 s / 5 MB window ends before the audio's first frames, so every audio stream came back with no channel count, and the equal-weight mono mix divided by it | a probe that finds such a stream is run again with `analyzeduration=100M`, `probesize=200M`, kept for the extraction; all four then report their real channel counts (Le Samourai: LPCM mono). A stream still without one is refused in words |
| Prizzi's Honor, Silkwood | `ffmpeg error` | AC-3 with layout `4 channels (FL+FR+LFE+BC)`, which the layout table did not name, so the mono mix was `None` (`pan ... Expected in channel name, got "None"`) | a layout ffmpeg describes by its channels is parsed (`layout_channels`): mixed with the LFE weighted where it is, and its channels named FL/FR/LFE/BC; a layout naming no channels is mixed from the channel count |
| John Wick: Chapter 3, The Dark Knight (`_bak`) | `ffprobe error` | AACS-encrypted backups (1 transport-stream sync byte in 32 per aligned unit) | refused with "is AACS-encrypted ... decrypt the rip first" (`model.bdmv.is_aacs_encrypted`) |
| House Party (`House_Party.mkv`) | `ffprobe error` | the file JRiver lists is gone (the title now lives in `D1`/`D2`) | not code: the item goes when JRiver stops listing it |

Every ffmpeg/ffprobe failure a run records now carries the last lines ffmpeg wrote to stderr, banner left out
(`pipeline.library.failure.failure_message`), where it used to say only "see stderr output for detail".

A failure is remembered against the source and settings, not the code, so these titles are retried only by a run with
`retry_failed`, once the service runs this build.

Tests: `src/test/python/model/test_ffmpeg_probe_failures.py` (each case above, including a real AC-3 track with the
custom layout extracted to mono and to its channels), `test_pipeline_library_failure.py` (the failure text),
`test_pipeline_library_run.py::test_an_ffmpeg_failure_is_remembered_with_what_ffmpeg_said`. Full suite: 3134 passed,
1 skipped.
