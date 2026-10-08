# JRiver MCWS fixture (TODO E1)

Real responses of a JRiver Media Center server, sanitised, replayed by
`test_pipeline_library_jriver_fixture.py`. `capture.py` both captures and
sanitises; rerun it to refresh the fixture.

## Capture, 2026-10-08

- Server: JRiver Media Center 36.0.38 on Windows (`/Alive`'s `ProgramVersion`,
  `Platform`), library version 24. Captured with the owner's authorisation,
  using the login of the library profile's `jriver` source, from the
  maintainer's desktop on the same LAN.
- Requests, as the pipeline makes them: `/Alive`, `Library/Fields`,
  `Browse/Children?Version=2&ErrorOnMissing=0` for the root and every node
  two levels below it (64 nodes), and `Browse/Files?Action=JSON` for the
  source's browse node (1004, *Video > Movies*), with exactly the fields
  `JRiverLibrarySource` requests (`capture.json`). 1,291 rows came back.
- Raw responses were written to a scratch directory outside the repository
  and are not kept.

## Sanitisation

- `/Alive`: `FriendlyName` → `MEDIA-SERVER`, `AccessKey` → `ACCESSKEY`,
  `RuntimeGUID` zeroed. The field set and order are as sent.
- `Library/Fields`: only the fields the source requests or reads.
- `Browse/Children`: only the root, *Video* (3) and *Movies* (1004), the nodes
  on the way to the browse node. Other branches of the library are dropped.
- `Browse/Files`: 132 rows, at most two per combination of path form,
  `Playback Info` structure, image form, ids present, series and multi-stream
  audio (87 combinations), chosen by `Key`. `Description` is dropped from every
  row. One `Playback Info` `JRVRProfiles` record named a display's hardware id;
  it is replaced by `DISPLAY#DEVICE` with its nesting and lengths kept. Titles
  and file names under the media share (`W:\`) are public catalogue names and
  kept as reported. No host, credential or path outside the share remains
  (`test_the_fixture_is_sanitised`).

## What it showed

- Fields: asked for `Year`, MC answers `Date (year)`. A field with no value is
  absent from the row rather than empty: of the requested fields, `Rating`
  (5 rows), `Series` (12), `Season`, `Episode`, `Playback Info` (238) and
  `TheMovieDB Movie ID` (290) are mostly absent; `Length`, `Audio Format` and
  `Edition` never appear. `IMDb ID` (a default id field) is present on 1,252.
  Numbers come back as JSON numbers, not strings.
- Paths: all under the share root, as `W:\` (1,241) or `w:\` (50). Forms:
  `.mkv` 544; `BDMV\index.bdmv` 396; `BDMV\index.bluray;1` 346 (never another
  number); one `.mp4`; one `BDMV\PLAYLIST\00305.mpls` (a short listed from
  another film's disc). There are no `BDMV\PLAYLIST\index.bluray;N`
  pseudo-paths (E4's premise) and no DVDs (E3). Two disc folders are listed by
  more than one entry.
- Audio streams (recaptured the same day once the source asked for them,
  W2): `Audio Codec`, `Audio Channels`, `Audio Sample Rate`, `Audio Bitrate`,
  `Audio Language` and `Audio Title` are semicolon lists in stream order. An
  entry is blank where MC has nothing: a lossless stream has no bitrate, and
  titles are mostly blank (`;;;;`).
- Artwork: `Image File` is a bare file name beside the media (`index.jpg`) on
  1,289 rows and absent on 2. There is no `INTERNAL` artwork.
- Identity: `Key` is unique per row.
- Browse tree: no node has two children with the same name. The test makes one
  from a real entry to check that both ids survive. Before this change, hamcws
  read `Browse/Children` into a dict keyed by name and kept only the last.
- `Playback Info` (J2, E4): a length-prefixed record, `(1:N)` then N
  name/value pairs, with names including `Streams`, `CC`, `Subtitles`,
  `BlurayPlaylist`, `CenterOffset`, `ZoomPercent`, `AspectRatio` and
  `JRVRProfiles`, in any order (211 with `Streams`). A `Streams` value is
  three comma-separated numbers (`0,1,3`, 205 rows) or two (`0,1`, 6). Which
  container streams they are needs ffprobe on the same files (J2); the media
  share was not mounted where this was captured. `BlurayPlaylist` names an
  `.mpls` (13 rows).
