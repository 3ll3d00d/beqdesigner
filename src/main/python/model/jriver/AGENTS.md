# `model/jriver` — JRiver Media Center DSP config integration

This module reads and writes JRiver Media Center (JRMC) "DSP Studio" preset
files (`.dsp`) and MCWS-served DSP presets, translating between JRMC's own
on-disk/on-wire format and BEQDesigner's `Filter`/`FilterGraph` object model.
It is the only place in the codebase that understands the JRMC file format,
so it is the module to read/update whenever JRiver changes that format in a
new MC release.

Read this before touching parsing, encoding, or channel/format logic here —
the format has several non-obvious layers and a couple of sharp edges that
are easy to reintroduce a regression in.

## Module map

| File | Responsibility |
|---|---|
| `dsp.py` | `JRiverDSP` — top level orchestrator. Parses a whole `.dsp` config into one `FilterGraph` per enabled PEQ block, and serializes graphs back into config text. |
| `codec.py` | Low level, format-only helpers: XPath lookups into the `Preset` XML, the `(length:content)` micro-format used inside `<Value>` text, and splicing new filter text back into the original document. No knowledge of `Filter` classes. |
| `filter.py` | The `Filter` class hierarchy (one class per JRMC filter `Type` code), `ComplexFilter`/`Divider` protocol for grouping filters, `FilterGraph`, the crossover/bass-management synthesis engine (`MultiChannelSystem`, `MultiwayFilter`, `XO`/`MDSXO`), and `FilterOp` (in-process simulation of what JRMC's DSP engine would do to a signal). |
| `formats.py` | Channel name/index tables and `OutputFormat` (maps JRMC's "Output Channels"/"Output Padding Channels"/"Output Channel Layout" trio to a named speaker layout). |
| `routing.py` | `Matrix` — an editor-side input/way/output routing model used when the user designs a crossover in the UI. Independent of the XML format; feeds into `MultiChannelSystem`. |
| `render.py` | Graphviz `dot` rendering of a `FilterGraph`, for the UI's visual filter chain. Not part of the round-trip path. |
| `mcws.py` | `MediaServer` — talks to a running JRMC instance over the MCWS HTTP API (auth, zones, get/set DSP preset) and includes JRMC's own tolerant XML-equality check (`__compare_xml`) used after pushing a preset, as a live sanity check that JRMC accepted it unchanged. |
| `parser.py` | Adapters that turn *other* filter sources (MiniDSP exports, MSO exports) into JRMC `Filter` objects via `convert_filter_to_mc_dsp`. |

## The file format, from the outside in

A `.dsp` file is real XML, but only down to a point:

```
<Preset>
  <Key Name="Audio Settings">
    <Data><Name>Output Channels</Name><Value>6</Value></Data>
    <Data><Name>Output Padding Channels</Name><Value>0</Value></Data>
  </Key>
  <Key Name="Parametric Equalizer">
    <Data><Name>Enabled</Name><Value>1</Value></Data>
    <Data><Name>Filters</Name><Value>(1:1)(2:0)...</Value></Data>
  </Key>
  <Key Name="Parametric Equalizer 2"> ... </Key>
  <Key Name="DSP Studio">
    <Data><Name>Plugin Order</Name><Value>(...)(Parametric Equalizer)(...)</Value></Data>
  </Key>
</Preset>
```

Every setting lives at `/Preset/Key[@Name=X]/Data/Name[.=Y]/../Value`
(`codec.xpath_to_key_data_value`). There are up to **two independent PEQ
("Parametric Equalizer") blocks**, block 0 = `Parametric Equalizer`, block 1
= `Parametric Equalizer 2` (`codec.get_peq_key_name`). Only PEQ blocks with a
sibling `Enabled` `Data` set to `"1"` are active (`codec.get_peq_block_order`);
if both are active, the `DSP Studio`/`Plugin Order` value's token order
decides which one runs first (i.e. which one's *output* channels become the
other's *input* channels) — `JRiverDSP` builds one `FilterGraph` per active
block in that order, but each graph's `.stage` records the real block number
(0 or 1), not its position in the list. **Don't confuse graph list index
with `.stage`** — `config_txt()` writes back using `.stage`, and a reversed
plugin order is a real, if unusual, configuration.

Inside a PEQ block's `<Value>`, JRMC uses its **own length-prefixed
micro-format**, not further XML: a sequence of `(length:content)` tokens.
The first two tokens are a fixed header (`(1:1)(N:count)` where `count` is
the total number of filter entries that follow); everything after that is
one `(length:<XMLPH version="1.1">...</XMLPH>)` token per filter, where the
inner `<XMLPH>` blob is itself a small XML document of `<Item Name="K">V</Item>`
pairs — see `codec.filt_to_xml`/`codec.item_to_dicts`. So a single `<Value>`
text node is: real XML → custom length-prefixed text tokens → embedded XML
again. `codec.extract_filters`/`codec.include_filters_in_dsp` are the only
places that touch this middle layer.

Each `<Item>` dict has a `Type` code identifying a `Filter` subclass in
`filter.py` (`filter_classes_by_type`, keyed by `Filter.TYPE`):

| Type | Class | Notes |
|---|---|---|
| 1 | `LowPass` | order 1/2 native; `Slope` encodes order×6 |
| 2 | `HighPass` | as above |
| 3 | `Peak` | parametric EQ band |
| 4 | `Gain` | |
| 5 | `Mute` | |
| 6 | `Mix` | add/copy/move/swap/subtract between two channels |
| 7 | `Delay` | |
| 8 | `LinkwitzTransform` | |
| 9 | `Limiter` | |
| 10 | `LowShelf` | |
| 11 | `HighShelf` | |
| 12 | `Order` | channel reorder |
| 13 | `BitdepthSimulator` | |
| 14 | `SubwooferLimiter` | |
| 15 | `Polarity` | |
| 16 | `LinkwitzRiley` | rarely produced by BEQD; parsed for completeness |
| 17 | `AllPass` | |
| 18 | `MidSideEncoding` | |
| 19 | `MidSideDecoding` | |
| 20 | `Divider` | see below — the load-bearing one |

## The `Divider` protocol — how BEQD's compound filters survive JRMC

JRMC only understands a **flat list** of atomic filters; it has no concept
of BEQDesigner's higher level objects (a multiway crossover, a GEQ, a
compound bass-management routing block, an odd-order/Bessel pass filter
built from several biquads). BEQD fakes grouping by writing plain JRMC
`Divider` (type 20, normally just a visual separator in JRMC's own UI) filters
with a specially-formatted `Text`:

```
***<CustomType>_START|<metadata>***
  ... child filters ...
***<CustomType>_END|<metadata>***
```

`ComplexFilter.custom_type()`/`.create()` per subclass
(`GEQFilter`='GEQ', `MultiwayFilter`='Multiway', `XOFilter`='XO',
`CompoundRoutingFilter`='XOBM', `CustomPassFilter`='PASS',
`MSOFilter`='MSO') define the marker and how `metadata` (a JSON blob for
`MultiwayFilter`, a slash-delimited string for `XOFilter`/`CustomPassFilter`)
round-trips. Parsing is a single-pass stack machine in
`dsp.JRiverDSP.__extract_custom_filters`/`__handle_divider`: unrecognised
divider text is passed through as a plain filter, whether it's nested inside
a recognised complex filter's start/end pair (appended to the current
buffer) or sitting outside any BEQD complex-filter block entirely (appended
to the top-level filter list) — e.g. a user's own "---" separator added
directly in JRMC survives a round trip like any other native-only filter.
See `test_native_divider_outside_complex_filter_is_preserved`.

`complex_filter_classes_by_type`/`filter_classes_by_type` are built via
`all_subclasses()` reflection over `ComplexFilter`/`Filter` — **a new
`Filter`/`ComplexFilter` subclass registers itself automatically just by
existing**, so a new JRMC filter type or a new BEQD compound filter only
needs a class added here, not a registry edited.

## Round-trip mechanics (`JRiverDSP`)

- **Parse**: `JRiverDSP.__init__` reads `Audio Settings` for the
  `OutputFormat` (`codec.get_output_format`), determines active PEQ blocks
  and their order, then per block: `codec.extract_filters` gets the raw
  `<Value>` text, splits it into `(length:content)` tokens, decodes each into
  a dict (`codec.item_to_dicts`), builds a `Filter` (`filter.create_single_filter`),
  and folds the flat list into a tree via the `Divider` protocol above.
- **Write**: `JRiverDSP.config_txt()` walks each `FilterGraph`, re-encodes its
  top-level filters (`Filter.get_all_vals()` → `codec.filts_to_xml`), and
  splices the result back into the **original** document text via
  `codec.include_filters_in_dsp`.

### Known non-invariants — don't assert byte-identical round trips blindly

1. **Numeric normalisation is intentional.** Every numeric field is parsed
   once (`s2f`, which also tolerates a comma decimal separator) and
   re-emitted with a fixed `%g` precision (`.7g` for frequency/gain/delay,
   `.12g`/`.4g` for Q depending on filter type). A value with more precision
   than that in the source file will be truncated on the first write. This
   is expected; round-trip tests should compare parsed `Filter.get_all_vals()`
   (or use values already at target precision), not raw text.
2. **Whole-document re-serialization.** `include_filters_in_dsp` only
   rewrites a block's `<Value>` text, but it does so by parsing the *entire*
   document with `xml.etree.ElementTree` and calling `et.tostring()` on the
   whole tree — this happens once per PEQ block **that has at least one
   filter**. Untouched elements survive *semantically* (same tags,
   attributes, text) but not necessarily byte-for-byte (XML declaration
   quoting, self-closing empty-element style can change). If **no** PEQ
   block has any filters, `config_txt()` is a complete no-op and returns the
   original text unchanged (`include_filters_in_dsp` short-circuits on an
   empty `xml_filts` list) — that path *is* byte-identical.
3. **`Item` key order is canonicalised, not preserved.** Re-encoding always
   uses each `Filter` subclass's fixed `key_order`, regardless of what order
   the source file's `<Item>`s were in. Round trip is a stable fixed point
   from the *second* write onwards, not necessarily on the first.
4. **Newlines.** Files are read with `Path.read_text()` (universal newline
   translation to `\n`) and written with `newline='\r\n'`
   (`JRiverDSP.write_to_file`). `config_txt()` itself returns `\n`-terminated
   text (`ElementTree`'s default) — callers that send it elsewhere (e.g.
   `mcws.MediaServer.set_dsp`) must convert `\n` → `\r\n` themselves, and do.

### A live sharp edge: `convert_q` is not remembered by the instance

`JRiverDSP` takes `convert_q` at **parse** time (decodes MC28-style shelf
"S" parameters and 2nd-order pass-filter Q into true Q using
`iir.s_to_q`/`q_to_s` and `Pass.from_jriver_q`), but `config_txt()` takes an
**independent** `convert_q` argument at **write** time, defaulting to
`False`. `JRiverDSP` does not remember how it was parsed. Callers must pass
matching flags themselves:

- `mcws.MediaServer.set_dsp` does this correctly — it derives `convert_q`
  from `mc_version` at push time and passes it to the `txt_provider`.
- `ui.py`'s plain "Save DSP File" path (`self.dsp.write_to_file()`) calls
  `config_txt()` with **no arguments**, i.e. `convert_q=False`, regardless of
  the `convert_q` the file was originally loaded with. Loading a legacy
  MC28 file (`convert_q=True`) and saving it via this path re-encodes Q
  values without converting back to the "S" convention — silently
  corrupting shelf/pass-filter Q for MC28. This is a real, currently
  unfixed bug, pinned by
  `test_convert_q_write_default_does_not_match_legacy_parse` as documented
  (not endorsed) behaviour.

## Output format / channel layout

`formats.OUTPUT_FORMATS` enumerates named layouts (`STEREO`, `FIVE_ONE`,
`SEVEN_ONE`, the MC35 immersive layouts `FIVE_ONE_TWO`/`SEVEN_ONE_FOUR`/
`NINE_ONE_SIX`, padded variants, etc). `codec.get_output_format` decodes
`Audio Settings` into one of these:

- **Legacy path** (`allow_padding=False`, roughly MC ≤ 28):
  `codec.get_legacy_output_format` — a fixed lookup table keyed on
  `(output_channels, padding, layout)`.
- **Padded path** (`allow_padding=True`, MC ≥ 29, `MediaServer.can_pad_output_channels`):
  arbitrary channel counts with an explicit padding count, templated off the
  legacy formats.

`mcws.MediaServer.mc_version`/`can_pad_output_channels`/`convert_q` gate
which path/convention is active for a live JRMC connection; `ui.py` derives
the same booleans from a user-facing "legacy" checkbox when working from a
file rather than a live connection.

### Channels beyond the base 8: legacy numbered vs MC 35.0.39+ Atmos + Extra

Indexes 2-9 (base 8 surround) and 11-12 (user channels) never change. Beyond
that, JRMC has had (at least) three eras of naming for the "extra" channels a
>8-channel format needs, all still present in `formats.py` since raw
*indexes* never get remapped between JRMC versions — only which ones are
reachable via JRMC's own UI/picker changes:

| Indexes | Name | Works from | Notes |
|---|---|---|---|
| 13-36 | `Channel 9`..`Channel 32` (`NUMBER_CHANNELS`) | always | legacy generic numbering |
| 54-57 | `LTF`/`RTF`/`LTR`/`RTR` (Atmos, first 4) | ~MC34 | JRiver's own short codes don't match its long names (`'Left Height Front'` etc) — `SHORT_ATMOS_CHANNELS` mirrors the short codes verbatim, not the "logically correct" ones derived from the long name |
| 58-61 | `LTM`/`RTM`/`LW`/`RW` (Atmos, remaining 4, completing 9.1.6) | **MC 35.0.39** | broken before 35.0.39 — selecting them in the JRMC UI just re-picks 56/57 |
| 37-52 | `X1`..`X16` (`EXTRA_CHANNELS`) | **MC 35.0.39** | unreachable before 35.0.39 — selecting them falls back to L/R |

The true cutover is the **35.0.39 point release**, not major version 36 —
`mcws.ATMOS_CHANNELS_MIN_VERSION = (35, 0, 39)` is the source of truth,
compared against the full parsed `ProgramVersion` semver
(`mcws.parse_version`), not just the major version. A prior pass through this
doc generalized from the two captures below (which happen to bracket
35.0.39 but don't pin it down themselves) to "MC36" everywhere; that's now
corrected throughout this file.

Confirmed empirically (2026-07) from two real captures, one PeakingEQ filter
per channel as a marker, `all_35.dsp` (MC 35.0.38 — just *before* the 35.0.39
cutover) vs `all_36.dsp` (MC 36.0.14 — well after it) — scrubbed copies live
at `src/test/python/model/jriver/resources/mc{35,36}_all_channels.dsp`.
**JRiver itself** never remaps a raw index between versions — everything
above just becomes newly reachable, and the file format's meaning of any given
raw index is stable. That's a separate question from whether **BEQD** should
carry an existing filter's *role* (e.g. "the 3rd scratch/bass-management
channel") forward onto the new pool when reinterpreting an older config with
`use_atmos_channels=True` — it should, and does: see
`OutputFormat.migrate_channel_index` further down. So reading a pre-35.0.39
file and writing it back with `use_atmos_channels=True` *does* rewrite any
legacy-pool filter channels it finds, by design.

### What each output format *shape* actually uses, pre- vs post-35.0.39

The table above is by raw index range; this one is by the shape of format a
user actually picks, since that's what determines which pool(s) apply. "✔
real capture" means a real `.dsp` (or, for the JRiver-Analyzer row, a live
JRiver Analyzer screenshot cross-checked against one) backs the row; anything
else is inferred from `formats.py`'s static tables/`OUTPUT_FORMATS` design
intent, not independently verified against a real JRiver instance.

| Format shape | ≤ MC33 | MC34 - 35.0.38 | 35.0.39+ | Evidence |
|---|---|---|---|---|
| Base 8 (mono..7.1, idx 2-9) | L/R/C/SW/SL/SR/RL/RR | same | same | always true, never version-gated |
| User channels (idx 11-12) | U1/U2 | same | same | always true, additive to any format |
| Generic `"N.N + N padding"` (`codec.get_output_format`'s padded branch — e.g. 7.1+2, 7.1+8, 5.1+8) | legacy numbered, `Channel 9`.. (idx 13-36) | same | **Extra only**, `X1`.. (idx 37-52) — legacy pool NOT used, Atmos NOT used | ✔ real capture both sides: `mc35_mixed_channel_eras.dsp` (7.1+8 on MC 35.0.38, `C9`-`C13` used for real bass management) and `mc36_seven_one_plus_two_padding.dsp` (7.1+2 on MC 36.0.14, Peak filter + live Analyzer confirm `X1`/`X2`) — the actual 35.0.39 cutover itself hasn't been captured on either side, see caveat below |
| Static named immersive layout height/width slots (`FIVE_ONE_TWO`/`SEVEN_ONE_FOUR`/`NINE_ONE_SIX`, and the `THIRTY_TWO` static entry) | n/a (didn't exist) | first 4 Atmos slots only (`LTF`/`RTF`/`LTR`/`RTR`, idx 54-57) reachable from ~MC34; remaining 4 (idx 58-61) broken, re-pick 56/57 | `FIVE_ONE_TWO` (5.1.2): 5.1 bed (no `RL`/`RR`) + `LTF`/`RTF` only, 8 total. `SEVEN_ONE_FOUR`/`NINE_ONE_SIX`/`THIRTY_TWO`: full 7.1 bed + Atmos (then Extra for `THIRTY_TWO`) | ✔ real JRiver Analyzer review, one format at a time (2026-07): 5.1.2=`LTF`/`RTF`; 5.1.4/7.1.4=`LTF`/`RTF`/`LTR`/`RTR`; 9.1.6=`LW`/`RW`/`LTF`/`RTF`/`LTM`/`RTM`/`LTR`/`RTR` (all 8, different *display* order than `SHORT_ATMOS_CHANNELS` but same raw index set); 32ch=9.1.6 set + Extra appended. Settles open question 2 below. |
| Static MC28-only fixed layouts (`TEN`/`TWELVE`/`FOURTEEN`/`SIXTEEN`/`EIGHTEEN`/`TWENTY`/`TWENTY_TWO`/`TWENTY_FOUR`, and `STEREO_IN_*`/`FIVE_ONE_IN_SEVEN`) | legacy numbered | n/a — `OutputFormat.is_compatible()` only offers these for `version == 28` (`paddings` is empty), so BEQD's own `ui.py` flow never reaches them on 34+ | not offered by BEQD regardless of what real JRiver would do with them | not independently tested post-35.0.39 — moot only because of BEQD's own version gate, not because JRiver itself would necessarily behave this way |
| Extra channels (idx 37-52) reachability, standalone | unreachable — picking one falls back to L/R | unreachable | reachable | ✔ `mc35_all_channels.dsp` (35.0.38) vs `mc36_all_channels.dsp` (36.0.14) |
| Atmos remaining 4 (idx 58-61), standalone | broken/unreachable | broken — re-picks 56/57 | reachable | ✔ same two captures |

**Caveat on the "35.0.39" number itself**: the two real captures backing this
table bracket 35.0.39 (35.0.38 just below it, 36.0.14 well above it) but
neither one *is* 35.0.39, so they can't independently pin down that exact
point release as the cutover — that figure comes from the user, not from
BEQD's own captures. `mcws.ATMOS_CHANNELS_MIN_VERSION` encodes it as the
source of truth; if a future capture on an actual 35.0.3x install disagrees,
trust the capture and update the constant.

One open question left, one now settled:
1. **Open**: whether the "New Extra Channels System" ini/options toggle
   (mentioned above the `JRIVER_NAMED_CHANNELS` table) gates Extra-channel
   reachability *independently* of version, or whether it's simply
   on-by-default from 35.0.39 onward — not yet tested with the toggle
   deliberately flipped.
2. **Settled (2026-07)**: picking a real named format from JRiver's own
   Output Format wizard does *not* always assign Atmos channels via a simple
   "first N of a fixed 8-slot order" prefix, contrary to the prior assumption.
   `FIVE_ONE_TWO` in particular *drops `RL`/`RR` entirely* rather than
   appending its 2 height channels on top of a full 7.1 bed — a real bug, not
   just an ordering nuance: with the old fixed 8-channel base,
   `output_channels=8` for 5.1.2 was already exhausted by the base alone, so
   `get_output_channel_indexes(use_atmos_channels=True)` silently returned
   zero Atmos channels. Fixed by making the base size format-aware
   (`OutputFormat.__init__`'s new `base_channels` param, 6 for `FIVE_ONE_TWO`,
   8 for everything else) — see `get_all_channel_names`'s
   `base_channel_count`. `SEVEN_ONE_FOUR`/`NINE_ONE_SIX`/`THIRTY_TWO` turned
   out already correct: they keep the full 7.1 bed, and since they consume
   Atmos slots in full multiples of what `SHORT_ATMOS_CHANNELS` already
   provides as a prefix (4 of 8, or all 8), the *display* order JRiver's
   Analyzer showed (`LW`/`RW`/`LTF`/`RTF`/`LTM`/`RTM`/`LTR`/`RTR` for 9.1.6,
   vs. code's `LTF`/`RTF`/`LTR`/`RTR`/`LTM`/`RTM`/`LW`/`RW`) doesn't change
   which raw indices get selected, since `get_output_channel_indexes` sorts
   its result and 9.1.6/32ch consume the entire 8-slot pool regardless of
   internal order. See `test_five_one_two_drops_rear_surrounds_for_atmos_height_channels`
   and `test_seven_one_four_and_nine_one_six_keep_full_seven_one_bed` in
   `test_dsp_roundtrip.py`.

   **The same bug existed a second time**, independently, in the dynamically
   padded path: `codec.get_output_format`'s padded branch (a real "N.N + N
   padding" file, e.g. 5.1+10) builds a `template=False` `OutputFormat` from
   whichever static template matches the file's pre-padding channel count
   (`OutputFormat.from_output_channels`), but never passed that template's own
   bed size through as `base_channels` — so a 5.1+10 file always got the fixed
   8-channel 7.1 base regardless of the real underlying format being 5.1.
   Confirmed the same way: a real 5.1+10 capture uses the 6-channel 5.1 bed
   (`L/R/C/SW/SL/SR`, no `RL`/`RR`) plus 10 Extra channels (`X1`-`X10`), not an
    8-channel base plus only 8 Extra. Fixed by passing
   `base_channels=template.output_channels` through in `codec.py` — see
   `test_five_one_plus_padding_uses_five_one_bed_not_seven_one`. This is a
   distinct call site from `FIVE_ONE_TWO`'s fix above (one is a static
   `OUTPUT_FORMATS` entry, the other a dynamically-constructed instance), so
   fixing one did not fix the other - **grep for other places `OutputFormat(`
   is constructed with a bed smaller than 8 before assuming this class of bug
   is fully closed.**

   **`TWO_ONE` (2.1) is the one template reachable from `codec.py`'s padded
   branch where `input_channels` (3) != `output_channels` (6)** - the static,
   unpadded `TWO_ONE` entry already treats 2.1 as living in a 6-channel
   container (`L/R/C/SW/SL/SR`), not the 3-channel `L/R/SW` its name implies
   (a pre-existing quirk, not something this session changed).
   `base_channels=template.output_channels` (6) was chosen for consistency
   with that existing static behavior, which the user then confirmed
   (2026-07): **JRiver always sends 2.1 as a 6-channel container**, so the
   first 3 channels of "padding" beyond the 2.1 signal's own 3 real channels
   are a genuine nop (already accounted for by the container) - only padding
   beyond that produces a new Extra channel. Since `paddings` steps by 2,
   this only bites at `padding=2` (nop, stays entirely within the base, no
   Extra channel at all) vs. `padding=4` (1 new Extra channel, `X1`) - exactly
   what `base_channels=template.output_channels` already produces, with no
   further change needed. See
   `test_two_one_plus_padding_treats_first_3_padding_channels_as_a_nop`.

`formats.get_all_channel_names(use_atmos_channels=...)` /
`OutputFormat.get_output_channel_indexes(use_atmos_channels=...)` /
`get_input_channel_indexes(use_atmos_channels=...)` pick which ordering a
>8-channel format's channel set is built from: legacy numbered (`False`) or
the full 9.1.6 layout followed by the Extra channels (`True`) — base-8 + 8
Atmos + 16 Extra = 32, exactly the largest format currently defined. This
mirrors the `convert_q`/`allow_padding` pattern: `mcws.MediaServer.use_atmos_channels`
gates it for a live connection by comparing the full parsed `ProgramVersion`
semver against `mcws.ATMOS_CHANNELS_MIN_VERSION = (35, 0, 39)` (NOT the major
version alone — `mc_version` is only an int and can't distinguish 35.0.38
from 35.0.39). For the file-based flow, `ui.py`'s `__pick_mc_version` asks two
separate questions: the MC major version (28/29/30/36 — this still only
drives `is_compatible`/padding/which default config template is loaded, none
of which changed at 35.0.39) and, independently, `<=35.0.38` vs `>=35.0.39`
for `use_atmos_channels` — the two are decoupled because the 35.0.39 cutover
doesn't align with any major-version boundary the rest of the picker cares
about. `JRiverDSP(use_atmos_channels=...)` threads the result to
`dsp.py:channel_names()`.
Unlike `convert_q`, this one *is* remembered consistently — there's no
separate write-time argument to forget to match, because there's no
parse-vs-write asymmetry in what it controls (which channels a format
exposes for editing), not a per-filter numeric convention.

**`use_atmos_channels` picks a different pool for a dynamically padded
instance than for a static, deliberately-immersive format — it is never
simply ignored.** `OutputFormat.__init__`'s `template` flag distinguishes the
two: a static `OUTPUT_FORMATS` entry (`FIVE_ONE_TWO`/`SEVEN_ONE_FOUR`/
`NINE_ONE_SIX`/`THIRTY_TWO`, or any other, `template=True`) represents a
deliberately-chosen full layout and draws Atmos channels first, then Extra
(`get_all_channel_names(use_atmos_channels=True)`); a `"+N padding channels"`
instance dynamically built by `codec.get_output_format`'s padded branch
(`template=False`) represents JRiver's generic "N extra scratch channels"
mechanism, which draws from a *version-dependent* pool: legacy numbered
(`Channel 9`..) pre-35.0.39, or **Extra channels alone — never Atmos** —
35.0.39+. `get_output_channel_indexes`/`get_input_channel_indexes` pass
`padding_only=self.__is_padded_instance` through to `get_all_channel_names`,
which only changes the `use_atmos_channels=True` branch: Extra-only for a
padded instance, Atmos+Extra combined otherwise.

This was originally found from a real MC35 production config (a 7.1 +
8-padding format) whose PEQ block used *both* legacy spare channels (`C9`-`C13`,
via `XOBM`/`Multiway`) *and* one isolated Atmos-channel `Mix` (a height-channel
downmix stub, channel 57/`RTR`) at once — proving the padding channels and the
Atmos/Extra channels are two independent pools, not alternates selected by MC
version. The first fix drawn from that file went too far, though: it made
`use_atmos_channels` a no-op for *any* padded instance regardless of version,
reasoning that padding "has always meant legacy-numbered channels". That was
never actually tested against a real MC36 padded capture — it was an
extrapolation from an MC35-only file (where `use_atmos_channels` is `False`
anyway, so the padded-instance special case was never exercised for `True`).
A real MC36 capture (`mc36_seven_one_plus_two_padding.dsp` — `Output
Channels=8`, `Output Padding Channels=2`, i.e. 7.1+2, built via JRiver's own
DSP Studio UI with a Peak filter deliberately placed on the 2 padding
channels, cross-checked against JRiver's own Analyzer channel meters)
disproved it: the 2 padding channels resolved to `X1`/`X2` (raw indexes
37/38, the Extra pool), not `Channel 9`/`10` (13/14, legacy) and not `LTF`/
`RTF` (54/55, Atmos). So from 35.0.39 onward, padding *did* change pool — just to Extra,
not to Atmos+Extra combined the way a static immersive format does. Getting
either version of this wrong isn't just cosmetic: the original bug (always
Atmos+Extra) silently discarded `C9`-`C13` from the declared channel set for
an MC35 file loaded with `use_atmos_channels=True`; the intermediate "always
legacy for padding" fix would silently mis-declare `X1`/`X2` as `Channel 9`/
`10` for a real MC36 padded file — both are exactly the kind of file BEQD
itself produces via `XOBM`/`Multiway` bass management.

That same real file also surfaced a second, independent bug: `OutputFormat`'s
channel *index* assignment for a >8-channel format is fundamentally a
heuristic — "take the first `output_channels` names from one fixed
ordering" — not derived from what a specific file's filters actually
reference, and no ordering fix removes that limitation, since the isolated
`RTR` (57) Mix in that same file falls outside *either* ordering's guess for
a 16-channel format regardless of which one is "correct" here. That file
failed to even load — a `KeyError` deep in `render.GraphRenderer.generate()`,
which built `nodes_by_channel` only for channels in the graph's *declared*
`output_channels`/`input_channels` and crashed on any filter touching
something outside that guess. Since `FilterGraph.__init__` calls `__regen()`
(which renders) eagerly, this broke loading the file at all, not just
visualizing it. Fixed by making `nodes_by_channel` a `defaultdict(list)` (a
channel outside the declared set just doesn't get an `IN:` seed node, and
doesn't get a terminal `OUT:` edge either — it's scratch space, not a real
output) instead of assuming completeness.

**The exact same assumption existed one layer down, in simulation, and wasn't
covered by the render.py fix**: `FilterGraph.simulate()` builds a per-channel
`signals` dict, *also* only for `output_channels`, and `__simulate_filter`
indexes into it directly (`signals[c]`, `signals[dst_channel]`) with no
fallback — so loading the file was fine, but activating/simulating it (`ui.py`
calls this via `show_filters()`/`activate()`, needed just to display the
filter list, not only to view the graph) still raised `KeyError` for the
isolated `RTR` Mix. Fixed the same way: `_ScratchChannelSignals` (a `dict`
subclass with `__missing__`) lazily defaults an unlisted channel to a silent
signal instead of raising. **When you fix one "declared channels is a closed
set" assumption in this module, grep for the others** — `render.py` and
`filter.py`'s `simulate()` had the identical bug independently; there may be
more (e.g. anywhere else that builds a dict/set keyed by
`output_channels`/`input_channels` and then indexes it with a filter's actual
`channel_names`).

See `test_padded_instance_uses_extra_channels_not_legacy_on_mc36`,
`test_real_capture_with_mixed_channel_eras_loads_via_full_jriverdsp` (fixture:
`mc35_mixed_channel_eras.dsp`) and
`test_real_capture_seven_one_plus_two_padding_uses_extra_channels` (fixture:
`mc36_seven_one_plus_two_padding.dsp`) in `test_dsp_roundtrip.py` — the latter
two deliberately go through the full `JRiverDSP`/`FilterGraph`/`render.py`
pipeline including `activate()`, unlike the all-channels captures' tests,
which decode at the `codec`/`filter` layer directly.

When downloading a config over MCWS (`ui.py:__show_zone_dialog`'s `on_select`),
the real `mc_version`/`use_atmos_channels` are known directly from the live
connection's full semver (see `mcws.MediaServer.use_atmos_channels` above), so
there's no need to ask the user to pick a version like the file-based flow
does — except when `use_atmos_channels` is `False` and `mc_version < 36`,
where BEQD asks (`QMessageBox.question`) whether to opt in anyway for editing
purposes, since a user may be on an untracked 35.0.39+ install (major version
alone can't tell), about to route filters onto Atmos/Extra channels ahead of
upgrading their JRMC install, or managing configs for a mix of zones on
different versions. Saying yes changes `use_atmos_channels` for the loaded
session (i.e. which channels the routing UI offers for anything newly
added/edited) **and** migrates any existing filter already sitting on the
legacy numbered pool onto the new scheme — see
`OutputFormat.migrate_channel_index`/`JRiverDSP.__migrate_channels` below.

**Existing filters *are* migrated when `use_atmos_channels` flips to `True`
(2026-07, corrected from an earlier, wrong design decision).** A prior pass
through this file argued that since JRiver itself never remaps a raw index's
*meaning* between versions (still true — see the table above), BEQD didn't
need to either, and a filter recorded against `Channel 9` (idx 13) could just
be left there once the declared channel set moved on to `X1` (idx 37) —
becoming an orphaned "scratch channel" (rendered/simulated fine via the
`defaultdict`/`_ScratchChannelSignals` fixes above, but invisible in the
channel-list widget and not actually applied to any real output anymore). That
was wrong: those two facts don't imply each other. JRiver not remapping raw
indexes is a statement about the *file format*; it says nothing about whether
a filter that was deliberately placed on "the Nth scratch/extra channel" by an
older client should keep tracking that *role* once the pool backing it changes.
For BEQD's own bass-management routing (`XOBM`/`Multiway`, which is exactly
what generic padding/scratch channels are used for — see `template=False`
above) the answer is yes: the filter needs to keep landing on a real,
in-pipeline channel, not silently fall out of the declared set.

`OutputFormat.migrate_channel_index(idx, use_atmos_channels)` (`formats.py`)
does the actual remap: if `use_atmos_channels` is `True` and `idx` falls in
the legacy pool (13-36), it's translated by *position* (Channel 9 = position
0, Channel 10 = position 1, ...) onto the equivalent position in whichever
pool this format's `use_atmos_channels=True` branch actually draws from —
Extra-only for a padded instance, Atmos+Extra for a static immersive format
(the same `padding_only` distinction as `get_all_channel_names`). Indexes
outside the legacy pool, or `use_atmos_channels=False`, are a no-op.
`JRiverDSP.__migrate_channels` (`dsp.py`) applies this to every parsed
filter's raw channel value(s) before constructing the `Filter` object — most
filter types carry channels in a semicolon-joined `Channels` value, but `Mix`
uses single-channel `Source`/`Destination` values instead, so both are
covered. This runs identically whether the config came from a `.dsp` file or
a live MCWS download (`JRiverDSP.__init__` → `__parse_peq` → this), since both
paths converge on the same constructor.

Deliberately **not** migrated in the reverse direction (Atmos/Extra idx →
legacy, when `use_atmos_channels=False`): an index in the Atmos/Extra pool
found in a file that's being *viewed* under the legacy scheme is far more
likely to be a real, deliberate Atmos-channel assignment (e.g. a height-channel
downmix `Mix`, made directly through JRiver's own channel picker, independent
of BEQD's own scratch-channel numbering) than something that needs migrating
back — see the isolated `RTR` (57) `Mix` in `mc35_mixed_channel_eras.dsp`,
which must stay exactly where it is regardless of `use_atmos_channels`. See
`test_migrate_channel_index_maps_legacy_pool_onto_atmos_or_extra_pool` and the
updated `test_real_capture_with_mixed_channel_eras_loads_via_full_jriverdsp`.

**Per-filter migration alone wasn't sufficient — complex filters can cache a
channel name a second time, in their own Divider metadata.** Reported
directly by a user testing against a real MC35 instance: after the fix above,
an `XOBM` filter's own channel list showed a mix of eras at once, e.g.
`[L, R, C, SW, RL, RR, C11, C12, C15]` where 3 of those 9 should've been
`X`-named like the rest. Root cause: `MultiwayFilter`/`XOFilter`/
`CompoundRoutingFilter` (XOBM) don't derive their displayed `channel_names`
purely from their constituent filters' raw indexes the way `GEQFilter`/
`MSOFilter` do — they cache a channel *name* directly in the JSON/`/`-joined
metadata text embedded in the JRiver `Divider` marker that brackets the
complex filter (`MultiwayFilter`'s `"i"`/`"o"` fields, `XOFilter`'s leading
`/`-token, `CompoundRoutingFilter`'s `"e"[].u` speaker-group lists and
`"r"` route strings, e.g. `"R/2/C9"` = input `R`, way `2`, output `C9`).
`JRiverDSP.__migrate_channels` never touched this text — it only migrates a
leaf filter's `Channels`/`Source`/`Destination` numeric value(s), so the
complex filter's own constituent `Gain`/`Mix` filters ended up migrated while
its cached metadata name(s) didn't, producing exactly this kind of mixed
result. Fixed via `ComplexFilter.migrate_channel_metadata(data, migrate_name)`
— a no-op by default, overridden by the three types above to parse their own
metadata text, migrate whichever tokens are channel names (via a
`get_channel_idx` → `OutputFormat.migrate_channel_index` → `get_channel_name`
round trip, wrapped so a non-channel token is left alone), and re-serialize.
Called from `JRiverDSP.__handle_divider` right before `filt_cls.create(...)`,
using the same `use_atmos_channels`/`OutputFormat` this `JRiverDSP` was
constructed with. Because each of these types' own `metadata()` method
re-serializes fresh from its own fields on save (not the original raw text),
migrating at parse time is enough — no separate write-time step needed.
See the `xobm_repr` assertions added to
`test_real_capture_with_mixed_channel_eras_loads_via_full_jriverdsp`. **If a
new complex filter type is added that caches a channel name in its own
metadata rather than deriving it from constituent filters, it needs its own
`migrate_channel_metadata` override too** — the base class default silently
does nothing, so this is easy to miss.

## When JRiver changes the format again

Likely touch points, roughly in order of how often JRiver has changed them
historically:

1. New output/channel layout → add an entry to `formats.OUTPUT_FORMATS` and
   extend `codec.get_output_format`/`get_legacy_output_format` if the
   channel-count/padding/layout combination is ambiguous with an existing one.
2. New filter type in JRMC's own PEQ → add a `Filter` subclass in `filter.py`
   with the right `TYPE` code; it registers itself automatically
   (`filter_classes_by_type`). Add it to `convert_q_types` only if it has a
   version-dependent Q convention.
3. Change to how PEQ blocks/plugin ordering are exposed → `codec.get_peq_block_order`,
   `codec.get_peq_key_name`.
4. New MC major version behaviour differences (padding support, Q
   convention, anything else version-gated) → `mcws.MediaServer` version
   gates (`__is_29`-style checks) and the `convert_q`/`allow_padding` flags
   threaded through `dsp.py`/`ui.py`.

Whatever changes, re-run (and extend) the round-trip tests in
`src/test/python/model/jriver/test_dsp_roundtrip.py` first — they're the
closest thing this module has to an executable spec of the file format. The
MC36 Atmos/Extra channel work above is a worked example of touchpoint 1 and
4 together: a one-off PeakingEQ-per-channel capture from each JRiver version
(see "Channels beyond the base 8" above) pinned down exactly which raw
indexes changed reachability, which confirmed JRiver's own file format never
remaps a raw index's meaning between versions — completing `formats.py`'s
lookup tables and adding a `convert_q`-style boolean (`use_atmos_channels`) to
gate which ordering BEQD itself uses when *constructing* a new format's
channel set was enough for that part. It was a separate, later finding
(2026-07, see the migration paragraph above) that BEQD's *own* existing
filters still needed an explicit remap step when reinterpreting an older
config under the new scheme — don't conflate "JRiver doesn't remap" with
"BEQD doesn't need to either".

## Testing

- `test_dsp_roundtrip.py` — parse → re-encode → re-parse fidelity for the
  file format itself (this is where new format-compatibility tests belong).
  There were **no tests exercising `dsp.py` or `codec.py` at all** before
  this file existed (coverage was 0% and 8% respectively) — the crossover
  synthesis engine in `filter.py` (`MultiChannelSystem` etc) was tested via
  `test_xo.py`'s in-memory filter-list assertions, but that never touched
  XML serialization, so a bug in the `Divider`/complex-filter encode/decode
  path could pass all existing tests.
- `test_dsp_roundtrip.py`'s `resources/` dir holds real, scrubbed (machine
  paths/timestamps only) JRMC captures — currently the MC35/MC36 "every
  channel at once" pair described above. Prefer adding real captures here
  over hand-built XML when investigating a specific JRiver version's actual
  behaviour; hand-built fixtures are still the right tool for testing the
  format mechanics in isolation (see the rest of this file).
- `test_xo.py` — crossover/bass-management filter *synthesis* correctness
  (what filters get generated for a given routing matrix), independent of
  the XML format.
