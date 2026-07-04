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
closest thing this module has to an executable spec of the file format.

## Testing

- `test_dsp_roundtrip.py` — parse → re-encode → re-parse fidelity for the
  file format itself (this is where new format-compatibility tests belong).
  There were **no tests exercising `dsp.py` or `codec.py` at all** before
  this file existed (coverage was 0% and 8% respectively) — the crossover
  synthesis engine in `filter.py` (`MultiChannelSystem` etc) was tested via
  `test_xo.py`'s in-memory filter-list assertions, but that never touched
  XML serialization, so a bug in the `Divider`/complex-filter encode/decode
  path could pass all existing tests.
- `test_xo.py` — crossover/bass-management filter *synthesis* correctness
  (what filters get generated for a given routing matrix), independent of
  the XML format.
