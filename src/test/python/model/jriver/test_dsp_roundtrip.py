'''
Round trip tests for the JRiver Media Center DSP config format - i.e. verifying that a range of DSP
configurations (individual filters, compound/complex filters, native JRMC-only filters, multiple PEQ
blocks, various output channel formats) survive JRiverDSP parse -> config_txt() re-encode -> re-parse
without losing or corrupting information.

See model/jriver/AGENTS.md for the format itself; these tests are the closest thing to an executable
spec of that format.
'''
import math
import xml.etree.ElementTree as et
from pathlib import Path

import pytest

from model.iir import ComplexHighPass, ComplexLowPass, FilterType, q_to_s
from model.jriver import JRIVER_FS
from model.jriver.codec import (extract_filters, filts_to_xml, get_output_format, get_peq_block_order,
                                get_peq_key_name, include_filters_in_dsp, item_to_dicts)
from model.jriver.dsp import JRiverDSP
from model.jriver.filter import (AllPass, BitdepthSimulator, CompositeXODescriptor, CustomPassFilter, Delay,
                                  Divider, Gain, GEQFilter, HighPass, HighShelf, LimiterMode, Limiter,
                                  LinkwitzRiley, LowPass, LowShelf, MidSideDecoding, MidSideEncoding, Mix,
                                  MixType, MultiChannelSystem, MultiwayFilter, Mute, Order, Peak, Polarity,
                                  SubwooferLimiter, WayDescriptor, WayValues, XODescriptor,
                                  convert_filter_to_mc_dsp, create_single_filter)
from model.jriver.formats import OUTPUT_FORMATS, get_all_channel_names, get_channel_idx, get_channel_name
from model.jriver.routing import Matrix

PEQ1 = 'Parametric Equalizer'
PEQ2 = 'Parametric Equalizer 2'
RESOURCES = Path(__file__).parent / 'resources'


def base_config(output_channels: int = 6, padding: int = 0, layout: int = None, peq2_enabled: bool = False,
                 plugin_order: str = None, extra_key: str = None) -> str:
    '''Builds a minimal, structurally valid JRiver .dsp config with empty PEQ blocks.'''
    layout_xml = f'''
    <Data>
      <Name>Output Channel Layout</Name>
      <Value>{layout}</Value>
    </Data>''' if layout is not None else ''
    plugin_order_xml = f'''
  <Key Name="DSP Studio">
    <Data>
      <Name>Plugin Order</Name>
      <Value>{plugin_order}</Value>
    </Data>
  </Key>''' if plugin_order else ''
    extra_xml = f'''
  <Key Name="Zzz Custom Setting">
    <Data>
      <Name>Marker</Name>
      <Value>{extra_key}</Value>
    </Data>
  </Key>''' if extra_key else ''
    return f'''<?xml version="1.0" encoding="utf-8"?>
<DSP>
<Preset>
  <Key Name="Audio Settings">
    <Data>
      <Name>Output Channels</Name>
      <Value>{output_channels}</Value>
    </Data>
    <Data>
      <Name>Output Padding Channels</Name>
      <Value>{padding}</Value>
    </Data>{layout_xml}
  </Key>
  <Key Name="{PEQ1}">
    <Data>
      <Name>Enabled</Name>
      <Value>1</Value>
    </Data>
    <Data>
      <Name>Filters</Name>
      <Value>(1:1)(2:0)</Value>
    </Data>
  </Key>
  <Key Name="{PEQ2}">
    <Data>
      <Name>Enabled</Name>
      <Value>{1 if peq2_enabled else 0}</Value>
    </Data>
    <Data>
      <Name>Filters</Name>
      <Value>(1:1)(2:0)</Value>
    </Data>
  </Key>{plugin_order_xml}{extra_xml}
</Preset>
</DSP>
'''


def seed(config_txt: str, peq_key: str, filters: list, convert_q: bool = False) -> str:
    '''Seeds a PEQ block with the given top level filters, using the library's own encoder - this is
    exactly what config_txt() does internally, used here to construct a fixture "as if" the filters
    were already present in a JRMC-authored file. Note: Filter.encode() always uses convert_q=False,
    so a convert_q=True fixture (mimicking an MC28-style file) must go via get_all_vals() directly.'''
    from model.jriver.codec import filts_to_xml
    xml_filts = [filts_to_xml(f.get_all_vals(convert_q=convert_q)) for f in filters]
    return include_filters_in_dsp(peq_key, config_txt, xml_filts, replace=True)


def load(config_txt: str, **kwargs) -> JRiverDSP:
    return JRiverDSP('test.dsp', lambda: config_txt, **kwargs)


def vals(cls, **overrides) -> dict:
    return {**cls.default_values(), **overrides}


# ---------------------------------------------------------------------------------------------------
# Output format decoding
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('output_channels,padding,layout,expected', [
    (2, 0, None, 'STEREO'),
    (1, 0, None, 'MONO'),
    (6, 0, None, 'FIVE_ONE'),
    (8, 0, None, 'SEVEN_ONE'),
    (3, 0, None, 'TWO_ONE'),
    (4, 0, 15, 'THREE_ONE'),
    (2, 2, None, 'STEREO_IN_FOUR'),
    (2, 4, None, 'STEREO_IN_FIVE'),
    (6, 2, None, 'FIVE_ONE_IN_SEVEN'),
])
def test_legacy_output_format_roundtrip(output_channels, padding, layout, expected):
    txt = base_config(output_channels=output_channels, padding=padding, layout=layout)
    fmt = get_output_format(txt, allow_padding=False)
    assert fmt.display_name == OUTPUT_FORMATS[expected].display_name
    assert fmt.output_channels == OUTPUT_FORMATS[expected].output_channels


@pytest.mark.parametrize('output_channels,padding', [
    (2, 0),
    (6, 0),
    (8, 4),
    (12, 16),
])
def test_padded_output_format_roundtrip(output_channels, padding):
    txt = base_config(output_channels=output_channels, padding=padding)
    fmt = get_output_format(txt, allow_padding=True)
    assert fmt.input_channels + padding == fmt.output_channels
    # re-derive the same format from a dsp instance built with allow_padding
    dsp = load(txt, allow_padding=True)
    assert dsp.output_format.output_channels == fmt.output_channels


def test_mc35_immersive_output_formats_are_registered():
    for key in ['FIVE_ONE_TWO', 'SEVEN_ONE_FOUR', 'NINE_ONE_SIX']:
        fmt = OUTPUT_FORMATS[key]
        txt = base_config(output_channels=fmt.output_channels, padding=0)
        decoded = get_output_format(txt, allow_padding=False)
        assert decoded.output_channels == fmt.output_channels


# ---------------------------------------------------------------------------------------------------
# PEQ block enablement & ordering
# ---------------------------------------------------------------------------------------------------

def test_single_enabled_peq_block():
    txt = base_config(peq2_enabled=False)
    assert get_peq_block_order(txt) == [0]
    dsp = load(txt)
    assert dsp.graph_count == 1
    assert dsp.graph(0).stage == 0


def test_dual_peq_block_default_order():
    txt = base_config(peq2_enabled=True)
    assert get_peq_block_order(txt) == [0, 1]
    dsp = load(txt)
    assert dsp.graph_count == 2
    assert [dsp.graph(i).stage for i in range(2)] == [0, 1]


def test_dual_peq_block_reversed_order_via_plugin_order():
    plugin_order = f'(Some Plugin)({PEQ2})(Another Plugin)({PEQ1})(Output Format)'
    txt = base_config(peq2_enabled=True, plugin_order=plugin_order)
    assert get_peq_block_order(txt) == [1, 0]
    dsp = load(txt)
    assert dsp.graph_count == 2
    # graph list order follows plugin order, but .stage always identifies the real XML block
    assert [dsp.graph(i).stage for i in range(2)] == [1, 0]


def test_no_enabled_peq_block_raises():
    txt = base_config(peq2_enabled=False).replace(
        f'<Key Name="{PEQ1}">\n    <Data>\n      <Name>Enabled</Name>\n      <Value>1</Value>',
        f'<Key Name="{PEQ1}">\n    <Data>\n      <Name>Enabled</Name>\n      <Value>0</Value>')
    with pytest.raises(ValueError):
        get_peq_block_order(txt)


# ---------------------------------------------------------------------------------------------------
# Empty config is a true no-op
# ---------------------------------------------------------------------------------------------------

def test_empty_peq_blocks_round_trip_is_byte_identical_noop():
    txt = base_config(peq2_enabled=False)
    dsp = load(txt)
    assert dsp.graph(0).filters == []
    assert dsp.config_txt() == txt


# ---------------------------------------------------------------------------------------------------
# Atomic filter round trips
# ---------------------------------------------------------------------------------------------------

def _l_r(channels=('L', 'R')):
    return ';'.join(str(get_channel_idx(c)) for c in channels)


def _assert_filter_list_round_trips(txt: str, original_filters: list, peq_key: str = PEQ1):
    seeded = seed(txt, peq_key, original_filters)
    dsp = load(seeded)
    parsed = dsp.graph(0).filters
    assert len(parsed) == len(original_filters)
    for original, decoded in zip(original_filters, parsed):
        assert type(decoded) is type(original)
        assert decoded == original

    round_tripped_txt = dsp.config_txt()
    dsp2 = load(round_tripped_txt)
    assert dsp2.graph(0).filters == parsed
    # second pass onwards is a stable fixed point
    assert dsp2.config_txt() == round_tripped_txt
    return dsp, round_tripped_txt


@pytest.mark.parametrize('build', [
    lambda: Peak(vals(Peak, Channels=_l_r(('L',)), Frequency='100', Q='1.41', Gain='3.2')),
    lambda: LowShelf(vals(LowShelf, Channels=_l_r(('SW',)), Frequency='30', Q='0.7', Gain='-4.5')),
    lambda: HighShelf(vals(HighShelf, Channels=_l_r(('C',)), Frequency='8000', Q='0.7', Gain='2')),
    lambda: LowPass(vals(LowPass, Channels=_l_r(('SW',)), Slope='24', Frequency='80', Q='0.707', Gain='0')),
    lambda: HighPass(vals(HighPass, Channels=_l_r(('L', 'R')), Slope='12', Frequency='40', Q='0.707', Gain='0')),
    lambda: Gain(vals(Gain, Channels=_l_r(('L', 'R', 'C')), Gain='-6.5')),
    lambda: Mute(vals(Mute, Channels=_l_r(('SL',)))),
    lambda: Delay(vals(Delay, Channels=_l_r(('R',)), Delay='4.242')),
    lambda: Polarity(vals(Polarity, Channels=_l_r(('SW',)))),
    lambda: AllPass(vals(AllPass, Channels=_l_r(('L',)), Frequency='120', Q='1.2')),
], ids=['peak', 'lowshelf', 'highshelf', 'lowpass', 'highpass', 'gain', 'mute', 'delay', 'polarity', 'allpass'])
def test_atomic_channel_filter_roundtrip(build):
    txt = base_config()
    f = build()
    _assert_filter_list_round_trips(txt, [f])


def test_linkwitz_transform_roundtrip():
    txt = base_config()
    from model.jriver.filter import LinkwitzTransform
    f = LinkwitzTransform({
        'Enabled': '1', 'Type': LinkwitzTransform.TYPE, 'Channels': _l_r(('SW',)),
        'Fz': '20', 'Qz': '0.5', 'Fp': '25', 'Qp': '0.707', 'PreventClipping': '0'
    })
    _assert_filter_list_round_trips(txt, [f])


def test_mix_filter_roundtrip():
    txt = base_config()
    for mix_type in MixType:
        f = Mix({
            'Enabled': '1', 'Type': Mix.TYPE, 'Source': str(get_channel_idx('L')),
            'Destination': str(get_channel_idx('SW')), 'Gain': '-1.5', 'Mode': str(mix_type.value)
        })
        _assert_filter_list_round_trips(txt, [f])


def test_disabled_filter_preserves_enabled_flag():
    txt = base_config()
    f = Peak(vals(Peak, Enabled='0', Channels=_l_r(('L',)), Frequency='100', Q='1.0', Gain='3'))
    dsp, _ = _assert_filter_list_round_trips(txt, [f])
    assert dsp.graph(0).filters[0].enabled is False


def test_multiple_atomic_filters_preserve_order():
    txt = base_config()
    filters = [
        Peak(vals(Peak, Channels=_l_r(('L',)), Frequency='100', Q='1.0', Gain='3')),
        Gain(vals(Gain, Channels=_l_r(('L', 'R')), Gain='-2')),
        Delay(vals(Delay, Channels=_l_r(('R',)), Delay='1.5')),
        Mute(vals(Mute, Channels=_l_r(('SW',)))),
    ]
    _assert_filter_list_round_trips(txt, filters)


# ---------------------------------------------------------------------------------------------------
# Native-JRMC-only filter types (never produced by BEQD but must not be corrupted if present)
# ---------------------------------------------------------------------------------------------------

def test_native_only_filter_types_roundtrip():
    txt = base_config()
    filters = [
        Limiter({
            'Enabled': '1', 'Type': Limiter.TYPE, 'Channels': _l_r(('L', 'R')),
            'Hold': '20', 'Mode': str(LimiterMode.ADAPTIVE.value), 'Level': '-1', 'Release': '100', 'Attack': '5'
        }),
        BitdepthSimulator({'Enabled': '1', 'Type': BitdepthSimulator.TYPE, 'Bits': '16', 'Dither': '1'}),
        Order({'Enabled': '1', 'Type': Order.TYPE,
               'Order': ','.join(str(get_channel_idx(c)) for c in ('R', 'L'))}),
        MidSideEncoding({'Enabled': '1', 'Type': MidSideEncoding.TYPE}),
        MidSideDecoding({'Enabled': '1', 'Type': MidSideDecoding.TYPE}),
        LinkwitzRiley({'Enabled': '1', 'Type': LinkwitzRiley.TYPE, 'Frequency': '80'}),
        SubwooferLimiter({'Enabled': '1', 'Type': SubwooferLimiter.TYPE, 'Channels': _l_r(('SW',)), 'Level': '-3'}),
    ]
    _assert_filter_list_round_trips(txt, filters)


def test_native_divider_outside_complex_filter_is_preserved():
    '''
    A plain divider (e.g. a user's manual "---" separator added directly in JRMC's own PEQ UI) that
    isn't part of one of BEQD's recognised complex-filter start/end pairs must survive a round trip
    unchanged, just like any other native-only filter type.
    '''
    txt = base_config()
    manual_divider = Divider({'Enabled': '1', 'Type': Divider.TYPE, 'Text': '--- my manual separator ---'})
    peak = Peak(vals(Peak, Channels=_l_r(('L',)), Frequency='100', Q='1.0', Gain='3'))
    _assert_filter_list_round_trips(txt, [manual_divider, peak])


# ---------------------------------------------------------------------------------------------------
# Complex (compound) filter round trips - the Divider start/end protocol
# ---------------------------------------------------------------------------------------------------

def test_geq_filter_roundtrip():
    txt = base_config()
    bands = [Peak(vals(Peak, Channels=_l_r(('L',)), Frequency=str(f), Q='4.3', Gain=str(g)))
             for f, g in [(25, 2.0), (63, -1.5), (1000, 0.5)]]
    geq = GEQFilter(bands)
    dsp, _ = _assert_filter_list_round_trips(txt, [geq])
    decoded = dsp.graph(0).filters[0]
    assert isinstance(decoded, GEQFilter)
    assert len(decoded.filters) == 3
    assert [f.freq for f in decoded.filters] == [25.0, 63.0, 1000.0]


def test_custom_pass_filter_roundtrip_lr4():
    '''LR4 (and any non-4th/6th/8th-order Butterworth, or any Bessel pass filter) is encoded as a
    CustomPassFilter wrapping the underlying biquad-stage native filters, with the original filter
    design smuggled through as Divider metadata (e.g. "LP/LR/4/2000/0.7071").'''
    txt = base_config()
    target = _l_r(('L', 'R'))
    lp = ComplexLowPass(FilterType.LINKWITZ_RILEY, 4, JRIVER_FS, 2000.0)
    mc_filter = convert_filter_to_mc_dsp(lp, target)
    assert isinstance(mc_filter, CustomPassFilter)
    dsp, _ = _assert_filter_list_round_trips(txt, [mc_filter])
    decoded = dsp.graph(0).filters[0]
    assert isinstance(decoded, CustomPassFilter)
    editable = decoded.get_editable_filter()
    assert isinstance(editable, ComplexLowPass)
    assert editable.type == FilterType.LINKWITZ_RILEY
    assert editable.order == 4
    assert math.isclose(editable.freq, 2000.0, rel_tol=1e-6)


def test_custom_pass_filter_roundtrip_bessel_highpass():
    txt = base_config()
    target = _l_r(('SW',))
    hp = ComplexHighPass(FilterType.BESSEL_MAG3, 3, JRIVER_FS, 45.0)
    mc_filter = convert_filter_to_mc_dsp(hp, target)
    assert isinstance(mc_filter, CustomPassFilter)
    dsp, _ = _assert_filter_list_round_trips(txt, [mc_filter])
    decoded = dsp.graph(0).filters[0]
    editable = decoded.get_editable_filter()
    assert isinstance(editable, ComplexHighPass)
    assert editable.type == FilterType.BESSEL_MAG3
    assert editable.order == 3


def test_multiway_crossover_roundtrip():
    '''
    The most realistic real-world case: a crossover designed in the UI, synthesised into a
    MultiwayFilter/CompoundRoutingFilter tree by MultiChannelSystem, then written to and read back
    from a JRMC config. This is the scenario test_xo.py exercises but never serializes to XML.
    '''
    matrix = Matrix({'L': 2, 'R': 2}, ['L', 'R', 'C', 'SW'])

    def p(freq):
        return ['Linkwitz-Riley', 4, freq]

    xo_desc = [
        CompositeXODescriptor({
            'L': XODescriptor('L', [WayDescriptor(WayValues(0, lp=p(2000.0)), ['L']),
                                     WayDescriptor(WayValues(1, hp=p(2000.0)), ['C'])]),
            'R': XODescriptor('R', [WayDescriptor(WayValues(0, lp=p(2000.0)), ['R']),
                                     WayDescriptor(WayValues(1, hp=p(2000.0)), ['SW'])]),
        }, [None] * 2, [])
    ]
    for d in xo_desc:
        for desc in d.xo_descriptors.values():
            for i, way in enumerate(desc.ways):
                for o in way.out_channels:
                    matrix.enable(desc.in_channel, i, o)

    mcs = MultiChannelSystem(xo_desc)
    mc_filter = mcs.calculate_filters(matrix)
    assert mc_filter.filters
    assert all(isinstance(f, MultiwayFilter) for f in mc_filter.filters)

    txt = base_config(output_channels=4)
    dsp, _ = _assert_filter_list_round_trips(txt, mc_filter.filters)
    decoded = dsp.graph(0).filters
    assert len(decoded) == len(mc_filter.filters)
    for original, decoded_f in zip(mc_filter.filters, decoded):
        assert isinstance(decoded_f, MultiwayFilter)
        assert decoded_f.input_channel == original.input_channel
        assert decoded_f.output_channels == original.output_channels
        assert len(decoded_f.filters) == len(original.filters)


# ---------------------------------------------------------------------------------------------------
# Dual PEQ block round trip - independence of blocks & the untouched-document guarantee
# ---------------------------------------------------------------------------------------------------

def test_dual_peq_block_independent_roundtrip():
    txt = base_config(peq2_enabled=True, extra_key='do-not-touch-me')
    peq1_filters = [Peak(vals(Peak, Channels=_l_r(('L',)), Frequency='100', Q='1.0', Gain='3'))]
    peq2_filters = [Gain(vals(Gain, Channels=_l_r(('L', 'R')), Gain='-4'))]
    seeded = seed(txt, PEQ1, peq1_filters)
    seeded = seed(seeded, PEQ2, peq2_filters)

    dsp = load(seeded)
    assert dsp.graph(0).filters == peq1_filters
    assert dsp.graph(1).filters == peq2_filters

    round_tripped = dsp.config_txt()
    dsp2 = load(round_tripped)
    assert dsp2.graph(0).filters == peq1_filters
    assert dsp2.graph(1).filters == peq2_filters
    # unrelated document content is semantically untouched even though the whole tree gets
    # re-serialized by ElementTree as a side effect of splicing the Filters value
    root = et.fromstring(round_tripped)
    marker = root.find('./Preset/Key[@Name="Zzz Custom Setting"]/Data/Value')
    assert marker.text == 'do-not-touch-me'
    # stable fixed point on a second pass
    assert load(round_tripped).config_txt() == round_tripped


# ---------------------------------------------------------------------------------------------------
# convert_q (MC28 shelf/pass "S" <-> true Q) round trip, including the documented footgun
# ---------------------------------------------------------------------------------------------------

def test_convert_q_shelf_roundtrip_when_flags_match():
    gain = -6.0
    true_q = 0.9
    s = q_to_s(true_q, gain)
    txt = base_config()
    f = LowShelf(vals(LowShelf, Channels=_l_r(('L',)), Frequency='80', Q=f'{s:.12g}', Gain=str(gain)),
                 convert_q=True)
    assert math.isclose(f.q, true_q, rel_tol=1e-9)

    seeded = seed(txt, PEQ1, [f], convert_q=True)
    dsp = load(seeded, convert_q=True)
    decoded = dsp.graph(0).filters[0]
    assert math.isclose(decoded.q, true_q, rel_tol=1e-9)

    # write back WITH matching convert_q -> the "S" value is preserved
    round_tripped = dsp.config_txt(convert_q=True)
    dsp2 = load(round_tripped, convert_q=True)
    assert math.isclose(dsp2.graph(0).filters[0].q, true_q, rel_tol=1e-9)
    raw_q_in_file = et.fromstring(round_tripped).find(
        f'./Preset/Key[@Name="{PEQ1}"]/Data/Name[.="Filters"]/../Value').text
    assert f'{s:.12g}'[:8] in raw_q_in_file


def test_convert_q_write_default_does_not_match_legacy_parse():
    '''
    Documents a real, currently unfixed bug (see AGENTS.md "A live sharp edge"): JRiverDSP does not
    remember the convert_q it was parsed with. ui.py's plain "save" path calls config_txt() with no
    arguments (convert_q=False) regardless of how the file was loaded, so round-tripping a legacy
    MC28 file through the default write path silently changes the encoded Q semantics.
    '''
    gain = -6.0
    true_q = 0.9
    s = q_to_s(true_q, gain)
    txt = base_config()
    f = LowShelf(vals(LowShelf, Channels=_l_r(('L',)), Frequency='80', Q=f'{s:.12g}', Gain=str(gain)),
                 convert_q=True)
    seeded = seed(txt, PEQ1, [f], convert_q=True)

    dsp = load(seeded, convert_q=True)
    assert math.isclose(dsp.graph(0).filters[0].q, true_q, rel_tol=1e-9)

    # the buggy call site: config_txt() with no convert_q argument
    naive_round_trip = dsp.config_txt()
    raw_q_in_file = et.fromstring(naive_round_trip).find(
        f'./Preset/Key[@Name="{PEQ1}"]/Data/Name[.="Filters"]/../Value').text
    # the true Q (0.9) was written directly instead of being re-encoded back to the "S" convention -
    # it does NOT match the original S value, demonstrating the corruption
    assert f'{s:.12g}'[:8] not in raw_q_in_file
    reparsed_without_convert_q = load(naive_round_trip)
    assert math.isclose(reparsed_without_convert_q.graph(0).filters[0].q, true_q, rel_tol=1e-9)
    # ...but if MC itself still treats this file as legacy (convert_q=True on the next load), the
    # value is now double-interpreted and no longer represents the original filter design
    reparsed_as_legacy_again = load(naive_round_trip, convert_q=True)
    assert not math.isclose(reparsed_as_legacy_again.graph(0).filters[0].q, true_q, rel_tol=1e-9)


def test_convert_q_second_order_pass_filter_roundtrip():
    txt = base_config()
    jriver_q = 1.0  # value as stored by MC29+ (true Q)
    f = LowPass(vals(LowPass, Channels=_l_r(('SW',)), Slope='12', Frequency='80', Q=str(jriver_q), Gain='0'),
                convert_q=True)
    expected_q = jriver_q / (2 ** 0.5)
    assert math.isclose(f.q, expected_q, rel_tol=1e-9)
    seeded = seed(txt, PEQ1, [f], convert_q=True)
    dsp = load(seeded, convert_q=True)
    decoded = dsp.graph(0).filters[0]
    assert math.isclose(decoded.q, expected_q, rel_tol=1e-9)
    round_tripped = dsp.config_txt(convert_q=True)
    dsp2 = load(round_tripped, convert_q=True)
    assert math.isclose(dsp2.graph(0).filters[0].q, expected_q, rel_tol=1e-9)


# ---------------------------------------------------------------------------------------------------
# MC36 channel scheme: completed Atmos channels (54-61) + Extra channels (37-52) - see AGENTS.md
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('idx,short_name', [
    (37, 'X1'), (44, 'X8'), (52, 'X16'),
    (54, 'LTF'), (55, 'RTF'), (56, 'LTR'), (57, 'RTR'), (58, 'LTM'), (59, 'RTM'), (60, 'LW'), (61, 'RW'),
])
def test_atmos_and_extra_channel_names_resolve(idx, short_name):
    assert get_channel_name(idx) == short_name
    assert get_channel_idx(short_name) == idx


def test_get_all_channel_names_legacy_vs_atmos_ordering():
    legacy = get_all_channel_names(use_atmos_channels=False)
    atmos = get_all_channel_names(use_atmos_channels=True)
    assert legacy[:8] == atmos[:8] == ['L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR']
    # legacy: base 8 then the generically numbered channels
    assert legacy[8:12] == ['C9', 'C10', 'C11', 'C12']
    # atmos: base 8 then the full 9.1.6 layout, then the Extra channels
    assert atmos[8:16] == ['LTF', 'RTF', 'LTR', 'RTR', 'LTM', 'RTM', 'LW', 'RW']
    assert atmos[16:19] == ['X1', 'X2', 'X3']
    assert 'LTF' not in legacy
    assert 'C9' not in atmos


def test_output_format_channel_indexes_are_version_gated():
    fmt = OUTPUT_FORMATS['THIRTY_TWO']
    legacy_indexes = fmt.get_output_channel_indexes(use_atmos_channels=False)
    modern_indexes = fmt.get_output_channel_indexes(use_atmos_channels=True)
    assert 13 in legacy_indexes and 54 not in legacy_indexes and 37 not in legacy_indexes
    assert 54 in modern_indexes and 37 in modern_indexes and 13 not in modern_indexes
    # the bare property (no args) is unaffected - still legacy, for backwards compatibility
    assert fmt.output_channel_indexes == legacy_indexes
    assert fmt.input_channel_indexes == fmt.get_input_channel_indexes(use_atmos_channels=False)


# ---------------------------------------------------------------------------------------------------
# Real captures from MC 35.0.38 and MC 36.0.14 - one PeakingEQ filter per channel, gain used as a
# per-channel marker, covering every addressable channel range in both versions. See AGENTS.md for
# the full analysis. Decoded at the codec/filter layer directly (not through JRiverDSP/FilterGraph):
# these are deliberately exhaustive "every channel at once" diagnostic captures that straddle multiple
# channel-naming eras within a single PEQ block, which the (pre-existing) OutputFormat channel-index
# heuristic - a single contiguous "first N channels of one ordering" guess - can't represent all of
# at once. That's a real, separate, pre-existing limitation (see AGENTS.md); it doesn't affect parsing
# or round-tripping the filters themselves, which is what's under test here.
# ---------------------------------------------------------------------------------------------------

def _decode_raw_filters(config_txt: str, block: int):
    peq_block = get_peq_key_name(block)
    _, filt_element = extract_filters(config_txt, peq_block)
    frags = [v + ')' for v in filt_element.text.split(')') if v]
    return [create_single_filter(d) for d in (item_to_dicts(f) for f in frags[2:]) if d]


@pytest.mark.parametrize('fixture,expected', [
    ('mc35_all_channels.dsp', {
        1: ['L'], 8: ['RR'], 9: ['LTF'], 10: ['RTF'], 11: ['LTR'], 12: ['RTR'],
        # 58-61 (LTM/RTM/LW/RW) don't work in 35.0.38 - selecting them re-picks LTR/RTR (56/57)
        13: ['LTR'], 14: ['RTR'], 15: ['LTR'], 16: ['RTR'],
        17: ['L', 'R'],  # Extra channels don't work in 35.0.38 - falls back to L,R
        19: ['U1', 'U2'],
    }),
    ('mc36_all_channels.dsp', {
        1: ['L'], 8: ['RR'], 9: ['LTF'], 10: ['RTF'], 11: ['LTR'], 12: ['RTR'],
        13: ['LTM'], 14: ['RTM'], 15: ['LW'], 16: ['RW'],
        17: [f'X{i}' for i in range(1, 17)],
        19: ['U1', 'U2'],
    }),
])
def test_real_capture_channel_names_resolve(fixture, expected):
    txt = (RESOURCES / fixture).read_text()
    block = get_peq_block_order(txt)[0]
    filters = _decode_raw_filters(txt, block)
    by_gain = {int(f.gain): f for f in filters}
    for gain, channel_names in expected.items():
        assert by_gain[gain].channel_names == channel_names, f"gain={gain}"


@pytest.mark.parametrize('fixture', ['mc35_all_channels.dsp', 'mc36_all_channels.dsp'])
def test_real_capture_filters_roundtrip(fixture):
    txt = (RESOURCES / fixture).read_text()
    block = get_peq_block_order(txt)[0]
    filters = _decode_raw_filters(txt, block)
    assert len(filters) == 19

    peq_key = get_peq_key_name(block)
    xml_filts = [filts_to_xml(f.get_all_vals()) for f in filters]
    round_tripped_txt = include_filters_in_dsp(peq_key, txt, xml_filts, replace=True)
    filters2 = _decode_raw_filters(round_tripped_txt, block)
    assert filters2 == filters

    # stable fixed point on a second pass
    xml_filts2 = [filts_to_xml(f.get_all_vals()) for f in filters2]
    round_tripped_txt2 = include_filters_in_dsp(peq_key, round_tripped_txt, xml_filts2, replace=True)
    assert round_tripped_txt2 == round_tripped_txt
