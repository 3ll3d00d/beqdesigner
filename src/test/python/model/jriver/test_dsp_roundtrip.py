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


def test_five_one_plus_padding_uses_five_one_bed_not_seven_one():
    '''
    Same bug class as FIVE_ONE_TWO (a 5.1 bed has no RL/RR), but in the dynamically-constructed
    "N.N + N padding" path (codec.get_output_format's padded branch) rather than a static
    OUTPUT_FORMATS entry: a real 5.1+10 capture uses the 6-channel 5.1 bed (L/R/C/SW/SL/SR) plus
    10 Extra channels (X1-X10), not a padded-out 8-channel 7.1 bed plus only 8 Extra channels.
    '''
    txt = base_config(output_channels=6, padding=10)
    fmt = get_output_format(txt, allow_padding=True)
    names = [get_channel_name(i) for i in fmt.get_output_channel_indexes(use_atmos_channels=True)]
    assert 'RL' not in names and 'RR' not in names
    assert all(f'X{i}' in names for i in range(1, 11))
    assert 'X11' not in names


@pytest.mark.parametrize('padding,expected_extra', [
    (2, []),
    (4, ['X1']),
    (6, ['X1', 'X2', 'X3']),
])
def test_two_one_plus_padding_treats_first_3_padding_channels_as_a_nop(padding, expected_extra):
    '''
    Confirmed by the user: 2.1 is always sent by JRiver as a 6-channel container (L/R/C/SW/SL/SR),
    not a bare 3-channel L/R/SW signal - so the first 3 channels of "padding" beyond the 2.1 signal's
    own 3 channels are a nop (already accounted for by the container), and only padding beyond that
    (paddings step by 2, so this only ever bites at 4+) produces a genuinely new Extra channel.
    '''
    txt = base_config(output_channels=3, padding=padding)
    fmt = get_output_format(txt, allow_padding=True)
    names = [get_channel_name(i) for i in fmt.get_output_channel_indexes(use_atmos_channels=True)]
    assert [n for n in names if n.startswith('X')] == expected_extra


def test_mc35_immersive_output_formats_are_registered():
    for key in ['FIVE_ONE_TWO', 'SEVEN_ONE_FOUR', 'NINE_ONE_SIX']:
        fmt = OUTPUT_FORMATS[key]
        txt = base_config(output_channels=fmt.output_channels, padding=0)
        decoded = get_output_format(txt, allow_padding=False)
        assert decoded.output_channels == fmt.output_channels


def test_five_one_two_drops_rear_surrounds_for_atmos_height_channels():
    '''
    5.1.2 is a 5.1 bed (no rear surrounds) + 2 height channels, confirmed via a real JRiver
    Analyzer capture: L/R/C/SW/SL/SR/LTF/RTF (8 total), not the full 7.1 bed plus 2 more. Before
    the base_channels fix, get_all_channel_names always sliced from the fixed 8-channel surround
    base first, so a padded_instance-free = 8 output channel format like this one exhausted the
    slice on RL/RR before ever reaching the Atmos pool.
    '''
    fmt = OUTPUT_FORMATS['FIVE_ONE_TWO']
    names = [get_channel_name(i) for i in fmt.get_output_channel_indexes(use_atmos_channels=True)]
    assert 'LTF' in names and 'RTF' in names
    assert 'RL' not in names and 'RR' not in names
    input_names = [get_channel_name(i) for i in fmt.get_input_channel_indexes(use_atmos_channels=True)]
    assert input_names == ['L', 'R', 'C', 'SW', 'SL', 'SR', 'LTF', 'RTF']


def test_seven_one_four_and_nine_one_six_keep_full_seven_one_bed():
    for key, expected_atmos in [('SEVEN_ONE_FOUR', {'LTF', 'RTF', 'LTR', 'RTR'}),
                                 ('NINE_ONE_SIX', {'LTF', 'RTF', 'LTR', 'RTR', 'LTM', 'RTM', 'LW', 'RW'})]:
        fmt = OUTPUT_FORMATS[key]
        names = [get_channel_name(i) for i in fmt.get_output_channel_indexes(use_atmos_channels=True)]
        assert {'RL', 'RR'}.issubset(names)
        assert expected_atmos.issubset(names)


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


@pytest.mark.parametrize('use_atmos_channels', [False, True])
def test_real_capture_with_mixed_channel_eras_loads_via_full_jriverdsp(use_atmos_channels):
    '''
    A real-world MC35 config (not a synthetic diagnostic file): a 16-channel format (7.1 + 8 padding)
    whose Parametric Equalizer block includes BEQD's own XOBM/Multiway bass-management routing (using
    legacy spare channels C9-C12), a GEQ, an MSO import, and one isolated Atmos-channel Mix (a height
    channel downmix stub, channel 57/RTR) - all in the same PEQ block. This is a completely ordinary
    real config, not the exhaustive all-channels probe used elsewhere in this file. It's exactly the
    kind of file that broke end-to-end (JRiverDSP construction, not just decode) regardless of
    use_atmos_channels, for reasons now fixed:
      1. GraphRenderer assumed every channel a filter touches is a subset of the declared
         output_channels (fixed: nodes_by_channel is a defaultdict).
      2. OutputFormat.get_output_channel_indexes(use_atmos_channels=True) used to replace the legacy
         numbered ordering with atmos+extra entirely for a "+N padding channels" instance, discarding
         the C9-C12 range this file's own bass management depends on. This file is an MC35 capture, so
         its filters were originally recorded against the legacy pool; with use_atmos_channels=True
         (i.e. this same live/captured config now viewed under the 35.0.39+ scheme, e.g. after
         importing from a live MC35 instance over MCWS), OutputFormat.migrate_channel_index remaps
         those legacy indexes (13-16, C9-C12) onto their equivalent position in the newly-declared
         Extra pool (37-40, X1-X4) - see JRiverDSP._JRiverDSP__migrate_channels - so the filters keep
         controlling the same real channel instead of silently falling outside the declared set and
         becoming orphaned. The isolated RTR Mix (idx 57, already an Atmos-pool index) is untouched by
         migration either way and remains a genuine scratch channel - handled by the
         defaultdict/lazy-signals fixes below.
      2b. That per-filter migration alone wasn't enough: XOBM/Multiway complex filters additionally
          cache their own routing description (which channel each "way" is built from/goes to) as
          channel *names* directly in their Divider metadata JSON (e.g. Multiway's "i"/"o" fields,
          XOBM's "r" route list) - reported by a real user as a mixed-era result, e.g. an XOBM filter
          showing `[..., C11, C12, C15]` next to already-migrated `X`-named channels from its own
          constituent filters. Fixed via `ComplexFilter.migrate_channel_metadata`, overridden by
          `MultiwayFilter`/`XOFilter`/`CompoundRoutingFilter` to migrate the channel name(s) in their
          own metadata the same way, called from `JRiverDSP.__handle_divider` right before
          `filt_cls.create(...)`.
      3. FilterGraph.simulate() had the exact same "declared channels only" assumption as
         GraphRenderer, one layer down - its per-channel `signals` dict was built only from
         output_channels, so activating/simulating the graph (not just rendering it) raised a
         KeyError for the isolated RTR Mix. Fixed the same way: _ScratchChannelSignals lazily
         defaults an unlisted channel to silence instead of raising.
    '''
    txt = (RESOURCES / 'mc35_mixed_channel_eras.dsp').read_text()
    dsp = load(txt, allow_padding=True, use_atmos_channels=use_atmos_channels)
    assert dsp.output_format.output_channels == 16
    filters = dsp.graph(0).filters
    assert len(filters) == 10
    names = dsp.channel_names(output=True)
    def touched_indexes(f):
        idxs = list(getattr(f, 'channels', None) or [])
        if type(f).__name__ == 'Mix':
            idxs = [f.src_idx, f.dst_idx]
        return idxs

    all_touched = {c for f in filters for c in touched_indexes(f)}
    legacy_bass_mgmt = all_touched & {13, 14, 15, 16, 17}
    extra_bass_mgmt = all_touched & {37, 38, 39, 40, 41}
    if use_atmos_channels:
        assert all(c in names for c in ['X1', 'X2', 'X3', 'X4', 'X5'])
        assert not any(c in names for c in ['C9', 'C10', 'C11', 'C12', 'C13'])
        # migrated: no filter is still sat on the legacy C9-C13 indexes (including the Mix filter's
        # Destination), and they now show up on the equivalent Extra indexes (X1-X5) instead - not
        # left as orphaned scratch channels.
        assert not legacy_bass_mgmt
        assert extra_bass_mgmt == {37, 38, 39, 40, 41}
    else:
        assert all(c in names for c in ['C9', 'C10', 'C11', 'C12', 'C13'])
        assert legacy_bass_mgmt == {13, 14, 15, 16, 17}
        assert not extra_bass_mgmt
    # the isolated Atmos-channel Mix (idx 57/RTR) is already Atmos-pool and unaffected either way
    assert any(57 in touched_indexes(f) for f in filters)

    # the XOBM complex filter's own repr (built from its cached Divider-metadata channel names, not
    # just its constituent filters' raw indexes - see ComplexFilter.migrate_channel_metadata) must be
    # fully migrated too, not a mix of eras (this was the actual bug: leaf filters migrated correctly
    # while Multiway/XOBM's own "i"/"o"/"r" metadata stayed on the old names, e.g. "C11" and "X3" both
    # showing up for what should be the same, single, migrated channel).
    xobm_repr = next(repr(f) for f in filters if type(f).__name__ == 'CompoundRoutingFilter')
    if use_atmos_channels:
        assert 'C11' not in xobm_repr and 'C12' not in xobm_repr and 'C15' not in xobm_repr
        assert 'X3' in xobm_repr and 'X4' in xobm_repr and 'X7' in xobm_repr
    else:
        assert 'C11' in xobm_repr and 'C12' in xobm_repr and 'C15' in xobm_repr
        assert 'X3' not in xobm_repr and 'X4' not in xobm_repr and 'X7' not in xobm_repr

    dsp.activate(0)
    assert dsp.signals

    round_tripped = dsp.config_txt()
    dsp2 = load(round_tripped, allow_padding=True, use_atmos_channels=use_atmos_channels)
    assert dsp2.graph(0).filters == filters
    assert dsp2.config_txt() == round_tripped
    dsp2.activate(0)
    assert dsp2.signals


def test_migrate_channel_index_maps_legacy_pool_onto_atmos_or_extra_pool():
    '''
    OutputFormat.migrate_channel_index is what lets a filter recorded against the legacy generically-
    numbered pool (13-36, pre-35.0.39) keep controlling the same real channel once use_atmos_channels
    flips to True - see JRiverDSP.__migrate_channels and
    test_real_capture_with_mixed_channel_eras_loads_via_full_jriverdsp for the end-to-end version of
    this. Position within the legacy pool (Channel 9 = position 0, Channel 10 = position 1, ...) maps
    onto the same position within whichever pool the target format actually draws from:
      - a static, deliberately-immersive format (e.g. THIRTY_TWO) draws Atmos channels first, then Extra.
      - a dynamically-constructed "+N padding channels" instance draws Extra channels only (see
        test_padded_instance_uses_extra_channels_not_legacy_on_mc36).
    Indexes outside the legacy pool, or use_atmos_channels=False, are always a no-op - migration never
    touches base surround/user channels or already-migrated indexes.
    '''
    static_fmt = OUTPUT_FORMATS['THIRTY_TWO']
    # position 0 (Channel 9) -> LTF (Atmos-first pool)
    assert static_fmt.migrate_channel_index(13, True) == get_channel_idx('LTF')
    # position 7 (Channel 16) -> RW (last Atmos slot)
    assert static_fmt.migrate_channel_index(20, True) == get_channel_idx('RW')
    # position 8 (Channel 17) -> X1 (first Extra slot, after the 8 Atmos slots)
    assert static_fmt.migrate_channel_index(21, True) == get_channel_idx('X1')

    txt = base_config(output_channels=8, padding=8)
    padded_fmt = get_output_format(txt, allow_padding=True)
    # position 0 (Channel 9) -> X1 (Extra-only pool for a padded instance, no Atmos)
    assert padded_fmt.migrate_channel_index(13, True) == get_channel_idx('X1')
    assert padded_fmt.migrate_channel_index(20, True) == get_channel_idx('X8')

    # no-ops: use_atmos_channels=False, or an index outside the legacy pool entirely
    assert padded_fmt.migrate_channel_index(13, False) == 13
    assert padded_fmt.migrate_channel_index(2, True) == 2
    assert padded_fmt.migrate_channel_index(37, True) == 37


def test_padded_instance_uses_extra_channels_not_legacy_on_mc36():
    '''
    use_atmos_channels applies differently to a static, deliberately-immersive format (5.1.2/7.1.4/
    9.1.6/the 32-channel one) than to a dynamically-constructed "+N padding channels" instance:
      - static format: Atmos channels first, then Extra (unchanged - confirmed via the real
        mc35/mc36_all_channels.dsp captures).
      - padded instance: Extra channels only, never Atmos - confirmed empirically against a real MC36
        7.1+2 capture (mc36_seven_one_plus_two_padding.dsp): the 2 padding channels resolved to X1/X2,
        not Channel 9/10 (the pre-36 legacy pool) and not LTF/RTF (Atmos, reserved for static
        immersive layouts - a generic "+N padding channels" pick was never an immersive layout
        choice). See test_real_capture_with_mixed_channel_eras_loads_via_full_jriverdsp for the
        MC35 (legacy pool) side of this, and AGENTS.md "Channels beyond the base 8".
    '''
    static_fmt = OUTPUT_FORMATS['THIRTY_TWO']
    assert 54 in static_fmt.get_output_channel_indexes(use_atmos_channels=True)
    assert 13 not in static_fmt.get_output_channel_indexes(use_atmos_channels=True)
    assert 13 in static_fmt.get_output_channel_indexes(use_atmos_channels=False)

    txt = base_config(output_channels=8, padding=8)
    padded_fmt = get_output_format(txt, allow_padding=True)
    assert 13 in padded_fmt.get_output_channel_indexes(use_atmos_channels=False)
    assert 37 not in padded_fmt.get_output_channel_indexes(use_atmos_channels=False)
    assert 37 in padded_fmt.get_output_channel_indexes(use_atmos_channels=True)
    assert 13 not in padded_fmt.get_output_channel_indexes(use_atmos_channels=True)
    assert 54 not in padded_fmt.get_output_channel_indexes(use_atmos_channels=True)


def test_real_capture_seven_one_plus_two_padding_uses_extra_channels():
    '''
    A real MC36 capture: Output Channels=8, Output Padding Channels=2 (i.e. 7.1 + 2, picked via
    JRiver's own DSP Studio UI), with a single Peak filter deliberately placed on the 2 padding
    channels to test what they resolve to. JRiver's own Analyzer confirmed (screenshot, not part of
    this repo) the 2 extra meters for this config are labelled X1/X2 - and this file's Peak filter
    (in the enabled "Parametric Equalizer 2" block) targets raw indexes 37;38 accordingly, not 13;14
    (Channel 9/10) or 54;55 (LTF/RTF). This is the capture that grounds
    test_padded_instance_uses_extra_channels_not_legacy_on_mc36.
    '''
    txt = (RESOURCES / 'mc36_seven_one_plus_two_padding.dsp').read_text()
    dsp = load(txt, allow_padding=True, use_atmos_channels=True)
    assert dsp.output_format.output_channels == 10
    filters = dsp.graph(0).filters
    assert len(filters) == 1
    assert filters[0].channels == [37, 38]
    names = dsp.channel_names(output=True)
    assert 'X1' in names and 'X2' in names
    assert 'C9' not in names and 'C10' not in names

    dsp.activate(0)
    assert dsp.signals

    round_tripped = dsp.config_txt()
    dsp2 = load(round_tripped, allow_padding=True, use_atmos_channels=True)
    assert dsp2.graph(0).filters == filters
    assert dsp2.config_txt() == round_tripped
