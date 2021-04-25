from model.jriver.common import SHORT_USER_CHANNELS, get_channel_idx
from model.jriver.filter import FilterGraph, Mix, MixType, print_node, Peak, XOFilter, LowPass, HighPass
from model.jriver.render import GraphRenderer
from model.jriver.routing import Matrix, calculate_compound_routing_filter


def test_simple():
    channels = ['L', 'R']

    def peak(freq: int, c: str):
        return Peak({
            **Peak.default_values(),
            'Q': '1.0',
            'Gain': '5.0',
            'Frequency': f"{freq}",
            'Channels': f"{get_channel_idx(c)}"
        })

    filters = [peak(100, 'L'), peak(1000, 'R'), peak(500, 'L')]
    graph = FilterGraph(0, channels + SHORT_USER_CHANNELS, channels + SHORT_USER_CHANNELS, filters)
    assert graph.nodes_by_channel
    for c in ['L', 'R']:
        printed = print_node(c, graph.nodes_by_channel[c])
        print()
        print(printed)
    gz = GraphRenderer(graph).generate(False)
    assert gz
    signals_by_channel = graph.simulate()
    assert signals_by_channel


def test_circular_add():
    channels = ['L', 'R']

    def mix(f: str, t: str, mt: MixType = MixType.ADD):
        return Mix({
            **Mix.default_values(),
            'Source': str(get_channel_idx(f)),
            'Destination': str(get_channel_idx(t)),
            'Mode': str(mt.value)
        })

    filters = [mix('L', 'U1'), mix('R', 'U1'), mix('U1', 'R', MixType.COPY)]
    graph = FilterGraph(0, channels + SHORT_USER_CHANNELS, channels + SHORT_USER_CHANNELS, filters)
    assert graph.nodes_by_channel
    for c in ['L', 'R', 'U1']:
        printed = print_node(c, graph.nodes_by_channel[c])
        print()
        print(printed)
    gz = GraphRenderer(graph).generate(False)
    assert gz
    signals_by_channel = graph.simulate()
    assert signals_by_channel


def test_circular_copy():
    def mix(f: str, t: str, mt: MixType = MixType.ADD):
        return Mix({
            **Mix.default_values(),
            'Source': str(get_channel_idx(f)),
            'Destination': str(get_channel_idx(t)),
            'Mode': str(mt.value)
        })

    filters = [mix('L', 'U1', MixType.COPY), mix('R', 'C'), mix('L', 'C'), mix('U1', 'L', MixType.COPY)]
    graph = FilterGraph(0, ['L', 'R'] + SHORT_USER_CHANNELS, ['L', 'R', 'C', 'SW'] + SHORT_USER_CHANNELS, filters)
    assert graph.nodes_by_channel
    gz = GraphRenderer(graph).generate(False)
    assert gz
    signals_by_channel = graph.simulate()
    assert signals_by_channel


def test_stereo_subs():
    mains = ['L', 'R', 'C', 'SL', 'SR']
    input_channels = mains + ['SW']
    output_channels = input_channels + ['RL', 'RR']
    matrix = Matrix({'L': 2, 'R': 2, 'C': 2, 'SW': 1, 'SL': 2, 'SR': 2}, output_channels)
    for c in ['L', 'C', 'SW', 'SL']:
        matrix.enable(c, 0, 'SW')
    for c in ['R', 'C', 'SW', 'SR']:
        matrix.enable(c, 0, 'RR')
    for c in ['L', 'R', 'C', 'SL', 'SR']:
        matrix.enable(c, 1, c)
    xo_filters = [XOFilter('SW', 0, [low_pass(2, 80, 'SW')])] + [XOFilter(c, 1, [high_pass(2, 80, c)]) for c in mains]
    crf = calculate_compound_routing_filter(matrix, xo_filters=xo_filters, lfe_channel_idx=5)
    crf.id = 1
    for i, f in enumerate(crf.filters):
        f.id = i + 2
    graph = FilterGraph(0, input_channels + SHORT_USER_CHANNELS, output_channels + SHORT_USER_CHANNELS, [crf])
    assert graph.nodes_by_channel
    signals_by_channel = graph.simulate()
    assert signals_by_channel
    gz = GraphRenderer(graph).generate(False)
    print()
    print(gz)
    assert gz


def low_pass(order: int, freq: int, channel: str) -> LowPass:
    return LowPass({
        **LowPass.default_values(),
        'Slope': f"{order * 6}",
        'Q': '1',
        'Gain': '0.0',
        'Frequency': f"{freq}",
        'Channels': f"{get_channel_idx(channel)}"
    })


def high_pass(order: int, freq: int, channel: str) -> HighPass:
    return HighPass({
        **HighPass.default_values(),
        'Slope': f"{order * 6}",
        'Q': '1',
        'Gain': '0.0',
        'Frequency': f"{freq}",
        'Channels': f"{get_channel_idx(channel)}"
    })
