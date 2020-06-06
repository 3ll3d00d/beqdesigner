import json

from model.codec import filter_from_json, signaldata_from_json, signaldata_to_json
from model.iir import ComplexLowPass, FilterType, ComplexHighPass, Passthrough, PeakingEQ, FirstOrder_LowPass, \
    FirstOrder_HighPass, SecondOrder_LowPass, SecondOrder_HighPass, AllPass, LowShelf, CompleteFilter, HighShelf, Gain
from model.signal import SingleChannelSignalData


def test_codec_Passthrough():
    filter = Passthrough()
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "Passthrough", "fs": 1000}'
    decoded = filter_from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, Passthrough)
    assert filter.fs == decoded.fs


def test_codec_Passthrough_with_fs():
    filter = Passthrough(fs=2000)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "Passthrough", "fs": 2000}'
    decoded = filter_from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, Passthrough)
    assert filter.fs == decoded.fs


def test_codec_Gain():
    filter = Gain(1000, 10.0)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "Gain", "fs": 1000, "gain": 10.0}'
    decoded = filter_from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, Gain)
    assert filter.fs == decoded.fs
    assert filter.gain == decoded.gain
    assert decoded.get_transfer_function() is not None


def test_codec_PeakingEQ():
    filter = PeakingEQ(48000, 100, 0.707, 4.3)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "PeakingEQ", "fs": 48000, "fc": 100.0, "q": 0.707, "gain": 4.3}'
    decoded = filter_from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, PeakingEQ)
    assert filter.fs == decoded.fs
    assert filter.q == decoded.q
    assert filter.gain == decoded.gain
    assert filter.freq == decoded.freq
    assert decoded.get_transfer_function() is not None


def test_codec_LowShelf():
    filter = LowShelf(48000, 20, 1.5, 2.5)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "LowShelf", "fs": 48000, "fc": 20.0, "q": 1.5, "gain": 2.5, "count": 1}'
    decoded = filter_from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, LowShelf)
    assert filter.fs == decoded.fs
    assert filter.q == decoded.q
    assert filter.gain == decoded.gain
    assert filter.freq == decoded.freq
    assert filter.count == decoded.count
    assert decoded.get_transfer_function() is not None


def test_codec_StackedLowShelf():
    filter = LowShelf(48000, 20, 1.5, 2.5, count=5)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "LowShelf", "fs": 48000, "fc": 20.0, "q": 1.5, "gain": 2.5, "count": 5}'
    decoded = filter_from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, LowShelf)
    assert filter.fs == decoded.fs
    assert filter.q == decoded.q
    assert filter.gain == decoded.gain
    assert filter.freq == decoded.freq
    assert filter.count == decoded.count
    assert decoded.get_transfer_function() is not None


def test_codec_HighShelf():
    filter = HighShelf(48000, 20, 1.5, 2.5)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "HighShelf", "fs": 48000, "fc": 20.0, "q": 1.5, "gain": 2.5, "count": 1}'
    decoded = filter_from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, HighShelf)
    assert filter.fs == decoded.fs
    assert filter.q == decoded.q
    assert filter.gain == decoded.gain
    assert filter.freq == decoded.freq
    assert filter.count == decoded.count
    assert decoded.get_transfer_function() is not None


def test_codec_StackedHighShelf():
    filter = HighShelf(48000, 20, 1.5, 2.5, count=5)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "HighShelf", "fs": 48000, "fc": 20.0, "q": 1.5, "gain": 2.5, "count": 5}'
    decoded = filter_from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, HighShelf)
    assert filter.fs == decoded.fs
    assert filter.q == decoded.q
    assert filter.gain == decoded.gain
    assert filter.freq == decoded.freq
    assert filter.count == decoded.count
    assert decoded.get_transfer_function() is not None


def test_codec_FirstOrderLowPass():
    filter = FirstOrder_LowPass(48000, 2000, 1.5)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "FirstOrder_LowPass", "fs": 48000, "fc": 2000.0, "q": 1.5}'
    decoded = filter_from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, FirstOrder_LowPass)
    assert filter.fs == decoded.fs
    assert filter.q == decoded.q
    assert filter.freq == decoded.freq
    assert decoded.get_transfer_function() is not None


def test_codec_FirstOrderHighPass():
    filter = FirstOrder_HighPass(48000, 2000, 1.5)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "FirstOrder_HighPass", "fs": 48000, "fc": 2000.0, "q": 1.5}'
    decoded = filter_from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, FirstOrder_HighPass)
    assert filter.fs == decoded.fs
    assert filter.q == decoded.q
    assert filter.freq == decoded.freq
    assert decoded.get_transfer_function() is not None


def test_codec_SecondOrderLowPass():
    filter = SecondOrder_LowPass(48000, 2000, 1.5)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "SecondOrder_LowPass", "fs": 48000, "fc": 2000.0, "q": 1.5}'
    decoded = filter_from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, SecondOrder_LowPass)
    assert filter.fs == decoded.fs
    assert filter.q == decoded.q
    assert filter.freq == decoded.freq
    assert decoded.get_transfer_function() is not None


def test_codec_SecondOrderHighPass():
    filter = SecondOrder_HighPass(48000, 2000, 1.5)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "SecondOrder_HighPass", "fs": 48000, "fc": 2000.0, "q": 1.5}'
    decoded = filter_from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, SecondOrder_HighPass)
    assert filter.fs == decoded.fs
    assert filter.q == decoded.q
    assert filter.freq == decoded.freq
    assert decoded.get_transfer_function() is not None


def test_codec_AllPass():
    filter = AllPass(1000, 250, 1.5)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "AllPass", "fs": 1000, "fc": 250.0, "q": 1.5}'
    decoded = filter_from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, AllPass)
    assert filter.fs == decoded.fs
    assert filter.q == decoded.q
    assert filter.freq == decoded.freq
    assert decoded.get_transfer_function() is not None


def test_codec_ComplexLowPass():
    filter = ComplexLowPass(FilterType.BUTTERWORTH, 2, 48000, 100)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "ComplexLowPass", "filter_type": "BW", "order": 2, "fs": 48000, "fc": 100.0}'
    decoded = filter_from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, ComplexLowPass)
    assert filter.fs == decoded.fs
    assert filter.type == decoded.type
    assert filter.order == decoded.order
    assert filter.freq == decoded.freq
    assert decoded.get_transfer_function() is not None


def test_codec_ComplexHighPass():
    filter = ComplexHighPass(FilterType.LINKWITZ_RILEY, 2, 48000, 100)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "ComplexHighPass", "filter_type": "LR", "order": 2, "fs": 48000, "fc": 100.0}'
    decoded = filter_from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, ComplexHighPass)
    assert filter.fs == decoded.fs
    assert filter.type == decoded.type
    assert filter.order == decoded.order
    assert filter.freq == decoded.freq
    assert decoded.get_transfer_function() is not None


def test_codec_CompleteFilter():
    filter = CompleteFilter(filters=[PeakingEQ(1000, 50, 3.2, -5),
                                     LowShelf(1000, 25, 1, 3.2, count=3),
                                     ComplexHighPass(FilterType.BUTTERWORTH, 6, 1000, 12)],
                            description='Hello from me')
    output = json.dumps(filter.to_json())
    expected = '{"_type": "CompleteFilter", "description": "Hello from me", "fs": 1000, "filters": [' \
               '{"_type": "ComplexHighPass", "filter_type": "BW", "order": 6, "fs": 1000, "fc": 12.0}, ' \
               '{"_type": "LowShelf", "fs": 1000, "fc": 25.0, "q": 1.0, "gain": 3.2, "count": 3}, ' \
               '{"_type": "PeakingEQ", "fs": 1000, "fc": 50.0, "q": 3.2, "gain": -5.0}' \
               ']}'
    assert output == expected
    decoded = filter_from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, CompleteFilter)
    assert decoded.description == 'Hello from me'
    assert decoded.filters is not None
    assert len(decoded.filters) == len(filter.filters)
    assert decoded.get_transfer_function() is not None
    assert isinstance(decoded.filters[0], filter.filters[0].__class__)
    assert filter.filters[0].fs == decoded.filters[0].fs
    assert filter.filters[0].type == decoded.filters[0].type
    assert filter.filters[0].order == decoded.filters[0].order
    assert filter.filters[0].freq == decoded.filters[0].freq
    assert isinstance(decoded.filters[1], filter.filters[1].__class__)
    assert filter.filters[1].fs == decoded.filters[1].fs
    assert filter.filters[1].q == decoded.filters[1].q
    assert filter.filters[1].gain == decoded.filters[1].gain
    assert filter.filters[1].freq == decoded.filters[1].freq
    assert filter.filters[1].count == decoded.filters[1].count
    assert isinstance(decoded.filters[2], filter.filters[2].__class__)
    assert filter.filters[2].fs == decoded.filters[2].fs
    assert filter.filters[2].q == decoded.filters[2].q
    assert filter.filters[2].gain == decoded.filters[2].gain
    assert filter.filters[2].freq == decoded.filters[2].freq


def test_codec_signal():
    fs = 1000
    peak = LowShelf(fs, 30, 1, 10, count=2).get_transfer_function().get_magnitude()
    avg = LowShelf(fs, 30, 1, 10).get_transfer_function().get_magnitude()
    median = LowShelf(fs, 30, 1, 10).get_transfer_function().get_magnitude()
    filt = CompleteFilter()
    filt.save(HighShelf(fs, 60, 1, 5, count=2))
    data = SingleChannelSignalData('test', fs, xy_data=[avg, peak, median], filter=filt, duration_seconds=123456, start_seconds=123, offset=4.2)
    output = json.dumps(signaldata_to_json(data))
    assert output is not None
    decoded = signaldata_from_json(json.loads(output), None)
    assert decoded is not None
    assert isinstance(decoded, SingleChannelSignalData)
    assert decoded.name == data.name
    assert decoded.fs == data.fs
    assert decoded.filter is not None
    assert type(decoded.filter) is type(data.filter)
    assert decoded.filter.id != -1
    assert decoded.filter.description == data.filter.description
    assert decoded.filter.filters == data.filter.filters
    assert decoded.current_unfiltered is not None
    assert len(decoded.current_unfiltered) == 3
    assert decoded.current_unfiltered == data.current_unfiltered
    assert decoded.duration_hhmmss == data.duration_hhmmss
    assert decoded.start_hhmmss == data.start_hhmmss
    assert decoded.end_hhmmss == data.end_hhmmss
    assert decoded.offset == data.offset
