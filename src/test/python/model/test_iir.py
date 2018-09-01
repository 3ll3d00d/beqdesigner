import json

from model.iir import ComplexLowPass, FilterType, from_json, ComplexHighPass, Passthrough, PeakingEQ, LowShelf, \
    HighShelf, FirstOrder_LowPass, FirstOrder_HighPass, SecondOrder_LowPass, SecondOrder_HighPass, AllPass, \
    CompleteFilter


def test_codec_Passthrough():
    filter = Passthrough()
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "Passthrough"}'
    decoded = from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, Passthrough)


def test_codec_PeakingEQ():
    filter = PeakingEQ(48000, 100, 0.707, 4.3)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "PeakingEQ", "fs": 48000, "fc": 100, "q": 0.707, "gain": 4.3}'
    decoded = from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, PeakingEQ)
    assert filter.fs == decoded.fs
    assert filter.q == decoded.q
    assert filter.gain == decoded.gain
    assert filter.freq == decoded.freq
    assert decoded.getTransferFunction() is not None


def test_codec_LowShelf():
    filter = LowShelf(48000, 20, 1.5, 2.5)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "LowShelf", "fs": 48000, "fc": 20, "q": 1.5, "gain": 2.5, "count": 1}'
    decoded = from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, LowShelf)
    assert filter.fs == decoded.fs
    assert filter.q == decoded.q
    assert filter.gain == decoded.gain
    assert filter.freq == decoded.freq
    assert filter.count == decoded.count
    assert decoded.getTransferFunction() is not None


def test_codec_StackedLowShelf():
    filter = LowShelf(48000, 20, 1.5, 2.5, count=5)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "LowShelf", "fs": 48000, "fc": 20, "q": 1.5, "gain": 2.5, "count": 5}'
    decoded = from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, LowShelf)
    assert filter.fs == decoded.fs
    assert filter.q == decoded.q
    assert filter.gain == decoded.gain
    assert filter.freq == decoded.freq
    assert filter.count == decoded.count
    assert decoded.getTransferFunction() is not None


def test_codec_HighShelf():
    filter = HighShelf(48000, 20, 1.5, 2.5)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "HighShelf", "fs": 48000, "fc": 20, "q": 1.5, "gain": 2.5, "count": 1}'
    decoded = from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, HighShelf)
    assert filter.fs == decoded.fs
    assert filter.q == decoded.q
    assert filter.gain == decoded.gain
    assert filter.freq == decoded.freq
    assert filter.count == decoded.count
    assert decoded.getTransferFunction() is not None


def test_codec_StackedHighShelf():
    filter = HighShelf(48000, 20, 1.5, 2.5, count=5)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "HighShelf", "fs": 48000, "fc": 20, "q": 1.5, "gain": 2.5, "count": 5}'
    decoded = from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, HighShelf)
    assert filter.fs == decoded.fs
    assert filter.q == decoded.q
    assert filter.gain == decoded.gain
    assert filter.freq == decoded.freq
    assert filter.count == decoded.count
    assert decoded.getTransferFunction() is not None


def test_codec_FirstOrderLowPass():
    filter = FirstOrder_LowPass(48000, 2000, 1.5)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "FirstOrder_LowPass", "fs": 48000, "fc": 2000, "q": 1.5}'
    decoded = from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, FirstOrder_LowPass)
    assert filter.fs == decoded.fs
    assert filter.q == decoded.q
    assert filter.freq == decoded.freq
    assert decoded.getTransferFunction() is not None


def test_codec_FirstOrderHighPass():
    filter = FirstOrder_HighPass(48000, 2000, 1.5)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "FirstOrder_HighPass", "fs": 48000, "fc": 2000, "q": 1.5}'
    decoded = from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, FirstOrder_HighPass)
    assert filter.fs == decoded.fs
    assert filter.q == decoded.q
    assert filter.freq == decoded.freq
    assert decoded.getTransferFunction() is not None


def test_codec_SecondOrderLowPass():
    filter = SecondOrder_LowPass(48000, 2000, 1.5)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "SecondOrder_LowPass", "fs": 48000, "fc": 2000, "q": 1.5}'
    decoded = from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, SecondOrder_LowPass)
    assert filter.fs == decoded.fs
    assert filter.q == decoded.q
    assert filter.freq == decoded.freq
    assert decoded.getTransferFunction() is not None


def test_codec_SecondOrderHighPass():
    filter = SecondOrder_HighPass(48000, 2000, 1.5)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "SecondOrder_HighPass", "fs": 48000, "fc": 2000, "q": 1.5}'
    decoded = from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, SecondOrder_HighPass)
    assert filter.fs == decoded.fs
    assert filter.q == decoded.q
    assert filter.freq == decoded.freq
    assert decoded.getTransferFunction() is not None


def test_codec_AllPass():
    filter = AllPass(1000, 250, 1.5)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "AllPass", "fs": 1000, "fc": 250, "q": 1.5}'
    decoded = from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, AllPass)
    assert filter.fs == decoded.fs
    assert filter.q == decoded.q
    assert filter.freq == decoded.freq
    assert decoded.getTransferFunction() is not None


def test_codec_ComplexLowPass():
    filter = ComplexLowPass(FilterType.BUTTERWORTH, 2, 48000, 100)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "ComplexLowPass", "filter_type": "BW", "order": 2, "fs": 48000, "fc": 100}'
    decoded = from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, ComplexLowPass)
    assert filter.fs == decoded.fs
    assert filter.type == decoded.type
    assert filter.order == decoded.order
    assert filter.freq == decoded.freq
    assert decoded.getTransferFunction() is not None


def test_codec_ComplexHighPass():
    filter = ComplexHighPass(FilterType.LINKWITZ_RILEY, 2, 48000, 100)
    output = json.dumps(filter.to_json())
    assert output == '{"_type": "ComplexHighPass", "filter_type": "LR", "order": 2, "fs": 48000, "fc": 100}'
    decoded = from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, ComplexHighPass)
    assert filter.fs == decoded.fs
    assert filter.type == decoded.type
    assert filter.order == decoded.order
    assert filter.freq == decoded.freq
    assert decoded.getTransferFunction() is not None


def test_codec_CompleteFilter():
    filters = [PeakingEQ(1000, 50, 3.2, -5), LowShelf(1000, 25, 1, 3.2, count=3),
               ComplexHighPass(FilterType.BUTTERWORTH, 6, 1000, 12)]
    filter = CompleteFilter(filters=filters, description='Hello from me')
    output = json.dumps(filter.to_json())
    expected = '{"_type": "CompleteFilter", "description": "Hello from me", "filters": [' \
               '{"_type": "PeakingEQ", "fs": 1000, "fc": 50, "q": 3.2, "gain": -5}, ' \
               '{"_type": "LowShelf", "fs": 1000, "fc": 25, "q": 1, "gain": 3.2, "count": 3}, ' \
               '{"_type": "ComplexHighPass", "filter_type": "BW", "order": 6, "fs": 1000, "fc": 12}' \
               ']}'
    assert output == expected
    decoded = from_json(json.loads(output))
    assert decoded is not None
    assert isinstance(decoded, CompleteFilter)
    assert decoded.description == 'Hello from me'
    assert decoded.filters is not None
    assert len(decoded.filters) == len(filter.filters)
    assert decoded.getTransferFunction() is not None
    assert isinstance(decoded.filters[0], filter.filters[0].__class__)
    assert filter.filters[0].fs == decoded.filters[0].fs
    assert filter.filters[0].q == decoded.filters[0].q
    assert filter.filters[0].gain == decoded.filters[0].gain
    assert filter.filters[0].freq == decoded.filters[0].freq
    assert isinstance(decoded.filters[1], filter.filters[1].__class__)
    assert filter.filters[1].fs == decoded.filters[1].fs
    assert filter.filters[1].q == decoded.filters[1].q
    assert filter.filters[1].gain == decoded.filters[1].gain
    assert filter.filters[1].freq == decoded.filters[1].freq
    assert filter.filters[1].count == decoded.filters[1].count
    assert isinstance(decoded.filters[2], filter.filters[2].__class__)
    assert filter.filters[2].fs == decoded.filters[2].fs
    assert filter.filters[2].type == decoded.filters[2].type
    assert filter.filters[2].order == decoded.filters[2].order
    assert filter.filters[2].freq == decoded.filters[2].freq
