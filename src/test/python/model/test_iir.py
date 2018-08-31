import json

from model.iir import ComplexLowPass, FilterType, from_json, ComplexHighPass, Passthrough, PeakingEQ, LowShelf, \
    HighShelf, FirstOrder_LowPass, FirstOrder_HighPass, SecondOrder_LowPass, SecondOrder_HighPass, AllPass


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
