import json

from model.iir import ComplexLowPass, FilterType, from_json, ComplexHighPass, Passthrough, PeakingEQ


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
