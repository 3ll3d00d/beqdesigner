import json
import unittest

from model.iir import ComplexLowPass, FilterType, from_json, ComplexHighPass, Passthrough, PeakingEQ


class TestFilterEncoding(unittest.TestCase):
    def test_codec_Passthrough(self):
        filter = Passthrough()
        output = json.dumps(filter.to_json())
        self.assertEqual(output, '{"_type": "Passthrough"}')
        decoded = from_json(json.loads(output))
        self.assertIsNotNone(decoded)
        self.assertIsInstance(decoded, Passthrough)

    def test_codec_PeakingEQ(self):
        filter = PeakingEQ(48000, 100, 0.707, 4.3)
        output = json.dumps(filter.to_json())
        self.assertEqual(output, '{"_type": "PeakingEQ", "fs": 48000, "fc": 100, "q": 0.707, "gain": 4.3}')
        decoded = from_json(json.loads(output))
        self.assertIsNotNone(decoded)
        self.assertIsInstance(decoded, PeakingEQ)
        self.assertEqual(filter.fs, decoded.fs)
        self.assertEqual(filter.q, decoded.q)
        self.assertEqual(filter.gain, decoded.gain)
        self.assertEqual(filter.freq, decoded.freq)
        self.assertIsNotNone(decoded.getTransferFunction())

    def test_codec_ComplexLowPass(self):
        filter = ComplexLowPass(FilterType.BUTTERWORTH, 2, 48000, 100)
        output = json.dumps(filter.to_json())
        self.assertEqual(output, '{"_type": "ComplexLowPass", "filter_type": "BW", "order": 2, "fs": 48000, "fc": 100}')
        decoded = from_json(json.loads(output))
        self.assertIsNotNone(decoded)
        self.assertIsInstance(decoded, ComplexLowPass)
        self.assertEqual(filter.fs, decoded.fs)
        self.assertEqual(filter.type, decoded.type)
        self.assertEqual(filter.order, decoded.order)
        self.assertEqual(filter.freq, decoded.freq)
        self.assertIsNotNone(decoded.getTransferFunction())

    def test_codec_ComplexHighPass(self):
        filter = ComplexHighPass(FilterType.LINKWITZ_RILEY, 2, 48000, 100)
        output = json.dumps(filter.to_json())
        self.assertEqual(output,
                         '{"_type": "ComplexHighPass", "filter_type": "LR", "order": 2, "fs": 48000, "fc": 100}')
        decoded = from_json(json.loads(output))
        self.assertIsNotNone(decoded)
        self.assertIsInstance(decoded, ComplexHighPass)
        self.assertEqual(filter.fs, decoded.fs)
        self.assertEqual(filter.type, decoded.type)
        self.assertEqual(filter.order, decoded.order)
        self.assertEqual(filter.freq, decoded.freq)
        self.assertIsNotNone(decoded.getTransferFunction())
