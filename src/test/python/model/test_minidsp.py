import io
import os

from model.iir import PeakingEQ, LowShelf
from model.merge import DspType
from model.minidsp import xml_to_filt, HDXmlParser, pad_with_passthrough


def test_codec_minidsp_xml():
    filts = xml_to_filt(os.path.join(os.path.dirname(__file__), 'minidsp.xml'))
    assert filts
    assert len(filts) == 2
    assert type(filts[0]) is PeakingEQ
    assert filts[0].freq == 45.0
    assert filts[0].gain == 0.4
    assert filts[0].q == 1.0
    assert type(filts[1]) is LowShelf
    assert filts[1].freq == 19.0
    assert filts[1].gain == 3.8
    assert filts[1].q == 0.9
    assert filts[1].count == 3


def test_merge_2x4hd():
    dsp_type = DspType.MINIDSP_TWO_BY_FOUR_HD
    parser = HDXmlParser(dsp_type, False, None)
    filt = xml_to_filt(os.path.join(os.path.dirname(__file__), 'input.xml'))
    filts = pad_with_passthrough(filt, dsp_type.target_fs, dsp_type.filters_required)
    with open('MiniDSP-2x4HD-setting.xml', 'r') as f1:
        with open('expected_output_2x4HD.xml', 'r') as f2:
            convert_and_compare(f1, f2, filts, parser)


def test_merge_2x4hd_output_with_output():
    dsp_type = DspType.MINIDSP_TWO_BY_FOUR_HD
    parser = HDXmlParser(dsp_type, False, [str(i) for i in range(1, 7)])
    filt = xml_to_filt(os.path.join(os.path.dirname(__file__), 'input.xml'))
    filts = pad_with_passthrough(filt, dsp_type.target_fs, dsp_type.filters_required)
    with open('MiniDSP-2x4HD-setting.xml', 'r') as f1:
        with open('expected_output_2x4HD_output.xml', 'r') as f2:
            convert_and_compare(f1, f2, filts, parser)


def convert_and_compare(f1, f2, filts, parser):
    dst = io.StringIO(f1.read())
    expected = f2.read()
    actual, optimised = parser.convert(dst, filts)
    assert optimised is False
    assert len(actual) == len(expected)
    assert actual == expected
