from model.minidsp import xml_to_filt
import os
from model.iir import PeakingEQ, LowShelf


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
