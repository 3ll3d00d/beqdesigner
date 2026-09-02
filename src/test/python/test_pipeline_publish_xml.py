'''
Phase 2 (item 2c) of design/pipeline-implementation-plan.md: to_beq_xml()
wraps the existing HDXmlParser path. Re-runs test_pipeline_publish_roundtrip.py's
RP1 check end to end through the new pipeline code -- from a fake designer's
DesignResponse, through the contract boundary (pipeline.designer.convert),
through pipeline.metadata.BeqMetadata, to a beqcatalogue XML string -- and
must agree with the Phase 0 baseline.
'''
import os

from model.iir import LowShelf, PeakingEQ
from model.minidsp import xml_to_filt
from pipeline.designer.contract import BiquadSpec, DesignResponse
from pipeline.designer.convert import to_complete_filter
from pipeline.metadata import BeqMetadata
from pipeline.publish.xml import flat24hd_template_path, to_beq_xml


def test_flat24hd_template_path_resolves_regardless_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # prove this doesn't depend on the process cwd
    path = flat24hd_template_path()
    assert os.path.isfile(path), path


def test_to_beq_xml_end_to_end_from_a_fake_designer_response():
    '''
    The same RP1 numbers as test_pipeline_publish_roundtrip.py, but produced
    via a DesignResponse -> to_complete_filter -> to_beq_xml, not built by
    hand. Both paths must publish the same filters.
    '''
    response = DesignResponse(
        contract_version='1.0',
        filters=[
            BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7),
            BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7),
            BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7),
            BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7),
            BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7),
            BiquadSpec(type='peaking_eq', freq_hz=40.0, gain_db=-3.0, q=2.0),
        ],
        confidence=0.9,
        mv_adjust_db=22.5,
        method='fitted',
    )
    complete_filter = to_complete_filter(response, fs=96000)

    meta = BeqMetadata(title='Ready Player One', year='2018', audio_types=['Atmos'],
                       genres=[{'id': 28, 'name': 'Action'}, {'id': 878, 'name': 'Science Fiction'}])

    output_xml = to_beq_xml(complete_filter, meta)
    assert output_xml
    assert '<beq_title>Ready Player One</beq_title>' in output_xml
    assert '<genre id="28">Action</genre>' in output_xml

    # round-trip it, same assertions as the Phase 0 baseline
    written = os.path.join(os.path.dirname(flat24hd_template_path()), '__test_output_2c__.xml')
    with open(written, 'w', encoding='utf-8') as f:
        f.write(output_xml)
    try:
        read_back = xml_to_filt(written, fs=96000)
    finally:
        os.remove(written)

    low_shelves = [f for f in read_back if isinstance(f, LowShelf)]
    peaks = [f for f in read_back if isinstance(f, PeakingEQ)]
    assert len(low_shelves) == 1
    assert low_shelves[0].count == 5
    assert low_shelves[0].gain * low_shelves[0].count == 22.5
    assert len(peaks) == 1


def test_to_beq_xml_rejects_invalid_metadata():
    from pipeline.metadata import BeqMetadata
    import pytest

    meta = BeqMetadata(title='', year='', audio_types=[])
    with pytest.raises(ValueError, match='Invalid metadata'):
        to_beq_xml([PeakingEQ(96000, 40.0, 2.0, -3.0)], meta)


def test_pipeline_publish_xml_module_has_no_qtpy_import():
    import ast
    import pathlib
    source = (pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python' / 'pipeline' / 'publish' / 'xml.py').resolve()
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith('qtpy') for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith('qtpy')
