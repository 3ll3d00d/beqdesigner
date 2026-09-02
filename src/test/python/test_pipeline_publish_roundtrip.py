'''
Phase 0 of design/pipeline-implementation-plan.md: pin the beqcatalogue
publish contract against the code as it stands today, before any of the
pipeline/ refactoring starts. No production code is exercised here beyond
what already exists — model.minidsp/model.iir/model.merge.

Uses the Ready Player One filter values from docs/workflow/beq.md (a
LowShelf stacked 5x plus a PeakingEQ) as the worked example, matching
design/api-headless-pipeline.md §11.1.
'''
import os


FLAT24HD_XML = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'main', 'xml', 'flat24hd.xml'))


def build_rp1_filters(fs):
    from model.iir import LowShelf, PeakingEQ
    return [
        LowShelf(fs, 18.0, 0.7, 4.5, count=5),
        PeakingEQ(fs, 40.0, 2.0, -3.0),
    ]


def build_metadata():
    return {
        'beq_title': 'Ready Player One',
        'beq_alt_title': '',
        'beq_sortTitle': 'Ready Player One',
        'beq_year': '2018',
        'beq_spectrumURL': '',
        'beq_pvaURL': '',
        'beq_edition': '',
        'beq_season': '',
        'beq_note': '',
        'beq_warning': '',
        'beq_gain': '-4.0',
        'beq_language': 'English',
        'beq_source': 'Disc',
        'beq_overview': '',
        'beq_rating': '',
        'beq_author': '',
        'beq_avs': '',
        'beq_theMovieDB': '335984',
        'beq_poster': '',
        'beq_runtime': '140',
        'beq_collection': None,
        'beq_audioTypes': ['Atmos'],
        'beq_genres': [{'id': 28, 'name': 'Action'}, {'id': 878, 'name': 'Science Fiction'}],
    }


def test_publish_round_trip():
    '''
    Build filters + metadata by hand, write beqcatalogue XML via HDXmlParser
    against the bundled flat24hd.xml template, read it back with
    xml_to_filt, and assert the filters and metadata survive.
    '''
    from model.merge import DspType
    from model.minidsp import HDXmlParser, xml_to_filt
    from model.iir import LowShelf, PeakingEQ

    assert os.path.isfile(FLAT24HD_XML), f'missing template at {FLAT24HD_XML}'

    dsp_type = DspType.MINIDSP_TWO_BY_FOUR_HD
    filt = build_rp1_filters(fs=1000)
    metadata = build_metadata()

    parser = HDXmlParser(dsp_type, False)
    output_xml, was_optimised = parser.convert(FLAT24HD_XML, filt, metadata, pretty=True)

    assert was_optimised is False
    assert output_xml

    # --- filters survive the round trip, including stacked shelf count ---
    # unroll=False (the default) keeps the LowShelf's count intact rather
    # than exploding it into 5 separate biquads -- losing that count is the
    # failure mode that matters, since it would silently publish a 4.5 dB
    # filter instead of a 22.5 dB one.
    written = os.path.join(os.path.dirname(FLAT24HD_XML), '__test_output__.xml')
    with open(written, 'w', encoding='utf-8') as f:
        f.write(output_xml)
    try:
        read_back = xml_to_filt(written, fs=dsp_type.target_fs)
    finally:
        os.remove(written)

    low_shelves = [f for f in read_back if isinstance(f, LowShelf)]
    peaks = [f for f in read_back if isinstance(f, PeakingEQ)]
    assert len(low_shelves) == 1
    assert len(peaks) == 1

    shelf = low_shelves[0]
    assert shelf.freq == 18.0
    assert shelf.q == 0.7
    assert shelf.gain == 4.5
    assert shelf.count == 5
    assert shelf.gain * shelf.count == 22.5

    peak = peaks[0]
    assert peak.freq == 40.0
    assert peak.q == 2.0
    assert peak.gain == -3.0

    # --- biquad budget: 5 (unrolled shelf) + 1 (peak) = 6, well under 10 ---
    from model.minidsp import flatten_filters
    assert len(flatten_filters(filt)) == 6

    # --- metadata survives, including nested id-bearing genre elements ---
    assert '<beq_title>Ready Player One</beq_title>' in output_xml
    assert '<beq_year>2018</beq_year>' in output_xml
    assert '<beq_gain>-4.0</beq_gain>' in output_xml
    assert '<genre id="28">Action</genre>' in output_xml
    assert '<genre id="878">Science Fiction</genre>' in output_xml
    assert '<audioType>Atmos</audioType>' in output_xml
