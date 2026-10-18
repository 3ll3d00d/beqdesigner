'''
A design's fit (designer contract 1.3): carried from the designer's response into the review queue, written only when
measured (so an older reader of the queue never meets it otherwise), read tolerantly, and said in one line.
'''
import hashlib
import json
from dataclasses import asdict, replace

import pytest

from pipeline.review import CandidateSummary, QueueEntry, _entry_from_dict, _entry_to_dict, fit_text, has_measured_fit

FIT = dict(residual_db=1.1, residual_band_hz=(5.0, 212.0), residual_at_hz=9.0, residual_direction='under',
           residual_rms_db=0.3, residual_within_tolerance=0.96, residual_tolerance_db=0.5)


def _summary(**fields):
    return CandidateSummary(filters={'filters': []}, confidence=0.9, method='fitted', mv_adjust_db=3.0, **fields)


@pytest.mark.parametrize('fields, said', [
    (FIT, '0.3 dB RMS over 5-212 Hz; 96% of the boost within ±0.5 dB; worst 1.1 dB short at 9 Hz'),
    ({**FIT, 'residual_direction': 'over', 'residual_at_hz': 17.25}, '0.3 dB RMS over 5-212 Hz; 96% of the boost within '
                                                                     '±0.5 dB; worst 1.1 dB too much at 17.25 Hz'),
    ({**FIT, 'residual_rms_db': None, 'residual_within_tolerance': None, 'residual_tolerance_db': None},
     'worst 1.1 dB short at 9 Hz over 5-212 Hz'),
    (dict(residual_db=0.4, residual_band_hz=(5.0, 200.0)), 'worst 0.4 dB over 5-200 Hz'),   # before 1.3: residual only
    ({}, ''),
])
def test_the_fit_is_said_in_one_line(fields, said):
    assert fit_text(_summary(**fields)) == said


def test_a_measured_fit_is_kept_in_the_queue_and_read_back():
    entry = QueueEntry(id='t', fs=1000, meta={}, curve={}, candidates=[_summary(**FIT)])

    back = _entry_from_dict(json.loads(json.dumps(_entry_to_dict(entry))))

    assert asdict(back.candidates[0]) == {**asdict(entry.candidates[0]), 'residual_band_hz': [5.0, 212.0]}   # JSON list
    assert has_measured_fit(back.candidates[0])


def test_an_unmeasured_fit_is_not_written_so_an_older_reader_never_meets_it():
    written = _entry_to_dict(QueueEntry(id='t', fs=1000, meta={}, curve={}, candidates=[_summary(residual_db=0.4)]))
    assert not {'residual_at_hz', 'residual_direction', 'residual_rms_db', 'residual_within_tolerance',
                'residual_tolerance_db'} & set(written['candidates'][0])


def test_a_field_written_by_a_later_version_is_ignored_on_reading():
    written = _entry_to_dict(QueueEntry(id='t', fs=1000, meta={}, curve={}, candidates=[_summary(**FIT)]))
    written['candidates'][0]['residual_from_the_future'] = 3

    assert _entry_from_dict(written).candidates[0].residual_rms_db == 0.3


def test_the_offered_digest_of_an_entry_without_a_fit_is_what_it_was_before_1_3():
    from pipeline.library.decide import offered_digest
    entry = QueueEntry(id='t', fs=1000, meta={}, curve={}, candidates=[_summary(residual_db=0.4, residual_band_hz=(5.0, 200.0))])
    before_1_3 = {k: v for k, v in asdict(entry.candidates[0]).items()
                  if k not in ('residual_at_hz', 'residual_direction', 'residual_rms_db', 'residual_within_tolerance',
                               'residual_tolerance_db')}
    content = {'fs': 1000, 'offered': [before_1_3]}
    assert offered_digest(entry) == hashlib.sha256(
        json.dumps(content, sort_keys=True, default=list).encode('utf-8')).hexdigest()[:32]
    assert offered_digest(replace(entry, candidates=[_summary(**FIT)])) != offered_digest(entry)


def test_a_designers_fit_reaches_the_queue_entry(tmp_path):
    import numpy as np
    from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
    from pipeline.orchestrate import Applied
    from pipeline.review import _outcome_to_entry
    from pipeline.orchestrate import _design_of, fit_of
    from model.iir import CompleteFilter

    candidate = DesignCandidate(filters=[BiquadSpec('low_shelf', 20.0, 6.0, 0.7)], confidence=0.9, mv_adjust_db=6.0,
                                method='fitted', **FIT)
    design = _design_of(candidate, CompleteFilter(fs=1000))
    applied = Applied(filters=CompleteFilter(fs=1000), confidence=0.9, method='fitted', mv_adjust_db=6.0,
                      residual_db=1.1, residual_band_hz=(5.0, 212.0), alternatives=(design,), **fit_of(candidate))

    entry = _outcome_to_entry('t', 1000, {}, {}, applied)

    assert [c.residual_rms_db for c in entry.candidates] == [0.3, 0.3]
    assert entry.candidates[1].residual_direction == 'under'


def test_the_service_says_the_fit_and_drops_the_designers_repeat_of_it():
    from pipeline.service.api import _without_repeated_fit
    measured = _summary(**FIT, commentary={'fit': 'matches its target to 0.3 dB RMS', 'correction': 'restores 9 dB'})
    unmeasured = _summary(residual_db=0.4, commentary={'fit': 'a designer note'})

    assert _without_repeated_fit(measured) == {'correction': 'restores 9 dB'}
    assert _without_repeated_fit(unmeasured) == {'fit': 'a designer note'}   # nothing measured: its note is all there is


def test_the_title_page_shows_the_fit_first_without_the_designers_repeat_of_it():
    from model.worklist_title_text import candidate_commentary_html
    measured = _summary(**FIT, commentary={'fit': 'matches its target to 0.3 dB RMS', 'correction': 'restores 9 dB'})

    html = candidate_commentary_html(measured)

    assert html.startswith('<p style="margin-bottom:2px"><b>Fit</b></p>')
    assert 'worst 1.1 dB short at 9 Hz' in html and 'restores 9 dB' in html
    assert 'matches its target' not in html
    assert '<b>Fit</b>' not in candidate_commentary_html(_summary(commentary={'note': 'x'}))
