'''
A designer's design identity (design/design-version.md, contract 1.4): carried from its response to the queue entry,
backfilled from the build for entries made before it, and judged out of date against what the designers up now offer.
'''
import json

import numpy as np
import pytest
import soundfile as sf

from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
from pipeline.designer.convert import ContractViolation, validate_response
from pipeline.designer.http_binding import _response_from_json
from pipeline.designer.identity import CurrentIdentity, DesignIdentity, EndpointIdentity, backfill_version, \
    current_identity, identity_from_health, is_out_of_date, out_of_date_reasons
from pipeline.review import QueueEntry, _entry_from_dict, _entry_to_dict

CANDIDATE = {'filters': [{'type': 'low_shelf', 'freq_hz': 20.0, 'gain_db': 6.0, 'q': 0.7}], 'confidence': 0.9,
             'mv_adjust_db': 6.0, 'method': 'fitted'}


# --- the builds a library holds from before 1.4 ---------------------------------------------------------------------

@pytest.mark.parametrize('build, version', [
    ('unknown+src:eeb06d1cb9a6', 1),
    ('unknown+src:c1ae12bafc27', 2),
    ('beqforge_revision: unknown+src:c1ae12bafc27; run_record: /work/records/designer-d305dceefef0923a.run.json.gz', 2),
    ('unknown+src:745e6c61342a', 3),
    ('unknown+src:3f0cbd9447c7', 4),
    ('2f1a0ccefd69+src:35c4437ecbb6 (frozen build)', 4),
    ('a110332e314d+src:04ac01018d49 (frozen build)', 4),
    ('beqforge_revision: e1a2c24b1172+src:008f1c30d377 (frozen build)', 4),
    ('unknown+src:76b2edd7e060', 4),
    ('unknown+src:0123456789ab', None),   # a build not in the table: unknown, never guessed
    (None, None),
    ('', None),
])
def test_a_build_from_before_1_4_is_given_the_version_beqforge_declared_for_it(build, version):
    assert backfill_version(build) == version


def test_an_entrys_own_version_wins_over_its_build():
    assert DesignIdentity.of(7, 'abc', 'unknown+src:c1ae12bafc27') == DesignIdentity(7, 'abc', 'unknown+src:c1ae12bafc27')
    assert DesignIdentity.of(None, None, 'unknown+src:c1ae12bafc27') == \
        DesignIdentity(2, None, 'unknown+src:c1ae12bafc27', backfilled=True)
    assert DesignIdentity.of(True, '', None) == DesignIdentity(None)   # nothing valid: unknown


# --- out of date ----------------------------------------------------------------------------------------------------

DEFAULTS_NOW = CurrentIdentity(4, 'd0', {})        # every setting at its default
TUNED_NOW = CurrentIdentity(4, 'd1', {'strategy': ['flatten']})


@pytest.mark.parametrize('identity, current, out', [
    (DesignIdentity(None), DEFAULTS_NOW, True),                      # unknown
    (DesignIdentity(3, 'd0'), DEFAULTS_NOW, True),                   # older
    (DesignIdentity(5, 'd0'), DEFAULTS_NOW, True),                   # newer than anything up: not reproducible now
    (DesignIdentity(4, 'd0'), DEFAULTS_NOW, False),
    (DesignIdentity(4, 'd1'), DEFAULTS_NOW, True),                   # other settings
    (DesignIdentity(4, None), CurrentIdentity(4), False),            # neither has settings
    (DesignIdentity(4, None, backfilled=True), DEFAULTS_NOW, False),  # before 1.4, designer now at its defaults
    (DesignIdentity(4, None, backfilled=True), TUNED_NOW, True),
    (DesignIdentity(4, None, backfilled=True), CurrentIdentity(4, 'd0'), True),   # defaults not said: cannot tell
    (DesignIdentity(2, None, backfilled=True), DEFAULTS_NOW, True),
])
def test_a_design_is_out_of_date_when_a_redesign_now_could_answer_differently(identity, current, out):
    assert is_out_of_date(identity, current) is out
    assert bool(out_of_date_reasons(identity, current)) is out


def test_the_reason_names_the_versions():
    assert out_of_date_reasons(DesignIdentity(2, None, backfilled=True), DEFAULTS_NOW) == \
        ['designed with version 2; the designer now designs with 4']


# --- what the designers up now offer --------------------------------------------------------------------------------

def test_health_reports_an_identity_only_with_a_valid_version():
    assert identity_from_health('u', {'design_version': 4, 'design_settings': 'd0', 'design_settings_detail': {}}) == \
        EndpointIdentity('u', 4, 'd0', {})
    assert identity_from_health('u', {'contract_version': '1.2', 'parallel': 2}) is None   # a designer before 1.4
    assert identity_from_health('u', {'design_version': '4'}) is None
    assert identity_from_health('u', {'design_version': -1}) is None
    assert identity_from_health('u', None) is None


def test_the_current_identity_is_the_highest_version_up_and_a_disagreement_is_said():
    a = EndpointIdentity('http://a/design', 4, 'd0', {})
    b = EndpointIdentity('http://b/design', 5, 'd1', {'strategy': ['flatten']})

    agreed = current_identity([a, EndpointIdentity('http://c/design', 4, 'd0', {})])
    differ = current_identity([a, b, None])

    assert agreed == CurrentIdentity(4, 'd0', {}) and not agreed.disagreement
    assert (differ.version, differ.settings) == (5, 'd1')
    assert differ.disagreement == ("http://a/design designs with version 4, default settings",
                                   "http://b/design designs with version 5, settings {'strategy': ['flatten']}")
    assert current_identity([None, None]) is None   # no designer says: nothing can be judged out of date
    assert current_identity([EndpointIdentity('http://d/design', 4, 'd9'), b]).disagreement[0] == \
        'http://d/design designs with version 4, settings d9'   # no detail: its digest


# --- the contract fields ------------------------------------------------------------------------------------------

def test_a_1_4_response_says_its_identity_on_either_shape():
    designed = _response_from_json({'contract_version': '1.1', 'candidates': [CANDIDATE], 'design_version': 4,
                                    'design_settings': 'd0', 'designer_build': '0.2.6 (frozen build)'})
    declined = _response_from_json({'contract_version': '1.1', 'decline_reason': 'no_rolloff_detected',
                                    'design_version': 4})
    older = _response_from_json({'contract_version': '1.1', 'candidates': [CANDIDATE]})

    assert (designed.design_version, designed.design_settings, designed.designer_build) == (4, 'd0', '0.2.6 (frozen build)')
    assert declined.design_version == 4 and declined.design_settings is None
    assert (older.design_version, older.design_settings, older.designer_build) == (None, None, None)
    for response in (designed, declined, older):
        validate_response(response)


@pytest.mark.parametrize('field, value', [('design_version', -1), ('design_version', '4'), ('design_version', True),
                                          ('design_version', 4.0), ('design_settings', ''), ('designer_build', 3)])
def test_an_identity_the_contract_does_not_allow_is_refused(field, value):
    response = DesignResponse('1.1', candidates=[DesignCandidate(filters=[BiquadSpec('low_shelf', 20.0, 6.0, 0.7)],
                                                                 confidence=0.9, mv_adjust_db=6.0, method='fitted')],
                              **{field: value})
    with pytest.raises(ContractViolation, match=field):
        validate_response(response)


# --- the queue entry ------------------------------------------------------------------------------------------------

def _entry(**fields):
    return QueueEntry(id='t', fs=1000, meta={}, curve={}, **fields)


def test_an_identity_is_written_only_when_said_so_an_older_reader_never_meets_it():
    assert not {'design_version', 'design_settings'} & set(_entry_to_dict(_entry()))
    written = json.loads(json.dumps(_entry_to_dict(_entry(design_version=4, design_settings='d0', designer_build='b'))))

    back = _entry_from_dict(written)

    assert (back.design_version, back.design_settings, back.designer_build) == (4, 'd0', 'b')
    assert back.design_identity == DesignIdentity(4, 'd0', 'b')


def test_an_entry_from_before_1_4_has_its_version_from_its_build():
    entry = _entry(designer_build='e1a2c24b1172+src:008f1c30d377 (frozen build)')
    assert entry.design_identity == DesignIdentity(4, None, entry.designer_build, backfilled=True)


def test_a_field_a_later_version_writes_is_left_alone_on_reading():
    written = _entry_to_dict(_entry(design_version=4))
    written['design_from_the_future'] = {'x': 1}
    assert _entry_from_dict(written).design_version == 4


@pytest.mark.parametrize('declines', [False, True])
def test_a_designers_identity_reaches_the_queue_entry(tmp_path, declines):
    from pipeline.designer.registry import register_designer, unregister_designer
    from pipeline.orchestrate import Session
    from pipeline.review import design_and_queue, read_entry

    def designer(request):
        identity = {'design_version': 4, 'design_settings': 'd0', 'designer_build': '0.2.6+src:abc'}
        if declines:
            return DesignResponse('1.1', decline_reason='no_rolloff_detected', **identity)
        return DesignResponse('1.1', candidates=[DesignCandidate(filters=[BiquadSpec('low_shelf', 20.0, 6.0, 0.7)],
                                                                 confidence=0.9, mv_adjust_db=6.0, method='fitted',
                                                                 commentary={'beqforge_revision': 'older key'})],
                              **identity)

    wav = str(tmp_path / 'mono.wav')
    sf.write(wav, np.random.default_rng(0).normal(0, 0.05, (1000 * 5, 1)), 1000, subtype='PCM_24')
    register_designer('test.identity', designer)
    try:
        design_and_queue(Session(), 't', wav, 'test.identity', str(tmp_path / 'queue'))
    finally:
        unregister_designer('test.identity')

    entry = read_entry(str(tmp_path / 'queue'), 't')
    assert (entry.design_version, entry.design_settings) == (4, 'd0')
    assert entry.designer_build == '0.2.6+src:abc'   # the 1.4 field, not the commentary key


def test_an_index_summary_cached_before_the_identity_was_kept_is_read_again():
    ''' A miss, so the entry is read again and its identity found, rather than every title looking unknown. '''
    from dataclasses import asdict
    from pipeline.library.status import EntryFacts
    facts = EntryFacts.of(_entry(design_version=4, designer_build='b'), mtime_ns=1, size=2)
    cached = asdict(facts)

    assert EntryFacts.from_json(json.dumps(cached)) == facts
    for name in ('design_version', 'design_settings', 'designer_build'):
        cached.pop(name)
    assert EntryFacts.from_json(json.dumps(cached)) is None
