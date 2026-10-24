'''
Contract 1.6's *reviewed at* (design/designer-interface.md §3, "Reviewed designs"; design/redesign-scope.md step 2): a
person may keep an out-of-date design at the current identity, after which it is not out of date against it; nothing
else about the design changes, and a redesign or an edit starts afresh.
'''
import json

import pytest

from pipeline.designer.identity import CurrentIdentity, DesignIdentity, is_out_of_date, reviewed_identities
from pipeline.library.reviewed import ReviewRefused, mark_reviewed, review_refused
from pipeline.review import CandidateSummary, QueueEntry, _entry_from_dict, _entry_to_dict, read_entry, \
    write_queue_entry

V7 = CurrentIdentity(7, 's0')
CANDIDATE = CandidateSummary(filters=[{'_type': 'LowShelf', 'fs': 1000, 'fc': 20.0, 'gain': 6.0, 'q': 0.7}],
                             confidence=0.9, method='fitted', mv_adjust_db=6.0)


def _entry(**fields):
    defaults = dict(id='t', fs=1000, meta={}, curve={'x': [1.0]}, candidates=[CANDIDATE], design_version=6,
                    design_settings='s0', designer_build='b', variant_id='v1')
    return QueueEntry(**{**defaults, **fields})


# --- the out-of-date rule ------------------------------------------------------------------------------------------

@pytest.mark.parametrize('reviewed, current, out', [
    ((), V7, True),
    (((7, 's0'),), V7, False),                      # kept at the identity a redesign now would use
    (((7, 's1'),), V7, True),                       # kept at other settings
    (((7, 's0'),), CurrentIdentity(8, 's0'), True),  # a newer version since
    (((6, 's9'), (7, 's0')), V7, False),            # reviewed at several over time: any of them
    (((7, None),), CurrentIdentity(7, None), False),
])
def test_a_design_is_not_out_of_date_against_an_identity_it_was_reviewed_at(reviewed, current, out):
    assert is_out_of_date(DesignIdentity(6, 's0', reviewed=reviewed), current) is out


def test_a_design_from_before_versioning_kept_at_the_current_identity_is_current():
    unknown = DesignIdentity.of(None, None, 'a build nobody knows', [{'design_version': 7, 'design_settings': 's0'}])
    assert unknown.version is None and not is_out_of_date(unknown, V7)


@pytest.mark.parametrize('stamps, kept', [
    (None, ()),
    ([{'design_version': 7, 'design_settings': 's0', 'at': 'x', 'by': 'me'}], ((7, 's0'),)),
    ([[7, 's0']], ((7, 's0'),)),
    ([{'design_version': -1}, {'design_version': True}, {'design_settings': 's'}, 'junk', [1, 2, 3]], ()),
    ([{'design_version': 7, 'design_settings': ''}], ((7, None),)),
])
def test_only_stamps_that_name_a_valid_version_count(stamps, kept):
    assert reviewed_identities(stamps) == kept


# --- the entry --------------------------------------------------------------------------------------------------------

def test_reviewed_at_is_written_only_when_there_is_one_so_an_older_reader_never_meets_it():
    assert 'reviewed_at' not in _entry_to_dict(_entry())
    stamp = {'design_version': 7, 'design_settings': 's0', 'at': '2026-10-25T00:00:00+00:00', 'by': 'matt'}
    back = _entry_from_dict(json.loads(json.dumps(_entry_to_dict(_entry(reviewed_at=[stamp])))))
    assert back.reviewed_at == [stamp] and back.design_identity.reviewed == ((7, 's0'),)


def test_keeping_a_design_stamps_it_and_changes_nothing_else(tmp_path):
    queue = str(tmp_path)
    before = _entry(rejected=[CANDIDATE], run_record='records/r-v6.run.json.gz', reviewer_note='hmm')
    write_queue_entry(queue, before)

    kept = mark_reviewed(queue, 't', V7, by='matt', now=lambda: '2026-10-25T00:00:00+00:00')

    assert kept.reviewed_at == [{'design_version': 7, 'design_settings': 's0', 'at': '2026-10-25T00:00:00+00:00',
                                 'by': 'matt'}]
    again = read_entry(queue, 't')
    assert again.reviewed_at == kept.reviewed_at and not is_out_of_date(again.design_identity, V7)
    for name in ('candidates', 'rejected', 'design_version', 'design_settings', 'designer_build', 'run_record',
                 'status', 'reviewer_note', 'variant_id'):
        assert getattr(again, name) == getattr(before, name), name


def test_a_second_review_at_a_later_identity_is_added_to_the_first(tmp_path):
    queue = str(tmp_path)
    write_queue_entry(queue, _entry())
    mark_reviewed(queue, 't', V7)
    later = mark_reviewed(queue, 't', CurrentIdentity(8, 's0'))
    assert [s['design_version'] for s in later.reviewed_at] == [7, 8]


@pytest.mark.parametrize('entry, current, says', [
    (_entry(), None, 'no designer'),
    (_entry(), CurrentIdentity(7, 's0', disagreement=('a designs with 7', 'b designs with 6')), 'would design differently'),
    (_entry(design_version=7), V7, 'not out of date'),
    (_entry(reviewed_at=[{'design_version': 7, 'design_settings': 's0'}]), V7, 'not out of date'),
    (_entry(curve={}), V7, 'no design'),
])
def test_a_design_that_cannot_be_kept_says_why(tmp_path, entry, current, says):
    assert says in review_refused(entry, current)
    write_queue_entry(str(tmp_path), entry)
    with pytest.raises(ReviewRefused, match=says):
        mark_reviewed(str(tmp_path), 't', current)
    assert read_entry(str(tmp_path), 't').reviewed_at == entry.reviewed_at


def test_an_edit_starts_with_no_review_and_the_reviewed_run_keeps_its_own(tmp_path):
    from pipeline.designer.contract import BiquadSpec
    from pipeline.library.variants import save_edited, use_variant
    queue = str(tmp_path)
    write_queue_entry(queue, _entry(reviewed_at=[{'design_version': 7, 'design_settings': 's0'}]))
    edited = save_edited(queue, 't', 0, [BiquadSpec('low_shelf', 20.0, 3.0, 0.7)])
    assert edited.reviewed_at == [] and edited.variants[0]['reviewed_at'] == [{'design_version': 7,
                                                                              'design_settings': 's0'}]
    assert use_variant(queue, 't', 'v1').design_identity.reviewed == ((7, 's0'),)


def test_the_index_summary_carries_the_reviews_so_the_out_of_date_filter_follows_them():
    from pipeline.library.index import _design_identity
    from pipeline.library.status import EntryFacts
    facts = EntryFacts.of(_entry(reviewed_at=[{'design_version': 7, 'design_settings': 's0'}],
                                 run_record='records/r-v6.run.json.gz'), mtime_ns=1, size=2)
    text = facts.to_json()
    assert EntryFacts.from_json(text) == facts and facts.run_record == 'records/r-v6.run.json.gz'
    identity = _design_identity(text)
    assert identity.reviewed == ((7, 's0'),) and not is_out_of_date(identity, V7)


def test_a_summary_cached_before_reviews_were_kept_reads_as_none():
    from dataclasses import asdict
    from pipeline.library.status import EntryFacts
    cached = asdict(EntryFacts.of(_entry(), mtime_ns=1, size=2))
    for name in ('reviewed', 'run_record'):
        cached.pop(name)
    facts = EntryFacts.from_json(json.dumps(cached))
    assert facts.reviewed == () and facts.run_record is None
