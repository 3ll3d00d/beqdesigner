'''
The needs/tier table of design/library-sync/workflow-rework §12.6 as a pure function: one case per row, in table order,
and the flags that take a title out of the work. No disk, no index -- derive_needs() is a function of StageStates.
'''
import pytest

from pipeline.library.state import COMMIT_STATES, DESIGN_STATES, EXTRACT_STATES, NEEDS, PUBLISH_STATES, REVIEW_STATES, \
    TIER_OF_NEEDS, StageStates, derive_needs

# an accepted title whose files are written and pushed: the resting state everything else is a departure from
DONE = dict(extract='current', design='protected', review='accepted', publish='written', commit='pushed')


def _needs(**states):
    result = derive_needs(StageStates(**states))
    assert result.tier == TIER_OF_NEEDS[result.needs]
    return result.needs, result.detail


def test_every_needs_value_has_a_tier_and_the_vocabularies_are_the_documented_ones():
    assert NEEDS == ('attention', 'review', 'extract', 'design', 'publish', 'commit', 'done')
    assert set(TIER_OF_NEEDS.values()) == {'attention', 'human', 'machine', 'done'}
    assert EXTRACT_STATES == ('none', 'current', 'stale', 'failed')
    assert DESIGN_STATES == ('none', 'current', 'stale', 'failed', 'protected')
    assert REVIEW_STATES == ('none', 'pending', 'accepted', 'skipped', 'rejected')
    assert PUBLISH_STATES == ('none', 'not_written', 'written', 'out_of_date')
    assert COMMIT_STATES == ('none', 'uncommitted', 'committed', 'pushed', 'unknown')


# --- attention -----------------------------------------------------------------------------------------------------

def test_a_failed_extract_needs_attention_with_the_reason():
    assert _needs(extract='failed', failure='ValueError: no such file') == \
           ('attention', 'extract failed: ValueError: no such file')


def test_a_failed_design_needs_attention_with_the_reason():
    assert _needs(extract='current', design='failed', failure='HTTPError: 500') == \
           ('attention', 'design failed: HTTPError: 500')


def test_a_project_conflict_needs_attention():
    needs, detail = _needs(**{**DONE, 'commit': 'uncommitted'}, project_conflict=True)

    assert needs == 'attention' and 'disagree' in detail


def test_a_source_changed_since_accepted_needs_attention_whether_or_not_it_was_published():
    assert _needs(extract='current', design='protected', review='accepted', publish='not_written',
                  source_changed=True) == ('attention', 'source changed since accepted')
    assert _needs(**DONE, source_changed=True) == ('attention', 'source changed since published')


def test_attention_beats_every_row_below_it():
    assert _needs(extract='failed', review='pending', design='current')[0] == 'attention'


# --- human -----------------------------------------------------------------------------------------------------------

def test_a_current_design_awaiting_review_needs_review_with_its_confidence_and_candidates():
    assert _needs(extract='current', design='current', review='pending', confidence=0.62, candidates=3) == \
           ('review', 'conf 0.62 - 3 candidates')


def test_a_designer_decline_is_still_a_review():
    assert _needs(extract='current', design='current', review='pending', decline='no signal below 20 Hz') == \
           ('review', 'designer declined: no signal below 20 Hz')


def test_incomplete_metadata_on_a_pending_title_is_named_in_its_review():
    needs, detail = _needs(extract='current', design='current', review='pending', confidence=0.9, candidates=1,
                           metadata_problems=('year is required',))

    assert needs == 'review' and detail.endswith('metadata incomplete: year is required')


def test_an_accepted_title_with_incomplete_metadata_goes_back_to_a_human():
    assert _needs(extract='current', design='protected', review='accepted', publish='not_written',
                  metadata_problems=('title is required', 'year is required')) == \
           ('review', 'metadata incomplete: title is required; year is required')


# --- machine ---------------------------------------------------------------------------------------------------------

def test_a_new_title_needs_extract():
    assert _needs() == ('extract', 'new')


def test_a_stale_extract_needs_extract_again():
    assert _needs(extract='stale', design='stale', review='pending') == ('extract', 'source or settings changed')


def test_a_current_extract_with_no_design_needs_design():
    assert _needs(extract='current') == ('design', 'new')


def test_a_stale_design_needs_design_again():
    assert _needs(extract='current', design='stale', review='pending') == ('design', 'source or settings changed')


def test_an_accepted_title_not_yet_written_needs_publish():
    assert _needs(extract='current', design='protected', review='accepted', publish='not_written')[0] == 'publish'


def test_a_published_title_whose_inputs_changed_needs_publish_again():
    assert _needs(**{**DONE, 'publish': 'out_of_date'}, out_of_date='changed since it was published') == \
           ('publish', 'changed since it was published')


@pytest.mark.parametrize('commit, detail', [('uncommitted', 'written, not committed'),
                                            ('committed', 'committed, not pushed'),
                                            ('unknown', 'cannot tell whether it is committed and pushed')])
def test_a_written_title_needs_commit_until_it_is_pushed(commit, detail):
    assert _needs(**{**DONE, 'commit': commit}) == ('commit', detail)


# --- done ------------------------------------------------------------------------------------------------------------

def test_a_pushed_title_is_done():
    assert _needs(**DONE) == ('done', 'pushed')


@pytest.mark.parametrize('review', ['skipped', 'rejected'])
def test_a_skipped_or_rejected_title_is_done(review):
    assert _needs(extract='current', design='current', review=review) == ('done', review)


@pytest.mark.parametrize('extract, design', [('stale', 'stale'), ('none', 'none')])
def test_a_decided_title_is_not_made_stale_by_a_settings_change(extract, design):
    ''' Only a source change re-enters a done title (§12.6): extract/design staleness counts for titles in play. '''
    assert _needs(**{**DONE, 'extract': extract, 'design': design})[0] == 'done'
    assert _needs(extract=extract, design=design, review='skipped')[0] == 'done'


# --- flags -----------------------------------------------------------------------------------------------------------

def test_an_ignored_title_is_done_whatever_its_stages_say():
    assert _needs(ignored='ignored by rule: kind tv', extract='failed', review='pending', design='current') == \
           ('done', 'ignored by rule: kind tv')


def test_a_shadowed_title_is_done_and_names_its_owner():
    needs, detail = _needs(shadowed_by='jriver-abc-1', extract='none')

    assert needs == 'done' and 'jriver-abc-1' in detail


def test_a_title_gone_from_its_source_is_done():
    assert _needs(gone=True, review='pending', extract='current', design='current') == ('done', 'gone from source')
