'''
How the episodes a filter covers reach the catalogue (plan §11.9): beqcatalogue's structured season
`<beq_season id><number/><episodes count>..</episodes></beq_season>`, and the legacy note it can also read.
'''
import xml.etree.ElementTree as ET

import pytest

from model.iir import CompleteFilter, PeakingEQ
from pipeline.metadata import BeqMetadata
from pipeline.publish.xml import to_beq_xml

FILTERS = CompleteFilter(fs=1000, filters=[PeakingEQ(1000, 40.0, 2.0, -3.0)])


def _meta(**kwargs):
    return BeqMetadata(title='Some Show', year='2015', audio_types=['DD 5.1'], **kwargs)


def _metadata_xml(meta):
    root = ET.fromstring(to_beq_xml(FILTERS, meta))
    return root.find('beq_metadata')


# --- the value to_dict() hands the writer -------------------------------------------------------------------------

def test_a_show_with_no_season_is_written_as_before():
    assert _meta().to_dict()['beq_season'] == ''
    assert _meta().to_dict()['beq_note'] == ''


def test_a_bare_season_is_the_plain_text():
    assert _meta(season='2').to_dict()['beq_season'] == '2'


def test_a_season_with_tmdb_details_is_structured_with_its_episodes_sorted_and_counted():
    meta = _meta(season='1', season_id='92137', season_episode_count=8, episodes=[3, 1, 2, 2])

    assert meta.to_dict()['beq_season'] == {'id': '92137', 'number': '1', 'episode_count': 8, 'episodes': '1,2,3'}
    assert meta.to_dict()['beq_note'] == ''  # the structured form needs no note


def test_a_structured_season_without_episodes_has_none_meaning_the_whole_season():
    meta = _meta(season='1', season_id='92137', season_episode_count=8)

    assert meta.to_dict()['beq_season'] == {'id': '92137', 'number': '1'}


@pytest.mark.parametrize('kwargs', [
    {'season': '1', 'season_id': '92137'},  # no episode count, so "complete" could not be judged
    {'season': '1', 'season_episode_count': 8},  # no season id
    {'season_id': '92137', 'season_episode_count': 8},  # no season
])
def test_it_is_only_structured_when_the_catalogue_can_use_it(kwargs):
    meta = _meta(episodes=[1, 2], **kwargs)

    assert not meta.structured_season
    assert not isinstance(meta.to_dict()['beq_season'], dict)


# --- the legacy note ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize('episodes, note', [([3], 'E3'), ([1, 2, 3], 'E1-3'), ([5, 4, 6], 'E4-6')])
def test_without_the_structured_season_a_contiguous_range_goes_in_the_note(episodes, note):
    meta = _meta(season='1', episodes=episodes)

    assert meta.to_dict()['beq_season'] == '1'
    assert meta.to_dict()['beq_note'] == note


def test_a_range_with_gaps_cannot_be_written_as_a_note():
    assert _meta(season='1', episodes=[1, 2, 4]).to_dict()['beq_note'] == ''


def test_an_existing_note_is_never_replaced():
    assert _meta(season='1', episodes=[3], note='Extended cut').to_dict()['beq_note'] == 'Extended cut'


def test_no_episode_note_without_a_season_or_without_episodes():
    assert _meta(episodes=[3]).to_dict()['beq_note'] == ''
    assert _meta(season='1').to_dict()['beq_note'] == ''


# --- the XML -----------------------------------------------------------------------------------------------------

def test_the_xml_carries_the_structured_season_for_a_partial_season():
    tag = _metadata_xml(_meta(season='1', season_id='92137', season_episode_count=8, episodes=[3])).find('beq_season')

    assert tag.attrib == {'id': '92137'}
    assert tag.find('number').text == '1'
    assert tag.find('episodes').attrib == {'count': '8'}
    assert tag.find('episodes').text == '3'


def test_the_xml_lists_every_episode_for_a_whole_season():
    tag = _metadata_xml(_meta(season='1', season_id='92137', season_episode_count=4,
                              episodes=[1, 2, 3, 4])).find('beq_season')

    assert tag.find('episodes').attrib == {'count': '4'}
    assert tag.find('episodes').text == '1,2,3,4'


def test_the_xml_omits_episodes_when_none_are_given():
    tag = _metadata_xml(_meta(season='1', season_id='92137', season_episode_count=8)).find('beq_season')

    assert tag.find('number').text == '1'
    assert tag.find('episodes') is None


def test_the_xml_falls_back_to_the_plain_season_and_the_note():
    metadata = _metadata_xml(_meta(season='2', episodes=[1, 2, 3]))

    assert metadata.find('beq_season').text == '2'
    assert list(metadata.find('beq_season')) == []
    assert metadata.find('beq_note').text == 'E1-3'


def test_a_film_or_a_show_without_a_season_writes_the_same_elements_as_before():
    metadata = _metadata_xml(_meta())

    assert metadata.find('beq_season').text is None or metadata.find('beq_season').text == ''
    assert [c.tag for c in metadata] .count('beq_season') == 1
