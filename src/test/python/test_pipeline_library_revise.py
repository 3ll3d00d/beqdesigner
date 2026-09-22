'''
Sending a title back into the work list (design/library-sync/workflow-rework §12.8): reopen for review, redesign,
re-extract -- from each starting state, against real temp git repos where the catalogue files matter.
'''
import json
import os
import subprocess

import pytest

from model.iir import CompleteFilter, LowShelf
from pipeline.config import AnalysisConfig
from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.library.design_cache import design_if_needed
from pipeline.library.extract_cache import extract_if_needed, invalidate_extract, read_source_channel_count
from pipeline.library.revise import REVISE_TARGETS, redesign_entry, reopen_entry, revise_entry
from pipeline.library.season import invalidate_season_track
from pipeline.library.source import LibraryItem
from pipeline.library.sync import commit_library, publish_library
from pipeline.orchestrate import Session
from pipeline.publish.git import RepoTarget, repo_state
from pipeline.publish.project import read_project_filter
from pipeline.review import design_and_queue, read_entry, update_entry
from test_pipeline_library_commit import OWNER, IMAGES_NAME, _commits, _on_remote, _publish, _queue_entry, _repo, _track
from test_pipeline_publish_project import _HUMAN_FILTER, _hand_edit_filter, _write_mono_wav

DESIGNER = 'test.revise'


@pytest.fixture
def repos(tmp_path):
    xml, xml_bare = _repo(tmp_path, 'xml')
    images, images_bare = _repo(tmp_path, 'images')
    return xml, xml_bare, images, images_bare


@pytest.fixture
def designer():
    register_designer(DESIGNER, lambda request: DesignResponse(contract_version='1.0', candidates=[
        DesignCandidate(filters=[BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7)], confidence=0.9,
                        mv_adjust_db=4.0, gain_reduction_db=-1.5, method='fitted', commentary={})]))
    yield
    unregister_designer(DESIGNER)


def _where(repos):
    xml, _, images, _ = repos
    return dict(xml_repo=xml, images_repo=images, xml_dir='xml', image_dir='img')


def _published_and_committed(tmp_path, repos):
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'))
    commit_library(queue_dir, repos[0], images_repo=repos[2], xml_dir='xml', image_dir='img')
    _track(repos[0])
    _track(repos[2])
    return queue_dir


# --- reopen: from every starting state -----------------------------------------------------------------------

def test_reopening_an_accepted_entry_puts_it_back_to_pending_with_a_note(tmp_path):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'one', 'Heat')

    result = reopen_entry(queue_dir, 'one', 'wrong poster')

    entry = read_entry(queue_dir, 'one')
    assert result.entry == entry
    assert (entry.status, entry.chosen_candidate_index, entry.revision) == ('pending', None, 0)
    assert entry.reviewer_note == 'Reopened for review: wrong poster'
    assert result.reverted == []


def test_reopening_keeps_the_candidates_metadata_and_artwork_and_appends_to_an_existing_note(tmp_path):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'one', 'Heat')
    before = update_entry(queue_dir, 'one', art_path='/art/poster.jpg', art_overridden=True, reviewer_note='first look')

    reopen_entry(queue_dir, 'one')

    after = read_entry(queue_dir, 'one')
    assert (after.candidates, after.meta, after.curve) == (before.candidates, before.meta, before.curve)
    assert (after.art_path, after.art_overridden) == ('/art/poster.jpg', True)
    assert after.reviewer_note == 'first look\nReopened for review'


@pytest.mark.parametrize('status', ['skipped', 'rejected'])
def test_a_skipped_or_rejected_entry_can_be_reopened(tmp_path, status):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'one', 'Heat', status='pending')
    update_entry(queue_dir, 'one', status=status)

    reopen_entry(queue_dir, 'one')

    assert read_entry(queue_dir, 'one').status == 'pending'


def test_an_entry_that_is_already_pending_or_missing_is_refused(tmp_path):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'one', 'Heat', status='pending')

    with pytest.raises(ValueError, match='already pending'):
        reopen_entry(queue_dir, 'one')
    with pytest.raises(FileNotFoundError):
        reopen_entry(queue_dir, 'nothing-here')


def test_a_published_entry_cannot_be_reopened_without_saying_where_its_files_are(tmp_path, repos):
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'))

    with pytest.raises(ValueError, match='xml_repo'):
        reopen_entry(queue_dir, 'one')

    assert read_entry(queue_dir, 'one').status == 'published'  # untouched
    assert (tmp_path / 'xml' / 'xml' / 'one.json').exists()


# --- reopen a published entry: written but not committed ---------------------------------------------------

def test_reopening_a_published_but_uncommitted_entry_takes_its_files_back_out_of_the_repos(tmp_path, repos):
    xml, _, images, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'))
    assert repo_state(xml).uncommitted == {'xml/one.json', 'xml/database.json'}

    result = reopen_entry(queue_dir, 'one', **_where(repos))

    assert sorted(result.reverted) == ['img/one.png', 'xml/one.json']
    assert not (tmp_path / 'xml' / 'xml' / 'one.json').exists()
    assert not (tmp_path / 'images' / 'img' / 'one.png').exists()
    assert repo_state(xml).uncommitted == {'xml/database.json'} and repo_state(images).uncommitted == frozenset()
    entry = read_entry(queue_dir, 'one')
    assert (entry.status, entry.revision) == ('pending', 0)  # nothing was ever in the catalogue
    assert (entry.published_digest, entry.published_at) == (None, None)


def test_reopening_removes_a_file_that_was_staged_but_not_committed(tmp_path, repos):
    xml, _, _, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'), with_images=False)
    subprocess.run(['git', '-C', xml.local_path, 'add', 'xml/one.json'], check=True)

    reopen_entry(queue_dir, 'one', xml_repo=xml, xml_dir='xml')

    assert not (tmp_path / 'xml' / 'xml' / 'one.json').exists()
    assert repo_state(xml).uncommitted == {'xml/database.json'}


def test_reopening_leaves_other_uncommitted_files_in_the_repos_alone(tmp_path, repos):
    xml, _, _, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'), ('two', 'Alien'), with_images=False)
    (tmp_path / 'xml' / 'notes.txt').write_text('mine')

    reopen_entry(queue_dir, 'one', xml_repo=xml, xml_dir='xml')

    assert repo_state(xml).uncommitted == {'xml/two.json', 'xml/database.json', 'notes.txt'}


# --- reopen a published entry: committed and pushed = a revision -----------------------------------------------

def test_reopening_a_committed_entry_leaves_the_catalogue_alone_and_starts_a_revision(tmp_path, repos):
    xml, xml_bare, images, _ = repos
    queue_dir = _published_and_committed(tmp_path, repos)

    result = reopen_entry(queue_dir, 'one', 'edition was wrong', **_where(repos))

    assert result.reverted == []
    assert (tmp_path / 'xml' / 'xml' / 'one.json').exists() and (tmp_path / 'images' / 'img' / 'one.png').exists()
    assert len(_commits(xml)) == 1
    entry = read_entry(queue_dir, 'one')
    assert (entry.status, entry.revision, entry.published_digest) == ('pending', 1, None)


def test_a_reopened_pushed_entry_publishes_to_the_same_path_as_a_new_commit(tmp_path, repos):
    xml, xml_bare, images, images_bare = repos
    queue_dir = _published_and_committed(tmp_path, repos)
    first_commit = _commits(xml)[0]
    reopen_entry(queue_dir, 'one', **_where(repos))
    update_entry(queue_dir, 'one', status='accepted', chosen_candidate_index=0,
                 meta={'title': 'Heat (1995)', 'year': '1995', 'audio_types': ['Atmos']})

    results = publish_library(str(queue_dir), xml, images_repo=images, xml_dir='xml', image_dir='img',
                              image_owner=OWNER, image_repo_name=IMAGES_NAME)
    committed = commit_library(queue_dir, xml, images_repo=images, xml_dir='xml', image_dir='img')

    assert [r['id'] for r in results] == ['one']
    assert committed.xml.paths == ['xml/one.json', 'xml/database.json'] and committed.xml.commit not in (None, first_commit)
    assert b'Heat (1995)' in _on_remote(xml_bare, 'xml/one.json')
    assert _on_remote(xml_bare, 'xml/two.json') == b''  # one title, one path, however many revisions
    assert read_entry(queue_dir, 'one').revision == 1


def test_reopening_a_revision_that_is_written_but_not_committed_restores_the_committed_version(tmp_path, repos):
    xml, xml_bare, images, _ = repos
    queue_dir = _published_and_committed(tmp_path, repos)
    committed_xml = (tmp_path / 'xml' / 'xml' / 'one.json').read_text()
    reopen_entry(queue_dir, 'one', **_where(repos))
    update_entry(queue_dir, 'one', status='accepted', chosen_candidate_index=0,
                 meta={'title': 'Heat (1995)', 'year': '1995', 'audio_types': ['Atmos']})
    publish_library(queue_dir, xml, images_repo=images, xml_dir='xml', image_dir='img', image_owner=OWNER,
                    image_repo_name=IMAGES_NAME)
    assert (tmp_path / 'xml' / 'xml' / 'one.json').read_text() != committed_xml

    result = reopen_entry(queue_dir, 'one', 'changed my mind', **_where(repos))

    assert 'xml/one.json' in result.reverted
    assert (tmp_path / 'xml' / 'xml' / 'one.json').read_text() == committed_xml  # the catalogue's own version again
    assert repo_state(xml).uncommitted == {'xml/database.json'}
    assert read_entry(queue_dir, 'one').revision == 1  # already counted when the revision began; not twice


# --- redesign -------------------------------------------------------------------------------------------------

def test_redesigning_drops_protection_and_marks_the_design_stale(tmp_path):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'one', 'Heat')
    update_entry(queue_dir, 'one', design_fingerprint='current')

    redesign_entry(queue_dir, 'one', 'new designer')

    entry = read_entry(queue_dir, 'one')
    assert (entry.status, entry.chosen_candidate_index, entry.design_fingerprint) == ('pending', None, None)
    assert entry.reviewer_note == 'Sent back for redesign: new designer'


def test_a_redesign_actually_happens_on_the_next_design_pass_and_keeps_what_a_person_set(tmp_path, monkeypatch):
    from dataclasses import replace
    from pipeline.review import QueueEntry, write_queue_entry
    calls = []

    def fake_design(session, entry_id, wav_path, designer, queue_dir, **kwargs):
        calls.append(entry_id)
        entry = QueueEntry(id=entry_id, fs=1000, meta=kwargs.get('meta') or {}, curve={})
        write_queue_entry(queue_dir, entry)
        return entry

    monkeypatch.setattr('pipeline.library.design_cache.design_and_queue', fake_design)
    monkeypatch.setattr('pipeline.library.design_cache.resolve_art', lambda *args, **kwargs: None)
    source = tmp_path / 'source.wav'
    source.write_bytes(b'x')
    item = LibraryItem(id='one', source_path=str(source), display_name='Heat', fingerprint='f1')
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'one', 'Heat')
    first = design_if_needed(None, item, '/w/mono.wav', 'd', queue_dir, AnalysisConfig())
    assert first.designed is False and first.protected is True  # accepted: never redesigned, even with force
    update_entry(queue_dir, 'one', design_fingerprint=first.entry.design_fingerprint, revision=2)

    redesign_entry(queue_dir, 'one', 'better designer')
    second = design_if_needed(None, item, '/w/mono.wav', 'd', queue_dir, AnalysisConfig())

    assert (calls, second.designed) == (['one'], True)
    assert second.entry.revision == 2  # a redesign is still the same catalogue title
    assert second.entry.reviewer_note == 'Sent back for redesign: better designer'
    assert second.entry.meta['title'] == 'Heat'


def test_a_hand_edited_project_survives_a_redesign_of_an_accepted_entry(tmp_path, designer, monkeypatch):
    monkeypatch.setattr('pipeline.library.design_cache.resolve_art', lambda *args, **kwargs: None)
    session = Session(AnalysisConfig())
    wav = str(tmp_path / 'mono.wav')
    _write_mono_wav(wav)
    project_dir = str(tmp_path / 'work' / 'one')
    os.makedirs(project_dir)
    queue_dir = str(tmp_path / 'queue')
    design_and_queue(session, 'one', wav, DESIGNER, queue_dir, meta={'title': 'Heat'}, project_dir=project_dir)
    project = os.path.join(project_dir, 'one.mono.beq')
    _hand_edit_filter(project, _HUMAN_FILTER)
    update_entry(queue_dir, 'one', status='accepted', chosen_candidate_index=0)
    item = LibraryItem(id='one', source_path=wav, display_name='Heat', fingerprint='f1')

    redesign_entry(queue_dir, 'one')
    result = design_if_needed(session, item, wav, DESIGNER, queue_dir, AnalysisConfig(), project_dir=project_dir)

    assert result.designed is True and result.project_edit_preserved is True
    assert read_project_filter(project)[0].to_json() == _HUMAN_FILTER.to_json()
    assert read_entry(queue_dir, 'one').status == 'pending'  # back to a person to decide, on the new candidates


# --- re-extract -----------------------------------------------------------------------------------------------

def _manifest(directory, **entries):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, 'manifest.json'), 'w') as f:
        json.dump(entries, f)


def _read_manifest(directory):
    with open(os.path.join(directory, 'manifest.json')) as f:
        return json.load(f)


def test_invalidating_an_extract_forgets_the_audio_but_keeps_what_describes_the_source(tmp_path):
    directory = str(tmp_path / 'one')
    _manifest(directory, mono_source_fingerprint='f', mono_params_hash='p', mono_extracted_at=1.0,
              multichannel_source_fingerprint='f', multichannel_params_hash='p', source_channel_count=6,
              channel_layout_name='5.1')

    assert invalidate_extract(directory) is True

    assert _read_manifest(directory) == {'source_channel_count': 6, 'channel_layout_name': '5.1'}
    assert read_source_channel_count(directory) == 6
    assert invalidate_extract(directory) is False  # nothing left to forget
    assert invalidate_extract(str(tmp_path / 'never-extracted')) is False


def test_an_invalidated_extract_runs_ffmpeg_again_even_though_nothing_changed(tmp_path):
    from test_pipeline_library_extract_cache import _mono_item, _write_synthetic_wav
    source = str(tmp_path / 'source.wav')
    _write_synthetic_wav(source)
    directory = str(tmp_path / 'work')
    session, item = Session(AnalysisConfig()), _mono_item(source)
    extract_if_needed(session, item, directory, AnalysisConfig(), mono_mix=True)
    assert extract_if_needed(session, item, directory, AnalysisConfig(), mono_mix=True)[1] is True

    invalidate_extract(directory)

    assert extract_if_needed(session, item, directory, AnalysisConfig(), mono_mix=True)[1] is False


def test_revising_to_extract_sends_the_entry_back_through_design_and_forgets_the_extraction(tmp_path):
    queue_dir, work_dir = str(tmp_path / 'queue'), str(tmp_path / 'work')
    _queue_entry(queue_dir, 'one', 'Heat')
    update_entry(queue_dir, 'one', design_fingerprint='current')
    _manifest(os.path.join(work_dir, 'one'), mono_source_fingerprint='f', mono_params_hash='p')
    _manifest(os.path.join(work_dir, 'other'), mono_source_fingerprint='f', mono_params_hash='p')

    result = revise_entry(queue_dir, 'one', 'extract', 'try the commentary track', work_dir=work_dir)

    assert result.extract_invalidated is True
    entry = read_entry(queue_dir, 'one')
    assert (entry.status, entry.design_fingerprint) == ('pending', None)
    assert entry.reviewer_note == 'Sent back for re-extraction: try the commentary track'
    assert _read_manifest(os.path.join(work_dir, 'one')) == {}
    assert _read_manifest(os.path.join(work_dir, 'other')) != {}  # only this title's


def test_re_extracting_a_season_forgets_its_joined_track_too(tmp_path):
    queue_dir, work_dir = str(tmp_path / 'queue'), str(tmp_path / 'work')
    _queue_entry(queue_dir, 'show-s01-abc123', 'Show')
    season_dir = tmp_path / 'work' / 'show-s01-abc123'
    season_dir.mkdir(parents=True)
    (season_dir / 'season_track.json').write_text('{"fingerprint": "f"}')

    result = revise_entry(queue_dir, 'show-s01-abc123', 'extract', work_dir=work_dir)

    assert result.extract_invalidated is True and not (season_dir / 'season_track.json').exists()
    assert invalidate_season_track(str(season_dir)) is False


def test_re_extracting_needs_a_work_dir_and_a_published_entry_needs_its_repos_before_anything_is_forgotten(
        tmp_path, repos):
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'))
    work_dir = str(tmp_path / 'work')
    _manifest(os.path.join(work_dir, 'one'), mono_source_fingerprint='f')

    with pytest.raises(ValueError, match='work_dir'):
        revise_entry(queue_dir, 'one', 'extract')
    with pytest.raises(ValueError, match='xml_repo'):
        revise_entry(queue_dir, 'one', 'extract', work_dir=work_dir)

    assert _read_manifest(os.path.join(work_dir, 'one')) == {'mono_source_fingerprint': 'f'}
    assert read_entry(queue_dir, 'one').status == 'published'


def test_revise_entry_dispatches_on_the_target_and_rejects_an_unknown_one(tmp_path):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'one', 'Heat')
    update_entry(queue_dir, 'one', design_fingerprint='current')

    revise_entry(queue_dir, 'one', 'review')
    assert read_entry(queue_dir, 'one').design_fingerprint == 'current'  # review does not touch the design
    with pytest.raises(ValueError, match='to must be one of'):
        revise_entry(queue_dir, 'one', 'everything')
    assert REVISE_TARGETS == ('review', 'design', 'extract')
