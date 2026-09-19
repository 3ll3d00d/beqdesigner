'''
current_publish_digest() is publish_reviewed_queue()'s digest computation factored out so discovery can ask "would
publishing change the catalogue?" without publishing (design/library-sync/workflow-rework §12.5/§12.6). If it ever
drifted from what publish records, every published title would look out of date -- so each case publishes for real
and compares.
'''
import os

import pytest

from model.iir import CompleteFilter, PeakingEQ
from pipeline.config import AnalysisConfig
from pipeline.orchestrate import Session
from pipeline.publish.project import ProjectFilterConflict
from pipeline.review import current_publish_digest, design_and_queue, publish_reviewed_queue, read_entry, update_entry
from test_pipeline_review import DESIGNER_NAME, _designers, _hand_edit_project_filter, _init_repo_with_remote, \
    _write_synthetic_wav  # noqa: F401 (_designers is an autouse fixture)

META = {'title': 'Ready Player One', 'year': '2018', 'audio_types': ['Atmos']}
ID = 'ready-player-one'


def _designed(tmp_path, multichannel=False):
    work_dir = str(tmp_path / 'work')
    project_dir = os.path.join(work_dir, ID)
    os.makedirs(project_dir)
    mono = os.path.join(project_dir, 'mono.wav')
    _write_synthetic_wav(mono, channel_values=(1000,))
    kwargs = {}
    if multichannel:
        kept = os.path.join(project_dir, 'multichannel.wav')
        _write_synthetic_wav(kept, channel_values=(1000, 2000))
        kwargs = dict(multichannel_wav_path=kept, channel_layout_name='stereo')
    design_and_queue(Session(AnalysisConfig()), ID, mono, DESIGNER_NAME, str(tmp_path / 'queue'), meta=META,
                     project_dir=project_dir, **kwargs)
    update_entry(str(tmp_path / 'queue'), ID, status='accepted', chosen_candidate_index=0)
    return str(tmp_path / 'queue'), work_dir, project_dir


def _publish(tmp_path, queue_dir, **kwargs):
    xml_repo, _ = _init_repo_with_remote(tmp_path, 'xml_repo')
    return publish_reviewed_queue(queue_dir, xml_repo, xml_dir='xml', **kwargs)


def test_the_digest_without_a_work_directory_is_what_publish_records(tmp_path):
    queue_dir, _, _ = _designed(tmp_path)
    _publish(tmp_path, queue_dir)

    entry = read_entry(queue_dir, ID)

    assert current_publish_digest(entry) == entry.published_digest


def test_the_digest_with_pipeline_pure_projects_is_what_publish_records(tmp_path):
    queue_dir, work_dir, _ = _designed(tmp_path, multichannel=True)
    _publish(tmp_path, queue_dir, work_dir=work_dir)

    entry = read_entry(queue_dir, ID)

    assert current_publish_digest(entry, work_dir=work_dir) == entry.published_digest


def test_the_digest_with_a_hand_edited_project_is_what_publish_records_and_moves_with_the_edit(tmp_path):
    queue_dir, work_dir, project_dir = _designed(tmp_path)
    mono = os.path.join(project_dir, f'{ID}.mono.beq')
    _hand_edit_project_filter(mono, CompleteFilter(fs=1000, filters=[PeakingEQ(1000, 55.0, 1.4, -6.0)]))
    _publish(tmp_path, queue_dir, work_dir=work_dir)
    entry = read_entry(queue_dir, ID)
    assert current_publish_digest(entry, work_dir=work_dir) == entry.published_digest

    _hand_edit_project_filter(mono, CompleteFilter(fs=1000, filters=[PeakingEQ(1000, 60.0, 1.4, -6.0)]))

    assert current_publish_digest(entry, work_dir=work_dir) != entry.published_digest  # out of date


def test_asking_for_the_digest_writes_nothing(tmp_path):
    queue_dir, work_dir, project_dir = _designed(tmp_path, multichannel=True)
    os.remove(os.path.join(project_dir, f'{ID}.multichannel.beq'))  # publish would recreate it; the preview must not
    before = sorted((p, os.stat(p).st_mtime_ns) for root, _, names in os.walk(tmp_path)
                    for p in (os.path.join(root, n) for n in names))

    current_publish_digest(read_entry(queue_dir, ID), work_dir=work_dir)

    assert sorted((p, os.stat(p).st_mtime_ns) for root, _, names in os.walk(tmp_path)
                  for p in (os.path.join(root, n) for n in names)) == before


def test_the_digest_follows_the_metadata_the_artwork_and_the_inputs_publish_is_given(tmp_path):
    queue_dir, _, _ = _designed(tmp_path)
    _publish(tmp_path, queue_dir)
    entry = read_entry(queue_dir, ID)

    assert current_publish_digest(entry) == entry.published_digest
    assert current_publish_digest(entry, meta_defaults={'author': 'me'}) != entry.published_digest
    assert current_publish_digest(entry, has_image=True) != entry.published_digest
    art = tmp_path / 'poster.png'
    art.write_bytes(b'poster')
    assert current_publish_digest(update_entry(queue_dir, ID, art_path=str(art))) != entry.published_digest
    typo = update_entry(queue_dir, ID, meta={**META, 'title': 'Ready Player Won'})
    assert current_publish_digest(typo) != entry.published_digest


def test_an_entry_that_is_not_accepted_or_published_has_no_digest(tmp_path):
    queue_dir, _, _ = _designed(tmp_path)
    pending = update_entry(queue_dir, ID, status='pending', chosen_candidate_index=None)

    with pytest.raises(ValueError):
        current_publish_digest(pending)


def test_independently_edited_projects_that_disagree_are_a_conflict_not_a_digest(tmp_path):
    queue_dir, work_dir, project_dir = _designed(tmp_path, multichannel=True)
    _hand_edit_project_filter(os.path.join(project_dir, f'{ID}.mono.beq'),
                              CompleteFilter(fs=1000, filters=[PeakingEQ(1000, 55.0, 1.4, -6.0)]))
    _hand_edit_project_filter(os.path.join(project_dir, f'{ID}.multichannel.beq'),
                              CompleteFilter(fs=1000, filters=[PeakingEQ(1000, 30.0, 1.4, -6.0)]))

    with pytest.raises(ProjectFilterConflict):
        current_publish_digest(read_entry(queue_dir, ID), work_dir=work_dir)
