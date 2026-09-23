'''
Phases 1-2 of design/candidate-review-plan.md: QueueEntry persistence, the
batch_design() driver, and applying/publishing a human's pick -- exercised
without any GUI, since the review dialog (Phase 3) is just a reader/writer
of the same queue.
'''
import gzip
import io
import json
import os
import subprocess
import wave

import numpy as np
import pytest
from PIL import Image

from model.iir import CompleteFilter, LowShelf, PeakingEQ
from pipeline.config import AnalysisConfig
from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.orchestrate import Session
from pipeline.publish.git import RepoTarget
from pipeline.review import CandidateSummary, QueueEntry, apply_reviewed_entry, batch_design, design_and_queue, \
    describe_publish_error, publish_reviewed_queue, read_entry, read_queue, split_publish_results, update_entry, \
    write_queue_entry

DESIGNER_NAME = 'test.review'
DECLINE_DESIGNER_NAME = 'test.review.decline'


def _multi_candidate_designer(request):
    return DesignResponse(contract_version='1.0', candidates=[
        DesignCandidate(filters=[BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7)],
                        confidence=0.9, mv_adjust_db=4.0, gain_reduction_db=-1.5, method='fitted',
                        commentary={'note': 'primary'}),
        DesignCandidate(filters=[BiquadSpec(type='peaking_eq', freq_hz=40.0, gain_db=-3.0, q=2.0)],
                        confidence=0.4, mv_adjust_db=1.0, gain_reduction_db=0.0, method='fitted',
                        commentary={'note': 'alternative'}),
    ])


def _decline_designer(request):
    return DesignResponse(contract_version='1.0', decline_reason='no_rolloff_detected',
                          decline_message='nothing to correct')


@pytest.fixture(autouse=True)
def _designers():
    register_designer(DESIGNER_NAME, _multi_candidate_designer)
    register_designer(DECLINE_DESIGNER_NAME, _decline_designer)
    yield
    unregister_designer(DESIGNER_NAME)
    unregister_designer(DECLINE_DESIGNER_NAME)


def _write_synthetic_wav(path, fs=48000, duration_s=0.25, channel_values=(1000, 2000, 3000, 4000, 5000, 6000)):
    n_frames = int(fs * duration_s)
    frame = np.array(channel_values, dtype=np.int16)
    data = np.tile(frame, (n_frames, 1)).astype('<i2').tobytes()
    with wave.open(path, 'wb') as w:
        w.setnchannels(len(channel_values))
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(data)


def _hand_edit_project_filter(path, new_filter):
    '''
    Simulates a human opening a .beq project in the interactive app and re-exporting via app.py's
    exportProject() -- which calls the generic model.codec.signaldata_to_json(), so pipeline_filter_hash
    (an additive key that function knows nothing about) is never re-emitted.
    '''
    with gzip.open(path, 'rb') as f:
        data = json.loads(f.read().decode('utf-8'))
    master = data[0]
    master['filter_presets'][master['active_filter_preset']] = new_filter.to_json()
    master.pop('pipeline_filter_hash', None)
    with gzip.open(path, 'wb') as f:
        f.write(json.dumps(data).encode('utf-8'))


# --- QueueEntry construction/validation -----------------------------------

def _candidate(**overrides):
    defaults = dict(filters={'_type': 'CompleteFilter', 'fs': 1000, 'description': 'designer', 'filters': []},
                    confidence=0.9, method='fitted', mv_adjust_db=4.0)
    defaults.update(overrides)
    return CandidateSummary(**defaults)


def test_accepted_entry_requires_chosen_candidate_index():
    with pytest.raises(ValueError, match='chosen_candidate_index'):
        QueueEntry(id='t1', fs=1000, meta={}, curve={}, candidates=[_candidate()], status='accepted')


def test_accepted_entry_index_out_of_range_rejected():
    with pytest.raises(ValueError, match='out of range'):
        QueueEntry(id='t1', fs=1000, meta={}, curve={}, candidates=[_candidate()], status='accepted',
                  chosen_candidate_index=1)


def test_invalid_status_rejected():
    with pytest.raises(ValueError, match='status'):
        QueueEntry(id='t1', fs=1000, meta={}, curve={}, status='not-a-status')


# --- persistence -----------------------------------------------------------

def test_queue_entry_round_trips_multiple_candidates_and_commentary(tmp_path):
    entry = QueueEntry(
        id='ready-player-one', fs=48000, meta={'beq_title': 'Ready Player One'},
        curve={'_type': 'MagnitudeData', 'name': 'avg', 'description': '', 'x': [1.0, 2.0], 'y': [0.1, 0.2],
              'colour': None, 'linestyle': '-'},
        peak_curve={'_type': 'MagnitudeData', 'name': 'peak', 'description': '', 'x': [1.0, 2.0], 'y': [1.1, 1.2],
                    'colour': None, 'linestyle': '-'},
        audio_stream=2,
        candidates=[
            _candidate(confidence=0.9, commentary={'alignment': 'LR4', 'knee_hz': '25.0'}),
            _candidate(confidence=0.4, commentary={'note': 'alternative'}),
        ],
    )
    write_queue_entry(str(tmp_path), entry)

    read_back = read_entry(str(tmp_path), 'ready-player-one')

    assert read_back == entry
    assert read_back.audio_stream == 2
    assert read_back.peak_curve == entry.peak_curve
    assert [c.confidence for c in read_back.candidates] == [0.9, 0.4]
    assert read_back.candidates[0].commentary == {'alignment': 'LR4', 'knee_hz': '25.0'}


def test_read_entry_missing_id_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_entry(str(tmp_path), 'does-not-exist')


def test_read_queue_sorts_pending_first(tmp_path):
    write_queue_entry(str(tmp_path), QueueEntry(id='b', fs=1000, meta={}, curve={}, status='pending'))
    write_queue_entry(str(tmp_path), QueueEntry(id='a', fs=1000, meta={}, curve={},
                                                candidates=[_candidate()], status='accepted',
                                                chosen_candidate_index=0))
    write_queue_entry(str(tmp_path), QueueEntry(id='c', fs=1000, meta={}, curve={}, status='pending'))

    ids = [e.id for e in read_queue(str(tmp_path))]

    assert ids == ['b', 'c', 'a']  # both pending entries (alphabetical), then the accepted one


def test_update_entry_rewrites_the_file(tmp_path):
    write_queue_entry(str(tmp_path), QueueEntry(id='t1', fs=1000, meta={}, curve={},
                                                candidates=[_candidate(), _candidate()], status='pending'))

    updated = update_entry(str(tmp_path), 't1', status='accepted', chosen_candidate_index=1)

    assert updated.status == 'accepted'
    assert updated.chosen_candidate_index == 1
    assert read_entry(str(tmp_path), 't1') == updated


def test_update_entry_missing_id_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        update_entry(str(tmp_path), 'does-not-exist', status='skipped')


def test_queue_entry_round_trips_art_path_and_overridden(tmp_path):
    entry = QueueEntry(id='t1', fs=1000, meta={}, curve={}, art_path='/tmp/x.jpg', art_overridden=True)
    write_queue_entry(str(tmp_path), entry)

    read_back = read_entry(str(tmp_path), 't1')

    assert read_back.art_path == '/tmp/x.jpg'
    assert read_back.art_overridden is True


def test_reading_a_pre_existing_entry_without_art_fields_defaults_them(tmp_path):
    import json
    path = tmp_path / 't1.json'
    path.write_text(json.dumps({
        'id': 't1', 'fs': 1000, 'meta': {}, 'curve': {}, 'candidates': [], 'status': 'pending',
    }))

    read_back = read_entry(str(tmp_path), 't1')

    assert read_back.art_path is None
    assert read_back.art_overridden is False
    assert read_back.peak_curve is None


# --- batch_design ------------------------------------------------------------

def test_batch_design_writes_one_pending_entry_per_title(tmp_path):
    source_wav = str(tmp_path / 'source.wav')
    _write_synthetic_wav(source_wav)
    queue_dir = str(tmp_path / 'queue')
    work_dir = str(tmp_path / 'work')

    written = batch_design(
        [('title-one', source_wav, {'beq_title': 'Title One'}), ('title-two', source_wav, None)],
        DESIGNER_NAME, queue_dir, work_dir)

    assert written == ['title-one', 'title-two']
    entries = {e.id: e for e in read_queue(queue_dir)}
    assert set(entries) == {'title-one', 'title-two'}
    for entry in entries.values():
        assert entry.status == 'pending'
        assert len(entry.candidates) == 2
        assert entry.candidates[0].confidence == 0.9  # top-ranked first
        assert entry.candidates[0].gain_reduction_db == -1.5
        assert entry.candidates[1].confidence == 0.4
        assert entry.candidates[1].gain_reduction_db == 0.0
        assert entry.curve['_type'] == 'MagnitudeData'
        assert entry.peak_curve['_type'] == 'MagnitudeData'
        assert entry.peak_curve['description'] == 'peak'
    assert entries['title-one'].meta == {'beq_title': 'Title One'}
    assert entries['title-two'].meta == {}


def test_batch_design_calls_on_item_done_after_each_entry_is_written(tmp_path):
    source_wav = str(tmp_path / 'source.wav')
    _write_synthetic_wav(source_wav)
    queue_dir = str(tmp_path / 'queue')
    work_dir = str(tmp_path / 'work')
    seen = []

    def on_item_done(entry_id):
        seen.append((entry_id, read_entry(queue_dir, entry_id).status))

    batch_design([('title-one', source_wav, None), ('title-two', source_wav, None)],
                 DESIGNER_NAME, queue_dir, work_dir, on_item_done=on_item_done)

    assert seen == [('title-one', 'pending'), ('title-two', 'pending')]


def test_batch_design_declined_title_carries_the_reason(tmp_path):
    source_wav = str(tmp_path / 'source.wav')
    _write_synthetic_wav(source_wav)
    queue_dir = str(tmp_path / 'queue')
    work_dir = str(tmp_path / 'work')

    batch_design([('declined-title', source_wav, None)], DECLINE_DESIGNER_NAME, queue_dir, work_dir)

    entry = read_entry(queue_dir, 'declined-title')
    assert entry.status == 'pending'
    assert entry.candidates == []
    assert entry.decline_reason == 'no_rolloff_detected'
    assert entry.decline_message == 'nothing to correct'


def test_design_and_queue_threads_channels_to_the_request(tmp_path):
    ''' design_and_queue()'s channels param (a caller's own per-channel decomposition, e.g. model/batch.py's
    ExtractCandidate.design() via Session.load_channels()) reaches the designer as DesignRequest.channels. '''
    from pipeline.config import AnalysisConfig
    from pipeline.orchestrate import Session
    from pipeline.review import design_and_queue

    seen_requests = []

    def recording_designer(request):
        seen_requests.append(request)
        return DesignResponse(contract_version='1.0', candidates=[
            DesignCandidate(filters=[BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7)],
                            confidence=0.9, mv_adjust_db=4.0, method='fitted'),
        ])

    register_designer('test.channels_through_queue', recording_designer)
    try:
        wav_path = str(tmp_path / 'mono.wav')
        _write_synthetic_wav(wav_path, channel_values=(1000,))
        queue_dir = str(tmp_path / 'queue')
        channels = {'FL': np.zeros(5), 'FR': np.ones(5)}

        design_and_queue(Session(AnalysisConfig()), 'title-one', wav_path, 'test.channels_through_queue',
                         queue_dir, channels=channels)
    finally:
        unregister_designer('test.channels_through_queue')

    assert seen_requests[0].channels is channels


# --- design_and_queue: output 1's .beq project files ------------------------

def test_design_and_queue_writes_a_mono_project_when_project_dir_given(tmp_path):
    from pipeline.publish.project import read_project_filter

    wav_path = str(tmp_path / 'mono.wav')
    _write_synthetic_wav(wav_path, channel_values=(1000,))
    queue_dir = str(tmp_path / 'queue')
    project_dir = str(tmp_path / 'projects')

    entry = design_and_queue(Session(AnalysisConfig()), 'title-one', wav_path, DESIGNER_NAME, queue_dir,
                             project_dir=project_dir)

    mono_project = os.path.join(project_dir, 'title-one.mono.beq')
    assert os.path.isfile(mono_project)
    filt, is_pure = read_project_filter(mono_project)
    assert is_pure is True
    assert filt.to_json() == entry.candidates[0].filters


def test_design_and_queue_writes_a_multichannel_project_when_given_one(tmp_path):
    mono_wav = str(tmp_path / 'mono.wav')
    _write_synthetic_wav(mono_wav, channel_values=(1000,))
    mc_wav = str(tmp_path / 'multi.wav')
    _write_synthetic_wav(mc_wav, channel_values=(1000, 2000, 3000, 4000, 5000, 6000))
    queue_dir = str(tmp_path / 'queue')
    project_dir = str(tmp_path / 'projects')

    design_and_queue(Session(AnalysisConfig()), 'title-one', mono_wav, DESIGNER_NAME, queue_dir,
                     multichannel_wav_path=mc_wav, channel_layout_name='5.1', project_dir=project_dir)

    assert os.path.isfile(os.path.join(project_dir, 'title-one.mono.beq'))
    assert os.path.isfile(os.path.join(project_dir, 'title-one.multichannel.beq'))


def test_design_and_queue_does_not_write_projects_without_project_dir(tmp_path):
    ''' Backward compatibility: today's call shape (project_dir omitted) writes no .beq files. '''
    wav_path = str(tmp_path / 'source.wav')
    _write_synthetic_wav(wav_path)
    queue_dir = str(tmp_path / 'queue')
    work_dir = str(tmp_path / 'work')

    batch_design([('title-one', wav_path, None)], DESIGNER_NAME, queue_dir, work_dir)

    beq_files = [f for _, _, files in os.walk(str(tmp_path)) for f in files if f.endswith('.beq')]
    assert beq_files == []


def test_pipeline_review_module_has_no_qtpy_import():
    import ast
    import pathlib
    source = (pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python' / 'pipeline' / 'review.py').resolve()
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith('qtpy') for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith('qtpy')


# --- Phase 2: apply_reviewed_entry / publish_reviewed_queue ----------------

def _init_repo_with_remote(tmp_path, subdir):
    bare = tmp_path / f'{subdir}.git'
    subprocess.run(['git', 'init', '--bare', '-q', str(bare)], check=True, capture_output=True)
    work = tmp_path / subdir
    work.mkdir()
    subprocess.run(['git', 'init', '-q', str(work)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(work), 'config', 'user.email', 'test@example.com'], check=True,
                   capture_output=True)
    subprocess.run(['git', '-C', str(work), 'config', 'user.name', 'Test'], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(work), 'remote', 'add', 'origin', str(bare)], check=True, capture_output=True)
    return RepoTarget(local_path=str(work)), bare


def _designed_entry(tmp_path, entry_id='ready-player-one', meta=None):
    ''' A real, batch_design()-produced entry -- realistic CompleteFilter JSON and curve, not hand-built. '''
    source_wav = str(tmp_path / 'source.wav')
    _write_synthetic_wav(source_wav)
    queue_dir = str(tmp_path / 'queue')
    batch_design([(entry_id, source_wav, meta)], DESIGNER_NAME, queue_dir, str(tmp_path / 'work'))
    return queue_dir, entry_id


def test_apply_reviewed_entry_uses_the_chosen_candidate_not_the_top_one(tmp_path):
    queue_dir, entry_id = _designed_entry(tmp_path)
    update_entry(queue_dir, entry_id, status='accepted', chosen_candidate_index=1)

    complete_filter = apply_reviewed_entry(read_entry(queue_dir, entry_id))

    filters = list(complete_filter)
    assert not any(isinstance(f, LowShelf) for f in filters)  # candidates[0]'s filter
    assert any(isinstance(f, PeakingEQ) for f in filters)     # candidates[1]'s filter


def test_apply_reviewed_entry_top_candidate(tmp_path):
    queue_dir, entry_id = _designed_entry(tmp_path)
    update_entry(queue_dir, entry_id, status='accepted', chosen_candidate_index=0)

    complete_filter = apply_reviewed_entry(read_entry(queue_dir, entry_id))

    assert any(isinstance(f, LowShelf) for f in list(complete_filter))


def test_apply_reviewed_entry_requires_accepted_status(tmp_path):
    queue_dir, entry_id = _designed_entry(tmp_path)

    with pytest.raises(ValueError, match='accepted'):
        apply_reviewed_entry(read_entry(queue_dir, entry_id))


def test_publish_reviewed_queue_xml_only(tmp_path):
    queue_dir, entry_id = _designed_entry(tmp_path, meta={'title': 'Ready Player One', 'year': '2018',
                                                          'audio_types': ['Atmos']})
    update_entry(queue_dir, entry_id, status='accepted', chosen_candidate_index=0)
    xml_repo, xml_bare = _init_repo_with_remote(tmp_path, 'xml_repo')

    results = publish_reviewed_queue(queue_dir, xml_repo, xml_dir='xml')

    assert len(results) == 1
    assert results[0]['id'] == entry_id
    assert '<beq_title>Ready Player One</beq_title>' in results[0]['xml']
    assert 'image_url' not in results[0]
    xml_on_remote = subprocess.run(
        ['git', '-C', str(xml_bare), 'cat-file', '-p', f"{results[0]['xml_commit']}:xml/{entry_id}.xml"],
        check=True, capture_output=True, text=True).stdout
    assert xml_on_remote == results[0]['xml']
    assert read_entry(queue_dir, entry_id).status == 'published'


def test_publish_reviewed_queue_defaults_gain_from_chosen_candidates_mv_adjust_db(tmp_path):
    queue_dir, entry_id = _designed_entry(tmp_path, meta={'title': 'Ready Player One', 'year': '2018',
                                                          'audio_types': ['Atmos']})
    update_entry(queue_dir, entry_id, status='accepted', chosen_candidate_index=0)
    xml_repo, _ = _init_repo_with_remote(tmp_path, 'xml_repo')

    results = publish_reviewed_queue(queue_dir, xml_repo, xml_dir='xml')

    assert '<beq_gain>+4</beq_gain>' in results[0]['xml']  # candidates[0].mv_adjust_db == 4.0


def test_publish_reviewed_queue_with_image(tmp_path):
    queue_dir, entry_id = _designed_entry(tmp_path, meta={'title': 'Ready Player One', 'year': '2018',
                                                          'audio_types': ['Atmos']})
    update_entry(queue_dir, entry_id, status='accepted', chosen_candidate_index=0)
    xml_repo, _ = _init_repo_with_remote(tmp_path, 'xml_repo')
    images_repo, images_bare = _init_repo_with_remote(tmp_path, 'images_repo')

    results = publish_reviewed_queue(queue_dir, xml_repo, xml_dir='xml', images_repo=images_repo, image_dir='img',
                                     image_owner='3ll3d00d', image_repo_name='beq-images')

    assert results[0]['image_url'].endswith(f'img/{entry_id}.png')
    pushed_png = subprocess.run(
        ['git', '-C', str(images_bare), 'cat-file', '-p', f'HEAD:img/{entry_id}.png'],
        check=True, capture_output=True).stdout
    image = Image.open(io.BytesIO(pushed_png))
    assert image.format == 'PNG'


def test_publish_reviewed_queue_passes_entry_art_path_as_poster(tmp_path):
    queue_dir, entry_id = _designed_entry(tmp_path, meta={'title': 'Ready Player One', 'year': '2018',
                                                          'audio_types': ['Atmos']})
    update_entry(queue_dir, entry_id, status='accepted', chosen_candidate_index=0)
    poster_path = str(tmp_path / 'poster.jpg')
    Image.new('RGB', (300, 450), color=(10, 20, 30)).save(poster_path, format='JPEG')
    update_entry(queue_dir, entry_id, art_path=poster_path, art_overridden=True)
    xml_repo, _ = _init_repo_with_remote(tmp_path, 'xml_repo')
    images_repo, images_bare = _init_repo_with_remote(tmp_path, 'images_repo')

    publish_reviewed_queue(queue_dir, xml_repo, xml_dir='xml', images_repo=images_repo, image_dir='img',
                           image_owner='3ll3d00d', image_repo_name='beq-images')

    pushed_png = subprocess.run(
        ['git', '-C', str(images_bare), 'cat-file', '-p', f'HEAD:img/{entry_id}.png'],
        check=True, capture_output=True).stdout
    image = Image.open(io.BytesIO(pushed_png))
    # top-left pixel of a composed (poster-on-top) image should be the poster's fill colour, not the
    # chart's white background -- proof poster_path actually reached render_report()/compose_with_poster().
    assert image.convert('RGB').getpixel((0, 0)) != (255, 255, 255)


def test_publish_reviewed_queue_skips_non_accepted_entries(tmp_path):
    queue_dir, accepted_id = _designed_entry(tmp_path, entry_id='accepted-title')
    update_entry(queue_dir, accepted_id, status='accepted', chosen_candidate_index=0)
    source_wav = str(tmp_path / 'source.wav')
    batch_design([('pending-title', source_wav, None)], DESIGNER_NAME, queue_dir, str(tmp_path / 'work2'))
    xml_repo, _ = _init_repo_with_remote(tmp_path, 'xml_repo')

    results = publish_reviewed_queue(queue_dir, xml_repo,
                                     meta_defaults={'title': 'T', 'year': '2020', 'audio_types': ['Atmos']})

    assert [r['id'] for r in results] == ['accepted-title']
    assert read_entry(queue_dir, 'pending-title').status == 'pending'


def test_publish_reviewed_queue_is_idempotent(tmp_path):
    queue_dir, entry_id = _designed_entry(tmp_path, meta={'title': 'Ready Player One', 'year': '2018',
                                                          'audio_types': ['Atmos']})
    update_entry(queue_dir, entry_id, status='accepted', chosen_candidate_index=0)
    xml_repo, _ = _init_repo_with_remote(tmp_path, 'xml_repo')

    first = publish_reviewed_queue(queue_dir, xml_repo, xml_dir='xml')
    second = publish_reviewed_queue(queue_dir, xml_repo, xml_dir='xml')

    assert len(first) == 1
    assert second == []


# --- publish_reviewed_queue: reading the filter from a .beq project (work_dir) ---

def test_publish_reviewed_queue_without_work_dir_uses_apply_reviewed_entry_as_before(tmp_path):
    ''' Backward compatibility: omitting work_dir (today's call shape) publishes exactly what it does
    today -- protects every existing test_publish_reviewed_queue_* test in this file from regressing. '''
    from model.minidsp import xml_to_filt

    queue_dir, entry_id = _designed_entry(tmp_path, meta={'title': 'Ready Player One', 'year': '2018',
                                                          'audio_types': ['Atmos']})
    update_entry(queue_dir, entry_id, status='accepted', chosen_candidate_index=0)
    xml_repo, _ = _init_repo_with_remote(tmp_path, 'xml_repo')

    results = publish_reviewed_queue(queue_dir, xml_repo, xml_dir='xml')

    written = str(tmp_path / 'out.xml')
    with open(written, 'w', encoding='utf-8') as f:
        f.write(results[0]['xml'])
    read_back = xml_to_filt(written, fs=1000)
    assert any(isinstance(f, LowShelf) for f in read_back)  # candidates[0]'s filter, untouched by chunk 2
    assert read_entry(queue_dir, entry_id).status == 'published'


def test_publish_reviewed_queue_reads_the_edited_mono_project_when_work_dir_given(tmp_path):
    from model.minidsp import xml_to_filt

    entry_id = 'ready-player-one'
    queue_dir = str(tmp_path / 'queue')
    work_dir = str(tmp_path / 'work')
    project_dir = os.path.join(work_dir, entry_id)
    os.makedirs(project_dir)
    mono_wav = os.path.join(project_dir, 'mono.wav')
    _write_synthetic_wav(mono_wav, channel_values=(1000,))

    session = Session(AnalysisConfig())
    design_and_queue(session, entry_id, mono_wav, DESIGNER_NAME, queue_dir,
                     meta={'title': 'Ready Player One', 'year': '2018', 'audio_types': ['Atmos']},
                     project_dir=project_dir)
    update_entry(queue_dir, entry_id, status='accepted', chosen_candidate_index=0)

    mono_project = os.path.join(project_dir, f'{entry_id}.mono.beq')
    edited_filter = CompleteFilter(fs=1000, filters=[PeakingEQ(1000, 55.0, 1.4, -6.0)])
    _hand_edit_project_filter(mono_project, edited_filter)

    xml_repo, _ = _init_repo_with_remote(tmp_path, 'xml_repo')

    results = publish_reviewed_queue(queue_dir, xml_repo, xml_dir='xml', work_dir=work_dir)

    assert 'error' not in results[0]
    written = str(tmp_path / 'out.xml')
    with open(written, 'w', encoding='utf-8') as f:
        f.write(results[0]['xml'])
    read_back = xml_to_filt(written, fs=1000)
    assert any(isinstance(f, PeakingEQ) for f in read_back)  # the human edit
    assert not any(isinstance(f, LowShelf) for f in read_back)  # the designer's original top pick, gone
    assert read_entry(queue_dir, entry_id).status == 'published'


def test_publish_reviewed_queue_reports_a_project_conflict_without_publishing(tmp_path):
    entry_id = 'ready-player-one'
    queue_dir = str(tmp_path / 'queue')
    work_dir = str(tmp_path / 'work')
    project_dir = os.path.join(work_dir, entry_id)
    os.makedirs(project_dir)
    mono_wav = os.path.join(project_dir, 'mono.wav')
    _write_synthetic_wav(mono_wav, channel_values=(1000,))
    mc_wav = os.path.join(project_dir, 'multichannel.wav')
    _write_synthetic_wav(mc_wav, channel_values=(1000, 2000, 3000, 4000, 5000, 6000))

    session = Session(AnalysisConfig())
    design_and_queue(session, entry_id, mono_wav, DESIGNER_NAME, queue_dir,
                     meta={'title': 'Ready Player One', 'year': '2018', 'audio_types': ['Atmos']},
                     multichannel_wav_path=mc_wav, channel_layout_name='5.1', project_dir=project_dir)
    update_entry(queue_dir, entry_id, status='accepted', chosen_candidate_index=0)

    mono_project = os.path.join(project_dir, f'{entry_id}.mono.beq')
    mc_project = os.path.join(project_dir, f'{entry_id}.multichannel.beq')
    _hand_edit_project_filter(mono_project, CompleteFilter(fs=1000, filters=[PeakingEQ(1000, 55.0, 1.4, -6.0)]))
    _hand_edit_project_filter(mc_project, CompleteFilter(fs=1000, filters=[PeakingEQ(1000, 80.0, 0.9, 3.0)]))

    xml_repo, _ = _init_repo_with_remote(tmp_path, 'xml_repo')

    results = publish_reviewed_queue(queue_dir, xml_repo, xml_dir='xml', work_dir=work_dir)

    assert results == [{'id': entry_id, 'error': 'project_conflict'}]
    assert read_entry(queue_dir, entry_id).status == 'accepted'  # not marked published -- a rerun will retry it


def _accepted_title_with_projects(tmp_path, multichannel=True):
    entry_id = 'ready-player-one'
    queue_dir = str(tmp_path / 'queue')
    work_dir = str(tmp_path / 'work')
    project_dir = os.path.join(work_dir, entry_id)
    os.makedirs(project_dir)
    mono_wav = os.path.join(project_dir, 'mono.wav')
    _write_synthetic_wav(mono_wav, channel_values=(1000,))
    kwargs = {}
    if multichannel:
        mc_wav = os.path.join(project_dir, 'multichannel.wav')
        _write_synthetic_wav(mc_wav, channel_values=(1000, 2000, 3000, 4000, 5000, 6000))
        kwargs = {'multichannel_wav_path': mc_wav, 'channel_layout_name': '5.1'}
    design_and_queue(Session(AnalysisConfig()), entry_id, mono_wav, DESIGNER_NAME, queue_dir,
                     meta={'title': 'Ready Player One', 'year': '2018', 'audio_types': ['Atmos']},
                     project_dir=project_dir, **kwargs)
    update_entry(queue_dir, entry_id, status='accepted', chosen_candidate_index=0)
    return entry_id, queue_dir, work_dir, project_dir


def test_design_and_queue_reports_which_projects_it_wrote_and_which_it_left_alone(tmp_path):
    entry_id, queue_dir, work_dir, project_dir = _accepted_title_with_projects(tmp_path, multichannel=False)
    mono_project = os.path.join(project_dir, f'{entry_id}.mono.beq')
    seen = []
    session = Session(AnalysisConfig())
    mono_wav = os.path.join(project_dir, 'mono.wav')

    design_and_queue(session, entry_id, mono_wav, DESIGNER_NAME, queue_dir, project_dir=project_dir,
                     on_projects=seen.append)
    _hand_edit_project_filter(mono_project, CompleteFilter(fs=1000, filters=[PeakingEQ(1000, 55.0, 1.4, -6.0)]))
    design_and_queue(session, entry_id, mono_wav, DESIGNER_NAME, queue_dir, project_dir=project_dir,
                     on_projects=seen.append)

    assert seen == [{'mono': True, 'multichannel': None}, {'mono': False, 'multichannel': None}]


def test_publish_reviewed_queue_says_when_it_published_a_human_edit_and_aligns_the_sibling(tmp_path):
    from pipeline.publish.project import read_project_filter

    entry_id, queue_dir, work_dir, project_dir = _accepted_title_with_projects(tmp_path)
    mono_project = os.path.join(project_dir, f'{entry_id}.mono.beq')
    mc_project = os.path.join(project_dir, f'{entry_id}.multichannel.beq')
    edited = CompleteFilter(fs=1000, filters=[PeakingEQ(1000, 55.0, 1.4, -6.0)])
    _hand_edit_project_filter(mono_project, edited)
    xml_repo, _ = _init_repo_with_remote(tmp_path, 'xml_repo')

    results = publish_reviewed_queue(queue_dir, xml_repo, xml_dir='xml', work_dir=work_dir)

    assert results[0]['edited_project'] == 'mono'
    assert results[0]['projects_aligned'] == ['multichannel']
    mc_filter, mc_pure = read_project_filter(mc_project)
    assert mc_filter.to_json() == edited.to_json()
    assert mc_pure is True


def test_publish_reviewed_queue_adds_no_project_notes_when_nothing_was_edited(tmp_path):
    entry_id, queue_dir, work_dir, project_dir = _accepted_title_with_projects(tmp_path)
    xml_repo, _ = _init_repo_with_remote(tmp_path, 'xml_repo')

    results = publish_reviewed_queue(queue_dir, xml_repo, xml_dir='xml', work_dir=work_dir)

    assert 'edited_project' not in results[0]
    assert 'projects_aligned' not in results[0]


def test_split_publish_results_separates_published_from_needs_attention():
    results = [{'id': 'a', 'xml_sha': '1'}, {'id': 'b', 'error': 'project_conflict'}, {'id': 'c'}]

    published, needs_attention = split_publish_results(results)

    assert [r['id'] for r in published] == ['a', 'c']
    assert needs_attention == [{'id': 'b', 'error': 'project_conflict'}]


def test_describe_publish_error_explains_a_project_conflict_and_passes_an_unknown_code_through():
    assert describe_publish_error({'id': 'b', 'error': 'project_conflict'}).startswith('b: the mono and multichannel')
    assert describe_publish_error({'id': 'x', 'error': 'something_new'}) == 'x: something_new'
