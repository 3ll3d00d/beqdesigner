'''
design/library-sync-pipeline-plan.md Appendix B (chunk 2): `.beq` project
files as the published source. Covers pipeline.publish.project's write/read/
resolve functions in isolation -- pipeline.review's design_and_queue()/
publish_reviewed_queue() wiring is covered in test_pipeline_review.py.
'''
import gzip
import json
import wave

import numpy as np
import pytest

from model.codec import signalmodel_from_json
from model.iir import CompleteFilter, LowShelf, PeakingEQ
from pipeline.config import AnalysisConfig
from pipeline.orchestrate import Session
from pipeline.publish.project import (
    ProjectFilterConflict,
    align_projects,
    read_project_filter,
    resolve_published_filter,
    resolve_published_projects,
    write_mono_project,
    write_multichannel_project,
    write_title_projects_if_safe,
)

_PIPELINE_FILTER = CompleteFilter(fs=1000, filters=[PeakingEQ(1000, 40.0, 2.0, -3.0)])
_HUMAN_FILTER = CompleteFilter(fs=1000, filters=[LowShelf(1000, 18.0, 0.7, 4.5)])
_OTHER_HUMAN_FILTER = CompleteFilter(fs=1000, filters=[LowShelf(1000, 30.0, 1.2, 6.0)])


def _write_mono_wav(path, fs=48000, duration_s=0.25):
    n_frames = int(fs * duration_s)
    samples = (0.2 * np.sin(2 * np.pi * 40 * np.linspace(0, duration_s, n_frames, endpoint=False))).astype(np.float32)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes((samples * 32767).astype('<i2').tobytes())


def _write_multichannel_wav(path, channel_values, fs=48000, duration_s=0.1):
    n_frames = int(fs * duration_s)
    frame = np.array(channel_values, dtype=np.int16)
    data = np.tile(frame, (n_frames, 1)).astype('<i2').tobytes()
    with wave.open(path, 'wb') as w:
        w.setnchannels(len(channel_values))
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(data)


def _read_raw(path):
    with gzip.open(path, 'rb') as f:
        return json.loads(f.read().decode('utf-8'))


def _write_raw(path, data):
    with gzip.open(path, 'wb') as f:
        f.write(json.dumps(data).encode('utf-8'))


def _hand_edit_filter(path, new_filter):
    '''
    Simulates a human opening the project in the interactive app and re-exporting via app.py's
    exportProject() -- which calls the generic model.codec.signaldata_to_json(), so pipeline_filter_hash
    (an additive key that function knows nothing about) is never re-emitted.
    '''
    data = _read_raw(path)
    master = data[0]
    master['filter_presets'][master['active_filter_preset']] = new_filter.to_json()
    master.pop('pipeline_filter_hash', None)
    _write_raw(path, data)


# --- write_project / read_project_filter ------------------------------------

def test_write_and_read_project_round_trips_the_filter(tmp_path):
    session = Session(AnalysisConfig())
    wav_path = str(tmp_path / 'mono.wav')
    _write_mono_wav(wav_path)
    out_path = str(tmp_path / 'title.mono.beq')

    write_mono_project(session, wav_path, _PIPELINE_FILTER, out_path)
    read_filter, is_pure = read_project_filter(out_path)

    assert is_pure is True
    assert read_filter.to_json() == _PIPELINE_FILTER.to_json()


def test_read_project_filter_detects_a_human_edit(tmp_path):
    session = Session(AnalysisConfig())
    wav_path = str(tmp_path / 'mono.wav')
    _write_mono_wav(wav_path)
    out_path = str(tmp_path / 'title.mono.beq')
    write_mono_project(session, wav_path, _PIPELINE_FILTER, out_path)

    _hand_edit_filter(out_path, _HUMAN_FILTER)
    read_filter, is_pure = read_project_filter(out_path)

    assert is_pure is False
    assert read_filter.to_json() == _HUMAN_FILTER.to_json()


def test_write_multichannel_project_enslaves_every_channel_to_the_master(tmp_path):
    session = Session(AnalysisConfig())
    wav_path = str(tmp_path / 'multi.wav')
    _write_multichannel_wav(wav_path, (1000, 2000, 3000, 4000, 5000, 6000))
    out_path = str(tmp_path / 'title.multichannel.beq')

    write_multichannel_project(session, wav_path, _PIPELINE_FILTER, '5.1', out_path)

    raw = _read_raw(out_path)
    signals = signalmodel_from_json(raw, None)
    master = signals[0]
    others = signals[1:]
    assert len(others) == 5
    assert all(s.master is master for s in others)
    assert {s.name for s in master.slaves} == {s.name for s in others}
    assert any(s.name.endswith('_LFE') for s in others)


# --- write_title_projects_if_safe --------------------------------------------

def test_write_title_projects_if_safe_skips_an_edited_target(tmp_path):
    session = Session(AnalysisConfig())
    wav_path = str(tmp_path / 'mono.wav')
    _write_mono_wav(wav_path)
    mono_out = str(tmp_path / 'title.mono.beq')
    write_mono_project(session, wav_path, _PIPELINE_FILTER, mono_out)
    _hand_edit_filter(mono_out, _HUMAN_FILTER)

    result = write_title_projects_if_safe(session, wav_path, _OTHER_HUMAN_FILTER, mono_out)

    assert result == {'mono': False, 'multichannel': None}
    read_filter, _ = read_project_filter(mono_out)
    assert read_filter.to_json() == _HUMAN_FILTER.to_json()


def test_write_title_projects_if_safe_writes_the_other_target_independently(tmp_path):
    session = Session(AnalysisConfig())
    mono_wav = str(tmp_path / 'mono.wav')
    _write_mono_wav(mono_wav)
    mc_wav = str(tmp_path / 'multi.wav')
    _write_multichannel_wav(mc_wav, (1000, 2000, 3000, 4000, 5000, 6000))
    mono_out = str(tmp_path / 'title.mono.beq')
    mc_out = str(tmp_path / 'title.multichannel.beq')

    write_title_projects_if_safe(session, mono_wav, _PIPELINE_FILTER, mono_out,
                                 multichannel_wav_path=mc_wav, channel_layout_name='5.1',
                                 multichannel_out_path=mc_out)
    _hand_edit_filter(mono_out, _HUMAN_FILTER)

    result = write_title_projects_if_safe(session, mono_wav, _OTHER_HUMAN_FILTER, mono_out,
                                          multichannel_wav_path=mc_wav, channel_layout_name='5.1',
                                          multichannel_out_path=mc_out)

    assert result == {'mono': False, 'multichannel': True}
    mono_filter, _ = read_project_filter(mono_out)
    assert mono_filter.to_json() == _HUMAN_FILTER.to_json()
    mc_filter, mc_is_pure = read_project_filter(mc_out)
    assert mc_is_pure is True
    assert mc_filter.to_json() == _OTHER_HUMAN_FILTER.to_json()


# --- resolve_published_filter -------------------------------------------------

def test_resolve_published_filter_prefers_the_edited_side(tmp_path):
    session = Session(AnalysisConfig())
    mono_wav = str(tmp_path / 'mono.wav')
    _write_mono_wav(mono_wav)
    mc_wav = str(tmp_path / 'multi.wav')
    _write_multichannel_wav(mc_wav, (1000, 2000, 3000, 4000, 5000, 6000))

    # case 1: mono edited, multichannel absent -> mono's edit wins
    mono_out_1 = str(tmp_path / 'a.mono.beq')
    write_mono_project(session, mono_wav, _PIPELINE_FILTER, mono_out_1)
    _hand_edit_filter(mono_out_1, _HUMAN_FILTER)

    filt, is_human_edit = resolve_published_filter(mono_out_1)

    assert is_human_edit is True
    assert filt.to_json() == _HUMAN_FILTER.to_json()

    # case 2: mono edited, multichannel still pure -> mono's edit still wins
    mono_out_2 = str(tmp_path / 'b.mono.beq')
    mc_out_2 = str(tmp_path / 'b.multichannel.beq')
    write_mono_project(session, mono_wav, _PIPELINE_FILTER, mono_out_2)
    write_multichannel_project(session, mc_wav, _PIPELINE_FILTER, '5.1', mc_out_2)
    _hand_edit_filter(mono_out_2, _HUMAN_FILTER)

    filt, is_human_edit = resolve_published_filter(mono_out_2, mc_out_2)

    assert is_human_edit is True
    assert filt.to_json() == _HUMAN_FILTER.to_json()

    # mirror case: multichannel edited, mono pure -> multichannel's edit wins
    mono_out_3 = str(tmp_path / 'c.mono.beq')
    mc_out_3 = str(tmp_path / 'c.multichannel.beq')
    write_mono_project(session, mono_wav, _PIPELINE_FILTER, mono_out_3)
    write_multichannel_project(session, mc_wav, _PIPELINE_FILTER, '5.1', mc_out_3)
    _hand_edit_filter(mc_out_3, _HUMAN_FILTER)

    filt, is_human_edit = resolve_published_filter(mono_out_3, mc_out_3)

    assert is_human_edit is True
    assert filt.to_json() == _HUMAN_FILTER.to_json()


def test_resolve_published_filter_raises_on_disagreeing_edits(tmp_path):
    session = Session(AnalysisConfig())
    mono_wav = str(tmp_path / 'mono.wav')
    _write_mono_wav(mono_wav)
    mc_wav = str(tmp_path / 'multi.wav')
    _write_multichannel_wav(mc_wav, (1000, 2000, 3000, 4000, 5000, 6000))
    mono_out = str(tmp_path / 'title.mono.beq')
    mc_out = str(tmp_path / 'title.multichannel.beq')
    write_mono_project(session, mono_wav, _PIPELINE_FILTER, mono_out)
    write_multichannel_project(session, mc_wav, _PIPELINE_FILTER, '5.1', mc_out)
    _hand_edit_filter(mono_out, _HUMAN_FILTER)
    _hand_edit_filter(mc_out, _OTHER_HUMAN_FILTER)

    with pytest.raises(ProjectFilterConflict) as exc_info:
        resolve_published_filter(mono_out, mc_out)

    assert exc_info.value.mono_filter.to_json() == _HUMAN_FILTER.to_json()
    assert exc_info.value.multichannel_filter.to_json() == _OTHER_HUMAN_FILTER.to_json()


def _both_projects(tmp_path):
    session = Session(AnalysisConfig())
    mono_wav = str(tmp_path / 'mono.wav')
    _write_mono_wav(mono_wav)
    mc_wav = str(tmp_path / 'multi.wav')
    _write_multichannel_wav(mc_wav, (1000, 2000, 3000, 4000, 5000, 6000))
    mono_out = str(tmp_path / 'title.mono.beq')
    mc_out = str(tmp_path / 'title.multichannel.beq')
    write_mono_project(session, mono_wav, _PIPELINE_FILTER, mono_out)
    write_multichannel_project(session, mc_wav, _PIPELINE_FILTER, '5.1', mc_out)
    return session, mono_wav, mc_wav, mono_out, mc_out


def test_resolve_published_projects_says_which_side_carried_the_edit(tmp_path):
    session, mono_wav, mc_wav, mono_out, mc_out = _both_projects(tmp_path)

    assert resolve_published_projects(mono_out, mc_out).edited_side is None
    assert resolve_published_projects(mono_out).edited_side is None

    _hand_edit_filter(mono_out, _HUMAN_FILTER)
    assert resolve_published_projects(mono_out, mc_out).edited_side == 'mono'
    assert resolve_published_projects(mono_out).edited_side == 'mono'

    _hand_edit_filter(mc_out, _HUMAN_FILTER)  # the same edit on both sides is not a conflict
    assert resolve_published_projects(mono_out, mc_out).edited_side == 'both'


def test_resolve_published_projects_names_the_multichannel_side(tmp_path):
    session, mono_wav, mc_wav, mono_out, mc_out = _both_projects(tmp_path)
    _hand_edit_filter(mc_out, _HUMAN_FILTER)

    published = resolve_published_projects(mono_out, mc_out)

    assert published.edited_side == 'multichannel'
    assert published.filter.to_json() == _HUMAN_FILTER.to_json()


def test_align_projects_writes_a_mono_edit_into_the_multichannel_project(tmp_path):
    session, mono_wav, mc_wav, mono_out, mc_out = _both_projects(tmp_path)
    _hand_edit_filter(mono_out, _HUMAN_FILTER)
    published = resolve_published_projects(mono_out, mc_out)

    aligned = align_projects(session, published, mono_out, mono_wav, mc_out, mc_wav, '5.1')

    assert aligned == ['multichannel']
    mc_filter, mc_pure = read_project_filter(mc_out)
    assert mc_filter.to_json() == _HUMAN_FILTER.to_json()
    assert mc_pure is True  # pipeline-written again, so a later regeneration is still allowed
    mono_filter, mono_pure = read_project_filter(mono_out)
    assert mono_filter.to_json() == _HUMAN_FILTER.to_json()
    assert mono_pure is False  # the human's own file is untouched
    # every channel of the rewritten multichannel project is still linked to its master
    raw = _read_raw(mc_out)
    assert raw[0]['slave_names'] and all(r['master_name'] == raw[0]['name'] for r in raw[1:])


def test_align_projects_writes_a_multichannel_edit_into_the_mono_project(tmp_path):
    session, mono_wav, mc_wav, mono_out, mc_out = _both_projects(tmp_path)
    _hand_edit_filter(mc_out, _HUMAN_FILTER)
    published = resolve_published_projects(mono_out, mc_out)

    aligned = align_projects(session, published, mono_out, mono_wav, mc_out, mc_wav, '5.1')

    assert aligned == ['mono']
    assert read_project_filter(mono_out)[0].to_json() == _HUMAN_FILTER.to_json()
    assert read_project_filter(mc_out)[1] is False  # the human's file is untouched


def test_align_projects_does_nothing_when_nothing_was_edited_or_there_is_no_sibling(tmp_path):
    session, mono_wav, mc_wav, mono_out, mc_out = _both_projects(tmp_path)

    assert align_projects(session, resolve_published_projects(mono_out, mc_out), mono_out, mono_wav,
                          mc_out, mc_wav, '5.1') == []

    _hand_edit_filter(mono_out, _HUMAN_FILTER)
    assert align_projects(session, resolve_published_projects(mono_out), mono_out, mono_wav) == []


def test_pipeline_publish_project_module_has_no_qtpy_import():
    import ast
    import pathlib
    source = (pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python' / 'pipeline' / 'publish' /
             'project.py').resolve()
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith('qtpy') for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith('qtpy')
