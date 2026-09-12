'''
Phase 5's acceptance test (design/pipeline-implementation-plan.md, §11.2 of
design/api-headless-pipeline.md): the Ready Player One worked example from
docs/workflow/beq.md, scripted end to end through pipeline.orchestrate.Session
-- extract -> load -> design -> set_filters -> stats -> to_beq_xml -> report
-> publish -- using a fake in-process designer that returns the documented
filter values, asserting the documented numbers reproduce at every stage.

This is the closest thing to a full spec test for the pipeline: it composes
every phase 0-4 module through Session rather than re-testing any one of
them in isolation (each already has its own dedicated test file).
'''
import io
import subprocess
import wave

import numpy as np
import pytest
from PIL import Image

from model.iir import LowShelf, PeakingEQ
from model.minidsp import xml_to_filt
from pipeline.config import AnalysisConfig
from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.metadata import BeqMetadata
from pipeline.orchestrate import Applied, Declined, Session
from pipeline.publish.git import RepoTarget, current_branch

RP1_DESIGNER_NAME = 'acceptance.rp1'
RP1_DECLINE_DESIGNER_NAME = 'acceptance.decline'


def _rp1_design_response() -> DesignResponse:
    ''' docs/workflow/beq.md's Ready Player One values, in DesignResponse/BiquadSpec form. '''
    return DesignResponse(
        contract_version='1.0',
        candidates=[DesignCandidate(
            filters=[BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7) for _ in range(5)]
                   + [BiquadSpec(type='peaking_eq', freq_hz=40.0, gain_db=-3.0, q=2.0)],
            confidence=0.9,
            mv_adjust_db=4.0,  # docs/workflow/beq.md step 7: "we need to reduce by ~4dB"
            method='fitted',
        )],
    )


def _rp1_designer(request):
    return _rp1_design_response()


def _decline_designer(request):
    return DesignResponse(contract_version='1.0', decline_reason='no_coherent_rolloff',
                          decline_message='coherent band does not span the knee')


@pytest.fixture(autouse=True)
def _rp1_designers():
    register_designer(RP1_DESIGNER_NAME, _rp1_designer)
    register_designer(RP1_DECLINE_DESIGNER_NAME, _decline_designer)
    yield
    unregister_designer(RP1_DESIGNER_NAME)
    unregister_designer(RP1_DECLINE_DESIGNER_NAME)


def _write_synthetic_5_1_wav(path, fs=48000, duration_s=1.0, channel_values=(1000, 2000, 3000, 4000, 5000, 6000)):
    n_frames = int(fs * duration_s)
    frame = np.array(channel_values, dtype=np.int16)
    data = np.tile(frame, (n_frames, 1)).astype('<i2').tobytes()
    with wave.open(path, 'wb') as w:
        w.setnchannels(6)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(data)


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


def test_design_declines_produces_a_first_class_declined_outcome(tmp_path):
    '''
    §15.4: a decline is a value, not an exception -- and nothing downstream
    runs on it. This test only asserts the outcome; it deliberately does not
    call set_filters/to_beq_xml/publish afterwards, because a Declined
    outcome carries nothing they could operate on.
    '''
    session = Session(AnalysisConfig())
    wav_path = str(tmp_path / 'synthetic.wav')
    _write_synthetic_5_1_wav(wav_path)
    extracted = session.extract(wav_path, str(tmp_path / 'extracted'))
    sig = session.load(extracted)

    outcome = session.design(sig, RP1_DECLINE_DESIGNER_NAME)

    assert isinstance(outcome, Declined)
    assert outcome.reason == 'no_coherent_rolloff'
    assert sig.filter is None or len(list(sig.filter)) == 0


def test_ready_player_one_end_to_end(tmp_path):
    session = Session(AnalysisConfig())

    # 1: extract + load, headlessly
    source_wav = str(tmp_path / 'source.wav')
    _write_synthetic_5_1_wav(source_wav)
    extracted = session.extract(source_wav, str(tmp_path / 'extracted'))
    sig = session.load(extracted)
    assert sig.signal.fs == 1000  # AnalysisConfig() default target_fs

    # 2: design -- Applied, not Declined, and matches the documented RP1 filter values
    outcome = session.design(sig, RP1_DESIGNER_NAME)
    assert isinstance(outcome, Applied)
    assert outcome.confidence == 0.9
    assert outcome.method == 'fitted'
    assert outcome.mv_adjust_db == 4.0

    filters = list(outcome.filters)
    shelves = [f for f in filters if isinstance(f, LowShelf)]
    peaks = [f for f in filters if isinstance(f, PeakingEQ)]
    assert len(shelves) == 1 and shelves[0].count == 5
    assert shelves[0].freq == 18.0 and shelves[0].gain == 4.5
    assert shelves[0].gain * shelves[0].count == 22.5
    assert len(peaks) == 1 and peaks[0].freq == 40.0 and peaks[0].gain == -3.0

    # apply it, and confirm curves/stats now reflect the filtered signal
    session.set_filters(sig, outcome.filters)
    unfiltered_avg = session.curves(sig, kind='avg', filtered=False)
    filtered_avg = session.curves(sig, kind='avg', filtered=True)
    assert filtered_avg.name.startswith(unfiltered_avg.name)
    assert not np.array_equal(unfiltered_avg.y, filtered_avg.y)

    stats_before = session.stats(sig, filtered=False)
    stats_after = session.stats(sig, filtered=True)
    assert stats_before.fs == 1000 and stats_after.fs == 1000

    # 3: metadata + XML
    meta = BeqMetadata(title='Ready Player One', year='2018', audio_types=['Atmos'],
                       genres=[{'id': 28, 'name': 'Action'}, {'id': 878, 'name': 'Science Fiction'}],
                       gain=f"{outcome.mv_adjust_db:+g}")
    xml = session.to_beq_xml(outcome.filters, meta)
    assert '<beq_title>Ready Player One</beq_title>' in xml
    assert '<genre id="28">Action</genre>' in xml

    written = str(tmp_path / 'rp1_check.xml')
    with open(written, 'w', encoding='utf-8') as f:
        f.write(xml)
    read_back = xml_to_filt(written, fs=96000)
    back_shelves = [f for f in read_back if isinstance(f, LowShelf)]
    back_peaks = [f for f in read_back if isinstance(f, PeakingEQ)]
    assert len(back_shelves) == 1 and back_shelves[0].count == 5
    assert back_shelves[0].gain * back_shelves[0].count == 22.5
    assert len(back_peaks) == 1

    # 4: report -- a real, correctly-sized PNG, headlessly
    report_png = session.report([unfiltered_avg, filtered_avg], outcome.filters, meta=meta,
                                mv_offset=outcome.mv_adjust_db)
    report_image = Image.open(io.BytesIO(report_png))
    assert report_image.format == 'PNG'

    # 5: publish -- real local git repos (a bare "remote" + a working clone each),
    # proving Session.publish() sequences image-push -> URL -> meta -> XML -> XML-push
    xml_repo, xml_bare = _init_repo_with_remote(tmp_path, 'xml_repo')
    images_repo, images_bare = _init_repo_with_remote(tmp_path, 'images_repo')

    result = session.publish(outcome.filters, meta, xml_repo, 'xml/ready-player-one.xml',
                             images_repo=images_repo, image_relative_path='img/ready-player-one.png',
                             image_png=report_png, image_owner='3ll3d00d', image_repo_name='beq-images')

    branch = current_branch(images_repo)
    assert result['image_url'] == f'https://raw.githubusercontent.com/3ll3d00d/beq-images/{branch}/img/ready-player-one.png'
    assert meta.spectrum_url == result['image_url']
    assert meta.pva_url == result['image_url']
    assert '<beq_spectrumURL>' in result['xml']

    xml_on_remote = subprocess.run(
        ['git', '-C', str(xml_bare), 'cat-file', '-p', f"{result['xml_commit']}:xml/ready-player-one.xml"],
        check=True, capture_output=True, text=True).stdout
    assert xml_on_remote == result['xml']


def test_session_headless_run_constructs_no_qapplication():
    '''
    Deliberately a subprocess test (see test_pipeline_qt_boundary.py's
    docstring): a real Qt widget test elsewhere in the same pytest session
    (e.g. src/test/python/gui/) legitimately constructs a QApplication via
    pytest-qt's qapp fixture, which then persists for the rest of that
    process -- an in-process assert here would fail depending on test
    order/selection, not on anything Session actually did.
    '''
    import subprocess
    import sys

    script = (
        "import tempfile, os\n"
        "tmp_path = tempfile.mkdtemp()\n"
        "from pipeline.config import AnalysisConfig\n"
        "from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse\n"
        "from pipeline.designer.registry import register_designer\n"
        "from pipeline.metadata import BeqMetadata\n"
        "from pipeline.orchestrate import Session\n"
        "import numpy as np, wave\n"
        "def rp1_designer(request):\n"
        "    return DesignResponse(contract_version='1.0', candidates=[DesignCandidate(\n"
        "        filters=[BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7) for _ in range(5)]\n"
        "               + [BiquadSpec(type='peaking_eq', freq_hz=40.0, gain_db=-3.0, q=2.0)],\n"
        "        confidence=0.9, mv_adjust_db=4.0, method='fitted')])\n"
        "register_designer('acceptance.rp1', rp1_designer)\n"
        "source_wav = os.path.join(tmp_path, 'source.wav')\n"
        "fs = 48000\n"
        "n_frames = fs\n"
        "frame = np.array([1000, 2000, 3000, 4000, 5000, 6000], dtype=np.int16)\n"
        "data = np.tile(frame, (n_frames, 1)).astype('<i2').tobytes()\n"
        "with wave.open(source_wav, 'wb') as w:\n"
        "    w.setnchannels(6); w.setsampwidth(2); w.setframerate(fs); w.writeframes(data)\n"
        "session = Session(AnalysisConfig())\n"
        "extracted = session.extract(source_wav, os.path.join(tmp_path, 'extracted'))\n"
        "sig = session.load(extracted)\n"
        "outcome = session.design(sig, 'acceptance.rp1')\n"
        "session.set_filters(sig, outcome.filters)\n"
        "session.stats(sig, filtered=True)\n"
        "meta = BeqMetadata(title='Ready Player One', year='2018', audio_types=['Atmos'])\n"
        "session.to_beq_xml(outcome.filters, meta)\n"
        "session.report([session.curves(sig)], outcome.filters, meta=meta)\n"
        "from qtpy.QtWidgets import QApplication\n"
        "assert QApplication.instance() is None, 'Session run constructed a QApplication'\n"
        "print('OK')\n"
    )
    env = _env_without_display()
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=30, env=env)
    assert result.returncode == 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert 'OK' in result.stdout


def _env_without_display():
    import os
    import pathlib
    env = dict(os.environ)
    for key in ('DISPLAY', 'WAYLAND_DISPLAY', 'QT_QPA_PLATFORM'):
        env.pop(key, None)
    src_main = str((pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python').resolve())
    env['PYTHONPATH'] = src_main
    return env


def test_pipeline_orchestrate_module_has_no_qtpy_import():
    import ast
    import pathlib
    source = (pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python' / 'pipeline' / 'orchestrate.py').resolve()
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith('qtpy') for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith('qtpy')
