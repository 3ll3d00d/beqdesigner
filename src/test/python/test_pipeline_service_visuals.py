'''
A title's heatmap and spectrogram over HTTP (design/web-parity.md WP2): rendered off the request (202, then the file),
kept in the title's work folder under a name of what it depends on, and a failed render said once.
'''
import io
import os
import struct
import threading
import time

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from pipeline.library.spectrum import GRID_HEADER, SpectrogramGrid
from pipeline.library.workdir import entry_directory
from pipeline.service import visuals
from test_pipeline_service_review import AUTH, Gate, service, setup  # noqa: F401 (fixtures)


@pytest.fixture
def audio(setup):
    _, work, _ = setup
    folder = entry_directory(str(work), 'a')
    os.makedirs(folder, exist_ok=True)
    t = np.arange(1000 * 30) / 1000
    sf.write(os.path.join(folder, 'mono.wav'), 0.1 * np.sin(2 * np.pi * 25 * t) + 0.05 * np.sin(2 * np.pi * 60 * t),
             1000, subtype='FLOAT')
    return folder


def _until_drawn(client, path, **params):
    first = client.get(path, params=params, headers=AUTH)
    assert first.status_code == 202 and first.headers['Retry-After'] == str(visuals.RETRY_SECONDS)
    for _ in range(300):
        answer = client.get(path, params=params, headers=AUTH)
        if answer.status_code != 202:
            return answer
        time.sleep(0.05)
    pytest.fail('never drawn')


def test_the_heatmap_is_drawn_off_the_request_then_served_from_the_title_folder(audio, service):
    client, *_ = service
    answer = _until_drawn(client, '/v1/titles/a/heatmap.png', candidate=1)
    assert answer.status_code == 200, answer.text
    assert Image.open(io.BytesIO(answer.content)).format == 'PNG'
    kept = os.listdir(os.path.join(audio, 'web'))
    assert len(kept) == 1 and kept[0].startswith('heatmap-')
    again = client.get('/v1/titles/a/heatmap.png', params={'candidate': 1}, headers=AUTH)
    assert again.status_code == 200 and again.content == answer.content
    assert client.get('/v1/titles/a/heatmap.png', params={'candidate': 0}, headers=AUTH).status_code == 202  # another design


def test_the_spectrogram_is_a_byte_grid_before_and_after_the_design(audio, service):
    client, *_ = service
    raw = SpectrogramGrid.decode(_until_drawn(client, '/v1/titles/a/spectrogram', candidate=0, max_hz=100).content)
    after = SpectrogramGrid.decode(_until_drawn(client, '/v1/titles/a/spectrogram', candidate=0, filtered='true',
                                                max_hz=100).content)
    assert raw.freqs == after.freqs and raw.times == after.times and raw.times > 1
    assert raw.f0 == 0 and raw.f0 + raw.df * (raw.freqs - 1) <= 100
    cells = np.frombuffer(raw.cells, dtype=np.uint8).reshape(raw.times, raw.freqs)
    tone = int(round(25 / raw.df))
    assert cells[:, tone].mean() > cells[:, int(round(80 / raw.df))].mean()   # the 25 Hz tone is the loud one
    assert raw.cells != after.cells                                           # the low shelf lifts it
    assert struct.calcsize(GRID_HEADER) == 40


@pytest.mark.parametrize('candidate, status', [(9, 422)])
def test_a_design_not_offered_is_refused(audio, service, candidate, status):
    client, *_ = service
    answer = client.get('/v1/titles/a/heatmap.png', params={'candidate': candidate}, headers=AUTH)
    assert answer.status_code == status


def test_a_title_with_no_extracted_audio_says_so(service):
    client, *_ = service
    answer = client.get('/v1/titles/b/spectrogram', headers=AUTH)
    assert (answer.status_code, answer.json()['title']) == (409, 'Nothing to draw')
    assert 'No extracted mono audio' in answer.json()['detail']


def test_a_render_that_fails_says_why_once_then_may_be_asked_again(audio, service, monkeypatch):
    client, *_ = service
    monkeypatch.setattr(visuals, 'render_comparison', lambda request: (_ for _ in ()).throw(ValueError('no projects')))
    failed = _until_drawn(client, '/v1/titles/a/heatmap.png')
    assert (failed.status_code, failed.json()['detail']) == (422, 'no projects')
    assert client.get('/v1/titles/a/heatmap.png', headers=AUTH).status_code == 202


def test_the_queue_makes_each_file_once_while_it_is_asked_for(tmp_path):
    queue, gate, calls = visuals.RenderQueue(), threading.Event(), []

    def make():
        calls.append(1)
        gate.wait(5)
        return b'drawn'
    path = str(tmp_path / 'web' / 'x.bin')
    for _ in range(3):
        with pytest.raises(visuals.Pending):
            queue.get(path, make)
    gate.set()
    for _ in range(100):
        try:
            assert queue.get(path, make) == b'drawn'
            break
        except visuals.Pending:
            time.sleep(0.02)
    assert calls == [1] and open(path, 'rb').read() == b'drawn'
    queue.stop()
