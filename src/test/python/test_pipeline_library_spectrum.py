'''
pipeline/library/spectrum.py -- the publication heatmap of one design, shared by the title page and the service
(design/web-parity.md WP0/WP2). The page's use of it is gui/test_worklist_spectrum.py; this is the module with no widgets.
'''
import io

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from pipeline.config import AnalysisConfig
from pipeline.library.spectrum import comparison_request, render_comparison
from pipeline.publish.heatmap import HeatmapSpec
from pipeline.review import read_entry
from review_entry_fixture import write_entry


def _title(tmp_path):
    write_entry(str(tmp_path / 'queue'), 'r-alien')
    directory = tmp_path / 'work' / 'r-alien'
    directory.mkdir(parents=True)
    t = np.arange(8000) / 1000
    sf.write(directory / 'mono.wav', 0.15 * np.sin(2 * np.pi * 25 * t), 1000, subtype='FLOAT')
    return read_entry(str(tmp_path / 'queue'), 'r-alien')


def test_a_request_takes_the_heatmap_spec_without_desktop_preferences_and_renders_a_png(tmp_path):
    entry = _title(tmp_path)
    spec = HeatmapSpec()
    request = comparison_request(str(tmp_path / 'work'), entry, 0, 'Alien', AnalysisConfig(), spec=spec)
    assert request.spec is spec and request.mono_wav.endswith('mono.wav')
    png, note = render_comparison(request)
    assert Image.open(io.BytesIO(png)).format == 'PNG' and note.startswith('Selected design')


@pytest.mark.parametrize('picked, work, says', [(5, 'work', 'No design to compare'), (0, '', 'No work directory'),
                                                (0, 'elsewhere', 'No extracted mono audio')])
def test_what_cannot_be_compared_says_why(tmp_path, picked, work, says):
    entry = _title(tmp_path)
    with pytest.raises(ValueError, match=says):
        comparison_request(str(tmp_path / work) if work else '', entry, picked, 'Alien', AnalysisConfig(), spec=HeatmapSpec())


# --- the spectrogram grid (WP2) -----------------------------------------------------------------------------------------

def test_reducing_keeps_the_loudest_of_each_group_so_a_short_hit_survives():
    from pipeline.library.spectrum import _max_pool
    values = np.full((2, 10), -80.0)
    values[1, 7] = -3.0                                    # one loud instant
    pooled, run = _max_pool(values, 1, 3)
    assert run == 4 and pooled.shape == (2, 3) and pooled[1].tolist() == [-80.0, -3.0, -80.0]
    same, run = _max_pool(values, 1, 20)
    assert run == 1 and same is values


def test_a_grid_is_at_most_the_cells_asked_for_and_round_trips(tmp_path):
    from pipeline.library.spectrum import GRID_RANGE_DB, SpectrogramGrid, spectrogram_grid
    entry = _title(tmp_path)
    request = comparison_request(str(tmp_path / 'work'), entry, 0, 'Alien', AnalysisConfig(), spec=HeatmapSpec())
    grid = spectrogram_grid(request, filtered=False, max_hz=50, max_times=4, max_freqs=8)
    assert grid.times <= 4 and grid.freqs <= 8 and len(grid.cells) == grid.times * grid.freqs
    assert grid.step_db == pytest.approx(GRID_RANGE_DB / 254) and max(grid.cells) == 255
    back = SpectrogramGrid.decode(grid.encode())
    assert (back.times, back.freqs, back.cells) == (grid.times, grid.freqs, grid.cells)
    assert back.df == pytest.approx(grid.df) and back.floor_db == pytest.approx(grid.floor_db, rel=1e-6)
    with pytest.raises(ValueError, match='not a version 1'):
        SpectrogramGrid.decode(b'NOPE' + grid.encode()[4:])
