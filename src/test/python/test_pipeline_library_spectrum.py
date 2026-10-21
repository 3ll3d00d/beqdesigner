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
