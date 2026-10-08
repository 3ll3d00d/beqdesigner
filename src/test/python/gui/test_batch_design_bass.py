'''
TODO R8: Batch Design (and Extract Audio's design step) send the app's crossover as the designer's bass management.

`import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import numpy as np
import soundfile as sf
from types import SimpleNamespace

from model.batch import DesignJob
from pipeline.config import AnalysisConfig
from pipeline.designer.contract import DesignResponse
from pipeline.designer.registry import register_designer
from pipeline.library.bass import DEFAULTS
from pipeline.orchestrate import Session
from pipeline.review import read_entry


def test_a_batch_design_sends_the_crossover_its_session_was_built_with(qtbot, tmp_path):
    seen = []
    register_designer('capture', lambda request: seen.append(request) or DesignResponse(
        '1.0', decline_reason='no_rolloff_detected', decline_message='nothing to restore'))
    wav = tmp_path / 'mono.wav'
    sf.write(str(wav), np.random.default_rng(1).normal(0, 0.05, (1000 * 5, 1)), 1000, subtype='PCM_24')
    session = Session(AnalysisConfig(target_fs=1000), bm_lpf_fs=110, bm_lpf_position='After')
    outcome = []
    candidate = SimpleNamespace(design_started=lambda: None, design_complete=outcome.append,
                                design_failed=lambda error: outcome.append(error))
    job = DesignJob(candidate, session, 'one', str(wav), None, 'unknown', str(wav), 0, str(tmp_path), 'capture',
                    str(tmp_path / 'queue'))

    job.run()

    assert seen[0].bass_management == {**DEFAULTS, 'lpf_fs': 110.0, 'lpf_position': 'After'}
    assert read_entry(str(tmp_path / 'queue'), 'one').bass_management == seen[0].bass_management
