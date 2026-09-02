'''
Phase 1 (B5) of design/pipeline-implementation-plan.md: signal_stats(),
extracted from model/waveform.py's __recalc_stats (+ the model/analysis.py
and model/extract.py duplicates).
'''
import math

import numpy as np


def test_signal_stats_known_signal():
    from pipeline.stats import signal_stats

    fs = 1000
    # a full-scale square wave: peak == rms == 1.0, so crest factor is 0 dB
    # and headroom is 0 dB (no margin left before clipping).
    samples = np.ones(fs, dtype=np.float64)
    stats = signal_stats(samples, fs)

    assert stats.fs == fs
    assert math.isclose(stats.peak, 1.0)
    assert math.isclose(stats.rms, 0.0, abs_tol=1e-9)
    assert math.isclose(stats.crest, 0.0, abs_tol=1e-9)
    assert math.isclose(stats.headroom, 0.0, abs_tol=1e-9)


def test_signal_stats_half_scale_sine_has_positive_headroom():
    from pipeline.stats import signal_stats

    fs = 48000
    t = np.arange(fs) / fs
    samples = 0.5 * np.sin(2 * np.pi * 100 * t)
    stats = signal_stats(samples, fs)

    assert math.isclose(stats.peak, 0.5, rel_tol=1e-6)
    # RMS of a sine at amplitude A is A/sqrt(2)
    assert math.isclose(stats.rms, 20 * math.log10(0.5 / math.sqrt(2)), rel_tol=1e-3)
    # 6 dB of headroom (peak at 0.5 of full scale)
    assert math.isclose(stats.headroom, 20 * math.log10(2.0), rel_tol=1e-3)


def test_pipeline_stats_module_has_no_qtpy_import():
    import ast
    import pathlib
    source = (pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python' / 'pipeline' / 'stats.py').resolve()
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith('qtpy') for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith('qtpy')
