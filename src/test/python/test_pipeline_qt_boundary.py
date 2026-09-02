'''
Phase 1 (B1) of design/pipeline-implementation-plan.md: importing model.iir
(the filter maths every pipeline step depends on) must not have the side
effect of reading the desktop user's real QSettings. Before the fix,
model/xy.py:15 constructed a `Preferences(QSettings(...))` at import time and
installed it as a process-global singleton -- merely importing model.iir
silently read and cached the interactive user's real preferences.

This is deliberately a subprocess test: model.iir (and model.xy, which it
imports) may already be imported by something else in the same test session,
so asserting "not yet imported" only means anything in a fresh interpreter.
'''
import subprocess
import sys


def test_importing_iir_does_not_construct_preferences_singleton():
    script = (
        "import model.iir\n"
        "import model.xy as xy\n"
        "assert xy.__dict__.get('__preferences') is None, "
        "'model.xy constructed its lazy Preferences singleton on import alone'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True, text=True, timeout=30,
        env=_env_without_display(),
    )
    assert result.returncode == 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'
    assert 'OK' in result.stdout


def _env_without_display():
    import os
    env = dict(os.environ)
    for key in ('DISPLAY', 'WAYLAND_DISPLAY', 'QT_QPA_PLATFORM'):
        env.pop(key, None)
    import pathlib
    src_main = str((pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python').resolve())
    env['PYTHONPATH'] = src_main
    return env
