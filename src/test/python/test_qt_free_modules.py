'''
The headless pipeline must import and run with no Qt installed: the pipeline service's Docker image has none
(design/pipeline-service/docker.md §10.1). Every module of `pipeline/`, and the `model/` modules it uses, is imported in a fresh
interpreter whose import system refuses every Qt package and the generated `ui` forms, so a Qt import anywhere down the chain
fails the test with the chain in the message; and a whole extract-load-design-publish run is made the same way.

A subprocess, because this session's own QApplication and imports would hide an import made elsewhere (AGENTS.md
"Testing", gotcha 1).
'''
import os
import pathlib
import subprocess
import sys

import pytest

_SRC = (pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python').resolve()


def _pipeline_modules():
    ''' Every module under pipeline/, found on disk (importing them to list them would defeat the test). '''
    for path in sorted((_SRC / 'pipeline').rglob('*.py')):
        parts = path.relative_to(_SRC).with_suffix('').parts
        yield '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)


# The model/ modules the pipeline relies on, named so a failure points at the module itself. Add one when the pipeline starts
# to use it.
QT_FREE_MODEL = (
    'model.preferences',
    'model.limits',
    'model.dsp_type',
    'model.minidsp',
    'model.ffmpeg',
    'model.signal',
)

_BLOCKED = ('qtpy', 'PyQt6', 'PyQt5', 'PySide6', 'qtawesome', 'pyqtgraph', 'ui')

_SCRIPT = '''
import importlib, sys, traceback

class _NoQt:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {blocked!r}:
            raise ImportError(f'Qt-free import reached {{name}}')

sys.meta_path.insert(0, _NoQt())
try:
{body}
except ImportError:
    traceback.print_exc()
    sys.exit(1)
print('OK')
'''


def _env():
    env = dict(os.environ)
    for key in ('DISPLAY', 'WAYLAND_DISPLAY', 'QT_QPA_PLATFORM'):
        env.pop(key, None)
    env['PYTHONPATH'] = str(_SRC)
    return env


def _run_without_qt(code: str) -> subprocess.CompletedProcess:
    body = '\n'.join('    ' + line for line in code.strip().splitlines())
    return subprocess.run([sys.executable, '-c', _SCRIPT.format(blocked=_BLOCKED, body=body)],
                          capture_output=True, text=True, timeout=60, env=_env())


def _import_without_qt(module: str) -> subprocess.CompletedProcess:
    return _run_without_qt(f'importlib.import_module({module!r})')


@pytest.mark.parametrize('module', (*_pipeline_modules(), *QT_FREE_MODEL))
def test_module_imports_without_qt(module):
    result = _import_without_qt(module)
    assert result.returncode == 0 and 'OK' in result.stdout, result.stderr


def test_the_blocker_catches_a_qt_import():
    ''' The guard itself: a module that imports Qt must fail, or every test above passes vacuously. '''
    result = _import_without_qt('model.limits_dialog')
    assert result.returncode == 1 and 'Qt-free import reached' in result.stderr


_SESSION_RUN = """
import os, tempfile, wave
import numpy as np
from pipeline.config import AnalysisConfig
from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse
from pipeline.designer.registry import register_designer
from pipeline.metadata import BeqMetadata
from pipeline.orchestrate import Session

def designer(request):
    return DesignResponse(contract_version='1.0', candidates=[DesignCandidate(
        filters=[BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=4.5, q=0.7)], confidence=0.9, mv_adjust_db=4.0,
        method='fitted')])

register_designer('no-qt', designer)
tmp = tempfile.mkdtemp()
source = os.path.join(tmp, 'source.wav')
with wave.open(source, 'wb') as w:     # one second of 5.1 at 48 kHz
    w.setnchannels(6); w.setsampwidth(2); w.setframerate(48000)
    w.writeframes(np.tile(np.array([1000, 2000, 3000, 4000, 5000, 6000], dtype='<i2'), (48000, 1)).tobytes())
session = Session(AnalysisConfig())
sig = session.load(session.extract(source, os.path.join(tmp, 'extracted')))
channels = session.load_channels(source)
outcome = session.design(sig, 'no-qt')
session.set_filters(sig, outcome.filters)
session.stats(sig, filtered=True)
meta = BeqMetadata(title='Ready Player One', year='2018', audio_types=['Atmos'])
session.to_beq_xml(outcome.filters, meta)
session.report([session.curves(sig)], outcome.filters, meta=meta)
"""


@pytest.mark.requires_ffmpeg
def test_a_session_extracts_loads_designs_and_publishes_without_qt():
    result = _run_without_qt(_SESSION_RUN)
    assert result.returncode == 0 and 'OK' in result.stdout, result.stderr
