'''
The modules the headless pipeline reaches must import with no Qt installed: the pipeline service's Docker image has none
(design/pipeline-service/docker.md §10.1). Each is imported in a fresh interpreter whose import system refuses every Qt
package and the generated `ui` forms, so a Qt import anywhere down the chain fails the test with the chain in the message.

A subprocess, because this session's own QApplication and imports would hide an import made elsewhere (AGENTS.md
"Testing", gotcha 1).
'''
import os
import pathlib
import subprocess
import sys

import pytest

# Modules that must stay Qt-free. Add one here when the pipeline starts to use it.
QT_FREE = (
    'model.preferences',
    'model.limits',
    'model.dsp_type',
    'model.minidsp',
    'pipeline.publish.xml',
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
    importlib.import_module({module!r})
except ImportError:
    traceback.print_exc()
    sys.exit(1)
print('OK')
'''


def _env():
    env = dict(os.environ)
    for key in ('DISPLAY', 'WAYLAND_DISPLAY', 'QT_QPA_PLATFORM'):
        env.pop(key, None)
    env['PYTHONPATH'] = str((pathlib.Path(__file__).parent / '..' / '..' / 'main' / 'python').resolve())
    return env


def _import_without_qt(module: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, '-c', _SCRIPT.format(blocked=_BLOCKED, module=module)],
                          capture_output=True, text=True, timeout=60, env=_env())


@pytest.mark.parametrize('module', QT_FREE)
def test_module_imports_without_qt(module):
    result = _import_without_qt(module)
    assert result.returncode == 0 and 'OK' in result.stdout, result.stderr


def test_the_blocker_catches_a_qt_import():
    ''' The guard itself: a module that imports Qt must fail, or every test above passes vacuously. '''
    result = _import_without_qt('model.limits_dialog')
    assert result.returncode == 1 and 'Qt-free import reached' in result.stderr
