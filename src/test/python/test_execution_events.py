from types import SimpleNamespace

import pytest

from model.execution_events import emit_execution_event, event_scope, execution_event_context
from model.ffmpeg import Executor
from pipeline.publish.git import RepoTarget, _git_raw


def test_execution_event_context_attaches_run_title_and_stage():
    seen = []

    with execution_event_context('run-1', seen.append):
        with event_scope(title_id='title-1', stage='extract'):
            emit_execution_event('command_finished', command=('ffmpeg', '-i', 'source.mkv'),
                                 stdout='out', stderr='warn', exit_code=0)

    assert len(seen) == 1
    event = seen[0]
    assert (event.run_id, event.title_id, event.stage, event.kind) == ('run-1', 'title-1', 'extract',
                                                                       'command_finished')
    assert event.command == ('ffmpeg', '-i', 'source.mkv')
    assert (event.stdout, event.stderr, event.exit_code) == ('out', 'warn', 0)
    assert event.timestamp > 0


def test_executor_reports_command_and_response(monkeypatch):
    class Command:
        def compile(self, *, overwrite_output):
            assert overwrite_output is True
            return ['ffmpeg', '-i', 'source.mkv', 'mono.wav']

        def run(self, **kwargs):
            assert kwargs == {'overwrite_output': True, 'quiet': True}
            return b'audio output', b'ffmpeg warning'

    executor = Executor.__new__(Executor)
    executor._Executor__ffmpeg_cmd = Command()
    executor._Executor__is_remux = False
    executor._Executor__progress_handler = None
    executor._Executor__progress_port = 12001
    seen = []

    with execution_event_context('run-2', seen.append), event_scope(title_id='title-2', stage='extract'):
        assert executor.run_sync() == (b'audio output', b'ffmpeg warning')

    assert [event.kind for event in seen] == ['command_started', 'command_finished']
    assert seen[0].command == ('ffmpeg', '-i', 'source.mkv', 'mono.wav')
    assert (seen[1].stdout, seen[1].stderr, seen[1].exit_code) == ('audio output', 'ffmpeg warning', 0)


@pytest.mark.parametrize('failure', [TypeError('invalid ffmpeg option'), OSError('could not compile command')])
def test_executor_reports_command_preparation_failure(failure):
    class Command:
        def compile(self, *, overwrite_output):
            raise failure

    executor = Executor.__new__(Executor)
    executor._Executor__ffmpeg_cmd = Command()
    executor._Executor__is_remux = False
    executor._Executor__progress_handler = None
    executor._Executor__progress_port = 12001
    seen = []

    with execution_event_context('run-2', seen.append), event_scope(title_id='title-2', stage='extract'):
        with pytest.raises(type(failure), match=str(failure)):
            executor.run_sync()

    assert [(event.title_id, event.stage, event.kind) for event in seen] == [
        ('title-2', 'extract', 'command_preparation_failed')]
    assert seen[0].message == f'Could not prepare ffmpeg command: {type(failure).__name__}: {failure}'


def test_git_command_reports_command_and_result(monkeypatch):
    seen = []
    monkeypatch.setattr('pipeline.publish.git.subprocess.run', lambda *a, **k: SimpleNamespace(
        returncode=0, stdout='main\n', stderr=''))

    with execution_event_context('run-3', seen.append), event_scope(title_id='title-3', stage='publish'):
        assert _git_raw(RepoTarget('/repo'), 'branch', '--show-current') == 'main\n'

    assert [event.kind for event in seen] == ['command_started', 'command_finished']
    assert seen[0].command == ('git', '--literal-pathspecs', '-C', '/repo', 'branch', '--show-current')
    assert (seen[1].stdout, seen[1].stderr, seen[1].exit_code) == ('main\n', '', 0)
