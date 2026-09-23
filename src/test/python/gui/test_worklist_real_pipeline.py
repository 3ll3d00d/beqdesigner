'''
model/worklist.py, chunk 26b review fix: Publish and then Commit driven from the window through the REAL `run_stages`, the
real discovery index (a scan of a fake source over real outputs) and real git repositories (a bare "remote" and a working
clone each), nothing faked but the source's listing. The other tests in this folder hand the window a fake pipeline; this is
the one that proves the window and the pipeline agree about what a title needs, what a run did and what to say about it.

The outputs (an extracted manifest, an accepted queue entry) are built with the helpers of `test_pipeline_library_index.py`
and `test_pipeline_library_commit.py`, which are what the pipeline's own tests scan. `import ui.beq` first: AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import json
import os
import subprocess
import time
from types import SimpleNamespace
from typing import List

import pytest
from qtpy.QtCore import QSettings, Qt, QTimer
from qtpy.QtWidgets import QApplication

from model.preferences import DESIGNER_QUEUE_DIR, LIBRARY_FILESYSTEM_GLOBS, LIBRARY_IMAGES_REPO, LIBRARY_PROFILE_PATH, \
    LIBRARY_WORK_DIR, LIBRARY_XML_REPO, Preferences
from model.worklist import WorkListWindow
from model.worklist_confirm import ConfirmDialog
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.library.stages import run_stages
from pipeline.publish.git import RepoTarget
from test_pipeline_library_commit import OWNER, IMAGES_NAME, _repo, repos  # noqa: F401 (a fixture)
from test_pipeline_library_index import DESIGNER, FakeSource, _entry, _extracted, _item, _ready
from test_pipeline_publish_project import _write_mono_wav

XML_DIR, IMAGE_DIR = 'xml', 'img'


@pytest.fixture(autouse=True)
def _designer():
    register_designer(DESIGNER, lambda request: None)
    yield
    unregister_designer(DESIGNER)


def _git(*args) -> str:
    return subprocess.run(['git', *args], check=True, capture_output=True, text=True).stdout


def _commit_count(bare) -> int:
    return int(_git('-C', str(bare), 'rev-list', '--count', 'HEAD'))


def _subjects(bare) -> List[str]:
    return _git('-C', str(bare), 'log', '--format=%s').splitlines()


def _files_on_remote(bare) -> List[str]:
    return sorted(_git('-C', str(bare), 'ls-tree', '-r', '--name-only', 'HEAD').split())


def _profile_file(tmp_path, xml, images) -> str:
    ''' A profile file, because it is the one place the images repository's owner and name can be given. '''
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps({
        'sources': [{'name': 'films', 'kind': 'filesystem', 'globs': [str(tmp_path / 'films' / '*.mkv')]}],
        'run': {'work_dir': str(tmp_path / 'work'), 'queue_dir': str(tmp_path / 'queue'), 'designer': DESIGNER},
        'sync': {'xml_repo': xml, 'xml_dir': XML_DIR, 'images_repo': images, 'image_dir': IMAGE_DIR,
                 'image_owner': OWNER, 'image_repo_name': IMAGES_NAME}}))
    return str(path)


def _prefs(tmp_path, **more) -> Preferences:
    prefs = Preferences(QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat))
    for key, value in more.items():
        prefs.set(key, value)
    return prefs


def _world(tmp_path, *names):
    ''' The outputs a design and an accept leave for each title, in the work and queue directories a profile names. '''
    world = SimpleNamespace(work=str(tmp_path / 'work'), queue=str(tmp_path / 'queue'))
    os.makedirs(world.work, exist_ok=True)
    world.items = [_item(name) for name in names]
    for item in world.items:
        _extracted(world, item)
        _write_mono_wav(os.path.join(world.work, item.id, 'mono.wav'))     # publish reads the project's audio
        _entry(world, item, status='accepted')
    return world


def _open(qtbot, prefs, items, source='films', run_stages_fn=run_stages):
    window = WorkListWindow(None, prefs, sources={source: FakeSource(items)}, auto_scan=False, clock=time.time,
                            run_stages_fn=run_stages_fn)
    qtbot.addWidget(window)
    window.show()
    with qtbot.waitSignal(window.scan_finished, timeout=30000):
        window.rescan()
    return window


def _capture_pipeline_events(events):
    '''Wrap the real pipeline while preserving the window's event signal path.'''
    def runner(*args, on_event, **kwargs):
        def capture(event):
            events.append(event)
            on_event(event)
        return run_stages(*args, on_event=capture, **kwargs)
    return runner


def _record_run_counts(window):
    snapshots = []
    update = window._update_run_summary

    def record():
        update()
        snapshots.append(dict(window._run_outcomes))

    window._update_run_summary = record
    return snapshots


def _answer(accept: bool, seen: list, tick=None) -> None:
    def respond():
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, ConfirmDialog), dialog
        seen.append(dialog.text)
        if tick is not None:
            dialog.checkbox.setChecked(tick)
        (dialog.ok_button if accept else dialog.cancel_button).click()

    QTimer.singleShot(0, respond)


def _click(qtbot, button):
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)


def _needs(window):
    return {row.id: row.needs for row in window.model.rows}


def test_publish_then_commit_and_push_through_the_real_pipeline_one_commit_per_repository(qtbot, tmp_path, repos):
    xml, xml_bare, images, images_bare = repos
    _ready(repos)
    before = (_commit_count(xml_bare), _commit_count(images_bare))
    world = _world(tmp_path, 'a', 'b')
    prefs = _prefs(tmp_path, **{LIBRARY_PROFILE_PATH: _profile_file(tmp_path, xml.local_path, images.local_path)})
    events = []
    window = _open(qtbot, prefs, world.items, run_stages_fn=_capture_pipeline_events(events))
    assert _needs(window) == {'fs-a': 'publish', 'fs-b': 'publish'}
    assert window.publishButton.text() == 'Publish 2' and window.commitButton.text() == 'Commit 0'

    # Publish: the confirmation names both repositories; the files are written and nothing is committed
    seen: list = []
    _answer(True, seen)
    with qtbot.waitSignal(window.run_finished, timeout=60000) as published:
        _click(qtbot, window.publishButton)

    assert seen[0].startswith('Publish 2 titles?')
    assert xml.local_path in seen[0] and images.local_path in seen[0]
    assert sorted(p['id'] for p in published.args[0].published) == ['fs-a', 'fs-b'] and not published.args[0].publish_errors
    assert window.runStatusLabel.text() == 'Publish finished: 2 published'
    assert sorted((l.title, l.outcome) for l in window.results) == [('Film a', 'Published'), ('Film b', 'Published')]
    assert _needs(window) == {'fs-a': 'commit', 'fs-b': 'commit'}          # moved on, without a rescan
    assert (window.publishButton.text(), window.commitButton.text()) == ('Publish 0', 'Commit 2')
    assert os.path.isfile(os.path.join(xml.local_path, XML_DIR, 'fs-a.json'))
    assert (_commit_count(xml_bare), _commit_count(images_bare)) == before   # nothing committed or pushed yet

    # Commit with push ticked: one commit per repository, the remotes receive them
    seen = []
    events.clear()
    count_snapshots = _record_run_counts(window)
    _answer(True, seen, tick=True)
    with qtbot.waitSignal(window.run_finished, timeout=60000) as committed:
        _click(qtbot, window.commitButton)

    assert seen[0].startswith('Commit 2 titles?') and seen[0].index('Images repository') < seen[0].index('XML repository')
    report = committed.args[0]
    assert report.commit_error == '' and report.committed.xml.pushed and report.committed.images.pushed
    assert _commit_count(xml_bare) == before[0] + 1 and _commit_count(images_bare) == before[1] + 1
    assert _subjects(xml_bare)[0].startswith('Publish ') and 'BEQ filters' in _subjects(xml_bare)[0]
    assert _subjects(images_bare)[0].startswith('Publish ') and 'report images' in _subjects(images_bare)[0]
    assert f'{XML_DIR}/fs-a.json' in _files_on_remote(xml_bare) and f'{IMAGE_DIR}/fs-b.png' in _files_on_remote(images_bare)
    assert _needs(window) == {'fs-a': 'done', 'fs-b': 'done'}
    assert window.runStatusLabel.text() == 'Commit finished: 2 committed, 2 pushed'
    assert any(not event.title_id and event.stage == 'commit' and event.kind == 'stage_started' for event in events)
    assert {event.title_id for event in events if event.stage == 'commit' and event.kind.startswith('command_')} == {
        'fs-a', 'fs-b'}
    assert count_snapshots and all(set(snapshot) <= {'fs-a', 'fs-b'} and len(snapshot) == 2
                                   for snapshot in count_snapshots)
    assert all('Command: git' in window._event_buffers[title_id].text() for title_id in ('fs-a', 'fs-b'))
    assert window.runCountsLabel.text() == '2 succeeded'
    by_key = {l.id or l.title: l for l in window.results}
    assert by_key['fs-a'].outcome == 'Committed, pushed' and by_key['fs-b'].outcome == 'Committed, pushed'
    assert by_key['XML repository'].detail.startswith('commit ') and 'pushed (' in by_key['XML repository'].detail
    assert window.results.index(by_key['Images repository']) < window.results.index(by_key['XML repository'])
    assert not window.is_running and window.chip_counts()['Commit'] == 0


def test_commit_with_push_unticked_then_a_push_only_commit_is_worded_as_a_push(qtbot, tmp_path, repos):
    xml, xml_bare, images, images_bare = repos
    _ready(repos)
    before = (_commit_count(xml_bare), _commit_count(images_bare))
    world = _world(tmp_path, 'a')
    prefs = _prefs(tmp_path, **{LIBRARY_PROFILE_PATH: _profile_file(tmp_path, xml.local_path, images.local_path)})
    window = _open(qtbot, prefs, world.items)
    _answer(True, [])
    with qtbot.waitSignal(window.run_finished, timeout=60000):
        window.publish_selected()

    _answer(True, [], tick=False)
    with qtbot.waitSignal(window.run_finished, timeout=60000):
        window.commit_selected()

    assert (_commit_count(xml_bare), _commit_count(images_bare)) == before        # committed locally only
    assert window.runStatusLabel.text() == 'Commit finished: 1 committed, not pushed'
    assert _needs(window) == {'fs-a': 'commit'}         # committed, not pushed: still to do
    assert window.commitButton.text() == 'Push 1'

    seen: list = []
    _answer(True, seen, tick=True)
    with qtbot.waitSignal(window.run_finished, timeout=60000):
        window.commit_selected()

    assert seen[0].startswith('Push 1 title?') and 'Makes one commit' not in seen[0]
    assert (_commit_count(xml_bare), _commit_count(images_bare)) == (before[0] + 1, before[1] + 1)
    assert window.runStatusLabel.text() == 'Commit finished: 1 pushed'
    assert [l.outcome for l in window.results if l.id] == ['Pushed']
    assert _needs(window) == {'fs-a': 'done'}


def test_a_commit_that_fails_in_git_is_a_clean_per_title_failure_and_the_window_carries_on(qtbot, tmp_path, repos):
    ''' The XML "repository" is a plain folder, not a git repository: publish writes into it, commit cannot. '''
    _, _, images, _ = repos
    _ready(repos)
    not_git = tmp_path / 'not-a-repo'
    not_git.mkdir()
    world = _world(tmp_path, 'a', 'b')
    prefs = _prefs(tmp_path, **{LIBRARY_PROFILE_PATH: _profile_file(tmp_path, str(not_git), images.local_path)})
    events = []
    window = _open(qtbot, prefs, world.items, run_stages_fn=_capture_pipeline_events(events))
    _answer(True, [])
    with qtbot.waitSignal(window.run_finished, timeout=60000):
        window.publish_selected()
    assert os.path.isfile(not_git / XML_DIR / 'fs-a.json')
    assert _needs(window) == {'fs-a': 'commit', 'fs-b': 'commit'}

    _answer(True, [], tick=True)
    events.clear()
    count_snapshots = _record_run_counts(window)
    with qtbot.waitSignal(window.run_finished, timeout=60000) as committed:      # nothing raised, nothing crashed
        window.commit_selected()

    assert committed.args[0].commit_error.startswith('git failed:')
    assert any(not event.title_id and event.stage == 'commit' and event.kind == 'failed' for event in events)
    assert count_snapshots and all(set(snapshot) <= {'fs-a', 'fs-b'} and len(snapshot) == 2
                                   for snapshot in count_snapshots)
    assert window.runCountsLabel.text() == '2 failed'
    lines = {l.id or l.title: l for l in window.results}
    for title_id in ('fs-a', 'fs-b'):
        assert lines[title_id].outcome == 'Not committed' and lines[title_id].level == 'error'
        assert 'not a git repository' in lines[title_id].detail and '\n' not in lines[title_id].detail
    assert lines['Commit'].outcome == 'failed' and lines['Commit'].level == 'error'
    assert 'the commit failed' in window.runStatusLabel.text()
    assert window.results[0].level == 'error'                 # problems first
    # the window is usable: nothing is running, the titles are where they were, and it can try again
    assert not window.is_running and window.model.running == {}
    assert _needs(window) == {'fs-a': 'commit', 'fs-b': 'commit'}
    assert window.commitButton.isEnabled() and window.runProgress.isVisibleTo(window) is False


def test_publishing_with_a_remote_that_is_not_github_says_what_to_set_against_every_title(qtbot, tmp_path, repos):
    ''' The preferences bootstrap has no image owner/name (real GitHub remotes are parsed); a local remote cannot be. '''
    xml, _, images, images_bare = repos
    _ready(repos)
    world = _world(tmp_path, 'a', 'b')
    prefs = _prefs(tmp_path, **{LIBRARY_WORK_DIR: world.work, DESIGNER_QUEUE_DIR: world.queue,
                                LIBRARY_FILESYSTEM_GLOBS: [str(tmp_path / 'films' / '*.mkv')],
                                LIBRARY_XML_REPO: xml.local_path, LIBRARY_IMAGES_REPO: images.local_path})
    from model.preferences import DESIGNER_DEFAULT
    prefs.set(DESIGNER_DEFAULT, DESIGNER)
    window = _open(qtbot, prefs, world.items, source='filesystem')   # what the bootstrap names its one source
    assert window.setup.origin == 'preferences' and window.setup.ready
    assert _needs(window) == {'fs-a': 'publish', 'fs-b': 'publish'}

    _answer(True, [])
    with qtbot.waitSignal(window.run_finished, timeout=60000):
        window.publish_selected()

    lines = {l.id: l for l in window.results if l.id}
    assert set(lines) == {'fs-a', 'fs-b'}
    for line in lines.values():
        assert line.outcome == 'Publish failed' and line.level == 'error'
        assert line.detail.startswith('Set image_owner and image_repo_name') and 'Unrecognised' not in line.detail
        assert str(images_bare) in line.detail and 'github.com' in line.detail
    assert _needs(window) == {'fs-a': 'publish', 'fs-b': 'publish'}    # still waiting: nothing was written
