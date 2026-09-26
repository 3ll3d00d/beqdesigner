'''
model/worklist_autopublish.py: accepting a title is the whole of it -- its files are written into the catalogue repositories and
committed locally (never pushed) on a worker, without the Publish and Commit buttons. `import ui.beq` first: AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import os
import subprocess
from types import SimpleNamespace

import pytest
from qtpy.QtCore import Qt

import model.worklist_autopublish as autopublish
from model.worklist_autopublish import publish_and_commit_locally
from pipeline.library.stages import PublishSettings
from pipeline.publish.catalogue import CategoryFolders
from pipeline.review import read_entry
from test_pipeline_library_commit import _commits, _queue_entry, _repo, repos  # noqa: F401 (a fixture)
from test_worklist_title import REVIEWABLE, _click, _open, _window


def _settings(repos, **extra):
    xml, _, images, _ = repos
    return PublishSettings(xml, images, 'o', 'r', 'xml', 'img', category_folders=CategoryFolders(), heatmap_spec=None,
                           push=False, **extra)


def _config():
    from pipeline.config import AnalysisConfig
    return AnalysisConfig()


# --- the work, with no widgets ---------------------------------------------------------------------------------------

def test_accepted_titles_are_written_and_committed_locally_and_nothing_is_pushed(tmp_path, repos):
    xml, xml_bare, images, images_bare = repos
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'a', 'Heat')
    _queue_entry(queue_dir, 'b', 'Alien')

    outcome = publish_and_commit_locally(['a', 'b'], queue_dir, None, _settings(repos), _config())

    assert outcome.ok and [r['id'] for r in outcome.published] == ['a', 'b']
    assert [read_entry(queue_dir, i).status for i in 'ab'] == ['published', 'published']
    assert len(_commits(xml)) == 1 and len(_commits(images)) == 1          # one commit per repository, whatever the count
    for bare in (xml_bare, images_bare):   # nothing reached either remote
        assert subprocess.run(['git', '-C', str(bare), 'for-each-ref'], capture_output=True, text=True).stdout == ''
    assert 'Published 2 titles' in outcome.describe()


def test_a_title_that_cannot_be_published_is_reported_and_the_rest_go_ahead(tmp_path, repos):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'a', 'Heat')
    _queue_entry(queue_dir, 'b', '')

    outcome = publish_and_commit_locally(['a', 'b'], queue_dir, None, _settings(repos), _config())

    assert not outcome.ok
    assert [r['id'] for r in outcome.published] == ['a'] and [r['id'] for r in outcome.errors] == ['b']
    assert read_entry(queue_dir, 'b').status == 'accepted'                  # still there for Publish to retry
    assert 'Accepted, but not fully published' in outcome.describe({'b': 'Untitled'})


def test_a_failed_local_commit_leaves_the_entries_written_for_the_commit_button(tmp_path, repos, monkeypatch):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'a', 'Heat')

    def refuse(*args, **kwargs):
        raise subprocess.CalledProcessError(1, 'git')
    monkeypatch.setattr(autopublish, 'commit_library', refuse)

    outcome = publish_and_commit_locally(['a'], queue_dir, None, _settings(repos), _config())

    assert outcome.commit_error and not outcome.ok
    assert read_entry(queue_dir, 'a').status == 'published'
    assert 'the local commit failed' in outcome.describe()


# --- the window -------------------------------------------------------------------------------------------------------

@pytest.fixture
def stubbed(monkeypatch):
    ''' The window has a repository set up and the work is a stub that records what it was asked to do. '''
    calls = []

    def work(ids, queue_dir, work_dir, settings, config):
        calls.append(list(ids))
        return autopublish.AutoPublishOutcome(list(ids), [{'id': i} for i in ids], None)
    monkeypatch.setattr(autopublish, 'publish_problem', lambda setup: '')
    monkeypatch.setattr(autopublish, 'build_publish_settings', lambda setup, push=True, preferences=None: SimpleNamespace(push=push))
    monkeypatch.setattr(autopublish, 'publish_and_commit_locally', work)
    return calls


def test_accept_and_next_writes_and_commits_the_title_without_a_publish_step(qtbot, tmp_path, stubbed):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')

    with qtbot.waitSignal(window.auto_published, timeout=10000) as signal:
        _click(qtbot, page.acceptButton)

    assert stubbed == [['r-alien']]
    assert [r['id'] for r in signal.args[0].published] == ['r-alien']
    assert 'Published' in window.statusBar.currentMessage()


def test_skip_and_reject_publish_nothing(qtbot, tmp_path, stubbed):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')

    assert page.skip() and page.reject()

    assert stubbed == [] and not window.auto_publishing


def test_no_repository_means_the_title_is_accepted_and_the_reason_is_said(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(autopublish, 'publish_problem', lambda setup: 'No filter-record repository is set.')
    monkeypatch.setattr(autopublish, 'publish_and_commit_locally', lambda *a: pytest.fail('published'))
    window = _window(qtbot, tmp_path, REVIEWABLE)
    page = _open(qtbot, window, 'r-alien')

    _click(qtbot, page.acceptButton)

    assert read_entry(str(tmp_path / 'queue'), 'r-alien').status == 'accepted'
    assert 'Accepted, not published: No filter-record repository is set.' in window.statusBar.currentMessage()


def test_titles_accepted_while_one_is_publishing_wait_their_turn_and_go_in_one_job(qtbot, tmp_path, stubbed):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    window._auto_job = object()          # a job in flight

    assert window.auto_publish(['a']) and window.auto_publish(['b', 'a'])
    assert stubbed == []

    window._auto_job = None
    with qtbot.waitSignal(window.auto_published, timeout=10000):
        window._start_auto_publish()

    assert stubbed == [['a', 'b']]


def test_they_also_wait_for_a_run_that_uses_the_repositories(qtbot, tmp_path, stubbed):
    window = _window(qtbot, tmp_path, REVIEWABLE)
    window._job = object()               # a publish/commit run in flight

    window.auto_publish(['a'])
    assert stubbed == []

    window._job = None
    with qtbot.waitSignal(window.auto_published, timeout=10000):
        window._start_auto_publish()
    assert stubbed == [['a']]


def test_a_failure_of_the_job_is_said_and_the_window_carries_on(qtbot, tmp_path, monkeypatch, stubbed):
    def boom(*args):
        raise RuntimeError('git is not installed')
    monkeypatch.setattr(autopublish, 'publish_and_commit_locally', boom)
    window = _window(qtbot, tmp_path, REVIEWABLE)

    window.auto_publish(['a'])
    qtbot.waitUntil(lambda: not window.auto_publishing, timeout=10000)

    assert 'publishing failed: RuntimeError: git is not installed' in window.statusBar.currentMessage()
