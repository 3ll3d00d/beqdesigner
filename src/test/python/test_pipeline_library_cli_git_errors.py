'''
Review fixes in the CLI (publish / commit / sync): git failures are a one-line error and a distinct exit status (3), not
a traceback; the JSON reports what got done; the XML is not repeated in the JSON; the work directory falls back to
`run:` so a hand edit is not reverted. Real temp git repos with a bad remote.
'''
import json
import subprocess

import pytest

from pipeline.library import cli
from test_pipeline_library_commit import _publish, _queue_entry, _run, repos  # noqa: F401 (a fixture)


def _args(command, queue_dir, repos, *extra):
    xml, _, images, _ = repos
    return [command, '--queue-dir', queue_dir, '--xml-repo', xml.local_path, '--xml-dir', 'xml',
            '--images-repo', images.local_path, '--image-dir', 'img', *extra]


def _break_remote(target, tmp_path):
    _run('git', '-C', target.local_path, 'remote', 'set-url', 'origin', str(tmp_path / 'nowhere.git'))


def test_commit_with_a_bad_remote_prints_one_line_and_exits_3_with_what_was_committed(tmp_path, repos, capsys):
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'))
    _break_remote(repos[0], tmp_path)

    code = cli.main(_args('commit', queue_dir, repos))

    captured = capsys.readouterr()
    assert code == cli.GIT_FAILED == 3
    assert captured.err.startswith('error: git ') and 'nowhere.git' in captured.err and captured.err.count('\n') == 1
    out = json.loads(captured.out)
    assert out['images']['pushed'] is True and out['xml']['commit'] and out['xml']['pushed'] is False
    assert 'nowhere.git' in out['error']


def test_sync_with_a_bad_remote_exits_3_and_still_reports_the_published_entries(tmp_path, repos, capsys):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'one', 'Heat')
    _break_remote(repos[0], tmp_path)

    code = cli.main(_args('sync', queue_dir, repos, '--image-owner', 'o', '--image-repo-name', 'r'))

    captured = capsys.readouterr()
    assert code == 3 and 'nowhere.git' in captured.err
    results = json.loads(captured.out)
    assert [r['id'] for r in results] == ['one'] and results[0]['image_commit']  # the image made it


def test_publish_with_a_broken_images_repo_is_a_per_entry_git_failure_and_exit_3(tmp_path, repos, capsys):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'one', 'Heat')
    _queue_entry(queue_dir, 'two', 'Ronin')
    plain = tmp_path / 'plain'
    plain.mkdir()

    code = cli.main(['publish', '--queue-dir', queue_dir, '--xml-repo', repos[0].local_path, '--xml-dir', 'xml',
                     '--images-repo', str(plain), '--image-dir', 'img', '--image-owner', 'o', '--image-repo-name', 'r'])

    results = json.loads(capsys.readouterr().out)
    assert code == 3
    assert [(r['id'], r['error']) for r in results] == [('one', 'git_failed'), ('two', 'git_failed')]  # not a traceback


def test_a_publish_that_is_merely_refused_still_exits_1(tmp_path, repos, capsys):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'one', '')

    assert cli.main(['publish', '--queue-dir', queue_dir, '--xml-repo', repos[0].local_path]) == 1
    assert 'title is required' in capsys.readouterr().err


def test_publish_and_sync_json_omits_the_whole_xml(tmp_path, repos, capsys):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'one', 'Heat')

    assert cli.main(['publish', '--queue-dir', queue_dir, '--xml-repo', repos[0].local_path, '--xml-dir', 'xml']) == 0

    (result,) = json.loads(capsys.readouterr().out)
    assert result['id'] == 'one' and 'xml' not in result
    assert (tmp_path / 'xml' / 'xml' / 'one.json').read_text().startswith('{')   # it is in the repository


@pytest.mark.parametrize('command', ['publish', 'sync'])
def test_the_work_dir_falls_back_to_the_run_section_and_a_flag_wins(tmp_path, monkeypatch, capsys, command):
    seen = []
    monkeypatch.setattr(cli, 'publish_library', lambda *a, **k: seen.append(k['work_dir']) or [])
    monkeypatch.setattr(cli, 'sync_library', lambda *a, **k: seen.append(k['work_dir']) or [])
    config = tmp_path / 'c.json'
    config.write_text(json.dumps({'run': {'work_dir': '/from-run'}, 'sync': {'queue_dir': '/q', 'xml_repo': '/x'}}))

    assert cli.main(['--config', str(config), command]) == 0
    assert cli.main(['--config', str(config), command, '--work-dir', '/flag']) == 0
    config.write_text(json.dumps({'run': {'work_dir': '/from-run'},
                                  'sync': {'queue_dir': '/q', 'xml_repo': '/x', 'work_dir': '/from-sync'}}))
    assert cli.main(['--config', str(config), command]) == 0

    assert seen == ['/from-run', '/flag', '/from-sync']


def test_the_help_documents_the_exit_status_and_the_work_dir(capsys):
    for command in ('publish', 'commit', 'sync'):
        with pytest.raises(SystemExit):
            cli.main([command, '--help'])
        text = capsys.readouterr().out
        assert '3 ' in text and 'git' in text
    with pytest.raises(SystemExit):
        cli.main(['publish', '--help'])
    assert 'hand-edited' in capsys.readouterr().out
