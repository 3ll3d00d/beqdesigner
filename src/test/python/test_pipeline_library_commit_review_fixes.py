'''
Review fixes for commit and revise (chunks 21/22): git errors carry what git said and leave a partial result, files git
ignores are reported not silently skipped, a subdirectory repo works, an XML naming an image is warned about when no
images repo is given, revise validates before it changes anything, and the revision counter follows one rule.
'''
import subprocess

import pytest

from pipeline.library.commit import commit_catalogue
from pipeline.library.revise import reopen_entry
from pipeline.library.sync import commit_library, publish_library
from pipeline.publish.git import RepoTarget, repo_state
from pipeline.review import publish_reviewed_queue, read_entry, update_entry
from test_pipeline_library_commit import (IMAGES_NAME, OWNER, _commits, _on_remote, _publish, _queue_entry, _repo, _run,
                                          _track, repos)  # noqa: F401 (repos is a fixture)
from test_pipeline_library_revise import _published_and_committed, _where


# --- 3: a failed push carries the message and what was done ------------------------------------------------------

def test_a_failed_push_carries_git_s_message_and_what_was_committed_before_it(tmp_path, repos):
    xml, _, images, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'))
    _run('git', '-C', xml.local_path, 'remote', 'set-url', 'origin', str(tmp_path / 'nowhere.git'))

    with pytest.raises(subprocess.CalledProcessError) as error:
        commit_catalogue(queue_dir, xml, images, xml_dir='xml', image_dir='img')

    assert 'nowhere.git' in str(error.value)
    partial = error.value.partial
    assert partial.images.pushed and partial.images.commit           # the images repo got all the way
    assert partial.xml.commit and partial.xml.pushed is False          # the XML was committed, not pushed
    assert partial.xml.paths == ['xml/one.xml']


# --- 4: ignored files, subdirectory repos ----------------------------------------------------------------------------

def test_a_published_file_that_git_ignores_is_reported_not_silently_skipped(tmp_path, repos):
    xml, _, images, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'), ('two', 'Ronin'))
    (tmp_path / 'xml' / '.gitignore').write_text('xml/two.xml\n')

    result = commit_catalogue(queue_dir, xml, images, xml_dir='xml', image_dir='img', push=False)

    assert result.xml.paths == ['xml/one.xml']   # `two` never got in: it is published but not in the catalogue
    assert result.not_committed == ['xml/two.xml']


def test_a_gitignored_xml_is_the_cli_s_git_failure_exit(tmp_path, repos, capsys):
    from pipeline.library import cli
    xml, _, images, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'))
    (tmp_path / 'xml' / '.gitignore').write_text('xml/one.xml\n')

    code = cli.main(['commit', '--queue-dir', queue_dir, '--xml-repo', xml.local_path, '--xml-dir', 'xml',
                     '--images-repo', images.local_path, '--image-dir', 'img', '--no-push'])

    assert code == cli.GIT_FAILED == 3
    assert 'xml/one.xml' in capsys.readouterr().err


def test_a_repo_that_is_a_subdirectory_of_a_clone_commits_its_files(tmp_path, repos):
    root, root_bare, images, _ = repos
    sub = RepoTarget(str(tmp_path / 'xml' / 'cat'), root.remote)
    queue_dir, _ = _publish(tmp_path, (sub, root_bare, images, None), ('one', 'Heat'))
    assert repo_state(sub).uncommitted == {'xml/one.xml'}

    result = commit_catalogue(queue_dir, sub, images, xml_dir='xml', image_dir='img', push=False)

    assert result.xml.paths == ['xml/one.xml'] and result.xml.commit and result.not_committed == []
    assert repo_state(sub).uncommitted == frozenset()


# --- 5: an XML that names an image, committed without an images repo ------------------------------------------------

def test_committing_an_xml_that_names_an_image_without_an_images_repo_warns(tmp_path, repos):
    xml, _, images, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'))

    result = commit_library(queue_dir, xml, xml_dir='xml', image_dir='img', push=False)  # no images_repo

    assert result.xml.commit  # not blocked
    assert len(result.warnings) == 1 and 'xml/one.xml' in result.warnings[0] and 'images' in result.warnings[0]


def test_no_warning_when_the_images_repo_is_given_or_the_xml_has_no_image(tmp_path, repos):
    xml, _, images, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'))
    assert commit_library(queue_dir, xml, images_repo=images, xml_dir='xml', image_dir='img',
                          push=False).warnings == []
    queue_dir, _ = _publish(tmp_path, repos, ('two', 'Ronin'), with_images=False)
    assert commit_library(queue_dir, xml, xml_dir='xml', image_dir='img', push=False, ids=['two']).warnings == []


def test_the_cli_prints_that_warning_to_stderr(tmp_path, repos, capsys):
    from pipeline.library import cli
    xml, _, _, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'))

    assert cli.main(['commit', '--queue-dir', queue_dir, '--xml-repo', xml.local_path, '--xml-dir', 'xml',
                     '--no-push']) == 0
    assert 'warning' in capsys.readouterr().err


# --- 2: revise validates both repos before it changes anything ----------------------------------------------------

def test_reopen_with_a_bad_images_repo_changes_nothing(tmp_path, repos):
    xml, _, images, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'))   # written, never committed: it would be deleted
    not_a_repo = tmp_path / 'plain'
    not_a_repo.mkdir()

    with pytest.raises(ValueError, match='images_repo'):
        reopen_entry(queue_dir, 'one', xml_repo=xml, images_repo=RepoTarget(str(not_a_repo)), xml_dir='xml',
                     image_dir='img')

    assert (tmp_path / 'xml' / 'xml' / 'one.xml').exists()   # not deleted
    assert read_entry(queue_dir, 'one').status == 'published'

    with pytest.raises(ValueError, match='xml_repo'):
        reopen_entry(queue_dir, 'one', xml_repo=RepoTarget(str(tmp_path / 'missing')), xml_dir='xml')
    assert read_entry(queue_dir, 'one').status == 'published'


def test_revise_reports_a_git_failure_per_id_and_carries_on(tmp_path, repos, capsys, monkeypatch):
    from pipeline.library import cli
    xml, _, images, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'), ('two', 'Ronin'))
    real = cli.revise_entry

    def flaky(queue, entry_id, *args, **kwargs):
        if entry_id == 'one':
            raise subprocess.CalledProcessError(128, ['git', 'checkout'], stderr='boom')
        return real(queue, entry_id, *args, **kwargs)

    monkeypatch.setattr(cli, 'revise_entry', flaky)
    code = cli.main(['revise', '--queue-dir', queue_dir, '--id', 'one', '--id', 'two', '--to', 'review',
                     '--xml-repo', xml.local_path, '--xml-dir', 'xml', '--images-repo', images.local_path,
                     '--image-dir', 'img'])

    import json
    results = json.loads(capsys.readouterr().out)
    assert code == 1 and 'error' in results[0] and results[1]['status'] == 'pending'


def test_a_reopen_that_fails_after_the_image_discard_can_simply_be_run_again(tmp_path, repos, monkeypatch):
    from pipeline.library import revise
    xml, _, images, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'))
    real = revise.discard_changes

    def fail_on_xml(target, paths):
        if target is xml:
            raise subprocess.CalledProcessError(1, ['git'], stderr='disk on fire')
        return real(target, paths)

    monkeypatch.setattr(revise, 'discard_changes', fail_on_xml)
    with pytest.raises(subprocess.CalledProcessError):
        reopen_entry(queue_dir, 'one', **_where(repos))
    assert read_entry(queue_dir, 'one').status == 'published'   # still published: nothing half-recorded

    monkeypatch.setattr(revise, 'discard_changes', real)
    result = reopen_entry(queue_dir, 'one', **_where(repos))
    assert result.reverted == ['xml/one.xml']   # the image was put back by the first attempt
    assert read_entry(queue_dir, 'one').status == 'pending'


# --- 6: one revision rule --------------------------------------------------------------------------------------------

def test_republishing_a_committed_entry_begins_a_revision_and_reopening_it_then_does_not_count_again(tmp_path, repos):
    xml, _, images, _ = repos
    queue_dir = _published_and_committed(tmp_path, repos)
    update_entry(queue_dir, 'one', meta={'title': 'Heat (fixed)', 'year': '2018', 'audio_types': ['Atmos']})

    results = publish_reviewed_queue(queue_dir, xml, images_repo=images, xml_dir='xml', image_dir='img', push=False,
                                     image_owner=OWNER, image_repo_name=IMAGES_NAME, republish=True)
    assert results[0]['republished'] is True
    assert read_entry(queue_dir, 'one').revision == 1          # the committed copy is being superseded

    update_entry(queue_dir, 'one', meta={'title': 'Heat (fixed again)', 'year': '2018', 'audio_types': ['Atmos']})
    publish_reviewed_queue(queue_dir, xml, images_repo=images, xml_dir='xml', image_dir='img', push=False,
                           image_owner=OWNER, image_repo_name=IMAGES_NAME, republish=True)
    assert read_entry(queue_dir, 'one').revision == 1          # still the same uncommitted revision

    result = reopen_entry(queue_dir, 'one', **_where(repos))
    assert 'xml/one.xml' in result.reverted
    assert read_entry(queue_dir, 'one').revision == 1          # counted already, by the republish
