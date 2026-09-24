'''
The publish/commit split (design/library-sync/workflow-rework §12.7): publish writes into the repos' working trees,
commit makes one commit and one push per repo, images first. Real local git repos (a bare "remote" plus a working
clone), so the git behaviour is genuinely exercised -- no mocking of subprocess.
'''
import io
import json
import os
import subprocess

import numpy as np
import pytest
from PIL import Image

from model.codec import xydata_to_json
from model.iir import CompleteFilter, LowShelf
from model.xy import MagnitudeData
from pipeline.library import commit as commit_module
from pipeline.library.commit import commit_catalogue
from pipeline.library.sync import commit_library, publish_library, sync_library
from pipeline.publish.catalogue import catalogue_paths, publish_digest
from pipeline.publish.git import RepoTarget, current_branch, repo_state
from pipeline.metadata import BeqMetadata
from pipeline.review import CandidateSummary, QueueEntry, publish_reviewed_queue, read_entry, update_entry, \
    write_queue_entry

OWNER, IMAGES_NAME = '3ll3d00d', 'beq-images'


def _run(*args):
    subprocess.run(list(args), check=True, capture_output=True)


def _out(*args):
    return subprocess.run(list(args), check=True, capture_output=True, text=True).stdout


def _repo(tmp_path, name):
    bare = tmp_path / f'{name}.git'
    work = tmp_path / name
    _run('git', 'init', '--bare', '-q', str(bare))
    work.mkdir()
    _run('git', 'init', '-q', str(work))
    _run('git', '-C', str(work), 'config', 'user.email', 'test@example.com')
    _run('git', '-C', str(work), 'config', 'user.name', 'Test')
    _run('git', '-C', str(work), 'remote', 'add', 'origin', str(bare))
    return RepoTarget(str(work)), bare


def _track(target):
    ''' Gives the clone an upstream, so repo_state() can tell what is unpushed (a fresh `git init` has none). '''
    _run('git', '-C', target.local_path, 'branch', f'--set-upstream-to=origin/{current_branch(target)}')


def _queue_entry(queue_dir, entry_id, title, status='accepted', gain=-3.0):
    low_shelf = CompleteFilter(fs=1000, filters=[LowShelf(1000, 18, 0.7, gain)])
    x = np.linspace(1.0, 500.0, 50)
    entry = QueueEntry(
        id=entry_id, fs=1000, meta={'title': title, 'year': '2018', 'audio_types': ['Atmos']},
        curve=xydata_to_json(MagnitudeData('avg', '', x, np.zeros_like(x))),
        candidates=[CandidateSummary(filters=low_shelf.to_json(), confidence=0.9, method='fitted', mv_adjust_db=4.0,
                                     gain_reduction_db=-1.0, commentary={})],
        status=status, chosen_candidate_index=0 if status in ('accepted', 'published') else None)
    write_queue_entry(queue_dir, entry)


@pytest.fixture
def repos(tmp_path):
    xml, xml_bare = _repo(tmp_path, 'xml')
    images, images_bare = _repo(tmp_path, 'images')
    return xml, xml_bare, images, images_bare


def _publish(tmp_path, repos, *entry_titles, push=False, with_images=True, **extra):
    xml, _, images, _ = repos
    queue_dir = str(tmp_path / 'queue')
    for entry_id, title in entry_titles:
        _queue_entry(queue_dir, entry_id, title)
    kwargs = dict(xml_dir='xml', image_dir='img', image_owner=OWNER, image_repo_name=IMAGES_NAME, **extra)
    if with_images:
        kwargs['images_repo'] = images
    if push:
        return queue_dir, publish_reviewed_queue(queue_dir, xml, **kwargs)
    return queue_dir, publish_library(queue_dir, xml, **kwargs)


def _commits(target):
    out = subprocess.run(['git', '-C', target.local_path, 'log', '--format=%H'], capture_output=True, text=True)
    return out.stdout.split() if out.returncode == 0 else []


def _files_in(target, sha):
    return sorted(_out('git', '-C', target.local_path, 'show', '--name-only', '--format=', sha).split())


def _on_remote(bare, path):
    return subprocess.run(['git', '-C', str(bare), 'cat-file', '-p', f'HEAD:{path}'], capture_output=True).stdout


# --- publish: write only ---------------------------------------------------------------------------------------

def test_publish_writes_the_files_but_commits_and_pushes_nothing(tmp_path, repos):
    xml, xml_bare, images, images_bare = repos

    queue_dir, results = _publish(tmp_path, repos, ('one', 'Heat'))

    assert [r['id'] for r in results] == ['one']
    assert results[0]['record']['title'] == 'Heat'
    assert 'filter_commit' not in results[0]
    assert json.loads((tmp_path / 'xml' / 'xml' / 'one.json').read_text()) == results[0]['record']
    assert Image.open(io.BytesIO((tmp_path / 'images' / 'img' / 'one.png').read_bytes())).format == 'PNG'
    assert _commits(xml) == [] and _commits(images) == []
    assert repo_state(xml).uncommitted == {'xml/one.json', 'xml/database.json'}
    assert read_entry(queue_dir, 'one').status == 'published'


def test_the_image_url_in_the_xml_is_known_without_pushing(tmp_path, repos):
    xml, _, images, _ = repos

    _, results = _publish(tmp_path, repos, ('one', 'Heat'))

    expected = f'https://raw.githubusercontent.com/{OWNER}/{IMAGES_NAME}/{current_branch(images)}/img/one.png'
    assert results[0]['image_url'] == expected
    assert expected in results[0]['record']['images']


def test_publishing_records_a_digest_and_when(tmp_path, repos):
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'))

    entry = read_entry(queue_dir, 'one')
    assert len(entry.published_digest) == 64
    assert entry.published_at.endswith('+00:00')


def test_the_default_publish_still_commits_and_pushes_each_file_and_records_the_digest(tmp_path, repos):
    xml, xml_bare, images, images_bare = repos

    queue_dir, results = _publish(tmp_path, repos, ('one', 'Heat'), push=True)

    assert results[0]['filter_commit'] in _commits(xml)
    assert json.loads(_on_remote(xml_bare, 'xml/one.json')) == results[0]['record']
    assert _on_remote(images_bare, 'img/one.png')
    assert read_entry(queue_dir, 'one').published_digest


# --- commit ---------------------------------------------------------------------------------------------------

def test_commit_makes_one_commit_and_one_push_per_repo_for_the_whole_batch(tmp_path, repos):
    xml, xml_bare, images, images_bare = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'), ('two', 'Alien'), ('three', 'Dune'))

    result = commit_catalogue(queue_dir, xml, images, xml_dir='xml', image_dir='img')

    assert len(_commits(xml)) == 1 and len(_commits(images)) == 1
    assert _files_in(xml, result.xml.commit) == ['xml/database.json', 'xml/one.json', 'xml/three.json', 'xml/two.json']
    assert _files_in(images, result.images.commit) == ['img/one.png', 'img/three.png', 'img/two.png']
    assert (result.xml.pushed, result.images.pushed) == (True, True)
    assert _out('git', '-C', str(xml_bare), 'rev-parse', current_branch(xml)).strip() == result.xml.commit
    assert _out('git', '-C', str(images_bare), 'rev-parse', current_branch(images)).strip() == result.images.commit
    assert result.missing == []


def test_the_commit_message_names_the_titles(tmp_path, repos):
    xml, _, images, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'), ('two', 'Alien'))

    result = commit_catalogue(queue_dir, xml, images, xml_dir='xml', image_dir='img')

    assert _out('git', '-C', xml.local_path, 'log', '-1', '--format=%s', result.xml.commit).strip() == \
        'Publish 2 BEQ filters: Heat, Alien'
    assert _out('git', '-C', images.local_path, 'log', '-1', '--format=%s', result.images.commit).strip() == \
        'Publish 2 report images: Heat, Alien'


def test_a_long_batch_summarises_its_titles_in_the_message(tmp_path, repos):
    xml, _, _, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, *[(f'id{i:02d}', f'Title {i:02d}') for i in range(11)], with_images=False)

    result = commit_catalogue(queue_dir, xml, xml_dir='xml')

    subject = _out('git', '-C', xml.local_path, 'log', '-1', '--format=%s', result.xml.commit).strip()
    assert subject.startswith('Publish 11 BEQ filters: Title 00, Title 01') and subject.endswith('and 3 more')


def test_the_images_repo_is_pushed_before_the_xml_repo(tmp_path, repos, monkeypatch):
    xml, _, images, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'))
    pushed = []
    monkeypatch.setattr(commit_module, 'push', lambda target: pushed.append(target.local_path))

    commit_catalogue(queue_dir, xml, images, xml_dir='xml', image_dir='img')

    assert pushed == [images.local_path, xml.local_path]


def test_a_failed_images_push_leaves_the_xml_repo_untouched(tmp_path, repos):
    xml, xml_bare, images, images_bare = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'))
    _run('git', '-C', images.local_path, 'remote', 'set-url', 'origin', str(tmp_path / 'nowhere.git'))

    with pytest.raises(subprocess.CalledProcessError):
        commit_catalogue(queue_dir, xml, images, xml_dir='xml', image_dir='img')

    assert len(_commits(images)) == 1  # committed locally, not lost
    assert _commits(xml) == []  # the XML never reached even a local commit, so nothing can reference a missing image
    assert _on_remote(xml_bare, 'xml/one.json') == b''


def test_a_rejected_push_is_retried_by_running_commit_again(tmp_path, repos):
    xml, xml_bare, images, images_bare = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'))
    good_url = str(xml_bare)
    _run('git', '-C', xml.local_path, 'remote', 'set-url', 'origin', str(tmp_path / 'nowhere.git'))
    with pytest.raises(subprocess.CalledProcessError):
        commit_catalogue(queue_dir, xml, images, xml_dir='xml', image_dir='img')
    first_commit = _commits(xml)[0]
    _run('git', '-C', xml.local_path, 'remote', 'set-url', 'origin', good_url)

    result = commit_catalogue(queue_dir, xml, images, xml_dir='xml', image_dir='img')

    assert result.xml.commit is None  # already committed by the failed run
    assert result.xml.pushed is True
    assert _commits(xml) == [first_commit]
    assert _on_remote(xml_bare, 'xml/one.json')


def test_committing_again_is_a_no_op_once_everything_is_pushed(tmp_path, repos):
    xml, xml_bare, images, images_bare = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'))
    commit_catalogue(queue_dir, xml, images, xml_dir='xml', image_dir='img')
    _track(xml)
    _track(images)

    again = commit_catalogue(queue_dir, xml, images, xml_dir='xml', image_dir='img')

    assert again.xml.commit is None and again.xml.pushed is False and again.xml.paths == []
    assert again.images.commit is None and again.images.pushed is False
    assert len(_commits(xml)) == 1


def test_without_an_upstream_a_never_pushed_repo_is_pushed_and_then_known_to_be_pushed(tmp_path, repos):
    xml, _, images, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'))
    first = commit_catalogue(queue_dir, xml, images, xml_dir='xml', image_dir='img')
    assert first.xml.pushed is True  # nothing to compare with yet: cannot tell, so it pushes

    again = commit_catalogue(queue_dir, xml, images, xml_dir='xml', image_dir='img')

    # no @{upstream} is configured, but the push updated origin/<branch>, which is what "pushed" is measured against
    assert again.xml.commit is None and again.xml.pushed is False and again.xml.paths == []
    assert len(_commits(xml)) == 1


def test_a_revision_is_a_second_commit_at_the_same_path(tmp_path, repos):
    xml, xml_bare, images, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'), with_images=False)
    first = commit_catalogue(queue_dir, xml, xml_dir='xml')
    _track(xml)
    update_entry(queue_dir, 'one', status='accepted', meta={'title': 'Heat (1995)', 'year': '1995',
                                                            'audio_types': ['Atmos']})
    publish_library(queue_dir, xml, xml_dir='xml')

    second = commit_catalogue(queue_dir, xml, xml_dir='xml')

    assert second.xml.commit not in (None, first.xml.commit)
    assert second.xml.paths == ['xml/one.json', 'xml/database.json']
    assert b'Heat (1995)' in _on_remote(xml_bare, 'xml/one.json')


def test_a_commit_leaves_other_staged_and_untracked_files_alone(tmp_path, repos):
    xml, _, _, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'), with_images=False)
    (tmp_path / 'xml' / 'notes.txt').write_text('staged by someone')
    _run('git', '-C', xml.local_path, 'add', 'notes.txt')
    (tmp_path / 'xml' / 'scratch.txt').write_text('untracked')

    result = commit_catalogue(queue_dir, xml, xml_dir='xml')

    assert _files_in(xml, result.xml.commit) == ['xml/database.json', 'xml/one.json']
    assert repo_state(xml).uncommitted == {'notes.txt', 'scratch.txt'}


def test_only_published_entries_are_committed(tmp_path, repos):
    xml, _, _, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'), with_images=False)
    _queue_entry(queue_dir, 'pending-one', 'Alien', status='pending')
    (tmp_path / 'xml' / 'xml' / 'pending-one.json').write_text('a stray file for an unpublished entry')

    result = commit_catalogue(queue_dir, xml, xml_dir='xml')

    assert result.xml.paths == ['xml/one.json', 'xml/database.json']


def test_a_published_entry_whose_file_is_gone_is_reported_not_fatal(tmp_path, repos):
    xml, _, _, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'), ('two', 'Alien'), with_images=False)
    os.remove(tmp_path / 'xml' / 'xml' / 'two.json')

    result = commit_catalogue(queue_dir, xml, xml_dir='xml')

    assert result.missing == [os.path.join('xml', 'two.json')]
    assert result.xml.paths == ['xml/one.json', 'xml/database.json']


def test_no_push_commits_locally_only(tmp_path, repos):
    xml, xml_bare, _, _ = repos
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'), with_images=False)

    result = commit_catalogue(queue_dir, xml, xml_dir='xml', push=False)

    assert result.xml.commit and result.xml.pushed is False
    assert _on_remote(xml_bare, 'xml/one.json') == b''


def test_nothing_published_means_nothing_to_do(tmp_path, repos):
    xml, _, images, _ = repos

    result = commit_catalogue(str(tmp_path / 'queue'), xml, images)

    assert (result.xml.commit, result.xml.paths, result.xml.pushed) == (None, [], False)


# --- sync = publish + commit, and matches what the one-step publish produced ----------------------------------------

def _tree(bare, prefix):
    names = _out('git', '-C', str(bare), 'ls-tree', '-r', '--name-only', 'HEAD', prefix).split()
    return {name: _on_remote(bare, name) for name in names}


def _records_without_generated_timestamps(bare):
    records = {name: json.loads(content) for name, content in _tree(bare, 'xml').items()}
    for record in records.values():
        for item in record if isinstance(record, list) else [record]:
            item.pop('created_at', None)
            item.pop('updated_at', None)
    return records


def test_sync_puts_the_same_records_on_the_remotes_as_per_file_publish(tmp_path):
    old = tmp_path / 'old'
    new = tmp_path / 'new'
    old.mkdir(), new.mkdir()
    old_repos, new_repos = (_repo(old, 'xml') + _repo(old, 'images'), _repo(new, 'xml') + _repo(new, 'images'))
    _publish(old, old_repos, ('one', 'Heat'), ('two', 'Alien'), push=True)
    new_queue = str(new / 'queue')
    for entry_id, title in (('one', 'Heat'), ('two', 'Alien')):
        _queue_entry(new_queue, entry_id, title)

    sync_library(new_queue, new_repos[0], images_repo=new_repos[2], xml_dir='xml', image_dir='img',
                 image_owner=OWNER, image_repo_name=IMAGES_NAME)

    assert _records_without_generated_timestamps(new_repos[1]) == \
        _records_without_generated_timestamps(old_repos[1]) != {}
    assert _tree(new_repos[3], 'img').keys() == _tree(old_repos[3], 'img').keys()
    assert {n: Image.open(io.BytesIO(b)).size for n, b in _tree(new_repos[3], 'img').items()} == \
        {n: Image.open(io.BytesIO(b)).size for n, b in _tree(old_repos[3], 'img').items()}
    assert len(_commits(new_repos[0])) == 1 and len(_commits(old_repos[0])) == 2  # a batch, not a commit per file


def test_sync_reports_each_published_entry_with_the_batchs_commit(tmp_path, repos):
    xml, _, images, _ = repos
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'one', 'Heat')
    _queue_entry(queue_dir, 'two', 'Alien')

    results = sync_library(queue_dir, xml, images_repo=images, xml_dir='xml', image_dir='img',
                           image_owner=OWNER, image_repo_name=IMAGES_NAME)

    assert [r['id'] for r in results] == ['one', 'two']
    assert {r['xml_commit'] for r in results} == {_commits(xml)[0]}
    assert {r['image_commit'] for r in results} == {_commits(images)[0]}


def test_sync_can_be_told_not_to_push(tmp_path, repos):
    xml, xml_bare, _, _ = repos
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'one', 'Heat')

    results = sync_library(queue_dir, xml, xml_dir='xml', push=False)

    assert results[0]['xml_commit'] in _commits(xml)
    assert _on_remote(xml_bare, 'xml/one.json') == b''


# --- the publish digest --------------------------------------------------------------------------------------

def _meta(**changes):
    fields = dict(title='Heat', year='1995', audio_types=['Atmos'])
    fields.update(changes)
    return BeqMetadata(**fields)


def _filter_json(gain=-3.0):
    return CompleteFilter(fs=1000, filters=[LowShelf(1000, 18, 0.7, gain)]).to_json()


def _digest(filter_json=None, meta=None, art_path=None, has_image=True, mv_offset=4.0):
    return publish_digest(filter_json or _filter_json(), meta or _meta(), art_path, has_image, mv_offset)


def test_the_digest_is_stable_for_identical_inputs():
    assert _digest() == _digest()
    assert len(_digest()) == 64


@pytest.mark.parametrize('changed', [
    dict(filter_json=_filter_json(gain=-4.0)),
    dict(meta=_meta(title='Heat (1995)')),
    dict(meta=_meta(edition='Director\'s Cut')),
    dict(meta=_meta(audio_types=['Atmos', 'TrueHD 7.1'])),
    dict(has_image=False),
    dict(mv_offset=5.0),
])
def test_the_digest_changes_when_anything_published_changes(changed):
    assert _digest(**changed) != _digest()


def test_the_digest_follows_the_artwork_content_not_its_timestamp(tmp_path):
    poster = tmp_path / 'poster.jpg'
    poster.write_bytes(b'first poster')
    first = _digest(art_path=str(poster))
    os.utime(poster, (1, 1))
    assert _digest(art_path=str(poster)) == first

    poster.write_bytes(b'another poster')
    assert _digest(art_path=str(poster)) != first
    assert _digest(art_path=str(tmp_path / 'missing.jpg')) == _digest()  # no file: the same as no artwork


def test_republishing_an_edited_title_records_a_new_digest(tmp_path, repos):
    queue_dir, _ = _publish(tmp_path, repos, ('one', 'Heat'), with_images=False)
    first = read_entry(queue_dir, 'one').published_digest
    update_entry(queue_dir, 'one', status='accepted')
    publish_library(queue_dir, repos[0], xml_dir='xml')
    assert read_entry(queue_dir, 'one').published_digest == first  # same inputs, same digest

    update_entry(queue_dir, 'one', status='accepted',
                 meta={'title': 'Heat', 'year': '1995', 'audio_types': ['Atmos']})
    publish_library(queue_dir, repos[0], xml_dir='xml')

    assert read_entry(queue_dir, 'one').published_digest != first


def test_catalogue_paths_are_the_entry_id_under_each_directory():
    assert catalogue_paths('jriver-3fa9c2-1234', 'filters', 'images') == (
        'filters/jriver-3fa9c2-1234.json', 'images/jriver-3fa9c2-1234.png')
    assert catalogue_paths('x') == ('x.json', 'x.png')
