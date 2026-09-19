'''
publish_reviewed_queue() beyond "publish what is accepted" (design.md §12.6/§12.7, chunk 25): a per-title refusal for
incomplete metadata instead of aborting the batch, an `ids` restriction, and `republish` -- writing a *published* title
whose catalogue copy is out of date again, at the same path, without a second review.
'''
import os

import pytest

from pipeline.library.commit import commit_catalogue
from pipeline.publish.catalogue import catalogue_paths
from pipeline.review import describe_publish_error, publish_reviewed_queue, read_entry, split_publish_results, \
    update_entry
from test_pipeline_library_commit import _commits, _on_remote, _publish, _queue_entry, repos  # noqa: F401 (a fixture)


def _xml(repos, entry_id):
    with open(os.path.join(repos[0].local_path, 'xml', f'{entry_id}.xml'), encoding='utf-8') as f:
        return f.read()


def _edit_title(queue_dir, entry_id, title):
    entry = read_entry(queue_dir, entry_id)
    update_entry(queue_dir, entry_id, meta={**entry.meta, 'title': title})


# --- one incomplete title does not abort the batch ------------------------------------------------------------------

def test_an_entry_with_incomplete_metadata_is_refused_on_its_own_and_the_others_are_published(tmp_path, repos):
    xml, _, _, _ = repos
    queue_dir = str(tmp_path / 'queue')
    for entry_id, title in (('a', 'Alien'), ('b', ''), ('c', 'Cube')):
        _queue_entry(queue_dir, entry_id, title)

    results = publish_reviewed_queue(queue_dir, xml, xml_dir='xml', push=False)

    published, refused = split_publish_results(results)
    assert [r['id'] for r in published] == ['a', 'c']  # the batch ran to the end
    assert refused == [{'id': 'b', 'error': 'invalid_metadata', 'problems': ['title is required']}]
    assert read_entry(queue_dir, 'b').status == 'accepted'  # untouched: it can be fixed and published later
    assert not os.path.exists(os.path.join(xml.local_path, 'xml', 'b.xml'))


def test_the_refusal_describes_itself_for_a_reviewer(tmp_path, repos):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'b', '')
    update_entry(queue_dir, 'b', meta={'title': '', 'year': '', 'audio_types': []})

    (result,) = publish_reviewed_queue(queue_dir, repos[0], xml_dir='xml', push=False)

    text = describe_publish_error(result)
    assert text.startswith('b: the metadata is not complete enough to publish: ')
    assert 'title is required' in text and 'year is required' in text and 'audio type' in text


def test_a_refused_entry_is_not_half_published_an_image_is_not_written_for_it(tmp_path, repos):
    xml, _, images, _ = repos
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'b', '')

    publish_reviewed_queue(queue_dir, xml, images_repo=images, xml_dir='xml', image_dir='img', push=False,
                           image_owner='o', image_repo_name='r')

    assert not os.path.exists(os.path.join(images.local_path, 'img', 'b.png'))


# --- ids ------------------------------------------------------------------------------------------------------------

def test_ids_restricts_what_is_published_and_ignores_an_unknown_id(tmp_path, repos):
    xml, _, _, _ = repos
    queue_dir = str(tmp_path / 'queue')
    for entry_id in ('a', 'b', 'c'):
        _queue_entry(queue_dir, entry_id, entry_id.upper() * 3)

    results = publish_reviewed_queue(queue_dir, xml, xml_dir='xml', push=False, ids=['b', 'nope', 'b'])

    assert [r['id'] for r in results] == ['b']
    assert [read_entry(queue_dir, i).status for i in 'abc'] == ['accepted', 'published', 'accepted']


def test_ids_takes_only_what_is_accepted_or_republishable(tmp_path, repos):
    xml, _, _, _ = repos
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'p', 'Pending', status='pending')

    assert publish_reviewed_queue(queue_dir, xml, xml_dir='xml', push=False, ids=['p'], republish=True) == []


# --- republish ------------------------------------------------------------------------------------------------------

def _published(tmp_path, repos, **extra):
    queue_dir, results = _publish(tmp_path, repos, ('a', 'Alien'), with_images=False, **extra)
    assert [r['id'] for r in results] == ['a']
    return queue_dir


def test_a_published_title_is_left_alone_unless_asked_to_republish(tmp_path, repos):
    xml, _, _, _ = repos
    queue_dir = _published(tmp_path, repos)
    _edit_title(queue_dir, 'a', 'Aliens')

    assert publish_reviewed_queue(queue_dir, xml, xml_dir='xml', push=False) == []
    assert '<beq_title>Alien</beq_title>' in _xml(repos, 'a')


def test_republish_writes_a_changed_published_title_again_at_the_same_path_and_keeps_it_published(tmp_path, repos):
    xml, _, _, _ = repos
    queue_dir = _published(tmp_path, repos)
    before = read_entry(queue_dir, 'a')
    _edit_title(queue_dir, 'a', 'Aliens')

    (result,) = publish_reviewed_queue(queue_dir, xml, xml_dir='xml', push=False, republish=True)

    after = read_entry(queue_dir, 'a')
    assert result['id'] == 'a' and result['republished'] is True
    assert '<beq_title>Aliens</beq_title>' in _xml(repos, 'a')  # the same file, rewritten
    assert after.status == 'published' and after.revision == before.revision
    assert after.published_digest and after.published_digest != before.published_digest
    assert after.published_at
    assert sorted(os.listdir(os.path.join(xml.local_path, 'xml'))) == ['a.xml']  # no second file


def test_republish_then_commit_records_a_revision_at_the_same_path(tmp_path, repos):
    xml, xml_bare, _, _ = repos
    queue_dir = _published(tmp_path, repos)
    commit_catalogue(queue_dir, xml, xml_dir='xml', push=True)
    _edit_title(queue_dir, 'a', 'Aliens')

    publish_reviewed_queue(queue_dir, xml, xml_dir='xml', push=False, republish=True)
    committed = commit_catalogue(queue_dir, xml, xml_dir='xml', push=True)

    assert committed.xml.paths == ['xml/a.xml'] and len(_commits(xml)) == 2
    assert b'Aliens' in _on_remote(xml_bare, 'xml/a.xml')


def test_republish_does_nothing_for_a_published_title_that_is_not_out_of_date(tmp_path, repos):
    xml, _, _, _ = repos
    queue_dir = _published(tmp_path, repos)
    before = read_entry(queue_dir, 'a')

    assert publish_reviewed_queue(queue_dir, xml, xml_dir='xml', push=False, republish=True) == []
    assert read_entry(queue_dir, 'a') == before


def test_republish_writes_a_title_whose_file_left_the_repository(tmp_path, repos):
    xml, _, _, _ = repos
    queue_dir = _published(tmp_path, repos)
    os.remove(os.path.join(xml.local_path, *catalogue_paths('a', 'xml')[0].split('/')))

    (result,) = publish_reviewed_queue(queue_dir, xml, xml_dir='xml', push=False, republish=True)

    assert result['republished'] and os.path.isfile(os.path.join(xml.local_path, 'xml', 'a.xml'))


def test_republish_leaves_alone_a_title_published_before_digests_were_recorded(tmp_path, repos):
    xml, _, _, _ = repos
    queue_dir = _published(tmp_path, repos)
    update_entry(queue_dir, 'a', published_digest=None)
    _edit_title(queue_dir, 'a', 'Aliens')

    assert publish_reviewed_queue(queue_dir, xml, xml_dir='xml', push=False, republish=True) == []  # cannot tell


def test_republish_of_a_title_whose_metadata_became_incomplete_is_refused_and_left_published(tmp_path, repos):
    xml, _, _, _ = repos
    queue_dir = _published(tmp_path, repos)
    _edit_title(queue_dir, 'a', '')

    (result,) = publish_reviewed_queue(queue_dir, xml, xml_dir='xml', push=False, republish=True)

    assert result['error'] == 'invalid_metadata'
    assert read_entry(queue_dir, 'a').status == 'published' and '<beq_title>Alien</beq_title>' in _xml(repos, 'a')


def test_republish_takes_an_accepted_and_a_changed_published_title_in_one_batch(tmp_path, repos):
    xml, _, _, _ = repos
    queue_dir = _published(tmp_path, repos)
    _queue_entry(queue_dir, 'b', 'Blade')
    _edit_title(queue_dir, 'a', 'Aliens')

    results = publish_reviewed_queue(queue_dir, xml, xml_dir='xml', push=False, republish=True)

    assert {r['id']: r.get('republished', False) for r in results} == {'a': True, 'b': False}


# --- progress hooks -------------------------------------------------------------------------------------------------

def test_on_entry_names_each_entry_about_to_be_published_and_not_a_skipped_one(tmp_path, repos):
    xml, _, _, _ = repos
    queue_dir = str(tmp_path / 'queue')
    for entry_id in ('a', 'b'):
        _queue_entry(queue_dir, entry_id, entry_id.upper() * 3)
    _queue_entry(queue_dir, 'p', 'Pending', status='pending')
    seen = []

    publish_reviewed_queue(queue_dir, xml, xml_dir='xml', push=False, on_entry=seen.append)

    assert seen == ['a', 'b']


def test_should_cancel_stops_before_the_next_entry_and_leaves_each_published_one_complete(tmp_path, repos):
    xml, _, _, _ = repos
    queue_dir = str(tmp_path / 'queue')
    for entry_id in ('a', 'b', 'c'):
        _queue_entry(queue_dir, entry_id, entry_id.upper() * 3)
    published = []

    results = publish_reviewed_queue(queue_dir, xml, xml_dir='xml', push=False, on_entry=published.append,
                                     should_cancel=lambda: len(published) >= 2)

    assert [r['id'] for r in results] == ['a', 'b']
    assert [read_entry(queue_dir, i).status for i in 'abc'] == ['published', 'published', 'accepted']
    assert not os.path.exists(os.path.join(xml.local_path, 'xml', 'c.xml'))


# --- commit_catalogue(ids=) -----------------------------------------------------------------------------------------

def test_commit_ids_commits_only_the_named_titles_files(tmp_path, repos):
    xml, _, _, _ = repos
    queue_dir = str(tmp_path / 'queue')
    for entry_id in ('a', 'b'):
        _queue_entry(queue_dir, entry_id, entry_id.upper() * 3)
    publish_reviewed_queue(queue_dir, xml, xml_dir='xml', push=False)

    committed = commit_catalogue(queue_dir, xml, xml_dir='xml', push=False, ids=['a', 'nope'])

    assert committed.xml.paths == ['xml/a.xml'] and len(_commits(xml)) == 1
    rest = commit_catalogue(queue_dir, xml, xml_dir='xml', push=False)
    assert rest.xml.paths == ['xml/b.xml'] and len(_commits(xml)) == 2


@pytest.mark.parametrize('ids', [[], ['nope']])
def test_commit_ids_that_match_nothing_commit_nothing(tmp_path, repos, ids):
    xml, _, _, _ = repos
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'a', 'Alien')
    publish_reviewed_queue(queue_dir, xml, xml_dir='xml', push=False)

    committed = commit_catalogue(queue_dir, xml, xml_dir='xml', push=False, ids=ids)

    assert committed.xml.paths == [] and committed.xml.commit is None and _commits(xml) == []
