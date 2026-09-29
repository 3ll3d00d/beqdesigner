'''
What a publish leaves in the catalogue repositories now: readable names, in configurable movies/tv folders and then a folder
per first letter, one heatmap per
title beside its report image, a `database.json` written once per batch rather than once per title, and projects named after
the track's work folder -- publish_reviewed_queue() and commit_catalogue() over real local git repos.
'''
import json
import os

import pytest

from pipeline.library.commit import commit_catalogue
from pipeline.orchestrate import Session
from pipeline.publish.catalogue import CategoryFolders
from pipeline.review import project_name, project_paths, publish_reviewed_queue, read_entry, update_entry
from test_pipeline_library_commit import _commits, _out, _queue_entry, _run, repos  # noqa: F401 (a fixture)
from test_pipeline_publish_project import _write_mono_wav

OWNER, IMAGES_NAME = 'o', 'r'


def _files_in(target, sha):
    ''' The paths in a commit (one per line: the names have spaces). '''
    return sorted(_out('git', '-C', target.local_path, 'show', '--name-only', '--format=', sha).splitlines())


def _publish(queue_dir, repos, folders=True, **extra):
    xml, _, images, _ = repos
    return publish_reviewed_queue(queue_dir, xml, images_repo=images, xml_dir='xml', image_dir='img', image_owner=OWNER,
                                  image_repo_name=IMAGES_NAME, category_folders=folders, push=False, heatmap_spec=None,
                                  **extra)


def _tree(target):
    found = []
    for folder, _, names in os.walk(target.local_path):
        if '.git' in folder.split(os.sep):
            continue
        found += [os.path.relpath(os.path.join(folder, n), target.local_path).replace(os.sep, '/') for n in names]
    return sorted(found)


def test_a_title_is_published_under_its_name_in_the_movies_folder(tmp_path, repos):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'jriver-abc-123', 'Heat', readable=True)

    (result,) = _publish(queue_dir, repos)

    assert not result.get('error')
    assert _tree(repos[0]) == ['xml/movies/H/Heat (2018) Atmos.json', 'xml/movies/database.json']
    assert _tree(repos[2]) == ['img/movies/H/Heat (2018) Atmos.png']
    assert read_entry(queue_dir, 'jriver-abc-123').published_stem == 'H/Heat (2018) Atmos'
    assert result['image_url'].endswith('/img/movies/H/Heat%20%282018%29%20Atmos.png')


def test_tv_goes_in_the_tv_folder_and_the_folder_names_are_configurable(tmp_path, repos):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'f', 'Heat', readable=True)
    _queue_entry(queue_dir, 's', 'Show', readable=True)
    update_entry(queue_dir, 's', meta={'title': 'Show', 'year': '2015', 'audio_types': ['DD+'], 'season': '2'})

    _publish(queue_dir, repos, folders=CategoryFolders('Movie BEQs', 'TV Shows BEQ'))

    assert _tree(repos[0]) == ['xml/Movie BEQs/H/Heat (2018) Atmos.json', 'xml/Movie BEQs/database.json',
                               'xml/TV Shows BEQ/S/Show (2015) S02 DD+.json', 'xml/TV Shows BEQ/database.json']


def test_flat_when_category_folders_are_off(tmp_path, repos):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'f', 'Heat', readable=True)

    _publish(queue_dir, repos, folders=False)

    assert _tree(repos[0]) == ['xml/H/Heat (2018) Atmos.json', 'xml/database.json']


def test_two_titles_with_one_name_do_not_overwrite_each_other(tmp_path, repos):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'a', 'Heat', readable=True)
    _queue_entry(queue_dir, 'b', 'Heat', readable=True)

    _publish(queue_dir, repos)

    assert [read_entry(queue_dir, i).published_stem for i in 'ab'] == ['H/Heat (2018) Atmos', 'H/Heat (2018) Atmos (2)']
    assert len([f for f in _tree(repos[0]) if f.endswith('.json') and 'database' not in f]) == 2


def test_a_name_already_in_the_repository_is_not_taken(tmp_path, repos):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'a', 'Heat', readable=True)
    folder = os.path.join(repos[0].local_path, 'xml', 'movies', 'H')
    os.makedirs(folder)
    with open(os.path.join(folder, 'Heat (2018) Atmos.json'), 'w') as f:
        f.write('{"someone": "else"}')

    _publish(queue_dir, repos)

    assert read_entry(queue_dir, 'a').published_stem == 'H/Heat (2018) Atmos (2)'
    with open(os.path.join(folder, 'Heat (2018) Atmos.json')) as f:
        assert json.load(f) == {'someone': 'else'}


def test_editing_the_title_after_publishing_does_not_move_the_file(tmp_path, repos):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'a', 'Heat', readable=True)
    _publish(queue_dir, repos)
    update_entry(queue_dir, 'a', meta={'title': 'Heat Redux', 'year': '2018', 'audio_types': ['Atmos']})

    (result,) = _publish(queue_dir, repos, republish=True)

    assert result['republished'] is True
    assert 'xml/movies/H/Heat (2018) Atmos.json' in _tree(repos[0])
    assert not any('Redux' in f.split('/')[-1] for f in _tree(repos[0]))
    with open(os.path.join(repos[0].local_path, 'xml', 'movies', 'H', 'Heat (2018) Atmos.json')) as f:
        assert json.load(f)['title'] == 'Heat Redux'


def test_a_title_published_before_names_were_readable_stays_at_its_id(tmp_path, repos):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'legacy-id', 'Heat', readable=True)
    _publish(queue_dir, repos)
    # as an old publish left it: its files at the id, no stem recorded
    for target, kind in ((repos[0], 'json'), (repos[2], 'png')):
        folder = os.path.join(target.local_path, 'xml' if kind == 'json' else 'img', 'movies')
        os.replace(os.path.join(folder, 'H', f'Heat (2018) Atmos.{kind}'), os.path.join(folder, f'legacy-id.{kind}'))
    update_entry(queue_dir, 'legacy-id', published_stem=None)
    update_entry(queue_dir, 'legacy-id', meta={'title': 'Heat', 'year': '2018', 'audio_types': ['Atmos'], 'note': 'x'})

    (result,) = _publish(queue_dir, repos, republish=True)

    assert result['republished'] is True
    assert 'xml/movies/legacy-id.json' in _tree(repos[0])
    assert not any('Atmos' in f for f in _tree(repos[0]) if f.endswith('.json'))


def test_the_aggregate_is_read_and_written_once_per_batch_not_once_per_title(tmp_path, repos, monkeypatch):
    import pipeline.orchestrate as orchestrate
    queue_dir = str(tmp_path / 'queue')
    for i in range(6):
        _queue_entry(queue_dir, f'e{i}', f'Title {i}', readable=True)
    reads = []
    real = orchestrate.read_records
    monkeypatch.setattr(orchestrate, 'read_records', lambda *args: reads.append(args) or real(*args))

    results = _publish(queue_dir, repos)

    assert len([r for r in results if not r.get('error')]) == 6
    assert len(reads) == 1
    with open(os.path.join(repos[0].local_path, 'xml', 'movies', 'database.json')) as f:
        assert sorted(r['title'] for r in json.load(f)) == [f'Title {i}' for i in range(6)]


def test_a_cancelled_batch_still_writes_the_aggregate_for_what_it_published(tmp_path, repos):
    queue_dir = str(tmp_path / 'queue')
    for i in range(3):
        _queue_entry(queue_dir, f'e{i}', f'Title {i}', readable=True)
    calls = []

    def cancel():
        calls.append(1)
        return len(calls) > 1   # after the first title

    results = _publish(queue_dir, repos, should_cancel=cancel)

    assert len(results) == 1
    with open(os.path.join(repos[0].local_path, 'xml', 'movies', 'database.json')) as f:
        assert len(json.load(f)) == 1


def test_titles_in_several_letter_folders_share_one_aggregate_in_their_category_folder(tmp_path, repos):
    queue_dir = str(tmp_path / 'queue')
    for entry_id, title in (('a', 'Alien'), ('b', 'Heat'), ('c', '1917'), ('d', 'Élite'), ('e', '[REC]')):
        _queue_entry(queue_dir, entry_id, title, readable=True)

    results = _publish(queue_dir, repos)

    assert not any(r.get('error') for r in results)
    assert _tree(repos[0]) == ['xml/movies/#/[REC] (2018) Atmos.json',
                               'xml/movies/0-9/1917 (2018) Atmos.json', 'xml/movies/A/Alien (2018) Atmos.json',
                               'xml/movies/E/Élite (2018) Atmos.json', 'xml/movies/H/Heat (2018) Atmos.json',
                               'xml/movies/database.json']
    with open(os.path.join(repos[0].local_path, 'xml', 'movies', 'database.json')) as f:
        assert sorted(r['title'] for r in json.load(f)) == ['1917', 'Alien', 'Heat', '[REC]', 'Élite']


def test_a_title_published_before_letter_folders_is_republished_where_it_is(tmp_path, repos):
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'a', 'Heat', readable=True)
    _publish(queue_dir, repos)
    # as a publish before letter folders left it: the file in the category folder, the stem without a letter
    for target, kind in ((repos[0], 'json'), (repos[2], 'png')):
        folder = os.path.join(target.local_path, 'xml' if kind == 'json' else 'img', 'movies')
        os.replace(os.path.join(folder, 'H', f'Heat (2018) Atmos.{kind}'), os.path.join(folder, f'Heat (2018) Atmos.{kind}'))
    update_entry(queue_dir, 'a', published_stem='Heat (2018) Atmos')
    update_entry(queue_dir, 'a', meta={'title': 'Heat', 'year': '2018', 'audio_types': ['Atmos'], 'note': 'x'})

    (result,) = _publish(queue_dir, repos, republish=True)

    assert result['republished'] is True
    assert _tree(repos[0]) == ['xml/movies/Heat (2018) Atmos.json', 'xml/movies/database.json']
    assert _tree(repos[2]) == ['img/movies/Heat (2018) Atmos.png']
    assert read_entry(queue_dir, 'a').published_stem == 'Heat (2018) Atmos'


def test_a_single_publish_into_a_letter_folder_writes_the_aggregate_in_the_folder_it_names(tmp_path, repos):
    from model.codec import filter_from_json
    from pipeline.metadata import BeqMetadata
    xml = repos[0]
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'a', 'Heat', readable=True)
    entry = read_entry(queue_dir, 'a')

    Session().publish(filter_from_json(entry.candidates[0].filters),
                      BeqMetadata(title='Heat', year='2018', audio_types=['Atmos']), xml, 'xml/movies/H/Heat.json',
                      push=False, record_dir='xml/movies')

    assert _tree(xml) == ['xml/movies/H/Heat.json', 'xml/movies/database.json']


def test_a_single_publish_call_still_writes_the_aggregate_by_default(tmp_path, repos):
    from model.codec import filter_from_json
    from pipeline.metadata import BeqMetadata
    xml = repos[0]
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'a', 'Heat', readable=True)
    entry = read_entry(queue_dir, 'a')

    Session().publish(filter_from_json(entry.candidates[0].filters),
                      BeqMetadata(title='Heat', year='2018', audio_types=['Atmos']), xml, 'xml/Heat.json', push=False)

    assert 'xml/database.json' in _tree(xml)


# --- the heatmap ----------------------------------------------------------------------------------------------

def _with_work_dir(tmp_path, queue_dir, entry_id):
    work = tmp_path / 'work'
    (work / entry_id).mkdir(parents=True)
    _write_mono_wav(str(work / entry_id / 'mono.wav'))
    return str(work)


def test_the_heatmap_is_written_beside_the_report_image_and_is_the_records_second_image(tmp_path, repos, monkeypatch):
    import pipeline.review as review
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'a', 'Heat', readable=True)
    work = _with_work_dir(tmp_path, queue_dir, 'a')
    drawn = []
    monkeypatch.setattr(review, 'heatmap_for', lambda sig, flt, spec, title='': drawn.append(title) or b'HEATMAP')
    from pipeline.publish.heatmap import HeatmapSpec

    (result,) = publish_reviewed_queue(
        queue_dir, repos[0], images_repo=repos[2], xml_dir='xml', image_dir='img', image_owner=OWNER,
        image_repo_name=IMAGES_NAME, category_folders=True, push=False, work_dir=work, heatmap_spec=HeatmapSpec())

    assert not result.get('error'), result
    assert drawn == ['Heat']
    assert _tree(repos[2]) == ['img/movies/H/Heat (2018) Atmos heatmap.png', 'img/movies/H/Heat (2018) Atmos.png']
    with open(os.path.join(repos[2].local_path, 'img', 'movies', 'H', 'Heat (2018) Atmos heatmap.png'), 'rb') as f:
        assert f.read() == b'HEATMAP'
    with open(os.path.join(repos[0].local_path, 'xml', 'movies', 'H', 'Heat (2018) Atmos.json')) as f:
        images = json.load(f)['images']
    assert [u.rsplit('/', 1)[1] for u in images] == ['Heat%20%282018%29%20Atmos.png', 'Heat%20%282018%29%20Atmos%20heatmap.png']


def test_a_heatmap_that_cannot_be_drawn_does_not_stop_the_title(tmp_path, repos, monkeypatch):
    import pipeline.review as review
    from pipeline.publish.heatmap import HeatmapSpec
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'a', 'Heat', readable=True)
    work = _with_work_dir(tmp_path, queue_dir, 'a')

    def boom(*args, **kwargs):
        raise RuntimeError('no spectrogram')
    monkeypatch.setattr(review, 'heatmap_for', boom)

    (result,) = publish_reviewed_queue(queue_dir, repos[0], images_repo=repos[2], xml_dir='xml', image_dir='img',
                                       image_owner=OWNER, image_repo_name=IMAGES_NAME, category_folders=True,
                                       push=False, work_dir=work, heatmap_spec=HeatmapSpec())

    assert 'error' not in result and result['heatmap_error'] == 'RuntimeError: no spectrogram'
    assert read_entry(queue_dir, 'a').status == 'published'
    assert _tree(repos[2]) == ['img/movies/H/Heat (2018) Atmos.png']


def test_no_heatmap_without_a_track_or_without_an_images_repo(tmp_path, repos, monkeypatch):
    import pipeline.review as review
    from pipeline.publish.heatmap import HeatmapSpec
    monkeypatch.setattr(review, 'heatmap_for', lambda *a, **k: pytest.fail('drawn'))
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'a', 'Heat', readable=True)

    publish_reviewed_queue(queue_dir, repos[0], images_repo=repos[2], xml_dir='xml', image_dir='img', push=False,
                           image_owner=OWNER, image_repo_name=IMAGES_NAME, heatmap_spec=HeatmapSpec())  # no work_dir
    _queue_entry(queue_dir, 'b', 'Other', readable=True)
    work = _with_work_dir(tmp_path, queue_dir, 'b')
    publish_reviewed_queue(queue_dir, repos[0], xml_dir='xml', push=False, work_dir=work,
                           heatmap_spec=HeatmapSpec())  # no images repo


def test_commit_takes_the_heatmap_with_the_report_image_and_names_it_in_the_commit(tmp_path, repos, monkeypatch):
    import pipeline.review as review
    from pipeline.publish.heatmap import HeatmapSpec
    monkeypatch.setattr(review, 'heatmap_for', lambda *a, **k: b'HEATMAP')
    queue_dir = str(tmp_path / 'queue')
    _queue_entry(queue_dir, 'a', 'Heat', readable=True)
    _queue_entry(queue_dir, 'b', 'Plain', readable=True)
    work = _with_work_dir(tmp_path, queue_dir, 'a')
    publish_reviewed_queue(queue_dir, repos[0], images_repo=repos[2], xml_dir='xml', image_dir='img', image_owner=OWNER,
                           image_repo_name=IMAGES_NAME, category_folders=True, push=False, work_dir=work,
                           heatmap_spec=HeatmapSpec(), ids=['a'])
    publish_reviewed_queue(queue_dir, repos[0], images_repo=repos[2], xml_dir='xml', image_dir='img', image_owner=OWNER,
                           image_repo_name=IMAGES_NAME, category_folders=True, push=False, ids=['b'])

    result = commit_catalogue(queue_dir, repos[0], repos[2], xml_dir='xml', image_dir='img', push=False,
                              category_folders=True)

    assert result.missing == [] and result.not_committed == []
    assert _files_in(repos[2], _commits(repos[2])[0]) == [
        'img/movies/H/Heat (2018) Atmos heatmap.png', 'img/movies/H/Heat (2018) Atmos.png',
        'img/movies/P/Plain (2018) Atmos.png']
    assert _files_in(repos[0], _commits(repos[0])[0]) == [
        'xml/movies/H/Heat (2018) Atmos.json', 'xml/movies/P/Plain (2018) Atmos.json', 'xml/movies/database.json']


# --- projects --------------------------------------------------------------------------------------------------

def test_projects_are_named_after_the_readable_work_folder(tmp_path):
    work = tmp_path / 'work'
    (work / 'Heat - audio 1').mkdir(parents=True)
    (work / 'Heat - audio 1' / '.beq-title-id').write_text('jriver-1-2')

    directory, mono, multichannel, _ = project_paths(str(work), 'jriver-1-2')

    assert os.path.basename(directory) == 'Heat - audio 1'
    assert os.path.basename(mono) == 'Heat - audio 1.mono.beq'
    assert multichannel is None


def test_projects_already_written_under_the_id_keep_it(tmp_path):
    folder = tmp_path / 'work' / 'Heat - audio 1'
    folder.mkdir(parents=True)
    (folder / '.beq-title-id').write_text('jriver-1-2')
    (folder / 'jriver-1-2.mono.beq').write_text('{}')

    assert project_name(str(folder), 'jriver-1-2') == 'jriver-1-2'
