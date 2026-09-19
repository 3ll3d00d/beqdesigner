'''
Discovery (design/library-sync/workflow-rework §12.5/§12.6): scan() reads sources and outputs into the SQLite index
and says what every title needs next, without extracting or designing anything. Each case builds the outputs the
state describes on disk (a manifest, a queue entry, a real temp git repo) and scans, so what is tested is what
`run`/`publish`/`commit` would leave behind, not a hand-made row.
'''
import json
import os
import sqlite3
import subprocess
import time

import pytest

from pipeline.config import AnalysisConfig
from pipeline.library import index as index_module
from pipeline.library.design_cache import design_fingerprint
from pipeline.library.extract_cache import extract_params_hash
from pipeline.library.ignore import IgnoreRule
from pipeline.library.index import INDEX_FILE_NAME, SCHEMA_VERSION, LibraryIndex, index_path
from pipeline.library.profile import Profile, SourceSpec
from pipeline.library.source import LibraryItem
from pipeline.library.state import NEEDS
from pipeline.library.status import ScanSettings, failure_key, unit_fingerprint
from pipeline.publish.git import commit_paths
from pipeline.review import read_entry, update_entry, write_queue_entry
from test_pipeline_library_commit import _publish, _queue_entry, _repo, _track, repos  # noqa: F401 (a fixture)

CONFIG = AnalysisConfig()
DESIGNER = 'test.designer'


class FakeSource:
    ''' A LibrarySource over a fixed list; `error` makes listing fail, as a source that is down does. '''

    def __init__(self, items=(), error=None):
        self.items, self.error, self.listed = list(items), error, 0

    def list_items(self, **query):
        self.listed += 1
        if self.error:
            raise self.error
        return list(self.items)


def _item(name, **overrides):
    fields = dict(id=f'fs-{name}', source_path=f'/films/{name}.mkv', display_name=f'Film {name}', title=f'Film {name}',
                  year='2001', fingerprint=f'fp-{name}-1')
    fields.update(overrides)
    return LibraryItem(**fields)


@pytest.fixture
def env(tmp_path):
    class Env:
        pass

    e = Env()
    e.tmp = tmp_path
    e.work, e.queue = str(tmp_path / 'work'), str(tmp_path / 'queue')
    e.settings = ScanSettings(work_dir=e.work, queue_dir=e.queue, designer=DESIGNER)
    e.index = LibraryIndex(index_path(e.work))
    yield e
    e.index.close()


def _profile(env, *names, ignore=(), ignored_titles=None, **settings):
    return Profile(sources=tuple(SourceSpec(n, 'filesystem', {'globs': ['x']}) for n in names or ('films',)),
                   ignore=tuple(ignore), ignored_titles=dict(ignored_titles or {}), work_dir=env.work,
                   queue_dir=env.queue, **settings)


def _scan(env, *items, sources=None, profile=None, settings=None, **kwargs):
    ''' Scans `items` as the one source `films` (or `sources`, {name: FakeSource}). '''
    sources = sources if sources is not None else {'films': FakeSource(items)}
    profile = profile or _profile(env, *sources)
    return env.index.scan(profile, settings or env.settings, sources=sources, **kwargs)


def _extracted(env, item, fingerprint=None, channels=2):
    ''' What a completed extraction leaves: the manifest and the wav. '''
    directory = os.path.join(env.work, item.id)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, 'mono.wav'), 'wb') as f:
        f.write(b'wav')
    with open(os.path.join(directory, 'manifest.json'), 'w') as f:
        json.dump({'mono_source_fingerprint': fingerprint or item.fingerprint,
                   'mono_params_hash': extract_params_hash(item, CONFIG, True), 'source_channel_count': channels}, f)


def _ready(repos):
    ''' Gives both repos a first commit on a branch that tracks the remote, as a real clone has. '''
    for repo in (repos[0], repos[2]):
        with open(os.path.join(repo.local_path, 'README'), 'w') as f:
            f.write('catalogue')
        commit_paths(repo, ['README'], 'first')
        subprocess.run(['git', '-C', repo.local_path, 'push', '-q', '-u', 'origin', 'HEAD'], check=True,
                       capture_output=True)


def _entry(env, item, status='pending', fingerprint=None, candidates=1, confidence=0.7, **fields):
    ''' What a design leaves: a queue entry carrying the design fingerprint (of the current inputs unless given). '''
    _queue_entry(env.queue, item.id, item.title, status='pending')
    fingerprint = fingerprint or item.fingerprint
    entry = read_entry(env.queue, item.id)
    if candidates:
        entry.candidates = entry.candidates[:candidates]
        entry.candidates[0].confidence = confidence
    else:
        entry.candidates = []
    accepted = status in ('accepted', 'published')
    write_queue_entry(env.queue, entry)
    return update_entry(
        env.queue, item.id, status=status, chosen_candidate_index=0 if accepted else None,
        design_fingerprint=design_fingerprint(item, DESIGNER, CONFIG, 'complete_programme', source=fingerprint),
        source_fingerprint=fingerprint, **fields)


def _row(env, item_id):
    return env.index.title(item_id)


def _needs(env, item_id):
    row = _row(env, item_id)
    return row.needs, row.detail


# --- every row of the needs table, from real outputs -----------------------------------------------------------------

def test_a_new_title_needs_extract(env):
    result = _scan(env, _item('a'))

    assert _needs(env, 'fs-a') == ('extract', 'new')
    assert _row(env, 'fs-a').tier == 'machine'
    assert result.new == ['fs-a'] and result.counts['extract'] == 1


def test_a_stale_extract_after_the_source_changed_needs_extract_again(env):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item)

    _scan(env, _item('a', fingerprint='fp-a-2'))  # re-ripped

    assert _needs(env, 'fs-a') == ('extract', 'source or settings changed')
    assert _row(env, 'fs-a').extract_state == 'stale' and _row(env, 'fs-a').design_state == 'stale'


def test_an_extraction_at_other_settings_is_stale(env):
    item = _item('a')
    _extracted(env, item)

    _scan(env, item, settings=ScanSettings(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER,
                                           config=AnalysisConfig(target_fs=500)))

    assert _row(env, 'fs-a').extract_state == 'stale'


def test_a_current_extract_with_no_entry_needs_design(env):
    item = _item('a')
    _extracted(env, item)

    _scan(env, item)

    assert _needs(env, 'fs-a') == ('design', 'new')
    assert _row(env, 'fs-a').extract_state == 'current'


def test_a_design_at_other_settings_needs_design_again(env):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item)

    _scan(env, item, settings=ScanSettings(work_dir=env.work, queue_dir=env.queue, designer='another'))

    assert _needs(env, 'fs-a') == ('design', 'source or settings changed')


def test_a_current_design_awaiting_review_needs_review_and_shows_its_confidence(env):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item, confidence=0.62)

    _scan(env, item)

    row = _row(env, 'fs-a')
    assert (row.needs, row.tier, row.detail) == ('review', 'human', 'conf 0.62 - 1 candidate')
    assert (row.confidence, row.candidate_count) == (0.62, 1)


def test_a_designer_decline_needs_review_and_says_so(env):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item, candidates=0, decline_reason='no_signal', decline_message='nothing below 20 Hz')

    _scan(env, item)

    assert _needs(env, 'fs-a') == ('review', 'designer declined: nothing below 20 Hz')


def test_an_accepted_title_with_incomplete_metadata_goes_back_to_a_human(env):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item, status='accepted', meta={'title': 'Film a', 'year': '2001'})  # no audio types

    _scan(env, item)

    assert _needs(env, 'fs-a') == ('review', 'metadata incomplete: at least one audio type is required')


def test_an_accepted_title_needs_publish(env):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item, status='accepted')

    _scan(env, item)

    assert _needs(env, 'fs-a')[0] == 'publish'
    assert (_row(env, 'fs-a').review_state, _row(env, 'fs-a').publish_state) == ('accepted', 'not_written')


def _published(env, repos, *names, **extra):
    ''' Publishes `names` (through the real publish_reviewed_queue) and returns settings that know the repos. '''
    xml, _, images, _ = repos
    items = [_item(n) for n in names]
    for item in items:
        _extracted(env, item)
    queue, _ = _publish(env.tmp, repos, *[(i.id, i.title) for i in items], **extra)
    assert queue == env.queue
    for item in items:
        update_entry(env.queue, item.id, source_fingerprint=item.fingerprint)
    settings = ScanSettings(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER, xml_repo=xml.local_path,
                            xml_dir='xml', images_repo=images.local_path, image_dir='img')
    return items, settings


def test_a_written_title_needs_commit_until_it_is_committed_and_then_pushed(env, repos):
    from pipeline.library.sync import commit_library
    xml, _, images, _ = repos
    _ready(repos)
    (item,), settings = _published(env, repos, 'a')

    _scan(env, item, settings=settings)
    assert _needs(env, 'fs-a') == ('commit', 'written, not committed')
    assert (_row(env, 'fs-a').publish_state, _row(env, 'fs-a').commit_state) == ('written', 'uncommitted')

    commit_library(env.queue, xml, images_repo=images, xml_dir='xml', image_dir='img', push=False)
    _scan(env, item, settings=settings)
    assert _needs(env, 'fs-a') == ('commit', 'committed, not pushed')

    commit_library(env.queue, xml, images_repo=images, xml_dir='xml', image_dir='img', push=True)
    _scan(env, item, settings=settings)
    assert _needs(env, 'fs-a') == ('done', 'pushed')
    assert _row(env, 'fs-a').tier == 'done'


def test_an_image_that_is_not_committed_holds_the_title_at_commit(env, repos):
    from pipeline.publish.git import commit_paths, push
    xml, _, images, _ = repos
    _ready(repos)
    (item,), settings = _published(env, repos, 'a')
    commit_paths(xml, ['xml/fs-a.xml'], 'xml only')
    push(xml)

    _scan(env, item, settings=settings)

    assert _needs(env, 'fs-a') == ('commit', 'written, not committed')  # the XML is pushed, the image is not


def test_a_repo_with_no_upstream_leaves_the_push_state_unknown(env, repos):
    from pipeline.publish.git import commit_paths
    xml, _, images, _ = repos
    (item,), settings = _published(env, repos, 'a')
    commit_paths(xml, ['xml/fs-a.xml'], 'x'), commit_paths(images, ['img/fs-a.png'], 'x')

    _scan(env, item, settings=settings)

    assert _row(env, 'fs-a').commit_state == 'unknown'
    assert _needs(env, 'fs-a')[0] == 'commit' and 'upstream' in _row(env, 'fs-a').detail


def test_a_metadata_edit_on_a_published_title_makes_publish_out_of_date(env, repos):
    xml, _, images, _ = repos
    _ready(repos)
    (item,), settings = _published(env, repos, 'a')
    _scan(env, item, settings=settings)
    assert _row(env, 'fs-a').publish_state == 'written'
    entry = read_entry(env.queue, 'fs-a')

    update_entry(env.queue, 'fs-a', meta={**entry.meta, 'title': 'A Typo Fixed'})
    _scan(env, item, settings=settings)

    assert _needs(env, 'fs-a') == ('publish', 'changed since it was published')
    assert _row(env, 'fs-a').publish_state == 'out_of_date'


def test_the_current_digest_matches_what_publish_recorded_even_with_a_project_and_defaults(env, repos):
    ''' Guards the factoring of the digest out of publish_reviewed_queue(): if it drifted, every title would look stale. '''
    from pipeline.review import current_publish_digest
    xml, _, images, _ = repos
    (item,), settings = _published(env, repos, 'a', meta_defaults={'source': 'Blu-ray'})
    entry = read_entry(env.queue, 'fs-a')

    assert current_publish_digest(entry, meta_defaults={'source': 'Blu-ray'}, has_image=True) == entry.published_digest
    assert current_publish_digest(entry, meta_defaults={'source': 'Disc'}, has_image=True) != entry.published_digest
    assert current_publish_digest(entry, meta_defaults={'source': 'Blu-ray'}, has_image=False) != entry.published_digest


def test_a_published_title_whose_file_left_the_repository_needs_publish(env, repos):
    xml, _, images, _ = repos
    (item,), settings = _published(env, repos, 'a')
    os.remove(os.path.join(xml.local_path, 'xml', 'fs-a.xml'))

    _scan(env, item, settings=settings)

    assert _needs(env, 'fs-a') == ('publish', 'its file is missing from the repository')


def test_an_entry_published_before_digests_were_recorded_is_not_called_out_of_date(env, repos):
    xml, _, images, _ = repos
    _ready(repos)
    (item,), settings = _published(env, repos, 'a')
    update_entry(env.queue, 'fs-a', published_digest=None)

    _scan(env, item, settings=settings)

    assert _row(env, 'fs-a').publish_state == 'written'


@pytest.mark.parametrize('status', ['skipped', 'rejected'])
def test_a_skipped_or_rejected_title_is_done(env, status):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item, status=status)

    _scan(env, item)

    assert _needs(env, 'fs-a') == ('done', status)


def test_a_remembered_failure_needs_attention_until_the_source_or_settings_change(env):
    item = _item('a')
    key = failure_key('extract', item, config=CONFIG, designer=DESIGNER, coverage='complete_programme',
                      keep_multichannel=False)
    env.index.record_failure('fs-a', 'extract', 'ValueError: file not found (path mapping?)', 'fp-a-1', key)

    _scan(env, item)
    assert _needs(env, 'fs-a') == ('attention', 'extract failed: ValueError: file not found (path mapping?)')
    assert _row(env, 'fs-a').extract_state == 'failed' and _row(env, 'fs-a').failure

    _scan(env, item)  # nothing changed: still failed, not retried
    assert _row(env, 'fs-a').needs == 'attention'

    _scan(env, _item('a', fingerprint='fp-a-2'))  # the source changed: the failure no longer applies
    assert _needs(env, 'fs-a') == ('extract', 'new')
    assert 'fs-a' not in env.index.failures()  # and it is forgotten, not kept for ever


def test_a_design_failure_is_remembered_against_the_designer_and_settings(env):
    item = _item('a')
    _extracted(env, item)
    key = failure_key('design', item, config=CONFIG, designer=DESIGNER, coverage='complete_programme',
                      keep_multichannel=False)
    env.index.record_failure('fs-a', 'design', 'HTTPError: 500', 'fp-a-1', key)

    _scan(env, item)
    assert _needs(env, 'fs-a') == ('attention', 'design failed: HTTPError: 500')

    _scan(env, item, settings=ScanSettings(work_dir=env.work, queue_dir=env.queue, designer='another'))
    assert _needs(env, 'fs-a')[0] == 'design'


def test_clearing_a_failure_is_how_retry_failed_works(env):
    item = _item('a')
    env.index.record_failure('fs-a', 'extract', 'boom', 'fp-a-1', failure_key(
        'extract', item, config=CONFIG, designer=DESIGNER, coverage='complete_programme', keep_multichannel=False))
    _scan(env, item)

    env.index.clear_failure('fs-a')
    _scan(env, item)

    assert _needs(env, 'fs-a') == ('extract', 'new')


def test_a_source_changed_since_accepted_needs_attention_even_after_publishing(env, repos):
    xml, _, images, _ = repos
    _ready(repos)
    (item,), settings = _published(env, repos, 'a')

    _scan(env, _item('a', fingerprint='fp-a-2'), settings=settings)  # re-ripped after it was published

    assert _needs(env, 'fs-a') == ('attention', 'source changed since published')


def test_a_source_changed_since_accepted_but_not_written_says_so(env):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item, status='accepted')

    _scan(env, _item('a', fingerprint='fp-a-2'))

    assert _needs(env, 'fs-a') == ('attention', 'source changed since accepted')


def test_an_accepted_entry_without_a_recorded_source_fingerprint_cannot_be_called_changed(env):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item, status='accepted')
    update_entry(env.queue, 'fs-a', source_fingerprint=None)  # designed before the field existed

    _scan(env, _item('a', fingerprint='fp-a-2'))

    assert _needs(env, 'fs-a')[0] == 'publish'


def test_a_settings_change_does_not_pull_an_accepted_title_back_into_the_list(env):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item, status='accepted')

    _scan(env, item, settings=ScanSettings(work_dir=env.work, queue_dir=env.queue, designer='another',
                                           config=AnalysisConfig(target_fs=500)))

    assert _needs(env, 'fs-a')[0] == 'publish'  # not extract/design: only a source change re-enters (§12.6)
    assert _row(env, 'fs-a').design_state == 'protected'


def test_a_project_conflict_needs_attention(env, tmp_path):
    from test_pipeline_publish_project import _OTHER_HUMAN_FILTER, _HUMAN_FILTER, _hand_edit_filter
    from pipeline.config import AnalysisConfig as Config
    from pipeline.orchestrate import Session
    from pipeline.publish.project import write_mono_project, write_multichannel_project
    from test_pipeline_publish_project import _write_mono_wav, _write_multichannel_wav
    item = _item('a')
    _extracted(env, item)
    _entry(env, item, status='accepted')
    directory = os.path.join(env.work, 'fs-a')
    _write_mono_wav(os.path.join(directory, 'mono.wav'))
    _write_multichannel_wav(os.path.join(directory, 'multichannel.wav'), (1000, 2000))
    session = Session(Config())
    mono, multichannel = os.path.join(directory, 'fs-a.mono.beq'), os.path.join(directory, 'fs-a.multichannel.beq')
    write_mono_project(session, os.path.join(directory, 'mono.wav'), _HUMAN_FILTER, mono)
    write_multichannel_project(session, os.path.join(directory, 'multichannel.wav'), _HUMAN_FILTER, 'stereo',
                               multichannel)
    _hand_edit_filter(mono, _HUMAN_FILTER)
    _hand_edit_filter(multichannel, _OTHER_HUMAN_FILTER)

    _scan(env, item)

    assert _needs(env, 'fs-a')[0] == 'attention' and 'disagree' in _row(env, 'fs-a').detail


# --- flags -------------------------------------------------------------------------------------------------------

def test_a_title_matching_an_ignore_rule_is_ignored_and_labelled_with_it(env):
    _scan(env, _item('a'), _item('kid', source_path='/films/Kids/kid.mkv'),
          profile=_profile(env, 'films', ignore=[IgnoreRule(path='/films/Kids/**', reason='not for the catalogue')]))

    row = _row(env, 'fs-kid')
    assert (row.needs, row.tier, row.flags) == ('done', 'done', ['Ignored'])
    assert row.ignored == 'ignored by rule: path /films/Kids/** (not for the catalogue)'
    assert _row(env, 'fs-a').flags == []


def test_deleting_the_rule_brings_the_title_back_on_the_next_scan(env):
    items = [_item('kid', source_path='/films/Kids/kid.mkv')]
    _scan(env, *items, profile=_profile(env, 'films', ignore=[IgnoreRule(path='/films/Kids')]))
    assert _row(env, 'fs-kid').needs == 'done'

    _scan(env, *items)

    assert _needs(env, 'fs-kid') == ('extract', 'new') and _row(env, 'fs-kid').flags == []


def test_a_title_ignored_by_id_is_ignored_with_the_reason(env):
    _scan(env, _item('a'), profile=_profile(env, 'films', ignored_titles={'fs-a': 'bad rip'}))

    assert _row(env, 'fs-a').ignored == 'ignored by you: bad rip'
    assert _row(env, 'fs-a').flags == ['Ignored']


def test_the_same_file_in_two_sources_is_one_title_and_the_other_is_shadowed(env):
    mine = _item('a', id='fs-mine', source_path='/films/a.mkv')
    theirs = _item('a', id='jriver-x-9', source_path='\\FILMS\\a.mkv'.replace('\\', '/'))

    _scan(env, sources={'films': FakeSource([mine]), 'jriver': FakeSource([theirs])})

    owner, shadow = _row(env, 'fs-mine'), _row(env, 'jriver-x-9')
    assert (owner.source, owner.also_in, owner.needs) == ('films', ('jriver',), 'extract')
    assert (shadow.flags, shadow.needs, shadow.tier, shadow.shadowed_by) == (['Shadowed'], 'done', 'done', 'fs-mine')
    assert 'fs-mine' in shadow.detail and shadow.source == 'jriver'


def test_two_files_for_the_same_title_are_flagged_as_possible_duplicates_and_both_stay(env):
    one = _item('a', external_ids={'tmdb': '603'})
    two = _item('a-extended', external_ids={'tmdb': '603'}, source_path='/films/a-extended.mkv')

    _scan(env, one, two)

    assert _row(env, 'fs-a').flags == ['Possible duplicate'] and _row(env, 'fs-a').duplicates == ('fs-a-extended',)
    assert _row(env, 'fs-a-extended').flags == ['Possible duplicate']
    assert _row(env, 'fs-a').needs == _row(env, 'fs-a-extended').needs == 'extract'


def test_a_title_that_left_its_source_with_outputs_is_kept_as_done_and_gone(env):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item)
    _scan(env, item)
    assert _needs(env, 'fs-a')[0] == 'review'

    result = _scan(env)  # the library no longer has it

    row = _row(env, 'fs-a')
    assert (row.needs, row.flags, row.detail) == ('done', ['Gone'], 'gone from source')
    assert result.gone == ['fs-a']
    assert row.review_state == 'pending'  # its state is as it was; it is just no longer this catalogue's work
    assert _scan(env).gone == []  # already reported once


def test_a_title_that_left_its_source_with_no_outputs_is_dropped(env):
    _scan(env, _item('a'), _item('b'))

    result = _scan(env, _item('b'))

    assert result.dropped == ['fs-a'] and _row(env, 'fs-a') is None


def _catalogue_xml(repo, name, tmdb, season=None):
    os.makedirs(repo, exist_ok=True)
    body = f'<beq_metadata><beq_theMovieDB>{tmdb}</beq_theMovieDB><beq_season>{season or ""}</beq_season></beq_metadata>'
    with open(os.path.join(repo, name), 'w') as f:
        f.write(f'<?xml version="1.0"?><filter>{body}</filter>')


def test_a_title_whose_tmdb_id_someone_else_published_is_already_in_the_catalogue(env):
    repo = str(env.tmp / 'xmlrepo')
    _catalogue_xml(repo, 'Heat (1995).xml', '949')
    settings = ScanSettings(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER, xml_repo=repo)

    _scan(env, _item('heat', external_ids={'tmdb': '949'}), _item('other', external_ids={'tmdb': '1'}),
          settings=settings)

    assert _row(env, 'fs-heat').flags == ['Already in catalogue']
    assert _row(env, 'fs-heat').needs == 'extract'  # informational: it stays in the list
    assert _row(env, 'fs-other').flags == []


def test_a_film_and_a_series_sharing_a_tmdb_id_are_not_confused(env):
    repo = str(env.tmp / 'xmlrepo')
    _catalogue_xml(repo, 'series.xml', '2316', season='1')  # a TV entry
    settings = ScanSettings(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER, xml_repo=repo)

    _scan(env, _item('film', external_ids={'tmdb': '2316'}), _item('show', kind='tv', season='1', episodes=(1,),
                                                                     external_ids={'tmdb': '2316'}),
          settings=settings)

    assert _row(env, 'fs-film').flags == []
    assert _row(env, 'fs-show').flags == ['Already in catalogue']


def test_what_this_profile_published_is_not_already_in_the_catalogue(env):
    repo = str(env.tmp / 'xmlrepo')
    _catalogue_xml(repo, 'fs-mine.xml', '949')  # our own entry id
    settings = ScanSettings(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER, xml_repo=repo)

    _scan(env, _item('mine', external_ids={'tmdb': '949'}), settings=settings)

    assert _row(env, 'fs-mine').flags == []


def test_an_xml_with_no_tmdb_id_can_never_match_and_a_bad_xml_is_skipped(env):
    repo = str(env.tmp / 'xmlrepo')
    _catalogue_xml(repo, 'blank.xml', '')
    os.makedirs(repo, exist_ok=True)
    with open(os.path.join(repo, 'broken.xml'), 'w') as f:
        f.write('<not closed')
    settings = ScanSettings(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER, xml_repo=repo)

    _scan(env, _item('a', external_ids={'tmdb': ''}), settings=settings)

    assert _row(env, 'fs-a').flags == []


def test_an_unchanged_catalogue_xml_is_not_parsed_again(env, monkeypatch):
    from pipeline import library
    from pipeline.library import catalogue_scan
    repo = str(env.tmp / 'xmlrepo')
    _catalogue_xml(repo, 'a.xml', '1')
    settings = ScanSettings(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER, xml_repo=repo)
    _scan(env, _item('a'), settings=settings)
    parsed = []
    real = catalogue_scan.parse_xml
    monkeypatch.setattr(catalogue_scan, 'parse_xml', lambda path: parsed.append(path) or real(path))

    _scan(env, _item('a'), settings=settings)
    assert parsed == []
    _catalogue_xml(repo, 'a.xml', '22222')  # a different size
    _scan(env, _item('a'), settings=settings)
    assert len(parsed) == 1


# --- sources: errors, last scanned, partial rescans -------------------------------------------------------------------

def test_a_source_that_is_down_keeps_its_titles_and_reports_why(env):
    down = FakeSource([_item('b', id='jriver-b')])
    _scan(env, sources={'films': FakeSource([_item('a')]), 'jriver': down})
    assert {r.id for r in env.index.titles()} == {'fs-a', 'jriver-b'}

    down.error = ConnectionError('server unreachable')
    result = _scan(env, sources={'films': FakeSource([_item('a')]), 'jriver': down})

    assert result.errors == {'jriver': 'ConnectionError: server unreachable'}
    assert {r.id for r in env.index.titles()} == {'fs-a', 'jriver-b'}  # nothing vanished
    assert _row(env, 'jriver-b').gone is False
    sources = {s.name: s for s in env.index.sources()}
    assert sources['films'].last_error == '' and sources['films'].last_ok == sources['films'].last_scanned
    assert 'server unreachable' in sources['jriver'].last_error
    assert sources['jriver'].last_ok < sources['jriver'].last_scanned
    assert sources['jriver'].item_count == 1  # what it last listed


def test_a_source_never_listed_successfully_has_no_titles_and_an_error(env):
    _scan(env, sources={'films': FakeSource(error=OSError('no route'))})

    assert env.index.titles() == []
    assert 'no route' in env.index.sources()[0].last_error and env.index.sources()[0].last_ok is None


def test_only_the_named_source_is_rescanned(env):
    films, other = FakeSource([_item('a')]), FakeSource([_item('b', id='jriver-b')])
    _scan(env, sources={'films': films, 'jriver': other}, now=100.0)
    films.items = [_item('a'), _item('c')]
    other.items = []  # would drop jriver-b if it were listed again

    result = _scan(env, sources={'films': films, 'jriver': other}, only=['films'], now=200.0)

    assert other.listed == 1 and films.listed == 2
    assert {r.id for r in env.index.titles()} == {'fs-a', 'fs-c', 'jriver-b'}
    assert result.new == ['fs-c']
    scanned = {s.name: s.last_scanned for s in env.index.sources()}
    assert scanned == {'films': 200.0, 'jriver': 100.0}


def test_a_partial_rescan_still_merges_against_the_sources_it_did_not_list(env):
    mine = _item('a', id='fs-a', source_path='/films/a.mkv')
    theirs = _item('a', id='jriver-a', source_path='/films/a.mkv')
    films, jriver = FakeSource([mine]), FakeSource([theirs])
    _scan(env, sources={'films': films, 'jriver': jriver})

    _scan(env, sources={'films': films, 'jriver': jriver}, only=['films'])

    assert _row(env, 'jriver-a').shadowed_by == 'fs-a' and _row(env, 'fs-a').also_in == ('jriver',)


def test_sources_are_recorded_in_priority_order_and_removed_with_the_profile(env):
    _scan(env, sources={'b': FakeSource(), 'a': FakeSource()})
    assert [s.name for s in env.index.sources()] == ['b', 'a']

    _scan(env, sources={'a': FakeSource()})

    assert [s.name for s in env.index.sources()] == ['a']


# --- state_since, new-since-scan, ordering, queries -------------------------------------------------------------------

def test_state_since_holds_while_the_state_holds_and_moves_when_it_changes(env):
    item = _item('a')
    _scan(env, item, now=100.0)
    _scan(env, item, now=200.0)
    assert _row(env, 'fs-a').state_since == 100.0

    _extracted(env, item)
    _scan(env, item, now=300.0)

    assert (_row(env, 'fs-a').needs, _row(env, 'fs-a').state_since) == ('design', 300.0)


def test_titles_first_seen_by_the_latest_scan_are_marked_new_until_the_next(env):
    _scan(env, _item('a'))
    _scan(env, _item('a'), _item('b'))

    assert {r.id for r in env.index.titles(new_only=True)} == {'fs-b'}
    assert _row(env, 'fs-b').is_new and not _row(env, 'fs-a').is_new
    assert env.index.summary().new == 1

    _scan(env, _item('a'), _item('b'))
    assert env.index.titles(new_only=True) == []


def test_the_work_list_is_ordered_by_tier_then_oldest_first(env):
    review, extract, done = _item('review'), _item('extract'), _item('done')
    for item in (review, done):
        _extracted(env, item)
    _entry(env, review)
    _entry(env, done, status='skipped')
    _scan(env, review, extract, done, now=100.0)
    failing = _item('bad')
    env.index.record_failure('fs-bad', 'extract', 'x', 'fp-bad-1', failure_key(
        'extract', failing, config=CONFIG, designer=DESIGNER, coverage='complete_programme', keep_multichannel=False))
    _extracted(env, extract)
    _scan(env, review, extract, done, failing, now=200.0)  # extract moves to design at 200

    assert [(r.id, r.tier) for r in env.index.titles()] == [
        ('fs-bad', 'attention'), ('fs-review', 'human'), ('fs-extract', 'machine'), ('fs-done', 'done')]


def test_queries_filter_by_needs_source_text_ids_and_done(env):
    _extracted(env, _item('a'))
    _entry(env, _item('a'))
    _entry(env, _item('z'), status='rejected')
    _scan(env, sources={'films': FakeSource([_item('a'), _item('z')]), 'jriver': FakeSource([
        _item('b', id='jriver-b', source_path='/x/b.mkv', title='Blade Runner')])})

    assert [r.id for r in env.index.titles(needs='review')] == ['fs-a']
    assert {r.id for r in env.index.titles(needs=['review', 'extract'])} == {'fs-a', 'jriver-b'}
    assert [r.id for r in env.index.titles(source='jriver')] == ['jriver-b']
    assert [r.id for r in env.index.titles(match='blade')] == ['jriver-b']
    assert [r.id for r in env.index.titles(match='100%')] == []  # % is text, not a wildcard
    assert {r.id for r in env.index.titles(ids=['fs-a', 'fs-z'])} == {'fs-a', 'fs-z'}
    assert 'fs-z' not in {r.id for r in env.index.titles(include_done=False)}
    assert env.index.titles(tier='done')[0].id == 'fs-z'


def test_the_summary_counts_every_needs_and_flag(env):
    _scan(env, _item('a'), _item('b'), _item('kid', source_path='/films/Kids/k.mkv'),
          profile=_profile(env, 'films', ignore=[IgnoreRule(path='/films/Kids')]))

    summary = env.index.summary()

    assert summary.titles == 3 and list(summary.counts) == list(NEEDS)
    assert summary.counts['extract'] == 2 and summary.counts['done'] == 1 and summary.counts['review'] == 0
    assert summary.flags['Ignored'] == 1 and summary.new == 3 and summary.generation == 1
    assert [s.name for s in summary.sources] == ['films']


def test_a_multichannel_extract_that_is_missing_makes_the_extract_stale(env):
    item = _item('a')
    _extracted(env, item, channels=6)
    settings = ScanSettings(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER, keep_multichannel=True)

    _scan(env, item, settings=settings)
    assert _row(env, 'fs-a').extract_state == 'stale'

    _scan(env, item, settings=ScanSettings(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER))
    assert _row(env, 'fs-a').extract_state == 'current'  # not asked to keep it


# --- seasons -----------------------------------------------------------------------------------------------------------

def _episode(number, **overrides):
    return _item(f'show-e{number}', title='Some Show', kind='tv', season='1', episodes=(number,),
                 source_path=f'/tv/show/e{number}.mkv', **overrides)


def test_in_season_mode_a_season_is_one_title_whose_extract_is_its_episodes(env):
    from pipeline.library.season import plan_units
    episodes = [_episode(1), _episode(2)]
    (group,) = plan_units(episodes, 'season')
    settings = ScanSettings(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER, tv_mode='season')

    _scan(env, *episodes, settings=settings)
    row = _row(env, group.item.id)
    assert (row.unit, row.needs, row.members) == ('season', 'extract', ('fs-show-e1', 'fs-show-e2'))
    assert row.season == '1' and row.episodes == (1, 2) and len(env.index.titles()) == 1

    for episode in episodes:
        _extracted(env, episode)
    _scan(env, *episodes, settings=settings)
    assert _row(env, group.item.id).extract_state == 'current'
    assert _row(env, group.item.id).needs == 'design'

    _scan(env, _episode(1), _episode(2, fingerprint='fp-changed'), settings=settings)
    assert _row(env, group.item.id).extract_state == 'stale'


def test_a_season_with_one_episode_that_never_extracts_is_not_stuck_needing_extract(env):
    from pipeline.library.season import plan_units
    episodes = [_episode(1), _episode(2)]
    (group,) = plan_units(episodes, 'season')
    _extracted(env, episodes[0])  # the second never extracted: a run leaves it out
    settings = ScanSettings(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER, tv_mode='season')

    _scan(env, *episodes, settings=settings)

    assert _row(env, group.item.id).extract_state == 'current'


def test_in_episode_mode_each_episode_is_its_own_title(env):
    _scan(env, _episode(1), _episode(2))

    assert {r.unit for r in env.index.titles()} == {'item'} and len(env.index.titles()) == 2


# --- the index is a cache -------------------------------------------------------------------------------------------------

def _states(index):
    return {r.id: (r.needs, r.tier, r.detail, r.extract_state, r.design_state, r.review_state, r.publish_state,
                   r.commit_state, r.confidence, r.candidate_count, r.title, r.year) for r in index.titles()}


def test_deleting_the_index_and_rebuilding_from_outputs_gives_the_same_states(env, repos):
    from pipeline.library.sync import commit_library
    xml, _, images, _ = repos
    _ready(repos)
    published, settings = _published(env, repos, 'pub')
    (pub,) = published
    pending, accepted, skipped, declined = _item('pending'), _item('accepted'), _item('skipped'), _item('declined')
    for item in (pending, accepted, skipped, declined):
        _extracted(env, item)
    _entry(env, pending, confidence=0.4)
    _entry(env, accepted, status='accepted')
    _entry(env, skipped, status='skipped')
    _entry(env, declined, candidates=0, decline_reason='no_signal', decline_message='nothing below 20 Hz')
    commit_library(env.queue, xml, images_repo=images, xml_dir='xml', image_dir='img', push=False)
    _scan(env, pub, pending, accepted, skipped, declined, settings=settings, now=100.0)
    before = _states(env.index)
    assert {v[0] for v in before.values()} == {'commit', 'review', 'publish', 'done'}
    env.index.close()

    os.remove(index_path(env.work))
    with LibraryIndex(index_path(env.work)) as fresh:
        assert fresh.generation == 0 and fresh.titles() == []
        assert fresh.rebuild_from_outputs(settings, now=500.0) == 5
        after = _states(fresh)
        assert after == before
        assert {r.state_since for r in fresh.titles()} == {500.0}  # accepted: state_since restarts at the rebuild
        assert fresh.generation == 0 and not any(r.is_new for r in fresh.titles())  # nothing is "new" after a rebuild
        assert {r.source for r in fresh.titles()} == {''}  # nobody knows the source until the next scan

        fresh.scan(_profile(env, 'films'), settings, sources={
            'films': FakeSource([pub, pending, accepted, skipped, declined])}, now=600.0)
        assert _states(fresh) == before  # the scan after a rebuild agrees with it, and attaches the sources
        assert {r.source for r in fresh.titles()} == {'films'}
        assert {r.state_since for r in fresh.titles()} == {500.0}  # (nothing changed state, so none moved)
    env.index = LibraryIndex(index_path(env.work))


def test_a_rebuild_reports_a_title_gone_at_the_next_scan_if_its_source_no_longer_lists_it(env):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item)
    env.index.rebuild_from_outputs(env.settings)

    result = _scan(env)

    assert result.gone == ['fs-a'] and _row(env, 'fs-a').flags == ['Gone']


def test_a_rebuild_leaves_out_what_has_no_queue_entry(env):
    _extracted(env, _item('a'))  # extracted, never designed

    assert env.index.rebuild_from_outputs(env.settings) == 0


# --- the schema ------------------------------------------------------------------------------------------------------------

def test_the_schema_is_versioned_by_pragma_user_version(env):
    db = sqlite3.connect(index_path(env.work))
    try:
        assert db.execute('PRAGMA user_version').fetchone()[0] == SCHEMA_VERSION == 1
        titles = [r[1] for r in db.execute('PRAGMA table_info(titles)')]
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        db.close()

    assert titles == list(index_module._TITLE_COLUMNS)
    assert tables == {'meta', 'sources', 'titles', 'failures', 'repo_xml'}


def test_an_index_with_another_schema_version_is_dropped_and_starts_afresh(env):
    _scan(env, _item('a'))
    env.index.close()
    db = sqlite3.connect(index_path(env.work))
    db.execute(f'PRAGMA user_version = {SCHEMA_VERSION + 1}')
    db.commit(), db.close()

    with LibraryIndex(index_path(env.work)) as fresh:
        assert fresh.titles() == [] and fresh.generation == 0
        assert sqlite3.connect(index_path(env.work)).execute('PRAGMA user_version').fetchone()[0] == SCHEMA_VERSION
    env.index = LibraryIndex(index_path(env.work))


def test_a_file_that_is_not_a_database_is_replaced(tmp_path):
    path = tmp_path / INDEX_FILE_NAME
    path.write_bytes(b'this is not sqlite' * 100)

    with LibraryIndex(str(path)) as index:
        assert index.titles() == []


def test_the_index_persists_between_openings(env):
    _scan(env, _item('a'))
    env.index.close()

    with LibraryIndex(index_path(env.work)) as again:
        assert [r.id for r in again.titles()] == ['fs-a'] and again.generation == 1
    env.index = LibraryIndex(index_path(env.work))


def test_a_scan_never_writes_to_the_outputs(env):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item)
    before = {p: os.stat(p).st_mtime_ns for root, _, names in os.walk(env.tmp)
              for p in (os.path.join(root, n) for n in names) if not p.endswith(INDEX_FILE_NAME)}

    _scan(env, item, _item('b'))

    after = {p: os.stat(p).st_mtime_ns for root, _, names in os.walk(env.tmp)
             for p in (os.path.join(root, n) for n in names) if not p.endswith(INDEX_FILE_NAME)}
    assert after == before


# --- failure memory from a run ---------------------------------------------------------------------------------------------

def test_run_library_remembers_a_failure_against_the_source_and_settings_and_forgets_it_on_success(env, monkeypatch):
    from pipeline.library.run import LibraryRunConfig, run_library

    def boom(session, item, *args, **kwargs):
        raise FileNotFoundError('W:\\films\\a.mkv')

    monkeypatch.setattr('pipeline.library.run.extract_if_needed', boom)
    run_config = LibraryRunConfig(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER)
    item = _item('a')

    report = run_library(FakeSource([item]), run_config, index=env.index)

    assert report.failed == [('fs-a', "FileNotFoundError: W:\\films\\a.mkv")]
    memory = env.index.failures()['fs-a']
    assert (memory.stage, memory.fingerprint) == ('extract', 'fp-a-1')
    assert memory.key == failure_key('extract', item, config=CONFIG, designer=DESIGNER,
                                     coverage='complete_programme', keep_multichannel=False)
    _scan(env, item)
    assert _needs(env, 'fs-a') == ('attention', "extract failed: FileNotFoundError: W:\\films\\a.mkv")

    monkeypatch.setattr('pipeline.library.run.extract_if_needed', lambda *a, **k: ('/w/mono.wav', True))
    monkeypatch.setattr('pipeline.library.run.design_if_needed',
                        lambda *a, **k: type('R', (), {'designed': True, 'project_edit_preserved': False})())
    assert run_library(FakeSource([item]), run_config, index=env.index).failed == []
    assert env.index.failures() == {}


def test_a_design_failure_is_recorded_as_a_design_failure(env, monkeypatch):
    from pipeline.library.run import LibraryRunConfig, run_library
    monkeypatch.setattr('pipeline.library.run.extract_if_needed', lambda *a, **k: ('/w/mono.wav', True))

    def boom(*args, **kwargs):
        raise RuntimeError('designer said no')

    monkeypatch.setattr('pipeline.library.run.design_if_needed', boom)

    run_library(FakeSource([_item('a')]), LibraryRunConfig(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER),
                index=env.index)

    assert env.index.failures()['fs-a'].stage == 'design'


def test_a_broken_index_never_sinks_a_run(env, monkeypatch):
    from pipeline.library.run import LibraryRunConfig, run_library

    class Broken:
        def record_failure(self, *args, **kwargs):
            raise sqlite3.OperationalError('database is locked')

        def clear_failure(self, *args):
            raise sqlite3.OperationalError('database is locked')

    monkeypatch.setattr('pipeline.library.run.extract_if_needed', lambda *a, **k: (_ for _ in ()).throw(ValueError('x')))
    report = run_library(FakeSource([_item('a')]), LibraryRunConfig(work_dir=env.work, queue_dir=env.queue,
                                                                       designer=DESIGNER), index=Broken())

    assert report.failed == [('fs-a', 'ValueError: x')]


def test_a_season_failure_is_remembered_against_the_season(env, monkeypatch):
    from pipeline.library.run import LibraryRunConfig, run_library
    from pipeline.library.season import plan_units
    episodes = [_episode(1), _episode(2)]
    (group,) = plan_units(episodes, 'season')

    def boom(*args, **kwargs):
        raise RuntimeError('cannot extract')

    monkeypatch.setattr('pipeline.library.run.extract_if_needed', boom)
    run_library(FakeSource(episodes), LibraryRunConfig(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER,
                                                       tv_mode='season'), index=env.index)

    memory = env.index.failures()[group.item.id]
    assert memory.stage == 'extract' and memory.fingerprint == unit_fingerprint(group)
    _scan(env, *episodes, settings=ScanSettings(work_dir=env.work, queue_dir=env.queue, designer=DESIGNER,
                                                tv_mode='season'))
    assert _needs(env, group.item.id)[0] == 'attention'


# --- performance ---------------------------------------------------------------------------------------------------------------

def test_a_5000_item_scan_and_a_rescan_stay_within_a_budget(env):
    items = [_item(f'{n:05d}', source_path=f'/films/{n // 100:03d}/{n:05d}.mkv') for n in range(5000)]
    # some outputs, as a real library has: extracted, designed, some decided
    for item in items[:100]:
        _extracted(env, item)
    for item in items[:50]:
        _entry(env, item)

    started = time.perf_counter()
    result = _scan(env, *items)
    first = time.perf_counter() - started
    started = time.perf_counter()
    _scan(env, *items)
    second = time.perf_counter() - started
    listed = env.index.titles()

    assert result.titles == 5000 and len(listed) == 5000
    assert result.counts['review'] == 50 and result.counts['design'] == 50 and result.counts['extract'] == 4900
    assert first < 6 and second < 6, (first, second)
    started = time.perf_counter()
    env.index.titles(needs='extract')
    env.index.summary()
    assert time.perf_counter() - started < 3


def test_the_frozen_schema_in_the_design_doc_is_the_one_in_the_code():
    ''' design.md §12.5 is where the schema is frozen for the UI chunks; it must not drift from index.SCHEMA. '''
    import re
    doc = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'design', 'library-sync', 'workflow-rework',
                       'design.md')
    with open(doc, encoding='utf-8') as f:
        text = f.read()
    blocks = re.findall(r'```sql\n(.*?)```', text, re.S)

    assert [b.strip() for b in blocks] == [index_module.SCHEMA.strip()]
    assert f'SCHEMA_VERSION = {SCHEMA_VERSION}' in text
