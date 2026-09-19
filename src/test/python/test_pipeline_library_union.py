'''
The union of several sources (design/library-sync/workflow-rework §12.4): hard and soft clashes, priority, sticky
ownership, ignore rules, claims read back from disk, and the LibrarySource wrapper run_library() uses.
'''
import os

import pytest

from pipeline.library.ignore import rules_from_config
from pipeline.library.pathmap import PathMapping, translate_path
from pipeline.library.profile import Profile, SourceSpec
from pipeline.library.season import is_season_id, plan_units, season_item_id
from pipeline.library.source import LibraryItem
from pipeline.library.union import Claims, UnionLibrarySource, clash_key, reconstruct_claims, union_items, union_of
from pipeline.review import QueueEntry, write_queue_entry


def _item(item_id, path, **fields):
    return LibraryItem(id=item_id, source_path=path, display_name=fields.pop('display_name', item_id), **fields)


def _ids(result):
    return [t.id for t in result.titles]


# --- hard clashes: the same file ---------------------------------------------------------------------------------

def test_a_jriver_item_and_a_filesystem_item_for_the_same_mapped_path_are_one_title():
    mapped = translate_path('W:\\Films\\Heat (1995).mkv', [PathMapping('W:\\Films', '/media/films')])
    jriver = _item('jriver-3fa9c2-1234', mapped, title='Heat', year='1995')
    filesystem = _item('fs-aaaa', '/media/films/Heat (1995).mkv')

    result = union_of([('films', [jriver]), ('disk', [filesystem])])

    assert _ids(result) == ['jriver-3fa9c2-1234']
    assert result.titles[0].source == 'films' and result.titles[0].also_in == ('disk',)
    assert [(s.item.id, s.source, s.owner) for s in result.shadowed] == [('fs-aaaa', 'disk', 'jriver-3fa9c2-1234')]


@pytest.mark.parametrize('other', [
    '/MEDIA/Films/heat.MKV', '\\media\\films\\heat.mkv', '/media//films/heat.mkv', '/media/films/heat.mkv/',
])
def test_case_separators_and_stray_slashes_do_not_hide_a_clash(other):
    result = union_of([('a', [_item('one', '/media/films/heat.mkv')]), ('b', [_item('two', other)])])

    assert _ids(result) == ['one'] and len(result.shadowed) == 1


@pytest.mark.parametrize('clip', [
    '/discs/Heat/BDMV/STREAM/00001.m2ts', '/discs/Heat/BDMV/index.bdmv', '/discs/Heat/BDMV',
    '/discs/heat/bdmv/PLAYLIST/00000.mpls', '/discs/Heat/VIDEO_TS/VTS_01_1.VOB',
])
def test_a_clip_inside_a_disc_rip_clashes_with_the_disc_folder(clip):
    disc = _item('fs-disc', '/discs/Heat')

    result = union_of([('disk', [disc]), ('films', [_item('jriver-x-1', clip)])])

    assert _ids(result) == ['fs-disc'] and [s.item.id for s in result.shadowed] == ['jriver-x-1']


def test_different_files_are_not_a_clash_even_with_a_shared_prefix():
    result = union_of([('a', [_item('one', '/films/Heat.mkv')]), ('b', [_item('two', '/films/Heat 2.mkv')])])

    assert _ids(result) == ['one', 'two'] and result.shadowed == []


def test_clash_key_folds_only_at_a_disc_folder_not_at_a_folder_that_merely_contains_the_word():
    assert clash_key('/films/BDMV Collection/a.mkv') == '/films/bdmv collection/a.mkv'
    assert clash_key('\\\\nas\\Media\\Heat\\BDMV\\STREAM\\1.m2ts') == '/nas/media/heat'


# --- priority ----------------------------------------------------------------------------------------------------

def test_the_first_source_wins_a_clash_and_the_order_of_titles_follows_priority_then_listing():
    first = [_item('a1', '/x/1.mkv'), _item('a2', '/x/2.mkv')]
    second = [_item('b0', '/x/0.mkv'), _item('b2', '/x/2.mkv')]

    result = union_of([('one', first), ('two', second)])

    assert _ids(result) == ['a1', 'a2', 'b0'] and [s.item.id for s in result.shadowed] == ['b2']
    assert _ids(union_of([('two', second), ('one', first)])) == ['b0', 'b2', 'a1']


def test_reordering_the_sources_keeps_the_ids_that_already_have_outputs():
    ours = _item('jriver-x-1', '/media/films/heat.mkv')
    theirs = _item('fs-1', '/media/films/heat.mkv')
    claims = Claims(ids=frozenset({'jriver-x-1'}))

    forward = union_of([('films', [ours]), ('disk', [theirs])], claims=claims)
    reordered = union_of([('disk', [theirs]), ('films', [ours])], claims=claims)

    assert _ids(forward) == _ids(reordered) == ['jriver-x-1']
    assert reordered.titles[0].source == 'films'  # the same item, not just the same id: its fingerprint stays valid
    assert reordered.titles[0].item is ours


def test_without_a_claim_reordering_does_change_the_owner():
    ours, theirs = _item('jriver-x-1', '/m/heat.mkv'), _item('fs-1', '/m/heat.mkv')

    assert _ids(union_of([('films', [ours]), ('disk', [theirs])])) == ['jriver-x-1']
    assert _ids(union_of([('disk', [theirs]), ('films', [ours])])) == ['fs-1']


def test_when_the_owners_item_disappears_the_other_source_takes_over_under_its_own_id():
    theirs = _item('fs-1', '/media/films/heat.mkv')
    claims = Claims(ids=frozenset({'jriver-x-1'}))  # the JRiver copy had outputs, and has since left the library

    result = union_of([('films', []), ('disk', [theirs])], claims=claims)

    assert _ids(result) == ['fs-1'] and result.titles[0].source == 'disk'


def test_two_claimed_copies_leave_one_owner_deterministically():
    a, b = _item('a', '/m/heat.mkv'), _item('b', '/m/heat.mkv')

    result = union_of([('one', [a]), ('two', [b])], claims=Claims(ids=frozenset({'a', 'b'})))

    assert _ids(result) == ['a'] and [s.item.id for s in result.shadowed] == ['b']


def test_a_source_listing_the_same_file_twice_is_one_title():
    result = union_of([('films', [_item('k1', '/m/heat.mkv'), _item('k2', '/m/heat.mkv')])])

    assert _ids(result) == ['k1'] and len(result.shadowed) == 1


# --- soft clashes: the same title in different files ----------------------------------------------------------------

def test_the_same_tmdb_id_in_different_files_is_flagged_and_both_stay_titles():
    a = _item('a', '/x/heat.mkv', external_ids={'tmdb': '949'})
    b = _item('b', '/y/heat-extended.mkv', external_ids={'tmdb': '949'})
    other = _item('c', '/x/alien.mkv', external_ids={'tmdb': '348'})

    result = union_of([('one', [a, other]), ('two', [b])])

    assert _ids(result) == ['a', 'c', 'b'] and result.shadowed == []
    by_id = {t.id: t for t in result.titles}
    assert by_id['a'].duplicates == ('b',) and by_id['b'].duplicates == ('a',) and by_id['c'].duplicates == ()
    assert result.duplicates == [('a', 'b')]


def test_imdb_and_then_title_and_year_identify_a_title_when_there_is_no_tmdb_id():
    imdb = union_of([('s', [_item('a', '/1.mkv', external_ids={'imdb': 'tt1'}),
                            _item('b', '/2.mkv', external_ids={'imdb': 'tt1'})])])
    named = union_of([('s', [_item('a', '/1.mkv', title='Heat', year='1995'),
                             _item('b', '/2.mkv', title=' heat ', year='1995')])])
    other_year = union_of([('s', [_item('a', '/1.mkv', title='Heat', year='1995'),
                                  _item('b', '/2.mkv', title='Heat', year='2013')])])

    assert imdb.duplicates == [('a', 'b')] and named.duplicates == [('a', 'b')] and other_year.duplicates == []


def test_a_title_with_nothing_to_go_on_is_never_called_a_duplicate():
    result = union_of([('s', [_item('a', '/1.mkv'), _item('b', '/2.mkv')])])

    assert result.duplicates == []


def test_different_episodes_of_one_series_are_not_duplicates_but_two_copies_of_one_episode_are():
    def episode(item_id, path, number, tmdb='1399'):
        return _item(item_id, path, kind='tv', title='Show', season='1', episodes=(number,),
                     external_ids={'tmdb': tmdb})

    result = union_of([('a', [episode('e1', '/1.mkv', 1), episode('e2', '/2.mkv', 2)]),
                       ('b', [episode('e1-copy', '/3.mkv', 1)])])

    assert result.duplicates == [('e1', 'e1-copy')]
    assert union_of([('a', [episode('e1', '/1.mkv', 1), episode('e2', '/2.mkv', 2)])]).duplicates == []


def test_a_movie_and_a_series_with_the_same_number_are_not_duplicates():
    result = union_of([('s', [_item('m', '/1.mkv', kind='movie', external_ids={'tmdb': '1'}),
                              _item('t', '/2.mkv', kind='tv', external_ids={'tmdb': '1'})])])

    assert result.duplicates == []


# --- ignore ------------------------------------------------------------------------------------------------------

def test_an_ignored_title_stays_in_the_list_labelled_with_the_rule():
    rules = rules_from_config([{'path': '/films/Kids/**', 'reason': 'kids'}, {'kind': 'tv'}])
    items = [_item('a', '/films/Kids/Shrek.mkv'), _item('b', '/films/Heat.mkv'), _item('c', '/tv/x.mkv', kind='tv')]

    result = union_of([('films', items)], ignore=rules)

    assert _ids(result) == ['a', 'b', 'c']
    assert [t.ignored for t in result.titles] == ['ignored by rule: path /films/Kids/** (kids)', '',
                                                   'ignored by rule: kind tv']
    assert [t.id for t in result.active()] == ['b']


def test_deleting_a_rule_brings_its_titles_back_because_nothing_is_stored():
    items = [_item('a', '/films/Kids/Shrek.mkv')]

    assert union_of([('f', items)], ignore=rules_from_config([{'path': '/films/Kids/**'}])).active() == []
    assert [t.id for t in union_of([('f', items)], ignore=[]).active()] == ['a']


def test_a_rule_can_name_a_source():
    rules = rules_from_config([{'source': 'disk'}])
    result = union_of([('films', [_item('a', '/1.mkv')]), ('disk', [_item('b', '/2.mkv')])], ignore=rules)

    assert [t.id for t in result.active()] == ['a']


def test_a_single_title_can_be_ignored_by_id_with_a_reason_and_the_rules_do_not_need_to_match():
    result = union_of([('f', [_item('a', '/1.mkv'), _item('b', '/2.mkv')])],
                      ignored_titles={'a': 'broken rip', 'b': ''})

    assert [t.ignored for t in result.titles] == ['ignored by you: broken rip', 'ignored by you']


def test_an_ignored_title_is_not_reported_as_a_duplicate():
    items = [_item('a', '/1.mkv', external_ids={'tmdb': '1'}), _item('b', '/2.mkv', external_ids={'tmdb': '1'})]

    result = union_of([('f', items)], ignored_titles={'b': ''})

    assert result.duplicates == [] and result.titles[0].duplicates == ()


# --- claims read back from disk ----------------------------------------------------------------------------------

def _entry(queue_dir, entry_id, **meta):
    write_queue_entry(queue_dir, QueueEntry(id=entry_id, fs=1000, meta=meta, curve={}))


def test_claims_are_every_queue_entry_and_every_work_directory(tmp_path):
    queue_dir, work_dir = str(tmp_path / 'queue'), tmp_path / 'work'
    _entry(queue_dir, 'jriver-x-1', title='Heat')
    (work_dir / 'fs-2').mkdir(parents=True)
    (work_dir / '.hidden').mkdir()
    (work_dir / 'stray-file.txt').write_text('not a directory')

    claims = reconstruct_claims(str(work_dir), queue_dir)

    assert claims.ids == {'jriver-x-1', 'fs-2'}


def test_missing_directories_are_simply_no_claims(tmp_path):
    assert reconstruct_claims(str(tmp_path / 'nope'), str(tmp_path / 'nada')) == Claims()
    assert reconstruct_claims(None, None) == Claims()


def test_reconstructed_claims_decide_a_clash_with_no_extra_state(tmp_path):
    queue_dir = str(tmp_path / 'queue')
    _entry(queue_dir, 'jriver-x-1', title='Heat')
    ours, theirs = _item('jriver-x-1', '/m/heat.mkv'), _item('fs-1', '/m/heat.mkv')

    claims = reconstruct_claims(str(tmp_path / 'work'), queue_dir)

    assert _ids(union_of([('disk', [theirs]), ('films', [ours])], claims=claims)) == ['jriver-x-1']


# --- season ids: the second id shape -------------------------------------------------------------------------

def test_the_season_id_shape_is_recognised_and_source_ids_are_not():
    assert is_season_id(season_item_id('Some Show', '1')) and is_season_id('some-show-s01-60ad24')
    assert not any(is_season_id(i) for i in ('jriver-3fa9c2-1234', 'fs-0123456789abcdef', 'heat', 'a-s1-60ad24'))


def test_a_season_keeps_its_id_when_the_series_title_is_corrected_in_the_library(tmp_path):
    queue_dir = str(tmp_path / 'queue')
    old_id = season_item_id('Some Show', '1')
    _entry(queue_dir, old_id, title='Some Show', season='1', the_movie_db='1399')
    claims = reconstruct_claims(None, queue_dir)
    corrected = [_item('e1', '/1.mkv', kind='tv', title='Some Show (2019)', season='1', episodes=(1,),
                       external_ids={'tmdb': '1399'})]

    kept = plan_units(corrected, 'season', claims.season_id)
    fresh = plan_units(corrected, 'season')

    assert kept[0].item.id == old_id and fresh[0].item.id != old_id


def test_a_season_is_found_by_title_when_there_is_no_tmdb_id():
    claims = Claims(seasons={('title:some show', '1'): 'some-show-s01-aaaaaa'})
    episode = _item('e1', '/1.mkv', kind='tv', title=' Some Show', season='01', episodes=(1,))

    assert claims.season_id(episode) == 'some-show-s01-aaaaaa'
    assert claims.season_id(_item('e', '/1.mkv', kind='tv', title='Other', season='1', episodes=(1,))) is None
    assert claims.season_id(_item('m', '/1.mkv', title='Some Show')) is None  # not a season


def test_an_episodes_own_queue_entry_does_not_claim_the_season(tmp_path):
    queue_dir = str(tmp_path / 'queue')
    _entry(queue_dir, 'jriver-x-9', title='Some Show', season='1', episodes=[3], the_movie_db='1399')

    assert reconstruct_claims(None, queue_dir).seasons == {}


def test_a_new_season_gets_the_default_id():
    episode = _item('e1', '/1.mkv', kind='tv', title='New Show', season='1', episodes=(1,))

    assert plan_units([episode], 'season', Claims().season_id)[0].item.id == season_item_id('New Show', '1')


# --- the profile-level entry points ----------------------------------------------------------------------------

class _FakeSource:
    def __init__(self, items):
        self.items, self.calls = items, 0

    def list_items(self, **query):
        self.calls += 1
        return self.items


def _profile(tmp_path, **fields):
    return Profile(sources=(SourceSpec('films', 'jriver'), SourceSpec('disk', 'filesystem')),
                   work_dir=str(tmp_path / 'work'), queue_dir=str(tmp_path / 'queue'), **fields)


def test_union_items_lists_every_source_in_the_profile_and_merges_them(tmp_path):
    films = _FakeSource([_item('j1', '/m/heat.mkv'), _item('j2', '/m/alien.mkv')])
    disk = _FakeSource([_item('f1', '/m/heat.mkv'), _item('f2', '/m/dune.mkv')])
    profile = _profile(tmp_path, ignore=tuple(rules_from_config([{'path': '/m/alien.mkv'}])))

    result = union_items(profile, {'films': films, 'disk': disk})

    assert _ids(result) == ['j1', 'j2', 'f2'] and [t.id for t in result.active()] == ['j1', 'f2']
    assert (films.calls, disk.calls) == (1, 1)


def test_union_items_reads_claims_from_the_profiles_directories(tmp_path):
    (tmp_path / 'work' / 'f1').mkdir(parents=True)
    films = _FakeSource([_item('j1', '/m/heat.mkv')])
    disk = _FakeSource([_item('f1', '/m/heat.mkv')])

    assert _ids(union_items(_profile(tmp_path), {'films': films, 'disk': disk})) == ['f1']


def test_a_source_that_cannot_be_listed_fails_the_whole_union_rather_than_looking_empty(tmp_path):
    class Down:
        def list_items(self, **query):
            raise ConnectionError('media server is down')

    with pytest.raises(ConnectionError):
        union_items(_profile(tmp_path), {'films': Down(), 'disk': _FakeSource([])})


def test_the_union_source_gives_run_library_the_titles_that_are_not_ignored(tmp_path):
    films = _FakeSource([_item('j1', '/m/heat.mkv'), _item('j2', '/kids/shrek.mkv')])
    disk = _FakeSource([_item('f1', '/m/heat.mkv')])
    profile = _profile(tmp_path, ignore=tuple(rules_from_config([{'path': '/kids'}])))

    items = UnionLibrarySource(profile, {'films': films, 'disk': disk}).list_items()

    assert [i.id for i in items] == ['j1']
    with pytest.raises(TypeError, match='accepts no query'):
        UnionLibrarySource(profile, {}).list_items(content_type='movie')
