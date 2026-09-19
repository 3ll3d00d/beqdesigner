'''TV seasons as a single track, or a filter per episode (plan §11.9).'''
import os

import numpy as np
import pytest
import soundfile as sf

from pipeline.library.season import SeasonGroup, plan_units, season_item_id, season_track_if_needed, with_extracted
from pipeline.library.source import LibraryItem


def _ep(series, season_no, episode_no, key=None, **overrides):
    values = dict(id=key or f'{series[:3].lower()}-{season_no}-{episode_no}',
                  source_path=f'/tv/{series}/{season_no}/{episode_no}.mkv',
                  display_name=f'{series} S{season_no}E{episode_no}', title=series, year='2015', kind='tv',
                  season=str(season_no), episodes=(episode_no,))
    values.update(overrides)
    return LibraryItem(**values)


def _film(key='film'):
    return LibraryItem(id=key, source_path='/films/a.mkv', display_name='A Film', title='A Film', kind='movie')


# --- planning -----------------------------------------------------------------------------------------------------

def test_episode_mode_leaves_every_item_alone():
    items = [_ep('Show', 1, 2), _film(), _ep('Show', 1, 1)]

    assert plan_units(items, 'episode') == items
    assert plan_units(items) == items  # it is the default


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="tv_mode must be one of episode, season"):
        plan_units([], 'series')


def test_season_mode_joins_one_series_and_season_in_episode_order():
    units = plan_units([_ep('Show', 1, 3), _ep('Show', 1, 1), _ep('Show', 1, 2)], 'season')

    assert len(units) == 1 and isinstance(units[0], SeasonGroup)
    group = units[0]
    assert [m.episodes for m in group.members] == [(1,), (2,), (3,)]
    assert group.item.episodes == (1, 2, 3)
    assert (group.item.title, group.item.season, group.item.kind) == ('Show', '1', 'tv')
    assert group.item.display_name == 'Show Season 1'
    assert group.item.id.startswith('show-s01-')


def test_different_seasons_and_series_are_separate_units_in_the_order_first_seen():
    film = _film()
    units = plan_units([_ep('Show', 1, 1), film, _ep('Other', 1, 1), _ep('Show', 2, 1), _ep('Show', 1, 2)], 'season')

    assert [u.item.display_name if isinstance(u, SeasonGroup) else u.id for u in units] == \
        ['Show Season 1', 'film', 'Other Season 1', 'Show Season 2']
    assert units[1] is film
    assert [m.episodes for m in units[0].members] == [(1,), (2,)]  # the later episode joined the first season's group


def test_series_names_group_regardless_of_case_and_padding_of_the_season():
    units = plan_units([_ep('Some Show', 1, 1), _ep('some show', 1, 2, season='01')], 'season')

    assert len(units) == 1 and len(units[0].members) == 2


@pytest.mark.parametrize('odd', [
    {'season': None},  # no season
    {'title': None},  # no series
    {'episodes': ()},  # no episode number
    {'episodes': (1, 2)},  # already covering several
    {'kind': 'movie'},
])
def test_items_that_are_not_a_numbered_episode_of_a_series_pass_through_untouched(odd):
    item = _ep('Show', 1, 1, **odd)

    assert plan_units([item], 'season') == [item]


def test_a_second_item_for_the_same_episode_is_ignored_and_the_first_wins():
    first, copy = _ep('Show', 1, 1, key='first'), _ep('Show', 1, 1, key='copy')

    (group,) = plan_units([first, copy, _ep('Show', 1, 2)], 'season')

    assert [m.id for m in group.members] == ['first', 'sho-1-2']
    assert group.item.episodes == (1, 2)


def test_a_season_with_a_single_episode_is_still_a_season_unit():
    (group,) = plan_units([_ep('Show', 1, 4)], 'season')

    assert isinstance(group, SeasonGroup) and group.item.episodes == (4,)


def test_the_season_item_takes_ids_year_and_art_from_the_members_that_have_them():
    members = [_ep('Show', 1, 1, year=None), _ep('Show', 1, 2, year='2016', external_ids={'tmdb': '66292'},
                                                 art_path='/art/a.jpg'),
               _ep('Show', 1, 3, external_ids={'tmdb': 'ignored', 'imdb': 'tt9'})]

    (group,) = plan_units(members, 'season')

    assert group.item.year == '2016'
    assert group.item.external_ids == {'tmdb': '66292', 'imdb': 'tt9'}  # first value per identifier wins
    assert group.item.art_path == '/art/a.jpg'


def test_the_season_id_is_stable_safe_and_independent_of_which_episodes_are_present():
    assert season_item_id('Some Show', '1') == season_item_id('some show', '01')
    assert season_item_id('Some Show', '1') != season_item_id('Some Show', '2')
    assert season_item_id('Some Show', '1') != season_item_id('Other Show', '1')
    assert season_item_id('Modern Family!', '3').startswith('modern-family-s03-')
    assert '/' not in season_item_id('A/B: C', '1') and ' ' not in season_item_id('A B', '1')
    (all_eps,) = plan_units([_ep('Show', 1, e) for e in (1, 2, 3)], 'season')
    (some_eps,) = plan_units([_ep('Show', 1, 2)], 'season')
    assert all_eps.item.id == some_eps.item.id


def test_with_extracted_narrows_the_episodes_and_sets_the_fingerprint():
    (group,) = plan_units([_ep('Show', 1, e) for e in (1, 2, 3)], 'season')

    item = with_extracted(group, [1, 3], 'abc')

    assert item.episodes == (1, 3) and item.fingerprint == 'abc' and item.id == group.item.id


# --- joining the tracks ---------------------------------------------------------------------------------------------

def _wav(path, value, seconds=1.0, fs=1000, channels=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.full((int(fs * seconds), channels), value, dtype='float64')
    sf.write(path, data, fs, subtype='PCM_24')
    return path


def test_the_episodes_are_joined_in_episode_order_into_one_wav(tmp_path):
    a = _wav(str(tmp_path / 'e1' / 'mono.wav'), 0.1, 1.0)
    b = _wav(str(tmp_path / 'e2' / 'mono.wav'), 0.2, 2.0)
    c = _wav(str(tmp_path / 'e3' / 'mono.wav'), 0.3, 0.5)

    track, fingerprint, up_to_date = season_track_if_needed([(3, c), (1, a), (2, b)], str(tmp_path / 'season'))

    assert track == str(tmp_path / 'season' / 'mono.wav') and up_to_date is False and fingerprint
    data, fs = sf.read(track)
    assert fs == 1000 and len(data) == 3500
    assert data[0] == pytest.approx(0.1, abs=1e-4) and data[1500] == pytest.approx(0.2, abs=1e-4)
    assert data[-1] == pytest.approx(0.3, abs=1e-4)
    assert sf.info(track).subtype == 'PCM_24'


def test_an_unchanged_set_of_episodes_is_not_joined_again(tmp_path):
    a = _wav(str(tmp_path / 'e1' / 'mono.wav'), 0.1)
    b = _wav(str(tmp_path / 'e2' / 'mono.wav'), 0.2)
    first = season_track_if_needed([(1, a), (2, b)], str(tmp_path / 'season'))
    mtime = os.path.getmtime(first[0])

    again = season_track_if_needed([(2, b), (1, a)], str(tmp_path / 'season'))

    assert again[2] is True and again[1] == first[1] and os.path.getmtime(first[0]) == mtime


def test_changing_dropping_or_adding_an_episode_rebuilds_the_track(tmp_path):
    a = _wav(str(tmp_path / 'e1' / 'mono.wav'), 0.1)
    b = _wav(str(tmp_path / 'e2' / 'mono.wav'), 0.2)
    season = str(tmp_path / 'season')
    base = season_track_if_needed([(1, a), (2, b)], season)

    dropped = season_track_if_needed([(1, a)], season)
    assert dropped[2] is False and dropped[1] != base[1] and len(sf.read(dropped[0])[0]) == 1000

    _wav(str(tmp_path / 'e1' / 'mono.wav'), 0.5, 1.5)  # re-extracted: a different file on disk
    changed = season_track_if_needed([(1, a)], season)
    assert changed[2] is False and changed[1] != dropped[1] and len(sf.read(changed[0])[0]) == 1500


def test_force_always_rebuilds(tmp_path):
    a = _wav(str(tmp_path / 'e1' / 'mono.wav'), 0.1)
    season = str(tmp_path / 'season')
    season_track_if_needed([(1, a)], season)

    assert season_track_if_needed([(1, a)], season, force=True)[2] is False


def test_episodes_that_disagree_on_rate_or_channels_are_rejected(tmp_path):
    a = _wav(str(tmp_path / 'e1' / 'mono.wav'), 0.1, fs=1000)
    other_rate = _wav(str(tmp_path / 'e2' / 'mono.wav'), 0.1, fs=2000)
    stereo = _wav(str(tmp_path / 'e3' / 'mono.wav'), 0.1, channels=2)

    with pytest.raises(ValueError, match="episode 2's audio"):
        season_track_if_needed([(1, a), (2, other_rate)], str(tmp_path / 's1'))
    with pytest.raises(ValueError, match="episode 2's audio"):
        season_track_if_needed([(1, a), (2, stereo)], str(tmp_path / 's2'))


def test_a_track_needs_an_episode(tmp_path):
    with pytest.raises(ValueError, match='at least one episode'):
        season_track_if_needed([], str(tmp_path / 'season'))
