'''model.dvd: recognising a DVD-Video rip, reading its titles and their durations, and choosing one.'''
import pytest

from dvd_fixtures import dvd_time, write_disc
from model.dvd import dvd_root, is_dvd_root, list_titles, pseudo_file_root, resolve_main_title, resolve_title

# A film disc: title 1 is the feature (VTS 1, 1h32m10s), title 2 a short bonus (VTS 1), title 3 a 10 s stub
# (VTS 2), title 4 an unreadable one (points at a title set with no IFO).
TITLES = [(1, 1, 24), (1, 2, 3), (2, 1, 1), (9, 1, 1)]
TITLE_SETS = {
    1: ([[(1, 1)], [(2, 1)]], [dvd_time(1, 32, 10), dvd_time(0, 4, 30, 12)]),
    2: ([[(1, 1)]], [dvd_time(0, 0, 10)]),
}


@pytest.fixture
def disc(tmp_path):
    return write_disc(tmp_path / 'Film Disc', TITLES, TITLE_SETS)


def test_a_folder_holding_video_ts_is_a_disc_root_and_so_is_video_ts_itself(disc):
    assert is_dvd_root(disc)
    assert dvd_root(disc) == disc
    assert dvd_root(disc + '/VIDEO_TS') == disc
    assert dvd_root(disc + '/VIDEO_TS/') == disc


def test_the_case_of_the_folder_and_file_names_does_not_matter(tmp_path):
    root = write_disc(tmp_path / 'lower', TITLES, TITLE_SETS, video_ts='video_ts')
    assert is_dvd_root(root)
    assert dvd_root(root + '/video_ts') == root


def test_other_folders_are_not_discs(tmp_path):
    (tmp_path / 'empty').mkdir()
    (tmp_path / 'has_video_ts' / 'VIDEO_TS').mkdir(parents=True)  # but no VIDEO_TS.IFO
    (tmp_path / 'file.mkv').write_bytes(b'x')

    assert not is_dvd_root(str(tmp_path / 'empty'))
    assert not is_dvd_root(str(tmp_path / 'has_video_ts'))
    assert not is_dvd_root(str(tmp_path / 'file.mkv'))
    assert not is_dvd_root(str(tmp_path / 'missing'))
    assert dvd_root(str(tmp_path / 'empty')) is None


def test_titles_come_back_longest_first_with_their_durations(disc):
    titles = list_titles(disc)

    assert [(t.number, t.vts, t.vts_title, t.chapters) for t in titles] == [(1, 1, 1, 24), (2, 1, 2, 3), (3, 2, 1, 1)]
    assert titles[0].duration_s == pytest.approx(1 * 3600 + 32 * 60 + 10)
    assert titles[1].duration_s == pytest.approx(4 * 60 + 30 + 12 / 25)  # 25 fps frames
    assert titles[2].duration_s == pytest.approx(10)


def test_thirty_frame_rate_discs_count_frames_at_29_97(tmp_path):
    root = write_disc(tmp_path / 'ntsc', [(1, 1, 1)], {1: ([[(1, 1)]], [dvd_time(0, 0, 1, 15, fps=30)])})

    assert list_titles(root)[0].duration_s == pytest.approx(1 + 15 / 29.97)


def test_a_title_in_a_missing_title_set_is_skipped_not_fatal(disc):
    assert 4 not in [t.number for t in list_titles(disc)]


def test_a_title_whose_program_chain_does_not_exist_is_skipped(tmp_path):
    root = write_disc(tmp_path / 'bad', [(1, 1, 1), (1, 2, 1)],
                      {1: ([[(1, 1)], [(7, 1)]], [dvd_time(0, 20)])})  # title 2 says PGC 7 of 1

    assert [t.number for t in list_titles(root)] == [1]


def test_a_vts_title_number_beyond_the_table_is_skipped(tmp_path):
    root = write_disc(tmp_path / 'bad', [(1, 1, 1), (1, 5, 1)], {1: ([[(1, 1)]], [dvd_time(0, 20)])})

    assert [t.number for t in list_titles(root)] == [1]


def test_equal_durations_keep_title_order(tmp_path):
    root = write_disc(tmp_path / 'tie', [(1, 1, 1), (1, 2, 1)],
                      {1: ([[(1, 1)], [(2, 1)]], [dvd_time(0, 25), dvd_time(0, 25)])})

    assert [t.number for t in list_titles(root)] == [1, 2]


def test_a_file_that_is_not_an_ifo_is_rejected(tmp_path):
    root = write_disc(tmp_path / 'junk', [(1, 1, 1)], {1: ([[(1, 1)]], [dvd_time(0, 20)])})
    (tmp_path / 'junk' / 'VIDEO_TS' / 'VIDEO_TS.IFO').write_bytes(b'not an ifo' + bytes(4096))

    with pytest.raises(ValueError, match='not a DVD-Video manager file'):
        list_titles(root)


def test_listing_titles_of_a_non_disc_says_so(tmp_path):
    with pytest.raises(ValueError, match='does not look like a DVD-Video rip'):
        list_titles(str(tmp_path))


def test_the_main_title_is_the_longest_and_reads_through_the_dvdvideo_demuxer(disc):
    resolved = resolve_main_title(disc)

    assert resolved.playlist.number == 1
    assert resolved.ffmpeg_input == disc  # the disc folder, not a file
    assert resolved.input_options == {'f': 'dvdvideo', 'title': 1}
    assert resolved.display_name == 'Film Disc_t01'
    assert resolved.playlist.duration_s == pytest.approx(5530)


def test_a_title_can_be_chosen_by_number_as_text_or_int(disc):
    assert resolve_main_title(disc, '2').input_options['title'] == 2
    assert resolve_main_title(disc, 2).playlist.name == '2'


def test_resolving_from_the_video_ts_folder_uses_the_disc_root(disc):
    assert resolve_main_title(disc + '/VIDEO_TS').ffmpeg_input == disc


def test_an_unknown_title_number_is_rejected(disc):
    with pytest.raises(ValueError, match='No title numbered 9'):
        resolve_main_title(disc, '9')
    with pytest.raises(ValueError, match='No title numbered 4'):  # listed on the disc but unreadable
        resolve_main_title(disc, '4')


def test_a_disc_with_no_readable_titles_is_rejected(tmp_path):
    root = write_disc(tmp_path / 'none', [(9, 1, 1)], {})

    with pytest.raises(ValueError, match='No playable titles'):
        resolve_main_title(root)


def test_resolving_a_non_disc_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='does not look like a DVD-Video rip'):
        resolve_main_title(str(tmp_path))


def test_resolve_title_is_usable_directly(disc):
    resolved = resolve_title(disc, list_titles(disc)[1])

    assert resolved.input_options == {'f': 'dvdvideo', 'title': 2}
    assert resolved.display_name == 'Film Disc_t02'


@pytest.mark.parametrize('filename, root', [
    ('W:\\Eastbound And Down\\S01\\D1\\VIDEO_TS\\VIDEO_TS.dvd;1', 'W:\\Eastbound And Down\\S01\\D1'),
    ('/mnt/films/Disc/video_ts/video_ts.DVD;2', '/mnt/films/Disc'),
    ('W:\\VIDEO_TS\\VIDEO_TS.dvd;1', 'W:\\'),  # a disc at a drive's top level keeps its separator
])
def test_a_jriver_dvd_pseudo_file_names_the_disc_folder(filename, root):
    assert pseudo_file_root(filename) == root


@pytest.mark.parametrize('filename', [
    'W:\\Films\\a.mkv', 'W:\\Disc\\VIDEO_TS\\VTS_01_1.VOB', '\\VIDEO_TS\\VIDEO_TS.dvd;1', 'W:\\Disc\\BDMV\\index.bluray;1',
])
def test_anything_else_is_not_a_dvd_pseudo_file(filename):
    assert pseudo_file_root(filename) is None
