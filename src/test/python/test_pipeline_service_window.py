'''
pipeline/service/window.py -- a job's active window (design/job-windows.md §2): daily ranges in the service's time zone,
read in wall-clock time.
'''
from datetime import datetime, timedelta, timezone, tzinfo

import pytest

from pipeline.service.window import ActiveWindow, parse_range, window_problem

UTC = timezone.utc


def at(text: str, tz: tzinfo = UTC) -> float:
    ''' The instant the wall clock in `tz` reads `text` (YYYY-MM-DD HH:MM). '''
    return datetime.strptime(text, '%Y-%m-%d %H:%M').replace(tzinfo=tz).timestamp()


def wall(t: float, tz: tzinfo = UTC) -> str:
    return datetime.fromtimestamp(t, tz).strftime('%Y-%m-%d %H:%M')


class Switches(tzinfo):
    ''' A zone whose offset changes from `before` to `after` hours at the UTC instant `when`: a daylight-saving change. '''

    def __init__(self, when: float, before: int, after: int):
        self.when, self.before, self.after = when, timedelta(hours=before), timedelta(hours=after)

    def fromutc(self, dt):
        utc = dt.replace(tzinfo=UTC).timestamp()
        return (dt + (self.before if utc < self.when else self.after)).replace(tzinfo=self)

    def utcoffset(self, dt):
        return self.before

    def dst(self, dt):
        return timedelta(0)


@pytest.mark.parametrize('text, minutes', [
    ('01:00-07:00', (60, 420)),
    ('22:00-06:00', (1320, 360)),        # crosses midnight
    (' 1:05 - 7:30 ', (65, 450)),        # written loosely
    ('00:00-24:00', (0, 1440)),          # the whole day
    ('23:00-24:00', (1380, 1440)),
])
def test_a_range_is_two_times_of_day(text, minutes):
    assert parse_range(text) == minutes


@pytest.mark.parametrize('text', ['01:00-01:00', '24:00-06:00', '01:60-02:00', '25:00-01:00', '0100-0700', '01:00',
                                  '', 'night', '24:30-01:00', '01:00-24:01'])
def test_anything_else_is_refused_and_says_what_a_range_is(text):
    with pytest.raises(ValueError, match='HH:MM|time of day|same time'):
        parse_range(text)
    assert window_problem([text])


def test_ranges_are_written_back_normalised():
    assert ActiveWindow.of([' 1:00 - 7:00', '22:00-06:00']).texts() == ['01:00-07:00', '22:00-06:00']


@pytest.mark.parametrize('ranges, when, is_open', [
    (['01:00-07:00'], '00:59', False),
    (['01:00-07:00'], '01:00', True),     # the start is inside
    (['01:00-07:00'], '06:59', True),
    (['01:00-07:00'], '07:00', False),    # the end is not
    (['22:00-06:00'], '23:30', True),
    (['22:00-06:00'], '03:00', True),
    (['22:00-06:00'], '06:00', False),
    (['22:00-06:00'], '21:59', False),
    (['01:00-03:00', '13:00-15:00'], '14:00', True),
    (['01:00-03:00', '13:00-15:00'], '09:00', False),
    (['00:00-24:00'], '23:59', True),
    ([], '12:00', True),                  # none: any time
])
def test_open_when_the_wall_clock_is_inside_any_range(ranges, when, is_open):
    assert ActiveWindow.of(ranges, UTC).is_open(at(f'2026-10-26 {when}')) is is_open


@pytest.mark.parametrize('ranges, now, opens, closes', [
    (['01:00-07:00'], '2026-10-26 12:00', '2026-10-27 01:00', None),
    (['01:00-07:00'], '2026-10-26 00:30', '2026-10-26 01:00', None),
    (['01:00-07:00'], '2026-10-26 03:00', '2026-10-26 03:00', '2026-10-26 07:00'),   # open now: opens is now
    (['22:00-06:00'], '2026-10-26 23:00', '2026-10-26 23:00', '2026-10-27 06:00'),
    (['01:00-03:00', '13:00-15:00'], '2026-10-26 04:00', '2026-10-26 13:00', None),
    (['01:00-03:00', '02:00-05:00'], '2026-10-26 01:30', '2026-10-26 01:30', '2026-10-26 05:00'),   # overlapping
    (['22:00-24:00', '00:00-02:00'], '2026-10-26 23:00', '2026-10-26 23:00', '2026-10-27 02:00'),   # meeting
])
def test_when_it_next_opens_and_closes(ranges, now, opens, closes):
    window = ActiveWindow.of(ranges, UTC)
    t = at(now)
    assert wall(window.next_open(t)) == opens
    assert (wall(window.next_close(t)) if window.next_close(t) else None) == closes


def test_any_time_and_the_whole_day_never_close():
    t = at('2026-10-26 12:00')
    for window in (ActiveWindow.of([], UTC), ActiveWindow.of(['00:00-24:00'], UTC)):
        assert window.next_open(t) == t and window.next_close(t) is None


def test_a_start_the_clocks_skip_opens_at_the_first_minute_after_them():
    # 01:00 UTC on 29 March: the wall clock goes from 00:59 GMT to 02:00 BST, so 01:30 never happens
    change = at('2026-03-29 01:00')
    zone = Switches(change, 0, 1)
    window = ActiveWindow.of(['01:30-03:00'], zone)
    before = at('2026-03-29 00:30')
    opens = window.next_open(before)
    assert opens == change and wall(opens, zone) == '2026-03-29 02:00'
    assert wall(window.next_close(opens), zone) == '2026-03-29 03:00'   # an hour open, not an hour and a half


def test_an_hour_the_clocks_repeat_is_open_both_times():
    # 01:00 UTC on 25 October: the wall clock goes from 01:59 BST back to 01:00 GMT, so 01:00-02:00 happens twice
    change = at('2026-10-25 01:00')
    zone = Switches(change, 1, 0)
    window = ActiveWindow.of(['01:00-02:00'], zone)
    first = at('2026-10-25 00:00')              # 01:00 BST
    assert window.is_open(first) and window.is_open(change)   # 01:00 BST, and 01:00 GMT an hour later
    assert window.next_close(first) == at('2026-10-25 02:00')   # 02:00 GMT: two hours after it opened
