'''
A job's active window (design/job-windows.md §2): the times of day it may run, as daily `"HH:MM-HH:MM"` ranges in the
service's time zone.

The window is read in local wall-clock time, a minute at a time: a range opens when the clock on the wall reads its start
and closes when it reads its end. So a range whose start a daylight-saving change skips opens at the first minute after
the change that is inside it, and an hour the change repeats is open, or not, both times, as the wall clock reads.
'''
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from typing import Iterable, List, Optional, Sequence, Tuple

_RANGE = re.compile(r'^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$')
DAY = 24 * 60
# how far ahead a change is looked for: a day and a bit covers every window (each range opens once a day), with room for
# a daylight-saving change between now and then
_LOOK_AHEAD_MINUTES = 2 * DAY + 120


def _minute(hours: str, minutes: str, text: str, *, end: bool) -> int:
    h, m = int(hours), int(minutes)
    if m > 59 or h > 24 or (h == 24 and (m or not end)):
        raise ValueError(f'{text!r}: not a time of day (00:00 to 23:59, and 24:00 as an end)')
    return h * 60 + m


def parse_range(text: str) -> Tuple[int, int]:
    '''
    :return: (start, end) in minutes since midnight; end may be DAY (24:00), and less than start for a range that crosses
        midnight.
    :raises ValueError: for anything but `HH:MM-HH:MM` with two different times.
    '''
    if not isinstance(text, str):
        raise ValueError(f'{text!r}: a range is a string, HH:MM-HH:MM')
    match = _RANGE.match(text)
    if not match:
        raise ValueError(f'{text!r}: a range is HH:MM-HH:MM, e.g. 01:00-07:00')
    start = _minute(match[1], match[2], text, end=False)
    end = _minute(match[3], match[4], text, end=True)
    if start == end:
        raise ValueError(f'{text!r}: the start and end are the same time; 00:00-24:00 is the whole day')
    return start, end


def _text(start: int, end: int) -> str:
    return f'{start // 60:02d}:{start % 60:02d}-{end // 60:02d}:{end % 60:02d}'


@dataclass(frozen=True)
class ActiveWindow:
    '''
    The ranges a job may run in; none is any time. Build it with `of()`, which checks and normalises them.
    :param tz: the zone the ranges are in; None is the service's (the process's local time).
    '''
    ranges: Tuple[Tuple[int, int], ...] = ()
    tz: Optional[tzinfo] = None

    @classmethod
    def of(cls, ranges: Optional[Iterable[str]], tz: Optional[tzinfo] = None) -> 'ActiveWindow':
        ''' :raises ValueError: naming the first range that is not one. '''
        return cls(tuple(parse_range(r) for r in (ranges or ())), tz)

    @property
    def any_time(self) -> bool:
        return not self.ranges

    def texts(self) -> List[str]:
        ''' The ranges as they are written (normalised: `1:00 - 7:00` is `01:00-07:00`). '''
        return [_text(start, end) for start, end in self.ranges]

    def __local_minute(self, t: float) -> int:
        local = datetime.fromtimestamp(t, self.tz) if self.tz is not None else datetime.fromtimestamp(t)
        return local.hour * 60 + local.minute

    def is_open(self, t: float) -> bool:
        if self.any_time:
            return True
        minute = self.__local_minute(t)
        return any(start <= minute < end if start < end else (minute >= start or minute < end)
                   for start, end in self.ranges)

    def __next_change(self, t: float, to_open: bool) -> Optional[float]:
        ''' The first whole minute after `t` at which the window is open (to_open) or closed; None if there is none. '''
        whole = (int(t) // 60 + 1) * 60
        for step in range(_LOOK_AHEAD_MINUTES):
            at = whole + step * 60
            if self.is_open(at) == to_open:
                return float(at)
        return None

    def next_open(self, t: float) -> Optional[float]:
        ''' `t` if the window is open then, else when it next opens. '''
        return t if self.is_open(t) else self.__next_change(t, True)

    def next_close(self, t: float) -> Optional[float]:
        ''' When the window next closes after `t`, if it is open then; None if it never does (any time, the whole day). '''
        if self.any_time or not self.is_open(t):
            return None
        return self.__next_change(t, False)


def window_problem(ranges: Optional[Sequence[str]]) -> str:
    ''' Why these ranges are not a window, or ''. '''
    try:
        ActiveWindow.of(ranges)
    except ValueError as error:
        return str(error)
    return ''
