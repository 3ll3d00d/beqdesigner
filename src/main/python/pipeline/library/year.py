'''
The year expressions that ignore rules, `run --year`/`accept --year` and the pipeline service's filter share
(design/pipeline-service.md §3), so there is one language to learn:

    2026          that year (also `=2026`, `==2026`)
    <1960 <=1960  before, or up to and including
    >1999 >=1999  after, or from and including
    1990-1999     from and to, inclusive

A title with no numeric year never matches an expression.
'''
import re
from dataclasses import dataclass
from typing import Optional

# Digits and spaces are spelled out, not \d and \s, which mean more (any Unicode digit or space) in Python than a title's year
# is read as, and differ between regex engines.
_YEAR = re.compile(r'^[ \t]*(?:(?P<op><=|>=|<|>|==|=)?[ \t]*(?P<a>[0-9]{4})|(?P<lo>[0-9]{4})[ \t]*-[ \t]*(?P<hi>[0-9]{4}))[ \t]*$')

# The same grammar as a JSON schema pattern (no named groups), which the pipeline service publishes; a test holds the two to
# the same answers.
YEAR_PATTERN = r'^[ \t]*(?:(?:<=|>=|<|>|==|=)?[ \t]*[0-9]{4}|[0-9]{4}[ \t]*-[ \t]*[0-9]{4})[ \t]*$'


def is_year_expression(expression: str) -> bool:
    ''' True if `expression` is in the grammar (a reversed range such as 1999-1990 is, and matches nothing). '''
    return _YEAR.fullmatch(str(expression)) is not None


def numeric_year(year: Optional[str]) -> Optional[int]:
    ''' The year as a number, or None for a missing or non-numeric one. '''
    if not year or not re.fullmatch(r'[0-9]+', str(year).strip()):  # ASCII digits: '²'.isdigit() is True
        return None
    return int(str(year).strip())


@dataclass(frozen=True)
class YearRange:
    ''' An expression as inclusive bounds; None is open. '''
    lo: Optional[int] = None
    hi: Optional[int] = None

    @classmethod
    def parse(cls, expression: str, *, allow_empty: bool = True) -> 'YearRange':
        '''
        :param allow_empty: False refuses a range that can match nothing (1999-1990), which is a typo in a selection.
        :raises ValueError: for anything not in the grammar.
        '''
        m = _YEAR.fullmatch(str(expression))
        if m is None:
            raise ValueError(f'year {expression!r} is not a year (2026), a comparison (<1960, >=1999) or a range (1990-1999)')
        if m.group('lo'):
            lo, hi = int(m.group('lo')), int(m.group('hi'))
            if lo > hi and not allow_empty:
                raise ValueError(f'year range {expression!r} is reversed: it would match nothing')
            return cls(lo, hi)
        limit, op = int(m.group('a')), m.group('op') or '='
        return {'<': cls(None, limit - 1), '<=': cls(None, limit), '>': cls(limit + 1, None), '>=': cls(limit, None),
                '=': cls(limit, limit), '==': cls(limit, limit)}[op]

    def contains(self, year: Optional[str]) -> bool:
        value = numeric_year(year)
        if value is None:
            return False
        return (self.lo is None or value >= self.lo) and (self.hi is None or value <= self.hi)


def year_matches(expression: str, year: Optional[str]) -> bool:
    ''' True if `year` is within `expression`; False for a missing or non-numeric year. '''
    return YearRange.parse(expression).contains(year)
