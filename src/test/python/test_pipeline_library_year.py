'''
pipeline/library/year.py: the one year language of ignore rules, `run --year` and the pipeline service
(design/pipeline-service.md §3).
'''
import re

import pytest

from pipeline.library.year import YEAR_PATTERN, YearRange, is_year_expression, numeric_year, year_matches


@pytest.mark.parametrize('expression, lo, hi', [
    ('2026', 2026, 2026), ('=2026', 2026, 2026), ('==2026', 2026, 2026), (' 2026 ', 2026, 2026),
    ('<1960', None, 1959), ('<=1960', None, 1960), ('>1999', 2000, None), ('>=1999', 1999, None),
    ('1990-1999', 1990, 1999), ('1990 - 1999', 1990, 1999), ('>= 2020', 2020, None),
])
def test_each_form_is_the_bounds_it_says(expression, lo, hi):
    assert YearRange.parse(expression) == YearRange(lo, hi)


@pytest.mark.parametrize('expression', ['', '26', '20266', 'nineteen', '<>2000', '1990-', '-1999', '2000-2010-2020',
                                        '≥2000', '２０２６', '1990..1999', '\u00a02026', '2026\n'])
def test_anything_else_is_refused(expression):
    with pytest.raises(ValueError, match='is not a year'):
        YearRange.parse(expression)
    assert not is_year_expression(expression)


def test_a_reversed_range_is_in_the_grammar_but_a_selection_refuses_it():
    ''' An ignore rule with one has always loaded (and matched nothing); a selection that can match nothing is a typo. '''
    assert is_year_expression('1999-1990') and not year_matches('1999-1990', '1995')
    with pytest.raises(ValueError, match='reversed'):
        YearRange.parse('1999-1990', allow_empty=False)


@pytest.mark.parametrize('year, expected', [('2026', True), (' 2026 ', True), ('2025', False), ('', False), (None, False),
                                            ('20xx', False), ('²⁰²⁶', False), ('TBA', False)])
def test_only_a_numeric_year_can_match(year, expected):
    assert year_matches('>=2026', year) is expected


def test_numeric_year_reads_ascii_digits_only():
    assert numeric_year(' 1984 ') == 1984 and numeric_year('١٩٨٤') is None and numeric_year('') is None


@pytest.mark.parametrize('expression', ['2026', '=2026', '==2026', ' <1960 ', '<=1960', '>1999', '>= 1999', '1990-1999',
                                        '1990 - 1999', '1999-1990', '', '26', '20266', '<>2000', '1990-', 'x', '≥2000',
                                        '２０２６', '\u00a02026', '\t2026', '2026\n'])
def test_the_published_pattern_and_the_parser_agree(expression):
    ''' The service publishes YEAR_PATTERN in its OpenAPI schema; a client that passes it must never be refused by the grammar. '''
    # fullmatch: a JSON schema's `$` is the end of the string; Python's re.match would also accept one final newline
    assert (re.fullmatch(YEAR_PATTERN, expression) is not None) is is_year_expression(expression)
