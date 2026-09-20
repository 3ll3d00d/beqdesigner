'''Ignore rules (design/library-sync/workflow-rework §12.4): each field, combinations, and bad rules.'''
import pytest

from pipeline.library.ignore import IgnoreRule, evaluate, explain, rule_from_config, rules_from_config
from pipeline.library.source import LibraryItem


def _item(path='/films/Heat.mkv', **fields):
    return LibraryItem(id='a', source_path=path, display_name=fields.pop('display_name', 'Heat'), **fields)


def _matches(rule, item=None, source='films'):
    return rule_from_config(rule).matches(item or _item(), source)


@pytest.mark.parametrize('pattern, path, expected', [
    ('/films/Kids', '/films/Kids/Shrek.mkv', True),       # a prefix: the folder and everything under it
    ('/films/Kids', '/films/Kids', True),
    ('/films/Kid', '/films/Kids/Shrek.mkv', False),        # whole components only
    ('/films/Kids/', '/films/kids/Shrek.mkv', True),       # case and a trailing separator do not matter
    ('/films/Kids/**', '/films/Kids/a/b/Shrek.mkv', True),
    ('/films/Kids/**', '/films/Adult/Shrek.mkv', False),
    ('/films/*/extras/**', '/films/Heat/extras/x.mkv', True),
    ('/films/*/extras/**', '/films/a/b/extras/x.mkv', False),  # one * is one folder
    ('/films/**/extras/**', '/films/a/b/extras/x.mkv', True),
    ('**/*.iso', '/films/Heat.iso', True),
    ('**/*.iso', '/films/Heat.mkv', False),
    ('/films/H?at.mkv', '/films/Heat.mkv', True),
    ('W:\\Films\\Kids', 'w:/films/kids/x.mkv', True),      # either separator, either side
    ('/films/Kids', 'W:\\Films\\Kids\\x.mkv', False),
])
def test_a_path_rule_is_a_folder_prefix_or_a_glob(pattern, path, expected):
    assert _matches({'path': pattern}, _item(path)) is expected


@pytest.mark.parametrize('expression, year, expected', [
    ('<1960', '1959', True), ('<1960', '1960', False), ('<=1960', '1960', True), ('>1999', '2000', True),
    ('>=1999', '1998', False), ('1995', '1995', True), ('1995', '1996', False), ('= 1995', '1995', True),
    ('1990-1999', '1990', True), ('1990-1999', '1999', True), ('1990-1999', '2000', False),
    ('<1960', None, False), ('<1960', '', False), ('<1960', 'unknown', False), ('>1900', 'unknown', False),
])
def test_a_year_rule_compares_numbers_and_never_matches_an_unknown_year(expression, year, expected):
    assert _matches({'year': expression}, _item(year=year)) is expected


def test_a_year_may_be_given_as_a_number_in_yaml():
    assert _matches({'year': 1995}, _item(year='1995')) is True


def test_a_title_rule_is_a_case_insensitive_search_not_an_anchored_match():
    assert _matches({'title': 'heat'}, _item(title='Heat (Extended)')) is True
    assert _matches({'title': '^heat$'}, _item(title='Heat (Extended)')) is False
    assert _matches({'title': 'trailer'}, _item(display_name='Movie Trailer')) is True  # falls back to the name


def test_kind_source_and_external_ids_rules():
    assert _matches({'kind': 'tv'}, _item(kind='tv')) and not _matches({'kind': 'tv'}, _item(kind='movie'))
    assert _matches({'source': 'films'}) and not _matches({'source': 'other'})
    item = _item(external_ids={'imdb': 'tt0113277', 'tmdb': '949'})
    assert _matches({'external_ids': {'imdb': 'tt0113277'}}, item)
    assert _matches({'external_ids': {'imdb': 'tt0113277', 'tmdb': 949}}, item)  # numbers compare as text
    assert not _matches({'external_ids': {'imdb': 'tt0113277', 'tmdb': '1'}}, item)
    assert not _matches({'external_ids': {'imdb': 'tt0113277'}}, _item())


def test_every_field_a_rule_lists_must_match():
    rule = {'path': '/films/Kids/**', 'kind': 'movie', 'year': '<2000'}

    assert _matches(rule, _item('/films/Kids/a.mkv', kind='movie', year='1999'))
    assert not _matches(rule, _item('/films/Kids/a.mkv', kind='movie', year='2001'))
    assert not _matches(rule, _item('/films/Kids/a.mkv', kind='tv', year='1999'))
    assert not _matches(rule, _item('/films/Adult/a.mkv', kind='movie', year='1999'))


def test_evaluate_returns_the_first_matching_rule_in_order():
    rules = rules_from_config([{'kind': 'tv'}, {'path': '/films/**', 'reason': 'films'}, {'year': '<2000'}])

    assert evaluate(rules, _item(kind='tv')) is rules[0]
    assert evaluate(rules, _item('/films/a.mkv', year='1990')) is rules[1]
    assert evaluate(rules, _item('/other/a.mkv', year='1990')) is rules[2]
    assert evaluate(rules, _item('/other/a.mkv', year='2010')) is None
    assert evaluate([], _item()) is None


def test_explain_names_the_rule_that_matched():
    rule = rule_from_config({'path': '/films/Kids/**', 'year': '<2000', 'reason': 'not for the catalogue'})

    assert explain(rule) == 'ignored by rule: path /films/Kids/** and year <2000 (not for the catalogue)'
    assert explain(rule_from_config({'external_ids': {'imdb': 'tt1'}})) == 'ignored by rule: ids imdb=tt1'
    assert explain(None) == ''


@pytest.mark.parametrize('bad, message', [
    ({}, 'at least one'),
    ({'reason': 'because'}, 'at least one'),
    ({'kind': 'music'}, 'kind must be one of'),
    ({'year': 'old'}, 'not a year'),
    ({'year': '19'}, 'not a year'),
    ({'title': '('}, 'regular expression'),
    ({'colour': 'red'}, 'unknown ignore rule key'),
    ({'path': '/a', 'external_ids': ['tt1']}, 'external_ids must be a mapping'),
])
def test_a_malformed_rule_is_refused_so_a_typo_cannot_silently_ignore_nothing(bad, message):
    with pytest.raises(ValueError, match=message):
        rule_from_config(bad)


def test_a_rule_that_is_not_a_mapping_is_refused():
    with pytest.raises(ValueError, match='must be a mapping'):
        rules_from_config(['/films/Kids'])


def test_a_rule_round_trips_through_its_config_form():
    config = {'source': 'films', 'path': '/films/Kids/**', 'title': '^A', 'year': '<2000', 'kind': 'movie',
              'external_ids': {'imdb': 'tt1'}, 'reason': 'why'}

    assert rule_from_config(config).to_config() == config
    assert rule_from_config(rule_from_config(config).to_config()) == rule_from_config(config)
    assert isinstance(rule_from_config(config), IgnoreRule)


# --- review fixes: a glob naming a folder ignores what is under it; brackets are literal; bounded and safe --------------

@pytest.mark.parametrize('pattern, path, expected', [
    ('/films/Kids [HD]', '/films/Kids [HD]/x.mkv', True),      # brackets are literal, so this folder is found
    ('/films/Kids [HD]', '/films/Kids H/x.mkv', False),         # ... and are not a character class
    ('/films/Kids [HD]/**', '/films/Kids [HD]/a/x.mkv', True),
    ('/films/*/extras', '/films/A/extras/x.mkv', True),         # the documented glob, naming a folder
    ('/films/*/extras', '/films/A/extras', True),
    ('/films/*/extras', '/films/A/other/x.mkv', False),
    ('/films/Kids*', '/films/Kids/x.mkv', True),
    ('/films/Kids*', '/films/Kids Corner/x.mkv', True),
    ('/films/Kids*', '/films/Adult/x.mkv', False),
    ('/films/[abc]*', '/films/[abc]1/x.mkv', True),
    ('/films/[abc]*', '/films/a1/x.mkv', False),
    ('/films/Kids/**', '/films/Kids', True),                    # the folder itself, e.g. a disc rip's root
    ('**/extras', '/films/A/extras/x.mkv', True),
])
def test_a_path_pattern_matches_the_item_or_any_folder_above_it(pattern, path, expected):
    assert _matches({'path': pattern}, _item(path)) is expected


def test_an_item_with_no_path_is_matched_only_by_a_rule_that_matches_the_empty_path():
    assert _matches({'path': '/films/Kids'}, _item('')) is False


def test_a_year_of_superscript_digits_does_not_abort_the_scan():
    # str.isdigit() is True for '²', and int('²') raises: one odd year in a library must not stop discovery
    assert _matches({'year': '<1960'}, _item(year='²')) is False
    assert _matches({'year': '<1960'}, _item(year='١٩٥٥')) is False    # non-ASCII decimal digits are not a year either


def test_a_title_rule_reads_only_the_first_300_characters_of_a_title():
    rule = {'title': 'needle'}
    assert _matches(rule, _item(title='needle' + 'x' * 10000)) is True
    assert _matches(rule, _item(title='x' * 10000 + 'needle')) is False


def test_a_title_rule_with_a_backtracking_pattern_never_sees_the_part_of_the_title_that_would_blow_it_up():
    import time
    started = time.perf_counter()
    # uncapped, `(a+)+$` on 40 'a' and a '!' is ~2**40 steps (minutes); the cap keeps it off this title's tail
    assert _matches({'title': '(a+)+$'}, _item(title='x' * 300 + 'a' * 40 + '!')) is False
    assert time.perf_counter() - started < 5
