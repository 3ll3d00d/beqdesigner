'''
The Titles filters on adjustments and design runs (design/web-parity.md WP7 follow-up): what the index keeps of each
entry (status.EntryFacts: adjusted, unmet, runs), what a row says of it, and Selection's three filters.
'''
import pytest

from pipeline.library.index import LibraryIndex, index_path
from pipeline.library.selection import Selection
from pipeline.library.status import EntryFacts, unmet_adjustments
from pipeline.review import read_entry, update_entry
from review_entry_fixture import write_entry
from worklist_fixture import make_index, title_row

ASKED = [{'id': 'a1', 'kind': 'max_boost', 'params': {'db': 10}}, {'id': 'a2', 'kind': 'note', 'text': 'x'}]


def _entry(tmp_path, entry_id, **fields):
    write_entry(str(tmp_path), entry_id)
    if fields:
        update_entry(str(tmp_path), entry_id, **fields)
    return read_entry(str(tmp_path), entry_id)


def test_an_entry_says_whether_it_was_adjusted_how_many_asks_went_unmet_and_its_runs(tmp_path):
    plain = EntryFacts.of(_entry(tmp_path, 'plain'), 1, 2)
    assert (plain.adjusted, plain.unmet, plain.runs) == (False, 0, 1)
    asked = _entry(tmp_path, 'asked', adjusted_with=ASKED, variants=[{'variant_id': 'v1'}, {'variant_id': 'v2'}],
                   adjustment_outcomes=[{'id': 'a1', 'outcome': 'honoured'}, {'id': 'a2', 'outcome': 'not_understood'}])
    facts = EntryFacts.of(asked, 1, 2)
    assert (facts.adjusted, facts.unmet, facts.runs) == (True, 1, 3)
    assert unmet_adjustments(_entry(tmp_path, 'unanswered', adjusted_with=ASKED)) == 2   # a designer before 1.5
    assert EntryFacts.from_json(facts.to_json()) == facts


@pytest.fixture
def index(tmp_path):
    def summary(entry_id, **fields):
        return EntryFacts.of(_entry(tmp_path / 'q', entry_id, **fields), 1, 2).to_json()
    rows = [dict(title_row('plain', 'Plain', needs='review'), entry_summary=summary('plain')),
            dict(title_row('asked', 'Asked', needs='review'), entry_summary=summary(
                'asked', adjusted_with=ASKED, adjustment_outcomes=[{'id': 'a1', 'outcome': 'honoured'},
                                                                   {'id': 'a2', 'outcome': 'honoured'}])),
            dict(title_row('unmet', 'Unmet', needs='review'), entry_summary=summary(
                'unmet', adjusted_with=ASKED[:1], variants=[{'variant_id': 'v1'}])),
            title_row('bare', 'Bare', needs='extract')]
    make_index(tmp_path, rows)
    with LibraryIndex(index_path(str(tmp_path)), readonly=True) as opened:
        yield opened


@pytest.mark.parametrize('selection, ids', [
    (Selection(adjusted=True), ['asked', 'unmet']),
    (Selection(adjusted=False), ['bare', 'plain']),
    (Selection(unmet=True), ['unmet']),
    (Selection(several_runs=True), ['unmet']),
    (Selection(several_runs=False), ['asked', 'bare', 'plain'])])
def test_the_filters_select_by_what_the_index_keeps(index, selection, ids):
    assert sorted(r.id for r in selection.rows(index)) == ids


def test_a_row_says_the_facts_and_a_title_with_no_entry_has_no_runs(index):
    rows = {r.id: r for r in index.titles()}
    assert (rows['unmet'].adjusted, rows['unmet'].unmet_adjustments, rows['unmet'].design_runs) == (True, 1, 2)
    assert rows['bare'].design_runs == 0
    assert Selection(adjusted=True, unmet=False, several_runs=True).describe() == \
        'asked to change something, with every adjustment honoured, with earlier design runs'


def test_the_designs_to_choose_from_count_every_kept_run(tmp_path):
    ''' The Titles list's Designs column: every run's candidates, not how many runs (the designer's rejected ones aside). '''
    one = EntryFacts.of(_entry(tmp_path, 'one'), 1, 2)
    assert one.choices == one.candidates
    earlier = _entry(tmp_path, 'earlier', variants=[{'variant_id': 'v1', 'candidates': [{}, {}], 'rejected': [{}]},
                                                    {'variant_id': 'v2', 'candidates': [{}]}])
    facts = EntryFacts.of(earlier, 1, 2)
    assert facts.choices == len(earlier.candidates) + 3


def test_a_row_counts_the_designs_of_every_run_and_falls_back_to_the_current_runs(tmp_path):
    def summary(entry_id, **fields):
        return EntryFacts.of(_entry(tmp_path / 'q', entry_id, **fields), 1, 2).to_json()
    old = EntryFacts.of(_entry(tmp_path / 'q', 'old'), 1, 2)
    old_summary = old.to_json().replace('"choices": %d' % old.choices, '"choices": 0')   # written before it was kept
    rows = [dict(title_row('two', 'Two', needs='review', candidate_count=1),
                 entry_summary=summary('two', variants=[{'variant_id': 'v1', 'candidates': [{}, {}]}])),
            dict(title_row('old', 'Old', needs='review', candidate_count=4), entry_summary=old_summary)]
    make_index(tmp_path, rows)
    with LibraryIndex(index_path(str(tmp_path)), readonly=True) as opened:
        found = {r.id: r for r in opened.titles()}
    assert found['two'].choices == found['two'].design_choices == 2 + len(read_entry(str(tmp_path / 'q'), 'two').candidates)
    assert (found['old'].design_choices, found['old'].choices) == (0, 4)
