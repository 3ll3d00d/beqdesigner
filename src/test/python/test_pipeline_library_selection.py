'''
Selection (design.md §12.7): the one selector vocabulary shared by the CLI and the work list, what a selection picks
from the index, what `through` does to each kind of title, and -- the parity table chunk 26 builds on -- that every CLI
selector flag and every strip chip mean the same `Selection`.
'''
import pytest

from pipeline.library import cli
from pipeline.library.selection import CHIP_NEW, CHIPS, THROUGH, Selection, plan_stages, selection_from_chip
from pipeline.library.state import NEEDS
from test_pipeline_library_index import FakeSource, _entry, _extracted, _item, _scan, env  # noqa: F401 (a fixture)


# --- the vocabulary -----------------------------------------------------------------------------------------------

def test_the_vocabulary_is_the_designs():
    assert NEEDS == ('attention', 'review', 'extract', 'design', 'publish', 'commit', 'done')
    assert THROUGH == ('extract', 'design', 'publish', 'commit')
    assert CHIPS == ('Attention', 'New', 'Extract', 'Design', 'Review', 'Publish', 'Commit', 'Done')


def test_a_selection_rejects_a_need_that_does_not_exist():
    with pytest.raises(ValueError, match='bogus'):
        Selection(needs=('bogus',))


def test_a_selection_normalises_its_fields():
    selection = Selection(needs=['design', 'design', 'review'], source='', match='', ids=['b', 'a', 'b'])

    assert selection == Selection(needs=('design', 'review'), ids=('b', 'a')) and selection.source is None
    assert Selection().is_empty and not Selection(match='x').is_empty


def test_describe_says_what_a_selection_picks():
    assert Selection().describe() == 'every title'
    assert Selection(needs=('review',), source='disk', match='alien', ids=('a', 'b'), new_since_scan=True).describe() \
        == 'needs review, from disk, matching "alien", 2 named titles, new since the last scan'


# --- what a selection picks from the index -----------------------------------------------------------------------

def _seed(env):
    fresh, extracted, reviewed = _item('alien'), _item('blade'), _item('cube')
    for item in (extracted, reviewed):
        _extracted(env, item)
    _entry(env, reviewed)
    _scan(env, sources={'films': FakeSource([fresh, extracted, reviewed]),
                        'disk': FakeSource([_item('dune', source_path='/disk/dune.mkv')])})


def test_each_field_narrows_the_rows_and_together_they_intersect(env):
    _seed(env)

    def ids(selection):
        return [row.id for row in selection.rows(env.index)]

    assert sorted(ids(Selection())) == ['fs-alien', 'fs-blade', 'fs-cube', 'fs-dune']
    assert ids(Selection(needs=('review',))) == ['fs-cube']
    assert sorted(ids(Selection(needs=('extract', 'design')))) == ['fs-alien', 'fs-blade', 'fs-dune']
    assert ids(Selection(source='disk')) == ['fs-dune']
    assert ids(Selection(match='LAD')) == ['fs-blade']
    assert sorted(ids(Selection(ids=('fs-cube', 'fs-alien')))) == ['fs-alien', 'fs-cube']
    assert sorted(ids(Selection(new_since_scan=True))) == ['fs-alien', 'fs-blade', 'fs-cube', 'fs-dune']
    assert ids(Selection(needs=('extract', 'design'), source='disk', match='dun')) == ['fs-dune']
    assert ids(Selection(needs=('review',), source='disk')) == []
    assert ids(Selection(ids=[])) != []  # no ids given is no constraint, not "nothing"


def test_new_since_scan_follows_the_generation(env):
    _scan(env, _item('a'))
    _scan(env, _item('a'), _item('b'))

    assert [r.id for r in Selection(new_since_scan=True).rows(env.index)] == ['fs-b']


# --- what `through` does to each kind of title --------------------------------------------------------------------

def _plan(env, through, *needs, retry_failed=False):
    rows = env.index.titles(needs=list(needs))
    return plan_stages(rows, through, retry_failed=retry_failed)


def test_an_unknown_through_is_refused():
    with pytest.raises(ValueError, match='through'):
        plan_stages([], 'review')


@pytest.mark.parametrize('through, stages', [
    ('extract', ('extract',)), ('design', ('extract', 'design')), ('publish', ('extract', 'design')),
    ('commit', ('extract', 'design'))])
def test_a_title_needing_extract_runs_the_machine_stages_up_to_through_and_never_past_design(env, through, stages):
    _scan(env, _item('a'))

    plan = _plan(env, through, 'extract')

    assert [p.stages for p in plan.planned] == [stages] and plan.skipped == []


@pytest.mark.parametrize('through, planned', [('extract', False), ('design', True), ('publish', True), ('commit', True)])
def test_a_title_needing_design_is_designed_from_design_on(env, through, planned):
    item = _item('a')
    _extracted(env, item)
    _scan(env, item)

    plan = _plan(env, through, 'design')

    assert bool(plan.planned) is planned and (not planned) == bool(plan.skipped)
    if planned:
        assert plan.planned[0].stages == ('design',)
    else:
        assert plan.skipped[0].reason == 'already extracted'


def test_a_title_waiting_for_review_is_never_run_and_says_why(env):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item)
    _scan(env, item)

    for through in THROUGH:
        plan = _plan(env, through, 'review')
        assert plan.planned == [] and plan.skipped[0].reason == 'waiting for a person to review it'


def test_a_done_title_is_skipped_with_its_reason(env):
    item = _item('a')
    _extracted(env, item)
    _entry(env, item, status='rejected')
    _scan(env, item)

    plan = _plan(env, 'commit', 'done')

    assert plan.planned == [] and plan.skipped[0].reason == 'rejected'


def test_the_label_names_the_action_and_what_it_leaves_out(env):
    _scan(env, _item('a'), _item('b'))
    extracted = _item('c')
    _extracted(env, extracted)
    _entry(env, extracted)
    _scan(env, _item('a'), _item('b'), extracted)

    rows = env.index.titles()

    assert plan_stages(rows, 'design').label == 'Extract & design 2 (1 of 3 skipped)'
    assert plan_stages([r for r in rows if r.needs == 'extract'], 'extract').label == 'Extract 2'


# --- parity: the CLI's flags and the GUI's chips are one vocabulary ----------------------------------------------

def _from_cli(*argv):
    args = cli.build_parser().parse_args(['run', *argv])
    return cli._selection(args, args.source)


CLI_FLAGS_AND_CHIPS = [
    # (what the user types for the CLI, what the strip's chip selects)
    (('--needs', 'attention'), 'Attention'),
    (('--new-since-scan',), CHIP_NEW),
    (('--needs', 'extract'), 'Extract'),
    (('--needs', 'design'), 'Design'),
    (('--needs', 'review'), 'Review'),
    (('--needs', 'publish'), 'Publish'),
    (('--needs', 'commit'), 'Commit'),
    (('--needs', 'done'), 'Done'),
]


@pytest.mark.parametrize('flags, chip', CLI_FLAGS_AND_CHIPS, ids=[c for _, c in CLI_FLAGS_AND_CHIPS])
def test_each_chip_selects_what_its_cli_flag_does(flags, chip):
    assert selection_from_chip(chip) == _from_cli(*flags)


def test_every_chip_and_every_needs_value_is_in_the_parity_table():
    assert {chip for _, chip in CLI_FLAGS_AND_CHIPS} == set(CHIPS)
    assert {flags[1] for flags, chip in CLI_FLAGS_AND_CHIPS if flags[0] == '--needs'} == set(NEEDS)


@pytest.mark.parametrize('flags, kwargs', [
    (('--source', 'disk'), {'source': 'disk'}),
    (('--match', 'alien'), {'match': 'alien'}),
    (('--id', 'a', '--id', 'b'), {'ids': ('a', 'b')}),
    (('--needs', 'review', '--source', 'disk', '--match', 'x'), {'needs': ('review',), 'source': 'disk', 'match': 'x'}),
    (('--needs', 'design', '--needs', 'extract'), {'needs': ('design', 'extract')}),
], ids=['source', 'match', 'ids', 'chip+source+match', 'two needs'])
def test_the_source_combo_search_box_and_row_selection_are_the_other_flags(flags, kwargs):
    assert _from_cli(*flags) == Selection(**kwargs)


def test_a_chip_narrowed_by_the_combo_search_and_rows_is_the_same_flags_together():
    assert selection_from_chip('Review', source='disk', match='x') == _from_cli(
        '--needs', 'review', '--source', 'disk', '--match', 'x')
    assert selection_from_chip('Design', ids=('a', 'b')) == _from_cli('--needs', 'design', '--id', 'a', '--id', 'b')


def test_new_since_scan_combines_with_the_other_selectors_like_a_chip_with_the_combo_and_search():
    assert selection_from_chip('New', source='disk', match='a', ids=('x',)) == _from_cli(
        '--new-since-scan', '--source', 'disk', '--match', 'a', '--id', 'x')


def test_no_selector_flag_is_the_empty_selection():
    assert _from_cli().is_empty


def test_an_unknown_chip_is_refused():
    with pytest.raises(ValueError, match='chips'):
        selection_from_chip('Everything')


def test_the_through_flag_is_the_action_buttons_vocabulary():
    parser = cli.build_parser()
    (run,) = [a for a in parser._actions if hasattr(a, 'choices') and a.choices and 'run' in a.choices]
    through = next(a for a in run.choices['run']._actions if '--through' in a.option_strings)

    assert tuple(through.choices) == THROUGH
    needs = next(a for a in run.choices['run']._actions if '--needs' in a.option_strings)
    assert tuple(needs.choices) == NEEDS
