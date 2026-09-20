'''
model/worklist_run.py and model/worklist_confirm.py, chunk 26b: the words the work list shows -- what a selection would do
and what it leaves out, what a run did to each title, the one-line outcome, and the confirmations' text. No window here.
`import ui.beq` first: see AGENTS.md gotcha 3.
'''
import ui.beq  # noqa: F401 (must come first)

import pytest

from model.worklist_confirm import commit_text, machine_text, publish_text
from model.worklist_run import ResultLine, RunRequest, build_publish_settings, describe_results, failed_titles, \
    headline, plan_label, publish_problem, skip_reason, summarise_report, summarise_skipped
from pipeline.library.commit import CatalogueCommit, RepoCommit
from pipeline.library.index import LibraryIndex
from pipeline.library.run import LibraryRunReport
from pipeline.library.selection import plan_stages
from pipeline.library.stages import PublishSettings, StagesReport
from pipeline.library.status import FailureMemory, ScanSettings
from pipeline.publish.git import RepoTarget
from worklist_fixture import make_index, title_row

NOW = 1_800_000_000.0


@pytest.fixture
def rows(tmp_path):
    fixtures = [
        title_row('x1', 'Gravity', 'extract'), title_row('x2', 'Tenet', 'extract'),
        title_row('d1', 'Speed', 'design'), title_row('r1', 'Alien', 'review'),
        title_row('p1', 'Sicario', 'publish'), title_row('c1', 'Jaws', 'commit', commit_state='uncommitted'),
        title_row('c2', 'Ronin', 'commit', commit_state='committed'), title_row('z1', 'Old', 'done'),
        title_row('f1', 'Dune', 'attention', extract_state='failed', failure='no audio'),
        title_row('a1', 'Heat', 'attention', detail='source changed'),
    ]
    path = make_index(tmp_path, fixtures, [], generation=2, last_scan_at=NOW)
    with LibraryIndex(path) as index:
        return {row.id: row for row in index.titles()}


def _plan(rows, through, *ids, retry_failed=False):
    return plan_stages([rows[i] for i in ids], through, retry_failed=retry_failed)


def test_the_skipped_summary_groups_the_reasons_in_the_words_of_the_work_list(rows):
    plan = _plan(rows, 'design', 'x1', 'x2', 'd1', 'r1', 'p1', 'c1', 'z1', 'f1', 'a1')

    assert plan_label(plan) == 'Extract & design 3 (6 of 9 skipped)'
    text = summarise_skipped(plan, rows)
    assert text == ('6 of 9 skipped: 1 already accepted, 1 already done, 1 already published, 1 failed before, '
                    '1 needs attention, 1 waiting for review')


def test_a_selection_with_nothing_skipped_says_nothing_and_the_label_has_no_parenthesis(rows):
    plan = _plan(rows, 'design', 'x1', 'd1')

    assert summarise_skipped(plan, rows) == '' and plan_label(plan) == 'Extract & design 2'


def test_labels_carry_thousands_separators_and_the_verb_of_the_stage(rows):
    assert plan_label(_plan(rows, 'publish', 'p1')) == 'Publish 1'
    assert plan_label(_plan(rows, 'commit', 'c1')) == 'Commit 1'
    assert plan_label(_plan(rows, 'extract', 'x1')) == 'Extract 1'


def test_a_failed_title_is_skipped_as_failed_before_until_retry_is_asked_for(rows):
    assert skip_reason(rows['f1']) == 'failed before'
    assert _plan(rows, 'design', 'f1').planned == []
    retried = _plan(rows, 'design', 'f1', retry_failed=True)
    assert [p.row.id for p in retried.planned] == ['f1']


def test_the_failures_panel_lists_failed_titles_with_the_index_memory_of_why(rows):
    memory = {'f1': FailureMemory('extract', 'RuntimeError: no audio stream', 'fp', 'k')}

    failed = failed_titles(list(rows.values()), memory)

    assert [(f.id, f.title, f.stage, f.reason) for f in failed] == [
        ('f1', 'Dune', 'extract', 'RuntimeError: no audio stream')]
    # with no memory (a row rebuilt from outputs), the index's own failure text is used
    assert failed_titles(list(rows.values()), {})[0].reason == 'no audio'


def test_a_run_needs_titles():
    with pytest.raises(ValueError):
        RunRequest('design', ())


def test_results_put_problems_first_and_say_what_happened_to_every_title(rows):
    plan = _plan(rows, 'design', 'x1', 'x2', 'd1', 'f1')
    report = StagesReport('design', 4, run=LibraryRunReport(
        extracted=['x1', 'x2'], designed=['x1'], design_cached=['d1'], failed=[('x2', 'RuntimeError: boom')],
        failed_earlier=[('f1', 'no audio')]), attempted=['x1', 'x2', 'd1', 'f1'])

    lines = describe_results(report, plan, ScanSettings('/w', '/q'), rows)

    assert [(l.id, l.outcome, l.level) for l in lines] == [
        ('x2', 'Failed', 'error'), ('f1', 'Not retried', 'warn'), ('x1', 'Designed', 'ok'),
        ('d1', 'Already designed', 'ok')]
    by_id = {l.id: l for l in lines}
    assert by_id['x2'].detail == 'RuntimeError: boom'
    assert by_id['f1'].detail == 'failed before: no audio. Retry failed runs it again.'


def test_a_cancelled_run_lists_what_was_not_run_and_the_summary_says_how_far_it_got(rows):
    plan = _plan(rows, 'design', 'x1', 'x2', 'd1')
    report = StagesReport('design', 3, run=LibraryRunReport(designed=['x1']), cancelled=True, attempted=['x1'],
                          not_run=['x2', 'd1'])

    lines = {l.id: l for l in describe_results(report, plan, ScanSettings('/w', '/q'), rows)}
    text, level = summarise_report(report, plan)

    assert lines['x2'].outcome == 'Not run' and lines['d1'].outcome == 'Not run' and lines['x1'].outcome == 'Designed'
    assert text == 'Stopped after 1 of 3 titles (2 not run): 1 designed' and level == 'warn'


def test_the_outcome_of_a_run_that_failed_is_an_error_level_line(rows):
    plan = _plan(rows, 'design', 'x1', 'x2')
    report = StagesReport('design', 2, run=LibraryRunReport(designed=['x1'], failed=[('x2', 'boom')]))

    assert summarise_report(report, plan) == ('Extract & design finished: 1 designed, 1 failed', 'error')


def test_an_episode_that_failed_inside_a_season_is_listed_under_its_own_id(rows):
    plan = _plan(rows, 'design', 'x1')
    report = StagesReport('design', 1, run=LibraryRunReport(designed=['x1'], failed=[('ep-3', 'no audio')]))

    lines = {l.id: l for l in describe_results(report, plan, ScanSettings('/w', '/q'), rows)}

    assert lines['ep-3'] == ResultLine('ep-3', 'ep-3', 'Failed', 'no audio', 'error')


def test_commit_results_say_committed_pushed_or_nothing_new_per_title_and_repository(rows):
    settings = ScanSettings('/w', '/q', xml_dir='xml', image_dir='img')
    plan = _plan(rows, 'commit', 'c1')
    committed = CatalogueCommit(xml=RepoCommit('/repo/xml', ['xml/c1.xml'], 'abcdef0123456789', False),
                                images=RepoCommit('/repo/img', [], None, False), missing=['xml/gone.xml'])
    report = StagesReport('commit', 1, committed=committed, attempted=['c1'])

    lines = describe_results(report, plan, settings, rows)

    by_key = {l.id or l.title: l for l in lines}
    assert by_key['c1'].outcome == 'Committed'   # not pushed: no "pushed"
    assert by_key['XML repository'].detail == 'commit abcdef01, not pushed (1 file) -- /repo/xml'
    assert by_key['Images repository'].outcome == 'unchanged' and by_key['Images repository'].detail.startswith(
        'nothing new to commit, not pushed (0 files)')
    assert by_key['Published files missing'].level == 'warn'


SETTINGS = ScanSettings('/w', '/q', xml_dir='xml', image_dir='img')
REJECTED = ("git failed: To /srv/beq-xml.git\n ! [rejected]        main -> main (fetch first)\n"
            "error: failed to push some refs to '/srv/beq-xml.git'\nhint: Updates were rejected because the remote "
            "contains work that you do not have locally.")


def test_a_multi_line_git_failure_is_one_line_in_each_row_and_whole_once_in_the_repository_line(rows):
    plan = _plan(rows, 'commit', 'c1', 'c2')
    report = StagesReport('commit', 2, commit_error=REJECTED)

    lines = {l.id or l.title: l for l in describe_results(report, plan, SETTINGS, rows)}

    for title_id in ('c1', 'c2'):
        assert lines[title_id].outcome == 'Not committed' and lines[title_id].level == 'error'
        assert '\n' not in lines[title_id].detail
        assert lines[title_id].detail == 'git failed: ! [rejected] main -> main (fetch first) ...'   # the useful line
        assert lines[title_id].full == ''       # not repeated on every title
    assert lines['Commit'].full == REJECTED.strip() and '\n' not in lines['Commit'].detail
    assert sum(1 for l in lines.values() if l.full) == 1


def test_the_headline_of_a_failure_is_its_first_line_unless_that_is_only_where_git_pushed_to():
    assert headline('one line') == 'one line'
    assert headline('first\nsecond') == 'first ...'
    assert headline('git failed: To /x\n ! [rejected] a -> a\nerror: failed') == 'git failed: ! [rejected] a -> a ...'
    assert headline('') == ''


def test_a_commit_that_only_pushes_says_pushed_not_committed_and_the_summary_says_so(rows):
    plan = _plan(rows, 'commit', 'c2')
    committed = CatalogueCommit(xml=RepoCommit('/repo/xml', ['xml/c2.xml'], None, True))
    report = StagesReport('commit', 1, committed=committed, attempted=['c2'])

    lines = {l.id or l.title: l for l in describe_results(report, plan, SETTINGS, rows)}

    assert lines['c2'].outcome == 'Pushed' and lines['c2'].level == 'ok'
    assert summarise_report(report, plan, SETTINGS) == ('Commit finished: 1 pushed', 'ok')


def test_a_commit_and_push_over_a_mix_counts_each_kind_separately(rows):
    plan = _plan(rows, 'commit', 'c1', 'c2')
    committed = CatalogueCommit(xml=RepoCommit('/repo/xml', ['xml/c1.xml', 'xml/c2.xml'], 'abc12345', True))
    report = StagesReport('commit', 2, committed=committed, attempted=['c1', 'c2'])

    lines = {l.id: l for l in describe_results(report, plan, SETTINGS, rows) if l.id}

    assert lines['c1'].outcome == 'Committed, pushed' and lines['c2'].outcome == 'Pushed'
    assert summarise_report(report, plan, SETTINGS)[0] == 'Commit finished: 1 committed, 2 pushed'


def test_a_commit_with_push_unticked_over_already_committed_titles_does_nothing_and_says_so(rows):
    plan = _plan(rows, 'commit', 'c2')
    committed = CatalogueCommit(xml=RepoCommit('/repo/xml', ['xml/c2.xml'], None, False))
    report = StagesReport('commit', 1, committed=committed, attempted=['c2'])

    line = next(l for l in describe_results(report, plan, SETTINGS, rows) if l.id == 'c2')

    assert line.outcome == 'Already committed' and line.level == 'warn' and 'not pushed' in line.detail
    assert summarise_report(report, plan, SETTINGS)[0] == 'Commit finished: nothing new to commit'


def test_a_published_file_git_ignores_is_an_error_against_its_title_and_turns_the_summary_red(rows):
    plan = _plan(rows, 'commit', 'c1', 'c2')
    committed = CatalogueCommit(xml=RepoCommit('/repo/xml', ['xml/c2.xml'], 'abc12345', True),
                                images=RepoCommit('/repo/img', [], None, False), not_committed=['img/c1.png'])
    report = StagesReport('commit', 2, committed=committed, attempted=['c1', 'c2'])

    lines = {l.id: l for l in describe_results(report, plan, SETTINGS, rows) if l.id}
    text, level = summarise_report(report, plan, SETTINGS)

    assert lines['c1'].outcome == 'Not committed' and lines['c1'].level == 'error'
    assert 'img/c1.png' in lines['c1'].detail and '.gitignore' in lines['c1'].detail
    assert lines['c2'].outcome == 'Pushed'      # committed earlier, pushed now
    assert level == 'error' and '1 not committed (git ignores them)' in text


def test_a_warning_about_an_image_url_is_a_notice_that_blocks_nothing(rows):
    plan = _plan(rows, 'commit', 'c1')
    warning = 'xml/c1.xml names a report image (beq_spectrumURL) but no images repository was given'
    committed = CatalogueCommit(xml=RepoCommit('/repo/xml', ['xml/c1.xml'], 'abc12345', True), warnings=[warning])
    report = StagesReport('commit', 1, committed=committed, attempted=['c1'])

    lines = describe_results(report, plan, SETTINGS, rows)
    text, level = summarise_report(report, plan, SETTINGS)

    notice = next(l for l in lines if l.title == 'Notice')
    assert notice.level == 'warn' and notice.detail == warning and not notice.id
    assert next(l for l in lines if l.id == 'c1').outcome == 'Committed, pushed'    # nothing was blocked
    assert level == 'warn' and text == 'Commit finished: 1 committed, 1 pushed, 1 warning (see Last run)'


def test_each_kind_of_publish_error_is_shown_against_its_title_in_the_words_of_the_pipeline(rows):
    plan = _plan(rows, 'publish', 'p1')
    report = StagesReport('publish', 3, publish_errors=[
        {'id': 'p1', 'error': 'git_failed', 'message': 'git add x failed (exit 128): fatal: unable to write'},
        {'id': 'q2', 'error': 'publish_failed', 'message': 'FileNotFoundError: poster.jpg'},
        {'id': 'q3', 'error': 'invalid_metadata', 'problems': ['unknown field \'x\'']}])

    lines = {l.id: l for l in describe_results(report, plan, SETTINGS, rows)}

    assert (lines['p1'].outcome, lines['p1'].level) == ('Git failed', 'error')
    assert lines['p1'].detail == 'git refused: git add x failed (exit 128): fatal: unable to write'
    assert lines['q2'].outcome == 'Publish failed' and lines['q2'].detail == 'publishing failed: FileNotFoundError: poster.jpg'
    assert lines['q3'].outcome == 'Refused' and "unknown field 'x'" in lines['q3'].detail
    assert summarise_report(report, plan, SETTINGS) == ('Publish finished: 1 refused, 2 failed', 'error')


def test_a_whole_publish_failure_is_one_line_and_every_title_that_was_not_published_says_so(rows):
    plan = _plan(rows, 'publish', 'p1')
    report = StagesReport('publish', 1, publish_errors=[
        {'id': '', 'error': 'publish_failed', 'message': 'OSError: the queue folder is unreadable'}])

    lines = describe_results(report, plan, SETTINGS, rows)
    by_key = {l.id or l.title: l for l in lines}

    assert by_key['Publish'].level == 'error' and 'the queue folder is unreadable' in by_key['Publish'].detail
    assert by_key['p1'].outcome == 'Not published' and by_key['p1'].level == 'error'
    assert not any(l.title == '' for l in lines)     # no nameless title row
    assert summarise_report(report, plan, SETTINGS) == ('Publish finished: 1 failed', 'error')


def test_a_remote_that_is_not_github_is_explained_once_per_title_as_what_to_set(rows):
    plan = _plan(rows, 'publish', 'p1')
    report = StagesReport('publish', 1, publish_errors=[
        {'id': 'p1', 'error': 'publish_failed',
         'message': "ValueError: Unrecognised GitHub remote URL: '/srv/beq-images.git'"}])

    line = next(l for l in describe_results(report, plan, SETTINGS, rows) if l.id == 'p1')

    assert 'Unrecognised' not in line.detail
    assert line.detail.startswith('Set image_owner and image_repo_name') and '/srv/beq-images.git' in line.detail


# --- the settings ---------------------------------------------------------------------------------------------------------

class _Setup:
    def __init__(self, settings, config=None):
        self.settings = settings
        self.profile = type('P', (), {'config': config or {}})()


def test_publish_settings_come_from_the_scan_settings_and_the_profile_and_say_whether_to_push():
    setup = _Setup(ScanSettings('/w', '/q', xml_repo='/r/xml', images_repo='/r/img', xml_dir='x', image_dir='i'),
                   {'sync': {'image_owner': 'me', 'image_repo_name': 'imgs'}})

    settings = build_publish_settings(setup, push=False)

    assert settings == PublishSettings(RepoTarget('/r/xml'), RepoTarget('/r/img'), 'me', 'imgs', 'x', 'i', None,
                                       push=False)
    assert publish_problem(setup) == ''
    assert 'No XML repository' in publish_problem(_Setup(ScanSettings('/w', '/q')))
    with pytest.raises(ValueError):
        build_publish_settings(_Setup(ScanSettings('/w', '/q')))


# --- the confirmations' words -----------------------------------------------------------------------------------------------

def _settings(images=True):
    return PublishSettings(RepoTarget('/repos/beq-xml'), RepoTarget('/repos/beq-images') if images else None,
                           xml_dir='filters', image_dir='images')


def test_the_publish_confirmation_names_both_repositories_their_directories_and_the_count():
    heading, body = publish_text(12, _settings(), republishing=2)

    assert heading == 'Publish 12 titles?'
    for text in ('/repos/beq-xml', '/repos/beq-images', 'filters', 'images', '2 titles of these are already published'):
        assert text in body
    assert 'Nothing is committed or pushed' in body
    assert publish_text(1, _settings(images=False))[0] == 'Publish 1 title?'
    assert 'No images repository is set' in publish_text(1, _settings(images=False))[1]


def test_the_commit_confirmation_lists_images_before_xml_and_says_one_commit_per_repository():
    heading, body = commit_text(3, _settings())

    assert heading == 'Commit 3 titles?'
    assert body.index('beq-images') < body.index('beq-xml') and 'one commit per repository' in body
    assert "these titles' files" in body and "this title's files" in commit_text(1, _settings())[1]


def test_the_machine_run_confirmation_says_it_is_everything_in_the_view():
    heading, body = machine_text(120, True, 'Extract, source films')

    assert heading == 'Extract and design 120 titles?'
    assert 'every title that needs it in the current view' in body and 'Extract, source films' in body


def test_the_commit_confirmation_for_titles_that_are_only_waiting_to_be_pushed_says_nothing_is_committed():
    heading, body = commit_text(2, _settings(), uncommitted=0)

    assert heading == 'Push 2 titles?'
    assert 'Makes one commit' not in body and 'already committed' in body and 'Nothing new is committed' in body
    heading, body = commit_text(3, _settings(), uncommitted=1)
    assert heading == 'Commit 3 titles?' and 'Makes one commit per repository' in body
    assert '2 titles of these are already committed and only need pushing' in body
