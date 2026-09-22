'''
Sending titles back -- design/library-sync/workflow-rework/design.md §12.8, chunk 27c. The work list's *Revise...* (on the
title page, and on the selected rows) and the "settings changed" banner all come here; the work itself is
`pipeline.library.revise`.

Three depths, each including the one before (the words are `CHOICES`):

* **Reopen for review** -- back to *Waiting for review*; the candidates, metadata, artwork and projects are kept.
* **Redesign** -- as that, and the design is marked out of date so *Extract & design* designs it again. What a
  person set (metadata, artwork, their note, an edit to a `.beq` project) is kept.
* **Re-extract** -- as that, and the extracted audio is forgotten so the next run runs ffmpeg again.

The pipeline work starts straight away in the Library Work List.  The Review Folder has no pipeline runner, so there these
only change state. What happens to a title's catalogue files depends on how far it got (`ReviseSummary` counts the cases and the dialog
says so *before* anything changes): a published title needs the XML repository (and the images repository, for its image),
because reopening one written but not committed puts its files back as git has them, and reopening one already committed
starts a *revision*, which publishing again writes to the same path.

`revise_titles()` never raises for a title: a title that cannot be sent back is a line in `ReviseOutcome.failed`, with why.
`ReviseDialog` is the question ("Reopen 3 titles?"): the depth, an optional reason (it goes into each title's reviewer
note) and, when a choice cannot be carried out, the reason, with OK disabled.
'''
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from qtpy.QtWidgets import QButtonGroup, QDialog, QDialogButtonBox, QLabel, QLineEdit, QRadioButton, QVBoxLayout
from qtpy.QtCore import Qt
from qtpy.QtGui import QTextDocument

from model.worklist_confirm import _plural
from model.worklist_run import LEVEL_ERROR, LEVEL_OK, LEVEL_WARN, ResultLine
from pipeline.library.index import TitleRow
from pipeline.library.revise import REVISE_TARGETS, revise_entry
from pipeline.publish.git import RepoTarget

logger = logging.getLogger('worklist')

# (label, what it does), in order of depth
CHOICES: Dict[str, Tuple[str, str]] = {
    'review': ('Reopen for review',
               'Back to "Waiting for review". Its candidates, metadata, artwork and projects are kept and nothing is '
               'redesigned; you pick again.'),
    'design': ('Redesign',
               'As reopening, and the design is marked out of date so Extract & design can design it again. '
               'What you set is kept: metadata, artwork, your note and any edit you made to a project.'),
    'extract': ('Re-extract and redesign',
                'As redesigning, and the extracted audio is forgotten, so the next run extracts it again with ffmpeg '
                '(another audio stream, say).'),
}
assert tuple(CHOICES) == REVISE_TARGETS

_DONE_WORDS = {'review': 'reopened for review', 'design': 'sent back for redesign', 'extract': 'sent back for re-extraction'}
_OUTCOMES = {'review': 'Reopened', 'design': 'Sent back for redesign', 'extract': 'Sent back for re-extraction'}


@dataclass(frozen=True)
class ReviseContext:
    ''' What revising needs to know of the setup: where the entries and audio are, and the repositories a published title's files are in. '''
    queue_dir: str
    work_dir: str = ''
    xml_repo: Optional[RepoTarget] = None
    images_repo: Optional[RepoTarget] = None
    xml_dir: str = ''
    image_dir: str = ''


def revise_context(setup) -> Optional[ReviseContext]:
    ''' From a `WorkListSetup`; None while there is no queue directory (nothing can be revised). '''
    settings = setup.settings if setup is not None else None
    if settings is None or not settings.queue_dir:
        return None
    return ReviseContext(settings.queue_dir, settings.work_dir, RepoTarget(settings.xml_repo) if settings.xml_repo else None,
                         RepoTarget(settings.images_repo) if settings.images_repo else None, settings.xml_dir,
                         settings.image_dir)


@dataclass(frozen=True)
class ReviseSummary:
    '''
    What the titles to revise are, in the terms that matter to what revising does. `published` are those whose files are
    in the catalogue repositories; `in_catalogue` counts those of them the repository has *committed* (reopening starts a
    revision) and the rest are written but not committed (reopening takes the files out again).
    '''
    count: int
    published: int = 0
    in_catalogue: int = 0
    accepted: int = 0
    pending: int = 0
    decided: int = 0        # skipped or rejected
    no_entry: int = 0
    first: str = ''         # the first title's name, for a heading about one

    @classmethod
    def of(cls, states: Sequence[Tuple[str, str]], first: str = '') -> 'ReviseSummary':
        '''
        :param states: one (queue entry status, commit state) per title; the status is `none` when there is no entry, and the
            commit state is the index's (`committed`, `pushed`, `uncommitted`, ... or '' if not known).
        '''
        published = [(s, c) for s, c in states if s == 'published']
        return cls(len(states), len(published), sum(1 for _, c in published if c in ('committed', 'pushed')),
                   sum(1 for s, _ in states if s == 'accepted'), sum(1 for s, _ in states if s == 'pending'),
                   sum(1 for s, _ in states if s in ('skipped', 'rejected')), sum(1 for s, _ in states if s == 'none'),
                   first)

    @classmethod
    def from_rows(cls, rows: Sequence[TitleRow]) -> 'ReviseSummary':
        ''' From the index's rows (which say published as an accepted title whose publish state is written or out of date). '''
        def status(row: TitleRow) -> str:
            if row.review_state == 'accepted' and row.publish_state in ('written', 'out_of_date'):
                return 'published'
            return row.review_state
        return cls.of([(status(r), r.commit_state) for r in rows], (rows[0].title or rows[0].id) if rows else '')

    @property
    def unchanged_by_reopen(self) -> int:
        ''' Titles a plain reopen leaves alone: already waiting, or with nothing designed. '''
        return self.pending + self.no_entry


def revise_problem(to: str, summary: ReviseSummary, context: Optional[ReviseContext]) -> str:
    ''' Why `to` cannot be carried out for these titles, in words for the dialog; empty if it can. Nothing has changed yet. '''
    if context is None:
        return 'No review queue directory is set (Settings > Locations).'
    if summary.count == 0:
        return 'No title is selected.'
    if summary.no_entry == summary.count:
        return 'Nothing has been designed for ' + ('this title' if summary.count == 1 else 'these titles') + \
               ' yet, so there is nothing to send back.'
    if to == 'review' and summary.unchanged_by_reopen == summary.count:
        return ('This title is' if summary.count == 1 else 'These titles are') + ' already waiting for review.'
    if summary.published and context.xml_repo is None:
        return (f'{_plural(summary.published, "title")} {"is" if summary.published == 1 else "are"} published, so '
                f'{"its files are" if summary.published == 1 else "their files are"} in the catalogue repositories: set '
                f'the XML repository in Settings > Locations first.')
    if to == 'extract' and not context.work_dir:
        return 'No work directory is set (Settings > Locations), so there is no extraction to forget.'
    return ''


def revise_text(to: str, summary: ReviseSummary, run_now: bool = False) -> Tuple[str, str]:
    ''' :return: (heading, body html) saying exactly what `to` will do to these titles. '''
    label, what = CHOICES[to]
    noun = summary.first if summary.count == 1 and summary.first else _plural(summary.count, 'title')
    if to in ('design', 'extract') and run_now:
        lines = [what, '<br>After this change, Extract &amp; design starts for these titles straight away. Review the new '
                       'candidates when it finishes; Publish and Commit remain separate.']
    else:
        lines = [what, '<br>Nothing is redesigned, extracted or published now: this only changes what each title needs next. '
                       'Run Extract &amp; design, then Publish and Commit, as usual.']
    if summary.accepted:
        lines.append(f'<br>{_plural(summary.accepted, "title")} {"is" if summary.accepted == 1 else "are"} accepted: '
                     f'the accept is undone and you decide again.')
    if summary.in_catalogue:
        lines.append(f'<br>{_plural(summary.in_catalogue, "title")} {"is" if summary.in_catalogue == 1 else "are"} '
                     f'already committed to the catalogue. {"Its" if summary.in_catalogue == 1 else "Their"} files stay '
                     f'there until you publish again, which writes the same path (a revision); then commit it.')
    written = summary.published - summary.in_catalogue
    if written > 0:
        lines.append(f'<br>{_plural(written, "title")} {"is" if written == 1 else "are"} published but not committed: '
                     f'{"its files are" if written == 1 else "their files are"} taken out of the repositories\' working '
                     f'trees again (deleted, or put back as last committed).')
    if to == 'review' and summary.unchanged_by_reopen:
        lines.append(f'<br>{_plural(summary.unchanged_by_reopen, "title")} {"is" if summary.unchanged_by_reopen == 1 else "are"} '
                     f'already waiting for review or not designed yet, and {"is" if summary.unchanged_by_reopen == 1 else "are"} '
                     f'left as {"it is" if summary.unchanged_by_reopen == 1 else "they are"}.')
    if summary.no_entry and to != 'review':
        lines.append(f'<br>{_plural(summary.no_entry, "title")} {"has" if summary.no_entry == 1 else "have"} nothing designed '
                     f'yet and {"is" if summary.no_entry == 1 else "are"} left as {"it is" if summary.no_entry == 1 else "they are"}.')
    lines.append('<br>A note with your reason is added to each title\'s reviewer note.')
    return f'{label}: {noun}?', '<br>'.join(lines)


# --- doing it --------------------------------------------------------------------------------------------------------------

@dataclass
class ReviseOutcome:
    to: str
    revised: List[str] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)              # (id, why)
    reverted: Dict[str, List[str]] = field(default_factory=dict)             # id -> catalogue files put back as git has them
    extract_forgotten: List[str] = field(default_factory=list)               # ids whose recorded extraction was forgotten
    revisions: Dict[str, int] = field(default_factory=dict)                  # id -> the revision number now


def _why(error: Exception) -> str:
    if isinstance(error, FileNotFoundError):
        return 'it has no queue entry: nothing has been designed for it'
    if isinstance(error, subprocess.CalledProcessError):
        return f'git failed: {error}'
    if isinstance(error, ValueError):
        return str(error)
    return f'{type(error).__name__}: {error}'


def revise_titles(context: ReviseContext, ids: Sequence[str], to: str, reason: str = '') -> ReviseOutcome:
    '''
    `pipeline.library.revise.revise_entry()` over each title. **One that cannot be sent back is reported, not raised**, and
    the others go on: an entry that is already pending (for `review`), one with no queue entry, a published one whose
    repository is missing or not a git repository, a git error. A published entry is left exactly as it was if it fails.
    '''
    outcome = ReviseOutcome(to)
    for title_id in dict.fromkeys(ids):
        try:
            result = revise_entry(context.queue_dir, title_id, to, reason.strip(), work_dir=context.work_dir or None,
                                  xml_repo=context.xml_repo, images_repo=context.images_repo, xml_dir=context.xml_dir,
                                  image_dir=context.image_dir)
        except Exception as error:   # per title: a bad one does not stop the rest, and nothing is raised into the window
            if not isinstance(error, (FileNotFoundError, ValueError)):
                logger.exception('Could not revise %s', title_id)
            outcome.failed.append((title_id, _why(error)))
            continue
        outcome.revised.append(title_id)
        outcome.revisions[title_id] = result.entry.revision
        if result.reverted:
            outcome.reverted[title_id] = list(result.reverted)
        if result.extract_invalidated:
            outcome.extract_forgotten.append(title_id)
    return outcome


def summarise_outcome(outcome: ReviseOutcome) -> Tuple[str, str]:
    ''' :return: (one line, level) -- what was done and what could not be. '''
    done, failed = len(outcome.revised), len(outcome.failed)
    if not done and not failed:
        return 'Nothing to do.', LEVEL_OK
    parts = []
    if done:
        parts.append(f'{_plural(done, "title")} {_DONE_WORDS[outcome.to]}')
    if failed:
        parts.append(f'{failed:,} could not be changed' + (' (see the Last run tab)' if failed > 1 or done else f': {outcome.failed[0][1]}'))
    text = '; '.join(parts)
    text = text[0].upper() + text[1:] + '.'
    if done and outcome.to == 'extract' and not outcome.extract_forgotten:
        text += ' (Their recorded extraction was already gone.)'
    return text, LEVEL_ERROR if failed and not done else LEVEL_WARN if failed else LEVEL_OK


def describe_outcome(outcome: ReviseOutcome, titles: Mapping[str, str]) -> List[ResultLine]:
    ''' One line per title for the *Last run* tab: what was done, or why not; problems first. '''
    lines = [ResultLine(i, titles.get(i, i), 'Not changed', why, LEVEL_ERROR) for i, why in outcome.failed]
    for title_id in outcome.revised:
        detail = []
        if outcome.reverted.get(title_id):
            detail.append('catalogue files put back as git has them: ' + ', '.join(outcome.reverted[title_id]))
        if outcome.revisions.get(title_id):
            detail.append(f'revision {outcome.revisions[title_id]}')
        lines.append(ResultLine(title_id, titles.get(title_id, title_id), _OUTCOMES[outcome.to], '; '.join(detail)))
    return lines


# --- the question -----------------------------------------------------------------------------------------------------------

class ReviseDialog(QDialog):
    '''
    :param summary: what the titles are (`ReviseSummary.of()` / `.from_rows()`).
    :param context: None if the setup cannot revise (no queue directory).
    :param default: the choice to start on.
    Modal; Cancel is the default button, so Enter never sends anything back. OK is disabled, with the reason shown, while
    the chosen depth cannot be carried out.
    '''

    def __init__(self, parent, summary: ReviseSummary, context: Optional[ReviseContext], default: str = 'review',
                 run_now: bool = False):
        super().__init__(parent)
        self._summary, self._context, self._run_now = summary, context, run_now
        self.setModal(True)
        layout = QVBoxLayout(self)
        self.heading = QLabel()
        layout.addWidget(self.heading)
        self.radios: Dict[str, QRadioButton] = {}
        group = QButtonGroup(self)
        for to, (label, what) in CHOICES.items():
            radio = QRadioButton(label)
            radio.setToolTip(what)
            group.addButton(radio)
            radio.toggled.connect(lambda checked, to=to: checked and self._changed())
            self.radios[to] = radio
            layout.addWidget(radio)
        self.body = QLabel()
        self.body.setTextFormat(Qt.TextFormat.RichText)
        self.body.setWordWrap(True)
        self.body.setMinimumWidth(480)
        layout.addWidget(self.body)
        self.problemLabel = QLabel()
        self.problemLabel.setWordWrap(True)
        layout.addWidget(self.problemLabel)
        self.reasonEdit = QLineEdit()
        self.reasonEdit.setPlaceholderText('Why? (optional: it goes into the reviewer note)')
        layout.addWidget(self.reasonEdit)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.ok_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.cancel_button = self.buttons.button(QDialogButtonBox.StandardButton.Cancel)
        self.cancel_button.setDefault(True)
        self.ok_button.setAutoDefault(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.set_choice(default if default in CHOICES else 'review')

    @property
    def choice(self) -> str:
        return next(to for to, radio in self.radios.items() if radio.isChecked())

    @property
    def reason(self) -> str:
        return self.reasonEdit.text().strip()

    @property
    def problem(self) -> str:
        return revise_problem(self.choice, self._summary, self._context)

    def set_choice(self, to: str) -> None:
        self.radios[to].setChecked(True)
        self._changed()

    def _changed(self) -> None:
        to = self.choice
        heading, body = revise_text(to, self._summary, self._run_now)
        self.heading.setText(f'<b>{heading}</b>')
        self.setWindowTitle(heading)
        self.body.setText(body)
        problem = self.problem
        self.problemLabel.setText(problem)
        self.problemLabel.setVisible(bool(problem))
        from model.worklist_model import warning_colour
        self.problemLabel.setStyleSheet(f'color: {warning_colour().name()}')
        self.ok_button.setEnabled(not problem)
        count = self._summary.count
        self.ok_button.setText(f'{CHOICES[to][0]} ({count:,})' if count != 1 else CHOICES[to][0])

    @property
    def text(self) -> str:
        ''' Everything the dialog says, as plain text (the tests read this). '''
        document = QTextDocument()
        document.setHtml(self.body.text())
        return f'{self.windowTitle()}\n{document.toPlainText()}\n{self.problemLabel.text()}'
