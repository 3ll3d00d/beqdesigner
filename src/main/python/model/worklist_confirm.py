'''
The work list's confirmations -- chunk 26b. Publish and commit write into the catalogue repositories, and a run over a whole
view can take hours, so each says exactly what it is about to do, to which repositories and how many titles, and does
nothing until the person agrees.

`ConfirmDialog` is a plain modal `QDialog` (heading, body, an optional checkbox, and OK / Cancel buttons named for the
action), so a test drives it as a user would; the `*_text()` functions build the words and are tested on their own.
'''
import html
import os
from typing import Optional, Tuple

from qtpy.QtCore import Qt
from qtpy.QtGui import QTextDocument
from qtpy.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QVBoxLayout

from pipeline.library.bulk import AcceptPlan
from pipeline.library.stages import PublishSettings


def _repo(label: str, path: str, directory: str) -> str:
    where = f' &nbsp;(directory <code>{html.escape(directory)}</code>)' if directory else ''
    return f'{html.escape(label)}: <code>{html.escape(os.path.normpath(path))}</code>{where}'


def _plural(count: int, noun: str) -> str:
    return f'{count:,} {noun}{"" if count == 1 else "s"}'


def publish_text(count: int, settings: PublishSettings, republishing: int = 0) -> Tuple[str, str]:
    '''
    :param republishing: how many of the `count` are already published and out of date in the repository (rewritten in place).
    :return: (heading, body html)
    '''
    lines = ["Writes each title's filter XML into the XML repository:",
             _repo('XML repository', settings.xml_repo.local_path, settings.xml_dir)]
    if settings.images_repo is not None:
        lines += ["and its report image into the images repository:",
                  _repo('Images repository', settings.images_repo.local_path, settings.image_dir)]
    else:
        lines.append('No images repository is set, so no report images are written.')
    if republishing:
        lines.append(f'<br>{_plural(republishing, "title")} of these {"is" if republishing == 1 else "are"} already '
                     f'published and out of date: their files are written again at the same path.')
    lines.append('<br>Nothing is committed or pushed. Commit is the next step, and separate.')
    return f'Publish {_plural(count, "title")}?', '<br>'.join(lines)


def commit_text(count: int, settings: PublishSettings, uncommitted: Optional[int] = None) -> Tuple[str, str]:
    '''
    :param uncommitted: how many of the `count` still have to be committed (the rest were committed earlier, with push
        unticked, and only need pushing); None if not known, which says what a commit does.
    '''
    push_only = uncommitted == 0
    lines = []
    if push_only:
        lines.append(f'{"This title is" if count == 1 else "These titles are"} already committed and only '
                     f'{"needs" if count == 1 else "need"} pushing. Nothing new is committed; each repository is pushed '
                     f'(if you leave the box below ticked):')
    else:
        whose = "this title's files" if count == 1 else "these titles' files"
        lines.append(f'Makes one commit per repository containing just {whose}:')
    if settings.images_repo is not None:
        lines.append('1. ' + _repo('Images repository', settings.images_repo.local_path, settings.image_dir))
        lines.append('2. ' + _repo('XML repository', settings.xml_repo.local_path, settings.xml_dir))
        lines.append('<br>The images repository goes first, so a pushed XML never points at a missing image.')
    else:
        lines.append(_repo('XML repository', settings.xml_repo.local_path, settings.xml_dir))
    if uncommitted and uncommitted < count:
        lines.append(f'<br>{_plural(count - uncommitted, "title")} of these {"is" if count - uncommitted == 1 else "are"} '
                     f'already committed and only {"needs" if count - uncommitted == 1 else "need"} pushing.')
    heading = f'Push {_plural(count, "title")}?' if push_only else f'Commit {_plural(count, "title")}?'
    return heading, '<br>'.join(lines)


def retry_text(count: int) -> Tuple[str, str]:
    lines = [f'Runs every title that failed before ({count:,}) again, <b>wherever it is in the library</b> -- not only '
             f'in the current view. This can take a long time; you can cancel, and the title in hand finishes first.']
    return f'Retry {_plural(count, "failed title")}?', '<br>'.join(lines)


def machine_text(count: int, everything_in_view: bool, view: str) -> Tuple[str, str]:
    lines = ['Extracts the audio and designs a filter for each. This can take a long time; you can cancel, and the title '
             'in hand finishes first.']
    if everything_in_view:
        lines.insert(0, f'Nothing is selected, so this works on <b>every title that needs it in the current view</b> '
                        f'({html.escape(view)}).')
    return f'Extract and design {_plural(count, "title")}?', '<br>'.join(lines)


def accept_text(plan: AcceptPlan) -> Tuple[str, str, str]:
    '''
    Bulk accept's confirmation (design.md §12.9): how many titles, at what confidence, and -- as the details -- every title that
    was left out and why, so nothing is accepted that a person was not told about.
    :return: (heading, body html, details plain text -- one line per left-out title, '' if none)
    '''
    count = len(plan.eligible)
    lines = [f'Chooses the designer\'s <b>top pick</b> for {_plural(count, "title")} whose confidence is '
             f'<b>{plan.threshold:.2f} or more</b> (the threshold is in Settings &gt; Locations), and marks '
             f'{"it" if count == 1 else "them"} accepted with the note "bulk accepted, confidence &gt;= {plan.threshold:.2f}".',
             'Nothing is published: Publish is the next step, and separate. You can still reopen any of them.']
    if plan.excluded:
        lines.append(f'<br>{_plural(len(plan.excluded), "title")} at or above the threshold {"is" if len(plan.excluded) == 1 else "are"} '
                     f'<b>left out</b> because {"it needs" if len(plan.excluded) == 1 else "they need"} a person to look (listed below).')
    if plan.below_threshold:
        lines.append(f'<br>{_plural(plan.below_threshold, "title")} waiting for review {"is" if plan.below_threshold == 1 else "are"} '
                     f'below the threshold and left for you.')
    if plan.not_for_review:
        lines.append(f'{_plural(plan.not_for_review, "title")} in the selection {"is" if plan.not_for_review == 1 else "are"} '
                     f'not waiting for review and {"is" if plan.not_for_review == 1 else "are"} not touched.')
    details = '\n'.join(f'{e.title}: {e.reason}' for e in plan.excluded)
    return f'Accept the top pick for {_plural(count, "title")}?', '<br>'.join(lines), details


def drift_text(count: int) -> str:
    ''' The banner: accepted or published titles that were designed under other settings (`pipeline.library.drift`). '''
    return (f'The settings changed since {_plural(count, "accepted or published title")} {"was" if count == 1 else "were"} designed. '
            f'{"It keeps" if count == 1 else "They keep"} the old design until you revise {"it" if count == 1 else "them"}.')


class ConfirmDialog(QDialog):
    '''
    :param ok_text: the accept button, named for what it does ("Publish 12 titles").
    :param checkbox: text of an optional checkbox (checked by `checkbox_checked`); its state is `checked` afterwards.
    :param details: plain text shown below the body in a scrolling box (the titles a bulk accept leaves out, and why).
    '''

    def __init__(self, parent, heading: str, body: str, ok_text: str, checkbox: Optional[str] = None,
                 checkbox_checked: bool = True, details: str = ''):
        super().__init__(parent)
        self.setWindowTitle(heading)
        self.setModal(True)
        layout = QVBoxLayout(self)
        title = QLabel(f'<b>{html.escape(heading)}</b>')
        layout.addWidget(title)
        self.body = QLabel(body)
        self.body.setTextFormat(Qt.TextFormat.RichText)
        self.body.setWordWrap(True)
        self.body.setMinimumWidth(460)
        layout.addWidget(self.body)
        self.details = None
        if details:
            self.details = QPlainTextEdit(details)
            self.details.setReadOnly(True)
            self.details.setMaximumHeight(170)
            self.details.setMinimumHeight(90)
            layout.addWidget(self.details)
        self.checkbox = None
        if checkbox:
            self.checkbox = QCheckBox(checkbox)
            self.checkbox.setChecked(checkbox_checked)
            layout.addWidget(self.checkbox)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.ok_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText(ok_text)
        self.cancel_button = self.buttons.button(QDialogButtonBox.StandardButton.Cancel)
        self.cancel_button.setDefault(True)   # Enter cancels; agreeing takes a deliberate click
        self.ok_button.setAutoDefault(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    @property
    def checked(self) -> bool:
        return self.checkbox is not None and self.checkbox.isChecked()

    @property
    def text(self) -> str:
        ''' Everything the dialog says, as plain text (the tests and the log read this). '''
        document = QTextDocument()
        document.setHtml(self.body.text())
        extra = f'\n{self.details.toPlainText()}' if self.details is not None else ''
        return f'{self.windowTitle()}\n{document.toPlainText()}{extra}'
