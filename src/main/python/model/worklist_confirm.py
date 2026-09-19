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
from qtpy.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout

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


def commit_text(count: int, settings: PublishSettings) -> Tuple[str, str]:
    whose = "this title's files" if count == 1 else "these titles' files"
    lines = [f'Makes one commit per repository containing just {whose}:']
    if settings.images_repo is not None:
        lines.append('1. ' + _repo('Images repository', settings.images_repo.local_path, settings.image_dir))
        lines.append('2. ' + _repo('XML repository', settings.xml_repo.local_path, settings.xml_dir))
        lines.append('<br>The images repository goes first, so a pushed XML never points at a missing image.')
    else:
        lines.append(_repo('XML repository', settings.xml_repo.local_path, settings.xml_dir))
    return f'Commit {_plural(count, "title")}?', '<br>'.join(lines)


def machine_text(count: int, everything_in_view: bool, view: str) -> Tuple[str, str]:
    lines = ['Extracts the audio and designs a filter for each. This can take a long time; you can cancel, and the title '
             'in hand finishes first.']
    if everything_in_view:
        lines.insert(0, f'Nothing is selected, so this works on <b>every title that needs it in the current view</b> '
                        f'({html.escape(view)}).')
    return f'Extract and design {_plural(count, "title")}?', '<br>'.join(lines)


class ConfirmDialog(QDialog):
    '''
    :param ok_text: the accept button, named for what it does ("Publish 12 titles").
    :param checkbox: text of an optional checkbox (checked by `checkbox_checked`); its state is `checked` afterwards.
    '''

    def __init__(self, parent, heading: str, body: str, ok_text: str, checkbox: Optional[str] = None,
                 checkbox_checked: bool = True):
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
        return f'{self.windowTitle()}\n{document.toPlainText()}'
