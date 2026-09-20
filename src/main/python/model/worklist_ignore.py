'''
The ignore half of the work list's settings drawer (chunk 26c): the profile's ignore rules and per-title ignores, and the
dialog that edits one rule with a live "would ignore N titles" count.

A rule is `pipeline.library.ignore.IgnoreRule`; this only builds one from the widgets with `rule_from_config` and shows its
ValueError where the person is typing. The count comes from the index's rows (`model.worklist_edit.preview_rules`), so it
needs no scan -- and says what it cannot judge from them.
'''
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout, \
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget

from model.worklist_edit import IgnorePreview, format_external_ids, parse_external_ids, preview_rules, prefill_from_row
from model.worklist_model import warning_colour
from pipeline.library.ignore import IgnoreRule, rule_from_config
from pipeline.library.index import TitleRow

RULE_HELP = ('A rule ignores every title it matches: all the fields you tick must match. An ignored title stays in the list '
             'under Done, labelled with the rule, and deleting the rule brings it back. Rules only take effect on titles '
             'when the library is scanned again.')


def describe_rule(rule: IgnoreRule) -> str:
    return rule.describe() or '(empty rule)'


class IgnoreRuleDialog(QDialog):
    '''
    Adds or edits one ignore rule.
    :param rows: the index's rows, for the live count.
    :param other_rules: the profile's other rules (they also ignore titles, so the count says what this one adds).
    :param existing: the rule being edited.
    :param prefill: values for the fields (`worklist_edit.prefill_from_row`); the folder's path is the one field ticked.
    :param row: when the dialog was opened for one work-list row: it offers "Ignore just this title" instead.
    '''
    FIELDS = ('source', 'path', 'title', 'year', 'kind', 'external_ids')

    def __init__(self, parent, rows: Sequence[TitleRow], source_names: Sequence[str], other_rules: Sequence[IgnoreRule],
                 ignored_titles: Mapping[str, str], existing: Optional[IgnoreRule] = None,
                 prefill: Optional[Mapping[str, str]] = None, row: Optional[TitleRow] = None):
        super().__init__(parent)
        self.setWindowTitle('Edit ignore rule' if existing else 'Ignore titles like this')
        self._rows, self._source_names = list(rows), list(source_names)
        self._other, self._ignored = list(other_rules), dict(ignored_titles)
        self._row = row
        self._rule: Optional[IgnoreRule] = None
        self.ignore_only_this = False
        self.checks: Dict[str, QCheckBox] = {}
        self.editors: Dict[str, QWidget] = {}
        self.sourceCombo = QComboBox()
        self.sourceCombo.setEditable(True)
        self.sourceCombo.addItems(self._source_names)
        self.pathEdit = QLineEdit()
        self.pathEdit.setPlaceholderText('/films/Kids  or  /films/*/extras  or  D:\\Films\\**\\Trailers')
        self.titleEdit = QLineEdit()
        self.titleEdit.setPlaceholderText('a regular expression, e.g. ^The .* Trilogy$')
        self.yearEdit = QLineEdit()
        self.yearEdit.setPlaceholderText('1960, <1960, >=1999 or 1990-1999')
        self.kindCombo = QComboBox()
        self.kindCombo.addItems(['movie', 'tv'])
        self.idsEdit = QLineEdit()
        self.idsEdit.setPlaceholderText('imdb=tt0113277, tmdb=603')
        self.reasonEdit = QLineEdit()
        self.reasonEdit.setPlaceholderText('optional: why, shown wherever the rule is named')
        labels = {'source': 'Source', 'path': 'Path (folder or glob)', 'title': 'Title matches', 'year': 'Year',
                  'kind': 'Kind', 'external_ids': 'External ids'}
        editors = {'source': self.sourceCombo, 'path': self.pathEdit, 'title': self.titleEdit, 'year': self.yearEdit,
                   'kind': self.kindCombo, 'external_ids': self.idsEdit}
        grid = QGridLayout()
        for number, name in enumerate(self.FIELDS):
            check = QCheckBox(labels[name])
            self.checks[name], self.editors[name] = check, editors[name]
            editors[name].setEnabled(False)
            check.toggled.connect(editors[name].setEnabled)
            check.toggled.connect(self.__update)
            grid.addWidget(check, number, 0)
            grid.addWidget(editors[name], number, 1)
        grid.addWidget(QLabel('Reason'), len(self.FIELDS), 0)
        grid.addWidget(self.reasonEdit, len(self.FIELDS), 1)
        grid.setColumnStretch(1, 1)
        self.helpLabel = QLabel(RULE_HELP)
        self.helpLabel.setWordWrap(True)
        self.errorLabel = QLabel('')
        self.previewLabel = QLabel('')
        self.notesLabel = QLabel('')
        for label in (self.errorLabel, self.previewLabel, self.notesLabel):
            label.setWordWrap(True)
        self.errorLabel.setStyleSheet(f'color: {warning_colour().name()}')
        self.notesLabel.setStyleSheet('color: palette(mid)')
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.thisTitleButton = QPushButton('Ignore just this title')
        self.thisTitleButton.setToolTip('Ignore only this one title, whatever else it shares with others')
        self.thisTitleButton.clicked.connect(self.__ignore_this_title)
        self.thisTitleButton.setVisible(row is not None)
        bottom = QHBoxLayout()
        bottom.addWidget(self.thisTitleButton)
        bottom.addStretch()
        bottom.addWidget(self.buttons)
        layout = QVBoxLayout(self)
        if row is not None:
            layout.addWidget(QLabel(f'<b>{_escape(row.title or row.display_name)}</b>'
                                    f'{" (" + row.year + ")" if row.year else ""} &nbsp; {_escape(row.path)}'))
        layout.addWidget(self.helpLabel)
        layout.addLayout(grid)
        layout.addWidget(self.errorLabel)
        layout.addWidget(self.previewLabel)
        layout.addWidget(self.notesLabel)
        layout.addLayout(bottom)
        for edit in (self.pathEdit, self.titleEdit, self.yearEdit, self.idsEdit, self.reasonEdit):
            edit.textChanged.connect(self.__update)
        self.kindCombo.currentTextChanged.connect(self.__update)
        self.sourceCombo.editTextChanged.connect(self.__update)
        if existing is not None:
            self.__load_rule(existing)
        elif prefill:
            self.__load_prefill(prefill)
        self.setMinimumWidth(560)
        self.__update()

    # --- filling and reading ----------------------------------------------------------------------------------------

    def __load_rule(self, rule: IgnoreRule) -> None:
        values = {'source': rule.source, 'path': rule.path, 'title': rule.title, 'year': rule.year, 'kind': rule.kind,
                  'external_ids': format_external_ids(dict(rule.external_ids)) if rule.external_ids else None}
        for name, value in values.items():
            if value is not None:
                self.set_value(name, value, True)
        self.reasonEdit.setText(rule.reason or '')

    def __load_prefill(self, values: Mapping[str, str]) -> None:
        ''' Every value the row offers is in its field; only the folder is ticked (ticking all would match one title). '''
        for name, value in values.items():
            if name in self.editors and value:
                self.set_value(name, value, name == 'path')

    def set_value(self, name: str, value: str, use: bool = True) -> None:
        ''' Puts `value` in the field and ticks (or not) that the rule constrains it. '''
        editor = self.editors[name]
        if isinstance(editor, QLineEdit):
            editor.setText(value)
        elif name == 'kind':
            editor.setCurrentText(value)
        else:
            editor.setCurrentText(value)
        self.checks[name].setChecked(use)

    def config(self) -> Dict[str, object]:
        ''' The rule as a profile's `ignore:` entry, from the ticked fields. '''
        entry: Dict[str, object] = {}
        for name in self.FIELDS:
            if not self.checks[name].isChecked():
                continue
            editor = self.editors[name]
            if name == 'external_ids':
                entry[name] = parse_external_ids(editor.text())
            elif isinstance(editor, QLineEdit):
                entry[name] = editor.text().strip()
            else:
                entry[name] = editor.currentText().strip()
        if self.reasonEdit.text().strip():
            entry['reason'] = self.reasonEdit.text().strip()
        return entry

    def build_rule(self) -> IgnoreRule:
        '''
        :raises ValueError: with what `rule_from_config` (or the field) says is wrong -- shown under the fields as you type.
        '''
        entry = self.config()
        blank = [name for name, value in entry.items() if name != 'reason' and value in ('', {})]
        if blank:
            raise ValueError(f'{blank[0].replace("_", " ")} is ticked but empty')
        return rule_from_config(entry)

    @property
    def rule(self) -> Optional[IgnoreRule]:
        ''' The rule, once the dialog was accepted (None if it was "just this title"). '''
        return self._rule

    @property
    def reason(self) -> str:
        return self.reasonEdit.text().strip()

    # --- the live check ---------------------------------------------------------------------------------------------

    def __update(self, *_) -> None:
        try:
            rule = self.build_rule()
        except ValueError as error:
            self.errorLabel.setText(str(error))
            self.previewLabel.setText('')
            self.notesLabel.setText('')
            self.notesLabel.setVisible(False)
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
            return
        self.errorLabel.setText('')
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        preview = preview_rules(self._rows, self._other + [rule], self._ignored, self._source_names)
        self.previewLabel.setText(rule_preview_text(preview, len(self._other)))
        mine = f'Rule {len(self._other) + 1}'
        notes = [n.replace(mine, 'This rule', 1) for n in preview.notes if n.startswith(mine)]
        self.notesLabel.setText('\n'.join(notes))
        self.notesLabel.setVisible(bool(notes))

    def accept(self) -> None:
        try:
            self._rule = self.build_rule()
        except ValueError as error:
            self.errorLabel.setText(str(error))
            return
        super().accept()

    def __ignore_this_title(self) -> None:
        self.ignore_only_this = True
        self._rule = None
        super().accept()

    @property
    def title_id(self) -> Optional[str]:
        return self._row.id if self._row is not None else None


def rule_preview_text(preview: IgnorePreview, other_count: int) -> str:
    ''' What one rule (the last of `preview`'s) matches, and what the rules and ids ignore altogether. '''
    if not preview.scanned:
        return preview.text()
    own = preview.per_rule[other_count] if len(preview.per_rule) > other_count else 0
    return f'This rule matches {own:,} title{"" if own == 1 else "s"}. ' + preview.text()


def _escape(text: str) -> str:
    import html
    return html.escape(text or '')


class IgnoreTab(QWidget):
    '''
    The profile's ignore rules and the titles ignored one by one. `changed(rules, ignored_titles)` says what they are now.
    '''
    changed = Signal(object, object)

    def __init__(self, prefs=None, parent=None, run_dialog: Optional[Callable[[QDialog], bool]] = None):
        super().__init__(parent)
        self._rules: List[IgnoreRule] = []
        self._ignored: Dict[str, str] = {}
        self._rows: List[TitleRow] = []
        self._source_names: List[str] = []
        self._run_dialog = run_dialog or (lambda dialog: dialog.exec() == QDialog.DialogCode.Accepted)
        self.helpLabel = QLabel(RULE_HELP)
        self.helpLabel.setWordWrap(True)
        self.ruleList = QListWidget()
        self.ruleList.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.addButton, self.editButton, self.removeButton = QPushButton('Add...'), QPushButton('Edit...'), \
            QPushButton('Remove')
        self.previewLabel = QLabel('')
        self.previewLabel.setWordWrap(True)
        self.notesLabel = QLabel('')
        self.notesLabel.setWordWrap(True)
        self.notesLabel.setStyleSheet('color: palette(mid)')
        self.titleList = QListWidget()
        self.titleList.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.titleList.setMaximumHeight(110)
        self.unignoreButton = QPushButton('Stop ignoring')
        buttons = QHBoxLayout()
        for button in (self.addButton, self.editButton, self.removeButton):
            buttons.addWidget(button)
        buttons.addStretch()
        title_buttons = QHBoxLayout()
        title_buttons.addWidget(QLabel('<b>Titles ignored one by one</b> (right-click a title in the list to add one)'))
        title_buttons.addStretch()
        title_buttons.addWidget(self.unignoreButton)
        layout = QVBoxLayout(self)
        layout.addWidget(self.helpLabel)
        layout.addWidget(self.ruleList, 1)
        layout.addLayout(buttons)
        layout.addWidget(self.previewLabel)
        layout.addWidget(self.notesLabel)
        layout.addLayout(title_buttons)
        layout.addWidget(self.titleList)
        self.ruleList.itemSelectionChanged.connect(self.__update_buttons)
        self.titleList.itemSelectionChanged.connect(self.__update_buttons)
        self.ruleList.itemDoubleClicked.connect(lambda _item: self.edit_selected())
        self.addButton.clicked.connect(lambda: self.add_rule())
        self.editButton.clicked.connect(lambda: self.edit_selected())
        self.removeButton.clicked.connect(lambda: self.remove_selected())
        self.unignoreButton.clicked.connect(lambda: self.unignore_selected())
        self.__update_buttons()

    # --- state ------------------------------------------------------------------------------------------------------

    @property
    def rules(self) -> tuple:
        return tuple(self._rules)

    @property
    def ignored_titles(self) -> Dict[str, str]:
        return dict(self._ignored)

    def set_state(self, rules: Sequence[IgnoreRule], ignored_titles: Mapping[str, str], rows: Sequence[TitleRow],
                  source_names: Sequence[str]) -> None:
        ''' Shows these rules and ignored ids over the index's rows; nothing is emitted. '''
        self._rules, self._ignored = list(rules), dict(ignored_titles)
        self.set_rows(rows, source_names)

    def set_rows(self, rows: Sequence[TitleRow], source_names: Optional[Sequence[str]] = None) -> None:
        self._rows = list(rows)
        if source_names is not None:
            self._source_names = list(source_names)
        self.refresh()

    def refresh(self) -> None:
        ''' The lists and the live preview. '''
        selected = self.ruleList.currentRow()
        self.ruleList.clear()
        for number, rule in enumerate(self._rules, 1):
            self.ruleList.addItem(f'{number}.  {describe_rule(rule)}')
        if 0 <= selected < self.ruleList.count():
            self.ruleList.setCurrentRow(selected)
        self.titleList.clear()
        names = {row.id: row.title or row.display_name for row in self._rows}
        for title_id, reason in self._ignored.items():
            item = QListWidgetItem(f'{names.get(title_id) or title_id}' + (f' - {reason}' if reason else ''))
            item.setData(Qt.ItemDataRole.UserRole, title_id)
            item.setToolTip(title_id)
            self.titleList.addItem(item)
        preview = self.preview()
        self.previewLabel.setText(preview.text() if (self._rules or self._ignored) else
                                  'No ignore rules: nothing is ignored.')
        self.notesLabel.setText('\n'.join(preview.notes))
        self.notesLabel.setVisible(bool(preview.notes))
        self.__update_buttons()

    def preview(self) -> IgnorePreview:
        ''' What the rules and the ignored ids would ignore of the index's rows now. '''
        return preview_rules(self._rows, self._rules, self._ignored, self._source_names)

    def __update_buttons(self) -> None:
        has_rule = self.ruleList.currentRow() >= 0 and bool(self.ruleList.selectedItems())
        self.editButton.setEnabled(has_rule)
        self.removeButton.setEnabled(has_rule)
        self.unignoreButton.setEnabled(bool(self.titleList.selectedItems()))

    def __emit(self) -> None:
        self.refresh()
        self.changed.emit(self.rules, self.ignored_titles)

    def rename_source(self, old: str, new: str) -> None:
        ''' Rules that named a source that was renamed follow it. Emits only if one did. '''
        renamed = False
        for i, rule in enumerate(self._rules):
            if rule.source == old:
                entry = dict(rule.to_config(), source=new)
                self._rules[i] = rule_from_config(entry)
                renamed = True
        if renamed:
            self.__emit()

    # --- rules ------------------------------------------------------------------------------------------------------

    def _dialog(self, existing: Optional[IgnoreRule] = None, prefill=None, row=None) -> IgnoreRuleDialog:
        others = [r for r in self._rules if r is not existing]
        return IgnoreRuleDialog(self, self._rows, self._source_names, others, self._ignored, existing, prefill, row)

    def add_rule(self, prefill: Optional[Mapping[str, str]] = None, row: Optional[TitleRow] = None) -> bool:
        '''
        Opens the rule editor for a new rule (optionally filled from a work-list row).
        :return: True if a rule or an ignored title was added.
        '''
        dialog = self._dialog(None, prefill, row)
        if not self._run_dialog(dialog):
            return False
        if dialog.ignore_only_this and dialog.title_id:
            self._ignored[dialog.title_id] = dialog.reason
        elif dialog.rule is not None:
            self._rules.append(dialog.rule)
        else:
            return False
        self.__emit()
        return True

    def ignore_like(self, row: TitleRow) -> bool:
        ''' "Ignore titles like this...": the rule editor filled from the row, with "just this title" as the other way. '''
        return self.add_rule(prefill_from_row(row), row)

    def edit_selected(self) -> bool:
        index = self.ruleList.currentRow()
        if index < 0 or not self.ruleList.selectedItems():
            return False
        dialog = self._dialog(self._rules[index])
        if not self._run_dialog(dialog) or dialog.rule is None:
            return False
        self._rules[index] = dialog.rule
        self.__emit()
        return True

    def remove_selected(self) -> bool:
        index = self.ruleList.currentRow()
        if index < 0 or not self.ruleList.selectedItems():
            return False
        del self._rules[index]
        self.__emit()
        return True

    # --- ids --------------------------------------------------------------------------------------------------------

    def ignore_ids(self, ids: Sequence[str], reason: str = '') -> None:
        ''' Ignores these titles one by one. '''
        for title_id in ids:
            self._ignored[title_id] = reason
        if ids:
            self.__emit()

    def unignore_selected(self) -> bool:
        ids = [item.data(Qt.ItemDataRole.UserRole) for item in self.titleList.selectedItems()]
        for title_id in ids:
            self._ignored.pop(title_id, None)
        if ids:
            self.__emit()
        return bool(ids)
