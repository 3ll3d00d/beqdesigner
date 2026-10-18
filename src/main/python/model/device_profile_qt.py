'''
The main window's device profile controls (model/device_profile.py): a Devices button whose menu ticks the profiles the
chart shows the filter on and offers to optimise the filter for them (model/device_optimise.py), a line under the filter
table with each one's largest error, and the dialogs that add, edit and remove a person's own profiles and that show and
apply an optimisation.
'''
from typing import Callable, List, Optional

from qtpy.QtCore import QObject, QRunnable, QThreadPool, Signal
from qtpy.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QHeaderView, QLabel, \
    QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox, QPushButton, QSpinBox, QTableWidget, \
    QTableWidgetItem, QToolButton, QVBoxLayout, QWidget

from model.device_profile import BUILTIN_PROFILES, RATES, DeviceProfile, all_profiles, custom_profiles, \
    save_custom_profiles, set_shown, shown_profile_names

MANAGE = 'Manage profiles…'
OPTIMISE = 'Optimise filter for these devices…'


class DeviceProfileControls:
    '''
    :param on_change: called when the shown profiles change (the window redraws its chart).
    :param summary: returns what to say under the filter table (FilterModel.device_summary).
    '''

    def __init__(self, parent: QWidget, preferences, on_change: Callable[[], None], summary: Callable[[], str],
                 on_optimise: Optional[Callable[[], None]] = None, can_optimise: Callable[[], bool] = lambda: False):
        self.parent, self.preferences, self.on_change, self.summary = parent, preferences, on_change, summary
        self.on_optimise, self.can_optimise = on_optimise, can_optimise
        self.button = QToolButton(parent)
        self.button.setObjectName('deviceProfilesButton')
        self.button.setText('Devices')
        self.button.setToolTip('Show the filter as a device stores it: its rate and coefficient precision')
        self.button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu = QMenu(self.button)
        self.button.setMenu(self.menu)
        self.label = QLabel(parent)
        self.label.setObjectName('deviceProfilesSummary')
        self.label.setWordWrap(True)
        self.label.setVisible(False)
        self.rebuild_menu()

    def rebuild_menu(self) -> None:
        self.menu.clear()
        shown = set(shown_profile_names(self.preferences))
        for profile in all_profiles(self.preferences):
            action = self.menu.addAction(f'{profile.name} ({profile.describe()})')
            action.setCheckable(True)
            action.setChecked(profile.name in shown)
            action.setData(profile.name)
            action.toggled.connect(lambda checked, name=profile.name: self.toggle(name, checked))
        self.menu.addSeparator()
        if self.on_optimise is not None:
            self.optimise_action = self.menu.addAction(OPTIMISE)
            self.optimise_action.triggered.connect(self.on_optimise)
            self.menu.aboutToShow.connect(self.enable_optimise)
            self.enable_optimise()
        self.menu.addAction(MANAGE).triggered.connect(self.manage)
        self.button.setText(f'Devices ({len(shown)})' if shown else 'Devices')

    def enable_optimise(self) -> None:
        ''' Only with a profile ticked and a peaking or shelf filter to adjust. '''
        self.optimise_action.setEnabled(bool(shown_profile_names(self.preferences)) and self.can_optimise())

    def toggle(self, name: str, shown: bool) -> None:
        set_shown(self.preferences, name, shown)
        count = len(shown_profile_names(self.preferences))
        self.button.setText(f'Devices ({count})' if count else 'Devices')
        self.on_change()

    def manage(self) -> None:
        if DeviceProfilesDialog(self.parent, self.preferences).exec() == QDialog.DialogCode.Accepted:
            self.rebuild_menu()
            self.on_change()

    def refresh_summary(self) -> None:
        text = self.summary()
        self.label.setText(text)
        self.label.setVisible(bool(text))


class DeviceProfilesDialog(QDialog):
    ''' The built-in profiles, read only, and a person's own, which can be added, edited and removed. '''

    def __init__(self, parent: Optional[QWidget], preferences):
        super().__init__(parent)
        self.preferences = preferences
        self.custom = list(custom_profiles(preferences))
        self.setWindowTitle('Device profiles')
        self.list = QListWidget(self)
        self.list.setObjectName('profiles')
        self.add_button, self.edit_button, self.delete_button = QPushButton('Add…'), QPushButton('Edit…'), \
            QPushButton('Delete')
        buttons = QHBoxLayout()
        for b in (self.add_button, self.edit_button, self.delete_button):
            buttons.addWidget(b)
        buttons.addStretch()
        self.box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Built-in profiles are the devices beqdesigner exports to; add your own for others.'))
        layout.addWidget(self.list)
        layout.addLayout(buttons)
        layout.addWidget(self.box)
        self.add_button.clicked.connect(self.add)
        self.edit_button.clicked.connect(self.edit)
        self.delete_button.clicked.connect(self.delete)
        self.list.currentRowChanged.connect(self.enable)
        self.box.accepted.connect(self.save)
        self.box.rejected.connect(self.reject)
        self.fill()

    def fill(self, select: Optional[str] = None) -> None:
        self.list.clear()
        for profile in (*BUILTIN_PROFILES, *self.custom):
            item = QListWidgetItem(f'{profile.name} - {profile.describe()}' + (' (built in)' if profile.builtin else ''))
            item.setData(256, profile.name)
            self.list.addItem(item)
            if profile.name == select:
                self.list.setCurrentItem(item)
        self.enable()

    def selected(self) -> Optional[DeviceProfile]:
        item = self.list.currentItem()
        name = item.data(256) if item is not None else None
        return next((p for p in self.custom if p.name == name), None)

    def enable(self, *_) -> None:
        editable = self.selected() is not None
        self.edit_button.setEnabled(editable)
        self.delete_button.setEnabled(editable)

    def add(self) -> None:
        profile = ProfileForm(self).ask()
        if profile is not None:
            self.custom.append(profile)
            self.fill(profile.name)

    def edit(self) -> None:
        current = self.selected()
        if current is None:
            return
        profile = ProfileForm(self, current).ask()
        if profile is not None:
            self.custom[self.custom.index(current)] = profile
            self.fill(profile.name)

    def delete(self) -> None:
        current = self.selected()
        if current is not None:
            self.custom.remove(current)
            self.fill()

    def save(self) -> None:
        try:
            save_custom_profiles(self.preferences, self.custom)
        except ValueError as error:
            QMessageBox.warning(self, 'Device profiles', str(error))
            return
        self.accept()


class ProfileForm(QDialog):
    ''' One profile: its name, the device's internal rate and how it stores coefficients. '''

    def __init__(self, parent: Optional[QWidget], profile: Optional[DeviceProfile] = None):
        super().__init__(parent)
        self.setWindowTitle('Device profile')
        self.name = QLineEdit(profile.name if profile else '')
        self.rate = QComboBox()
        for rate in RATES:
            self.rate.addItem(f'{rate // 1000} kHz', rate)
        self.storage = QComboBox()
        self.storage.addItem('float32', 'float32')
        self.storage.addItem('fixed point', 'fixed')
        self.integer_bits, self.fractional_bits = QSpinBox(), QSpinBox()
        self.integer_bits.setRange(1, 16)
        self.fractional_bits.setRange(0, 48)
        self.integer_bits.setValue(profile.integer_bits if profile else 5)
        self.fractional_bits.setValue(profile.fractional_bits if profile else 23)
        if profile:
            self.rate.setCurrentIndex(self.rate.findData(profile.rate))
            self.storage.setCurrentIndex(self.storage.findData(profile.storage))
        form = QFormLayout(self)
        form.addRow('Name', self.name)
        form.addRow('Internal rate', self.rate)
        form.addRow('Coefficients', self.storage)
        form.addRow('Integer bits (with sign)', self.integer_bits)
        form.addRow('Fractional bits', self.fractional_bits)
        self.box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        form.addRow(self.box)
        self.box.accepted.connect(self.accept)
        self.box.rejected.connect(self.reject)
        self.storage.currentIndexChanged.connect(self.enable)
        self.name.textChanged.connect(self.enable)
        self.enable()

    def enable(self, *_) -> None:
        fixed = self.storage.currentData() == 'fixed'
        self.integer_bits.setEnabled(fixed)
        self.fractional_bits.setEnabled(fixed)
        self.box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(self.name.text().strip()))

    def profile(self) -> DeviceProfile:
        return DeviceProfile(self.name.text().strip(), int(self.rate.currentData()), str(self.storage.currentData()),
                             self.integer_bits.value(), self.fractional_bits.value())

    def ask(self) -> Optional[DeviceProfile]:
        return self.profile() if self.exec() == QDialog.DialogCode.Accepted else None


class OptimiseSignals(QObject):
    done = Signal(object)
    failed = Signal(str)


class OptimiseJob(QRunnable):
    ''' One optimisation, off the window's thread: beqforge's search takes a few seconds for a long cascade. '''

    def __init__(self, filters: List, profiles: List):
        super().__init__()
        self.filters, self.profiles = filters, profiles
        self.signals = OptimiseSignals()

    def run(self):
        from model.device_optimise import optimise_for
        try:
            self.signals.done.emit(optimise_for(self.filters, self.profiles))
        except Exception as error:   # beqforge refuses a cascade it cannot model: say why, do not lose the window
            self.signals.failed.emit(f'{type(error).__name__}: {error}')


class OptimiseDialog(QDialog):
    '''
    The filter optimised for the chosen device profiles: each profile's largest error before and after, each parameter
    that changes, and Apply, offered only when the result is published and makes no profile worse.
    :param apply: called with the optimised filters (each with its original's id) when the person applies them.
    :param start: runs the job; default the global thread pool (tests run it in place).
    '''

    def __init__(self, parent: Optional[QWidget], filters: List, profiles: List, apply: Callable[[List], None],
                 start: Optional[Callable[[QRunnable], None]] = None):
        super().__init__(parent)
        self.setWindowTitle('Optimise for devices')
        self.apply_to, self.optimisation = apply, None
        self.status = QLabel(f'Optimising for {", ".join(p.name for p in profiles)}…')
        self.status.setObjectName('optimiseStatus')
        self.status.setWordWrap(True)
        self.errors = QTableWidget(0, 3)
        self.errors.setObjectName('optimiseErrors')
        self.errors.setHorizontalHeaderLabels(['Device', 'Error before (dB)', 'Error after (dB)'])
        self.changes = QTableWidget(0, 4)
        self.changes.setObjectName('optimiseChanges')
        self.changes.setHorizontalHeaderLabels(['Filter', 'Freq (Hz)', 'Q', 'Gain (dB)'])
        for table in (self.errors, self.changes):
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.box = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Close)
        self.apply_button = self.box.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply)
        self.box.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(QLabel('Largest error against the filter as designed, 2-200 Hz'))
        layout.addWidget(self.errors)
        layout.addWidget(QLabel('Parameters that change'))
        layout.addWidget(self.changes)
        layout.addWidget(self.box)
        self.resize(640, 420)
        self.job = OptimiseJob(list(filters), list(profiles))
        self.job.signals.done.connect(self.show_result)
        self.job.signals.failed.connect(self.show_failure)
        (start or QThreadPool.globalInstance().start)(self.job)

    def show_failure(self, message: str) -> None:
        self.status.setText(f'Could not optimise: {message}')

    def show_result(self, result) -> None:
        self.optimisation = result
        self.status.setText(result.describe() + ('' if result.applicable or not result.changed else
                                                 ' It would make a device worse, so it is shown, not offered.'))
        self.errors.setRowCount(len(result.profiles))
        for row, p in enumerate(result.profiles):
            after = '—' if p.after_db is None else f'{p.after_db:.2f}'
            for col, text in enumerate((f'{p.profile.name} ({p.profile.describe()})', _db(p.before_db), after)):
                self.errors.setItem(row, col, QTableWidgetItem(text))
        self.changes.setRowCount(len(result.changed))
        for row, (old, new) in enumerate(result.changed):
            cells = (old.filter_type + (f' x{old.count}' if getattr(old, 'count', 1) > 1 else ''),
                     f'{old.freq:g} → {new.freq:g}', f'{old.q:g} → {new.q:g}', f'{old.gain:g} → {new.gain:g}')
            for col, text in enumerate(cells):
                self.changes.setItem(row, col, QTableWidgetItem(text))
        self.apply_button.setEnabled(result.applicable)

    def apply(self) -> None:
        if self.optimisation is not None and self.optimisation.applicable:
            self.apply_to([new for _, new in self.optimisation.changed])
            self.accept()


def _db(value: float) -> str:
    return 'cannot be realised' if value == float('inf') else f'{value:.2f}'
