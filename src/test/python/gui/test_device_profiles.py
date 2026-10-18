'''
The main window's device profiles (model/device_profile_qt.py): a Devices menu beside Show Filters ticks the profiles
the chart shows the filter on, dashed beside it, with each one's largest error under the filter table; a dialog adds a
person's own profiles.
'''
import ui.beq  # noqa: F401 -- must come before app/model.filter (see test_filter_dialog.py)

import logging

import pytest

from qtpy.QtCore import QSettings
from qtpy.QtWidgets import QApplication, QDialog

from model.device_profile import BUILTIN_PROFILES, DeviceProfile, custom_profiles, set_shown
from model.device_profile_qt import MANAGE, OPTIMISE, DeviceProfileControls, DeviceProfilesDialog, OptimiseDialog, \
    ProfileForm
from model.filter import FilterModel
from model.iir import COMBINED, CompleteFilter, PeakingEQ
from model.preferences import SYSTEM_CHECK_FOR_UPDATES, Preferences

HD, FIXED = BUILTIN_PROFILES


def _prefs(tmp_path):
    return Preferences(QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat))


def _model(prefs):
    model = FilterModel(None, prefs)
    model.filter = CompleteFilter(fs=1000, filters=[PeakingEQ(1000, 5, 6, 12)])
    return model


def test_the_chart_draws_the_combined_filter_dashed_on_each_shown_profile(tmp_path):
    prefs = _prefs(tmp_path)
    model = _model(prefs)
    assert [m.name for m in model.get_curve_data()] == [COMBINED]

    set_shown(prefs, HD.name, True)
    set_shown(prefs, FIXED.name, True)
    curves = {m.name: m for m in model.get_curve_data()}

    assert list(curves) == [COMBINED, f'{COMBINED} on {HD.name}', f'{COMBINED} on {FIXED.name}']
    on_hd = curves[f'{COMBINED} on {HD.name}']
    assert on_hd.linestyle == '--' and on_hd.colour == curves[COMBINED].colour
    lines = model.device_summary().splitlines()
    assert len(lines) == 2 and lines[0].startswith(f'{HD.name} (float32 at 96 kHz): max error ')


def test_normalised_to_the_combined_filter_a_device_curve_is_its_error(tmp_path):
    prefs = _prefs(tmp_path)
    model = _model(prefs)
    set_shown(prefs, HD.name, True)

    curves = {m.name: m for m in model.get_curve_data(reference=COMBINED)}

    assert abs(curves[COMBINED].y).max() < 1e-9
    assert abs(curves[f'{COMBINED} on {HD.name}'].y).max() > 10   # the float32 damage, alone


def test_the_menu_ticks_profiles_and_the_summary_follows(qtbot, tmp_path):
    prefs = _prefs(tmp_path)
    model = _model(prefs)
    changes = []
    parent = QDialog()
    qtbot.addWidget(parent)
    controls = DeviceProfileControls(parent, prefs, on_change=lambda: changes.append(True), summary=model.device_summary)

    actions = [a for a in controls.menu.actions() if a.isCheckable()]
    assert [a.data() for a in actions] == [HD.name, FIXED.name] and not any(a.isChecked() for a in actions)
    assert controls.menu.actions()[-1].text() == MANAGE
    controls.refresh_summary()
    assert controls.label.isHidden()

    actions[0].setChecked(True)
    controls.refresh_summary()

    assert changes and controls.button.text() == 'Devices (1)'
    assert not controls.label.isHidden() and controls.label.text().startswith(HD.name)
    actions[0].setChecked(False)
    controls.refresh_summary()
    assert controls.label.isHidden() and controls.button.text() == 'Devices'


def test_the_dialog_adds_a_profile_and_keeps_the_built_ins_read_only(qtbot, tmp_path, monkeypatch):
    prefs = _prefs(tmp_path)
    dialog = DeviceProfilesDialog(None, prefs)
    qtbot.addWidget(dialog)
    dialog.list.setCurrentRow(0)
    assert not dialog.edit_button.isEnabled() and not dialog.delete_button.isEnabled()

    mine = DeviceProfile('DCX2496', 96000, 'fixed', 8, 24)
    monkeypatch.setattr(ProfileForm, 'ask', lambda self: mine)
    dialog.add()
    assert dialog.list.count() == 3 and dialog.edit_button.isEnabled()
    dialog.save()

    assert dialog.result() == QDialog.DialogCode.Accepted and custom_profiles(prefs) == [mine]


def test_a_duplicate_name_is_not_saved(qtbot, tmp_path, monkeypatch):
    from qtpy.QtWidgets import QMessageBox
    prefs = _prefs(tmp_path)
    dialog = DeviceProfilesDialog(None, prefs)
    qtbot.addWidget(dialog)
    warned = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warned.append(args[2]))
    monkeypatch.setattr(ProfileForm, 'ask', lambda self: DeviceProfile(HD.name, 48000, 'float32'))
    dialog.add()
    dialog.save()

    assert warned and 'already a device profile' in warned[0] and custom_profiles(prefs) == []


def test_the_form_builds_a_profile_and_needs_a_name(qtbot):
    form = ProfileForm(None)
    qtbot.addWidget(form)
    ok = form.box.button(form.box.StandardButton.Ok)
    assert not ok.isEnabled() and not form.integer_bits.isEnabled()

    form.name.setText('mine')
    form.rate.setCurrentIndex(form.rate.findData(96000))
    form.storage.setCurrentIndex(form.storage.findData('fixed'))
    form.integer_bits.setValue(8)

    assert ok.isEnabled() and form.integer_bits.isEnabled()
    assert form.profile() == DeviceProfile('mine', 96000, 'fixed', 8, 23)


def test_the_main_window_has_the_devices_menu_beside_show_filters(qtbot, tmp_path):
    import app as app_module
    root = logging.getLogger()
    handlers = list(root.handlers)
    prefs = _prefs(tmp_path)
    prefs.set(SYSTEM_CHECK_FOR_UPDATES, False)
    main = app_module.BeqDesigner(QApplication.instance(), prefs)
    qtbot.addWidget(main)
    try:
        controls = main._BeqDesigner__device_profiles
        assert main.selectorsLayout.indexOf(controls.button) != -1
        assert main.rightPane.indexOf(controls.label) == main.rightPane.indexOf(main.filterView) + 1

        next(a for a in controls.menu.actions() if a.data() == HD.name).setChecked(True)   # no filter yet: nothing to say

        assert controls.button.text() == 'Devices (1)' and controls.label.isHidden()
    finally:
        for handler in list(root.handlers):
            if handler not in handlers:
                root.removeHandler(handler)


def _run_in_place(job):
    job.run()


def test_the_optimise_dialog_shows_each_device_and_change_and_applies_them(qtbot):
    peak = PeakingEQ(1000, 5, 6, 12)
    applied = []
    dialog = OptimiseDialog(None, [peak], [HD], applied.extend, start=_run_in_place)
    qtbot.addWidget(dialog)

    assert dialog.errors.rowCount() == 1 and dialog.errors.item(0, 0).text().startswith(HD.name)
    assert float(dialog.errors.item(0, 2).text()) < float(dialog.errors.item(0, 1).text())
    assert dialog.changes.rowCount() == 1 and dialog.changes.item(0, 1).text().startswith('5 → ')
    assert dialog.apply_button.isEnabled()

    dialog.apply_button.click()

    assert len(applied) == 1 and applied[0].id == peak.id and dialog.result() == QDialog.DialogCode.Accepted


def test_a_result_that_cannot_be_applied_is_shown_without_apply(qtbot):
    from model.iir import SecondOrder_HighPass
    applied = []
    dialog = OptimiseDialog(None, [PeakingEQ(1000, 5, 6, 12), SecondOrder_HighPass(1000, 3)], [HD], applied.extend,
                            start=_run_in_place)
    qtbot.addWidget(dialog)

    assert not dialog.apply_button.isEnabled() and 'unstable' in dialog.status.text()
    assert dialog.errors.item(0, 1).text() == 'cannot be realised' and applied == []


def test_a_failure_is_said_in_the_dialog(qtbot, monkeypatch):
    import model.device_optimise as optimise
    monkeypatch.setattr(optimise, 'optimise_for', lambda *a: (_ for _ in ()).throw(ValueError('bad cascade')))
    dialog = OptimiseDialog(None, [PeakingEQ(1000, 5, 6, 12)], [HD], lambda f: None, start=_run_in_place)
    qtbot.addWidget(dialog)
    assert dialog.status.text() == 'Could not optimise: ValueError: bad cascade' and not dialog.apply_button.isEnabled()


def test_optimise_is_offered_only_with_a_profile_ticked_and_a_filter_to_adjust(qtbot, tmp_path):
    prefs = _prefs(tmp_path)
    adjustable = [True]
    parent = QDialog()
    qtbot.addWidget(parent)
    controls = DeviceProfileControls(parent, prefs, on_change=lambda: None, summary=lambda: '',
                                     on_optimise=lambda: None, can_optimise=lambda: adjustable[0])
    assert controls.optimise_action.text() == OPTIMISE and not controls.optimise_action.isEnabled()

    set_shown(prefs, HD.name, True)
    controls.enable_optimise()
    assert controls.optimise_action.isEnabled()
    adjustable[0] = False
    controls.enable_optimise()
    assert not controls.optimise_action.isEnabled()


def test_optimising_from_the_main_window_changes_its_filter(qtbot, tmp_path, monkeypatch):
    import app as app_module
    import model.device_profile_qt as qt_module
    root = logging.getLogger()
    handlers = list(root.handlers)
    prefs = _prefs(tmp_path)
    prefs.set(SYSTEM_CHECK_FOR_UPDATES, False)
    main = app_module.BeqDesigner(QApplication.instance(), prefs)
    qtbot.addWidget(main)
    try:
        model = main._BeqDesigner__filter_model
        fs = model.filter.fs
        model.filter = CompleteFilter(fs=fs, filters=[PeakingEQ(fs, 5, 6, 12)])
        set_shown(prefs, HD.name, True)
        peak_id = model.filter[0].id

        def run_and_apply(self):   # exec() would block: run the job in place and press Apply
            self.apply_button.click()
            return QDialog.DialogCode.Accepted
        original_init = qt_module.OptimiseDialog.__init__
        monkeypatch.setattr(qt_module.OptimiseDialog, '__init__',
                            lambda self, *a, **k: original_init(self, *a, **{**k, 'start': _run_in_place}))
        monkeypatch.setattr(qt_module.OptimiseDialog, 'exec', run_and_apply)

        controls = main._BeqDesigner__device_profiles
        controls.menu.aboutToShow.emit()   # as opening the menu does: a profile is ticked and there is a peak to adjust
        assert controls.optimise_action.isEnabled()
        controls.optimise_action.trigger()

        [peak] = list(model.filter)
        assert peak.id == peak_id and (peak.freq, peak.q, peak.gain) != (5, 6, 12)
    finally:
        for handler in list(root.handlers):
            if handler not in handlers:
                root.removeHandler(handler)


# --- the filter edit dialog ---------------------------------------------------------------------------------------------

def _filter_dialog(qtbot, tmp_path, small=False):
    import numpy as np
    from model.filter import FilterDialog
    from model.signal import Signal
    prefs = _prefs(tmp_path)
    model = FilterModel(None, prefs)
    peak = PeakingEQ(1000, 5, 6, 12)
    model.filter = CompleteFilter(fs=1000, filters=[peak])
    dialog = FilterDialog(prefs, Signal('test', np.zeros(1000), fs=1000), model, lambda: None, selected_filter=peak,
                          small=small)
    qtbot.addWidget(dialog)
    return dialog, prefs, model


def test_the_filter_dialog_draws_the_working_filter_on_each_ticked_device(qtbot, tmp_path):
    dialog, prefs, _ = _filter_dialog(qtbot, tmp_path)
    controls = dialog._FilterDialog__device_profiles
    assert dialog.checkboxLayout.indexOf(controls.button) != -1
    assert not any(c.name.startswith(f'{COMBINED} on') for c in dialog.get_curve_data('mag'))

    next(a for a in controls.menu.actions() if a.data() == HD.name).setChecked(True)

    names = [c.name for c in dialog.get_curve_data('mag')]
    assert f'{COMBINED} on {HD.name}' in names
    assert not any(n.startswith(f'{COMBINED} on') for n in [c.name for c in dialog.get_curve_data('phase')])
    assert not controls.label.isHidden() and controls.label.text().startswith(HD.name)


def test_optimising_in_the_filter_dialog_changes_only_its_working_filter(qtbot, tmp_path, monkeypatch):
    import model.device_profile_qt as qt_module
    dialog, prefs, model = _filter_dialog(qtbot, tmp_path)
    set_shown(prefs, HD.name, True)
    original_init = qt_module.OptimiseDialog.__init__
    monkeypatch.setattr(qt_module.OptimiseDialog, '__init__',
                        lambda self, *a, **k: original_init(self, *a, **{**k, 'start': _run_in_place}))
    monkeypatch.setattr(qt_module.OptimiseDialog, 'exec', lambda self: (self.apply_button.click(), 1)[1])
    controls = dialog._FilterDialog__device_profiles
    controls.menu.aboutToShow.emit()
    assert controls.optimise_action.isEnabled()

    controls.optimise_action.trigger()

    [working] = list(dialog._FilterDialog__working)
    assert (working.freq, working.q, working.gain) != (5, 6, 12)
    # the edit fields show the new values, so the next edit does not write the old ones back
    assert (dialog.freq.value(), dialog.filterQ.value(), dialog.filterGain.value()) == \
        pytest.approx((working.freq, working.q, working.gain))
    [unsaved] = list(model.filter)
    assert (unsaved.freq, unsaved.q, unsaved.gain) == (5, 6, 12)   # nothing reaches the main window until Save


def test_the_small_filter_dialog_has_no_device_controls(qtbot, tmp_path):
    dialog, _, _ = _filter_dialog(qtbot, tmp_path, small=True)
    assert dialog._FilterDialog__device_profiles is None
