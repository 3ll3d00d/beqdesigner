'''
The main window's device profiles (model/device_profile_qt.py): a Devices menu beside Show Filters ticks the profiles
the chart shows the filter on, dashed beside it, with each one's largest error under the filter table; a dialog adds a
person's own profiles.
'''
import ui.beq  # noqa: F401 -- must come before app/model.filter (see test_filter_dialog.py)

import logging

from qtpy.QtCore import QSettings
from qtpy.QtWidgets import QApplication, QDialog

from model.device_profile import BUILTIN_PROFILES, DeviceProfile, custom_profiles, set_shown
from model.device_profile_qt import MANAGE, DeviceProfileControls, DeviceProfilesDialog, ProfileForm
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
