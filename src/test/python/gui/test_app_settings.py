"""Application identity keeps Qt's config paths and existing preferences in the same namespace."""
import ui.beq  # noqa: F401 (must come first)

import os

from qtpy.QtCore import QSettings, QStandardPaths

import app as app_module


def test_startup_sets_identity_before_opening_the_existing_settings(qapp, tmp_path, monkeypatch):
    old_org, old_app = qapp.organizationName(), qapp.applicationName()
    old_font, old_icon = qapp.font(), qapp.windowIcon()
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    opened = []

    def open_settings(*args):
        opened.append((args, qapp.organizationName(), qapp.applicationName()))
        return settings

    monkeypatch.setattr(app_module, 'QApplication', lambda args: qapp)
    monkeypatch.setattr(app_module, 'QSettings', open_settings)
    try:
        made, _ = app_module.make_app()
        assert made is qapp
        assert opened == [((), '3ll3d00d', 'beqdesigner')]
        # The implicit settings constructor still resolves to the existing explicit namespace.
        assert QSettings().fileName() == QSettings('3ll3d00d', 'beqdesigner').fileName()
    finally:
        qapp.setOrganizationName(old_org)
        qapp.setApplicationName(old_app)
        qapp.setFont(old_font)
        qapp.setWindowIcon(old_icon)


def test_profile_default_shares_the_organisation_config_folder(qapp, monkeypatch, _configuration_folder_is_temporary):
    from model.worklist_settings import PROFILE_FILE_NAME, default_profile_path
    old_org, old_app = qapp.organizationName(), qapp.applicationName()
    monkeypatch.setattr(QStandardPaths, 'writableLocation', _configuration_folder_is_temporary)
    qapp.setOrganizationName('3ll3d00d')
    qapp.setApplicationName('beqdesigner')
    try:
        root = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
        assert default_profile_path() == os.path.join(os.path.dirname(root), PROFILE_FILE_NAME)
        assert os.path.basename(os.path.dirname(default_profile_path())) == '3ll3d00d'
    finally:
        qapp.setOrganizationName(old_org)
        qapp.setApplicationName(old_app)
