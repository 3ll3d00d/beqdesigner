# R10 — Configuration-folder hygiene

Completed 2026-10-12 in the implementation commit containing this record,
`Align application identity and library profile configuration paths`.

Desktop startup sets Qt's organization name to `3ll3d00d` and application
name to `beqdesigner` before reading settings or config locations. The main
preferences use `QSettings()` with that identity, preserving the existing
explicit `QSettings("3ll3d00d", "beqdesigner")` store. The default library
profile is in the organization config folder alongside the preferences file
on Linux, rather than an `app.py` folder. Explicit saved paths are retained.

GUI tests redirect `AppConfigLocation` into each test's temporary folder,
including the function used by the settings drawer's default argument. The
startup regression checks the identity at the moment settings are opened;
path tests check the real Qt naming and the temporary default save location.

Verification (with `PYTHONPATH=src/main/python`, `QT_QPA_PLATFORM=offscreen`,
`MPLCONFIGDIR=/tmp/beq-mpl`):

- `.venv/bin/python -m pytest src/test/python/gui/test_app_settings.py src/test/python/gui/test_worklist_settings.py -q`: 44 passed.
- `.venv/bin/python -m pytest src/test/python/gui -n 4 -q`: 791 passed (two existing matplotlib layout warnings).
