'''
Plain, Qt-free workflow configuration for the headless pipeline.

design/api-headless-pipeline.md §1 ("Configuration") identifies
ANALYSIS_TARGET_FS/ANALYSIS_RESOLUTION/avg-peak windows as workflow inputs
that must be supplyable per request, not read from a process-wide
Preferences singleton -- see B1 (model/xy.py's import-time Preferences
construction). AnalysisConfig is that explicit input: a plain dataclass a
caller builds directly (a script, a test, an API request) with no
dependency on QSettings/Preferences/qtpy at all.

The defaults below mirror the existing preference defaults
(model/preferences.py's DEFAULT_PREFS for the same keys) so a caller that
doesn't care can just use AnalysisConfig() and get today's behaviour.
Mapping a *user's actual* QSettings onto this is a GUI-side concern (design
doc §10) and deliberately does not live here.
'''
from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisConfig:
    target_fs: int = 1000
    resolution: float = 1.0
    avg_window: str = 'Default'
    peak_window: str = 'Default'
