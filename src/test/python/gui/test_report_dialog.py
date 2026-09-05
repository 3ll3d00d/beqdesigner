'''
Safety net for model/report.py's SaveReportDialog.__table_print(), which now
delegates to pipeline.publish.report.table_print() instead of its own inline
copy -- including a real bug that copy had (see table_print's docstring):
when gain == 0 for a filter reaching the "else" branch, the original did
`vals.append(f"..." if gain != 0 else vals.append('N/A'))`, appending both
'N/A' *and* the outer append's None, silently corrupting that row's column
alignment. A PeakingEQ with gain=0.0 below exercises exactly that path.
'''
import ui.beq  # noqa: F401 -- see test_filter_dialog.py's comment: must be imported before
               # model.report (-> ui.report -> app) to resolve their mid-file circular import.

from qtpy.QtCore import QSettings

from model.filter import FilterModel
from model.iir import CompleteFilter, PeakingEQ
from model.preferences import Preferences
from model.report import SaveReportDialog


def _make_preferences(tmp_path):
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    return Preferences(settings)


class _NoSignalsModel:
    def get_all_magnitude_data(self):
        return []


def _make_dialog(qtbot, tmp_path, filters):
    prefs = _make_preferences(tmp_path)
    filter_model = FilterModel(None, prefs)
    filter_model.filter = CompleteFilter(fs=1000, filters=filters)
    dialog = SaveReportDialog(None, prefs, _NoSignalsModel(), filter_model, status_bar=None, selected_signal=None)
    qtbot.addWidget(dialog)
    return dialog


def test_table_print_delegates_to_pipeline_publish_report(qtbot, tmp_path):
    dialog = _make_dialog(qtbot, tmp_path, [PeakingEQ(1000, 40.0, 2.0, -3.0)])
    dialog.showTableHeader.setChecked(True)

    row = dialog._SaveReportDialog__table_print(dialog._SaveReportDialog__filter_model.filter[0])

    assert row == ['40.0', '-3', '2.0', 'PEQ', '']


def test_table_print_handles_a_zero_gain_filter_without_corrupting_the_row(qtbot, tmp_path):
    '''
    The bug case: a filter reaching the gain-bearing branch with gain == 0
    exactly (e.g. a PEQ created with 0dB gain) used to append an extra
    None cell. Row must stay exactly 5 cells.
    '''
    dialog = _make_dialog(qtbot, tmp_path, [PeakingEQ(1000, 40.0, 2.0, 0.0)])
    dialog.showTableHeader.setChecked(True)

    row = dialog._SaveReportDialog__table_print(dialog._SaveReportDialog__filter_model.filter[0])

    assert row == ['40.0', 'N/A', '2.0', 'PEQ', '']
    assert len(row) == 5
    assert None not in row
