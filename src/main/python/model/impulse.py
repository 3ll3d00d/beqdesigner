from typing import List, Callable, Dict

from qtpy import QtCore
from qtpy.QtWidgets import QDialog, QAbstractItemView, QListWidgetItem, QWidget

from model.limits import ImpulseRangeCalculator
from model.magnitude import MagnitudeModel
from model.preferences import Preferences, IMPULSE_GRAPH_X_MIN, IMPULSE_GRAPH_X_MAX, get_filter_colour
from model.signal import Signal
from model.xy import MagnitudeData
from ui.channel_select import Ui_channelSelectDialog
from ui.impulse import Ui_impulseDialog


class ImpulseDialog(QDialog, Ui_impulseDialog):

    def __init__(self, parent: QWidget, prefs: Preferences, signals: List[Signal]):
        super(ImpulseDialog, self).__init__(parent)
        self.prefs = prefs
        self.setupUi(self)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowSystemMenuHint | QtCore.Qt.WindowMinMaxButtonsHint)
        self.__signals: Dict[Signal, bool] = {s: True for s in signals}
        self.__magnitude_model = MagnitudeModel('preview', self.previewChart, prefs,
                                                self.__get_data(), 'Signal', fill_primary=False,
                                                x_min_pref_key=IMPULSE_GRAPH_X_MIN, x_max_pref_key=IMPULSE_GRAPH_X_MAX,
                                                y_range_calc=ImpulseRangeCalculator())
        self.limitsButton.setToolTip('Set graph axis limits')

    def show_limits(self):
        ''' shows the limits dialog for the chart. '''
        self.__magnitude_model.show_limits()

    def update_chart(self):
        self.__magnitude_model.redraw()

    def __get_data(self):
        return lambda *args, **kwargs: self.get_curve_data(*args, **kwargs)

    def get_curve_data(self, reference=None):
        ''' preview of the filter to display on the chart '''
        result = []
        mode = 'I' if self.chartToggle.isChecked() else 'S'
        if self.__dsp:
            names = [n.text() for n in self.channelList.selectedItems()]
            for signal, selected in self.__signals.items():
                if selected:
                    result.append(MagnitudeData(signal.name, None, *signal.avg, colour=get_filter_colour(len(result))))
        return result

    def select_channels(self):
        '''
        Allow user to select signals to examine.
        '''
        def on_save(selected: List[str]):
            self.__signals = {s: s.name in selected for s in self.__signals.keys()}
            self.update_chart()

        SignalSelectorDialog(self, [s.name for s in self.__signals.keys()],
                             [s.name for s, b in self.__signals.items() if b], on_save).show()


class SignalSelectorDialog(QDialog, Ui_channelSelectDialog):

    def __init__(self, parent: QDialog, signals: List[str], selected_signals: List[str],
                 on_save: Callable[[List[str]], None]):
        super(SignalSelectorDialog, self).__init__(parent)
        self.setupUi(self)
        self.channelList.setSelectionMode(QAbstractItemView.MultiSelection)
        self.__on_save = on_save
        self.setWindowTitle('Select Signals')
        self.lfeChannel.hide()
        self.lfeChannelLabel.hide()
        for c in signals:
            self.channelList.addItem(c)
        for c in selected_signals:
            item: QListWidgetItem
            for item in self.channelList.findItems(c, QtCore.Qt.MatchCaseSensitive):
                item.setSelected(True)

    def accept(self):
        self.__on_save([i.text() for i in self.channelList.selectedItems()])
        super().accept()

