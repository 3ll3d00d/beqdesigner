from typing import List, Callable, Dict, Optional

import numpy as np
import qtawesome as qta
from qtpy import QtCore
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QDialog, QAbstractItemView, QListWidgetItem, QWidget

from model.limits import ImpulseRangeCalculator
from model.magnitude import MagnitudeModel
from model.preferences import Preferences, IMPULSE_GRAPH_X_MIN, IMPULSE_GRAPH_X_MAX, get_filter_colour
from model.signal import Signal
from model.xy import MagnitudeData
from ui.channel_select import Ui_channelSelectDialog
from ui.impulse import Ui_impulseDialog


class ImpulseDialog(QDialog, Ui_impulseDialog):

    def __init__(self, parent: QWidget, prefs: Preferences, signals: Dict[Signal, bool]):
        super(ImpulseDialog, self).__init__(parent)
        self.prefs = prefs
        self.__cid: list = []
        self.__in_active_area: bool = False
        self.__dragging: Optional[None] = None
        self.setupUi(self)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.WindowSystemMenuHint | QtCore.Qt.WindowType.WindowMinMaxButtonsHint)
        self.__signals = signals
        self.__magnitude_model = MagnitudeModel('preview', self.previewChart, prefs,
                                                self.__get_data(), '', primary_prefix='%', fill_primary=False,
                                                x_min_pref_key=IMPULSE_GRAPH_X_MIN, x_max_pref_key=IMPULSE_GRAPH_X_MAX,
                                                y_range_calc=ImpulseRangeCalculator(),
                                                x_axis_configurer=self.__configure_time_axis, x_scale='linear')
        self.__magnitude_model.limits.axes_1.spines['right'].set_color('none')
        self.__magnitude_model.limits.axes_1.spines['top'].set_color('none')
        self.limitsButton.setToolTip('Set graph axis limits')
        self.limitsButton.setIcon(qta.icon('fa5s.arrows-alt'))
        self.selectChannelsButton.setIcon(qta.icon('fa5s.filter'))
        self.zoomInButton.setIcon(qta.icon('fa5s.compress'))
        self.zoomOutButton.setIcon(qta.icon('fa5s.expand'))
        self.__left_marker = self.__magnitude_model.limits.axes_1.axvline(x=self.__magnitude_model.limits.x_min,
                                                                          color=get_filter_colour(0), lw=0.8, ls='--')
        self.__right_marker = self.__magnitude_model.limits.axes_1.axvline(x=self.__magnitude_model.limits.x_max,
                                                                           color=get_filter_colour(1), lw=0.8, ls='--')
        self.__left_x: float = self.__left_marker.get_xdata()[0]
        self.__right_x: float = self.__right_marker.get_xdata()[0]
        self.__init_x_lim = self.__left_x, self.__right_x
        self.__timer = QTimer(self)
        self.__timer.timeout.connect(self.__redraw)
        self.__connect_mouse()
        self.update_chart()

    def update_signals(self, signals: Dict[Signal, bool]):
        self.__signals = signals
        self.__redraw()

    def __redraw(self):
        self.__update_left_marker()
        self.__update_right_marker()
        v = self.rightTimeValue.value() - self.leftTimeValue.value()
        self.diffValue.setMinimum(v - 1)
        self.diffValue.setMaximum(v + 1)
        self.diffValue.setValue(v)
        self.__magnitude_model.redraw()

    def __update_left_marker(self):
        self.leftTimeValue.setValue(self.__left_x)
        self.__left_marker.set_xdata([self.__left_x, self.__left_x])

    def __update_right_marker(self):
        self.rightTimeValue.setValue(self.__right_x)
        self.__right_marker.set_xdata([self.__right_x, self.__right_x])

    @staticmethod
    def __configure_time_axis(axes, x_scale):
        axes.set_xscale(x_scale)
        axes.set_xlabel('Time (ms)')

    def show_limits(self):
        ''' shows the limits dialog for the chart. '''
        self.__magnitude_model.show_limits()

    def update_chart(self):
        if self.chartToggle.isChecked():
            self.chartToggle.setText('SR')
        else:
            self.chartToggle.setText('IR')
        self.__redraw()

    def __get_data(self):
        return lambda *args, **kwargs: self.get_curve_data(*args, **kwargs)

    def get_curve_data(self, reference=None):
        ''' preview of the filter to display on the chart '''
        if self.__signals:
            data = []
            for signal, selected in self.__signals.items():
                if selected:
                    t, y = signal.step_response if self.chartToggle.isChecked() else signal.waveform
                    if t[0] == 0.0:
                        t = (t - (signal.duration_seconds / 2.0)) * 1000
                    data.append((signal.name, t, y))
            max_y = max(np.max(np.abs(d[2])) for d in data)
            return [MagnitudeData(d[0], None, d[1], d[2] * (100.0 / max_y), colour=get_filter_colour(i + 2))
                    for i, d in enumerate(data)]
        return []

    def select_channels(self):
        '''
        Allow user to select signals to examine.
        '''

        def on_save(selected: List[str]):
            self.__signals = {s: s.name in selected for s in self.__signals.keys()}
            self.update_chart()

        SignalSelectorDialog(self, [s.name for s in self.__signals.keys()],
                             [s.name for s, b in self.__signals.items() if b], on_save).show()

    def __connect_mouse(self):
        self.__cid.append(self.previewChart.canvas.mpl_connect('motion_notify_event', self.__record_coords))
        self.__cid.append(self.previewChart.canvas.mpl_connect('button_press_event', self.__depress))
        self.__cid.append(self.previewChart.canvas.mpl_connect('button_release_event', self.__release))
        self.__cid.append(self.previewChart.canvas.mpl_connect('axes_enter_event', self.__enter_axes))
        self.__cid.append(self.previewChart.canvas.mpl_connect('axes_leave_event', self.__leave_axes))

    def __depress(self, event):
        if not event.dblclick:
            if event.button == 1:
                self.__dragging = 'L'
            elif event.button == 3:
                self.__dragging = 'R'
            else:
                self.__dragging = None
        if self.__dragging and self.__in_active_area:
            self.__timer.start(50)

    def __release(self, event):
        self.__dragging = None
        self.__timer.stop()

    def __enter_axes(self, event):
        self.__in_active_area = event.inaxes is self.__magnitude_model.limits.axes_1

    def __leave_axes(self, event):
        if event.inaxes is self.__magnitude_model.limits.axes_1:
            self.__in_active_area = False
            self.__timer.stop()

    def __record_coords(self, event):
        if event is not None and self.__in_active_area and self.__dragging:
            if self.__dragging == 'L':
                self.__left_x = event.xdata
            elif self.__dragging == 'R':
                self.__right_x = event.xdata

    def zoom_in(self):
        self.__magnitude_model.limits.update(x_min=self.__left_x, x_max=self.__right_x, draw=True)

    def zoom_out(self):
        self.__magnitude_model.limits.update(x_min=self.__init_x_lim[0], x_max=self.__init_x_lim[1], draw=True)


class SignalSelectorDialog(QDialog, Ui_channelSelectDialog):

    def __init__(self, parent: QDialog, signals: List[str], selected_signals: List[str],
                 on_save: Callable[[List[str]], None]):
        super(SignalSelectorDialog, self).__init__(parent)
        self.setupUi(self)
        self.channelList.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.__on_save = on_save
        self.setWindowTitle('Select Signals')
        self.lfeChannel.hide()
        self.lfeChannelLabel.hide()
        for c in signals:
            self.channelList.addItem(c)
        for c in selected_signals:
            item: QListWidgetItem
            for item in self.channelList.findItems(c, QtCore.Qt.MatchFlag.MatchCaseSensitive):
                item.setSelected(True)

    def accept(self):
        self.__on_save([i.text() for i in self.channelList.selectedItems()])
        super().accept()
