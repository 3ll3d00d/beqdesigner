import logging
from typing import List, Callable, Dict, Optional, Union

import math
import qtawesome as qta
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QFont, QCloseEvent
from qtpy.QtWidgets import QDialog, QFrame, QGridLayout, QHBoxLayout, QToolButton, QButtonGroup, QLabel, QSlider, \
    QDoubleSpinBox, QSpacerItem, QSizePolicy, QWidget, QAbstractSpinBox, QListWidgetItem

from acoustics.standards.iec_61260_1_2014 import NOMINAL_OCTAVE_CENTER_FREQUENCIES
from model.iir import LowShelf, HighShelf, PeakingEQ, CompleteFilter, Passthrough, SOS
from model.limits import PhaseRangeCalculator, dBRangeCalculator
from model.magnitude import MagnitudeModel
from model.preferences import GEQ_GEOMETRY, GEQ_GRAPH_X_MIN, GEQ_GRAPH_X_MAX, get_filter_colour, Preferences
from model.xy import MagnitudeData
from ui.geq import Ui_geqDialog

GEQ = 'GEQ'

logger = logging.getLogger('geq')


class GeqDialog(QDialog, Ui_geqDialog):

    def __init__(self, parent, prefs: Preferences, channels: Dict[str, bool], existing_filters: List[SOS],
                 on_save: Callable[[List[str], List[SOS]], None], preset: str = GEQ, **kwargs):
        super(GeqDialog, self).__init__(parent)
        self.__on_save = on_save
        self.prefs = prefs
        self.setupUi(self)
        from model.report import block_signals
        with block_signals(self.presetSelector):
            self.presetSelector.addItem('')
            self.presetSelector.addItem(GEQ)
            self.presetSelector.addItem('BEQ')
        for c in channels.keys():
            self.channelList.addItem(c)
        self.presetSelector.currentTextChanged.connect(self.__load_preset)
        self.showPhase.setIcon(qta.icon('mdi.cosine-wave'))
        self.advancedMode.setIcon(qta.icon('mdi.toggle-switch'))
        self.showIndividual.setIcon(qta.icon('fa5s.chart-line'))
        self.__mag_update_timer = QTimer(self)
        self.__mag_update_timer.setSingleShot(True)
        self.__peq_editors: List[PeqEditor] = []
        self.scrollableLayout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        if preset:
            self.presetSelector.setCurrentText(preset)
        if existing_filters:
            self.__load_filters(existing_filters)
        else:
            self.update_peq_editors()
        self.__magnitude_model = MagnitudeModel('preview', self.previewChart, prefs,
                                                self.__get_data(), 'Filter', fill_primary=False,
                                                x_min_pref_key=GEQ_GRAPH_X_MIN, x_max_pref_key=GEQ_GRAPH_X_MAX,
                                                secondary_data_provider=self.__get_data('phase'),
                                                secondary_name='Phase', secondary_prefix='deg', fill_secondary=False,
                                                db_range_calc=dBRangeCalculator(60),
                                                y2_range_calc=PhaseRangeCalculator(), show_y2_in_legend=False, **kwargs)
        self.showPhase.toggled.connect(self.__trigger_redraw)
        self.showPhase.setToolTip('Display phase response')
        self.showIndividual.toggled.connect(self.__trigger_redraw)
        self.showIndividual.setToolTip('Display individual filter responses')
        self.advancedMode.toggled.connect(lambda b: [p.advanced(b) for p in self.__peq_editors])
        self.advancedMode.toggled.connect(lambda b: self.advancedMode.setIcon(qta.icon(f"mdi.toggle-switch{'' if b else '-off'}")))
        self.advancedMode.setToolTip('Show Q and Frequency Sliders')
        self.channelList.itemSelectionChanged.connect(self.__trigger_redraw)
        selected_channels = [c for c, b in channels.items() if b]
        for i in range(self.channelList.count()):
            item: QListWidgetItem = self.channelList.item(i)
            item.setSelected(item.text() in selected_channels)
        self.__mag_update_timer.timeout.connect(self.__magnitude_model.redraw)

    def __load_filters(self, to_load: List[SOS]):
        valid = [f for f in to_load if isinstance(f, (PeakingEQ, LowShelf, HighShelf))]
        self.peqCount.setValue(len(valid))
        self.update_peq_editors()
        for i, f in enumerate(valid):
            self.__peq_editors[i].load(f)

    def __load_preset(self, preset: str):
        if preset == 'GEQ':
            freqs = NOMINAL_OCTAVE_CENTER_FREQUENCIES.tolist()
            self.peqCount.setValue(len(freqs) + 2)
            self.__peq_editors[0].reset('LS', 40, 0.707)
            for i, freq in enumerate(freqs):
                self.__peq_editors[i+1].reset('PEQ', freq, 1.0)
            self.__peq_editors[-1].reset('HS', 8000, 0.707)
        elif preset == 'BEQ':
            self.peqCount.setValue(10)
            for i in range(10):
                self.__peq_editors[i].reset('LS', 20, 0.8)

    def __trigger_redraw(self):
        if not self.__mag_update_timer.isActive():
            self.__mag_update_timer.start(20)

    def __get_data(self, mode='mag'):
        return lambda *args, **kwargs: self.get_curve_data(mode, *args, **kwargs)

    def get_curve_data(self, mode, reference=None):
        ''' preview of the filter to display on the chart '''
        result = []
        final_filter = CompleteFilter(fs=48000, filters=self.__get_filters(), sort_by_id=True)
        if mode == 'mag' or self.showPhase.isChecked():
            extra = 0
            if len(final_filter) > 0:
                result.append(final_filter.get_transfer_function()
                                          .get_data(mode=mode, colour=get_filter_colour(len(result))))
            else:
                extra += 1
            for i, f in enumerate(final_filter):
                if self.showIndividual.isChecked():
                    colour = get_filter_colour(len(result) + extra)
                    data: MagnitudeData = f.get_transfer_function().get_data(mode=mode, colour=colour, linestyle=':')
                    data.override_name = f"PEQ {i}"
                    result.append(data)
        return result

    def __get_filters(self, include_zero=False) -> List[SOS]:
        filters = [e.make_filter(include_zero) for i, e in enumerate(self.__peq_editors) if i < self.peqCount.value()]
        return [f for f in filters if f]

    def show_limits(self):
        ''' shows the limits dialog for the filter chart. '''
        self.__magnitude_model.show_limits()

    def update_peq_editors(self):
        for i in range(self.peqCount.value()):
            if i >= len(self.__peq_editors):
                self.__create_peq_editor(i)
            else:
                self.__peq_editors[i].show()
        if self.peqCount.value() < len(self.__peq_editors):
            for i in range(self.peqCount.value(), len(self.__peq_editors)):
                self.__peq_editors[i].hide()

    def __create_peq_editor(self, i: int):
        editor = PeqEditor(self.scrollable, i, self.__trigger_redraw)
        self.__peq_editors.append(editor)
        self.scrollableLayout.insertWidget(i, editor.widget)

    def __restore_geometry(self):
        ''' loads the saved window size '''
        geometry = self.prefs.get(GEQ_GEOMETRY)
        if geometry is not None:
            self.restoreGeometry(geometry)

    def accept(self):
        self.__on_save([c.text() for c in self.channelList.selectedItems()], self.__get_filters(include_zero=True))
        super().accept()

    def closeEvent(self, event: QCloseEvent):
        ''' Stores the window size on close. '''
        self.prefs.set(GEQ_GEOMETRY, self.saveGeometry())
        super().closeEvent(event)


class PeqEditor:

    def __init__(self, parent: QWidget, idx: int, on_change: Callable[[], None]):
        font = QFont()
        font.setPointSize(10)
        self.__idx = idx
        self.__geq_frame = QFrame(parent)
        self.__geq_frame.setFrameShape(QFrame.StyledPanel)
        self.__geq_frame.setFrameShadow(QFrame.Raised)
        self.__grid_layout = QGridLayout(self.__geq_frame)
        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setItalic(True)
        title_font.setBold(True)
        self.__title = QLabel(self.__geq_frame)
        self.__title.setFont(title_font)
        self.__title.setText(f"PEQ {idx + 1}")
        self.__title.setAlignment(Qt.AlignCenter)
        self.__filter_selector_layout = QHBoxLayout()
        self.__peq_button = QToolButton(self.__geq_frame)
        self.__peq_button.setCheckable(True)
        self.__peq_button.setChecked(True)
        self.__peq_button.setText("P")
        self.__button_group = QButtonGroup(self.__geq_frame)
        self.__button_group.addButton(self.__peq_button)
        self.__filter_selector_layout.addWidget(self.__peq_button)
        self.__ls_button = QToolButton(self.__geq_frame)
        self.__ls_button.setCheckable(True)
        self.__ls_button.setText("LS")
        self.__button_group.addButton(self.__ls_button)
        self.__filter_selector_layout.addWidget(self.__ls_button)
        self.__hs_button = QToolButton(self.__geq_frame)
        self.__hs_button.setCheckable(True)
        self.__hs_button.setText("HS")
        self.__button_group.addButton(self.__hs_button)
        self.__filter_selector_layout.addWidget(self.__hs_button)
        self.__grid_layout.addWidget(self.__title, 0, 0, 1, 4)
        self.__grid_layout.addLayout(self.__filter_selector_layout, 1, 0, 1, 4)

        self.__gain_label = QLabel(self.__geq_frame)
        self.__gain_label.setFont(font)
        self.__gain_label.setText("Gain (dB)")
        self.__gain_slider = QSlider(self.__geq_frame)
        self.__gain_slider.setMinimum(-3000)
        self.__gain_slider.setMaximum(3000)
        self.__gain_slider.setOrientation(Qt.Vertical)
        self.__gain_slider.setTickPosition(QSlider.TicksBelow)
        self.__gain_slider.setTickInterval(300)
        self.__gain_slider.setToolTip('Gain (dB)')
        self.__gain = QDoubleSpinBox(self.__geq_frame)
        self.__gain.setFont(font)
        self.__gain.setMinimum(-30)
        self.__gain.setMaximum(30)
        self.__gain.setSingleStep(0.1)
        self.__gain.setDecimals(2)

        self.__freq_slider = QSlider(self.__geq_frame)
        self.__freq_slider.setOrientation(Qt.Vertical)
        self.__freq_slider.setTickPosition(QSlider.TicksBelow)
        self.__freq_slider.setMinimum(1)
        self.__freq_slider.setMaximum(1500)
        self.__freq_slider.setTickInterval(100)
        self.__freq_slider.setToolTip('Frequency (Hz)')
        self.__freq_label = QLabel(self.__geq_frame)
        self.__freq_label.setFont(font)
        self.__freq_label.setText("Freq (Hz)")
        self.__freq = QDoubleSpinBox(self.__geq_frame)
        self.__freq.setFont(font)
        self.__freq.setMinimum(1)
        self.__freq.setMaximum(24000)
        self.__freq.setDecimals(1)
        self.__freq.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)

        self.__q_slider = QSlider(self.__geq_frame)
        self.__q_slider.setOrientation(Qt.Vertical)
        self.__q_slider.setTickPosition(QSlider.TicksBelow)
        self.__q_slider.setMinimum(1)
        self.__q_slider.setMaximum(20000000)
        self.__q_slider.setTickInterval(1000000)
        self.__q_slider.setToolTip('Q')
        self.__q_label = QLabel(self.__geq_frame)
        self.__q_label.setFont(font)
        self.__q_label.setText("Q")
        self.__q = QDoubleSpinBox(self.__geq_frame)
        self.__q.setFont(font)
        self.__q.setMinimum(0.001)
        self.__q.setMaximum(20)
        self.__q.setDecimals(3)
        self.__q.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)

        def to_q_slider(v: float) -> float:
            return (math.log((v + 0.15) / 0.15) / math.log(1.28)) * 1000000

        def from_q_slider(v: float) -> float:
            return (0.15 * (1.28 ** (v / 1000000))) - 0.15

        def to_freq_slider(v: float) -> float:
            return (math.log(v) / math.log(2)) * 100

        def from_freq_slider(v: float) -> float:
            return 2**(v/100)

        self.__freq_slider.valueChanged.connect(lambda v: self.__freq.setValue(from_freq_slider(v)))
        self.__q_slider.valueChanged.connect(lambda v: self.__q.setValue(from_q_slider(v)))
        self.__gain_slider.valueChanged.connect(lambda v: self.__gain.setValue(v/100))
        self.__gain.valueChanged.connect(on_change)
        self.__q.valueChanged.connect(on_change)
        self.__freq.valueChanged.connect(on_change)
        self.__gain.valueChanged.connect(lambda v: self.__update_slider(self.__gain_slider, v, lambda v: v*100))
        self.__q.valueChanged.connect(lambda v: self.__update_slider(self.__q_slider, v, to_q_slider))
        self.__freq.valueChanged.connect(lambda v: self.__update_slider(self.__freq_slider, v, to_freq_slider))
        self.__ls_button.toggled.connect(on_change)
        self.__hs_button.toggled.connect(on_change)
        self.__peq_button.toggled.connect(on_change)

        self.__freq_slider.setValue(1000)
        self.__q_slider.setValue(to_q_slider(0.707))

        self.__grid_layout.addWidget(self.__gain_label, 2, 0, 1, 1)
        self.__grid_layout.addWidget(self.__gain_slider, 2, 1, 7, 1)
        self.__grid_layout.addWidget(self.__freq_slider, 2, 2, 7, 1)
        self.__grid_layout.addWidget(self.__q_slider, 2, 3, 7, 1)
        self.__grid_layout.addWidget(self.__gain, 3, 0, 1, 1)
        self.__grid_layout.addWidget(self.__freq_label, 4, 0, 1, 1)
        self.__grid_layout.addWidget(self.__freq, 5, 0, 1, 1)
        self.__grid_layout.addWidget(self.__q_label, 6, 0, 1, 1)
        self.__grid_layout.addWidget(self.__q, 7, 0, 1, 1)
        self.__grid_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding), 8, 0, 1, 1)

    @staticmethod
    def __update_slider(slider: QSlider, v: float, translate: Callable[[float], float] = lambda x: x) -> None:
        from model.report import block_signals
        with block_signals(slider):
            slider.setValue(translate(v))

    def make_filter(self, include_zero) -> Optional[Union[LowShelf, HighShelf, PeakingEQ, Passthrough]]:
        if math.isclose(self.__gain.value(), 0.0) and not include_zero:
            return None
        if self.__ls_button.isChecked():
            return LowShelf(48000, self.__freq.value(), self.__q.value(), self.__gain.value(), f_id=self.__idx)
        elif self.__hs_button.isChecked():
            return HighShelf(48000, self.__freq.value(), self.__q.value(), self.__gain.value(), f_id=self.__idx)
        elif self.__peq_button.isChecked():
            return PeakingEQ(48000, self.__freq.value(), self.__q.value(), self.__gain.value(), f_id=self.__idx)
        else:
            return Passthrough(fs=48000)

    def show(self) -> None:
        self.__geq_frame.show()

    def hide(self) -> None:
        self.__geq_frame.hide()

    @property
    def widget(self) -> QWidget:
        return self.__geq_frame

    def reset(self, filter_type: str, freq: float, q: float, gain: float = 0.0) -> None:
        if filter_type == 'LS':
            self.__ls_button.setChecked(True)
        elif filter_type == 'HS':
            self.__hs_button.setChecked(True)
        else:
            self.__peq_button.setChecked(True)
        self.__freq.setValue(freq)
        self.__q.setValue(q)
        self.__gain.setValue(gain)

    def advanced(self, on: bool) -> None:
        self.__q_slider.setVisible(on)
        self.__freq_slider.setVisible(on)

    def load(self, f: Union[LowShelf, HighShelf, PeakingEQ]) -> None:
        self.reset(f.filter_type, f.freq, f.q, gain= f.gain)
