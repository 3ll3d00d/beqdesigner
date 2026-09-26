'''
The Qt side of `model.signal`: the signal table models, the add/merge/select signal dialogs and their loaders, and smoothing
on the thread pool. `model.signal` itself (signal data, `Signal`, wav loading) is Qt-free, so the headless pipeline loads audio
without Qt.
'''
import logging
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import List, Dict, Iterable, Any, Callable

import numpy as np
import qtawesome as qta
from qtpy import QtCore
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QVariant, Qt, QRunnable, QThreadPool
from qtpy.QtWidgets import QDialog, QFileDialog, QDialogButtonBox, QStatusBar

from model.codec import signaldata_to_json, bassmanagedsignaldata_to_json
from model.iir import CompleteFilter
from model.magnitude import MagnitudeModel
from model.preferences import get_avg_colour, get_peak_colour, DISPLAY_SHOW_SIGNALS, DISPLAY_SHOW_FILTERED_SIGNALS, \
    ANALYSIS_TARGET_FS, DISPLAY_SMOOTH_PRECALC, EXTRACTION_OUTPUT_DIR
from model.signal import AutoWavLoader, BassManagedSignalData, Signal, SingleChannelSignalData, \
    get_visible_signal_name_filter
from model.xy import MagnitudeData
from ui.merge_signals import Ui_MergeSignalDialog
from ui.signal import Ui_addSignalDialog
from ui.signal_viz import Ui_selectSignalsDialog

logger = logging.getLogger('signal')


class SignalModel(Sequence):
    '''
    A model to hold onto the signals.
    '''

    def __init__(self, view, default_signal, preferences, on_update=lambda _: True):
        self.__signals = []
        self.__visible_names = set()
        self.__bass_managed_signals = []
        self.default_signal = default_signal
        self.__view = view
        self.__on_update = on_update
        self.__preferences = preferences
        self.__table = None

    @property
    def visible_names(self) -> List[str]:
        return sorted([s for s in self.__visible_names])

    @visible_names.setter
    def visible_names(self, names: List[str]):
        self.__visible_names = set(names)

    @property
    def table(self):
        return self.__table

    @property
    def bass_managed_signals(self) -> Iterable[BassManagedSignalData]:
        return self.__bass_managed_signals

    @property
    def non_bm_signals(self):
        return [s for s in self.__signals if not self.__is_bass_managed(s)]

    @table.setter
    def table(self, table):
        self.__table = table

    def __getitem__(self, i):
        return self.__signals[i]

    def __len__(self):
        return len(self.__signals)

    def to_json(self):
        '''
        :return: a json compatible format of the data in the model.
        '''
        json = [bassmanagedsignaldata_to_json(x) for x in self.__bass_managed_signals]
        json += [signaldata_to_json(x) for x in self.__signals if not self.__is_bass_managed(x)]
        return json

    def __is_bass_managed(self, signal):
        '''
        :param signal: the signal.
        :return: true if this signal is found in a bass managed signal.
        '''
        is_bm = False
        for bm in self.__bass_managed_signals:
            for c in bm.channels:
                if c is signal:
                    is_bm = True
                    break
        return is_bm

    def free_all(self):
        '''
        Frees all signals from their masters.
        '''
        if self.__table is not None:
            self.__table.beginResetModel()
        for signal in self:
            signal.free_all()
        if self.__table is not None:
            self.__table.endResetModel()

    def enslave(self, master_name, slave_names):
        '''
        Enslaves the named slaves to the named master.
        :param master_name: the master.
        :param slave_names: the slaves.
        '''
        logger.info(f"Enslaving {slave_names} to {master_name}")
        if self.__table is not None:
            self.__table.beginResetModel()
        master = self.find_by_name(master_name)
        if master is not None:
            for slave_name in slave_names:
                slave = self.find_by_name(slave_name)
                if slave is not None:
                    master.enslave(slave)
        if self.__table is not None:
            self.__table.endResetModel()

    def add(self, signal):
        '''
        Add the supplied signals ot the model.
        :param signals: the signal.
        '''
        if isinstance(signal, BassManagedSignalData):
            # add the bass managed signal first because the selector refresh is driven by the signal model change
            self.__bass_managed_signals.append(signal)
            self.add_all(signal.channels)
        else:
            before_size = len(self.__signals)
            if self.__table is not None:
                self.__table.beginInsertRows(QModelIndex(), before_size, before_size)
            signal.reindex(before_size)
            self.__signals.append(signal)
            self.__visible_names.add(signal.name)
            self.post_update()
            if self.__table is not None:
                self.__table.endInsertRows()

    def add_all(self, signals):
        '''
        Add the supplied signals ot the model.
        :param signals: the signal.
        '''
        before_size = len(self.__signals)
        if self.__table is not None:
            self.__table.beginInsertRows(QModelIndex(), before_size, before_size + (len(signals) - 1))
        for s in signals:
            s.reindex(len(self.__signals))
            self.__signals.append(s)
            self.__visible_names.add(s.name)
        self.post_update()
        if self.__table is not None:
            self.__table.endInsertRows()

    def post_update(self):
        self.__on_update(self.get_visible_curve_names())

    def get_visible_curve_names(self) -> List[str]:
        from app import flatten
        show_signals = self.__preferences.get(DISPLAY_SHOW_SIGNALS)
        show_filtered_signals = self.__preferences.get(DISPLAY_SHOW_FILTERED_SIGNALS)
        pattern = get_visible_signal_name_filter(show_filtered_signals, show_signals)
        visible_signal_names = [x.name for x in flatten([y for x in self.__signals for y in x.get_all_xy()])]
        if pattern is not None:
            visible_signal_names = [x for x in visible_signal_names if pattern.match(x) is not None]
        return visible_signal_names

    def remove(self, signal):
        '''
        Remove the specified signal from the model.
        :param signal: the signal to remove.
        '''
        idx = self.__signals.index(signal)
        if self.__table is not None:
            self.__table.beginRemoveRows(QModelIndex(), idx, idx)
        self.__visible_names.remove(self.__signals[idx].name)
        del self.__signals[idx]
        for idx, s in enumerate(self.__signals):
            s.reindex(idx)
        self.__ensure_master_slave_integrity()
        self.post_update()
        if self.__table is not None:
            self.__table.endRemoveRows()

    def delete(self, indices):
        '''
        Delete the signals at the given indices.
        :param indices: the indices to remove.
        '''
        self.replace([s for idx, s in enumerate(self.__signals) if idx not in indices])

    def get_all_magnitude_data(self, visible_filter: bool = False):
        '''
        :return: the raw xy data.
        '''
        from app import flatten
        results = list(
            flatten([s.get_all_xy() for s in self.__signals if not visible_filter or s.name in self.__visible_names]))
        return results

    def get_curve_data(self, reference=None):
        '''
        :param reference: the curve against which to normalise.
        :return: the peak,  avg and median spectrum for the signals (if any) + the filter signals.
        '''
        results = self.get_all_magnitude_data(visible_filter=True)
        show_signals = self.__preferences.get(DISPLAY_SHOW_SIGNALS)
        show_filtered_signals = self.__preferences.get(DISPLAY_SHOW_FILTERED_SIGNALS)
        pattern = get_visible_signal_name_filter(show_filtered_signals, show_signals)
        if pattern is not None:
            results = [x for x in results if pattern.match(x.name) is not None]
        if reference is not None:
            ref_data = next((x for x in results if x.name == reference), None)
            if ref_data:
                results = [x.normalise(ref_data) for x in results]
        return results

    def replace(self, signals):
        '''
        Replaces the contents of the model with the supplied signals
        :param signals: the signals
        '''
        if self.__table is not None:
            self.__table.beginResetModel()
        self.__bass_managed_signals = [s for s in signals if isinstance(s, BassManagedSignalData)]
        sigs = [s for s in signals if isinstance(s, SingleChannelSignalData)]
        for b in self.__bass_managed_signals:
            sigs += b.channels
        self.__signals = sigs
        self.visible_names = [s.name for s in sigs]
        for idx, s in enumerate(self.__signals):
            if self.__preferences.get(DISPLAY_SMOOTH_PRECALC):
                QThreadPool.globalInstance().start(Smoother(s))
            s.reindex(idx)
        self.__ensure_master_slave_integrity()
        self.__discard_incomplete_bass_managed_signals()
        self.post_update()
        if self.__table is not None:
            self.__table.endResetModel()

    def __discard_incomplete_bass_managed_signals(self):
        ''' discards any bass managed signals that are missing child signals '''
        still_here = []
        for bm in self.__bass_managed_signals:
            if all(c in self.__signals for c in bm.channels):
                still_here.append(bm)
        self.__bass_managed_signals = still_here

    def __ensure_master_slave_integrity(self):
        '''
        Verifies that all master/slaves mentioned by signals actually exist in the model. Used when signals are deleted.
        '''
        for s in self.__signals:
            slave_count_before = len(s.slaves)
            if slave_count_before > 0:
                s.slaves = [slave for slave in s.slaves if slave in self.__signals]
                delta = slave_count_before - len(s.slaves)
                if delta > 0:
                    logger.info(f"Removed {delta} missing slaves from {s.name}")
            if s.master is not None:
                master = next((m for m in self.__signals if m.name == s.master.name), None)
                if master is None:
                    logger.info(f"Removing missing master {s.master.name} from {s.name}")
                    s.free()

    def find_by_name(self, name):
        '''
        :param name: the signal name.
        :return: the signal with that name (or None).
        '''
        return next((s for s in self if s.name == name), None)

    def tilt(self, tilt):
        '''
        Applies or removes the 3dB equal energy tilt.
        :param tilt: true or false.
        '''
        for s in self.__signals:
            s.tilt(tilt)


class SignalTableModel(QAbstractTableModel):
    '''
    A Qt table model to feed the signal view.
    '''

    def __init__(self, model, parent=None):
        super().__init__(parent=parent)
        self.__headers = ['Name', 'Linked', 'Fs', 'Duration', 'Offset']
        self.__signal_model = model
        self.__signal_model.table = self

    def rowCount(self, parent: QModelIndex = ...) -> int:
        return len(self.__signal_model)

    def columnCount(self, parent: QModelIndex = ...) -> int:
        return len(self.__headers)

    def flags(self, idx):
        flags = super().flags(idx)
        if idx.column() == 0:
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(self, idx, value, role=None):
        if idx.column() == 0:
            self.__signal_model[idx.row()].name = value
            self.dataChanged.emit(idx, idx, [])
            return True
        return super().setData(idx, value, role=role)

    def data(self, index: QModelIndex, role: int = ...) -> Any:
        if not index.isValid():
            return QVariant()
        elif role != Qt.ItemDataRole.DisplayRole:
            return QVariant()
        else:
            signal_at_row = self.__signal_model[index.row()]
            if index.column() == 0:
                return QVariant(signal_at_row.name)
            if index.column() == 1:
                if signal_at_row.master is not None:
                    return QVariant(f"S - {signal_at_row.master.name}")
                elif len(signal_at_row.slaves) > 0:
                    return QVariant(f"M {len(signal_at_row.slaves)}")
                else:
                    return QVariant('')
            elif index.column() == 2:
                return QVariant(signal_at_row.fs)
            elif index.column() == 3:
                return QVariant(signal_at_row.duration_hhmmss)
            elif index.column() == 4:
                return QVariant(signal_at_row.offset)
            else:
                return QVariant()

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return QVariant(self.__headers[section])
        return QVariant()


def select_file(owner, file_types, dir=None):
    '''
    Presents a file picker for selecting a file that contains a signal.
    '''
    dialog = QFileDialog(parent=owner)
    dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
    filt = ' '.join([f"*.{f}" for f in file_types])
    dialog.setNameFilter(f"Audio ({filt})")
    dialog.setWindowTitle(f"Select Signal File")
    if dir:
        dialog.setDirectory(dir)
    if dialog.exec():
        selected = dialog.selectedFiles()
        if len(selected) > 0:
            return selected[0]
    return None


class DialogWavLoaderBridge:
    '''
    Loads signals from wav files.
    '''

    def __init__(self, dialog, preferences, allow_multichannel=True):
        self.__preferences = preferences
        self.__dialog = dialog
        self.__auto_loader = AutoWavLoader(preferences)
        self.__duration = 0
        self.__allow_multichannel = allow_multichannel

    def toggle_decimate(self, channel_idx):
        self.__auto_loader.clear_cache()
        self.prepare_signal(channel_idx)

    def select_wav_file(self):
        out_dir = self.__preferences.get(EXTRACTION_OUTPUT_DIR)
        file = select_file(self.__dialog, ['wav', 'flac'], dir=out_dir)
        if file is not None:
            self.clear_signal()
            self.__dialog.wavFile.setText(file)
            self.init_time_range()
            self.__auto_loader.load(file)
            self.__load_info()

    def clear_signal(self):
        self.__auto_loader.reset()
        self.__duration = 0
        self.__dialog.wavStartTime.setEnabled(False)
        self.__dialog.wavEndTime.setEnabled(False)
        self.__dialog.gainOffset.setEnabled(False)
        self.__dialog.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    def __load_info(self):
        '''
        Loads metadata about the signal from the file and propagates it to the form fields.
        '''
        info = self.__auto_loader.info
        self.__dialog.wavFs.setText(f"{info.samplerate} Hz")
        self.__dialog.decimate.setEnabled(info.samplerate != self.__preferences.get(ANALYSIS_TARGET_FS))
        from model.report import block_signals
        with block_signals(self.__dialog.wavChannelSelector):
            self.__dialog.wavChannelSelector.clear()
            for i in range(0, info.channels):
                self.__dialog.wavChannelSelector.addItem(f"{i + 1}")
            self.__dialog.wavChannelSelector.setEnabled(info.channels > 1)
        self.__dialog.loadAllChannels.setEnabled(info.channels > 1 and self.__allow_multichannel)
        with block_signals(self.__dialog.wavStartTime):
            self.__dialog.wavStartTime.setTime(QtCore.QTime(0, 0, 0))
            self.__dialog.wavStartTime.setEnabled(True)
        self.__duration = math.floor(info.duration * 1000)
        with block_signals(self.__dialog.wavEndTime):
            self.__dialog.wavEndTime.setTime(QtCore.QTime(0, 0, 0).addMSecs(self.__duration))
            self.__dialog.wavEndTime.setEnabled(True)
        self.__dialog.wavSignalName.setEnabled(True)
        from pathlib import Path
        self.__dialog.wavSignalName.setText(str(Path(Path(info.name).name).stem))
        self.prepare_signal(int(self.__dialog.wavChannelSelector.currentText()))
        self.__dialog.applyTimeRangeButton.setEnabled(False)

    def init_time_range(self):
        ''' Initialises the time range on the auto loader. '''
        start = end = None
        start_millis = self.__dialog.wavStartTime.time().msecsSinceStartOfDay()
        if start_millis > 0:
            start = start_millis
        end_millis = self.__dialog.wavEndTime.time().msecsSinceStartOfDay()
        if end_millis < self.__duration or start is not None:
            end = end_millis
        self.__auto_loader.set_range(start=start, end=end)

    def prepare_signal(self, channel_idx):
        '''
        Reads the actual file and calculates the relevant peak/avg spectrum.
        '''
        self.__auto_loader.prepare(name=self.__dialog.wavSignalName.text(),
                                   channel=channel_idx,
                                   decimate=self.__dialog.decimate.isChecked())
        self.__dialog.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        self.__dialog.gainOffset.setEnabled(True)

    def __get_window(self, key):
        from model.preferences import ANALYSIS_WINDOW_DEFAULT
        window = self.__preferences.get(key)
        if window is None or window == ANALYSIS_WINDOW_DEFAULT:
            window = None
        else:
            if window == 'tukey':
                window = (window, 0.25)
        return window

    def get_magnitude_data(self):
        if self.__dialog.wavChannelSelector.count() > 0:
            return self.__auto_loader.get_magnitude_data(int(self.__dialog.wavChannelSelector.currentText()))
        else:
            return []

    def can_save(self):
        '''
        :return: true if we can save a new signal.
        '''
        return self.__auto_loader.has_signal()

    def enable_ok(self):
        enabled = len(
            self.__dialog.wavSignalName.text()) > 0 and not self.__dialog.applyTimeRangeButton.isEnabled() and self.can_save()
        self.__dialog.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(enabled)
        return enabled

    def get_signal(self, offset=0.0):
        '''
        Converts the loaded signal into a SignalData.
        :return: the signal data.
        '''
        if self.__dialog.loadAllChannels.isChecked() and self.__dialog.loadAllChannels.isEnabled():
            from model.extract import get_channel_name
            name_provider = lambda channel, channel_count: get_channel_name(self.__dialog.wavSignalName.text(), channel,
                                                                            channel_count)
            return self.__auto_loader.auto_load(name_provider,
                                                self.__dialog.decimate.isChecked(),
                                                offset=offset)
        else:
            return self.__auto_loader.get_signal(int(self.__dialog.wavChannelSelector.currentText()),
                                                 self.__dialog.wavSignalName.text(),
                                                 offset=offset)


class FrdLoader:
    '''
    Loads signals from frd files.
    '''

    def __init__(self, dialog):
        self.__dialog = dialog
        self.__peak = None
        self.__avg = None

    def _read_from_file(self):
        file = select_file(self.__dialog, ['frd'])
        if file is not None:
            comment_char = None
            with open(file) as f:
                c = f.read(1)
                if not c.isalnum():
                    comment_char = c
            f, m = np.genfromtxt(file, comments=comment_char, unpack=True)
            return file, f, m
        return None, None, None

    def select_peak_file(self):
        '''
        Asks the user to pick a file containing the peak series magnitude response.
        '''
        name, f, m = self._read_from_file()
        if name is not None:
            signal_name = Path(name).resolve().stem
            if signal_name.endswith('_filter_peak'):
                signal_name = signal_name[:-12]
            elif signal_name.endswith('_peak'):
                signal_name = signal_name[:-5]
            self.__peak = MagnitudeData(signal_name, 'peak', f, m, colour=get_peak_colour(0))
            self.__dialog.frdSignalName.setText(signal_name)
            self.__dialog.frdPeakFile.setText(name)
            self.__enable_fields()

    def select_avg_file(self):
        '''
        Asks the user to pick a file containing the avg series magnitude response.
        '''
        name, f, m = self._read_from_file()
        if name is not None:
            signal_name = Path(name).resolve().stem
            if signal_name.endswith('_filter_avg'):
                signal_name = signal_name[:-11]
            elif signal_name.endswith('_avg'):
                signal_name = signal_name[:-4]
            self.__avg = MagnitudeData(signal_name, 'avg', f, m, colour=get_avg_colour(0))
            self.__dialog.frdSignalName.setText(signal_name)
            self.__dialog.frdAvgFile.setText(name)
            self.__enable_fields()

    def __enable_fields(self):
        '''
        Enables the fs field if we have both measurements.
        '''
        self.__dialog.frdSignalName.setEnabled(self.__peak is not None or self.__avg is not None)
        if self.__peak is not None and self.__avg is not None:
            # TODO read the header?
            self.__dialog.frdFs.setValue(int(np.max(self.__peak.x) * 2))
            self.__dialog.frdFs.setEnabled(True)
        self.enable_ok()

    def enable_ok(self):
        enabled = len(self.__dialog.frdSignalName.text()) > 0 and self.can_save()
        self.__dialog.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(enabled)
        return enabled

    def clear_signal(self):
        self.__peak = None
        self.__avg = None
        self.__enable_fields()

    def get_magnitude_data(self):
        data = []
        if self.__avg is not None:
            data.append(self.__avg)
        if self.__peak is not None:
            data.append(self.__peak)
        return data

    def can_save(self):
        return self.__avg is not None and self.__peak is not None

    def get_signal(self, **kwargs):
        frd_name = self.__dialog.frdSignalName.text()
        self.__avg.internal_name = frd_name
        self.__peak.internal_name = frd_name
        return SingleChannelSignalData(name=frd_name,
                                       fs=self.__dialog.frdFs.value(),
                                       filter=CompleteFilter(fs=self.__dialog.frdFs.value()),
                                       xy_data=self.get_magnitude_data())


class PulseLoader:
    '''
    Generates pulse signals.
    '''

    def __init__(self, dialog, prefs):
        self.__dialog = dialog
        self.__prefs = prefs
        from scipy.signal import unit_impulse
        signal = Signal('signal', unit_impulse(4 * 48000, 'mid'), prefs, fs=48000)
        self.__pulse = SingleChannelSignalData(name=f"signal", signal=signal)

    def can_save(self):
        '''
        :return: true if we can save at least one signal.
        '''
        return len(self.__dialog.pulsePrefix.text()) > 0

    def enable_ok(self):
        enabled = self.can_save()
        self.__dialog.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(enabled)
        return enabled

    def get_magnitude_data(self):
        return self.__pulse.current_unfiltered

    def get_signal(self, **kwargs):
        from scipy.signal import unit_impulse
        fs = int(self.__dialog.pulseFs.currentText()[0:2]) * 1000
        samples = unit_impulse(4 * fs, 'mid')
        signal = Signal(self.__dialog.pulsePrefix.text(), samples, self.__prefs, fs=fs)
        return SingleChannelSignalData(name=signal.name, signal=signal)


class SignalDialog(QDialog, Ui_addSignalDialog):
    '''
    Alows user to extract a signal from a wav or frd.
    '''

    def __init__(self, preferences, signal_model, allow_multichannel=True, parent=None):
        super(SignalDialog, self).__init__(parent=parent)
        self.setupUi(self)
        self.statusBar = QStatusBar()
        self.verticalLayout.addWidget(self.statusBar)
        self.wavFilePicker.setIcon(qta.icon('fa5s.folder-open'))
        self.frdAvgFilePicker.setIcon(qta.icon('fa5s.folder-open'))
        self.frdPeakFilePicker.setIcon(qta.icon('fa5s.folder-open'))
        self.applyTimeRangeButton.setIcon(qta.icon('fa5s.cut'))
        self.applyTimeRangeButton.setEnabled(False)
        self.__preferences = preferences
        pulse_loader = PulseLoader(self, preferences)
        self.__loaders = [
            DialogWavLoaderBridge(self, preferences, allow_multichannel=allow_multichannel),
            FrdLoader(self),
            pulse_loader
        ]
        self.__loader_idx = self.signalTypeTabs.currentIndex()
        self.__magnitudeModel = MagnitudeModel('preview', self.previewChart, preferences, self.get_curve_data, 'Signal')
        self.__signal_model = signal_model
        if self.__signal_model is None:
            self.filterSelectLabel.setEnabled(False)
            self.filterSelect.setEnabled(False)
            self.linkedSignal.setEnabled(False)
        elif len(self.__signal_model) == 0:
            if len(self.__signal_model.default_signal.filter) > 0:
                self.filterSelect.addItem('Default')
            else:
                self.filterSelectLabel.setEnabled(False)
                self.filterSelect.setEnabled(False)
                self.linkedSignal.setEnabled(False)
        else:
            for s in self.__signal_model:
                if s.master is None:
                    self.filterSelect.addItem(s.name)
        self.clear_signal(draw=False)

    def changeLoader(self, idx):
        self.__loader_idx = idx
        self.__loaders[self.__loader_idx].enable_ok()
        self.__magnitudeModel.redraw()

    def selectFile(self):
        '''
        Presents a file picker for selecting a wav file that contains a signal.
        '''
        self.__loaders[self.__loader_idx].select_wav_file()
        self.enableOk()
        self.__magnitudeModel.redraw()

    def selectPeakFile(self):
        '''
        Presents a file picker for selecting a frd file that contains the peak signal.
        '''
        self.__loaders[self.__loader_idx].select_peak_file()
        self.__magnitudeModel.redraw()

    def selectAvgFile(self):
        '''
        Presents a file picker for selecting a frd file that contains the avg signal.
        '''
        self.__loaders[self.__loader_idx].select_avg_file()
        self.__magnitudeModel.redraw()

    def clear_signal(self, draw=True):
        ''' clears the current signal '''
        self.__loaders[self.__loader_idx].clear_signal()
        if draw:
            self.__magnitudeModel.redraw()

    def reject(self):
        ''' ensure signals are released from memory after we close the dialog '''
        self.__clear_down()
        super().reject()

    def __clear_down(self):
        for l in self.__loaders:
            try:
                l.clear_signal()
            except:
                pass

    def enableLimitTimeRangeButton(self):
        ''' enables the button whenever the time range changes. '''
        self.applyTimeRangeButton.setEnabled(True)
        self.statusBar.showMessage('Click the scissors button to change the slice of the source file to analyse', 8000)
        self.enableOk()

    def limitTimeRange(self):
        ''' changes the applied time range. '''
        self.previewChannel(self.wavChannelSelector.currentText())
        self.applyTimeRangeButton.setEnabled(False)
        self.statusBar.clearMessage()
        self.enableOk()

    def previewChannel(self, channel_idx):
        '''
        Selects the specified channel.
        :param channel_idx: the channel to display.
        '''
        from app import wait_cursor
        with wait_cursor('Preparing Signal'):
            self.__loaders[self.__loader_idx].init_time_range()
            self.__loaders[self.__loader_idx].prepare_signal(int(channel_idx))
            self.__magnitudeModel.redraw()

    def enableOk(self):
        '''
        Enables the ok button if we can save.
        '''
        self.__loaders[self.__loader_idx].enable_ok()

    def get_curve_data(self, reference=None):
        '''
        :param reference: ignored as we don't expose a normalisation control in this chart.
        :return: the peak and avg spectrum for the currently loaded signal (if any).
        '''
        return self.__loaders[self.__loader_idx].get_magnitude_data()

    def masterFilterChanged(self, idx):
        '''
        enables the linked signal checkbox if we have selected a filter.
        :param idx: the selected index.
        '''
        self.linkedSignal.setEnabled(idx > 0)

    def accept(self):
        '''
        Adds the signal to the model and exits if we have a signal (which we should because the button is disabled
        until we do).
        '''
        loader = self.__loaders[self.__loader_idx]
        if loader.can_save():
            from app import wait_cursor
            with wait_cursor(f"Saving signals"):
                signal = loader.get_signal(offset=self.gainOffset.value())
                if signal is not None:
                    self.save(signal)
                    QDialog.accept(self)
                else:
                    logger.warning(f"No signals produced by loader")
            self.__clear_down()

    def save(self, signal):
        ''' saves the specified signals in the signal model'''
        if self.__preferences.get(DISPLAY_SMOOTH_PRECALC):
            QThreadPool.globalInstance().start(Smoother(signal))
        selected_filter_idx = self.filterSelect.currentIndex()
        if selected_filter_idx > 0:  # 0 because the dropdown has a None value first
            if self.filterSelect.currentText() == 'Default':
                self.__apply_default_filter(signal)
            else:
                master = self.__signal_model[selected_filter_idx - 1]
                if isinstance(signal, BassManagedSignalData):
                    for s in signal.channels:
                        self.__copy_filter(master, s)
                else:
                    self.__copy_filter(master, signal)
        self.__signal_model.add(signal)

    def __copy_filter(self, master, signal):
        if self.linkedSignal.isChecked():
            master.enslave(signal)
        else:
            signal.filter = master.filter.resample(signal.fs)

    def __apply_default_filter(self, signal):
        '''
        Copies forward the default filter, using the 1st generated signal as the master if the user has chosen to link
        them.
        :param signal: the signal.
        '''
        if self.linkedSignal.isChecked():
            master = None
            for idx, s in enumerate(signal):
                if idx == 0:
                    s.filter = self.__signal_model.default_signal.filter.resample(s.fs)
                    master = s
                else:
                    master.enslave(s)
        else:
            for s in signal:
                s.filter = self.__signal_model.default_signal.filter.resample(s.fs)

    def toggleDecimate(self, state):
        ''' toggles whether to decimate '''
        if self.wavChannelSelector.count() > 0:
            from app import wait_cursor
            with (wait_cursor()):
                self.__loaders[self.__loader_idx].toggle_decimate(int(self.wavChannelSelector.currentText()))
                self.__magnitudeModel.redraw()


class Smoother(QRunnable):
    fractions = [0, 1, 2, 3, 6, 12, 24]

    '''
    Precalculates the fractional octave smoothing.
    '''

    def __init__(self, signal_data):
        super().__init__()
        self.__signal_data = signal_data

    def run(self):
        if isinstance(self.__signal_data, BassManagedSignalData):
            for c in self.__signal_data.channels:
                QThreadPool.globalInstance().start(Smoother(c))
        else:
            self.__smooth(self.__signal_data)

    def __smooth(self, signal_data):
        for fraction in self.fractions:
            start = time.time()
            signal_data.smooth(fraction, set_active=False)
            end = time.time()
            logger.info(f"Smoothed {signal_data} at {fraction} in {round(end - start, 3)}s")


class MergeSignalDialog(QDialog, Ui_MergeSignalDialog):
    '''
    Alows user to merge multiple signals.
    '''

    def __init__(self, preferences, signal_model: SignalModel, parent=None):
        super(MergeSignalDialog, self).__init__(parent=parent)
        self.setupUi(self)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.__prefs = preferences
        self.__signal_model = signal_model
        single_signals = {s.name: s for s in self.__signal_model.non_bm_signals if s.signal is not None}
        bm_signals = {c.name: c for bm in [bm.channels for bm in self.__signal_model.bass_managed_signals] for c in bm}
        self.__signals: Dict[str, SingleChannelSignalData] = {**single_signals, **bm_signals}
        for s in self.__signals.keys():
            self.signals.addItem(s)
        self.__validate()

    def __validate(self):
        self.buttonBox.button(QDialogButtonBox.StandardButton.Save).setEnabled(len(self.signals.selectedItems()) > 0)

    def calc_duration(self):
        duration = 0
        if len(self.signals.selectedItems()) > 0:
            duration = sum([self.__signals[s.text()].duration_seconds for s in self.signals.selectedItems()])
        self.duration.setTime(QtCore.QTime(0, 0, 0).addMSecs(int(duration * 1000)))
        self.__validate()

    def accept(self):
        selected_signals: List[Signal] = [self.__signals[s.text()].signal for s in self.signals.selectedItems()]
        logger.debug(f"Merging {','.join([s.name for s in selected_signals])}")
        samples = np.concatenate([s.samples for s in selected_signals])
        suffix = f"{len([s.name for s in self.__signal_model.non_bm_signals if s.name.startswith('merged')]) + 1}"
        output_signal = Signal(f"merged{suffix}", samples, self.__prefs, fs=selected_signals[0].fs)
        self.__signal_model.add(SingleChannelSignalData(f"merged{suffix}", signal=output_signal,
                                                        filter=CompleteFilter(fs=output_signal.fs)))
        super().accept()


class SelectSignalsDialog(QDialog, Ui_selectSignalsDialog):

    def __init__(self, parent, signal_model: SignalModel, redraw: Callable[[], None]):
        super(SelectSignalsDialog, self).__init__(parent)
        self.setupUi(self)
        self.__model = signal_model
        for s in signal_model:
            self.signals.addItem(s.name)
        if self.__model.visible_names:
            for i in range(self.signals.count()):
                item = self.signals.item(i)
                if item.text() in self.__model.visible_names:
                    item.setSelected(True)
        self.__redraw = redraw

    def accept(self):
        self.__model.visible_names = [i.text() for i in self.signals.selectedItems()]
        self.__model.post_update()
        self.__redraw()
        super().accept()
