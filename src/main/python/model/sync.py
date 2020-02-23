import json
import logging
from collections import OrderedDict

import qtawesome as qta
from qtpy import QtWebSockets
from qtpy.QtCore import QUrl, Qt
from qtpy.QtWidgets import QDialog, QAbstractItemView, QHeaderView, QLineEdit, QToolButton, QListWidgetItem, QMessageBox

from model.batch import StoppableSpin, stop_spinner
from model.filter import FilterModel, FilterTableModel
from model.iir import CompleteFilter, PeakingEQ
from model.limits import dBRangeCalculator
from model.magnitude import MagnitudeModel
from model.preferences import get_filter_colour, HTP1_ADDRESS
from ui.edit_mapping import Ui_editMappingDialog
from ui.syncdetails import Ui_syncDetailsDialog
from ui.synchtp1 import Ui_syncHtp1Dialog

HTP1_FS = 48000

logger = logging.getLogger('htp1sync')


class SyncHTP1Dialog(QDialog, Ui_syncHtp1Dialog):

    def __init__(self, parent, prefs, signal_model):
        super(SyncHTP1Dialog, self).__init__(parent)
        self.__preferences = prefs
        self.__signal_model = signal_model
        self.__filters_by_channel = {}
        self.__current_device_filters_by_channel = {}
        self.__spinner = None
        self.__beq_filter = None
        self.__signal_filter = None
        self.__last_requested_msoupdate = None
        self.__last_received_msoupdate = None
        self.__channel_to_signal = {}
        self.setupUi(self)
        self.syncStatus = qta.IconWidget('fa5s.unlink')
        self.syncLayout.addWidget(self.syncStatus)
        self.ipAddress.setText(self.__preferences.get(HTP1_ADDRESS))
        self.filterView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.__filters = FilterModel(self.filterView, self.__preferences)
        self.filterView.setModel(FilterTableModel(self.__filters))
        self.filterView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.__magnitude_model = MagnitudeModel('preview', self.previewChart, self.__preferences, self, 'Filter',
                                                db_range_calc=dBRangeCalculator(30, expand=True), fill_curves=True)
        self.connectButton.setIcon(qta.icon('fa5s.check'))
        self.disconnectButton.setIcon(qta.icon('fa5s.times'))
        self.resyncFilters.setIcon(qta.icon('fa5s.sync'))
        self.deleteFiltersButton.setIcon(qta.icon('fa5s.trash'))
        self.selectBeqButton.setIcon(qta.icon('fa5s.folder-open'))
        self.limitsButton.setIcon(qta.icon('fa5s.arrows-alt'))
        self.showDetailsButton.setIcon(qta.icon('fa5s.info'))
        self.createPulsesButton.setIcon(qta.icon('fa5s.wave-square'))
        self.fullRangeButton.setIcon(qta.icon('fa5s.expand'))
        self.subOnlyButton.setIcon(qta.icon('fa5s.compress'))
        self.__ws_client = QtWebSockets.QWebSocket('', QtWebSockets.QWebSocketProtocol.Version13, None)
        self.__ws_client.error.connect(self.__on_ws_error)
        self.__ws_client.connected.connect(self.__on_ws_connect)
        self.__ws_client.disconnected.connect(self.__on_ws_disconnect)
        self.__ws_client.textMessageReceived.connect(self.__on_ws_message)
        self.__disable_on_disconnect()
        self.filterMapping.itemDoubleClicked.connect(self.__show_mapping_dialog)

    def __on_ws_error(self, error_code):
        '''
        handles a websocket client error.
        :param error_code: the error code.
        '''
        logger.error(f"error code: {error_code}")
        logger.error(self.__ws_client.errorString())

    def __on_ws_connect(self):
        logger.info(f"Connected to {self.ipAddress.text()}")
        self.__preferences.set(HTP1_ADDRESS, self.ipAddress.text())
        self.__load_peq_slots()
        self.__enable_on_connect()
        self.syncStatus.setIcon(qta.icon('fa5s.link'))
        self.syncStatus.setEnabled(True)

    def __load_peq_slots(self):
        b = self.__ws_client.sendTextMessage('getmso')
        logger.debug(f"Sent {b} bytes")

    def __on_ws_disconnect(self):
        logger.info(f"Disconnected from {self.ipAddress.text()}")
        self.__disable_on_disconnect()

    def __disable_on_disconnect(self):
        ''' Clears all relevant state on disconnect. '''
        self.connectButton.setEnabled(True)
        self.disconnectButton.setEnabled(False)
        self.ipAddress.setReadOnly(False)
        self.resyncFilters.setEnabled(False)
        self.deleteFiltersButton.setEnabled(False)
        self.showDetailsButton.setEnabled(False)
        self.createPulsesButton.setEnabled(False)
        self.filtersetSelector.clear()
        self.__filters_by_channel = {}
        self.__current_device_filters_by_channel = {}
        self.beqFile.clear()
        self.__beq_filter = None
        self.selectBeqButton.setEnabled(False)
        self.addFilterButton.setEnabled(False)
        self.removeFilterButton.setEnabled(False)
        self.applyFiltersButton.setEnabled(False)
        self.__show_signal_mapping()
        self.syncStatus.setIcon(qta.icon('fa5s.unlink'))
        self.syncStatus.setEnabled(False)
        self.__filters.filter = CompleteFilter(fs=HTP1_FS, sort_by_id=True)
        self.__magnitude_model.redraw()

    def __enable_on_connect(self):
        ''' Prepares the UI for operation. '''
        self.connectButton.setEnabled(False)
        self.disconnectButton.setEnabled(True)
        self.deleteFiltersButton.setEnabled(True)
        self.ipAddress.setReadOnly(True)
        self.resyncFilters.setEnabled(True)
        self.selectBeqButton.setEnabled(True)
        self.createPulsesButton.setEnabled(True)
        self.__show_signal_mapping()
        self.on_signal_selected()

    def __on_ws_message(self, msg):
        '''
        Handles messages from the device.
        :param msg: the message.
        '''
        if msg.startswith('mso '):
            logger.debug(f"Processing mso {msg}")
            self.__on_mso(json.loads(msg[4:]))
        elif msg.startswith('msoupdate '):
            logger.debug(f"Processing msoupdate {msg}")
            self.__on_msoupdate(json.loads(msg[10:]))
        else:
            logger.warning(f"Unknown message {msg}")
            msg_box = QMessageBox(QMessageBox.Critical, 'Unknown Message',
                                  f"Received unexpected message from {self.ipAddress.text()}")
            msg_box.setDetailedText(f"<code>{msg}</code>")
            msg_box.setTextFormat(Qt.RichText)
            msg_box.exec()

    def __on_msoupdate(self, msoupdate):
        '''
        Handles msoupdate message sent after the device is updated.
        :param msoupdate: the update.
        '''
        self.__last_received_msoupdate = msoupdate
        was_requested = False
        if self.__spinner is not None:
            was_requested = True
            stop_spinner(self.__spinner, self.syncStatus)
            self.__spinner = None
        if was_requested:
            if not self.__msoupdate_matches(msoupdate):
                self.syncStatus.setIcon(qta.icon('fa5s.times', color='red'))
                self.show_sync_details()
            else:
                self.syncStatus.setIcon(qta.icon('fa5s.check', color='green'))
        else:
            self.syncStatus.setIcon(qta.icon('fa5s.question', color='red'))
        self.__update_device_state(msoupdate)
        self.showDetailsButton.setEnabled(True)

    def __update_device_state(self, msoupdate):
        ''' applies the delta from msoupdate to the local cache of the device state. '''
        for op in msoupdate:
            if 'path' in op and op['path'].startswith('/peq/slots'):
                path = op['path']
                if path[0] == '/':
                    path = path[1:]
                tokens = path.split('/')
                idx = int(tokens[2])
                channel = tokens[4]
                attrib = tokens[5]
                val = op['value']
                if channel in self.__current_device_filters_by_channel:
                    cf = self.__current_device_filters_by_channel[channel]
                    f = cf[idx].resample(HTP1_FS)
                    if attrib == 'gaindB':
                        f.gain = float(val)
                    elif attrib == 'Fc':
                        f.freq = float(val)
                    elif attrib == 'Q':
                        f.q = float(val)
                    cf.save(f)

    def show_sync_details(self):
        if self.__last_requested_msoupdate is not None and self.__last_received_msoupdate is not None:
            SyncDetailsDialog(self, self.__last_requested_msoupdate, self.__last_received_msoupdate).exec()

    def __msoupdate_matches(self, msoupdate):
        '''
        compares each operation for equivalence.
        :param msoupdate: the update
        :returns true if they match
        '''
        for idx, actual in enumerate(msoupdate):
            expected = self.__last_requested_msoupdate[idx]
            if actual['op'] != expected['op'] or actual['path'] != expected['path'] or actual['value'] != expected['value']:
                logger.error(f"msoupdate does not match {actual} vs {expected}")
                return False
        return True

    def __on_mso(self, mso):
        '''
        Handles mso message from the device sent after a getmso is issued.
        :param mso: the mso.
        '''
        speakers = mso['speakers']['groups']
        channels = ['lf', 'rf']
        for group in [s for s, v in speakers.items() if 'present' in v and v['present'] is True]:
            if group[0:2] == 'lr' and len(group) > 2:
                channels.append('l' + group[2:])
                channels.append('r' + group[2:])
            else:
                channels.append(group)

        peq_slots = mso['peq']['slots']

        class Filters(dict):
            def __init__(self):
                super().__init__()

            def __missing__(self, key):
                self[key] = CompleteFilter(fs=HTP1_FS, sort_by_id=True)
                return self[key]

        tmp1 = Filters()
        tmp2 = Filters()
        raw_filters = {c: [] for c in channels}
        unknown_channels = set()
        for idx, s in enumerate(peq_slots):
            for c in channels:
                if c in s['channels']:
                    tmp1[c].save(self.__convert_to_filter(s['channels'][c], idx))
                    tmp2[c].save(self.__convert_to_filter(s['channels'][c], idx))
                    raw_filters[c].append(s['channels'][c])
                else:
                    unknown_channels.add(c)
        if unknown_channels:
            peq_channels = peq_slots[0]['channels'].keys()
            logger.error(f"Unknown channels encountered [peq channels: {peq_channels}, unknown: {unknown_channels}]")

        from model.report import block_signals
        with block_signals(self.filtersetSelector):
            now = self.filtersetSelector.currentText()
            self.filtersetSelector.clear()
            now_idx = -1
            for idx, c in enumerate(channels):
                self.filtersetSelector.addItem(c)
                if c == now:
                    now_idx = idx
            if now_idx > -1:
                self.filtersetSelector.setCurrentIndex(now_idx)

        self.__filters_by_channel = tmp1
        self.__current_device_filters_by_channel = tmp2
        self.__filters.filter = self.__filters_by_channel[self.filtersetSelector.itemText(0)]
        if not self.__channel_to_signal:
            self.__recalc_mapping()
            self.__show_signal_mapping()
        self.__magnitude_model.redraw()
        self.syncStatus.setIcon(qta.icon('fa5s.link'))
        self.__last_received_msoupdate = None
        self.__last_requested_msoupdate = None
        self.showDetailsButton.setEnabled(False)

    @staticmethod
    def __convert_to_filter(filter_params, id):
        # TODO support different filter types
        return PeakingEQ(HTP1_FS, filter_params['Fc'], filter_params['Q'], filter_params['gaindB'], f_id=id)

    def add_filter(self):
        '''
        Adds the filter to the selected channel.
        '''
        selected_filter = self.__get_selected_filter()
        if selected_filter is not None:
            if self.__in_complex_mode():
                for i in self.filterMapping.selectedItems():
                    c = i.text().split(' ')[1]
                    s = self.__channel_to_signal[c]
                    if s is not None:
                        self.__apply_filter(selected_filter, s.filter)
                        if c == self.filtersetSelector.currentText():
                            self.__filters.filter = self.__filters.filter
            else:
                self.__apply_filter(selected_filter, self.__filters.filter)
                self.__filters.filter = self.__filters.filter
            self.__magnitude_model.redraw()

    @staticmethod
    def __apply_filter(source, target):
        '''
        Places the target filter in the source in the last n slots.
        :param source: the source filter.
        :param target: the target.
        '''
        max_idx = len(target)
        target.removeByIndex(range(max_idx-len(source), max_idx))
        for f in source:
            f.id = len(target)
            target.save(f)

    def __get_selected_filter(self):
        selected_filter = None
        if self.__signal_filter is not None:
            selected_filter = self.__signal_filter
        elif self.__beq_filter is not None:
            selected_filter = self.__beq_filter
        return selected_filter

    def remove_filter(self):
        '''
        Searches for the BEQ in the selected channel and highlights them for deletion.
        Auto deletes if in complex mode.
        '''
        selected_filter = self.__get_selected_filter()
        if selected_filter is not None:
            if self.__in_complex_mode():
                for i in self.filterMapping.selectedItems():
                    c = i.text().split(' ')[1]
                    s = self.__channel_to_signal[c]
                    if s is not None:
                        for r in self.__remove_from_filter(selected_filter, s.filter):
                            to_save = self.__filters if c == self.filtersetSelector.currentText() else s.filter
                            to_save.save(self.__make_passthrough(r))
                self.__magnitude_model.redraw()
            else:
                rows = self.__remove_from_filter(selected_filter, self.__filters.filter)
                self.filterView.clearSelection()
                if len(rows) > 0:
                    for r in rows:
                        self.filterView.selectRow(r)

    def __remove_from_filter(self, to_remove, target):
        filt_idx = 0
        rows = []
        for i, f in enumerate(target):
            if self.__is_equivalent(f, to_remove[filt_idx]):
                filt_idx += 1
                rows.append(i)
            if filt_idx >= len(to_remove):
                break
        return rows

    def select_filter(self, channel_name):
        '''
        loads a filter from the current signals.
        '''
        signal = self.__signal_model.find_by_name(channel_name)
        if signal is not None:
            self.__signal_filter = signal.filter
        else:
            self.__signal_filter = None
        self.__enable_filter_buttons()

    @staticmethod
    def __is_equivalent(a, b):
        return a.freq == b.freq and a.gain == b.gain and hasattr(a, 'q') and hasattr(b, 'q') and a.q == b.q

    def send_filters_to_device(self):
        '''
        Sends the selected filters to the device
        '''
        if self.__in_complex_mode():
            ops, non_peq_filter_types_by_channel = self.__convert_filter_mappings_to_ops()
            channels_to_update = [i.text().split(' ')[1] for i in self.filterMapping.selectedItems()]
            unsynced_channels = []
            for c, s in self.__channel_to_signal.items():
                if c not in channels_to_update:
                    if c in self.__current_device_filters_by_channel:
                        current = s.filter
                        device = self.__current_device_filters_by_channel[c]
                        if current != device:
                            unsynced_channels.append(c)
            if unsynced_channels:
                result = QMessageBox.question(self,
                                              'Sync all changed channels?',
                                              f"Filters in {len(unsynced_channels)} channels have changed but will not be "
                                              f"synced to the HTP-1."
                                              f"\n\nChannels: {', '.join(sorted([k for k in unsynced_channels]))}"
                                              f"\n\nDo you want to sync all changed channels? ",
                                              QMessageBox.Yes | QMessageBox.No,
                                              QMessageBox.No)
                if result == QMessageBox.Yes:
                    for c in unsynced_channels:
                        for i in range(self.filterMapping.count()):
                            item: QListWidgetItem = self.filterMapping.item(i)
                            if c == item.text().split(' ')[1]:
                                item.setSelected(True)
                    self.send_filters_to_device()
                    return
        else:
            ops, non_peq_filter_types_by_channel = self.__convert_current_filter_to_ops()
        do_send = True
        if non_peq_filter_types_by_channel:
            printed = '\n'.join([f"{k} - {', '.join(v)}" for k, v in non_peq_filter_types_by_channel.items()])
            result = QMessageBox.question(self,
                                          'Unsupported Filters Detected',
                                          f"HTP-1 supports PEQ filters only.\n\nUnsupported filters found:"
                                          f"\n\n{printed}"
                                          f"\n\nDo you want sync the PEQ filters only? ",
                                          QMessageBox.Yes | QMessageBox.No,
                                          QMessageBox.No)
            do_send = result == QMessageBox.Yes
        if do_send:
            from app import wait_cursor
            with wait_cursor():
                all_ops = [op for slot_ops in ops for op in slot_ops]
                self.__last_requested_msoupdate = all_ops
                msg = f"changemso {json.dumps(self.__last_requested_msoupdate)}"
                logger.debug(f"Sending to {self.ipAddress.text()} -> {msg}")
                self.__spinner = StoppableSpin(self.syncStatus, 'sync')
                spin_icon = qta.icon('fa5s.spinner', color='green', animation=self.__spinner)
                self.syncStatus.setIcon(spin_icon)
                self.__ws_client.sendTextMessage(msg)

    def __convert_current_filter_to_ops(self):
        '''
        Converts the selected filter to operations.
        :return: (the operations, channel with non PEQ filters)
        '''
        non_peq_filter_types_by_channel = {}
        selected_filter = self.filtersetSelector.currentText()
        ops = [self.__as_operation(idx, selected_filter, f) for idx, f in enumerate(self.__filters.filter)]
        non_peq_types = [f.filter_type for f in self.__filters.filter if f.filter_type != 'PEQ']
        if non_peq_types:
            non_peq_filter_types_by_channel[selected_filter] = set(non_peq_types)
        return ops, non_peq_filter_types_by_channel

    def __convert_filter_mappings_to_ops(self):
        '''
        Converts the selected filter mappings to operations.
        :return: (the operations, the channels with non PEQ filters)
        '''
        ops = []
        non_peq_filter_types_by_channel = OrderedDict()
        for i in self.filterMapping.selectedItems():
            c = i.text().split(' ')[1]
            s = self.__channel_to_signal[c]
            if s is not None:
                non_peq_types = [f.filter_type for f in s.filter if f.filter_type != 'PEQ']
                if non_peq_types:
                    non_peq_filter_types_by_channel[c] = sorted(set(non_peq_types))
                channel_ops = [self.__as_operation(idx, c, f) for idx, f in enumerate(s.filter)]
                remainder = 16 - len(channel_ops)
                if remainder > 0:
                    channel_ops += [self.__make_passthrough(i) for i in range(remainder)]
                ops += channel_ops
        return ops, non_peq_filter_types_by_channel

    @staticmethod
    def __make_passthrough(idx):
        return PeakingEQ(HTP1_FS, 100, 1, 0, f_id=idx)

    @staticmethod
    def __as_operation(idx, channel, f):
        prefix = f"/peq/slots/{idx}/channels/{channel}"
        return [
            {
                'op': 'replace',
                'path': f"{prefix}/Fc",
                'value': f.freq
            },
            {
                'op': 'replace',
                'path': f"{prefix}/Q",
                'value': f.q
            },
            {
                'op': 'replace',
                'path': f"{prefix}/gaindB",
                'value': f.gain
            }
        ]

    def clear_filters(self):
        '''
        Replaces the selected filters.
        '''
        selection_model = self.filterView.selectionModel()
        if selection_model.hasSelection():
            for x in selection_model.selectedRows():
                self.__filters.save(self.__make_passthrough(x.row()))
            self.__magnitude_model.redraw()

    def connect_htp1(self):
        '''
        Connects to the websocket of the specified ip:port.
        '''
        self.ipAddress.setReadOnly(True)
        logger.info(f"Connecting to {self.ipAddress.text()}")
        self.__ws_client.open(QUrl(f"ws://{self.ipAddress.text()}/ws/controller"))

    def disconnect_htp1(self):
        '''
        disconnects the ws
        '''
        logger.info(f"Closing connection to {self.ipAddress.text()}")
        self.__ws_client.close()
        logger.info(f"Closed connection to {self.ipAddress.text()}")
        self.ipAddress.setReadOnly(False)

    def display_filterset(self, filterset):
        '''
        Displays the selected filterset in the chart and the table.
        :param filterset: the filterset.
        '''
        if filterset in self.__filters_by_channel:
            self.__filters.filter = self.__filters_by_channel[filterset]
            self.__magnitude_model.redraw()
        else:
            logger.warning(f"Unknown filterset {filterset}")

    def resync_filters(self):
        '''
        reloads the PEQ slots
        '''
        self.__load_peq_slots()
        self.beqFile.clear()

    def select_beq(self):
        '''
        Selects a filter from the BEQ repos.
        '''
        from model.minidsp import load_as_filter

        filters, file_name = load_as_filter(self, self.__preferences, HTP1_FS)
        self.beqFile.setText(file_name)
        self.__beq_filter = filters
        self.__enable_filter_buttons()

    def __enable_filter_buttons(self):
        enable = self.__beq_filter is not None or self.__signal_filter is not None
        self.addFilterButton.setEnabled(enable)
        self.removeFilterButton.setEnabled(enable)

    def getMagnitudeData(self, reference=None):
        ''' preview of the filter to display on the chart '''
        result = []
        if len(self.__filters) > 0:
            result.append(self.__filters.getTransferFunction().getMagnitude(colour=get_filter_colour(len(result))))
            for f in self.__filters:
                result.append(f.getTransferFunction()
                               .getMagnitude(colour=get_filter_colour(len(result)), linestyle=':'))
        return result

    def reject(self):
        '''
        Ensures the HTP1 is disconnected before closing.
        '''
        self.disconnect_htp1()
        super().reject()

    def show_limits(self):
        '''
        Shows the limits dialog.
        '''
        self.__magnitude_model.show_limits()

    def show_full_range(self):
        ''' sets the limits to full range. '''
        self.__magnitude_model.show_full_range()

    def show_sub_only(self):
        ''' sets the limits to sub only. '''
        self.__magnitude_model.show_sub_only()

    def create_pulses(self):
        ''' Creates a dirac pulse with the specified filter for each channel and loads them into the signal model. '''
        for p in [self.__create_pulse(c, f) for c, f in self.__filters_by_channel.items()]:
            self.__signal_model.add(p)
        self.load_from_signals()

    def __recalc_mapping(self):
        for c in self.__filters_by_channel.keys():
            found = False
            for s in self.__signal_model:
                if s.name.endswith(f"_{c}"):
                    self.__channel_to_signal[c] = s
                    found = True
                    break
            if not found:
                self.__channel_to_signal[c] = None

    def __show_signal_mapping(self):
        show_it = self.__in_complex_mode()
        self.loadFromSignalsButton.setEnabled(show_it)
        self.filterMappingLabel.setVisible(show_it)
        self.filterMapping.setVisible(show_it)
        self.filterMappingLabel.setEnabled(show_it)
        self.filterMapping.setEnabled(show_it)
        self.selectNoneButton.setVisible(show_it)
        self.selectAllButton.setVisible(show_it)
        self.filterMapping.clear()
        for c, signal in self.__channel_to_signal.items():
            self.filterMapping.addItem(f"Channel {c} -> {signal.name if signal else 'No Filter'}")
        self.filtersetLabel.setText('Preview' if show_it else 'Channel')

    def __in_complex_mode(self):
        '''
        :return: true if we have signals and are connected.
        '''
        return len(self.__signal_model) > 0 and not self.connectButton.isEnabled()

    def __create_pulse(self, c, f):
        from scipy.signal import unit_impulse
        from model.signal import SingleChannelSignalData, Signal
        signal = Signal(f"pulse_{c}", unit_impulse(4 * HTP1_FS, 'mid'), self.__preferences, fs=HTP1_FS)
        return SingleChannelSignalData(name=f"pulse_{c}", filter=f, signal=signal)

    def load_from_signals(self):
        '''
        Maps the signals to the HTP1 channels.
        '''
        self.__recalc_mapping()
        self.__show_signal_mapping()
        self.on_signal_selected()

    def __show_mapping_dialog(self, item: QListWidgetItem):
        ''' Shows the edit mapping dialog '''
        if self.filterMapping.isEnabled():
            text = item.text()
            channel_name = text.split(' ')[1]
            mapped_signal = self.__channel_to_signal.get(channel_name, None)
            EditMappingDialog(self, self.__filters_by_channel.keys(), channel_name, self.__signal_model,
                              mapped_signal, self.__map_signal_to_channel).exec()

    def __map_signal_to_channel(self, channel_name, signal):
        '''
        updates the mapping of a signal to a channel.
        :param channel_name: the channel name.
        :param signal: the mapped signal.
        '''
        if signal:
            self.__channel_to_signal[channel_name] = signal
        else:
            self.__channel_to_signal[channel_name] = None
        self.__show_signal_mapping()

    def on_signal_selected(self):
        ''' if any channels are selected, enable the sync button. '''
        is_selected = self.filterMapping.selectionModel().hasSelection()
        enable = is_selected or self.filterMapping.count() == 0
        self.applyFiltersButton.setEnabled(enable)
        enable_beq = enable and self.__get_selected_filter() is not None
        self.addFilterButton.setEnabled(enable_beq)
        self.removeFilterButton.setEnabled(enable_beq)
        self.addFilterButton.setEnabled(enable_beq)
        self.removeFilterButton.setEnabled(enable_beq)

    def clear_sync_selection(self):
        ''' clears any selection for sync . '''
        self.filterMapping.selectionModel().clearSelection()

    def select_all_for_sync(self):
        ''' selects all channels for sync. '''
        self.filterMapping.selectAll()


class SyncDetailsDialog(QDialog, Ui_syncDetailsDialog):

    def __init__(self, parent, expected_ops, actual_ops):
        super(SyncDetailsDialog, self).__init__(parent)
        self.setupUi(self)
        for idx, actual_op in enumerate(actual_ops):
            actual = actual_ops[idx]
            expected = expected_ops[idx]
            path = QLineEdit(self.scrollAreaWidgetContents)
            path.setReadOnly(True)
            path.setEnabled(False)
            path.setText(actual['path'])
            requested = QLineEdit(self.scrollAreaWidgetContents)
            requested.setReadOnly(True)
            path.setEnabled(False)
            requested.setText(str(expected['value']))
            actual_text = QLineEdit(self.scrollAreaWidgetContents)
            actual_text.setReadOnly(True)
            path.setEnabled(False)
            actual_text.setText(str(actual['value']))
            status = QToolButton(self)
            if actual['op'] != expected['op'] or actual['path'] != expected['path'] or actual['value'] != expected['value']:
                status.setIcon(qta.icon('fa5s.times', color='red'))
            else:
                status.setIcon(qta.icon('fa5s.check', color='green'))
            self.gridLayout_2.addWidget(path, idx+1, 0, 1, 1)
            self.gridLayout_2.addWidget(requested, idx+1, 1, 1, 1)
            self.gridLayout_2.addWidget(actual_text, idx+1, 2, 1, 1)
            self.gridLayout_2.addWidget(status, idx+1, 3, 1, 1)


class EditMappingDialog(QDialog, Ui_editMappingDialog):
    ''' Allows the user to override the signal to channel mapping '''

    def __init__(self, parent, channels, channel_name, signal_model, selected_signal, on_change_handler):
        super(EditMappingDialog, self).__init__(parent)
        self.setupUi(self)
        self.__channels = channels
        self.__signal_model = signal_model
        self.signal.addItem('No Filter')
        if len(signal_model) > 0:
            for idx, s in enumerate(signal_model):
                self.signal.addItem(s.name)
        for idx, c in enumerate(channels):
            self.channels.addItem(c)
            if c == channel_name:
                self.channels.setCurrentRow(idx)
        if selected_signal is not None:
            self.signal.setCurrentText(selected_signal.name)
        self.on_change_handler = on_change_handler

    def accept(self):
        signal_name = None if self.signal.currentText() == 'No Filter' else self.signal.currentText()
        signal = None
        if len(self.__signal_model) > 0:
            signal = next((s for s in self.__signal_model if s.name == signal_name), None)
        for c in self.channels.selectedItems():
            self.on_change_handler(c.text(), signal)
        super().accept()
