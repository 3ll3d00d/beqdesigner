import json
import logging

import qtawesome as qta
from qtpy import QtWebSockets
from qtpy.QtCore import QUrl
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QDialog, QAbstractItemView, QHeaderView, QLineEdit, QToolButton

from model.batch import StoppableSpin, stop_spinner
from model.filter import FilterModel, FilterTableModel
from model.iir import CompleteFilter, PeakingEQ
from model.limits import dBRangeCalculator
from model.magnitude import MagnitudeModel
from model.preferences import get_filter_colour, HTP1_ADDRESS
from ui.syncdetails import Ui_syncDetailsDialog
from ui.synchtp1 import Ui_syncHtp1Dialog

HTP1_FS = 48000

logger = logging.getLogger('htp1sync')


class SyncHTP1Dialog(QDialog, Ui_syncHtp1Dialog):

    def __init__(self, parent, prefs):
        super(SyncHTP1Dialog, self).__init__(parent)
        self.__preferences = prefs
        self.__filters_by_channel = {}
        self.__spinner = None
        self.__beq_filter = None
        self.__last_requested_msoupdate = None
        self.__last_received_msoupdate = None
        self.setupUi(self)
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
        self.__ws_client = QtWebSockets.QWebSocket('', QtWebSockets.QWebSocketProtocol.Version13, None)
        self.__ws_client.error.connect(self.__on_ws_error)
        self.__ws_client.connected.connect(self.__on_ws_connect)
        self.__ws_client.disconnected.connect(self.__on_ws_disconnect)
        self.__ws_client.textMessageReceived.connect(self.__on_ws_message)
        self.__disable_on_disconnect()

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
        self.filtersetSelector.clear()
        self.__filters_by_channel = {}
        self.beqFile.clear()
        self.__beq_filter = None
        self.selectBeqButton.setEnabled(False)
        self.addBeqButton.setEnabled(False)
        self.removeBeqButton.setEnabled(False)
        self.applyFiltersButton.setEnabled(False)
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
        self.applyFiltersButton.setEnabled(True)

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
            logger.info(f"Unknown message {msg}")

    def __on_msoupdate(self, msoupdate):
        '''
        Handles msoupdate message sent after the device is updated.
        :param msoupdate: the update.
        '''
        self.__last_received_msoupdate = msoupdate
        was_requested = False
        if self.__spinner is not None:
            was_requested = True
            stop_spinner(self.__spinner, self.applyFiltersButton)
            self.applyFiltersButton.setIcon(QIcon())
            self.__spinner = None
        if self.__msoupdate_matches(msoupdate):
            pass
        else:
            if was_requested:
                self.show_sync_details()
            else:
                self.applyFiltersButton.setIcon(qta.icon('fa5s.times', color='red'))
        self.showDetailsButton.setEnabled(True)

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
        channels = ['lf', 'rf'] + [s for s, v in speakers.items() if 'present' in v and v['present'] is True]
        peq_slots = mso['peq']['slots']
        from model.report import block_signals
        with block_signals(self.filtersetSelector):
            now = self.filtersetSelector.currentText()
            self.filtersetSelector.clear()
            now_idx = -1
            for idx, c in enumerate(sorted(channels)):
                self.filtersetSelector.addItem(c)
                if c == now:
                    now_idx = idx
            if now_idx > -1:
                self.filtersetSelector.setCurrentIndex(now_idx)

        tmp = {c: CompleteFilter(fs=HTP1_FS, sort_by_id=True) for c in channels}
        raw_filters = {c: [] for c in channels}
        for idx, s in enumerate(peq_slots):
            for c in channels:
                tmp[c].save(self.__convert_to_filter(s['channels'][c], idx))
                raw_filters[c].append(s['channels'][c])
        # raw_filters_txt = {k: json.dumps(v) for k, v in raw_filters.items()}
        # from collections import defaultdict
        # filtersets = defaultdict(list)
        # for key, value in sorted(raw_filters_txt.items()):
        #     filtersets[value].append(key)
        # print(filtersets.values())
        # TODO provide a way to group and ungroup channels
        self.__filters_by_channel = tmp
        self.__filters.filter = self.__filters_by_channel[self.filtersetSelector.itemText(0)]
        self.__magnitude_model.redraw()
        self.applyFiltersButton.setIcon(QIcon())
        self.__last_received_msoupdate = None
        self.__last_requested_msoupdate = None
        self.showDetailsButton.setEnabled(False)

    @staticmethod
    def __convert_to_filter(filter_params, id):
        # TODO support different filter types
        return PeakingEQ(HTP1_FS, filter_params['Fc'], filter_params['Q'], filter_params['gaindB'], f_id=id)

    def add_beq(self):
        '''
        Adds the BEQ to the selected channel.
        '''
        if self.__beq_filter is not None:
            max_idx = len(self.__filters.filter)
            self.__filters.filter.removeByIndex(range(max_idx-len(self.__beq_filter), max_idx))
            for f in self.__beq_filter:
                f.id = len(self.__filters.filter)
                self.__filters.save(f)
            self.__magnitude_model.redraw()

    def remove_beq(self):
        '''
        Searches for the BEQ in the selected channel and highlights them for deletion.
        '''
        beq_filt_idx = 0
        rows = []
        for i, f in enumerate(self.__filters.filter):
            if self.__is_equivalent(f, self.__beq_filter[beq_filt_idx]):
                beq_filt_idx += 1
                rows.append(i)
            if beq_filt_idx >= len(self.__beq_filter):
                break
        self.filterView.clearSelection()
        if len(rows) > 0:
            for r in rows:
                self.filterView.selectRow(r)

    @staticmethod
    def __is_equivalent(a, b):
        return a.freq == b.freq and a.gain == b.gain and hasattr(a, 'q') and hasattr(b, 'q') and a.q == b.q

    def apply_filters(self):
        '''
        Sends the selected filters to the device
        '''
        from app import wait_cursor
        with wait_cursor():
            ops = [self.__as_operation(idx, self.filtersetSelector.currentText(), f) for idx, f in enumerate(self.__filters.filter)]
            all_ops = [op for slot_ops in ops for op in slot_ops]
            self.__last_requested_msoupdate = all_ops
            msg = f"changemso {json.dumps(self.__last_requested_msoupdate)}"
            logger.debug(f"Sending to {self.ipAddress.text()} -> {msg}")
            self.__spinner = StoppableSpin(self.applyFiltersButton, 'sync')
            spin_icon = qta.icon('fa5s.spinner', color='green', animation=self.__spinner)
            self.applyFiltersButton.setIcon(spin_icon)
            self.__ws_client.sendTextMessage(msg)

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
                self.__filters.save(PeakingEQ(HTP1_FS, 100, 1, 0, f_id=x.row()))
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
        self.addBeqButton.setEnabled(filters is not None)
        self.removeBeqButton.setEnabled(filters is not None)

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
