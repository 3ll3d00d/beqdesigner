'''
The Qt side of `model.ffmpeg`: running an extraction on the thread pool with progress signals (`Executor.execute()`), and the
probe and ffmpeg-command dialogs. `model.ffmpeg` itself is Qt-free, so the headless pipeline can extract with `Executor.run_sync()`.
'''
import os
import time

import ffmpeg
from qtpy import QtWidgets
from qtpy.QtCore import QObject, QRunnable, Signal
from qtpy.QtWidgets import QDialog, QTreeWidget, QTreeWidgetItem

from model.ffmpeg import FfmpegProgressBridge, SIGNAL_CANCELLED, SIGNAL_COMPLETE, SIGNAL_CONNECTED, SIGNAL_ERROR, \
    describe_missing_binary, logger
from ui.ffmpeg import Ui_ffmpegReportDialog


class JobSignals(QObject):
    on_progress = Signal(str, str, name='on_progress')


class AudioExtractor(QRunnable):
    '''
    Allows audio extraction to be performed outside the main UI thread.
    '''

    def __init__(self, executor, port=None, progress_handler=None, cancel=False):
        super().__init__()
        self.__executor = executor
        self.__progress_handler = progress_handler
        self.__signals = JobSignals()
        if self.__progress_handler is not None:
            self.__signals.on_progress.connect(self.__progress_handler)
            self.__signals.on_progress.emit(SIGNAL_CONNECTED, '')
        self.__socket_server = None
        self.__port = port
        self.__cancel = cancel

    def __del__(self):
        self.__stop_socket_server()

    def cancel(self):
        '''
        Attempts to cancel the job. Currently only has any effect if it hasn't already started.
        '''
        self.__cancel = True

    def enable(self):
        '''
        Enables the job. Currently only has any effect if it hasn't already started.
        '''
        self.__cancel = False

    def run(self):
        '''
        Executes the ffmpeg command.
        '''
        if self.__cancel is True:
            self.__signals.on_progress.emit(SIGNAL_CANCELLED, 'Cancelled')
        else:
            self.__start_socket_server()
            start = time.time()
            try:
                logger.info("Starting ffmpeg command")
                out, err = self.__executor.run_sync(start_progress_bridge=False)
                end = time.time()
                elapsed = round(end - start, 3)
                logger.info(f"Executed ffmpeg command in {elapsed}s")
                result = f"Command completed normally in {elapsed}s" + os.linesep + os.linesep
                result = self.__append_out_err(err, out, result)
                self.__signals.on_progress.emit(SIGNAL_COMPLETE, result)
            except ffmpeg.Error as e:
                end = time.time()
                elapsed = round(end - start, 3)
                logger.info(f"FAILED to execute ffmpeg command in {elapsed}s")
                result = f"Command FAILED in {elapsed}s" + os.linesep + os.linesep
                result = self.__append_out_err(e.stderr, e.stdout, result)
                self.__signals.on_progress.emit(SIGNAL_ERROR, result)
            except FileNotFoundError as e:
                end = time.time()
                elapsed = round(end - start, 3)
                logger.error(f"FAILED to execute ffmpeg command in {elapsed}s, {e.filename} not found")
                result = f"Command FAILED in {elapsed}s" + os.linesep + os.linesep + describe_missing_binary(e)
                self.__signals.on_progress.emit(SIGNAL_ERROR, result)
            finally:
                self.__stop_socket_server()

    def __append_out_err(self, err, out, result):
        result += 'STDOUT' + os.linesep + '------' + os.linesep + os.linesep
        if out is not None:
            result += out.decode() + os.linesep
        result += 'STDERR' + os.linesep + '------' + os.linesep + os.linesep
        if err is not None:
            result += err.decode() + os.linesep
        return result

    def __start_socket_server(self):
        if self.__progress_handler is not None:
            self.__socket_server = FfmpegProgressBridge(self.handle_progress_event, port=self.__port, auto=True)

    def __stop_socket_server(self):
        if self.__socket_server is not None:
            self.__socket_server.stop()

    def handle_progress_event(self, key, value):
        '''
        Callback from ffmpeg -progress, passes the value straight to the __progress_handler
        :param key: the key.
        :param value: the value.
        '''
        logger.debug(f"Received -- 127.0.0.1:{self.__port} -- {key}={value}")
        self.__signals.on_progress.emit(key, value)


class ViewProbeDialog(QDialog):
    '''
    Shows the tree widget in a separate dialog.
    '''

    def __init__(self, name, probe, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(f"ffprobe data {name}")
        self.resize(400, 600)
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setObjectName("gridLayout")
        self.probeTree = ViewTree(probe)
        self.gridLayout.addWidget(self.probeTree, 1, 1, 1, 1)


class ViewTree(QTreeWidget):
    '''
    Renders a dict as a tree, taken from https://stackoverflow.com/a/46096319/123054
    '''

    def __init__(self, value):
        super().__init__()

        def fill_item(item, value):
            def new_item(parent, text, val=None):
                child = QTreeWidgetItem([text])
                fill_item(child, val)
                parent.addChild(child)
                child.setExpanded(True)

            if value is None:
                return
            elif isinstance(value, dict):
                for key, val in sorted(value.items()):
                    new_item(item, str(key), val)
            elif isinstance(value, (list, tuple)):
                for val in value:
                    text = (str(val) if not isinstance(val, (dict, list, tuple))
                            else '[%s]' % type(val).__name__)
                    new_item(item, text, val)
            else:
                new_item(item, str(value))

        fill_item(self.invisibleRootItem(), value)


class FFMpegDetailsDialog(QDialog, Ui_ffmpegReportDialog):
    def __init__(self, name, parent):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.setWindowTitle(f"ffmpeg: {name}")
