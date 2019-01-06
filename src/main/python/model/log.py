import itertools
import logging

from qtpy import QtGui
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QMainWindow

from model.preferences import LOGGING_LEVEL
from ui.logs import Ui_logsForm


class LogViewer(QMainWindow, Ui_logsForm):
    '''
    A window which displays logging.
    '''

    def __init__(self, owner, max_size):
        super(LogViewer, self).__init__()
        self.setupUi(self)
        self.logViewer.setMaximumBlockCount(max_size)
        self.__owner = owner

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        '''
        Propagates the window close event.
        '''
        self.__owner.close_logs()
        super().closeEvent(a0)

    def setLogSize(self, size):
        '''
        Updates the log size.
        :param level: the new size.
        '''
        self.__owner.set_size(size)
        self.logViewer.setMaximumBlockCount(size)

    def setLogLevel(self, level):
        '''
        Updates the log level.
        :param level: the new level.
        '''
        if level:
            self.__owner.change_level(level)

    def refresh(self, data):
        '''
        Refreshes the displayed data.
        :param data: the data.
        '''
        self.logViewer.clear()
        for d in data:
            if d is not None:
                self.logViewer.appendPlainText(d)

    def appendMsg(self, msg):
        '''
        Shows the message.
        :param idx: the idx.
        :param msg: the msg.
        '''
        self.logViewer.appendPlainText(msg)
        self.logViewer.verticalScrollBar().setValue(self.logViewer.verticalScrollBar().maximum())


class MessageSignals(QObject):
    append_msg = Signal(str, name='append_msg')


class RollingLogger(logging.Handler):
    def __init__(self, preferences, size=1000, parent=None):
        super().__init__()
        self.__buffer = RingBuffer(size)
        self.__signals = MessageSignals()
        self.__visible = False
        self.__logWindow = None
        self.__preferences = preferences
        self.parent = parent
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(threadName)s	- %(name)s - %(funcName)s - %(message)s'))
        level = self.__preferences.get(LOGGING_LEVEL)
        if level is not None and level in logging._nameToLevel:
            level = logging._nameToLevel[level]
        else:
            level = logging.INFO
        self.__root = self.__init_root_logger(level)
        self.__levelName = logging.getLevelName(level)

    def __init_root_logger(self, level):
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.addHandler(self)
        return root_logger

    def emit(self, record):
        msg = self.format(record)
        self.__buffer.append(msg)
        self.__signals.append_msg.emit(msg)

    def show_logs(self):
        '''
        Creates a new log viewer window.
        '''
        if self.__logWindow is None:
            self.__logWindow = LogViewer(self, len(self.__buffer))
            self.__logWindow.maxRows.setValue(len(self.__buffer))
            self.__signals.append_msg.connect(self.__logWindow.appendMsg)
            levelIdx = self.__logWindow.logLevel.findText(self.__levelName)
            self.__logWindow.logLevel.setCurrentIndex(levelIdx)
            self.__logWindow.show()
            self.__logWindow.refresh(self.__buffer)

    def close_logs(self):
        '''
        Reacts to the closure of the window so we don't keep writing logs to something that doesn't exist.
        '''
        self.__logWindow = None
        self.__signals.append_msg.disconnect()

    def set_size(self, size):
        '''
        Changes the size of the log cache.
        '''
        if self.__buffer.set_size(size):
            if self.__logWindow is not None:
                self.__logWindow.refresh(self.__buffer)

    def change_level(self, level):
        '''
        Change the root logger level.
        :param level: the new level name.
        '''
        logging.info(f"Changing log level from {self.__levelName} to {level}")
        self.__root.setLevel(level)
        self.__levelName = level
        self.__preferences.set(LOGGING_LEVEL, self.__levelName)


class RingBuffer:
    '''
    A simple circular data structure which is iterable and resizeable.
    '''

    def __init__(self, size):
        self.__data = [None] * size
        self.__index = 0

    def __iter__(self):
        return get_iterable_for_rb(self.__data, self.__index)

    def __len__(self):
        return len(self.__data)

    def append(self, msg):
        '''
        Adds a new piece of data.
        :param msg: the msg
        '''
        self.__data[self.__index] = msg
        if self.__index == len(self) - 1:
            self.__index = 0
        else:
            self.__index += 1

    def set_size(self, new_size):
        '''
        Resizes the buffer if required.
        :param new_size: the new size.
        :return: true if it was resized.
        '''
        old_size = len(self)
        self.__index = resize(old_size, new_size, self.__data, self.__index)
        return old_size != new_size


def get_iterable_for_rb(data, idx):
    '''
    Provides an iterator on a circulr data structure.
    :param data: the data.
    :param idx: the current index.
    :return: an iterator.
    '''
    if idx == 0:
        return iter(data)
    else:
        return itertools.chain(data[idx:len(data)], data[0:idx])


def resize(old_size, new_size, data, idx):
    '''
    Resizes the circular data from currentSize to newSize
    :param old_size: the current size.
    :param new_size: the new size.
    :param data: the data
    :param idx: the circular buffer idx.
    '''
    if new_size > old_size:
        data.extend(([None] * (new_size - old_size)))
        return idx
    elif new_size < old_size:
        to_delete = old_size - new_size
        to_delete_at_end = old_size - idx
        if to_delete_at_end < to_delete:
            del data[idx:old_size]
            del data[0:(to_delete - to_delete_at_end)]
            return len(data) - 1
        else:
            del data[idx:idx + to_delete]
            return idx
    else:
        return idx


def to_millis(start, end):
    '''
    Calculates the differences in time in millis.
    :param start: start time in seconds.
    :param end: end time in seconds.
    :return: delta in millis.
    '''
    return round((end - start) * 1000)
