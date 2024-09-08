import logging
from collections.abc import Sequence

import numpy as np
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QMainWindow

from model.preferences import LOGGING_LEVEL
from ui.logs import Ui_logsForm

logger = logging.getLogger('log')


class LogViewer(QMainWindow, Ui_logsForm):
    change_level = Signal(str)
    set_size = Signal(int)
    set_exclude_filter = Signal(str)

    '''
    A window which displays logging.
    '''

    def __init__(self, max_size):
        super(LogViewer, self).__init__()
        self.setupUi(self)
        self.maxRows.setValue(max_size)
        self.logViewer.setMaximumBlockCount(max_size)

    def closeEvent(self, event):
        '''
        Propagates the window close event.
        '''
        self.hide()

    def set_log_size(self, size):
        '''
        Updates the log size.
        :param size: the new size.
        '''
        self.set_size.emit(size)
        self.logViewer.setMaximumBlockCount(size)

    def set_log_level(self, level):
        '''
        Updates the log level.
        :param level: the new level.
        '''
        self.change_level.emit(level)

    def set_excludes(self):
        self.set_exclude_filter.emit(self.excludes.text())

    def refresh(self, data):
        '''
        Refreshes the displayed data.
        :param data: the data.
        '''
        self.logViewer.clear()
        for d in data:
            if d is not None:
                self.logViewer.appendPlainText(d)

    def append_msg(self, msg):
        '''
        Shows the message.
        :param msg: the msg.
        '''
        self.logViewer.appendPlainText(msg)
        self.logViewer.verticalScrollBar().setValue(self.logViewer.verticalScrollBar().maximum())


class MessageSignals(QObject):
    append_msg = Signal(str, name='append_msg')


class RollingLogger(logging.Handler):
    def __init__(self, preferences, size=1000, parent=None):
        super().__init__()
        self.__buffer = RingBuffer(size, dtype=object)
        self.__signals = MessageSignals()
        self.__visible = False
        self.__window = None
        self.__preferences = preferences
        self.__excludes = []
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
        if not any(e in msg for e in self.__excludes):
            self.__buffer.append(msg)
            self.__signals.append_msg.emit(msg)

    def show_logs(self):
        '''
        Creates a new log viewer window.
        '''
        if self.__window is None:
            self.__window = LogViewer(self.__buffer.maxlen)
            self.__window.set_size.connect(self.set_size)
            self.__window.change_level.connect(self.change_level)
            self.__window.set_exclude_filter.connect(self.set_excludes)
            self.__signals.append_msg.connect(self.__window.append_msg)
            level_idx = self.__window.logLevel.findText(self.__levelName)
            self.__window.logLevel.setCurrentIndex(level_idx)
        self.__window.show()
        self.__window.refresh(self.__buffer)

    def set_excludes(self, excludes):
        self.__excludes = excludes.split(',')
        if len(self.__excludes) > 0:
            old_buf = self.__buffer
            self.__buffer = RingBuffer(old_buf.maxlen, dtype=object)
            for m in old_buf:
                if any(e in m for e in self.__excludes):
                    pass
                else:
                    self.__buffer.append(m)
            if self.__window is not None:
                self.__window.refresh(self.__buffer)

    def set_size(self, size):
        '''
        Changes the size of the log cache.
        '''
        old_buf = self.__buffer
        self.__buffer = RingBuffer(size, dtype=object)
        self.__buffer.extend(old_buf)
        if self.__window is not None:
            self.__window.refresh(self.__buffer)

    def change_level(self, level):
        '''
        Change the root logger level.
        :param level: the new level name.
        '''
        logging.info(f"Changing log level from {self.__levelName} to {level}")
        self.__root.setLevel(level)
        self.__levelName = level
        self.__preferences.set(LOGGING_LEVEL, self.__levelName)


def to_millis(start, end, precision=1):
    '''
    Calculates the differences in time in millis.
    :param start: start time in seconds.
    :param end: end time in seconds.
    :return: delta in millis.
    '''
    return round((end - start) * 1000, precision)


class RingBuffer(Sequence):
    def __init__(self, capacity, dtype=np.float64):
        """
        Create a new ring buffer with the given capacity and element type

        Parameters
        ----------
        capacity: int
            The maximum capacity of the ring buffer
        dtype: data-type, optional
            Desired type of buffer elements. Use a type like (float, 2) to
            produce a buffer with shape (N, 2)
        """
        self.__buffer = np.empty(capacity, dtype)
        self.__left_idx = 0
        self.__right_idx = 0
        self.__capacity = capacity
        self.__event_count = 0

    def unwrap(self):
        """ Copy the data from this buffer into unwrapped form """
        return np.concatenate((
            self.__buffer[self.__left_idx:min(self.__right_idx, self.__capacity)],
            self.__buffer[:max(self.__right_idx - self.__capacity, 0)]
        ))

    def take_event_count(self, if_multiple=None):
        '''
        :param if_multiple: if set, only take the event count if it is a multiple of the supplied value.
        :return: the count of items added since the last take if the count is taken.
        '''
        count = self.__event_count
        if if_multiple is None or count % if_multiple == 0:
            self.__event_count = 0
            return count
        else:
            return None

    def _fix_indices(self):
        """
        Enforce our invariant that 0 <= self._left_index < self._capacity
        """
        if self.__left_idx >= self.__capacity:
            self.__left_idx -= self.__capacity
            self.__right_idx -= self.__capacity
        elif self.__left_idx < 0:
            self.__left_idx += self.__capacity
            self.__right_idx += self.__capacity

    @property
    def idx(self):
        return self.__left_idx, self.__right_idx

    @property
    def is_full(self):
        """ True if there is no more space in the buffer """
        return len(self) == self.__capacity

    # numpy compatibility
    def __array__(self):
        return self.unwrap()

    @property
    def dtype(self):
        return self.__buffer.dtype

    @property
    def shape(self):
        return (len(self),) + self.__buffer.shape[1:]

    @property
    def maxlen(self):
        return self.__capacity

    def append(self, value):
        if self.is_full:
            if not len(self):
                return
            else:
                self.__left_idx += 1

        self.__buffer[self.__right_idx % self.__capacity] = value
        self.__right_idx += 1
        self.__event_count += 1
        self._fix_indices()

    def peek(self):
        if len(self) == 0:
            return None
        idx = (self.__right_idx % self.__capacity) - 1
        logger.debug(f"Peeking at idx {idx}")
        res = self.__buffer[idx]
        return res

    def append_left(self, value):
        if self.is_full:
            if not len(self):
                return
            else:
                self.__right_idx -= 1

        self.__left_idx -= 1
        self._fix_indices()
        self.__buffer[self.__left_idx] = value
        self.__event_count += 1

    def extend(self, values):
        lv = len(values)
        if len(self) + lv > self.__capacity:
            if not len(self):
                return
        if lv >= self.__capacity:
            # wipe the entire array! - this may not be threadsafe
            self.__buffer[...] = values[-self.__capacity:]
            self.__right_idx = self.__capacity
            self.__left_idx = 0
            return

        ri = self.__right_idx % self.__capacity
        sl1 = np.s_[ri:min(ri + lv, self.__capacity)]
        sl2 = np.s_[:max(ri + lv - self.__capacity, 0)]
        self.__buffer[sl1] = values[:sl1.stop - sl1.start]
        self.__buffer[sl2] = values[sl1.stop - sl1.start:]
        self.__right_idx += lv

        self.__left_idx = max(self.__left_idx, self.__right_idx - self.__capacity)
        self.__event_count += len(values)
        self._fix_indices()

    def extend_left(self, values):
        lv = len(values)
        if len(self) + lv > self.__capacity:
            if not len(self):
                return
        if lv >= self.__capacity:
            # wipe the entire array! - this may not be threadsafe
            self.__buffer[...] = values[:self.__capacity]
            self.__right_idx = self.__capacity
            self.__left_idx = 0
            return

        self.__left_idx -= lv
        self._fix_indices()
        li = self.__left_idx
        sl1 = np.s_[li:min(li + lv, self.__capacity)]
        sl2 = np.s_[:max(li + lv - self.__capacity, 0)]
        self.__buffer[sl1] = values[:sl1.stop - sl1.start]
        self.__buffer[sl2] = values[sl1.stop - sl1.start:]

        self.__right_idx = min(self.__right_idx, self.__left_idx + self.__capacity)
        self.__event_count += len(values)

    def __len__(self):
        return self.__right_idx - self.__left_idx

    def __getitem__(self, item):
        # handle simple (b[1]) and basic (b[np.array([1, 2, 3])]) fancy indexing specially
        if not isinstance(item, tuple):
            item_arr = np.asarray(item)
            if issubclass(item_arr.dtype.type, np.integer):
                item_arr = (item_arr + self.__left_idx) % self.__capacity
                return self.__buffer[item_arr]

        # for everything else, get it right at the expense of efficiency
        return self.unwrap()[item]

    def __iter__(self):
        # alarmingly, this is comparable in speed to using itertools.chain
        return iter(self.unwrap())

    # Everything else
    def __repr__(self):
        return '<RingBuffer of {!r}>'.format(np.asarray(self))

