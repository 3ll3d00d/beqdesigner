import typing

import qtawesome as qta
from qtpy.QtCore import QAbstractTableModel, QModelIndex, Qt
from qtpy.QtWidgets import QDialog, QPushButton, QHeaderView
from sortedcontainers import SortedDict, SortedSet

from ui.delegates import CheckBoxDelegate
from ui.link import Ui_linkSignalDialog


class LinkedSignalsModel:
    '''
    Converts the signal model into a grid of master/slave signals. Masters are rows, slaves are columns.
    '''

    def __init__(self, signal_model):
        self.__signal_model = signal_model
        self.rows = self.__convert(signal_model)
        self.row_keys = self.rows.keys()
        self.names = [s.name for s in signal_model]
        self.columns = SortedSet([s for s in self.names if s not in self.row_keys])
        self.table = None

    def __getitem__(self, i):
        return self.rows[self.row_keys[i]]

    def __len__(self):
        return len(self.rows)

    def is_slave(self, row_idx, col_idx):
        '''
        True if the entry at the specified cell is a slave.
        :param row_idx: the row index.
        :param col_idx: the column index.
        :return: True if that cell is a slave.
        '''
        master = self[row_idx]
        slave = self.columns[col_idx]
        return slave in master

    def toggle(self, row_idx, col_idx):
        '''
        Toggles the status at the given QModelIndex.
        :param row_idx: the row index.
        :param col_idx: the column index.
        '''
        master = self[row_idx]
        slave = self.columns[col_idx]
        if slave in master:
            master.remove(slave)
        else:
            for vals in self.rows.values():
                if slave in vals:
                    vals.remove(slave)
            master.append(slave)

    def __convert(self, signal_model):
        '''
        converts a signal_model into a list of master -> slave signal names.
        :param signal_model: the model to convert.
        :return: a dict.
        '''
        return SortedDict({s.name: [l.name for l in s.slaves] for s in signal_model if len(s.slaves) > 0})

    def save(self):
        pass

    def make_master(self, name):
        '''
        Moves the named signal from the slave list to the master list.
        :param name: the name.
        '''
        if name not in self.rows:
            # unslave it from any master
            for k, v in self.rows.items():
                if name in v:
                    v.remove(name)
            # promote to master
            if name not in self.rows:
                self.rows[name] = []
            # remove it from the columns
            self.columns.remove(name)

    def remove_master(self, name):
        '''
        Removes the named master.
        :param name: the name.
        :return Boolean: true if the master was removed.
        '''
        if name in self.rows:
            del self.rows[name]
            self.columns.add(name)
            return True
        return False


class LinkedSignalsTableModel(QAbstractTableModel):
    '''
    A Qt table model to feed the linked signals view.
    '''

    def __init__(self, model, parent=None):
        super().__init__(parent=parent)
        self.__model = model
        self.__model.table = self

    def rowCount(self, parent=None):
        return len(self.__model)

    def columnCount(self, parent=None):
        return len(self.__model.columns) + 2

    def flags(self, idx):
        flags = super().flags(idx)
        if idx.column() > 1:
            flags |= Qt.ItemIsEditable
        return flags

    def data(self, index: QModelIndex, role: int = ...) -> typing.Any:
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        else:
            if index.column() == 0:
                return None
            elif index.column() == 1:
                return self.__model.row_keys[index.row()]
            else:
                return self.__model.is_slave(index.row(), index.column() - 2)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...) -> typing.Any:
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if section == 0:
                return ''
            elif section == 1:
                return 'Master'
            else:
                if section - 2 < len(self.__model.columns):
                    return self.__model.columns[section - 2]
                else:
                    return None
        return None

    def delegate_to_checkbox(self, view):
        delegate = CheckBoxDelegate()
        for x in range(0, len(self.__model.columns)):
            view.setItemDelegateForColumn(x + 2, delegate)

    def toggle(self, idx):
        '''
        Toggles the model at the given index.
        :param idx: the index.
        '''
        self.__model.toggle(idx.row(), idx.column() - 2)
        self.dataChanged.emit(QModelIndex(), QModelIndex())


class LinkSignalsDialog(QDialog, Ui_linkSignalDialog):
    '''
    Alows user to link signals to the selected master.
    '''

    def __init__(self, signal_model, parent=None):
        super(LinkSignalsDialog, self).__init__(parent=parent)
        self.setupUi(self)
        self.__model = LinkedSignalsModel(signal_model)
        self.__signal_model = signal_model
        self.__table_model = LinkedSignalsTableModel(self.__model, parent=parent)
        self.linkSignals.setModel(self.__table_model)
        self.linkSignals.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.masterCandidates.addItem('')
        for x in self.__model.columns:
            self.masterCandidates.addItem(x)
        self.__table_model.delegate_to_checkbox(self.linkSignals)
        self.addToMaster.setIcon(qta.icon('fa5s.plus'))
        self.__manage_add_master_state()
        self.__make_delete_buttons()

    def addMaster(self):
        idx = self.masterCandidates.currentIndex()
        if idx > 0:
            self.__table_model.beginResetModel()
            # make the master
            master_name = self.masterCandidates.currentText()
            self.__model.make_master(master_name)
            # remove the new master from the dropdown
            self.masterCandidates.removeItem(idx)
            self.__manage_add_master_state()
            # trigger the update
            self.__table_model.endResetModel()
            # have to add this after the table is updated otherwise the index does not yet exist
            self.__make_delete_buttons()

    def __manage_add_master_state(self):
        '''
        Ensures the dropdown is enabled if there are multiple slaves left.
        '''
        multiple_slaves = len(self.__model.columns) > 1
        self.masterCandidates.setEnabled(multiple_slaves)
        self.addToMaster.setEnabled(multiple_slaves)

    def __make_delete_buttons(self):
        for idx, master_name in enumerate(self.__model.row_keys):
            remove_master_button = QPushButton()
            remove_master_button.setIcon(qta.icon('fa5s.trash'))
            remove_master_button.clicked.connect(self.remove_master(master_name))
            remove_master_button.setAutoFillBackground(True)
            self.linkSignals.setIndexWidget(self.__table_model.index(idx, 0), remove_master_button)

    def remove_master(self, name):
        '''
        :param name: the master name.
        :return: a function which will remove the master.
        '''

        def do_remove_master():
            self.__table_model.beginResetModel()
            if self.__model.remove_master(name):
                self.masterCandidates.addItem(name)
            self.__table_model.endResetModel()
            self.__make_delete_buttons()
            self.__manage_add_master_state()

        return do_remove_master

    def accept(self):
        '''
        Applies the changes to the master/slaves.
        '''
        self.__signal_model.free_all()
        for master_name, slaves in self.__model.rows.items():
            self.__signal_model.enslave(master_name, slaves)
        QDialog.accept(self)
