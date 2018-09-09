import logging

from PyQt5.QtGui import QRegExpValidator
from qtpy.QtCore import QEvent, Qt, QRegExp
from qtpy.QtWidgets import QItemDelegate, QStyledItemDelegate, QLineEdit

logger = logging.getLogger('delegates')


class CheckBoxDelegate(QItemDelegate):
    """
    A delegate that places a fully functioning QCheckBox cell of the column to which it's applied & which propagates
    state changes to the owning measurement model.
    from https://stackoverflow.com/questions/17748546/pyqt-column-of-checkboxes-in-a-qtableview

    YOU MUST ONLY CREATE ONE OF THESE PER VIEW OTHERWISE Qt CRASHES
    """

    def __init__(self):
        QItemDelegate.__init__(self)

    def createEditor(self, parent, option, index):
        """
        Important, otherwise an editor is created if the user clicks in this cell.
        """
        return None

    def paint(self, painter, option, index):
        """
        Paint a checkbox without the label.
        """
        data = index.data()
        if data is None:
            logger.error(f"No data found at {index.row()}, {index.column()}")
            data = 0
        self.drawCheck(painter, option, option.rect, Qt.Unchecked if int(data) == 0 else Qt.Checked)

    def editorEvent(self, event, model, option, index):
        '''
        Change the data in the model and the state of the checkbox
        if the user presses the left mousebutton and this cell is editable. Otherwise do nothing.
        '''
        if not int(index.flags() & Qt.ItemIsEditable) > 0:
            return False

        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            model.toggle(index)
            return True

        return False


class RegexValidator(QStyledItemDelegate):
    ''' Validates the input against the regex '''

    def __init__(self, regex):
        QStyledItemDelegate.__init__(self)
        self.__regex = regex

    def createEditor(self, widget, option, index):
        if not index.isValid():
            return 0
        if index.column() == 0:  # only on the cells in the first column
            editor = QLineEdit(widget)
            validator = QRegExpValidator(QRegExp(self.__regex), editor)
            editor.setValidator(validator)
            return editor
        return super(RegexValidator, self).createEditor(widget, option, index)
