from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from qtpy import QtWidgets
from qtpy.QtWidgets import QProxyStyle, QStyle


# Matplotlib canvas class to create figure
class MplCanvas(Canvas):
    def __init__(self):
        self.figure = Figure(tight_layout=True)
        Canvas.__init__(self, self.figure)
        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        Canvas.updateGeometry(self)


# Matplotlib widget
class MplWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.canvas = MplCanvas()
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.setContentsMargins(6, 0, 6, 0)
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)


# this seems like a) a massive hack, and b) the only way to avoid a windows only rendering flaw (bad selection box in each cell)
class NoCaretStyle(QProxyStyle):

    def drawControl(self, element, option, painter, widget=None):
        if element == QStyle.ControlElement.CE_ItemViewItem:
            # Completely override item rendering to avoid Windows-specific caret
            painter.save()

            # Draw selection background if selected
            if option.state & QStyle.StateFlag.State_Selected:
                painter.fillRect(option.rect, option.palette.highlight())

            # Draw text without any selection state
            text_rect = option.rect.adjusted(4, 0, -4, 0)  # Add some padding
            text_color = option.palette.highlightedText() if (option.state & QStyle.StateFlag.State_Selected) else option.palette.text()
            painter.setPen(text_color.color())
            painter.drawText(text_rect, int(option.displayAlignment), option.text)

            painter.restore()
            return

        super().drawControl(element, option, painter, widget)
