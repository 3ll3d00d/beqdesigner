from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from qtpy import QtWidgets


# Matplotlib canvas class to create figure
class MplCanvas(Canvas):
    def __init__(self):
        self.figure = Figure(tight_layout=True)
        Canvas.__init__(self, self.figure)
        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
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
