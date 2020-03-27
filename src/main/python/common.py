import pyqtgraph as pg


class PlotWidgetWithDateAxis(pg.PlotWidget):
    def __init__(self, parent=None, background='default', **kargs):
        super().__init__(parent=parent,
                         background=background,
                         axisItems={'bottom': TimeAxisItem(orientation='bottom')},
                         **kargs)


class TimeAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        import datetime
        return [str(datetime.timedelta(seconds=value)).split('.')[0] for value in values]