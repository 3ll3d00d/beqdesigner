import logging

from qtpy.QtCore import QByteArray, QObject, Signal, QRectF, QPoint, QTimer
from qtpy.QtGui import QBrush, QColor, QPalette
from qtpy.QtSvg import QSvgRenderer
from qtpy.QtSvgWidgets import QGraphicsSvgItem
from qtpy.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem, QWidget, QGraphicsSceneMouseEvent, \
    QGraphicsSceneContextMenuEvent, QApplication, QGraphicsSceneHoverEvent

logger = logging.getLogger('svg')


class NodeSignal(QObject):
    on_click = Signal(str, name='on_click')
    on_double_click = Signal(str, name='on_double_click')
    on_context = Signal(str, QPoint, name='on_context')


class SvgItem(QGraphicsSvgItem):

    def __init__(self, id, renderer, parent=None):
        super().__init__(parent)
        self.id = id
        self.setSharedRenderer(renderer)
        self.setElementId(id)
        bounds = renderer.boundsOnElement(id)
        self.setPos(bounds.topLeft())


class ClickableSvgItem(SvgItem):

    def __init__(self, id, renderer, signal: NodeSignal, node_name: str, parent=None):
        super().__init__(id, renderer, parent)
        self.__signal = signal
        self.__node_name = node_name
        self.__timer = QTimer(self)
        self.__timer.setSingleShot(True)
        self.__timer.timeout.connect(self.__on_single_click)
        self.__double_click_interval = QApplication.doubleClickInterval()
        self.setAcceptHoverEvents(True)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        event.accept()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        event.accept()
        if not self.__timer.isActive():
            self.__timer.start(self.__double_click_interval)
        else:
            self.__timer.stop()
            self.__on_double_click()

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        event.accept()
        self.__timer.stop()
        self.__on_double_click()

    def __on_single_click(self):
        self.__signal.on_click.emit(self.__node_name)

    def __on_double_click(self):
        self.__signal.on_double_click.emit(self.__node_name)

    def contextMenuEvent(self, event: QGraphicsSceneContextMenuEvent):
        self.__signal.on_context.emit(self.__node_name, event.screenPos())

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent):
        self.setToolTip(self.__node_name)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent):
        self.setToolTip('')


class SvgView(QGraphicsView):
    Native, OpenGL, Image = range(3)

    def __init__(self, parent=None):
        super(SvgView, self).__init__(parent)
        self.signal = NodeSignal()
        self.renderer = SvgView.Native
        self.__svg_items = []
        self.__wrapper_item = None
        self.__svg_renderer = QSvgRenderer()

        self.setScene(QGraphicsScene(self))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setViewport(QWidget())
        self.setBackgroundBrush(QBrush(QColor(QPalette().color(QPalette.ColorGroup.Active, QPalette.ColorRole.Window))))

    def render_bytes(self, svg_bytes: bytes):
        s = self.scene()
        s.clear()
        self.__svg_items = []
        self.resetTransform()

        self.__svg_renderer.load(QByteArray(svg_bytes))
        import xml.etree.ElementTree as ET
        g = "{http://www.w3.org/2000/svg}g"
        xml = svg_bytes.decode('utf-8')
        # logger.debug(xml)
        for i in ET.fromstring(xml).findall(f"./{g}/{g}"):
            if i.attrib['class'] == 'edge':
                item = SvgItem(i.attrib['id'], self.__svg_renderer)
            else:
                node_name = next((c.text for c in i if c.tag == '{http://www.w3.org/2000/svg}title'), None)
                if node_name:
                    # logger.debug(f"Adding clickable item for {node_name}")
                    item = ClickableSvgItem(i.attrib['id'], self.__svg_renderer, self.signal, node_name)
                else:
                    # logger.debug(f"Adding standard item for {i.attrib['id']}")
                    item = SvgItem(i.attrib['id'], self.__svg_renderer)
            item.setFlags(QGraphicsItem.GraphicsItemFlag.ItemClipsToShape)
            item.setCacheMode(QGraphicsItem.CacheMode.NoCache)
            item.setZValue(1)
            s.addItem(item)
        rect: QRectF = s.itemsBoundingRect()
        rect.adjust(-10, -10, 10, 10)
        s.setSceneRect(rect)

    def wheelEvent(self, event):
        factor = pow(1.2, event.angleDelta().y() / 240.0)
        self.scale(factor, factor)
        event.accept()
