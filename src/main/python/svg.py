import logging

from qtpy.QtCore import QByteArray, QObject, Signal, QRectF, QPoint
from qtpy.QtGui import QBrush, QColor, QPalette
from qtpy.QtSvg import QGraphicsSvgItem, QSvgRenderer
from qtpy.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem, QWidget, QGraphicsSceneMouseEvent, \
    QGraphicsSceneContextMenuEvent

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

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.__signal.on_click.emit(self.__node_name)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.__signal.on_double_click.emit(self.__node_name)

    def contextMenuEvent(self, event: QGraphicsSceneContextMenuEvent):
        self.__signal.on_context.emit(self.__node_name, event.screenPos())


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
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setViewport(QWidget())
        self.setBackgroundBrush(QBrush(QColor(QPalette().color(QPalette.Active, QPalette.Window))))

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
            item.setFlags(QGraphicsItem.ItemClipsToShape)
            item.setCacheMode(QGraphicsItem.NoCache)
            item.setZValue(0)
            s.addItem(item)
        rect: QRectF = s.itemsBoundingRect()
        rect.adjust(-10, -10, 10, 10)
        s.setSceneRect(rect)

    def wheelEvent(self, event):
        factor = pow(1.2, event.angleDelta().y() / 240.0)
        self.scale(factor, factor)
        event.accept()
