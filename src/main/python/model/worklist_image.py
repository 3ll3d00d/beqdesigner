"""An aspect-preserving image widget shared by the title page's preview tabs."""
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QLabel, QSizePolicy


class ScaledImage(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._original = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(1, 1)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def set_image(self, image):
        self._original = image
        if image.isNull():
            self.clear()
        self._scale()

    def _scale(self):
        if not self._original.isNull():
            self.setPixmap(self._original.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale()
