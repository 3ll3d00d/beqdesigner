import math

import matplotlib
import qtawesome as qta
from matplotlib.gridspec import GridSpec
from matplotlib.image import imread
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog, QFileDialog, QListWidgetItem

from model.magnitude import MagnitudeModel
from ui.report import Ui_saveReportDialog


class SaveReportDialog(QDialog, Ui_saveReportDialog):
    '''
    Save Report dialog
    '''

    def __init__(self, parent, preferences, signal_model, filter_model, status_bar):
        super(SaveReportDialog, self).__init__(parent)
        self.__table = None
        self.__magnitude_model = None
        self.__imshow_axes = None
        self.__tab_axes = None
        self.__signal_model = signal_model
        self.__filter_model = filter_model
        self.__preferences = preferences
        self.__status_bar = status_bar
        self.__xy_data = self.__signal_model.get_all_magnitude_data()
        self.__selected_xy = []
        self.setWindowFlags(self.windowFlags() | Qt.WindowSystemMenuHint | Qt.WindowMinMaxButtonsHint)
        self.setupUi(self)
        self.imagePicker.setIcon(qta.icon('fa.folder-open-o'))
        self.limitsButton.setIcon(qta.icon('ei.move'))
        self.filterFontSize.setValue(int(matplotlib.rcParams['font.size']))
        self.titleFontSize.setValue(12)
        for xy in self.__xy_data:
            self.curves.addItem(QListWidgetItem(xy.name, self.curves))
        self.__dpi = self.preview.canvas.figure.dpi
        self.__x, self.__y = self.preview.canvas.figure.get_size_inches() * self.preview.canvas.figure.dpi
        self.__aspectRatio = self.__x / self.__y
        self.widthPixels.setValue(self.__x)
        self.heightPixels.setValue(self.__y)
        self.redraw_all_axes()

    def redraw_all_axes(self):
        ''' Draws all charts. '''
        self.preview.canvas.figure.clear()
        self.preview.canvas.figure.tight_layout()
        if self.imageIsBackground.isChecked():
            self.imageRatio.setEnabled(False)
            self.filterLocationY.setEnabled(False)
            self.__imshow_axes = None
            if self.filterLocationX.currentText() == 'Left':
                gs = GridSpec(1, 2, width_ratios=[1, 3])
                table_spec = gs.new_subplotspec((0, 0), 1, 1)
                chart_spec = gs.new_subplotspec((0, 1), 1, 1)
            else:
                gs = GridSpec(1, 2, width_ratios=[3, 1])
                table_spec = gs.new_subplotspec((0, 1), 1, 1)
                chart_spec = gs.new_subplotspec((0, 0), 1, 1)
        else:
            self.imageRatio.setEnabled(True)
            self.filterLocationY.setEnabled(True)
            if self.filterLocationY.currentText() == 'Top':
                if self.filterLocationX.currentText() == 'Left':
                    gs = GridSpec(2, 2, width_ratios=[1, 3], height_ratios=[self.imageRatio.value(), 1])
                    table_spec = gs.new_subplotspec((0, 0), 1, 1)
                    image_spec = gs.new_subplotspec((0, 1), 1, 1)
                    chart_spec = gs.new_subplotspec((1, 0), 1, 2)
                else:
                    gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[self.imageRatio.value(), 1])
                    table_spec = gs.new_subplotspec((0, 1), 1, 1)
                    image_spec = gs.new_subplotspec((0, 0), 1, 1)
                    chart_spec = gs.new_subplotspec((1, 0), 1, 2)
            else:
                if self.filterLocationX.currentText() == 'Left':
                    gs = GridSpec(2, 2, width_ratios=[1, 3], height_ratios=[self.imageRatio.value(), 1])
                    image_spec = gs.new_subplotspec((0, 0), 1, 2)
                    table_spec = gs.new_subplotspec((1, 0), 1, 1)
                    chart_spec = gs.new_subplotspec((1, 1), 1, 1)
                else:
                    gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[self.imageRatio.value(), 1])
                    image_spec = gs.new_subplotspec((0, 0), 1, 2)
                    table_spec = gs.new_subplotspec((1, 1), 1, 1)
                    chart_spec = gs.new_subplotspec((1, 0), 1, 1)
            self.__imshow_axes = self.preview.canvas.figure.add_subplot(image_spec)
            self.__init_imshow_axes()

        self.__magnitude_model = MagnitudeModel('main', self.preview, self.__preferences, self, 'Signals',
                                                show_legend=lambda: self.showLegend.isChecked(),
                                                subplot_spec=chart_spec)
        self.__tab_axes = self.preview.canvas.figure.add_subplot(table_spec)
        self.__tab_axes.axis('off')
        self.__tab_axes.axis('tight')
        col_labels = ('Type', 'Freq', 'Q', 'Gain')
        cells = [self.__format_filter(f) for f in self.__filter_model]
        if len(cells) == 0:
            cells = [''] * len(col_labels)
        self.__table = self.__tab_axes.table(cellText=cells, colLabels=col_labels, loc='center')
        self.__table.auto_set_font_size(False)
        self.__table.set_fontsize(self.filterFontSize.value())
        fc = self.__magnitude_model.limits.axes_1.get_facecolor()
        cells = self.__table.get_celld()
        for cell in cells.values():
            cell.set_facecolor(fc)
            cell.set_edgecolor(matplotlib.rcParams['axes.edgecolor'])
        self.set_title()
        self.apply_image(draw=False)
        self.preview.canvas.draw_idle()

    def __init_imshow_axes(self):
        self.__imshow_axes.axis('off')
        # self.__imshow_axes.axis('tight')

    def __format_filter(self, filt):
        vals = [str(filt.filter_type), str(filt.freq)]
        vals.append(str(filt.q)) if hasattr(filt, 'q') else vals.append(str('N/A'))
        vals.append(str(filt.gain)) if hasattr(filt, 'gain') else vals.append(str('N/A'))
        return vals

    def getMagnitudeData(self, reference=None):
        return self.__selected_xy

    def set_selected(self):
        '''
        Updates the selected curves and redraws.
        '''
        selected = [x.text() for x in self.curves.selectedItems()]
        self.__selected_xy = [x for x in self.__xy_data if x.name in selected]
        self.redraw()

    def set_title(self):
        ''' sets the title text '''
        if self.imageIsBackground.isChecked():
            if self.__magnitude_model is not None:
                self.__magnitude_model.limits.axes_1.set_title(str(self.title.text()),
                                                               fontsize=self.titleFontSize.value())
        else:
            if self.__imshow_axes is not None:
                self.__imshow_axes.set_title(str(self.title.text()), fontsize=self.titleFontSize.value())
        self.preview.canvas.draw_idle()

    def redraw(self):
        '''
        triggers a redraw.
        '''
        self.__magnitude_model.redraw()

    def accept(self):
        formats = "Portable Network Graphic (*.png)"
        file_name = QFileDialog(parent=self).getSaveFileName(self, 'Export Report', 'report.png', formats)
        if file_name:
            output_file = str(file_name[0]).strip()
            if len(output_file) == 0:
                return
            else:
                scale_factor = self.widthPixels.value() / self.__x
                self.preview.canvas.figure.savefig(output_file, format='png', dpi=self.__dpi * scale_factor)
                self.__status_bar.showMessage(f"Saved report to {output_file}", 5000)
        QDialog.accept(self)

    def update_height(self, new_width):
        '''
        Updates the height as the width changes according to the aspect ratio.
        :param new_width: the new width.
        '''
        self.heightPixels.setValue(int(math.floor(new_width / self.__aspectRatio)))

    def choose_image(self):
        '''
        Pick an image and display it.
        '''
        image = QFileDialog.getOpenFileName(parent=self, caption='Choose Image',
                                            filter='Portable Network Graphic (*.png)')
        img_file = image[0] if image is not None and len(image) > 0 else None
        self.image.setText(img_file)
        self.apply_image()

    def apply_image(self, draw=True):
        '''
        Applies the image to the current charts.
        :param draw: true if we should redraw.
        '''
        img_file = self.image.text()
        if len(img_file) > 0:
            im = imread(img_file)
            if self.imageIsBackground.isChecked():
                self.__magnitude_model.limits.axes_1.imshow(im, aspect='equal')
            else:
                self.__imshow_axes.imshow(im, aspect='equal')
        else:
            if self.imageIsBackground.isChecked():
                pass
            else:
                self.__imshow_axes.clear()
                self.__init_imshow_axes()
        self.set_title()
        if draw:
            self.preview.canvas.draw_idle()

    def set_table_font_size(self, size):
        ''' changes the size of the font in the table '''
        if self.__table is not None:
            self.__table.auto_set_font_size(False)
            self.__table.set_fontsize(size)
            self.preview.canvas.draw_idle()

    def show_limits(self):
        ''' Show the limits dialog '''
        if self.__magnitude_model is not None:
            self.__magnitude_model.show_limits()
