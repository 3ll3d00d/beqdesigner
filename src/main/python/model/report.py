import math

import matplotlib
import qtawesome as qta
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.image import imread
from matplotlib.table import Table
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
        self.__first_create = True
        self.__magnitude_model = None
        self.__imshow_axes = None
        self.__filter_axes = None
        self.__image = None
        self.__dpi = None
        self.__x = None
        self.__y = None
        self.__aspect_ratio = None
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
        self.filterFontSize.setValue(18)
        self.titleFontSize.setValue(36)
        for xy in self.__xy_data:
            self.curves.addItem(QListWidgetItem(xy.name, self.curves))
        self.__canvas_size_to_xy()
        self.redraw_all_axes()

    def __canvas_size_to_xy(self):
        '''
        Calculates the current size of the image.
        '''
        self.__dpi = self.preview.canvas.figure.dpi
        self.__x, self.__y = self.preview.canvas.figure.get_size_inches() * self.preview.canvas.figure.dpi
        self.__aspect_ratio = self.__x / self.__y
        self.widthPixels.setValue(self.__x)
        self.heightPixels.setValue(self.__y)

    def redraw_all_axes(self):
        ''' Draws all charts. '''
        self.preview.canvas.figure.clear()
        self.preview.canvas.figure.tight_layout()
        image_spec, chart_spec, filter_spec = self.__calculate_layout()
        if image_spec is not None:
            self.__imshow_axes = self.preview.canvas.figure.add_subplot(image_spec)
            self.__init_imshow_axes()
        else:
            self.__imshow_axes = None
        self.__magnitude_model = MagnitudeModel('main', self.preview, self.__preferences, self, 'Signals',
                                                show_legend=lambda: self.showLegend.isChecked(),
                                                subplot_spec=chart_spec, redraw_listener=self.on_redraw)
        self.__filter_axes = self.preview.canvas.figure.add_subplot(filter_spec)
        self.__replace_table()
        self.set_title()
        self.apply_image(draw=False)
        self.preview.canvas.draw_idle()

    def on_redraw(self):
        '''
        fired when the magnitude chart redraws, basically means the limits have changed so we have to redraw the image
        if we have it in the chart to make sure the extents fit.
        '''
        if self.__imshow_axes is None and self.__image is not None:
            self.__image.set_extent(self.__make_extent(self.__magnitude_model.limits))

    def __replace_table(self):
        ''' Adds the table to the axis '''
        self.__filter_axes.clear()
        self.__filter_axes.axis('off')
        self.__filter_axes.add_table(self.__make_table())

    def __make_table(self):
        '''
        Transforms the filter model into a table. This code is based on the code in matplotlib.table
        :return: the table.
        '''
        fc = self.__magnitude_model.limits.axes_1.get_facecolor()
        table = Table(self.__filter_axes, loc='center')
        table.edges = 'closed'
        font_size = self.filterFontSize.value()
        # this is some hackery around the way the matplotlib table works
        multiplier = 1.2 * 1.85 if not self.__first_create else 1.2
        self.__first_create = False
        height = (font_size / 72.0 * self.preview.canvas.figure.dpi / self.__filter_axes.bbox.height * multiplier)
        font = FontProperties()
        font.set_size(font_size)
        cols = ('Type', 'Freq', 'Q', 'Gain')
        for idx, label in enumerate(cols):
            table.add_cell(0, idx, width=1 / len(cols), height=height, text=label, facecolor=fc, loc='center',
                           edgecolor=matplotlib.rcParams['axes.edgecolor'], fontproperties=font)
        cells = [self.__format_filter(f) for f in self.__filter_model]
        if len(cells) > 0:
            for idx, row in enumerate(cells):
                for col_idx, cell in enumerate(row):
                    table.add_cell(idx + 1, col_idx, width=1 / len(cols), height=height, text=cell, facecolor=fc,
                                   loc='center', edgecolor=matplotlib.rcParams['axes.edgecolor'], fontproperties=font)
        return table

    def __calculate_layout(self):
        '''
        Creates the subplot specs for the chart layout, options are

        3 panes, horizontal, split right

        -------------------
        |        |        |
        |        |--------|
        |        |        |
        -------------------

        3 panes, horizontal, split left

        -------------------
        |        |        |
        |--------|        |
        |        |        |
        -------------------

        3 panes, vertical, split bottom

        -------------------
        |                 |
        |-----------------|
        |        |        |
        -------------------

        3 panes, vertical, split top

        -------------------
        |        |        |
        |-----------------|
        |                 |
        -------------------

        2 panes, vertical

        -------------------
        |                 |
        |-----------------|
        |                 |
        -------------------

        2 panes, horizontal

        -------------------
        |        |        |
        |        |        |
        |        |        |
        -------------------

        :return: image_spec, chart_spec, filter_spec
        '''
        layout = self.chartLayout.currentText()
        filter_spec = None
        image_spec = None
        chart_spec = None
        if layout == "Image | Chart, Filters":
            image_spec, chart_spec, filter_spec = self.__get_one_pane_two_pane_spec()
        elif layout == "Image | Filters, Chart":
            image_spec, filter_spec, chart_spec = self.__get_one_pane_two_pane_spec()
        elif layout == "Chart | Image, Filter":
            chart_spec, image_spec, filter_spec = self.__get_one_pane_two_pane_spec()
        elif layout == "Chart | Filters, Image":
            chart_spec, filter_spec, image_spec = self.__get_one_pane_two_pane_spec()
        elif layout == "Filters | Image, Chart":
            filter_spec, image_spec, chart_spec = self.__get_one_pane_two_pane_spec()
        elif layout == "Filters | Chart, Image":
            filter_spec, chart_spec, image_spec = self.__get_one_pane_two_pane_spec()
        elif layout == 'Image, Filters | Chart':
            image_spec, filter_spec, chart_spec = self.__get_two_pane_one_pane_spec()
        elif layout == 'Filters, Image | Chart':
            filter_spec, image_spec, chart_spec = self.__get_two_pane_one_pane_spec()
        elif layout == 'Chart, Image | Filters':
            chart_spec, image_spec, filter_spec = self.__get_two_pane_one_pane_spec()
        elif layout == 'Image, Chart | Filters':
            image_spec, chart_spec, filter_spec = self.__get_two_pane_one_pane_spec()
        elif layout == 'Filters, Chart | Image':
            filter_spec, chart_spec, image_spec = self.__get_two_pane_one_pane_spec()
        elif layout == 'Chart, Filters | Image':
            chart_spec, filter_spec, image_spec = self.__get_two_pane_one_pane_spec()
        elif layout == "Chart | Filters":
            chart_spec, filter_spec = self.__get_two_pane_spec()
        elif layout == "Filters | Chart":
            filter_spec, chart_spec = self.__get_two_pane_spec()
        return image_spec, chart_spec, filter_spec

    def __get_one_pane_two_pane_spec(self):
        '''
        :return: layout spec with two panes with 1 axes in the first pane and 2 in the other.
        '''
        major_ratio = [self.majorSplitRatio.value(), 1]
        minor_ratio = [self.minorSplitRatio.value(), 1]
        if self.chartSplit.currentText() == 'Horizontal':
            gs = GridSpec(2, 2, width_ratios=major_ratio, height_ratios=minor_ratio)
            spec_1 = gs.new_subplotspec((0, 0), 2, 1)
            spec_2 = gs.new_subplotspec((0, 1), 1, 1)
            spec_3 = gs.new_subplotspec((1, 1), 1, 1)
        else:
            gs = GridSpec(2, 2, width_ratios=minor_ratio, height_ratios=major_ratio)
            spec_1 = gs.new_subplotspec((0, 0), 1, 2)
            spec_2 = gs.new_subplotspec((1, 0), 1, 1)
            spec_3 = gs.new_subplotspec((1, 1), 1, 1)
        return spec_1, spec_2, spec_3

    def __get_two_pane_one_pane_spec(self):
        '''
        :return: layout spec with two panes with 2 axes in the first pane and 1 in the other.
        '''
        major_ratio = [self.majorSplitRatio.value(), 1]
        minor_ratio = [self.minorSplitRatio.value(), 1]
        if self.chartSplit.currentText() == 'Horizontal':
            gs = GridSpec(2, 2, width_ratios=major_ratio, height_ratios=minor_ratio)
            spec_1 = gs.new_subplotspec((0, 0), 1, 1)
            spec_2 = gs.new_subplotspec((1, 0), 1, 1)
            spec_3 = gs.new_subplotspec((0, 1), 2, 1)
        else:
            gs = GridSpec(2, 2, width_ratios=minor_ratio, height_ratios=major_ratio)
            spec_1 = gs.new_subplotspec((0, 0), 1, 1)
            spec_2 = gs.new_subplotspec((0, 1), 1, 1)
            spec_3 = gs.new_subplotspec((1, 0), 2, 1)
        return spec_1, spec_2, spec_3

    def __get_two_pane_spec(self):
        '''
        :return: layout spec with two panes containing an axes in each.
        '''
        if self.chartSplit.currentText() == 'Horizontal':
            gs = GridSpec(1, 2, width_ratios=[self.majorSplitRatio.value(), 1])
            spec_1 = gs.new_subplotspec((0, 0), 1, 1)
            spec_2 = gs.new_subplotspec((0, 1), 1, 1)
        else:
            gs = GridSpec(2, 1, height_ratios=[self.majorSplitRatio.value(), 1])
            spec_1 = gs.new_subplotspec((0, 0), 1, 1)
            spec_2 = gs.new_subplotspec((1, 0), 1, 1)
        return spec_1, spec_2

    def __init_imshow_axes(self):
        self.__imshow_axes.axis('off')

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
        if self.__imshow_axes is None:
            if self.__magnitude_model is not None:
                self.__magnitude_model.limits.axes_1.set_title(str(self.title.text()),
                                                               fontsize=self.titleFontSize.value())
        else:
            self.__imshow_axes.set_title(str(self.title.text()), fontsize=self.titleFontSize.value())
        self.preview.canvas.draw_idle()

    def redraw(self):
        '''
        triggers a redraw.
        '''
        self.__magnitude_model.redraw()

    def resizeEvent(self, resizeEvent):
        super().resizeEvent(resizeEvent)
        self.__canvas_size_to_xy()
        self.__replace_table()

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
        self.heightPixels.setValue(int(math.floor(new_width / self.__aspect_ratio)))

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
            if self.__imshow_axes is None:
                if self.__magnitude_model is not None:
                    extent = self.__make_extent(self.__magnitude_model.limits)
                    self.__image = self.__magnitude_model.limits.axes_1.imshow(im, extent=extent)
            else:
                self.__image = self.__imshow_axes.imshow(im)
        else:
            if self.__imshow_axes is None:
                if self.__image is not None:
                    pass
            else:
                self.__imshow_axes.clear()
                self.__init_imshow_axes()
        self.set_title()
        if draw:
            self.preview.canvas.draw_idle()

    def __make_extent(self, limits):
        return (limits.x_min, limits.x_max, limits.y1_min, limits.y1_max)

    def set_table_font_size(self, size):
        ''' changes the size of the font in the table '''
        if self.__filter_axes is not None:
            self.__replace_table()
            self.preview.canvas.draw_idle()

    def show_limits(self):
        ''' Show the limits dialog '''
        if self.__magnitude_model is not None:
            self.__magnitude_model.show_limits()
