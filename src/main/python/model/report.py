import math

import matplotlib
import qtawesome as qta
from matplotlib.gridspec import GridSpec
from matplotlib.image import imread
from matplotlib.table import Table
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QListWidgetItem, QDialog, QFileDialog, QDialogButtonBox

from model.magnitude import MagnitudeModel
from model.preferences import REPORT_TITLE_FONT_SIZE, REPORT_IMAGE_ALPHA, REPORT_FILTER_ROW_HEIGHT_MULTIPLIER, \
    REPORT_FILTER_X0, REPORT_FILTER_X1, REPORT_FILTER_Y0, REPORT_FILTER_Y1, \
    REPORT_LAYOUT_MAJOR_RATIO, REPORT_LAYOUT_MINOR_RATIO, REPORT_CHART_GRID_ALPHA, REPORT_CHART_SHOW_LEGEND, \
    REPORT_GEOMETRY, REPORT_LAYOUT_SPLIT_DIRECTION, REPORT_LAYOUT_TYPE, REPORT_CHART_LIMITS_X0, \
    REPORT_CHART_LIMITS_X_SCALE, REPORT_CHART_LIMITS_X1
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
        self.saveLayout.setIcon(qta.icon('fa.floppy-o'))
        self.buttonBox.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.discard_layout)
        for xy in self.__xy_data:
            self.curves.addItem(QListWidgetItem(xy.name, self.curves))
        self.preview.canvas.mpl_connect('resize_event', self.__canvas_size_to_xy)
        # init fields
        self.__restore_geometry()
        self.restore_layout(redraw=True)
        self.filterRowHeightMultiplier.setValue(self.__preferences.get(REPORT_FILTER_ROW_HEIGHT_MULTIPLIER))

    def __restore_geometry(self):
        ''' loads the saved window size '''
        geometry = self.__preferences.get(REPORT_GEOMETRY)
        if geometry is not None:
            self.restoreGeometry(geometry)

    def __canvas_size_to_xy(self, event):
        '''
        Calculates the current size of the image.
        '''
        self.__dpi = self.preview.canvas.figure.dpi
        self.__x = event.width
        self.__y = event.height
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
                                                subplot_spec=chart_spec, redraw_listener=self.on_redraw,
                                                grid_alpha=self.gridOpacity.value(),
                                                x_min_pref_key=REPORT_CHART_LIMITS_X0,
                                                x_max_pref_key=REPORT_CHART_LIMITS_X1,
                                                x_scale_pref_key=REPORT_CHART_LIMITS_X_SCALE)
        if filter_spec is not None:
            self.__filter_axes = self.preview.canvas.figure.add_subplot(filter_spec)
            self.filterRowHeightMultiplier.setEnabled(True)
            self.x0.setEnabled(False)
            self.x1.setEnabled(False)
            self.y0.setEnabled(False)
            self.y1.setEnabled(False)
        else:
            self.__filter_axes = None
            self.filterRowHeightMultiplier.setEnabled(False)
            self.x0.setEnabled(True)
            self.x1.setEnabled(True)
            self.y0.setEnabled(True)
            self.y1.setEnabled(True)
        self.replace_table(draw=False)
        self.set_title(draw=False)
        self.apply_image(draw=False)
        self.preview.canvas.draw_idle()

    def on_redraw(self):
        '''
        fired when the magnitude chart redraws, basically means the limits have changed so we have to redraw the image
        if we have it in the chart to make sure the extents fit.
        '''
        if self.__imshow_axes is None and self.__image is not None:
            self.__image.set_extent(self.__make_extent(self.__magnitude_model.limits))

    def replace_table(self, draw=True):
        ''' Adds the table to the axis '''
        table = self.__make_table()
        if table is not None:
            if self.__filter_axes is not None:
                self.__filter_axes.clear()
                self.__filter_axes.axis('off')
                self.__filter_axes.add_table(table)
                self.__magnitude_model.limits.axes_1.spines['top'].set_visible(True)
                self.__magnitude_model.limits.axes_1.spines['right'].set_visible(True)
            elif self.__magnitude_model is not None:
                for t in self.__magnitude_model.limits.axes_1.tables:
                    t.remove()
                self.__magnitude_model.limits.axes_1.add_table(table)
                self.__magnitude_model.limits.axes_1.spines['top'].set_visible(False)
                self.__magnitude_model.limits.axes_1.spines['right'].set_visible(False)
            if draw:
                self.preview.canvas.draw_idle()

    def __make_table(self):
        '''
        Transforms the filter model into a table. This code is based on the code in matplotlib.table
        :return: the table.
        '''
        if len(self.__filter_model) > 0 and self.__magnitude_model is not None:
            fc = self.__magnitude_model.limits.axes_1.get_facecolor()
            cell_kwargs = {}

            if self.__filter_axes is not None:
                table_axes = self.__filter_axes
                table_loc = {'loc': 'center'}
            else:
                table_axes = self.__magnitude_model.limits.axes_1
                table_loc = {'bbox': (self.x0.value(), self.y0.value(),
                                      self.x1.value() - self.x0.value(), self.y1.value() - self.y0.value())}
            # this is some hackery around the way the matplotlib table works
            # multiplier = 1.2 * 1.85 if not self.__first_create else 1.2
            multiplier = self.filterRowHeightMultiplier.value()
            self.__first_create = False
            row_height = (matplotlib.rcParams[
                              'font.size'] / 72.0 * self.preview.canvas.figure.dpi / table_axes.bbox.height * multiplier)
            cell_kwargs['facecolor'] = fc

            table = Table(table_axes, **table_loc)
            self.__add_filters_to_table(table, row_height, cell_kwargs)
            return table
        return None

    def __add_filters_to_table(self, table, row_height, cell_kwargs):
        ''' renders the filters as a nicely formatted table '''
        cols = ('Freq', 'Gain (dB)', 'Q', 'Type', '')
        col_width = 1 / len(cols)
        if 'bbox' in cell_kwargs:
            col_width *= cell_kwargs['bbox'][2]
        cells = [self.__table_print(f) for f in self.__filter_model]
        for idx, label in enumerate(cols):
            cell = table.add_cell(0, idx, width=col_width, height=row_height, text=label, loc='center',
                                  edgecolor=matplotlib.rcParams['axes.edgecolor'], **cell_kwargs)
            if idx == 0:
                cell.visible_edges = 'LTB'
            elif idx == len(cols) - 1:
                cell.visible_edges = 'RTB'
            else:
                cell.visible_edges = 'TB'
        if len(cells) > 0:
            for idx, row in enumerate(cells):
                for col_idx, cell in enumerate(row):
                    cell = table.add_cell(idx + 1, col_idx, width=col_width, height=row_height, text=cell,
                                          loc='center', edgecolor=matplotlib.rcParams['axes.edgecolor'], **cell_kwargs)
                    cell.PAD = 0.02
                    edges = 'B'
                    if col_idx == 0:
                        edges += 'L'
                    elif col_idx == len(cols) - 1:
                        edges += 'R'
                    cell.visible_edges = edges
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

        1 pane

        -------------------
        |                 |
        |                 |
        |                 |
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
        elif layout == "Chart | Image":
            chart_spec, image_spec = self.__get_two_pane_spec()
        elif layout == "Image | Chart":
            image_spec, chart_spec = self.__get_two_pane_spec()
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

    def __pretty_print(self, filt):
        ''' formats the filter into a format suitable for rendering as a string '''
        txt = f"{filt.freq} Hz "
        if hasattr(filt, 'gain'):
            if filt.gain > 0:
                txt += '+'
            txt += f"{filt.gain}dB "
        if hasattr(filt, 'q'):
            txt += f"Q {filt.q} "
        txt += f"{filt.filter_type} "
        if len(filt) > 1:
            txt += f" x{len(filt)} (Total: {filt.gain * len(filt)} db)"
        return txt

    def __table_print(self, filt):
        ''' formats the filter into a format suitable for rendering in the table '''
        vals = [str(filt.freq)]
        gain = filt.gain if hasattr(filt, 'gain') else 0
        vals.append(str(gain)) if gain != 0 else vals.append(str('N/A'))
        vals.append(str(filt.q)) if hasattr(filt, 'q') else vals.append(str('N/A'))
        vals.append(str(filt.filter_type))
        if len(filt) > 1:
            vals.append(f"x{len(filt)}")
        else:
            vals.append('')
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

    def set_title(self, draw=True):
        ''' sets the title text '''
        if self.__imshow_axes is None:
            if self.__magnitude_model is not None:
                self.__magnitude_model.limits.axes_1.set_title(str(self.title.text()),
                                                               fontsize=self.titleFontSize.value())
        else:
            self.__imshow_axes.set_title(str(self.title.text()), fontsize=self.titleFontSize.value())
        if draw:
            self.preview.canvas.draw_idle()

    def redraw(self):
        '''
        triggers a redraw.
        '''
        self.__magnitude_model.redraw()

    def resizeEvent(self, resizeEvent):
        super().resizeEvent(resizeEvent)
        self.replace_table()

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

    def closeEvent(self, QCloseEvent):
        ''' Stores the window size on close '''
        self.__preferences.set(REPORT_GEOMETRY, self.saveGeometry())
        super().closeEvent(QCloseEvent)

    def save_layout(self):
        '''
        Saves the layout in the preferences.
        '''
        self.__preferences.set(REPORT_FILTER_ROW_HEIGHT_MULTIPLIER, self.filterRowHeightMultiplier.value())
        self.__preferences.set(REPORT_TITLE_FONT_SIZE, self.titleFontSize.value())
        self.__preferences.set(REPORT_IMAGE_ALPHA, self.imageOpacity.value())
        self.__preferences.set(REPORT_FILTER_X0, self.x0.value())
        self.__preferences.set(REPORT_FILTER_X1, self.x1.value())
        self.__preferences.set(REPORT_FILTER_Y0, self.y0.value())
        self.__preferences.set(REPORT_FILTER_Y1, self.y1.value())
        self.__preferences.set(REPORT_LAYOUT_MAJOR_RATIO, self.majorSplitRatio.value())
        self.__preferences.set(REPORT_LAYOUT_MINOR_RATIO, self.minorSplitRatio.value())
        self.__preferences.set(REPORT_LAYOUT_SPLIT_DIRECTION, self.chartSplit.currentText())
        self.__preferences.set(REPORT_LAYOUT_TYPE, self.chartLayout.currentText())
        self.__preferences.set(REPORT_CHART_GRID_ALPHA, self.gridOpacity.value())
        self.__preferences.set(REPORT_CHART_SHOW_LEGEND, self.showLegend.isChecked())
        if self.__magnitude_model is not None:
            self.__preferences.set(REPORT_CHART_LIMITS_X0, self.__magnitude_model.limits.x_min)
            self.__preferences.set(REPORT_CHART_LIMITS_X1, self.__magnitude_model.limits.x_max)
            self.__preferences.set(REPORT_CHART_LIMITS_X_SCALE, self.__magnitude_model.limits.x_scale)

    def discard_layout(self):
        '''
        Discards the stored layout in the preferences.
        '''
        self.__preferences.clear(REPORT_FILTER_ROW_HEIGHT_MULTIPLIER)
        self.__preferences.clear(REPORT_TITLE_FONT_SIZE)
        self.__preferences.clear(REPORT_IMAGE_ALPHA)
        self.__preferences.clear(REPORT_FILTER_X0)
        self.__preferences.clear(REPORT_FILTER_X1)
        self.__preferences.clear(REPORT_FILTER_Y0)
        self.__preferences.clear(REPORT_FILTER_Y1)
        self.__preferences.clear(REPORT_LAYOUT_MAJOR_RATIO)
        self.__preferences.clear(REPORT_LAYOUT_MINOR_RATIO)
        self.__preferences.clear(REPORT_LAYOUT_TYPE)
        self.__preferences.clear(REPORT_LAYOUT_SPLIT_DIRECTION)
        self.__preferences.clear(REPORT_CHART_GRID_ALPHA)
        self.__preferences.clear(REPORT_CHART_SHOW_LEGEND)
        self.__preferences.clear(REPORT_CHART_LIMITS_X0)
        self.__preferences.clear(REPORT_CHART_LIMITS_X1)
        self.__preferences.clear(REPORT_CHART_LIMITS_X_SCALE)
        self.restore_layout(redraw=True)

    def restore_layout(self, redraw=False):
        '''
        Restores the saved layout.
        :param redraw: if true, also redraw the report.
        '''
        self.filterRowHeightMultiplier.setValue(self.__preferences.get(REPORT_FILTER_ROW_HEIGHT_MULTIPLIER))
        if self.__first_create:
            self.filterRowHeightMultiplier.setValue(self.filterRowHeightMultiplier.value() * 1.85)
        self.titleFontSize.setValue(self.__preferences.get(REPORT_TITLE_FONT_SIZE))
        self.imageOpacity.setValue(self.__preferences.get(REPORT_IMAGE_ALPHA))
        self.x0.setValue(self.__preferences.get(REPORT_FILTER_X0))
        self.x1.setValue(self.__preferences.get(REPORT_FILTER_X1))
        self.y0.setValue(self.__preferences.get(REPORT_FILTER_Y0))
        self.y1.setValue(self.__preferences.get(REPORT_FILTER_Y1))
        self.majorSplitRatio.setValue(self.__preferences.get(REPORT_LAYOUT_MAJOR_RATIO))
        self.minorSplitRatio.setValue(self.__preferences.get(REPORT_LAYOUT_MINOR_RATIO))
        self.__restore_combo(REPORT_LAYOUT_SPLIT_DIRECTION, self.chartSplit)
        self.__restore_combo(REPORT_LAYOUT_TYPE, self.chartLayout)
        self.gridOpacity.setValue(self.__preferences.get(REPORT_CHART_GRID_ALPHA))
        self.showLegend.setChecked(self.__preferences.get(REPORT_CHART_SHOW_LEGEND))
        if redraw:
            self.redraw_all_axes()

    def __restore_combo(self, key, combo):
        value = self.__preferences.get(key)
        if value is not None:
            idx = combo.findText(value)
            if idx > -1:
                combo.setCurrentIndex(idx)

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
                                            filter='Images (*.png, *.jpeg, *.jpg)')
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
                    self.__image = self.__magnitude_model.limits.axes_1.imshow(im, extent=extent,
                                                                               alpha=self.imageOpacity.value())
            else:
                self.__image = self.__imshow_axes.imshow(im, alpha=self.imageOpacity.value())
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
        return limits.x_min, limits.x_max, limits.y1_min, limits.y1_max

    def set_table_font_size(self, size):
        ''' changes the size of the font in the table '''
        if self.__filter_axes is not None:
            self.replace_table()

    def show_limits(self):
        ''' Show the limits dialog '''
        if self.__magnitude_model is not None:
            self.__magnitude_model.show_limits()

    def set_image_opacity(self, value):
        ''' updates the image alpha '''
        if self.__image is not None:
            self.__image.set_alpha(value)
            self.preview.canvas.draw_idle()

    def set_grid_opacity(self, value):
        ''' updates the grid alpha '''
        if self.__magnitude_model is not None:
            self.__magnitude_model.limits.axes_1.grid(alpha=value)
            self.preview.canvas.draw_idle()
