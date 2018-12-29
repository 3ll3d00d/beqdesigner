import logging
import math
import os
import tempfile
from contextlib import contextmanager
from urllib.parse import urlparse

import matplotlib
import qtawesome as qta
import requests
from PIL import Image
from matplotlib.gridspec import GridSpec
from matplotlib.image import imread
from matplotlib.table import Table
from matplotlib.ticker import NullLocator
from qtpy.QtCore import Qt, QSize
from qtpy.QtWidgets import QDesktopWidget, QListWidgetItem, QDialog, QFileDialog, QDialogButtonBox, QMessageBox

from model.magnitude import MagnitudeModel
from model.preferences import REPORT_TITLE_FONT_SIZE, REPORT_IMAGE_ALPHA, REPORT_FILTER_ROW_HEIGHT_MULTIPLIER, \
    REPORT_FILTER_X0, REPORT_FILTER_X1, REPORT_FILTER_Y0, REPORT_FILTER_Y1, \
    REPORT_LAYOUT_MAJOR_RATIO, REPORT_LAYOUT_MINOR_RATIO, REPORT_CHART_GRID_ALPHA, REPORT_CHART_SHOW_LEGEND, \
    REPORT_GEOMETRY, REPORT_LAYOUT_SPLIT_DIRECTION, REPORT_LAYOUT_TYPE, REPORT_CHART_LIMITS_X0, \
    REPORT_CHART_LIMITS_X_SCALE, REPORT_CHART_LIMITS_X1, REPORT_FILTER_FONT_SIZE, REPORT_FILTER_SHOW_HEADER, \
    REPORT_GROUP, REPORT_LAYOUT_WSPACE, REPORT_LAYOUT_HSPACE
from ui.report import Ui_saveReportDialog

VALID_IMG_FORMATS = ['jpg', 'jpeg', 'png']

logger = logging.getLogger('report')


class SaveReportDialog(QDialog, Ui_saveReportDialog):
    '''
    Save Report dialog
    '''

    def __init__(self, parent, preferences, signal_model, filter_model, status_bar, selected_signal):
        super(SaveReportDialog, self).__init__(parent)
        self.__table = None
        self.__selected_signal = selected_signal
        self.__first_create = True
        self.__magnitude_model = None
        self.__imshow_axes = None
        self.__pixel_perfect_mode = False
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
        self.imagePicker.setIcon(qta.icon('fa5s.folder-open'))
        self.limitsButton.setIcon(qta.icon('fa5s.arrows-alt'))
        self.saveLayout.setIcon(qta.icon('fa5s.save'))
        self.loadURL.setIcon(qta.icon('fa5s.download'))
        self.loadURL.setEnabled(False)
        self.snapToImageSize.setIcon(qta.icon('fa5s.expand'))
        self.snapToImageSize.setEnabled(False)
        self.buttonBox.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.discard_layout)
        for xy in self.__xy_data:
            self.curves.addItem(QListWidgetItem(xy.name, self.curves))
        self.curves.selectAll()
        self.preview.canvas.mpl_connect('resize_event', self.__canvas_size_to_xy)
        # init fields
        self.__restore_geometry()
        self.restore_layout(redraw=True)
        self.__record_image_size()

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
        # have to clear the title first because the title can move from one axis to another (and clearing doesn't seem to remove that)
        self.set_title('')
        self.preview.canvas.figure.clear()
        image_spec, chart_spec, filter_spec = self.__calculate_layout()
        if image_spec is not None:
            self.__imshow_axes = self.preview.canvas.figure.add_subplot(image_spec)
            self.__init_imshow_axes()
        else:
            self.__imshow_axes = None
        self.__magnitude_model = MagnitudeModel('report', self.preview, self.__preferences, self, 'Signals',
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
        self.set_title(self.title.text(), draw=False)
        self.apply_image(draw=False)
        self.preview.canvas.draw_idle()
        self.__record_image_size()

    def on_redraw(self):
        '''
        fired when the magnitude chart redraws, basically means the limits have changed so we have to redraw the image
        if we have it in the chart to make sure the extents fit.
        '''
        if self.__imshow_axes is None and self.__image is not None:
            self.__image.set_extent(self.__make_extent(self.__magnitude_model.limits))

    def replace_table(self, *args, draw=True):
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
                tab = self.__magnitude_model.limits.axes_1.add_table(table)
                self.__magnitude_model.limits.axes_1.spines['top'].set_visible(False)
                self.__magnitude_model.limits.axes_1.spines['right'].set_visible(False)
                tab.set_zorder(1000)
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
            row_height = (
                    self.tableFontSize.value() / 72.0 * self.preview.canvas.figure.dpi / table_axes.bbox.height * multiplier)
            cell_kwargs['facecolor'] = fc

            table = Table(table_axes, **table_loc)
            table.set_zorder(1000)
            self.__add_filters_to_table(table, row_height, cell_kwargs)
            table.auto_set_font_size(False)
            table.set_fontsize(self.tableFontSize.value())
            return table
        return None

    def __add_filters_to_table(self, table, row_height, cell_kwargs):
        ''' renders the filters as a nicely formatted table '''
        cols = ('Freq', 'Gain', 'Q', 'Type', 'Total')
        col_width = 1 / len(cols)
        if 'bbox' in cell_kwargs:
            col_width *= cell_kwargs['bbox'][2]
        cells = [self.__table_print(f) for f in self.__filter_model]
        if self.__selected_signal is not None and not math.isclose(self.__selected_signal.offset, 0.0):
            cells.append(['', f"{self.__selected_signal.offset:+g}", '', 'MV', ''])
        show_header = self.showTableHeader.isChecked()
        ec = matplotlib.rcParams['axes.edgecolor']
        if show_header:
            for idx, label in enumerate(cols):
                cell = table.add_cell(0, idx, width=col_width, height=row_height, text=label, loc='center',
                                      edgecolor=ec, **cell_kwargs)
                cell.set_alpha(self.tableAlpha.value())
        if len(cells) > 0:
            for idx, row in enumerate(cells):
                for col_idx, cell in enumerate(row):
                    cell = table.add_cell(idx + (1 if show_header else 0), col_idx, width=col_width, height=row_height,
                                          text=cell, loc='center', edgecolor=ec, **cell_kwargs)
                    cell.PAD = 0.02
                    cell.set_alpha(self.tableAlpha.value())
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
        elif layout == "Pixel Perfect Image | Chart":
            gs = GridSpec(1, 1, wspace=self.widthSpacing.value(), hspace=self.heightSpacing.value())
            chart_spec = gs.new_subplotspec((0, 0), 1, 1)
            self.__prepare_for_pixel_perfect()
        return image_spec, chart_spec, filter_spec

    def __prepare_for_pixel_perfect(self):
        ''' puts the report into pixel perfect mode which means honour the image size. '''
        self.__pixel_perfect_mode = True
        self.widthSpacing.setValue(0.0)
        self.heightSpacing.setValue(0.0)
        self.__record_image_size()

    def __honour_image_aspect_ratio(self):
        ''' resizes the window to fit the image aspect ratio based on the chart size '''
        if len(self.image.text()) > 0:
            width, height = Image.open(self.image.text()).size
            width_delta = width - self.__x
            height_delta = height - self.__y
            if width_delta != 0 or height_delta != 0:
                win_size = self.size()
                target_size = QSize(win_size.width() + width_delta, win_size.height() + height_delta)
                available_size = QDesktopWidget().availableGeometry()
                if available_size.width() < target_size.width() or available_size.height() < target_size.height():
                    target_size.scale(available_size.width() - 48, available_size.height() - 48, Qt.KeepAspectRatio)
                self.resize(target_size)

    def __get_one_pane_two_pane_spec(self):
        '''
        :return: layout spec with two panes with 1 axes in the first pane and 2 in the other.
        '''
        major_ratio = [self.majorSplitRatio.value(), 1]
        minor_ratio = [self.minorSplitRatio.value(), 1]
        if self.chartSplit.currentText() == 'Horizontal':
            gs = GridSpec(2, 2, width_ratios=major_ratio, height_ratios=minor_ratio,
                          wspace=self.widthSpacing.value(), hspace=self.heightSpacing.value())
            spec_1 = gs.new_subplotspec((0, 0), 2, 1)
            spec_2 = gs.new_subplotspec((0, 1), 1, 1)
            spec_3 = gs.new_subplotspec((1, 1), 1, 1)
        else:
            gs = GridSpec(2, 2, width_ratios=minor_ratio, height_ratios=major_ratio,
                          wspace=self.widthSpacing.value(), hspace=self.heightSpacing.value())
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
            gs = GridSpec(2, 2, width_ratios=major_ratio, height_ratios=minor_ratio,
                          wspace=self.widthSpacing.value(), hspace=self.heightSpacing.value())
            spec_1 = gs.new_subplotspec((0, 0), 1, 1)
            spec_2 = gs.new_subplotspec((1, 0), 1, 1)
            spec_3 = gs.new_subplotspec((0, 1), 2, 1)
        else:
            gs = GridSpec(2, 2, width_ratios=minor_ratio, height_ratios=major_ratio,
                          wspace=self.widthSpacing.value(), hspace=self.heightSpacing.value())
            spec_1 = gs.new_subplotspec((0, 0), 1, 1)
            spec_2 = gs.new_subplotspec((0, 1), 1, 1)
            spec_3 = gs.new_subplotspec((1, 0), 2, 1)
        return spec_1, spec_2, spec_3

    def __get_two_pane_spec(self):
        '''
        :return: layout spec with two panes containing an axes in each.
        '''
        if self.chartSplit.currentText() == 'Horizontal':
            gs = GridSpec(1, 2, width_ratios=[self.majorSplitRatio.value(), 1],
                          wspace=self.widthSpacing.value(), hspace=self.heightSpacing.value())
            spec_1 = gs.new_subplotspec((0, 0), 1, 1)
            spec_2 = gs.new_subplotspec((0, 1), 1, 1)
        else:
            gs = GridSpec(2, 1, height_ratios=[self.majorSplitRatio.value(), 1],
                          wspace=self.widthSpacing.value(), hspace=self.heightSpacing.value())
            spec_1 = gs.new_subplotspec((0, 0), 1, 1)
            spec_2 = gs.new_subplotspec((1, 0), 1, 1)
        return spec_1, spec_2

    def __init_imshow_axes(self):
        if self.__imshow_axes is not None:
            if self.imageBorder.isChecked():
                self.__imshow_axes.axis('on')
                self.__imshow_axes.get_xaxis().set_major_locator(NullLocator())
                self.__imshow_axes.get_yaxis().set_major_locator(NullLocator())
            else:
                self.__imshow_axes.axis('off')

    def __table_print(self, filt):
        ''' formats the filter into a format suitable for rendering in the table '''
        header = self.showTableHeader.isChecked()
        vals = [str(filt.freq) if header else f"{filt.freq} Hz"] if hasattr(filt, 'freq') else ['']
        gain = filt.gain if hasattr(filt, 'gain') else 0
        g_suffix = ' dB' if gain != 0 and not header else ''
        vals.append(f"{gain:+g}{g_suffix}" if gain != 0 else vals.append(str('N/A')))
        vals.append(str(filt.q)) if hasattr(filt, 'q') else vals.append(str('N/A'))
        filter_type = filt.filter_type
        if len(filt) > 1:
            filter_type += f" x{len(filt)}"
        vals.append(filter_type)
        if gain != 0 and len(filt) > 1:
            vals.append(f"{len(filt)*gain:+g}{g_suffix}")
        else:
            vals.append('')
        return vals

    def getMagnitudeData(self, reference=None):
        ''' feeds the magnitude model with data '''
        return self.__selected_xy

    def set_selected(self):
        ''' Updates the selected curves and redraws. '''
        selected = [x.text() for x in self.curves.selectedItems()]
        self.__selected_xy = [x for x in self.__xy_data if x.name in selected]
        if self.__magnitude_model is not None:
            self.redraw()

    def set_title_size(self, _):
        ''' updates the title size '''
        self.set_title(self.title.text())

    def set_title(self, text, draw=True):
        ''' sets the title text '''
        if self.__imshow_axes is None:
            if self.__magnitude_model is not None:
                self.__magnitude_model.limits.axes_1.set_title(str(text), fontsize=self.titleFontSize.value())
        else:
            self.__imshow_axes.set_title(str(text), fontsize=self.titleFontSize.value())
        if draw:
            self.preview.canvas.draw_idle()

    def redraw(self):
        ''' triggers a redraw. '''
        self.__magnitude_model.redraw()

    def resizeEvent(self, resizeEvent):
        '''
        replaces the replace and updates axis sizes when the window resizes.
        :param resizeEvent: the event.
        '''
        super().resizeEvent(resizeEvent)
        self.replace_table()
        self.__record_image_size()

    def __record_image_size(self):
        ''' displays the image size on the form. '''
        if self.__image is None or self.__pixel_perfect_mode:
            self.imageWidthPixels.setValue(0)
            self.imageHeightPixels.setValue(0)
        else:
            width, height = get_ax_size(self.__image)
            self.imageWidthPixels.setValue(width)
            self.imageHeightPixels.setValue(height)

    def accept(self):
        ''' saves the report '''
        if self.__pixel_perfect_mode:
            self.__save_pixel_perfect()
        else:
            self.__save_report()

    def __save_report(self):
        ''' writes the figure to the specified format '''
        formats = "Report Files (*.png *.jpg *.jpeg)"
        file_name = QFileDialog.getSaveFileName(parent=self, caption='Export Report', filter=formats)
        if file_name:
            output_file = str(file_name[0]).strip()
            if len(output_file) == 0:
                return
            else:
                format = os.path.splitext(output_file)[1][1:].strip()
                if format in VALID_IMG_FORMATS:
                    scale_factor = self.widthPixels.value() / self.__x
                    from app import wait_cursor
                    with wait_cursor():
                        self.__status_bar.showMessage(f"Saving report to {output_file}", 5000)
                        self.preview.canvas.figure.savefig(output_file, format=format, dpi=self.__dpi * scale_factor,
                                                           pad_inches=0, bbox_inches='tight')
                        self.__status_bar.showMessage(f"Saved report to {output_file}", 5000)
                else:
                    msg_box = QMessageBox()
                    msg_box.setText(f"Invalid output file format - {output_file} is not one of {VALID_IMG_FORMATS}")
                    msg_box.setIcon(QMessageBox.Critical)
                    msg_box.setWindowTitle('Unexpected Error')
                    msg_box.exec()

    def __save_pixel_perfect(self):
        ''' saves an image based on passing the image through directly '''
        if len(self.image.text()) > 0:
            file_name = QFileDialog.getSaveFileName(parent=self, caption='Export Report',
                                                    filter='Report File (*.jpg *.png *.jpeg)')
            if file_name:
                output_file = str(file_name[0]).strip()
                if len(output_file) == 0:
                    return
                else:
                    format = os.path.splitext(output_file)[1][1:].strip()
                    if format in VALID_IMG_FORMATS:
                        from app import wait_cursor
                        with wait_cursor():
                            self.__status_bar.showMessage(f"Saving report to {output_file}", 5000)
                            self.preview.canvas.figure.savefig(output_file, format=format, dpi=self.__dpi)
                            if self.__concat_images(format, output_file):
                                self.__status_bar.showMessage(f"Saved report to {output_file}", 5000)
                    else:
                        msg_box = QMessageBox()
                        msg_box.setText(f"Invalid output file format - {output_file} is not one of {VALID_IMG_FORMATS}")
                        msg_box.setIcon(QMessageBox.Critical)
                        msg_box.setWindowTitle('Unexpected Error')
                        msg_box.exec()
        else:
            msg_box = QMessageBox()
            msg_box.setText('Unable to create report, no image selected')
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle('No Image')
            msg_box.exec()

    def __concat_images(self, format, output_file):
        ''' cats 2 images vertically '''
        im_image = Image.open(self.image.text())
        mp_image = Image.open(output_file)
        more_args = {}
        convert_to_rgb = False
        if format.lower() == 'jpg' or format.lower() == 'jpeg':
            more_args['subsampling'] = 0
            more_args['quality'] = 95
            if im_image.format.lower() != 'jpg' and im_image.format.lower() != 'jpeg':
                msg_box = QMessageBox()
                msg_box.setText(
                    f"Image format is {im_image.format}/{im_image.mode} but the desired output format is JPG<p/><p/>The image must be converted to RGB in order to proceed.")
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg_box.setWindowTitle('Do you want to convert?')
                decision = msg_box.exec()
                if decision == QMessageBox.Yes:
                    convert_to_rgb = True
                else:
                    return False
        if im_image.size[0] != mp_image.size[0]:
            new_height = round(im_image.size[1] * (mp_image.size[0] / im_image.size[0]))
            logger.debug(f"Resizing from {im_image.size} to match {mp_image.size}, new height {new_height}")
            im_image = im_image.resize((mp_image.size[0], new_height), Image.LANCZOS)
        if convert_to_rgb:
            im_image = im_image.convert('RGB')
        final_image = Image.new(im_image.mode, (im_image.size[0], im_image.size[1] + mp_image.size[1]))
        final_image.paste(im_image, (0, 0))
        final_image.paste(mp_image, (0, im_image.size[1]))
        final_image.save(output_file, **more_args)
        return True

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
        self.__preferences.set(REPORT_FILTER_SHOW_HEADER, self.showTableHeader.isChecked())
        self.__preferences.set(REPORT_FILTER_FONT_SIZE, self.tableFontSize.value())
        self.__preferences.set(REPORT_LAYOUT_HSPACE, self.heightSpacing.value())
        self.__preferences.set(REPORT_LAYOUT_WSPACE, self.widthSpacing.value())
        if self.__magnitude_model is not None:
            self.__preferences.set(REPORT_CHART_LIMITS_X0, self.__magnitude_model.limits.x_min)
            self.__preferences.set(REPORT_CHART_LIMITS_X1, self.__magnitude_model.limits.x_max)
            self.__preferences.set(REPORT_CHART_LIMITS_X_SCALE, self.__magnitude_model.limits.x_scale)

    def discard_layout(self):
        '''
        Discards the stored layout in the preferences.
        '''
        self.__preferences.clear_all(REPORT_GROUP)
        self.restore_layout(redraw=True)

    def restore_layout(self, redraw=False):
        '''
        Restores the saved layout.
        :param redraw: if true, also redraw the report.
        '''
        self.filterRowHeightMultiplier.blockSignals(True)
        self.filterRowHeightMultiplier.setValue(self.__preferences.get(REPORT_FILTER_ROW_HEIGHT_MULTIPLIER))
        if self.__first_create:
            self.filterRowHeightMultiplier.setValue(self.filterRowHeightMultiplier.value() * 1.85)
        self.filterRowHeightMultiplier.blockSignals(False)
        self.titleFontSize.setValue(self.__preferences.get(REPORT_TITLE_FONT_SIZE))
        with block_signals(self.tableFontSize):
            self.tableFontSize.setValue(self.__preferences.get(REPORT_FILTER_FONT_SIZE))
        with block_signals(self.showTableHeader):
            self.showTableHeader.setChecked(self.__preferences.get(REPORT_FILTER_SHOW_HEADER))
        self.imageOpacity.setValue(self.__preferences.get(REPORT_IMAGE_ALPHA))
        self.x0.setValue(self.__preferences.get(REPORT_FILTER_X0))
        self.x1.setValue(self.__preferences.get(REPORT_FILTER_X1))
        self.y0.setValue(self.__preferences.get(REPORT_FILTER_Y0))
        self.y1.setValue(self.__preferences.get(REPORT_FILTER_Y1))
        self.majorSplitRatio.setValue(self.__preferences.get(REPORT_LAYOUT_MAJOR_RATIO))
        with block_signals(self.minorSplitRatio):
            self.minorSplitRatio.setValue(self.__preferences.get(REPORT_LAYOUT_MINOR_RATIO))
        self.__restore_combo(REPORT_LAYOUT_SPLIT_DIRECTION, self.chartSplit)
        self.__restore_combo(REPORT_LAYOUT_TYPE, self.chartLayout)
        with block_signals(self.gridOpacity):
            self.gridOpacity.setValue(self.__preferences.get(REPORT_CHART_GRID_ALPHA))
        with block_signals(self.showLegend):
            self.showLegend.setChecked(self.__preferences.get(REPORT_CHART_SHOW_LEGEND))
        with block_signals(self.widthSpacing):
            self.widthSpacing.setValue(self.__preferences.get(REPORT_LAYOUT_WSPACE))
        with block_signals(self.heightSpacing):
            self.heightSpacing.setValue(self.__preferences.get(REPORT_LAYOUT_HSPACE))
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
                                            filter='Images (*.png *.jpeg *.jpg)')
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
            self.nativeImageWidth.setValue(im.shape[1])
            self.nativeImageHeight.setValue(im.shape[0])
            self.snapToImageSize.setEnabled(True)
            if self.__imshow_axes is None:
                if self.__magnitude_model is not None and not self.__pixel_perfect_mode:
                    extent = self.__make_extent(self.__magnitude_model.limits)
                    self.__image = self.__magnitude_model.limits.axes_1.imshow(im, extent=extent,
                                                                               alpha=self.imageOpacity.value())
            else:
                self.__image = self.__imshow_axes.imshow(im, alpha=self.imageOpacity.value())
        else:
            self.snapToImageSize.setEnabled(False)
            if self.__imshow_axes is None:
                if self.__image is not None:
                    pass
            else:
                self.nativeImageWidth.setValue(0)
                self.nativeImageHeight.setValue(0)
                self.__imshow_axes.clear()
                self.__init_imshow_axes()
        self.set_title(self.title.text())
        if draw:
            self.preview.canvas.draw_idle()
            self.__record_image_size()

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

    def set_image_border(self):
        ''' adds or removes the image border '''
        if self.__imshow_axes is not None:
            self.__init_imshow_axes()
            self.preview.canvas.draw_idle()
            self.__record_image_size()

    def set_grid_opacity(self, value):
        ''' updates the grid alpha '''
        if self.__magnitude_model is not None:
            self.__magnitude_model.limits.axes_1.grid(alpha=value)
            self.preview.canvas.draw_idle()

    def update_image_url(self, text):
        ''' changes the icon to open if the url is valid '''
        if len(text) > 0:
            o = urlparse(text)
            if len(o.scheme) > 0 and len(o.netloc) > 0:
                self.loadURL.setEnabled(True)
                if self.loadURL.signalsBlocked():
                    self.loadURL.setIcon(qta.icon('fa5s.download'))
                    self.loadURL.blockSignals(False)

    def load_image_from_url(self):
        ''' attempts to download the image and sets it as the file name '''
        tmp_image = self.__download_image()
        if tmp_image is not None:
            self.loadURL.setIcon(qta.icon('fa5s.check', color='green'))
            self.loadURL.blockSignals(True)
            self.image.setText(tmp_image)
            self.apply_image()
        else:
            self.loadURL.setIcon(qta.icon('fa5s.times', color='red'))

    def __download_image(self):
        '''
        Attempts to download the image.
        :return: the filename containing the downloaded data.
        '''
        url_file_type = None
        try:
            url_file_type = os.path.splitext(self.imageURL.text())[1].strip()
        except:
            pass
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=url_file_type)
        with open(tmp_file.name, 'wb') as f:
            try:
                f.write(requests.get(self.imageURL.text()).content)
                name = tmp_file.name
            except:
                logger.exception(f"Unable to download {self.imageURL.text()}")
                tmp_file.delete = True
                name = None
        return name

    def snap_to_image_size(self):
        ''' Snaps the dialog size to the image size. '''
        self.__honour_image_aspect_ratio()


@contextmanager
def block_signals(widget):
    '''
    blocks signals on a given widget
    :param widget: the widget.
    '''
    try:
        widget.blockSignals(True)
        yield
    finally:
        widget.blockSignals(False)


def get_ax_size(ax):
    '''
    :param ax: the axis.
    :return: width, height in pixels.
    '''
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= ax.figure.dpi
    height *= ax.figure.dpi
    return width, height
