'''
The Qt side of `model.minidsp`: pushing a filter to a connected minidsp with minidsp-rs on the thread pool, and picking a minidsp
XML file to load as a filter. `model.minidsp` itself is Qt-free (the headless pipeline writes XML with it).
'''
import logging
from typing import Callable, List, Optional, Tuple

from qtpy.QtCore import QObject, QRunnable, Signal
from qtpy.QtWidgets import QFileDialog

from model.iir import Biquad
from model.minidsp import load_filter_file
from model.preferences import BEQ_DOWNLOAD_DIR

logger = logging.getLogger('minidsp')


def load_as_filter(parent, preferences, fs, unroll=False) -> Tuple[Optional[List[Biquad]], Optional[str]]:
    '''
    allows user to select a minidsp xml file and load it as a filter.
    '''
    selected = QFileDialog.getOpenFileName(parent=parent, directory=preferences.get(BEQ_DOWNLOAD_DIR),
                                           caption='Load Minidsp XML Filter', filter='Filter (*.xml)')
    filt_file = selected[0] if selected is not None else None
    if filt_file is not None and len(filt_file) > 0:
        return load_filter_file(filt_file, fs, unroll=unroll), filt_file
    return None, None


class FilterPublisherSignals(QObject):
    ON_START: int = 1
    ON_COMPLETE: int = 0
    ON_ERROR: int = 2

    on_status = Signal(int, name='on_status')


class FilterPublisher(QRunnable):

    def __init__(self, filt: List[Biquad], slot: Optional[int], minidsp_rs_binary: str, minidsp_rs_options: str,
                 status_handler: Callable[[int], None]):
        super().__init__()
        self.__signals = FilterPublisherSignals()
        self.__slot = slot
        self.__filt = filt
        self.__signals.on_status.connect(status_handler)
        self.__signals.on_status.emit(FilterPublisherSignals.ON_START)
        from plumbum import local
        cmd = local[minidsp_rs_binary]
        if minidsp_rs_options:
            self.__runner = cmd[minidsp_rs_options.split(' ')]
        else:
            self.__runner = cmd

    def run(self):
        try:
            if self.__slot:
                self.__send_config()
            for c in range(2):
                idx = 0
                for f in self.__filt:
                    for bq in f.format_biquads(True, separator='|', show_index=False):
                        coeffs = bq.split('|')
                        if len(coeffs) != 5:
                            raise ValueError(f"Invalid coeff count {len(coeffs)} at idx {idx}")
                        else:
                            self.__send_biquad(str(c), str(idx), coeffs)
                            idx += 1
                for i in range(idx, 10):
                    self.__send_bypass(str(c), str(i), True)
            self.__signals.on_status.emit(FilterPublisherSignals.ON_COMPLETE)
        except Exception as e:
            logger.exception(f"Unexpected failure during filter publication")
            self.__signals.on_status.emit(FilterPublisherSignals.ON_ERROR)

    def __send_config(self):
        # minidsp config <slot>
        cmd = self.__runner['config', str(self.__slot)]
        logger.info(f"Executing {cmd}")
        cmd.run(timeout=5)

    def __send_biquad(self, channel: str, idx: str, coeffs: List[str]):
        # minidsp input <channel> peq <index> set -- <b0> <b1> <b2> <a1> <a2>
        cmd = self.__runner['input', channel, 'peq', idx, 'set', '--', coeffs]
        logger.info(f"Executing {cmd}")
        cmd.run(timeout=5)
        self.__send_bypass(channel, idx, False)

    def __send_bypass(self, channel: str, idx: str, bypass: bool):
        # minidsp input <channel> bypass on
        cmd = self.__runner['input', channel, 'peq', idx, 'bypass', 'on' if bypass else 'off']
        logger.info(f"Executing {cmd}")
        cmd.run(timeout=5)
