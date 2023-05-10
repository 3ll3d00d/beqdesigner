import logging
from qtawesome import Spin

logger = logging.getLogger('spin')


def stop_spinner(spinner, button):
    '''
    Stops the qtawesome spinner safely.
    :param spinner: the spinner.
    :param button: the button that owns the spinner.
    '''
    if spinner is not None:
        if isinstance(spinner, StoppableSpin):
            spinner.stop()
        else:
            if button in spinner.info:
                spinner.info[button][0].stop()


class StoppableSpin(Spin):

    def __init__(self, parent_widget, name):
        Spin.__init__(self, parent_widget, interval=25, step=5)
        self.__stopped = False
        self.__name = name

    def _update(self):
        if self.__stopped is False:
            super(StoppableSpin, self)._update()
        else:
            logger.debug(f"Ignoring update for stopped spinner - {self.__name}")

    def setup(self, icon_painter, painter, rect):
        if self.__stopped is True:
            logger.debug(f"Ignoring setup for stopped spinner - {self.__name}")
        else:
            logger.debug(f"Setting up spinner {self.__name} (has timer? {self.parent_widget in self.info})")
            super(StoppableSpin, self).setup(icon_painter, painter, rect)

    def stop(self):
        if self.__stopped is False:
            logger.debug(f"Stopping spinner {self.__name}")
            self.__stopped = True
            if self.parent_widget in self.info:
                self.info[self.parent_widget][0].stop()
            else:
                logger.debug(f"Unable to stop spinner - {self.__name}")
        else:
            logger.debug(f"Ignoring duplicate stop for spinner - {self.__name}")
