import logging

import requests
from qtpy.QtCore import Signal, QRunnable, QObject

logger = logging.getLogger('checker')

RELEASE_API = 'https://api.github.com/repos/3ll3d00d/beqdesigner/releases/latest'


class VersionSignals(QObject):
    on_old_version = Signal(str, str, name='on_old_version')
    on_error = Signal(str, name='on_error')


class VersionChecker(QRunnable):
    def __init__(self, on_old_handler, on_error_handler, current_version):
        super().__init__()
        self.__signals = VersionSignals()
        self.__version = current_version
        self.__signals.on_old_version.connect(on_old_handler)
        self.__signals.on_error.connect(on_error_handler)

    def run(self):
        '''
        Hit the Github release API and compare the latest to the current version
        if the new version is later then emit a signal
        '''
        try:
            r = requests.get(RELEASE_API)
            if r.status_code == 200:
                latest = r.json()
                if latest:
                    latest_tag = latest['tag_name']
                    if self.__is_new(latest_tag):
                        download_url = f"https://github.com/3ll3d00d/beqdesigner/releases/{latest_tag}"
                        self.__signals.on_old_version.emit(latest_tag, download_url)
            else:
                self.__signals.on_error.emit(f"Unable to hit Github Release API at: \n\n {RELEASE_API}")
        except:
            self.__signals.on_error.emit(f"Unable to contact Github Release API at: \n\n {RELEASE_API}")

    def __is_new(self, new_version):
        try:
            new_vers = [int(x) for x in new_version.split('.')]
            old_vers = [int(x) for x in self.__version.split('.')]
            return new_vers > old_vers
        except Exception as e:
            logger.exception(f"Unable to compare releases {new_version} to {self.__version}", e)
            return True
