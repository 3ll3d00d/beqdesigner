import logging

import requests
from qtpy.QtCore import Signal, QRunnable, QObject

logger = logging.getLogger('checker')

RELEASE_API = 'https://api.github.com/repos/3ll3d00d/beqdesigner/releases'
LATEST_RELEASE_API = f"{RELEASE_API}/latest"


class VersionSignals(QObject):
    on_old_version = Signal(str, str, name='on_old_version')
    on_error = Signal(str, name='on_error')


class VersionChecker(QRunnable):
    def __init__(self, beta, on_old_handler, on_error_handler, current_version):
        super().__init__()
        self.__check_beta = beta
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
            latest = self.__get_latest_tag() if self.__check_beta is True else self.__get_latest_release_tag()
            if latest:
                self.alert_if_new(latest)
        except:
            self.__signals.on_error.emit(f"Unable to contact Github Release API at: \n\n {RELEASE_API}")

    def __get_latest_release_tag(self):
        r = requests.get(LATEST_RELEASE_API)
        if r.status_code == 200:
            latest = r.json()
            if latest:
                import semver
                return semver.parse_version_info(latest['tag_name'])
        else:
            self.__signals.on_error.emit(f"Unable to hit Github Release API at: \n\n {LATEST_RELEASE_API}")
        return None

    def __get_latest_tag(self):
        import semver
        r = requests.get(RELEASE_API)
        if r.status_code == 200:
            latest = r.json()
            if latest:
                return sorted([semver.parse_version_info(t['tag_name']) for t in latest])[-1]
        else:
            self.__signals.on_error.emit(f"Unable to hit Github Release API at: \n\n {LATEST_RELEASE_API}")
        return None

    def alert_if_new(self, latest_tag):
        if self.__is_new(latest_tag) > 0:
            download_url = f"https://github.com/3ll3d00d/beqdesigner/releases/{latest_tag}"
            self.__signals.on_old_version.emit(f"{latest_tag}", download_url)

    def __is_new(self, new_version):
        try:
            import semver
            return new_version > semver.parse_version_info(self.__version)
        except Exception as e:
            logger.exception(f"Unable to compare releases {new_version} to {self.__version}", e)
            return True
