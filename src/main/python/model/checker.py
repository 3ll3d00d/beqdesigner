import logging
import platform

import requests
from qtpy.QtCore import Signal, QRunnable, QObject, QAbstractTableModel, QModelIndex, Qt, QVariant
from qtpy.QtWidgets import QDialog

from ui.newversion import Ui_newVersionDialog

logger = logging.getLogger('checker')

RELEASE_API = 'https://api.github.com/repos/3ll3d00d/beqdesigner/releases'
ISSUES_API = 'https://api.github.com/repos/3ll3d00d/beqdesigner/issues'


class VersionSignals(QObject):
    on_old_version = Signal(list, dict, name='on_old_version')
    on_error = Signal(str, name='on_error')


class VersionChecker(QRunnable):

    def __init__(self, beta, on_old_handler, on_error_handler, current_version, signal_anyway=False):
        super().__init__()
        self.__check_beta = beta
        self.__signals = VersionSignals()
        self.__version = current_version
        self.__signals.on_old_version.connect(on_old_handler)
        self.__signals.on_error.connect(on_error_handler)
        self.__signal_anyway = signal_anyway

    def run(self):
        '''
        Hit the Github release API and compare the latest to the current version
        if the new version is later then emit a signal
        '''
        try:
            versions = self.__get_versions()
            if versions[0]['current'] is False or self.__signal_anyway is True:
                self.__signals.on_old_version.emit(versions, self.__load_issues())
        except:
            logger.exception('Failed to hit GitHub API')
            self.__signals.on_error.emit(f"Unable to contact Github Release API at: \n\n {RELEASE_API}")

    @staticmethod
    def __load_issues():
        r = requests.get(ISSUES_API, params={'sort': 'updated', 'state': 'closed', 'per_page': '100'})
        if r.status_code == 200:
            issues = r.json()
            if issues:
                return {i['number']: {'title': i['title'], 'url': i['html_url']} for i in issues}
        return {}

    def __get_versions(self):
        r = requests.get(RELEASE_API)
        if r.status_code == 200:
            releases = r.json()
            if releases:
                return sorted([v for v in [self.__convert(r, self.__version) for r in releases if r['draft'] is False]
                               if self.__check_beta is True or v['version'].modifier_type is None],
                              key=lambda v: v['version'],
                              reverse=True)
        else:
            self.__signals.on_error.emit(f"Unable to hit Github Release API at: \n\n {RELEASE_API}")
        return None

    @staticmethod
    def __convert(release, current_version):
        from awesomeversion import AwesomeVersion
        return {
            'tag': release['tag_name'],
            'version': AwesomeVersion(release['tag_name']),
            'url': release['html_url'],
            'description': release['body'],
            'date': release['published_at'],
            'assets': [{'name': x['name'], 'url': x['browser_download_url']} for x in release['assets']],
            'current': release['tag_name'] == current_version
        }


class ReleaseNotesDialog(QDialog, Ui_newVersionDialog):
    '''
    shows release notes when we have a new build.
    '''

    def __init__(self, parent, new_versions, issues):
        super(ReleaseNotesDialog, self).__init__(parent)
        self.setupUi(self)
        self.__versions = new_versions
        for v in self.__versions:
            v['release_notes'] = self.__convert_to_html(self.__format(v), issues)
        self.__table_model = VersionTableModel(self.__versions)
        self.versionTable.setModel(self.__table_model)
        self.versionTable.selectionModel().selectionChanged.connect(self.on_version_selected)
        self.versionTable.resizeColumnsToContents()
        self.versionTable.selectRow(0)
        if self.__versions[0]['current'] is False:
            v = self.__versions[0]['version']
            self.message.setText(f"{v} is out, grab it while it is fresh!")
            if v.alpha:
                mod = 'Alpha'
            elif v.beta:
                mod = 'Beta'
            elif v.dev:
                mod = 'Dev'
            elif v.rc:
                mod = 'Release Candidate'
            else:
                mod = 'Final'
            self.setWindowTitle(f"New {mod} Version Available!")
        else:
            self.message.setText('')
            self.setWindowTitle('Release Notes')

    def on_version_selected(self):
        selection = self.versionTable.selectionModel()
        if selection.hasSelection() and len(selection.selectedRows()) > 0:
            self.releaseNotes.setHtml('<p/>'.join([self.__versions[r.row()]['release_notes']
                                                   for r in selection.selectedRows()]))

    @staticmethod
    def __format(v):
        download_link = _get_download_link(v['assets'], platform.system())
        if download_link is not None:
            download_link = f"Download from [GitHub]({download_link})\n"
        else:
            download_link = ''
        return f"## {v['tag']} - {v['date']}\n{download_link}\n{v['description']}\n"

    @staticmethod
    def __convert_to_html(text, issues):
        from markdown import markdown
        import re

        def inject_description(m):
            id = m.group(1)
            if int(id)in issues:
                issue = issues[int(id)]
                return f"[{id}]({issue['url']}) : {issue['title']}"
            else:
                return f"[{id}](https://github.com/3ll3d00d/beqdesigner/issues/{id})"

        return markdown(re.sub(r'#([0-9]+)', inject_description, text))


class VersionTableModel(QAbstractTableModel):
    '''
    A Qt table model to feed the signal view.
    '''

    def __init__(self, versions, parent=None):
        super().__init__(parent=parent)
        self.__headers = ['Version', 'Release Date', 'Win', 'OSX', 'Linux']
        self.__versions = versions

    def rowCount(self, parent: QModelIndex = ...):
        return len(self.__versions)

    def columnCount(self, parent: QModelIndex = ...):
        return len(self.__headers)

    def data(self, index: QModelIndex, role: int = ...):
        if not index.isValid():
            return QVariant()
        elif role != Qt.DisplayRole:
            return QVariant()
        else:
            at_row = self.__versions[index.row()]
            if index.column() == 0:
                return QVariant(at_row['tag'])
            elif index.column() == 1:
                return QVariant(at_row['date'])
            elif index.column() == 2:
                return QVariant('Y' if _has_os(at_row['assets'], 'win') is True else 'N')
            elif index.column() == 3:
                return QVariant('Y' if _has_os(at_row['assets'], 'osx') else 'N')
            elif index.column() == 4:
                return QVariant('Y' if _has_os(at_row['assets'], 'linux') else 'N')
            else:
                return QVariant()

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return QVariant(self.__headers[section])
        return QVariant()


def _get_download_link(assets, os):
    for a in assets:
        if os == 'Windows':
            if a['name'] == 'beqdesigner_small.exe':
                return a['url']
        elif os == 'Darwn':
            if a['name'] == 'beqdesigner.app.zip':
                return a['url']
        elif os.lower().startswith('linux'):
            if not a['name'].endswith('exe') and not a['name'].endswith('app.zip'):
                return a['url']
    return None


def _has_os(assets, os):
    for a in assets:
        if os == 'win':
            if a['name'].endswith('exe'):
                return True
        elif os == 'osx':
            if a['name'].endswith('app.zip'):
                return True
        elif os == 'linux':
            if not a['name'].endswith('exe') and not a['name'].endswith('app.zip'):
                return True
    return False
