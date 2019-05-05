import logging

import requests
from qtpy.QtCore import Signal, QRunnable, QObject, QAbstractTableModel, QModelIndex, Qt, QVariant
from qtpy.QtWidgets import QDialog, QHeaderView, QItemDelegate
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
                               if self.__check_beta is True or v['version'].prerelease is None],
                              key=lambda v: v['version'],
                              reverse=True)
        else:
            self.__signals.on_error.emit(f"Unable to hit Github Release API at: \n\n {RELEASE_API}")
        return None

    @staticmethod
    def __convert(release, current_version):
        import semver
        return {
            'tag': release['tag_name'],
            'version': semver.parse_version_info(release['tag_name']),
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

        if self.__versions[0]['current'] is False:
            self.message.setText(f"{self.__versions[0]['version']} is out, <a href='{self.__versions[0]['url']}'>download</a> it now!")
            self.setWindowTitle('New Version Available!')
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
        return f"## {v['tag']} - {v['date']}\nDownload from [GitHub]({v['url']})\n\n{v['description']}\n"

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
                return QVariant(self.__has_os(at_row['assets'], 'win'))
            elif index.column() == 3:
                return QVariant(self.__has_os(at_row['assets'], 'osx'))
            elif index.column() == 4:
                return QVariant(self.__has_os(at_row['assets'], 'linux'))
            else:
                return QVariant()

    def __has_os(self, assets, os):
        for a in assets:
            if os == 'win':
                if a['name'].endswith('exe'):
                    return 'Y'
            elif os == 'osx':
                if a['name'].endswith('app.zip'):
                    return 'Y'
            elif os == 'linux':
                if not a['name'].endswith('exe') and not a['name'].endswith('app.zip'):
                    return 'Y'
        return 'N'

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return QVariant(self.__headers[section])
        return QVariant()
