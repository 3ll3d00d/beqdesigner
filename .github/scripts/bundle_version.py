'''
The version the macOS .app's Info.plist carries, from the VERSION file the release workflow writes from its tag.

``CFBundleShortVersionString`` must be period-separated integers, so a tag like ``2.2.0-alpha.10`` gives ``2.2.0``;
``CFBundleVersion`` carries the whole string, so a crash report reads "Version: 2.2.0 (2.2.0-alpha.10)" rather than
"0.0.0 (???)" (issue #525). Without a VERSION file (a build from source) both fall back to the app's own default.
'''
import os
import re

DEFAULT_VERSION = '0.0.0-alpha.1'   # what app.py reports when it finds no VERSION file


def read_version(path: str) -> str:
    try:
        with open(path) as handle:
            return handle.read().strip() or DEFAULT_VERSION
    except FileNotFoundError:
        return DEFAULT_VERSION


def bundle_versions(version: str) -> tuple[str, str]:
    '''
    :return: (CFBundleShortVersionString, CFBundleVersion) for `version`: its leading x.y.z (missing parts as 0), and
        the version itself.
    '''
    # numbers only when the version is made of them, so a commit sha such as 1234abc is not read as 1234.0.0
    match = re.match(r'v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(?=$|[-+.])', version.strip())
    if match is None:
        return '0.0.0', version.strip()
    short = '.'.join(part or '0' for part in match.groups())
    return short, version.strip().removeprefix('v')


def bundle_versions_from(path: str) -> tuple[str, str]:
    return bundle_versions(read_version(path))


if __name__ == '__main__':
    import sys
    print(*bundle_versions_from(sys.argv[1] if len(sys.argv) > 1 else os.path.join('src', 'main', 'python', 'VERSION')))
