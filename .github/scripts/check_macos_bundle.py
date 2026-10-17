#!/usr/bin/env python3
'''
Checks a zipped macOS .app keeps the symlinks its frameworks are made of.

    python3 .github/scripts/check_macos_bundle.py dist/beqdesigner.app.zip_macos_26.6.2

A framework is a symlinked bundle: ``QtCore.framework/QtCore`` and ``/Resources`` link into ``Versions/Current``,
which links to ``Versions/A``. ``zip -r`` follows symlinks and stores copies, so every framework arrives as real
duplicate directories (and PyInstaller's ``Contents/Resources`` -> ``Contents/Frameworks`` links double the app).
CoreFoundation then may not find ``org.qt-project.QtCore`` by its identifier, and Qt passes the NULL it gets back to
``CFBundleCopyBundleURL``: on macOS 27 the app crashes on launch (issue #525). Exits 1, naming each framework, when a
framework's ``Versions/Current`` is not a symlink.
'''
import stat
import sys
import zipfile


def is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def problems(path: str) -> list[str]:
    '''
    :return: what is wrong with the zipped bundle at `path`, empty if nothing is.
    '''
    with zipfile.ZipFile(path) as archive:
        entries = {info.filename.rstrip('/'): info for info in archive.infolist()}
    frameworks = sorted({name[:name.index('.framework') + len('.framework')]
                         for name in entries if '.framework' in name})
    if not frameworks:
        return [f'{path}: no frameworks found, so this is not a packaged .app']
    found = []
    for framework in frameworks:
        current = entries.get(f'{framework}/Versions/Current')
        copied = any(name.startswith(f'{framework}/Versions/Current/') for name in entries)
        if current is None and not copied:
            found.append(f'{framework}: no Versions/Current')
        elif copied or not is_symlink(current):
            found.append(f'{framework}: Versions/Current is a copy, not a symlink')
    return found


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    failed = False
    for path in paths:
        for problem in problems(path):
            print(problem, file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
