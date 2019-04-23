# -*- mode: python -*-
import os
import platform
import distro

# work-around for https://github.com/pyinstaller/pyinstaller/issues/4064
import distutils

if distutils.distutils_path.endswith('__init__.py'):
    distutils.distutils_path = os.path.dirname(distutils.distutils_path)


generic_linux_excludes = [
    'libstdc++.so.',
    'libtinfo.so.',
    'libreadline.so.',
    'libdrm.so.'
]
fedora_excludes = [
    'libgio-2.0.so.',
    'libglib-2.0.so.',
    'libfreetype.so.',
    'libssl.so.',
    'libfontconfig.so.'
]
ubuntu_excludes = [
    'libgpg-error.so.',
    'libgtk-3.so.*',
    'libgio-2.0.so.*'
]
linux_excludes = {
    'ubuntu': generic_linux_excludes + ubuntu_excludes,
    'linuxmint': generic_linux_excludes + ubuntu_excludes,
    'fedora': generic_linux_excludes + fedora_excludes,
    'centos': generic_linux_excludes + fedora_excludes
}

# helper functions

def get_resampy_path():
    '''
    :return: gets the current path to the resampy module in order to find where the kaiser data files are.
    '''
    import resampy as _
    return _.__path__[0]


def get_sndfile_path():
    '''
    :return: gets the current path to the soundfile module in order to find where the bundled libsndfile binaries are.
    '''
    import soundfile as _
    return os.path.dirname(_.__file__)


def get_sndfile_data():
    '''
    :return: a tuple to add to the datas if we're on a platform which packages libsndfile binaries otherwise None.
    '''
    path = None
    if platform.system() == 'Windows':
        path = os.path.abspath(f"{get_sndfile_path()}/_soundfile_data")
    elif platform.system() == 'Darwin':
        path = os.path.abspath(f"{get_sndfile_path()}/_soundfile_data/libsndfile.dylib")
    return (path , '_soundfile_data') if path is not None else None


def get_icon_file():
    '''
    :return: the full path to the icon file for the current platform.
    '''
    return f"src/main/icons/{'icon.icns' if platform.system() == 'Darwin' else 'Icon.ico'}"


def use_nsis():
    '''
    :return: true if pyinstaller is being run in order to create an installer.
    '''
    return platform.system() == 'Windows' and 'USE_NSIS' in os.environ


def get_exe_args():
    '''
    :return: the *args to pass to EXE, varies according to whether we are in "create an installer" mode or not.
    '''
    return (a.scripts,) if use_nsis() is True else (a.scripts, a.binaries, a.zipfiles, a.datas)


def get_data_args():
    '''
    :return: the data array for the analysis.
    '''
    datas = [
        ('src/main/icons/Icon.ico', '.'),
        (os.path.abspath(f"{get_resampy_path()}/data/kaiser_fast.npz"), '_resampy_filters'),
        ('src/main/python/style', 'style'),
        ('src/main/python/VERSION', '.')
    ]
    sndfile_data = get_sndfile_data()
    if sndfile_data is not None:
        datas.append(sndfile_data)
    return datas


def should_keep_binary(x):
    '''
    :param x: the binary (from Analysis.binaries)
    :return: True if we should keep the given binary in the resulting output.
    '''
    if platform.system().lower().startswith('linux'):
        dist = distro.linux_distribution(full_distribution_name=False)[0]
        return not __is_exclude(x, linux_excludes.get(dist, generic_linux_excludes))
    return True


def __is_exclude(x, excludes):
    for exclude in excludes:
        if x[0].startswith(exclude):
            import sys
            print(f"EXCLUDING {x}", file=sys.stderr)
            return True
    return False


def remove_platform_specific_binaries(a):
    '''
    Removes elements from the analysis based on the current platform.
    Provides equivalent behaviour to https://github.com/mherrmann/fbs/tree/master/fbs/freeze
    :param a: the pyinstaller analysis.
    '''
    a.binaries = [x for x in a.binaries if should_keep_binary(x) is True]


block_cipher = None
spec_root = os.path.abspath(SPECPATH)

a = Analysis(['src/main/python/app.py'],
             pathex=[spec_root],
             binaries=None,
             datas=get_data_args(),
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

remove_platform_specific_binaries(a)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(pyz,
          *get_exe_args(),
          name='beqdesigner',
          debug=False,
          strip=False,
          upx=False,
          console=True,
          icon=get_icon_file())

if platform.system() == 'Darwin':
    app = BUNDLE(exe,
                 name='beqdesigner.app',
                 bundle_identifier='com.3ll3d00d.beqdesigner',
                 icon='src/main/icons/icon.icns',
                 info_plist={
                     'NSHighResolutionCapable': 'True',
                     'LSBackgroundOnly': 'False',
                     'NSRequiresAquaSystemAppearance': 'False',
                     'LSEnvironment': {
                         'PATH': '/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:'
                     }
                 })

if use_nsis() is True:
    coll = COLLECT(exe,
                   a.binaries,
                   a.zipfiles,
                   a.datas,
                   strip=False,
                   upx=False,
                   name='beqdesigner')
