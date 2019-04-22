# work-around for https://github.com/pyinstaller/pyinstaller/issues/4064
import distutils
if distutils.distutils_path.endswith('__init__.py'):
    distutils.distutils_path = os.path.dirname(distutils.distutils_path)

# -*- mode: python -*-
import os
import platform

def get_resampy_path():
    import resampy as _
    return _.__path__[0]

def get_sndfile_path():
    import soundfile as _
    return os.path.dirname(_.__file__)

block_cipher = None
spec_root = os.path.abspath(SPECPATH)

datas = [
    ('src/main/icons/Icon.ico', '.'),
    (os.path.abspath(f"{get_resampy_path()}/data/kaiser_fast.npz"), '_resampy_filters'),
    ('src/main/python/style', 'style'),
    ('src/main/python/VERSION', '.')
]
if platform.system() == 'Windows':
    datas.append((os.path.abspath(f"{get_sndfile_path()}/_soundfile_data"), '_soundfile_data'))
elif platform.system() == 'Darwin':
    datas.append((os.path.abspath(f"{get_sndfile_path()}/_soundfile_data/libsndfile.dylib"), '_soundfile_data'))

icon = f"src/main/icons/{'icon.icns' if platform.system() == 'Darwin' else 'Icon.ico'}"

a = Analysis(['src/main/python/app.py'],
             pathex=[spec_root],
             binaries=None,
             datas=datas,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

use_win_nsis = platform.system() == 'Windows' and 'USE_NSIS' in os.environ

if use_win_nsis is True:
    exe_args = (a.scripts,)
else:
    exe_args = (a.scripts, a.binaries, a.zipfiles, a.datas)

exe = EXE(pyz,
          *exe_args,
          name='beqdesigner',
          debug=False,
          strip=False,
          upx=platform.system() != 'Windows',
          console=True,
          icon=icon)

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
elif use_win_nsis is True:
    coll = COLLECT(exe,
                   a.binaries,
                   a.zipfiles,
                   a.datas,
                   strip=False,
                   upx=False,
                   name='beqdesigner')
