# -*- mode: python -*-
import os
import platform

def get_resampy_path():
    import resampy as _
    return _.__path__

def get_site_path():
    import site
    return site.USER_SITE

block_cipher = None
spec_root = os.path.abspath(SPECPATH)

datas = [
    ('src\\main\\icons\\Icon.ico', '.'),
    (os.path.abspath(f"{get_resampy_path()}/data/kaiser_fast.npz"), '_resampy_filters'),
    ('src\\main\\python\\style', 'style'),
    ('src\\main\\python\\VERSION', '.')
]
if platform.system() == 'Windows':
    data.append((os.path.abspath(f"{get_site_path()/_soundfile_data"), '_soundfile_data'))
elif platform.system() == 'Darwin':
    data.append((os.path.abspath(f"{get_site_path()/_soundfile_data/libsndfile.dylib"), '_soundfile_data'))

icon = 'icon.icns' if platform.system() == 'Darwin' else 'Icon.ico'

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
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='beqdesigner',
          debug=False,
          strip=False,
          upx=True,
          console=True,
          icon=f"src/main/icons/{icon}")

if platform.system() == 'Darwin':
    app = BUNDLE(exe,
                 name='BEQDesigner.app',
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