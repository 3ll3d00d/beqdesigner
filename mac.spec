# -*- mode: python -*-
import platform

block_cipher = None


a = Analysis(['src/main/python/app.py'],
             pathex=['/Users/bradmiller/Projects/beqdesigner'],
             binaries=[],
             datas=[
                ('src/main/icons/Icon.ico', '.'),
                ('/usr/local/lib/python3.7/site-packages/resampy/data/kaiser_fast.npz', '_resampy_filters'),
                ('src/main/python/style', 'style'),
                ('src/main/python/VERSION', '.')
             ],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='mac',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True,
          icon='src/main/icons/icon.icns')

if platform.system() == 'Darwin':
    app = BUNDLE(exe,
                 name='BEQDesigner.app',
                 bundle_identifier=None,
                 info_plist={
                  'NSHighResolutionCapable': 'True',
                  'LSBackgroundOnly': 'False'
                },
                icon='src/main/icons/icon.icns'
                )