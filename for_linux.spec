# -*- mode: python -*-

block_cipher = None

a = Analysis(['src/main/python/app.py'],
             pathex=['/home/mattk/github/beq'],
             binaries=None,
             datas=[
                ('src/main/icons/Icon.ico', '.'),
                ('/home/matt/python/beq/lib/python3.7/site-packages/resampy/data/kaiser_fast.npz', '_resampy_filters'),
                ('src/main/python/style', 'style'),
                ('src/main/python/VERSION', '.')
             ],
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
          upx=False,
          console=True,
          icon='src/main/icons/Icon.ico' )
