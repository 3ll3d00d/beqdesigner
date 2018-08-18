# -*- mode: python -*-

block_cipher = None

a = Analysis(['src\\main\\python\\app.py'],
             pathex=['C:\\Users\\mattk\\github\\beq'],
             binaries=[],
             datas=[('src\\main\\icons\\Icon.ico', '.')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='beqdesigner',
          debug=False,
          strip=False,
          upx=False,
          console=False,
          icon='src\\main\\icons\\Icon.ico' )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=False,
               name='beqdesigner')