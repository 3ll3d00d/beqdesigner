# -*- mode: python -*-

block_cipher = None

a = Analysis(['src\\main\\python\\app.py'],
             pathex=['C:\\Users\\mattk\\github\\beq'],
             binaries=None,
             datas=[
                ('src\\main\\icons\\Icon.ico', '.'),
                ('C:\\Users\\mattk\\Anaconda3_64\\envs\\beq\\Lib\\site-packages\\resampy\\data\\kaiser_fast.npz', '_resampy_filter'),
                ('C:\\Users\\mattk\\Anaconda3_64\\envs\\beq\\Lib\\site-packages\\_soundfile_data', '_soundfile_data')
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
          console=False,
          icon='src\\main\\icons\\Icon.ico' )
