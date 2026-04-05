# -*- mode: python -*-
import os
import platform
import distro

# work-around for https://github.com/pyinstaller/pyinstaller/issues/4064
import distutils

distutils_dir = getattr(distutils, 'distutils_path', None)
if distutils_dir is not None and distutils_dir.endswith('__init__.py'):
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
debian_excludes = [
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
    'centos': generic_linux_excludes + fedora_excludes,
    'debian': generic_linux_excludes + debian_excludes
}

# helper functions


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
    Decide whether EXE embeds all the binaries/data (onefile) or defers them to a later COLLECT
    step (onedir).

    PyInstaller has two packaging modes:

      * onefile  -- everything (the Python runtime, Qt, every wheel, icons, etc.) is stuffed into
                    one self-extracting executable. At launch, the bootloader picks a random temp
                    dir (e.g. $TMPDIR/_MEIxxxxx), extracts everything into it, fork/execs itself
                    and runs Python there, then deletes the temp dir on exit.
      * onedir   -- binaries/data sit on disk next to the executable at stable paths. No per-launch
                    extract step; the launcher just starts Python and lets it import from those
                    stable paths.

    macOS is forced to onedir because onefile is unusable-slow there:

      1. On onefile each launch unpacks ~200MB of dylibs into a fresh /var/folders/_MEIxxxxx path.
         dyld (the dynamic linker) cannot use its shared cache for libraries at previously-unseen
         paths, so it opens, hashes, maps and Gatekeeper-verifies every .so/.dylib from scratch
         each time. `sample` of such a launch is dominated by
         dyld4::JustInTimeLoader::makeJustInTimeLoaderDisk -> fcntl -> open.
      2. PyInstaller also ships a runtime hook (pyi_rth_mplconfig.py) that, in onefile mode,
         intentionally points MPLCONFIGDIR at a tempdir that is wiped at exit, forcing matplotlib
         to rebuild its font cache (~14s) on every launch. This was a workaround for matplotlib
         < 3.0 where font-cache entries embedded the install path; matplotlib fixed that in their
         PR #12472 (shipped in 3.0, 2018) but PyInstaller still applies the workaround
         unconditionally as of 6.19.0. Tracked at:
           https://github.com/pyinstaller/pyinstaller/issues/3959  (open since 2018)
           https://github.com/matplotlib/matplotlib/issues/13071   (closed: "punt to PyInstaller")
         We counter-override MPLCONFIGDIR back to ~/.matplotlib in app.py when sys.frozen; that
         mitigation is independent of the onefile/onedir choice here, but onedir is still the
         right choice to keep dyld happy.

    Together these made cold-start of the .app take ~16s. Building onedir + overriding
    MPLCONFIGDIR in app.py drops it to ~1.7s.

    Windows with NSIS also needs onedir -- COLLECT is what the NSIS installer packages into
    Program Files. Default Windows and Linux keep onefile (they do not suffer from the same dyld
    caching penalty on unknown paths).

    :return: the *args to pass to EXE. ``(a.scripts,)`` on macOS and NSIS (onedir: binaries get
             gathered by the later COLLECT step). ``(a.scripts, a.binaries, a.zipfiles, a.datas)``
             elsewhere (onefile: embed everything into the EXE).
    '''
    if use_nsis() is True or platform.system() == 'Darwin':
        return (a.scripts,)
    return (a.scripts, a.binaries, a.zipfiles, a.datas)


def get_data_args():
    '''
    :return: the data array for the analysis.
    '''
    vals = [
        ('src/main/icons/Icon.ico', '.'),
        ('src/main/python/style', 'style'),
        ('src/main/python/VERSION', '.'),
        ('src/main/xml/flat24hd.xml', '.'),
    ]
    import glob
    for p in glob.glob("src/main/xml/default_jriver_config_*.xml"):
        vals.append((p, '.'))
    return vals


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


def get_exe_name():
    '''
    Gets the executable name which is beqdesigner for osx & windows and has some dist specific suffix for linux.
    '''
    if platform.system().lower().startswith('linux'):
        linux_dist = distro.linux_distribution(full_distribution_name=False)
        return f"beqdesigner_{'_'.join((x for x in linux_dist if x is not None and len(x) > 0))}"
    elif platform.system().lower().startswith('windows'):
        return f"beqdesigner_{platform.win32_ver()[0].replace('.', '_')}"
    return 'beqdesigner'


def get_binaries():
    '''
    :return: the ssl binaries if we're on windows and they exist + graphviz (if available).
    '''
    ssl_dlls = get_ssl_dlls()
    gz_binaries = get_graphviz_binaries()
    return ssl_dlls + gz_binaries


def get_graphviz_binaries():
    '''
    :return: the graphviz binaries.
    '''
    if platform.system() == 'Windows':
        return get_graphviz_windows()
    return []


def get_graphviz_windows():
    path_to_gz = os.environ.get('GZ_PATH', 'c:/Program Files/Graphviz/bin')
    dot_exe = 'dot.exe'
    if os.path.isfile(os.path.join(path_to_gz, dot_exe)):
        gz = [
            (os.path.join(path_to_gz, 'cdt.dll'), '.'),
            (os.path.join(path_to_gz, 'cgraph.dll'), '.'),
            (os.path.join(path_to_gz, 'dot.exe'), '.'),
            (os.path.join(path_to_gz, 'gvc.dll'), '.'),
            (os.path.join(path_to_gz, 'libexpat.dll'), '.'),
            (os.path.join(path_to_gz, 'gvplugin_core.dll'), '.'),
            (os.path.join(path_to_gz, 'gvplugin_dot_layout.dll'), '.'),
            (os.path.join(path_to_gz, 'pathplan.dll'), '.'),
            (os.path.join(path_to_gz, 'xdot.dll'), '.'),
            (os.path.join(path_to_gz, 'zlib1.dll'), '.')
        ]
        return gz
    else:
        print(f"MISSING {os.path.join(path_to_gz, dot_exe)}")
        return []


def get_ssl_dlls():
    if platform.system() == 'Windows':
        import os
        ssl_dll = 'c:/Windows/System32/libssl-1_1-x64.dll'
        crypto_dll = 'c:/Windows/System32/libcrypto-1_1-x64.dll'
        if os.path.isfile(ssl_dll):
            if os.path.isfile(crypto_dll):
                return [
                    (ssl_dll, '.'),
                    (crypto_dll, '.'),
                ]
            else:
                print(f"MISSING libcrypto-1_1-x64.dll")
        else:
            print(f"MISSING libssl-1_1-x64.dll")
    return []


block_cipher = None
spec_root = os.path.abspath(SPECPATH)

a = Analysis(['src/main/python/app.py'],
             pathex=[spec_root],
             binaries=get_binaries(),
             datas=get_data_args(),
             hiddenimports=['numpy.random', 'pkg_resources.py2_warn', 'scipy.spatial.transform._rotation_groups'],
             hookspath=['hooks/'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

remove_platform_specific_binaries(a)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(pyz,
          *get_exe_args(),
          name=get_exe_name(),
          debug=False,
          strip=False,
          upx=False,
          console=True,
          exclude_binaries=use_nsis() or platform.system() == 'Darwin',
          icon=get_icon_file())

if platform.system() == 'Darwin':
    # Build a onedir .app: COLLECT lays the binaries/datas out at stable paths inside
    # beqdesigner.app/Contents/Frameworks so dyld can cache them, rather than PyInstaller
    # extracting everything to /var/folders/_MEIxxxxx on every launch (onefile mode, which is
    # very slow to cold-start on macOS).
    coll = COLLECT(exe,
                   a.binaries,
                   a.zipfiles,
                   a.datas,
                   strip=False,
                   upx=False,
                   name='beqdesigner')
    app = BUNDLE(coll,
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
