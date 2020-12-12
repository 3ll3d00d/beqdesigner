[![Build Status](https://github.com/3ll3d00d/beqdesigner/workflows/run%20tests/badge.svg
)](https://github.com/3ll3d00d/beqdesigner/actions)

# Developer Setup

## Windows

Install [Anaconda](https://repo.anaconda.com/archive/Anaconda3-5.2.0-Windows-x86_64.exe) then

    conda create -n beq numpy scipy qtpy mkl==2018.0.2 qtawesome pytest pytest-cov sortedcontainers pillow requests
    activate beq
    python -m pip install --upgrade pip
    pip install pyqt5 matplotlib ffmpeg-python soundfile resampy pyqtgraph pyinstaller semver dulwich
    
Note that pyinstaller must be >=4.0 and pyqt5 must be from pypi because of the issue noted in [the pyinstaller release notes](https://pyinstaller.readthedocs.io/en/stable/CHANGES.html#id1)

## Linux/OSX

Assuming pipenv and python3.8 is installed

    pipenv install

# Release

## Hack ffmpeg-python (on Windows only)

* hack ffmpeg to workaround https://github.com/kkroening/ffmpeg-python/issues/116 
  * in _run.py
```  
    stdin_stream = subprocess.PIPE
```
  * in _probe.py
```  
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
```

## Exe

to create an exe

    pyinstaller --clean --log-level=WARN -F for_exe.spec
    
produces 

    dist/beqdesigner.exe
    
## Installer 

to create an installer

    pyinstaller --clean --log-level=WARN -D for_nsis.spec

produces 

    dist/beqdesigner/*    
    
to create an installer

    makensis src\main\nsis\Installer.nsi
