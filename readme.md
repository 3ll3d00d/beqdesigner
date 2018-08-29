# setup

## Windows

Install https://repo.anaconda.com/archive/Anaconda3-5.2.0-Windows-x86_64.exe then

    conda create -n beq numpy colorcet scipy qtpy mkl==2018.0.2 qtawesome
    activate beq
    python -m pip install --upgrade pip
    pip install pyqt5 matplotlib ffmpeg-python soundfile resampy
    pip install https://github.com/pyinstaller/pyinstaller/tarball/develop

# todo

## load/save/import/export

* add frd import/export

* add load/save feature 
  * allow loading individual signals or filters only

* add filter presets

* allow addition of user supplied image with png (as per the avs thread style)

## core functionality 

* add gain adjust filter
* **BUG** multichannel export produces too many channels

## look and feel 

* come up with a sensible y axis tick scheme

## misc

* add a preferences model that stores the actual values and is passed around to avoid passing settings around

# Freeze

## Hack

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
    