# setup

    conda create -n beq numpy colorcet scipy qtpy mkl==2018.0.2
    activate beq
    python -m pip install --upgrade pip
    pip install pyqt5 matplotlib ffmpeg-python soundfile resampy
    pip install https://github.com/pyinstaller/pyinstaller/tarball/develop

# todo

* create biquad filter export
* add load/save feature 
  * allow loading individual signals or filters only
* add filter presets


# bugs

* 1st order filters do nothing

# Freeze

to create an exe

    pyinstaller --clean --log-level=WARN -F for_exe.spec
    
produces 

    dist/beqdesigner.exe
    
to create an installer

    pyinstaller --clean --log-level=WARN -D for_nsis.spec

produces 

    dist/beqdesigner/*    
    
to create an installer

    makensis src\main\nsis\Installer.nsi
    
