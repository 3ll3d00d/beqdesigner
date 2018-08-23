# setup

    conda create -n beq numpy colorcet scipy qtpy mkl==2018.0.2 qtawesome
    activate beq
    python -m pip install --upgrade pip
    pip install pyqt5 matplotlib ffmpeg-python soundfile resampy
    pip install https://github.com/pyinstaller/pyinstaller/tarball/develop

# todo

* add frd import
* create biquad filter export
* add load/save feature 
  * allow loading individual signals or filters only
* add filter presets
* save graph as png
  * with optional additional image (as per the avs thread style)
* add gain adjust filter
* add another table or list of fields or dialog that shows the value under the cursor or values at a user entered position
  * or a draggable line that reads the values
* get live update chart working efficiently
* add a preferences model that stores the actual values and is passed around to avoid passing settings around
* come up with a sensible y axis tick scheme

# bugs

* filter errors
  * odd order high passes are wrong (because the 1st order is wrong)
  * LR2/6/10/etc are wrong
* multichannel export produces too many channels and the preview graph is borked


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
    
