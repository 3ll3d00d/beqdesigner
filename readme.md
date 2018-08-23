# setup

    conda create -n beq numpy colorcet scipy qtpy mkl==2018.0.2 qtawesome
    activate beq
    python -m pip install --upgrade pip
    pip install pyqt5 matplotlib ffmpeg-python soundfile resampy
    pip install https://github.com/pyinstaller/pyinstaller/tarball/develop

# todo

## load/save/import/export

* add frd import/export
* create biquad filter export

* add load/save feature 
  * allow loading individual signals or filters only

* add filter presets

* save graph as png
  * with optional additional image (as per the avs thread style)

## core functionality 

* add gain adjust filter
* **BUG** multichannel export produces too many channels
* **BUG** LR2/6/10/etc are wrong

## look and feel 

* use animation on the charts so they can update live
* come up with a sensible y axis tick scheme

## misc

* add a preferences model that stores the actual values and is passed around to avoid passing settings around

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
    
## Notes

* replace the implementation of ``load_filter`` in ``resampy\filters.py`` with the implementation below and remove the ``import pkg_resources``::

```
    # hack in pyinstaller support
    if getattr(sys, 'frozen', False):
        data = np.load(os.path.join(sys._MEIPASS, '_resampy_filters', os.path.extsep.join([filter_name, 'npz'])))
    else:
        fname = os.path.join('data',
                             os.path.extsep.join([filter_name, 'npz']))
        import pkg_resources
        data = np.load(pkg_resources.resource_filename(__name__, fname))
    
    return data['half_window'], data['precision'], data['rolloff']
```
    
* add the following to the spec in the EXE section
```
    Tree('C:\\Users\\mattk\\Anaconda3_64\\envs\\beq\\Lib\\site-packages\\resampy\\data', prefix='_resampy_filters'),
    Tree('C:\\Users\\mattk\\Anaconda3_64\\envs\\beq\\Lib\\site-packages\\_soundfile_data', prefix='_soundfile_data'),
```
* hack ffmpeg _probe.py and _run.py so they always pass `stdin=subprocess.PIPE`
  * in _run.py
```  
    stdin_stream = subprocess.PIPE
```
  * in _probe.py
```  
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
```