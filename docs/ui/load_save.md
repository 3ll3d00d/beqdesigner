All load/save options are accessible via the File menu.

### Filter

Filters are saved as a file with the `.filt` extension via `File > Save Filter`. This saves the content of the currently selected filter table as a file. `File > Load Filter` can be used to reload this filter at a later date.

The data is stored as [json](https://www.json.org/) so is human readable.

### Signal

Signals are saved as a file with the `.signal` extension via `File > Save Signal`. On selecting this option, you will be presented with a dialog allowing you to select from the currently loaded signals. Clicking OK will allow you to choose where to save the signal to. 

The data is stored as gzipped [json](https://www.json.org/) so is human readable if decompressed.

A signal file contains:

* Basic metadata about the signal (e.g. name, fs)
* The calculated average and peak response curves
* The filters applied (if any)
* metadata about the underlying audio file (if the signal was loaded from an audio file)

`File > Load Signal` can be used to reload the signal at a later date. If the signal was loaded from an audio file then the underlying audio content will be reloaded **if** the audio file remains accessible in the original location (i.e. the file path the audio was loaded from originally). If the file has moved, the signal will still be loaded but without any audio data (hence the waveform view will not be available for this signal). The audio data can be reloaded after signal load via the relevant [waveform control](./waveform.md#chart-controls).

### Project

A BEQDesigner project contains all the currently loaded signals and their filters. It can be saved as a file with the `.beq` extension via `File > Save Project`. 

The data is stored as gzipped [json](https://www.json.org/) so is human readable if decompressed.

`File > Load Project` can be used to reload the project at a later date. Since a project contains signals, the same information regarding audio data loading applies here.

