# UI Tests

## Filter

* Add 
  * preview view updates on each change
  * can add multiple on one go 
* Edit
  * preview view shows combined effect
  * can turn off combined
* Delete
 
* main chart updates whenever filter changes

* can save preset
* can load preset
* preset button ticked when preset is loaded

## Signal

* Add from WAV
  * can handle mono or multichannel
  * can select a subset of a wav
  * can show preview
  * can add all channels in one go
* Add from TXT
* can copy filter on add
* can link filter on add
* can choose not to decimate on load

## Linking

* Can link signals on create
* Can link signals after creation
* Filter table shows the filter when any linked signal is selected
* Linked signals all update when the filter changes
* Linking survives import/export 
* Signal table shows link status
* Can remove slave signals from model
* Can remove master from model
 
## Reference

* dropdown shows all visible curves
* selecting dropdown normalises other curves against reference
* selecting None reverts all curves to unnormalised 

## Display

* can choose to show filters
* can choose to show legend
* can show peak or avg only or show both
* can show filtered or unfiltered or both
* can change graph limits
* can show values by frequency for each line

## Persist

* load/save project
* load/save signal
* load/save filter

## Export

* can export biquads of selected filter
* can export FRD of any signal
* can export chart

## Extract

* can open
  * audio only
    * wav
    * flac
    * mono
    * stereo
    * multichannel
  * video
    * mkv
    * m2ts
* extract
  * multichannel
    * to mono 
    * to mono + compress 
    * to mono + decimate
    * to mono + decimate + compress
    * passthrough (check result has same channel layout)
    * passthrough + compress 
    * passthrough + decimate
    * passthrough + decimate + compress
  * mono 
    * mono checkbox is disabled
    * compress
    * decimate
    * decimate + compress
  * can pick a stream 
  * can cut slice of track (start/end)
* with video
  * decimate not allowed
  * output file type is input file type
  * can cut slice of track (start/end)
* decimate honours analysis fs from preferences
  
* if sound is stored in preferences, it is played on complete
* dialog updates in realtime as the audio is extracted

## Remux

* video stream is selected by default
* filters are mapped to channels using naming convention
* can override filters
* can include original audio
* resulting audio stream is filtered
* can decimate and compress


## Preferences

* can change theme (and see it apply after restart)

## Analysis

* can show peak signals
* can show waveform

