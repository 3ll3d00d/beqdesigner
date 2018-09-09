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
* Add from TXT

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

* can extract mono audio
* user selected channel layouts
* can extract multichannel audio
* ffmpeg command specs
