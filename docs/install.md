## Installation

Releases are hosted on github, the latest official release will always be available via the [github releases page](https://github.com/3ll3d00d/beqdesigner/releases/latest)

3 binaries are provided for official releases:

File Name | OS | Description 
----------|----|------------
beqdesigner.exe | Windows | A portable, self contained exe 
beqdesignerSetup.exe | Windows | An installer, will startup more quickly than the portable exe 
beqdesigner | Linux | A portable, self contained binary (tested on Debian Stretch and Fedora 29) 

To use, simply download one of the above binaries and run it.

### Is there any difference between beqdesigner.exe and beqdesignerSetup.exe?

Functionally, no.

The only difference is that using beqdesignerSetup.exe should lead to faster startup times.

### Windows Smartscreen   

!!! warning
    Windows binaries are not signed so will be flagged as "Unknown" by Windows Defender. You will have to accept and disregard this warning in order to use BEQDesigner.

### OSX Builds

!!! note
    BEQDesigner is known to work on OSX but binaries are not provided by the author at this point in time. Some binaries are provided by community member bmiller (refer to the [BEQ slack group](https://join.slack.com/t/beqworkspace/shared_invite/enQtNTE3Mjg4MTgzMjY5LWIzZWZjYzNkOTQzZThkYzM5YTAwNzFmY2VlNjFkYTI1NWQ0NDU1ZTViMzg0OWUyMTdkZjQ5NDNmNGFmYzliODY)) for details.

## Optional Dependencies

If you intend to use BEQDesigner to extract audio from movie files and/or remux movie files then you will need a local installation of ffmpeg.

### Windows/OSX

ffmpeg binaries are available via [zeranoe](https://ffmpeg.zeranoe.com/builds/)

Download the latest version and extract to a known directory, make a note of this location as it will be required during BEQ configuration.  

### Linux

Refer to your distro for instructions

* [debian](https://wiki.debian.org/ffmpeg#Installation)
* [fedora via rpm-fusion](https://www.cyberciti.biz/faq/how-to-install-ffmpeg-on-fedora-linux-using-dnf)   

## Beta Releases

Beta releases are published more regularly than official releases. Beta releases provide early access to new or experimental features. Review the release notes for any particular beta release for more details.

They can be found via the full release page on [github](https://github.com/3ll3d00d/beqdesigner/releases).

Note that beta releases are typically shared in `beqdesigner.exe` format only.
