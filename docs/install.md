## Installation

Releases are hosted on github, the latest official release will always be available via the [github releases page](https://github.com/3ll3d00d/beqdesigner/releases/latest)

3 sets of binaries are provided for official releases:

File Name | OS | Description 
----------|----|------------
beqdesigner_*version*.exe | Windows | A portable, self contained exe which is built using the python.org python distribution. 
beqdesigner_*distro*\_*version*\_*name* | Linux | A portable, self contained binary built for the specified distribution
beqdesigner.app.zip_*version* | MacOS | A MacOS app bundle. 

For each OS, the *version* suffix denotes which version of that OS was used to build the exe. 

To run beqdesigner, take the following steps:

  1. download one of the above binaries. NB: where multiple versions for your OS are available, start by picking the version which corresponds to your own OS. If no such version is available, try each in turn.
  2. for MacOS only, rename the file to remove the version suffix and extract the contents of the zip file
  3. run it

It should now open normally. If it doesn't, repeat by running the file from a terminal/command prompt as this should provide more detailed error messages.

### Checking for Updates

Release notes and download links are accessible via the *Help > Release Notes* menu item. This is also displayed on startup if a new version is detected and the github release api is accessible.

![New Version](./img/new_version.png)

The OS columns show whether a binary is available for that operating system.

Multiple rows can be selected to see what has changed in each release.

![Many Releases](./img/show_release_notes.png)

If a binary is available for your operating system for any selected release, the download link will be provided alongside the release notes.

### Windows Smartscreen   

!!! warning
    Windows binaries are not signed so will be flagged as "Unknown" by Windows Defender. You will have to accept and disregard this warning in order to use BEQDesigner.

### OSX Builds

!!! note
    OSX binaries are not signed so the user has to explicitly allow it to run. Contributions are welcome to fix this issue, please track [the github issue](https://github.com/3ll3d00d/beqdesigner/issues/251) for more details.

### How can I trust these binaries?

All binaries are compiled on fresh VMs using [github actions](https://github.com/3ll3d00d/beqdesigner/actions) and publishes automatically to github.

Builds are created and published from tags in the repo.

This approach means the binaries are completely reproducible by anyone so feel free to make your own or run from source if you do not wish to trust the published binaries.
    
## Optional Dependencies

If you intend to use BEQDesigner to extract audio from movie files and/or remux movie files then you will need a local installation of ffmpeg.

### Windows/OSX

ffmpeg binaries are available via [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)

Download the latest *release* version and extract to a known directory, make a note of this location as it will be required during BEQ configuration.

!!! note
    gyan.dev publishes 2 builds, full and essentials. The essentials version does not contain soxr which is used for resampling audio therefore always download the full build.

### Linux

Refer to your distro for instructions

* [debian](https://wiki.debian.org/ffmpeg#Installation)
* [fedora via rpm-fusion](https://www.cyberciti.biz/faq/how-to-install-ffmpeg-on-fedora-linux-using-dnf)   

## Beta Releases

Beta releases are published more regularly than official releases. Beta releases provide early access to new or experimental features. Review the release notes for any particular beta release for more details.

They can be found via the full release page on [github](https://github.com/3ll3d00d/beqdesigner/releases).
