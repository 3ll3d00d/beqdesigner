## What is BEQ?

Bass EQ (BEQ) is a term coined on [data-bass](http://data-bass.ipbhost.com/topic/285-the-bass-eq-for-movies-thread/) in 2014 to describe a method for recovering low frequency content that has been filtered out during post production of the mix. 

BEQ has since gained a new lease of life via [avsforum](https://www.avsforum.com/forum/113-subwoofers-bass-transducers/2995212-bass-eq-filtered-movies.html). Hundreds of films have now been BEQ'ed with filters shared via [a github repo](https://github.com/bmiller/miniDSPBEQ).  

### When does this matter?

Playing back such a mix on a full bandwidth capable system results in an experience that has less weight and impact than might be expected to accompany the on screen content. 

### What is BEQDesigner?

BEQDesigner is a application which provides the means to:

* create BEQ filters
* analyse the effect of the BEQ filters on the audio track
* remux movie tracks to include the BEQ'ed audio track
* apply freely distributed BEQ filters to your own personal minidsp configuration

## Core Concepts

### Signals and Filters

BEQDesigner is an application that allows you to see the effects of filters on signals. 

#### What is a Signal?

A signal is a mono audio track. This typically means a signal is one of the following:
sm
* a single channel in a multichannel audio track 
* a combination of all channels from a multichannel audio track mixed into a mono audio file with channel levels adjusted to account for the LFE channel offset

The [Extract Audio](ui/extract_audio.md) utility is an easy to use interface for extracting such content from standard movie file formats.

#### What is a Filter?
  
A filter is a way of making particular sounds louder or quieter, i.e. it is a change in the frequency response of a signal.

#### What types of filters are supported?

BEQDesigner is designed to be used with commonly available and used DSP platforms. Such platforms typically allow the user to apply various types of parametric EQ filters which can change the frequency response of the signal in various ways.

The most commonly used device for this purpose is the [MiniDSP 2x4 HD](https://www.minidsp.com/products/minidsp-in-a-box/minidsp-2x4-hd) and BEQDesigner supports all the (IIR) filter types supported by this device. These are:

* Low and High Shelf filters
* Parametric EQ
* High and Low Pass Filters of the following types:
  * Butterworth (1st to 24th order)  
  * Linkwitz-Riley (2nd to 24th order)
  * User defined Q
  
Shelf filters and PEQ are implemented using the formulae shared in the [RBJ Cookbook](http://shepazu.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html)

!!! warning
    All filters are calculated using 64 bit floating point arithmetic. This ensures there are no biquad precision issues regardless of centre frequency or Q. However most commonly available DSP platforms do not operate at this precision and hence can suffer from inaccuracies, particularly at very low frequencies (i.e. <20Hz). Always be aware of the accuracy of your own hardware when applying such filters.      
  
### Linking Signals

BEQDesigner allows certain signals to be linked together. This means a filter can be defined once but applied to many signals. When signals are linked, one signal becomes the *master* signal and all linked signals become *slaves*. 

This terminology can be ignored for most practical purposes, it simply means that the filter applied is owned by the master signal.

Refer to [Link Signals](ui/main_window.md#linking-signals) for how to link signals in BEQDesigner.

#### Why is this useful? 

There are 2 common scenarios where this is relevant

  1. Applying a post bass management BEQ filter to the input channels (either when remuxing or when using a DSP device that sits before bass management in the signal chain)
  2. When designing a set of pre bass management BEQ filters and some subset of channels rolloff with a very similar shape. Common groups in this case are the LCR or the surrounds.

### Signal Analysis

All analysis methods are based on some form of [STFT](https://en.wikipedia.org/wiki/Short-time_Fourier_transform) analysis as provided by the [scipy](https://docs.scipy.org/doc/scipy/reference/index.html) library.

As the analysis is STFT based, the time vs frequency tradeoff is ever present, i.e. increasing the frequency resolution of the analysis reduces the temporal resolution and vice versa.

This is controlled via the relevant [Preferences](./ui/preferences.md#analysis).  

#### Avg Frequency Response

The avg frequency response is a measure of the average level by frequency of the entire track. It is calculated using [Welch's method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html) with *spectrum* scaling and *mean* averaging.

#### Peak Frequency Response

The peak frequency response shows the peak level by frequency reached in any single analysis time period. It is calculated by computing a [spectrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html#scipy.signal.spectrogram) and then calculating the max value on the frequency axis.

#### Spectrum

The spectrum view is essentially a spectrogram albeit one with some visual differences. See the [spectrum view](./ui/spectrum.md) for full details.
