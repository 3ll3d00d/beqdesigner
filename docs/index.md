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

### When is BEQ applied?

There are 2 fundamentally different styles of BEQ.

#### Pre Bass Management (or Input) BEQ

All BEQs published on [data-bass](http://data-bass.ipbhost.com/topic/285-the-bass-eq-for-movies-thread/) are filters that are applied to the input channels.

This approach has the benefit of accuracy, the precise rolloff for each channel can be reversed independently.

However it does carry some downsides

* accessibility: DSP needs to be available before decoding which is not something most people have access to
* complexity: such filters are harder to create and verify
* gain management: input channels typically reach full range so any filtering means losing (digital) signal level

#### Post Bass Management (or Output) BEQDesigner

All BEQs published on [avsforum](https://www.avsforum.com/forum/113-subwoofers-bass-transducers/2995212-bass-eq-filtered-movies.html) are filters that are applied to the subwoofer output channel.

This approach is the polar opposite of input BEQ in that it is:

* relatively simple
* accessible
* at relatively low risk (of clipping)
* will not be as accurate for some tracks as the filter will be dominated by the behaviour of the LFE channel

#### Does per channel accuracy matter?

There are certainly film soundtracks that have markedly different rolloffs per channel that, on paper, will benefit from input BEQ. However whether these differences are substantial enough to be audible is a question left to the reader to answer.

### Is BEQ safe?

If applied correctly then yes as the filters are simply recovering content that is already on the disk albeit at a very low level. 

!!! warning
    Applying a BEQ filter to the wrong content could result in egregious clipping at low frequencies and hence can pose a risk to your subs if combined with high playback levels. 
    Always take care to ensure you're using the right filter for the track you're playing.

### Does BEQ just boost noise?

Generally speaking, no, real content is restored to audible/tactile playback levels. However there is some risk of this, particularly with older (e.g. 80s and earlier) soundtracks. 
