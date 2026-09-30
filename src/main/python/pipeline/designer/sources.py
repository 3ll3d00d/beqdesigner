'''
Where a DesignRequest's arrays came from -- the WAV column each was loaded from, when it was loaded unchanged -- so a
designer that takes arrays by reference (designer-interface.md §7.1, 1.2) can be sent the file instead of the samples.
DesignRequest itself does not change (§2): this travels beside it, and only to a designer registered as taking it
(pipeline.designer.registry). design/designer-by-reference.md D1.2.
'''
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional


@dataclass(frozen=True)
class ArraySource:
    ''' One column of a WAV file: `channel` is zero-based. '''
    path: str
    channel: int


@dataclass(frozen=True)
class AudioSources:
    ''' The source of mono_mix, and of each array in channels by its label; either may be missing. '''
    mono: Optional[ArraySource] = None
    channels: Mapping[str, ArraySource] = field(default_factory=dict)

    @classmethod
    def from_wavs(cls, mono_path: Optional[str], multichannel_path: Optional[str] = None,
                  labels: Iterable[str] = ()) -> 'AudioSources':
        '''
        :param mono_path: the mono WAV mono_mix was loaded from (column 0).
        :param multichannel_path: the multichannel WAV channels was decomposed from, if any.
        :param labels: channels' labels in column order -- the order Session.load_channels() returns them in.
        '''
        mono = ArraySource(mono_path, 0) if mono_path else None
        channels = {label: ArraySource(multichannel_path, column) for column, label in enumerate(labels)} \
            if multichannel_path else {}
        return cls(mono=mono, channels=channels)
