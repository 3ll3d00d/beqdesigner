'''
Optimising a filter for device profiles (design/device-profiles.md, part b): new frequency, Q and gain for its peaking and
shelf filters, so that the filter as each chosen device stores it is as close as it can be to the filter as designed.

The search is beqforge's (`beqoptimiser.optimise_parameters`): it moves each parameter only a little (by default ±1%
frequency, ±2% Q, ±0.1 dB gain), keeps the results on the filter dialog's own grid (0.01 Hz, 0.0001, 0.01 dB), and
minimises the worst error over all the profiles at once, so a change that helps one device cannot quietly cost another.
A shelf stacked n times is n identical sections, which stay identical. Every other filter (a pass filter, a gain) is held
as it is but counts in the response.

The result is still parameters, so it saves, exports and publishes like any filter. It is applied only when beqforge
publishes it (a `replacement` or an `improvement`) and no chosen profile's error is larger than it was.

Qt-free.
'''
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from model.device_profile import DeviceProfile, RATES
from model.iir import HighShelf, LowShelf, PeakingEQ

SEARCHABLE = {PeakingEQ: 'PeakingEQ', LowShelf: 'LowShelf', HighShelf: 'HighShelf'}


def _kind(f) -> Optional[str]:
    return SEARCHABLE.get(type(f))


def to_sections(filters: Sequence) -> List:
    ''' The cascade beqforge searches: each peaking or shelf filter as Sections (a shelf x n as n), the rest Fixed. '''
    from beqoptimiser import Fixed, Section
    sections = []
    for f in filters:
        kind = _kind(f)
        if kind is not None:
            sections += [Section(kind, float(f.freq), float(f.q), float(f.gain))] * (f.count if kind != 'PeakingEQ' else 1)
        else:
            by_rate = {rate: f.resample(rate).get_sos() or [] for rate in RATES}
            for i in range(len(by_rate[RATES[0]])):
                sections.append(Fixed.of({rate: by_rate[rate][i] for rate in RATES}))
    return sections


def from_sections(filters: Sequence, sections: Sequence) -> List[Tuple[object, object]]:
    '''
    The filters with the parameters beqforge found, as (original, new) for each one that changed; the new one has the
    original's id and sample rate, so it replaces it.
    '''
    changed, at = [], 0
    for f in filters:
        kind = _kind(f)
        if kind is None:
            at += len(f.get_sos() or [])
            continue
        new = sections[at]
        at += 1 if kind == 'PeakingEQ' else f.count
        if (new.freq, new.q, new.gain) == (f.freq, f.q, f.gain):
            continue
        if kind == 'PeakingEQ':
            replacement = PeakingEQ(f.fs, new.freq, new.q, new.gain, f_id=f.id)
        else:
            replacement = type(f)(f.fs, new.freq, new.q, new.gain, count=f.count, f_id=f.id)
        changed.append((f, replacement))
    return changed


@dataclass(frozen=True)
class ProfileOutcome:
    profile: DeviceProfile
    before_db: float
    after_db: Optional[float]   # None: no candidate (nothing to publish)

    @property
    def worse(self) -> bool:
        return self.after_db is not None and self.after_db > self.before_db


@dataclass(frozen=True)
class Optimisation:
    outcome: str                      # beqforge's: replacement, improvement, within_margin, no_replacement, unresolved
    margin_db: float
    profiles: Tuple[ProfileOutcome, ...]
    changed: Tuple[Tuple[object, object], ...]   # (original filter, optimised filter)

    @property
    def applicable(self) -> bool:
        ''' A published result that changes something and makes no chosen profile worse. '''
        from beqoptimiser.parameters import PUBLISHED_OUTCOMES
        return self.outcome in PUBLISHED_OUTCOMES and bool(self.changed) and not any(p.worse for p in self.profiles)

    def describe(self) -> str:
        ''' What happened, in a sentence. '''
        unrealisable = [p.profile.name for p in self.profiles if p.before_db == float('inf')]
        if unrealisable:
            return f'As {", ".join(unrealisable)} stores it, the filter is unstable or cannot be realised, so it cannot ' \
                   'be compared, and only peaking and shelf filters can be adjusted: check its pass filters.'
        if self.outcome == 'within_margin':
            return f'Every device is already within {self.margin_db:g} dB of the filter as designed: nothing to do.'
        if self.outcome == 'replacement':
            return f'Found new parameters that bring every device within {self.margin_db:g} dB.'
        if self.outcome == 'improvement':
            return 'Found new parameters that reduce the worst error, though not to within ' \
                   f'{self.margin_db:g} dB on every device.'
        if self.outcome == 'unresolved':
            return 'The search could not tell whether a change helps (the errors are too close to call): nothing changed.'
        return 'No small change to the parameters makes the worst device better: nothing changed.'


def optimise_for(filters: Sequence, profiles: Sequence[DeviceProfile], settings=None) -> Optimisation:
    '''
    :param filters: the filter's children, in order (a CompleteFilter is one).
    :raises ValueError: for no profiles, or a filter beqforge cannot model (e.g. an unstable cascade).
    '''
    from beqoptimiser import optimise_parameters
    if not profiles:
        raise ValueError('choose at least one device profile')
    filters = list(filters)
    # cache=False: the result cache identifies its implementation by hashing its source, which a packaged app lacks.
    # errstate: before beqforge 0.2.6 a high-pass, zero at DC in both responses, made its guard error divide by zero
    # (-inf - -inf); only the reported guard error was affected, never the outcome
    import numpy as np
    with np.errstate(divide='ignore', invalid='ignore'):
        result = optimise_parameters(to_sections(filters), profiles=[(p.rate, p.precision()) for p in profiles],
                                     settings=settings, cache=False)
    by_key = {(e.rate, e.precision): e for e in result.profiles}
    outcomes = tuple(ProfileOutcome(p, by_key[(p.rate, p.precision().name)].original_error_db,
                                    by_key[(p.rate, p.precision().name)].candidate_error_db) for p in profiles)
    changed = tuple(from_sections(filters, result.sections)) if result.sections else ()
    return Optimisation(result.outcome, result.margin_db, outcomes, changed)
