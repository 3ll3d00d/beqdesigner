import numpy as np

from acoustics.standards.iec_61260_1_2014 import index_of_frequency, REFERENCE_FREQUENCY
from acoustics.standards import iec_61260_1_2014

REFERENCE_PRESSURE = 2.0e-5


def exact_center_frequency(frequency=None, fraction=1, n=None, ref=REFERENCE_FREQUENCY):
    """Exact center frequency.

    :param frequency: Frequency within the band.
    :param fraction: Band designator.
    :param n: Index of band.
    :param ref: Reference frequency.
    :return: Exact center frequency for the given frequency or band index.

    .. seealso:: :func:`iec_61260_1_2014.exact_center_frequency`
    .. seealso:: :func:`iec_61260_1_2014.index_of_frequency`

    """
    if frequency is not None:
        n = index_of_frequency(frequency, fraction=fraction, ref=ref)
    return iec_61260_1_2014.exact_center_frequency(n, fraction=fraction, ref=ref)


def nominal_center_frequency(frequency=None, fraction=1, n=None):
    """Nominal center frequency.

    :param frequency: Frequency within the band.
    :param fraction: Band designator.
    :param n: Index of band.
    :returns: The nominal center frequency for the given frequency or band index.

    .. seealso:: :func:`iec_61260_1_2014.exact_center_frequency`
    .. seealso:: :func:`iec_61260_1_2014.nominal_center_frequency`

    .. note:: Contrary to the other functions this function silently assumes 1000 Hz reference frequency.

    """
    center = exact_center_frequency(frequency, fraction, n)
    return iec_61260_1_2014.nominal_center_frequency(center, fraction)


def lower_frequency(frequency=None, fraction=1, n=None, ref=REFERENCE_FREQUENCY):
    """Lower band-edge frequency.

    :param frequency: Frequency within the band.
    :param fraction: Band designator.
    :param n: Index of band.
    :param ref: Reference frequency.
    :returns: Lower band-edge frequency for the given frequency or band index.

    .. seealso:: :func:`iec_61260_1_2014.exact_center_frequency`
    .. seealso:: :func:`iec_61260_1_2014.lower_frequency`

    """
    center = exact_center_frequency(frequency, fraction, n, ref=ref)
    return iec_61260_1_2014.lower_frequency(center, fraction)


def upper_frequency(frequency=None, fraction=1, n=None, ref=REFERENCE_FREQUENCY):
    """Upper band-edge frequency.

    :param frequency: Frequency within the band.
    :param fraction: Band designator.
    :param n: Index of band.
    :param ref: Reference frequency.
    :returns: Upper band-edge frequency for the given frequency or band index.

    .. seealso:: :func:`iec_61260_1_2014.exact_center_frequency`
    .. seealso:: :func:`iec_61260_1_2014.upper_frequency`

    """
    center = exact_center_frequency(frequency, fraction, n, ref=ref)
    return iec_61260_1_2014.upper_frequency(center, fraction)


class Frequencies:
    """
    Object describing frequency bands.
    """

    def __init__(self, center, lower, upper, bandwidth=None):
        self.center = np.asarray(center)
        """
        Center frequencies.
        """

        self.lower = np.asarray(lower)
        """
        Lower frequencies.
        """

        self.upper = np.asarray(upper)
        """
        Upper frequencies.
        """

        self.bandwidth = np.asarray(bandwidth) if bandwidth is not None else np.asarray(self.upper) - np.asarray(
            self.lower)
        """
        Bandwidth.
        """

    def __repr__(self):
        return f"Frequencies({self.center})"


class EqualBand(Frequencies):
    """
    Equal bandwidth spectrum. Generally used for narrowband data.
    """

    def __init__(self, center=None, fstart=None, fstop=None, nbands=None, bandwidth=None):
        """

        :param center: Vector of center frequencies.
        :param fstart: First center frequency.
        :param fstop: Last center frequency.
        :param nbands: Amount of frequency bands.
        :param bandwidth: Bandwidth of bands.

        """

        if center is not None:
            try:
                nbands = len(center)
            except TypeError:
                center = [center]
                nbands = 1

            u = np.unique(np.diff(center).round(decimals=3))
            n = len(u)
            if n == 1:
                bandwidth = u
            elif n > 1:
                raise ValueError("Given center frequencies are not equally spaced.")
            else:
                pass
            fstart = center[0]  # - bandwidth/2.0
            fstop = center[-1]  # + bandwidth/2.0
        elif fstart is not None and fstop is not None and nbands:
            bandwidth = (fstop - fstart) / (nbands - 1)
        elif fstart is not None and fstop is not None and bandwidth:
            nbands = round((fstop - fstart) / bandwidth) + 1
        elif fstart is not None and bandwidth and nbands:
            fstop = fstart + nbands * bandwidth
        elif fstop is not None and bandwidth and nbands:
            fstart = fstop - (nbands - 1) * bandwidth
        else:
            raise ValueError("Insufficient parameters. Cannot determine fstart, fstop, bandwidth.")

        center = fstart + np.arange(0, nbands) * bandwidth  # + bandwidth/2.0
        upper = fstart + np.arange(0, nbands) * bandwidth + bandwidth / 2.0
        lower = fstart + np.arange(0, nbands) * bandwidth - bandwidth / 2.0

        super(EqualBand, self).__init__(center, lower, upper, bandwidth)

    def __repr__(self):
        return f"EqualBand({self.center})"


class OctaveBand(Frequencies):
    """Fractional-octave band spectrum.
    """

    def __init__(self, center=None, fstart=None, fstop=None, nbands=None, fraction=1, reference=REFERENCE_FREQUENCY):

        if center is not None:
            try:
                nbands = len(center)
            except TypeError:
                center = [center]
            center = np.asarray(center)
            indices = index_of_frequency(center, fraction=fraction, ref=reference)
        elif fstart is not None and fstop is not None:
            nstart = index_of_frequency(fstart, fraction=fraction, ref=reference)
            nstop = index_of_frequency(fstop, fraction=fraction, ref=reference)
            indices = np.arange(nstart, nstop + 1)
        elif fstart is not None and nbands is not None:
            nstart = index_of_frequency(fstart, fraction=fraction, ref=reference)
            indices = np.arange(nstart, nstart + nbands)
        elif fstop is not None and nbands is not None:
            nstop = index_of_frequency(fstop, fraction=fraction, ref=reference)
            indices = np.arange(nstop - nbands, nstop)
        else:
            raise ValueError("Insufficient parameters. Cannot determine fstart and/or fstop.")

        center = exact_center_frequency(None, fraction=fraction, n=indices, ref=reference)
        lower = lower_frequency(center, fraction=fraction)
        upper = upper_frequency(center, fraction=fraction)
        bandwidth = upper - lower
        nominal = nominal_center_frequency(None, fraction, indices)

        super(OctaveBand, self).__init__(center, lower, upper, bandwidth)

        self.fraction = fraction
        """Fraction of fractional-octave filter.
        """

        self.reference = reference
        """Reference center frequency.
        """

        self.nominal = nominal
        """Nominal center frequencies.
        """

    def __repr__(self):
        return f"OctaveBand({self.center})"


def integrate_bands(data, a, b):
    """
    Reduce frequency resolution of power spectrum. Merges frequency bands by integration.

    :param data: Vector with narrowband powers.
    :param a: Instance of :class:`Frequencies`.
    :param b: Instance of :class:`Frequencies`.

    .. note:: Needs rewriting so that the summation goes over axis=1.

    """

    try:
        if b.fraction % a.fraction:
            raise NotImplementedError("Non-integer ratio of fractional-octaves are not supported.")
    except AttributeError:
        pass

    lower, _ = np.meshgrid(b.lower, a.center)
    upper, _ = np.meshgrid(b.upper, a.center)
    _, center = np.meshgrid(b.center, a.center)

    return ((lower < center) * (center <= upper) * data[..., None]).sum(axis=-2)


def fractional_octaves(f, p, fraction=3):
    """Calculate level per 1/N-octave in frequency domain`.
    """
    if fraction < 6:
        start = 5.0
    elif fraction == 6:
        start = 8.0
    elif fraction <= 12:
        start = 15.0
    elif fraction <= 24:
        start = 20.0
    else:
        raise ValueError(f'Unknown fraction {fraction}')
    stop = min(20000.0, f[-1])
    fob = OctaveBand(fstart=start, fstop=stop, fraction=fraction)
    fnb = EqualBand(f)
    power = integrate_bands(p, fnb, fob)
    return fob, power
