"""
https://gist.github.com/endolith/4982787
Calculate the f multipler and Q values for implementing Nth-order Bessel
filters as biquad second-order sections.

Based on description at http://freeverb3.sourceforge.net/iir_filter.shtml

For highpass filter, use fc/fmultiplier instead of fc*fmultiplier

Originally I back-calculated from the denominators produced by the bessel()
filter design tool, which stops at order 25.

Then I made a function to output Bessel polynomials directly, which can be
used to calculate Q, but not f.  (TODO: Create real Bessel filters directly
and calculate f.)  The two methods disagree with each other somewhat above
8th order.  I'm not sure which is more accurate.

Also, these are bilinear-transformed, so they're only good below fs/4
(https://gist.github.com/endolith/4964212)

Created on Mon Feb 18 21:34:15 2013

"""

from __future__ import division

from numpy import roots, reshape, sqrt, poly, polyval
from scipy.special import factorial
from scipy.optimize import brentq
from scipy.signal import bessel
from scipy.signal._filter_design import _cplxpair


MIN_ORDER = 1
MAX_ORDER = 26

def bessel_poly(n, reverse=False):
    """Return the Bessel polynomial of degree `n`

    If `reverse` is true, a reverse Bessel polynomial is output.

    Output is a list of coefficients:
    [1]                   = 1
    [1,  1]               = 1*s   +  1
    [1,  3,  3]           = 1*s^2 +  3*s   +  3
    [1,  6, 15, 15]       = 1*s^3 +  6*s^2 + 15*s   +  15
    [1, 10, 45, 105, 105] = 1*s^4 + 10*s^3 + 45*s^2 + 105*s + 105
    etc.

    Output is a Python list of arbitrary precision long ints, so n is only
    limited by your hardware's memory.

    Sequence is http://oeis.org/A001498 , and output can be confirmed to
    match http://oeis.org/A001498/b001498.txt :

    i = 0
    for n in xrange(51):
        for x in bessel_poly(n, reverse=True):
            print i, x
            i += 1

    """
    out = []
    for k in range(n + 1):
        num = factorial(2 * n - k, exact=True)
        den = 2 ** (n - k) * factorial(k, exact=True) * factorial(n - k, exact=True)
        out.append(num // den)

    if reverse:
        return list(reversed(out))
    else:
        return out


"""
Original:
    
1: ---
2: 0.58
3: --- 0.69
4: 0.81 0.52
5: ---- 0.92 0.56
6: 1.02 0.61 0.51
"""
print("Q for N = ")
for n in range(MIN_ORDER, MAX_ORDER):
    print(str(n).rjust(2) + ":", end=' ')
    #    b, a = bessel(n, 1, analog=True)
    a = bessel_poly(n, reverse=True)
    p = _cplxpair(roots(a))  # Poles, sorted into conjugate pairs
    if n % 2:
        # Odd-order, has a 1st-order stage
        print('-' * 14, end=' ')  # 1st-order stages don't have a Q
        # Remove 1st-order stage (single real pole at the end)
        p = reshape(p[:-1], (-1, 2))
    else:
        # Even-order, is all 2nd-order stages
        p = reshape(p, (-1, 2))

    for section in reversed(p):
        a = poly(section)  # Convert back into a polynomial
        """
        Polynomial is
             s^2 + wo/Q*s + wo^2 = 
        a[0]*s^2 + a[1]*s + a[2]
        so Q is:
        """
        print(str(sqrt(a[2]) / a[1]).ljust(14), end=' ')
    print()
print()

"""
The f requires two steps.  First calculate the f multiplier for each biquad
that produces a normalized Bessel filter (normalized so the asymptotes match 
a Butterworth)

With these settings, an LPF and HPF have the same phase curve vs frequency 
("phase-matched")

Numbers with asterisks are 1st-order filters
"""
print("f multiplier to get same asymptotes as Butterworth (LPF and HPF phase-matched), for N = ")
for n in range(MIN_ORDER, MAX_ORDER):
    print(str(n).rjust(2) + ":", end=' ')
    b, a = bessel(n, 1, analog=True)
    p = _cplxpair(roots(a))  # Poles, sorted into conjugate pairs
    if n % 2:
        # Odd-order, has a 1st-order stage
        print(str(abs(p[-1])) + '*'.ljust(14)),  # 1st-order stage
        # Remove 1st-order stage (single real pole at the end)
        p = reshape(p[:-1], (-1, 2))
    else:
        # Even-order, is all 2nd-order stages
        p = reshape(p, (-1, 2))

    for section in reversed(p):
        a = poly(section)  # Convert back into a polynomial
        """
        Polynomial is
             s^2 + wo/Q*s + wo^2 = 
        a[0]*s^2 + a[1]*s + a[2]
        so wo is sqrt(a[2]):
        """
        print(str(sqrt(a[2])).ljust(15), end=' ')
    print()
print()
"""
Second, measure the point at which the frequency response of the 
normalized filter = -3 dB, and calculate the frequency multiplier to 
shift it so that the -3 dB point is at the desired frequency.

This then matches the original values:

1: 1.00
2: 1.27
3: 1.32 1.45
4: 1.60 1.43
5: 1.50 1.76 1.56
6: 1.90 1.69 1.60
"""
print("f multiplier to get -6 dB at fc, for N = ")
for n in range(MIN_ORDER, MAX_ORDER):
    print(str(n).rjust(2) + ":", end=' ')
    print('[', end='')
    b, a = bessel(n, 1, analog=True)
    p = _cplxpair(roots(a))  # Poles, sorted into conjugate pairs

    # Measure frequency at which magnitude response = -6 dB
    db_point = 10**(-6/20)

    def H(w):
        """Output 0 when magnitude of frequency response is -6 dB"""
        # From scipy.signal.freqs:
        s = 1j * w
        return abs(polyval(b, s) / polyval(a, s)) - db_point


    w = brentq(H, 0, 5)

    # Invert to get frequency multiplier
    m = 1.0 / w

    if n % 2:
        # Odd-order, has a 1st-order stage
        print(str(m * abs(p[-1])), end=', ')  # 1st-order stage
        # Remove 1st-order stage (single real pole at the end)
        p = reshape(p[:-1], (-1, 2))
    else:
        # Even-order, is all 2nd-order stages
        p = reshape(p, (-1, 2))

    for section in reversed(p):
        a = poly(section)  # Convert back into a polynomial
        """
        Polynomial is
             s^2 + wo/Q*s + wo^2 = 
        a[0]*s^2 + a[1]*s + a[2]
        so wo is sqrt(a[2])

        then multiply by m so it's -3 dB
        """
        print(str(m * sqrt(a[2])).ljust(15), end=', ')
    print(']')
print()
