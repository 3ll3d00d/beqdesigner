from pytest import approx

from model.iir import LinkwitzTransform


def test_lt():
    lt = LinkwitzTransform(48000, 50.0, 1.0, 35.0, 0.707)
    a0, a1, a2 = lt.a
    b0, b1, b2 = lt.b
    assert a0 == approx(1.0)
    assert a1 == approx(-1.993519840635900, rel=1e-12)
    assert a2 == approx(0.993540762888275, rel=1e-12)
    assert b0 == approx(1.000037740241020, rel=1e-12)
    assert b1 == approx(-1.993508952524960, rel=1e-12)
    assert b2 == approx(0.993513910758182, rel=1e-12)
