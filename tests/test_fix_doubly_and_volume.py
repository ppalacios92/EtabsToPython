"""Regression tests: doubly reinforced flexure by strain compatibility and closed P-M-M volume.

- mn_doubly with equal top and bottom bars in a thin slab: the old "symmetric" branch
  (Mn = As fy (d - d')) dropped the concrete block and underestimated Mn; strain
  compatibility (§22.2) gives at least the singly reinforced value.
- Surface.volume(close_top=True) ends with a row Mx = My = 0 at phi*Pn,max.
"""
import numpy as np
import pytest

from design import Column, Concrete, Steel
from design.beams.flexure import mn_doubly, mn_singly


@pytest.fixture
def mats():
    return Concrete(fc=27.459), Steel(fy=411.88)


def test_symmetric_slab_not_below_singly(mats):
    c, s = mats
    as_ = 5 * np.pi * 6.0 ** 2                      # 5 phi12 per metre, both faces
    mn = mn_doubly(b=1000, d=169, d_prime=31, As=as_, As_prime=as_, concrete=c, steel=s)
    assert mn >= mn_singly(b=1000, d=169, As=as_, concrete=c, steel=s)
    assert mn > as_ * s.fy * (169 - 31) / 1e6 * 1.15   # well above the old As fy (d - d')


def test_doubly_equilibrium_and_yielding_compression(mats):
    c, s = mats
    # deep beam with large compression steel: result between singly and the yielded-compression bound
    mn = mn_doubly(b=300, d=540, d_prime=60, As=2280, As_prime=1140, concrete=c, steel=s)
    assert mn_singly(b=300, d=540, As=2280, concrete=c, steel=s) <= mn <= 2280 * s.fy * (540 - 30) / 1e6


def test_volume_closed_top(mats):
    c, s = mats
    col = Column.rectangular(b=700, h=700, cover=40, concrete=c, steel=s, n_x=6, n_y=6, db_long=22,
                             db_stirrup=12, n_legs_x=4, n_legs_y=4, s_stirrup=100, lu=3300)
    cap = col.capacity(biaxial=True)
    pu, mx, my = cap.surface.volume(n_Pu=15)
    assert np.allclose(mx[-1], 0.0) and np.allclose(my[-1], 0.0)
    assert np.isclose(pu[-1, 0], pu[-2, 0])
    pu2, mx2, _ = cap.surface.volume(n_Pu=15, close_top=False)
    assert mx2.shape[0] == mx.shape[0] - 1
