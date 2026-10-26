"""Regression tests for two-way shear, ACI 318-25 §22.6."""
from __future__ import annotations

import pytest

from design.common.materials import Concrete
from design.common.shear import bo_rectangular, lambda_s, vc_two_way, vc_two_way_force


def test_interior_punching_shear_at_square_column():
    concrete = Concrete(fc=27.459)
    d = 159.0
    bo = bo_rectangular(c1=700.0, c2=700.0, d=d)

    assert bo == pytest.approx(3436.0)
    assert lambda_s(d) == pytest.approx(1.0)
    assert vc_two_way(bo=bo, d=d, concrete=concrete) == pytest.approx(1.675, rel=0.005)
    assert vc_two_way_force(bo=bo, d=d, concrete=concrete) == pytest.approx(915.0, rel=0.005)


def test_beta_three_makes_second_two_way_expression_govern():
    concrete = Concrete(fc=27.459)
    # alpha_s*d/bo is made large enough that the third expression exceeds
    # the second; beta=3 then controls through the second expression.
    vc = vc_two_way(bo=1000.0, d=100.0, concrete=concrete, beta=3.0)
    expected = 0.17 * (1.0 + 2.0 / 3.0) * concrete.sqrt_fc
    assert vc == pytest.approx(expected)


def test_two_way_size_effect_is_below_one_for_deep_slab():
    assert lambda_s(400.0) < 1.0
