"""Regression tests for ACI 318-25 Table 22.5.5.1 one-way ``Vc``."""
from __future__ import annotations

from math import sqrt

import pytest

from design.common.materials import Concrete
from design.common.shear import lambda_s, vc_one_way


def test_vc_one_way_without_minimum_shear_reinforcement_uses_lambda_s():
    concrete = Concrete(fc=27.459)
    vc = vc_one_way(
        b=1000.0, d=169.0, rho_w=0.003346, concrete=concrete,
        av_ge_av_min=False,
    )
    expected = 0.66 * 1.0 * 0.003346 ** (1.0 / 3.0) * sqrt(27.459) * 1000.0 * 169.0 / 1000.0
    assert lambda_s(169.0) == pytest.approx(1.0)
    assert vc == pytest.approx(expected, rel=0.005)
    # The stated expression with the stated inputs evaluates to 87.42084 kN.
    assert vc == pytest.approx(87.42084, rel=0.005)


def test_vc_one_way_with_minimum_shear_reinforcement_takes_lesser_expression():
    concrete = Concrete(fc=27.459)
    b, d, rho_w = 700.0, 637.0, 0.005113
    vc = vc_one_way(
        b=b, d=d, rho_w=rho_w, concrete=concrete, av_ge_av_min=True,
    )
    expression_1 = 0.17 * sqrt(27.459) * b * d / 1000.0
    expression_2 = 0.66 * rho_w ** (1.0 / 3.0) * sqrt(27.459) * b * d / 1000.0
    assert expression_2 < expression_1
    assert vc == pytest.approx(expression_2, rel=1.0e-12)
    assert vc == pytest.approx(267.0, rel=0.01)


def test_vc_one_way_is_limited_by_table_cap():
    concrete = Concrete(fc=27.459)
    b, d = 700.0, 637.0
    vc = vc_one_way(
        b=b, d=d, rho_w=0.1, concrete=concrete, av_ge_av_min=True,
        Nu=1.0e9, Ag=1.0,
    )
    cap = 0.42 * sqrt(27.459) * b * d / 1000.0
    assert vc == pytest.approx(cap, rel=1.0e-12)
