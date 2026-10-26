"""Regression coverage for ACI 318-25 §18.10 wall shear/detailing fixes."""
from __future__ import annotations

import pytest

from design.common.materials import Concrete, Steel
from design.common.shear import alpha_sh
from design.walls.distributed import rho_min_distributed
from design.walls.wall import Wall


def _wall(*, n_walls_sharing: int = 1, seismic: bool = True, n_x_be: int = 3) -> Wall:
    return Wall.rectangular(
        lw=6500,
        tw=300,
        hw=13000,
        concrete=Concrete(fc=27.459),
        steel=Steel(fy=420, grade=60),
        n_y_web=8,
        n_walls_sharing=n_walls_sharing,
        seismic=seismic,
        n_x_be=n_x_be,
    )


def test_alpha_sh_flange_limits():
    assert alpha_sh(bw=300) == pytest.approx(1.0)
    assert alpha_sh(acf=0.5 * 600 * 300, bcf=600, bw=300) == pytest.approx(1.195, abs=0.001)
    assert alpha_sh(acf=10.0 * 600 * 300, bcf=600, bw=300) == pytest.approx(1.2)


def test_wall_shear_limits_include_common_lateral_force_group():
    alone = _wall(n_walls_sharing=1)
    pair = _wall(n_walls_sharing=2)

    assert alone.Vn_max == pytest.approx(8481, abs=2)
    assert alone.Vn_max_individual == pytest.approx(8481, abs=2)
    assert alone.Vn_max_group_per_wall == pytest.approx(8481, abs=2)
    assert pair.Vn_max == pytest.approx(6744, abs=2)
    assert pair.Vn_max_group_per_wall == pytest.approx(6744, abs=2)


def test_distributed_minimums_for_special_and_low_shear_wall():
    special = _wall(seismic=True)
    assert special.capacity().rho_t_min == pytest.approx(0.0025)

    rho_l, rho_t = rho_min_distributed(
        Vu=1.0,
        Acv=6500 * 300,
        fc=27.459,
        special=False,
        db=16.0,
        fy=420.0,
    )
    assert (rho_l, rho_t) == pytest.approx((0.0012, 0.0020))


def test_two_boundary_element_curtains_are_valid_but_one_is_not():
    assert _wall(n_x_be=2).n_x_be == 2
    with pytest.raises(ValueError, match="n_x_be >= 2"):
        _wall(n_x_be=1)
