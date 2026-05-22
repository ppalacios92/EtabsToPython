"""Functional smoke test for the design package.

Exercises every public surface — beams, columns (incl. PMM volume),
walls (incl. biaxial), coupling beams, wall piers, and the optional
plotting helpers.

Run from repo root:
    python -m design._smoke_test                                   # defaults
    python -m design._smoke_test --plot                            # + save PNGs
    python -m design._smoke_test --angles 72 --points 60 --plot    # high-res PMM
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Force UTF-8 stdout on Windows so '°', '§', '—' don't crash the print.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from design import Concrete, Steel, Bar, RectangularSection, WallSection
from design import Beam, Column, Wall
from design.beams.beam import BeamDemands
from design.columns.column import ColumnDemands
from design.walls.wall import WallDemands
from design.sections.reinforcement import (
    RebarLayout, two_layer_bars, perimeter_bars,
)
from design.walls.coupling_beam import CouplingBeam
from design.walls.wall_pier import WallPier


HEADER = "=" * 60


def banner(title: str) -> None:
    print(f"\n{HEADER}\n  {title}\n{HEADER}")


# ----------------------------------------------------------------- #
# BEAM
# ----------------------------------------------------------------- #
def test_beam():
    banner("BEAM — flexure + shear, three-mode design, SMF Ve")
    c = Concrete(fc=28)
    s = Steel(fy=420, grade=60)
    beam = Beam.rectangular(
        bw=300, h=600, cover=40,
        concrete=c, steel=s,
        n_top=2, db_top=20,
        n_bot=4, db_bot=20,
        db_stirrup=10, n_legs=2, s_stirrup=150,
        ln=5.0, seismic=True, Vg=80.0,
        units=6, label="B-1",
    )

    demands = BeamDemands(
        Mu_pos_i=80.0,   Mu_neg_i=320.0, Vu_i=210.0,
        Mu_pos_mid=350.0, Mu_neg_mid=50.0, Vu_mid=80.0,
        Mu_pos_j=80.0,   Mu_neg_j=320.0, Vu_j=205.0,
    )

    beam.summary()
    print()
    beam.print_checks(demands)
    print()
    results = beam.design(demands)
    beam.print_design(results, demands=demands)


# ----------------------------------------------------------------- #
# COLUMN — including PMM volume
# ----------------------------------------------------------------- #
def test_column(plot_dir: Path | None = None, *,
                n_angles: int = 13, n_points: int = 40):
    banner(f"COLUMN — uniaxial P-M, biaxial P-M-M surface  "
           f"(n_angles={n_angles}, n_points_per_curve={n_points})")
    c = Concrete(fc=28)
    s = Steel(fy=420, grade=60)
    group = perimeter_bars(
        b=500, h=500, cover=50, n_x=4, n_y=4,
        bar=Bar(diameter=25), steel=s,
    )
    sec = RectangularSection(b=500, h=500, concrete=c,
                             rebar=RebarLayout(groups=(group,)))
    col = Column(
        section=sec, lu=3.0, k=1.0,
        Av=2 * 71.0, fyt=420, s=100,
        seismic=True, hx=180.0,
    )

    t0 = time.perf_counter()
    cap = col.capacity(biaxial=True, n_angles=n_angles,
                       n_points_per_curve=n_points)
    dt = time.perf_counter() - t0
    print(f"  capacity() built in {dt*1000:.0f} ms "
          f"({n_angles*n_points} (angle,c) evaluations)")
    chk = col.check(ColumnDemands(Pu=2500.0, Mux=200.0, Muy=120.0, Vux=150.0))
    prop = col.design(ColumnDemands(Pu=3000.0, Mux=250.0))

    print(f"  Po             = {cap.Po:.0f} kN")
    print(f"  phi*Pn_max     = {cap.phi_Pn_max:.0f} kN")
    print(f"  To (pure tens.)= {cap.To:.0f} kN")
    print(f"  phi_Vn         = {cap.phi_Vn:.0f} kN")
    print(f"  PM @ theta=0:   max phi_Mn = {max(cap.diagram_x.phi_Mn):.0f} kN-m")
    print(f"  PM @ theta=90:  max phi_Mn = {max(cap.diagram_y.phi_Mn):.0f} kN-m")

    print()
    print("  --- PMM surface ---")
    print(f"  n_angles       = {len(col.surface.angles)}")
    Pu_grid, Mx_grid, My_grid = col.surface.volume(n_Pu=15)
    print(f"  volume shape   = {Mx_grid.shape}  (Pu_slices, theta points)")
    print(f"  Pu axis range  = [{Pu_grid.min():.0f}, {Pu_grid.max():.0f}] kN")

    # Biaxial slice at Pu = 2500 kN
    mx, my = col.surface.at_Pu(2500.0)
    print(f"  Slice @ Pu=2500 envelope: Mx in [{mx.min():.0f}, {mx.max():.0f}]  My in [{my.min():.0f}, {my.max():.0f}]")

    # Demand checks (biaxial)
    print(f"  ratio PMM      = {chk.ratio_pmm:.2f}")
    print(f"  ratio shear    = {chk.ratio_shear:.2f}")
    print(f"  passed         = {chk.passed}")

    # Three-mode design — show envelope
    results = col.design(ColumnDemands(Pu=2500, Mux=200, Muy=120, Vux=150))
    p = results.envelope
    print()
    print(f"  --- design (envelope of 3 modes) ---")
    print(f"  Ash_x required = {p.ash_required_x:.0f} mm²   provided {p.av_provided_x:.0f} mm²"
          f" ({p.n_legs_x} legs φ{p.db_hoop:.0f})")
    print(f"  Ash_y required = {p.ash_required_y:.0f} mm²   provided {p.av_provided_y:.0f} mm²"
          f" ({p.n_legs_y} legs φ{p.db_hoop:.0f})")
    if results.capacity.Ve_capacity:
        print(f"  Ve_capacity    = {results.capacity.Ve_capacity:.0f} kN  "
              f"(Mpr_top = {results.capacity.Mpr_top:.0f} kN·m)")
    print(f"  lo (confined)  = {p.lo:.0f} mm   hx_ok = {p.hx_ok}")

    if plot_dir is not None:
        _plot_column(col, cap, plot_dir)


# ----------------------------------------------------------------- #
# WALL — biaxial (in-plane + out-of-plane)
# ----------------------------------------------------------------- #
def test_wall(plot_dir: Path | None = None, *,
              n_angles: int = 13, n_points: int = 40):
    banner(f"WALL — in-plane PM, biaxial PMM, BE  "
           f"(n_angles={n_angles}, n_points_per_curve={n_points})")
    c = Concrete(fc=35)
    s = Steel(fy=420, grade=60)

    web = perimeter_bars(b=300, h=2400, cover=50, n_x=2, n_y=10,
                         bar=Bar(diameter=12), steel=s)
    be_top = perimeter_bars(b=400, h=600, cover=50, n_x=3, n_y=5,
                            bar=Bar(diameter=22), steel=s,
                            center=(0.0, 2400 / 2 - 300))
    be_bot = perimeter_bars(b=400, h=600, cover=50, n_x=3, n_y=5,
                            bar=Bar(diameter=22), steel=s,
                            center=(0.0, -2400 / 2 + 300))
    sec = WallSection(
        lw=4000, tw=300,
        concrete=c, rebar=RebarLayout(groups=(web, be_top, be_bot)),
        be_length_top=600, be_length_bot=600,
        be_thickness_top=400, be_thickness_bot=400,
    )
    wall = Wall(
        section=sec, hw=30_000, lu=3_000,
        rho_t=0.003, rho_l=0.003, fyt=420, seismic=True,
    )

    # In-plane only
    cap = wall.capacity(n_points=n_points)
    chk = wall.check(WallDemands(
        Pu=2000.0, Mu=8000.0, Vu=1200.0,
        delta_u=180.0, sigma_max=8.0,
    ))
    print("  --- In-plane only ---")
    print(f"  phi_Vn         = {cap.phi_Vn:.0f} kN   (cap = {cap.Vn_max:.0f} kN)")
    print(f"  max phi_Mn     = {max(cap.diagram.phi_Mn):.0f} kN-m")
    print(f"  ratio PM       = {chk.ratio_pm:.2f}")
    print(f"  ratio shear    = {chk.ratio_shear:.2f}")
    print(f"  BE disp / stress= {chk.be_required_disp} / {chk.be_required_stress}")
    print(f"  BE length req  = {chk.be_length_required:.0f} mm")
    print(f"  BE thickness min= {chk.be_thickness_min:.0f} mm")
    print(f"  double curtain = {cap.double_curtain_required}")

    # Biaxial — with out-of-plane moment
    t0 = time.perf_counter()
    cap_b = wall.capacity(biaxial=True, n_angles=n_angles, n_points=n_points)
    dt = time.perf_counter() - t0
    print(f"  capacity(biaxial) built in {dt*1000:.0f} ms")
    chk_b = wall.check(WallDemands(
        Pu=2000.0, Mu=8000.0, Mu_out=200.0, Vu=1500.0,
        delta_u=180.0, sigma_max=8.0,
    ))
    print()
    print("  --- Biaxial (with Mu_out = 200 kN-m) ---")
    print(f"  max phi_Mn in-plane    = {max(cap_b.diagram.phi_Mn):.0f} kN-m")
    print(f"  max phi_Mn out-of-plane= {max(cap_b.diagram_out.phi_Mn):.0f} kN-m")
    Pu_grid, Mx_grid, My_grid = cap_b.surface.volume_slices(n_Pu=12) if hasattr(cap_b.surface, 'volume_slices') else cap_b.surface.volume(n_Pu=12)
    print(f"  surface volume shape   = {Mx_grid.shape}")
    print(f"  biaxial ratio PMM      = {chk_b.ratio_pm:.2f}")
    print(f"  passed                 = {chk_b.passed}")

    results = wall.design(WallDemands(Pu=2000.0, Mu=8000.0, Vu=1500.0))
    env = results.envelope
    print()
    print("  --- Design proposal (envelope) ---")
    print(f"  rho_t required = {env.rho_t_required:.4f}")
    print(f"  rho_l required = {env.rho_l_required:.4f}")
    print(f"  BE required    = {env.be_required}")
    print(f"  notes          = {list(env.notes)}")

    if plot_dir is not None:
        _plot_wall(wall, cap_b, plot_dir)


# ----------------------------------------------------------------- #
# COUPLING BEAM & WALL PIER
# ----------------------------------------------------------------- #
def test_coupling_and_pier():
    banner("COUPLING BEAM & WALL PIER")
    c = Concrete(fc=35)
    s = Steel(fy=420)

    # 1) low ln/h, high Vu  → diagonal mandatory
    cb_short = CouplingBeam(ln=1500, h=900, b=300, concrete=c, steel=s)
    cls_short = cb_short.classify(Vu=600.0)
    Vn_diag = cb_short.diagonal_capacity(Avd=4 * 314.0, alpha_deg=25.0)
    print("  Short coupling beam:")
    print(f"    ln/h={cls_short.ln_over_h:.2f}  type={cls_short.type_required}  diag mandatory={cls_short.diagonal_mandatory}")
    print(f"    Vn diagonal (4 phi 20 @ 25deg) = {Vn_diag:.0f} kN")
    for n in cls_short.notes:
        print(f"      - {n}")

    # 2) long ln/h → behave as SMF beam
    cb_long = CouplingBeam(ln=4000, h=600, b=300, concrete=c, steel=s)
    cls_long = cb_long.classify(Vu=200.0)
    print(f"  Long coupling beam ln/h={cls_long.ln_over_h:.2f}  type={cls_long.type_required}")

    # 3) wall pier — column-like vs wall-like
    pier_col = WallPier(hw=2700, lw=600, bw=300).classify()
    pier_wall = WallPier(hw=2700, lw=1800, bw=300).classify()
    print("  Wall pier classification:")
    print(f"    short pier lw/bw=2 -> {pier_col.classification} ({pier_col.design_path})")
    print(f"    long  pier lw/bw=6 -> {pier_wall.classification} ({pier_wall.design_path})")


# ----------------------------------------------------------------- #
# Plot helpers (only when --plot)
# ----------------------------------------------------------------- #
def _plot_column(col, cap, plot_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from design.plotting import (
        plot_pm_diagram, plot_pmm_volume, plot_pmm_slice, plot_pmm_with_demand,
    )

    plot_dir.mkdir(exist_ok=True, parents=True)

    plot_pm_diagram(cap.diagram_x)
    plt.savefig(plot_dir / "col_pm_x.png", dpi=110, bbox_inches="tight")
    plt.close()

    plot_pmm_slice(cap.surface, Pu=2500, demand=(200, 120))
    plt.savefig(plot_dir / "col_slice_Pu2500.png", dpi=110, bbox_inches="tight")
    plt.close()

    plot_pmm_volume(cap.surface, n_Pu=25)
    plt.savefig(plot_dir / "col_pmm_volume.png", dpi=110, bbox_inches="tight")
    plt.close()

    plot_pmm_with_demand(cap.surface, Pu=2500, Mux=200, Muy=120)
    plt.savefig(plot_dir / "col_pmm_with_demand.png", dpi=110, bbox_inches="tight")
    plt.close()


def _plot_wall(wall, cap, plot_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from design.plotting import (
        plot_pm_diagram, plot_pmm_volume, plot_pmm_slice,
    )

    plot_dir.mkdir(exist_ok=True, parents=True)

    plot_pm_diagram(cap.diagram)
    plt.savefig(plot_dir / "wall_pm_in_plane.png", dpi=110, bbox_inches="tight")
    plt.close()

    if cap.diagram_out is not None:
        plot_pm_diagram(cap.diagram_out)
        plt.savefig(plot_dir / "wall_pm_out_of_plane.png", dpi=110, bbox_inches="tight")
        plt.close()

    if cap.surface is not None:
        plot_pmm_slice(cap.surface, Pu=2000, demand=(8000, 200))
        plt.savefig(plot_dir / "wall_slice_Pu2000.png", dpi=110, bbox_inches="tight")
        plt.close()

        plot_pmm_volume(cap.surface, n_Pu=20)
        plt.savefig(plot_dir / "wall_pmm_volume.png", dpi=110, bbox_inches="tight")
        plt.close()


# ----------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test for the design package."
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Also save PNGs of every interaction figure to ./_smoke_plots/",
    )
    parser.add_argument(
        "--angles", type=int, default=13,
        help="Number of neutral-axis angles for the PMM surface (0..180°). "
             "37 -> 5° step; 73 -> 2.5° step. Default: 13.",
    )
    parser.add_argument(
        "--points", type=int, default=40,
        help="Number of points per individual P-M curve (c sweep). Default: 40.",
    )
    args = parser.parse_args()

    plot_dir = Path("_smoke_plots") if args.plot else None
    if plot_dir is not None:
        print(f"Plot mode ON — figures will be saved to {plot_dir.resolve()}")

    print(f"PMM resolution: n_angles={args.angles}  n_points_per_curve={args.points}")
    print(f"  -> {args.angles} curves in 0..180°  "
          f"({(180/(args.angles-1) if args.angles>1 else 0):.2f}° step), "
          f"{args.angles*2} points in the closed 360° contour after mirror.")

    test_beam()
    test_column(plot_dir=plot_dir, n_angles=args.angles, n_points=args.points)
    test_wall(plot_dir=plot_dir, n_angles=args.angles, n_points=args.points)
    test_coupling_and_pier()

    banner("DONE")
    if plot_dir is not None:
        print(f"  Plots saved to: {plot_dir.resolve()}")
        for p in sorted(plot_dir.glob("*.png")):
            print(f"    - {p.name}")


if __name__ == "__main__":
    main()
