"""Build the wall example notebooks (simple + complete).

Run from repo root:
    python -m design._build_wall_notebooks

Generates:
    design/example_wall_simple.ipynb
    design/example_wall.ipynb
"""
from __future__ import annotations
import json
from pathlib import Path


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split(src),
    }


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _split(src),
    }


def _split(src: str) -> list[str]:
    lines = src.splitlines(keepends=True)
    return lines if lines else [src]


def write(path: Path, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {path}")


# ====================================================================== #
# SIMPLE NOTEBOOK — quick path with visual checks
# ====================================================================== #
def build_simple() -> list[dict]:
    cells: list[dict] = []

    cells.append(md(
        "# Wall design — quick path\n"
        "\n"
        "Strict step-by-step. We always **plot the original section** before\n"
        "any design call, and **plot the final section** after, so we can\n"
        "verify visually that the rebar layout is clean (no stacked bars).\n"
        "\n"
        "Convention: the wall starts at its web. When there is no BE, the\n"
        "distributed reinforcement runs the full `lw`. When a BE is added,\n"
        "web bars inside the new BE rectangle are dropped and the BE bars\n"
        "take their place."
    ))

    cells.append(md(
        "## 0. Path setup\n"
        "\n"
        "This notebook lives inside `design/`. We add the repo root to\n"
        "`sys.path` so `from design import ...` resolves regardless of\n"
        "where Jupyter was launched."
    ))
    cells.append(code(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "repo_root = Path.cwd().parent if Path.cwd().name == 'design' else Path.cwd()\n"
        "if str(repo_root) not in sys.path:\n"
        "    sys.path.insert(0, str(repo_root))\n"
        "print('repo root:', repo_root)"
    ))

    cells.append(code(
        "import matplotlib.pyplot as plt\n"
        "from design import Concrete, Steel, Bar, WallSection, Wall\n"
        "from design.walls import WallDemands\n"
        "from design.sections.reinforcement import RebarLayout, perimeter_bars\n"
        "\n"
        "DISPLAY_UNITS = 5    # kN_mm_C — keep moments in kN.mm for readability"
    ))

    cells.append(md(
        "## 1. Build the section — no boundary elements yet\n"
        "\n"
        "Web bars run edge to edge of the wall length `lw`. We will see\n"
        "later that, when the design forces a BE, these bars get trimmed\n"
        "at the BE boundary automatically."
    ))
    cells.append(code(
        "concrete = Concrete(fc=35)\n"
        "steel    = Steel(fy=420, grade=60)\n"
        "\n"
        "web = perimeter_bars(\n"
        "    b=300, h=4000, cover=50, n_x=2, n_y=14,\n"
        "    bar=Bar(diameter=12), steel=steel,\n"
        ")\n"
        "section = WallSection(\n"
        "    lw=4000, tw=300,\n"
        "    concrete=concrete, rebar=RebarLayout(groups=(web,)),\n"
        ")\n"
        "print(section)\n"
        "print('Ag =', section.gross_area(), 'mm^2  |  n bars =', sum(len(g.bars) for g in section.rebar.groups))"
    ))

    cells.append(md("### Plot the ORIGINAL section (visual sanity check)"))
    cells.append(code(
        "wall = Wall(\n"
        "    section=section, hw=30_000, lu=3_000,\n"
        "    rho_t=0.0025, rho_l=0.0025,\n"
        "    fyt=420, seismic=True, label='W-1',\n"
        "    units=DISPLAY_UNITS,\n"
        ")\n"
        "fig, ax = plt.subplots(figsize=(3, 8))\n"
        "wall.plot.section(ax=ax)\n"
        "plt.show()"
    ))

    cells.append(md("## 2. Capacity of the original geometry"))
    cells.append(code(
        "wall.run()\n"
        "wall.summary()"
    ))

    cells.append(md("### Interaction diagram (in-plane) — original geometry"))
    cells.append(code(
        "fig, ax = plt.subplots(figsize=(6, 6))\n"
        "wall.plot.pm(plane='in-plane', ax=ax)\n"
        "plt.show()"
    ))

    cells.append(md("### PMM volume — original geometry"))
    cells.append(code(
        "fig = plt.figure(figsize=(7, 6))\n"
        "ax = fig.add_subplot(111, projection='3d')\n"
        "wall.plot.pmm_volume(ax=ax, n_Pu=20)\n"
        "plt.show()"
    ))

    cells.append(md(
        "## 3. Define demands\n"
        "\n"
        "`delta_u` and `sigma_max` come from the analysis; they control\n"
        "the §18.10.6 BE-required checks."
    ))
    cells.append(code(
        "demands = WallDemands(\n"
        "    Pu=2000.0, Mu=8000.0, Vu=1500.0,\n"
        "    delta_u=180.0, sigma_max=8.0,\n"
        ")\n"
        "chk = wall.check(demands)\n"
        "print(f'ratio PM        = {chk.ratio_pm:.3f}')\n"
        "print(f'ratio shear     = {chk.ratio_shear:.3f}')\n"
        "print(f'BE disp/stress  = {chk.be_required_disp} / {chk.be_required_stress}')\n"
        "print(f'c at demand     = {chk.c_at_demand:.0f} mm')\n"
        "print(f'passed          = {chk.passed}')"
    ))

    cells.append(md(
        "## 4. Design with auto_update — BE is proposed and added\n"
        "\n"
        "If §18.10.6 triggers, `wall.design()` proposes BE geometry,\n"
        "rebuilds the section, and re-runs the PMM. Iterates until the\n"
        "BE length stabilises."
    ))
    cells.append(code(
        "results = wall.design(demands, auto_update=True)\n"
        "print(f'iterations: {results.iterations}, converged: {results.converged}')\n"
        "for i, w in enumerate(results.history):\n"
        "    print(f'  iter {i}: BE top = {w.section.be_length_top:>4.0f} mm, BE bot = {w.section.be_length_bot:>4.0f} mm')"
    ))

    cells.append(md("## 5. Plot the FINAL section — verify no stacked bars"))
    cells.append(code(
        "final = results.final_wall\n"
        "fig, ax = plt.subplots(figsize=(3, 8))\n"
        "final.plot.section(ax=ax)\n"
        "plt.show()\n"
        "\n"
        "n_total = sum(len(g.bars) for g in final.section.rebar.groups)\n"
        "n_web   = sum(len(g.bars) for g in final.section.web_rebar().groups)\n"
        "n_top   = sum(len(g.bars) for g in (final.section.be_top_rebar() or RebarLayout()).groups)\n"
        "n_bot   = sum(len(g.bars) for g in (final.section.be_bot_rebar() or RebarLayout()).groups)\n"
        "print(f'bars total = {n_total}  ({n_web} web + {n_top} BE top + {n_bot} BE bot)')"
    ))

    cells.append(md("## 6. Compare interaction diagrams: original vs final"))
    cells.append(code(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
        "results.original_wall.plot.pm(plane='in-plane', ax=axes[0])\n"
        "axes[0].set_title('ORIGINAL — no BE')\n"
        "final.plot.pm(plane='in-plane', ax=axes[1])\n"
        "axes[1].set_title('FINAL — with BE')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(md("## 7. Inspect the BE as a Column"))
    cells.append(code(
        "if final.has_boundary_elements:\n"
        "    be = final.be_top\n"
        "    print(be)\n"
        "    fig, ax = plt.subplots(figsize=(5, 5))\n"
        "    be.plot.section(ax=ax)\n"
        "    plt.show()\n"
        "else:\n"
        "    print('No BE was added — demands did not trigger §18.10.6.')"
    ))

    cells.append(md("## 8. Envelope summary"))
    cells.append(code(
        "env = results.envelope\n"
        "print(f'rho_t required     = {env.rho_t_required:.4f}')\n"
        "print(f'BE required        = {env.be_required}')\n"
        "print(f'BE length proposed = {env.be_length_proposed:.0f} mm')\n"
        "print(f'BE thick proposed  = {env.be_thickness_proposed:.0f} mm')\n"
        "print(f'Mpr in-plane       = {env.Mpr_in_plane:.0f} kN.m')\n"
        "print(f'Ve capacity        = {env.Ve_capacity:.0f} kN')\n"
        "print(f'phi_Vn             = {env.phi_Vn:.0f} kN  (cap = {env.phi_Vn_max:.0f} kN)')"
    ))

    return cells


# ====================================================================== #
# COMPLETE NOTEBOOK — full tour
# ====================================================================== #
def build_complete() -> list[dict]:
    cells: list[dict] = []

    cells.append(md(
        "# Wall design — complete tour\n"
        "\n"
        "Covers every constructor argument, every output, every design mode,\n"
        "manual overrides, unit hot-swap and the iteration loop.\n"
        "\n"
        "Pipeline:\n"
        "1. Materials + bar schedule\n"
        "2. Web rebar layout (no BE yet)\n"
        "3. WallSection construction\n"
        "4. Wall construction — every argument\n"
        "5. Lazy run + scalars\n"
        "6. PM curves (in-plane / out-of-plane)\n"
        "7. PMM surface — volume + slice\n"
        "8. wall.check(demands)\n"
        "9. wall.design() — three modes + envelope\n"
        "10. Iteration history\n"
        "11. Original vs final wall\n"
        "12. BE as a Column — design, optimize, plot\n"
        "13. evolve() manually\n"
        "14. Unit hot-swap\n"
        "15. report() and summary()"
    ))

    cells.append(md(
        "## 0. Path setup\n"
        "\n"
        "This notebook lives inside `design/`. We add the repo root to\n"
        "`sys.path` so `from design import ...` resolves regardless of\n"
        "where Jupyter was launched."
    ))
    cells.append(code(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "repo_root = Path.cwd().parent if Path.cwd().name == 'design' else Path.cwd()\n"
        "if str(repo_root) not in sys.path:\n"
        "    sys.path.insert(0, str(repo_root))\n"
        "print('repo root:', repo_root)"
    ))

    cells.append(code(
        "from design import (\n"
        "    Concrete, Steel, Bar, WallSection, Wall,\n"
        "    BarSchedule, PlotStyle, to_internal, units_from,\n"
        ")\n"
        "from design.walls import WallDemands\n"
        "from design.sections.reinforcement import RebarLayout, perimeter_bars\n"
        "\n"
        "INPUT_UNITS   = 5    # kN_mm_C — what we type below\n"
        "DISPLAY_UNITS = 11   # Tonf_mm_C — plots / reports / summaries"
    ))

    cells.append(md("## 1. Materials and bar schedule"))
    cells.append(code(
        "concrete = Concrete(fc=35, unit_weight=24)\n"
        "steel    = Steel(fy=420, grade=60, density=7850)\n"
        "\n"
        "schedule = BarSchedule(\n"
        "    longitudinal=[12, 16, 20, 22, 25, 28, 32],\n"
        "    hoops=[10, 12, 16],\n"
        ")\n"
        "concrete, steel, schedule"
    ))

    cells.append(md(
        "## 2. Web rebar layout\n"
        "\n"
        "Single perimeter group along the full wall length. No boundary "
        "elements yet — we let `wall.design()` propose them."
    ))
    cells.append(code(
        "web = perimeter_bars(\n"
        "    b=300, h=4000, cover=50, n_x=2, n_y=14,\n"
        "    bar=Bar(diameter=12), steel=steel,\n"
        ")\n"
        "print('web bars:', len(web.bars))\n"
        "print('web As  :', web.total_area, 'mm^2')"
    ))

    cells.append(md("## 3. WallSection"))
    cells.append(code(
        "section = WallSection(\n"
        "    lw=4000, tw=300,\n"
        "    concrete=concrete,\n"
        "    rebar=RebarLayout(groups=(web,)),\n"
        ")\n"
        "print(section)\n"
        "print('Ag   =', section.gross_area(), 'mm^2')\n"
        "print('Acv  =', section.Acv, 'mm^2')\n"
        "print('has BE?', section.has_boundary_elements)"
    ))

    cells.append(md("## 4. Wall — every argument"))
    cells.append(code(
        "wall = Wall(\n"
        "    section=section,\n"
        "    hw=30_000,                  # total wall height (mm)\n"
        "    lu=3_000,                   # unbraced storey height (mm)\n"
        "    rho_t=0.0025,               # initial distributed transverse ratio\n"
        "    rho_l=0.0025,\n"
        "    fyt=420.0,                  # MPa\n"
        "    seismic=True,\n"
        "    omega_v_factor=1.5,         # §18.10.3.1.2\n"
        "    hx=200.0,                   # §18.7.5.2 inside the BE\n"
        "    transverse_bar_diameter=10.0,\n"
        "    web_bar_diameter=12.0,\n"
        "    units=DISPLAY_UNITS,\n"
        "    bar_schedule=schedule,\n"
        "    label='W-1A',\n"
        ")\n"
        "wall"
    ))

    cells.append(md("## 5. Lazy run + scalars"))
    cells.append(code(
        "wall.run(n_angles=19, n_points=60)\n"
        "print(f'Po         = {wall.Po:.0f} kN')\n"
        "print(f'To         = {wall.To:.0f} kN')\n"
        "print(f'phi_Pn_max = {wall.phi_Pn_max:.0f} kN')\n"
        "print(f'phi_Vn     = {wall.phi_Vn:.0f} kN  (cap = {wall.Vn_max:.0f} kN)')"
    ))

    cells.append(md("## 6. PM curves (in-plane / out-of-plane)"))
    cells.append(code(
        "diagram_in  = wall.surface[0.0]\n"
        "diagram_out = wall.surface[90.0]\n"
        "print(f'in-plane:     max phi_Mn = {max(diagram_in.phi_Mn):.0f} kN.m, Po = {diagram_in.Po:.0f} kN')\n"
        "print(f'out-of-plane: max phi_Mn = {max(diagram_out.phi_Mn):.0f} kN.m')"
    ))

    cells.append(md("## 7. PMM surface — volume + slice"))
    cells.append(code(
        "Pu_grid, Mx_grid, My_grid = wall.surface.volume(n_Pu=20)\n"
        "print('grid shape:', Mx_grid.shape, '(Pu_slices, theta_points)')\n"
        "\n"
        "mx, my = wall.surface.at_Pu(2000.0)\n"
        "print(f'@ Pu=2000:  Mx in [{mx.min():.0f}, {mx.max():.0f}], My in [{my.min():.0f}, {my.max():.0f}]')\n"
        "\n"
        "ok, ratio = wall.surface.check(Pu=2000, Mux=8000, Muy=200)\n"
        "print(f'biaxial check: ok = {ok}, ratio = {ratio:.3f}')"
    ))

    cells.append(md("## 8. wall.check(demands)"))
    cells.append(code(
        "demands = WallDemands(\n"
        "    Pu=2000.0, Mu=8000.0, Mu_out=200.0,\n"
        "    Vu=1500.0,\n"
        "    delta_u=180.0,\n"
        "    sigma_max=8.0,\n"
        ")\n"
        "chk = wall.check(demands)\n"
        "print(f'ratio PM        = {chk.ratio_pm:.3f}')\n"
        "print(f'ratio shear     = {chk.ratio_shear:.3f}')\n"
        "print(f'BE disp/stress  = {chk.be_required_disp} / {chk.be_required_stress}')\n"
        "print(f'BE length req   = {chk.be_length_required:.0f} mm')\n"
        "print(f'BE thickness min= {chk.be_thickness_min:.0f} mm')\n"
        "print(f'distrib rho ok? {chk.distributed_rho_ok}')\n"
        "print(f'c at demand     = {chk.c_at_demand:.0f} mm')\n"
        "print(f'PASSED          = {chk.passed}')"
    ))

    cells.append(md(
        "## 9. wall.design() — three modes + envelope, with auto_update"
    ))
    cells.append(code(
        "results = wall.design(demands, auto_update=True, max_iter=5, tolerance=25.0)\n"
        "print(f'iterations : {results.iterations}')\n"
        "print(f'converged  : {results.converged}')\n"
        "print()\n"
        "print('--- ENVELOPE ---')\n"
        "env = results.envelope\n"
        "print(f'mode               : {env.mode}')\n"
        "print(f'rho_t required    : {env.rho_t_required:.4f}')\n"
        "print(f'rho_l required    : {env.rho_l_required:.4f}')\n"
        "print(f'double curtain?    : {env.double_curtain_required}')\n"
        "print(f'BE required        : {env.be_required}')\n"
        "print(f'BE length proposed : {env.be_length_proposed:.0f} mm')\n"
        "print(f'BE thick proposed  : {env.be_thickness_proposed:.0f} mm')\n"
        "print(f'Mpr in-plane       : {env.Mpr_in_plane:.0f} kN.m')\n"
        "print(f'omega_v            : {env.omega_v}')\n"
        "print(f'Ve capacity        : {env.Ve_capacity:.0f} kN')\n"
        "print(f'phi_Vn (proposed)  : {env.phi_Vn:.0f} kN')"
    ))

    cells.append(md("## 10. Iteration history"))
    cells.append(code(
        "for i, w in enumerate(results.history):\n"
        "    print(f'  iter {i}: BE top = {w.section.be_length_top:>4.0f} mm, BE bot = {w.section.be_length_bot:>4.0f} mm')"
    ))

    cells.append(md("## 11. Original vs final wall"))
    cells.append(code(
        "print('ORIGINAL:', results.original_wall)\n"
        "print('FINAL   :', results.final_wall)\n"
        "print()\n"
        "print('Original section:')\n"
        "results.original_wall.summary()\n"
        "print()\n"
        "print('Final section:')\n"
        "results.final_wall.summary()"
    ))

    cells.append(md(
        "## 12. BE as a Column — full columns/ API works\n"
        "\n"
        "We can design / optimize / inspect each boundary element exactly\n"
        "like a stand-alone SMF column."
    ))
    cells.append(code(
        "final = results.final_wall\n"
        "if final.has_boundary_elements:\n"
        "    be = final.be_top\n"
        "    print(be)\n"
        "    print()\n"
        "    be_results = be.design()\n"
        "    envb = be_results.envelope\n"
        "    print(f'BE detailing (envelope):')\n"
        "    print(f'  Ash_x req = {envb.ash_required_x:.0f} mm^2  provided {envb.av_provided_x:.0f} ({envb.n_legs_x} legs)')\n"
        "    print(f'  Ash_y req = {envb.ash_required_y:.0f} mm^2  provided {envb.av_provided_y:.0f} ({envb.n_legs_y} legs)')\n"
        "    print(f'  db_hoop   = {envb.db_hoop:.0f} mm')\n"
        "    print(f'  spacing   = {envb.spacing_confined:.0f} mm (lo)')\n"
        "    print(f'  hx ok?    = {envb.hx_ok}')\n"
        "else:\n"
        "    print('No BE — demands did not trigger §18.10.6.')"
    ))

    cells.append(md("## 13. evolve() manually"))
    cells.append(code(
        "# Suppose we want to explicitly grow the BE to 800 mm\n"
        "manual_wall = results.final_wall._with_section(\n"
        "    results.final_wall.section.with_boundary_elements(\n"
        "        top_length=800.0, bot_length=800.0,\n"
        "    )\n"
        ")\n"
        "print(manual_wall)\n"
        "manual_wall.run()\n"
        "print(f'Po = {manual_wall.Po:.0f} kN, phi_Vn = {manual_wall.phi_Vn:.0f} kN')"
    ))

    cells.append(md("## 14. Unit hot-swap"))
    cells.append(code(
        "for code in (5, 11, 6, 12):    # kN_mm_C, Tonf_mm_C, kN_m_C, Tonf_m_C\n"
        "    results.final_wall.set_units(code)\n"
        "    u = results.final_wall.units\n"
        "    print(f'{u.name:>11s}: Po = {results.final_wall.Po*u.force_factor:>8.0f} {u.force}')"
    ))

    cells.append(md("## 15. Report + plots"))
    cells.append(code(
        "results.final_wall.set_units(DISPLAY_UNITS)\n"
        "import json\n"
        "print(json.dumps(results.final_wall.report(), indent=2, default=str))"
    ))
    cells.append(code(
        "import matplotlib.pyplot as plt\n"
        "results.final_wall.plot.dashboard(demands=demands)\n"
        "plt.show()"
    ))
    cells.append(code(
        "results.final_wall.plot.iteration_history(results)\n"
        "plt.show()"
    ))
    cells.append(code(
        "if results.final_wall.has_boundary_elements:\n"
        "    results.final_wall.be_top.plot.section()\n"
        "    plt.show()"
    ))

    return cells


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    write(out_dir / "example_wall_simple.ipynb", build_simple())
    write(out_dir / "example_wall.ipynb", build_complete())


if __name__ == "__main__":
    main()
