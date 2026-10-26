"""Generate the three EXAMPLE_REVIEW notebooks (beam, column, wall).

Each notebook runs two exercises:
  1. evaluate a minimum-reinforcement section, apply demands, check, adjust;
  2. design from section + demands.

Results are presented in m / tonf (Tonf_m_C); material strengths in MPa.
Interaction diagrams and the Mpr at zero axial are shown for every element.

Run:  python design/EXAMPLE_REVIEW/_build_notebooks.py
Then execute with nbconvert (the build script does this automatically).
"""
from __future__ import annotations
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(text):
    return new_markdown_cell(text)


def code(src):
    return new_code_cell(src)


# ====================================================================== #
# Shared setup cell
# ====================================================================== #
SETUP = """\
import sys, pathlib
# Make the notebook self-contained: add the repo root (the folder that
# contains design/common) to sys.path regardless of the run cwd.
ROOT = pathlib.Path.cwd()
for _ in range(5):
    if (ROOT / 'design' / 'common').is_dir():
        break
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from design import Concrete, Steel, Bar, Beam, Column, Wall
from design import RectangularSection, WallSection
from design.beams.beam import BeamDemands
from design.columns.column import ColumnDemands
from design.walls.wall import WallDemands
from design.sections.reinforcement import RebarLayout, perimeter_bars, two_layer_bars
from design.common.units import units_from
from design.common.probable import mpr_at_axial, probable_interaction_diagram
from design.analysis.section_analysis import pm_curve

# Presentation: results in tonf / m ; materials in MPa (N/mm^2)
U = units_from('Tonf_m_C')      # forces -> tonf, moments -> tonf*m, lengths -> m
MAT = units_from('N_mm_C')      # stresses -> MPa
print('Results in:', U.name, '| materials in:', MAT.name)
print('1 tonf =', round(1/U.force_factor, 5), 'kN')
"""

PLOT_PM = """\
def plot_pm(diagram, *, title, demand=None, mark_p0=True, nominal=None):
    \"\"\"Plot a phi-reduced P-M interaction diagram (tonf, tonf*m), marking
    the Mpr at zero axial and an optional demand point.\"\"\"
    Mn = np.array(diagram.phi_Mn) * U.moment_factor
    Pn = np.array(diagram.phi_Pn) * U.force_factor
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    ax.plot(Mn, Pn, '-', lw=2, color='#1f77b4', label='phi P-M (design)')
    if nominal is not None:
        Mnn = np.array([p.Mn for p in nominal.points]) * U.moment_factor
        Pnn = np.array([p.Pn for p in nominal.points]) * U.force_factor
        order = np.argsort(Pnn)
        ax.plot(Mnn[order], Pnn[order], '--', lw=1.2, color='#888',
                label='nominal P-M')
    if mark_p0:
        # moment at P = 0 (Mpr-style) read off the nominal curve
        if nominal is not None:
            order = np.argsort([p.Pn for p in nominal.points])
            Pn_s = np.array([nominal.points[i].Pn for i in order])
            Mn_s = np.array([nominal.points[i].Mn for i in order])
            m0 = float(np.interp(0.0, Pn_s, Mn_s)) * U.moment_factor
            ax.plot(m0, 0.0, 'o', color='crimson', ms=8,
                    label=f'M at P=0 = {m0:.1f} tonf*m')
    if demand is not None:
        md_, pd_ = demand
        ax.plot(md_, pd_, 's', color='green', ms=9, label='demand')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('phi*Mn  [tonf*m]'); ax.set_ylabel('phi*Pn  [tonf]')
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()
"""


# ====================================================================== #
# BEAM notebook
# ====================================================================== #
def build_beam():
    nb = new_notebook()
    cells = [
        md("# Revisión de vigas — ACI 318-25\n"
           "Flexión + corte de viga sísmica (SMF, §18.6). "
           "Cláusulas: §9 (vigas), §22.2 (flexión), §22.5 (corte), "
           "§18.6.4 (confinamiento), §18.6.5 (Mpr, Ve).\n\n"
           "**Unidades:** resultados en m / tonf; materiales en MPa.\n"
           "**Mpr de viga = momento probable a axial cero (P=0).**"),
        code(SETUP),
        code(PLOT_PM),
        md("## Ejercicio 1 — Evaluar y verificar un armado mínimo\n"
           "Se parte de una viga 300×600 con la **armadura mínima** de flexión "
           "(§9.6.1.2: $A_{s,min}=\\max(0.25\\sqrt{f'_c}/f_y,\\,1.4/f_y)\\,b_w d$)."),
        code(
            "c = Concrete(fc=28); s = Steel(fy=420, grade=60)\n"
            "bw, h, cover = 300.0, 600.0, 40.0\n"
            "d = h - cover - 10 - 16/2\n"
            "As_min = max(0.25*np.sqrt(c.fc)/s.fy, 1.4/s.fy) * bw * d\n"
            "print(f'd = {d:.0f} mm   As_min = {As_min:.0f} mm^2  '\n"
            "      f'(=> 3 phi16 = {3*201:.0f} mm^2)')\n"
            "beam = Beam.rectangular(bw=bw, h=h, cover=cover, concrete=c, steel=s,\n"
            "    n_top=2, db_top=16, n_bot=3, db_bot=16,\n"
            "    db_stirrup=10, n_legs=2, s_stirrup=150, ln=5000,\n"
            "    seismic=True, Vg=60.0, label='V1-min')\n"
            "beam.run()\n"
            "beam.summary()"
        ),
        md("### 1.2 Capacidades y Mpr a P=0\n"
           "`capacity()` entrega $\\phi M_n$, $\\phi V_n$ y el momento probable "
           "$M_{pr}$ (con $f_{y,pr}=1.25 f_y$, $\\phi=1$). Para vigas el $M_{pr}$ "
           "es el momento del diagrama de interacción **cuando la axial es cero**."),
        code(
            "cap = beam.capacity()\n"
            "print('fc  =', MAT.fmt_stress(c.fc), '| fy =', MAT.fmt_stress(s.fy))\n"
            "print('phi_Mn_pos =', U.fmt_moment(cap.phi_Mn_pos),\n"
            "      '| phi_Mn_neg =', U.fmt_moment(cap.phi_Mn_neg))\n"
            "print('Mpr_pos    =', U.fmt_moment(cap.Mpr_pos),\n"
            "      '| Mpr_neg =', U.fmt_moment(cap.Mpr_neg))\n"
            "print('phi_Vn     =', U.fmt_force(cap.phi_Vn))\n"
            "# cross-check: Mpr_pos equals the probable moment at P=0 on the section\n"
            "print('Mpr@P=0 (section diagram) =', U.fmt_moment(mpr_at_axial(beam.section, Nu=0.0)))"
        ),
        md("### 1.3 Diagrama de interacción P-M con el Mpr a P=0\n"
           "El diagrama probable de la sección; el punto rojo es el $M_{pr}$ a $P=0$."),
        code(
            "diag = probable_interaction_diagram(beam.section, angle_deg=0.0)\n"
            "nom = pm_curve(beam.section, n_points=60, angle_deg=0.0)\n"
            "plot_pm(diag, title='Viga V1 — diagrama probable P-M (in-plane)', nominal=nom)"
        ),
        md("### 1.4 Demandas y verificación\n"
           "Se aplican momentos y cortantes mayorados (en tonf·m / tonf). "
           "`print_checks` reporta razón demanda/capacidad por estación."),
        code(
            "# demands in kN·m / kN internally; here given directly\n"
            "dem = BeamDemands(\n"
            "    Mu_pos_i=8.0,  Mu_neg_i=20.0, Vu_i=18.0,\n"
            "    Mu_pos_mid=22.0, Mu_neg_mid=5.0, Vu_mid=8.0,\n"
            "    Mu_pos_j=8.0,  Mu_neg_j=20.0, Vu_j=18.0)\n"
            "beam.print_checks(dem)"
        ),
        md("### 1.5 Ajuste del diseño para cumplir las demandas\n"
           "`design()` parte del mínimo y propone el armado que cumple."),
        code(
            "results = beam.design(dem)\n"
            "beam.print_design(results, demands=dem)"
        ),
        md("## Ejercicio 2 — Diseño desde la sección y las demandas\n"
           "Geometría fija (300×600) y demandas dadas → armado automático."),
        code(
            "dem2 = BeamDemands(\n"
            "    Mu_pos_i=12.0, Mu_neg_i=30.0, Vu_i=25.0,\n"
            "    Mu_pos_mid=35.0, Mu_neg_mid=8.0, Vu_mid=12.0,\n"
            "    Mu_pos_j=12.0, Mu_neg_j=30.0, Vu_j=25.0)\n"
            "res2 = beam.design(dem2)\n"
            "env = res2.envelope\n"
            "print('Top bars   :', env.n_top, 'phi', env.db_top)\n"
            "print('Bot bars   :', env.n_bot, 'phi', env.db_bot)\n"
            "print('Stirrups   : phi', env.db_stirrup, 'legs', env.n_legs,\n"
            "      '@', U.fmt_length(env.spacing_confined), '(confined)')\n"
            "print('Mpr_pos    =', U.fmt_moment(env.Mpr_pos), '| Mpr_neg =', U.fmt_moment(env.Mpr_neg))"
        ),
        md("## Resumen\n"
           "- $M_{pr}$ de viga verificado = momento a $P=0$ (§18.6.5.1).\n"
           "- Confinamiento $s\\le\\min(d/4,6d_b,150)$ con $d_b$ real (§18.6.4.4).\n"
           "- $V_c=0$ en la zona crítica cuando el sismo gobierna (§18.6.5.2).\n"
           "- Cuantía mínima $A_{s,min}$ (§9.6.1.2) y máxima 0.025 (§18.6.3.1)."),
    ]
    nb.cells = cells
    return nb


# ====================================================================== #
# COLUMN notebook
# ====================================================================== #
def build_column():
    nb = new_notebook()
    cells = [
        md("# Revisión de columnas — ACI 318-25\n"
           "P-M-M + corte + confinamiento (SMF, §18.7). "
           "Cláusulas: §10, §22.4 (interacción), §18.7.5 (confinamiento, Ash), "
           "§18.7.6 (Ve, Mpr), §21.2.2 (phi), §25.7.3 (espiral).\n\n"
           "**Unidades:** resultados en tonf / m; materiales en MPa."),
        code(SETUP),
        code(PLOT_PM),
        md("## Ejercicio 1 — Evaluar un armado mínimo\n"
           "Columna 600×600 con $\\rho_g\\approx1\\%$ (mínimo §10.6.1: "
           "$0.01\\le\\rho_g\\le0.06$ sísmico)."),
        code(
            "c = Concrete(fc=28); s = Steel(fy=420, grade=60)\n"
            "grp = perimeter_bars(b=600, h=600, cover=50, n_x=3, n_y=3, bar=Bar(25), steel=s)\n"
            "sec = RectangularSection(b=600, h=600, concrete=c, rebar=RebarLayout(groups=(grp,)))\n"
            "col = Column(section=sec, lu=3000, Av=2*71.0, fyt=420, s=100, seismic=True, hx=180.0)\n"
            "Ag = 600*600; rho_g = grp.total_area/Ag\n"
            "print(f'rho_g = {rho_g:.4f}  (min 0.01, max 0.06 seismic)  bars: 8 phi25')"
        ),
        md("### 1.2 Diagrama de interacción racional y Mpr\n"
           "El tope $\\phi P_{n,max}=\\phi\\,0.80\\,P_o$ (§22.4.2.1) debe ser una "
           "meseta plana; el $M_{pr}$ a $P=0$ y al $P_u$ actuante."),
        code(
            "cap = col.capacity()\n"
            "print('Po          =', U.fmt_force(cap.Po))\n"
            "print('phi_Pn_max  =', U.fmt_force(cap.phi_Pn_max),\n"
            "      '= 0.65*0.80*Po =', U.fmt_force(0.65*0.80*cap.Po))\n"
            "print('To          =', U.fmt_force(cap.To))\n"
            "print('max phi_Mn  =', U.fmt_moment(max(cap.diagram_x.phi_Mn)))\n"
            "print('Mpr @P=0    =', U.fmt_moment(mpr_at_axial(sec, Nu=0.0)))\n"
            "print('Mpr @Pu=2000=', U.fmt_moment(mpr_at_axial(sec, Nu=2000.0)))"
        ),
        code(
            "nom = pm_curve(sec, n_points=60, angle_deg=0.0)\n"
            "plot_pm(cap.diagram_x, title='Columna 600x600 — P-M racional (eje X)',\n"
            "        demand=(20.0, 250.0), nominal=nom)"
        ),
        md("### 1.3 Verificación frente a demandas\n"
           "Demanda biaxial $(P_u, M_{ux}, M_{uy}, V_u)$ en tonf·m."),
        code(
            "chk = col.check(ColumnDemands(Pu=2000.0, Mux=250.0, Muy=120.0, Vux=150.0))\n"
            "print('ratio PMM  =', round(chk.ratio_pmm,3), ' [', 'PASS' if chk.ratio_pmm<=1 else 'FAIL', ']')\n"
            "print('ratio shear=', round(chk.ratio_shear,3))\n"
            "print('Ash ok     =', chk.ash_ok, '| hx ok =', chk.hx_ok)\n"
            "print('passed     =', chk.passed)"
        ),
        md("## Ejercicio 2 — Diseño desde la sección y las demandas\n"
           "`design()` propone el confinamiento (Ash §18.7.5, hoops, lo) y el Ve."),
        code(
            "res = col.design(ColumnDemands(Pu=2000, Mux=250, Muy=120, Vux=150))\n"
            "p = res.envelope\n"
            "print('Ash_x req =', round(p.ash_required_x,0), 'mm^2  prov', round(p.av_provided_x,0),\n"
            "      f'({p.n_legs_x} legs phi{p.db_hoop:.0f})')\n"
            "print('Ash_y req =', round(p.ash_required_y,0), 'mm^2  prov', round(p.av_provided_y,0),\n"
            "      f'({p.n_legs_y} legs phi{p.db_hoop:.0f})')\n"
            "print('s confined=', U.fmt_length(p.spacing_confined), '| lo =', U.fmt_length(p.lo))\n"
            "if res.capacity.Ve_capacity:\n"
            "    print('Ve capacity=', U.fmt_force(res.capacity.Ve_capacity))"
        ),
        md("## Resumen\n"
           "- Diagrama P-M **racional**: tope $\\phi P_{n,max}=0.65\\cdot0.80\\,P_o$ "
           "como meseta plana (§22.4.2.1), transición $\\phi$ por $\\varepsilon_t$ "
           "(§21.2.2).\n"
           "- $A_{sh}$ por Tabla 18.7.5.4 incluyendo la expresión (c) con axial alta.\n"
           "- $M_{pr}$ disponible a $P=0$ y al $P_u$ actuante (§18.7.6.1.1)."),
    ]
    nb.cells = cells
    return nb


# ====================================================================== #
# WALL notebook
# ====================================================================== #
def build_wall():
    nb = new_notebook()
    cells = [
        md("# Revisión de muros especiales — ACI 318-25\n"
           "Muro estructural especial con elementos de borde (§18.10). "
           "Cláusulas: §11, §18.10.2 (web), §18.10.3/4 (corte, Ve), "
           "§18.10.6 (elementos de borde), §22.2 (interacción).\n\n"
           "**Unidades:** resultados en tonf / m; materiales en MPa.\n"
           "**El diagrama out-of-plane (90°) y el Mpr fuera del plano "
           "ahora se muestrean correctamente.**"),
        code(SETUP),
        code(PLOT_PM),
        md("## Ejercicio 1 — Evaluar un muro con armado mínimo\n"
           "Muro 300×4000 con $\\rho$ distribuido mínimo 0.0025 (§18.10.2.1) "
           "y elementos de borde de partida (línea base)."),
        code(
            "c = Concrete(fc=28); s = Steel(fy=420, grade=60)\n"
            "wall = Wall.rectangular(lw=4000, tw=300, hw=24000, lu=3000,\n"
            "    concrete=c, steel=s, n_y_web=12, db_web=12,\n"
            "    be_top_length=600, be_bot_length=600, n_y_be=5, db_be=25,\n"
            "    cover=40, seismic=True, units='Tonf_m_C')\n"
            "wall.run(); wall.summary()"
        ),
        md("### 1.2 Capacidad: BE evaluados, diagramas y Mpr in/out-of-plane\n"
           "`capacity()` ahora evalúa los elementos de borde (cada BE es una "
           "columna real) y reporta el $M_{pr}$ in-plane, out-of-plane y a $P=0$."),
        code(
            "cap = wall.capacity(biaxial=True)\n"
            "print('has BE          :', cap.has_boundary_elements)\n"
            "print('Po              =', U.fmt_force(cap.Po), '| To =', U.fmt_force(cap.To))\n"
            "print('phi_Vn          =', U.fmt_force(cap.phi_Vn), '(max', U.fmt_force(cap.Vn_max), ')')\n"
            "print('max phi_Mn in   =', U.fmt_moment(max(cap.diagram.phi_Mn)))\n"
            "print('max phi_Mn out  =', U.fmt_moment(max(cap.diagram_out.phi_Mn)))\n"
            "print('Mpr in-plane    =', U.fmt_moment(cap.Mpr_in_plane), '@Pu', U.fmt_force(cap.Pu_at_Mpr_in_plane))\n"
            "print('Mpr in @ P=0    =', U.fmt_moment(cap.Mpr_in_plane_at_zero))\n"
            "print('Mpr out-of-plane=', U.fmt_moment(cap.Mpr_out_of_plane))"
        ),
        md("### 1.3 Diagramas de interacción in-plane y out-of-plane\n"
           "El diagrama fuera del plano (90°) — corregido — coincide con la "
           "sección equivalente y ya no sale colapsado."),
        code(
            "plot_pm(cap.diagram, title='Muro — P-M in-plane (0 grados)')\n"
            "plot_pm(cap.diagram_out, title='Muro — P-M out-of-plane (90 grados)')"
        ),
        md("### 1.4 Verificación integral frente a demandas\n"
           "`check()` verifica el muro **y** sus elementos de borde: "
           "confinamiento (Ash §18.10.6.4(g)), $\\rho$ longitudinal del BE, "
           "capacidad de deriva (§18.10.6.2b) y ancho (§18.10.6.2(b)(ii))."),
        code(
            "dem = WallDemands(Pu=3000, Mu=8000, Vu=1500, delta_u=0.008*24000)\n"
            "chk = wall.check(dem)\n"
            "chk.summary(units=wall.units)"
        ),
        md("## Ejercicio 2 — Diseño integral desde la sección y las demandas\n"
           "`design()` propone el web, los elementos de borde (geometría + "
           "longitudinal desde la línea base + confinamiento Ash) y el Ve."),
        code(
            "res = wall.design(dem)\n"
            "wall.print_final(res.envelope, dem)"
        ),
        md("## Resumen\n"
           "- **Diseño integral del muro:** `capacity()` evalúa los BE, "
           "`check()` verifica su armado (confinamiento + $\\rho$ long.), "
           "`design()` los dimensiona desde una **línea base**.\n"
           "- **Disparador de BE §18.10.6.2a** con el factor 1.5; "
           "**capacidad de deriva §18.10.6.2b** y **ancho §18.10.6.2(b)(ii)**.\n"
           "- **Diagrama out-of-plane y $M_{pr}$ fuera del plano** correctos "
           "(corrección del muestreo de la profundidad rotada)."),
    ]
    nb.cells = cells
    return nb


def main():
    builders = {
        "01_vigas_review.ipynb": build_beam,
        "02_columnas_review.ipynb": build_column,
        "03_muros_review.ipynb": build_wall,
    }
    for name, builder in builders.items():
        nb = builder()
        path = HERE / name
        nbf.write(nb, path)
        print("wrote", path)


if __name__ == "__main__":
    main()
