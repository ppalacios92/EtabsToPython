# design/REFACTOR_GUIDE.md

**Plan maestro de unificación de `design/columns/`, `design/beams/`, `design/walls/`.**

Este documento es la referencia obligatoria para cualquier agente trabajando en el refactor. Léelo completo antes de tocar código.

---

## 0. Filosofía

- **No big-bang.** El código actual sigue funcionando en TODO momento gracias a shims de back-compat. Los shims se eliminan en la última fase.
- **Contrato escrito, testeado, ejecutable.** El "contrato bidirectional" hoy es prosa; lo convertimos en `Protocol` + `tests/test_contract.py`.
- **Cero superposición de archivos entre agentes paralelos.** La matriz de propiedad (sección 1) es ley.
- **Si necesitas algo nuevo en `common/`, NO lo agregues solo.** Detente y reporta. Solo Fase 0 agrega a common.

---

## 1. Matriz de propiedad de archivos

| Archivo | Fase 0 | Fase 1A col | Fase 1B beam | Fase 1C wall | Fase 2 |
|---|:---:|:---:|:---:|:---:|:---:|
| `design/AGENTS.md` | **CREATE** | — | — | — | EDIT |
| `design/common/contracts.py` | **CREATE** | — | — | — | — |
| `design/common/shear.py` | **CREATE** | — | — | — | — |
| `design/common/probable.py` | **CREATE** | — | — | — | — |
| `design/common/optimize.py` | **CREATE** | — | — | — | — |
| `design/common/plot_style.py` | **CREATE** | — | — | — | — |
| `design/common/spacing.py` | **CREATE** | — | — | — | — |
| `design/common/factors.py` | EDIT | — | — | — | — |
| `design/common/__init__.py` | EDIT | — | — | — | — |
| `design/columns/probable.py` | shim | — | — | — | — |
| `design/columns/style.py` | shim | — | — | — | — |
| `design/beams/style.py` | shim | — | — | — | — |
| `design/walls/probable.py` | shim verify | — | — | — | — |
| `design/columns/*.py` (resto) | — | **EDIT** | — | — | — |
| `design/beams/*.py` (resto) | — | — | **EDIT** | — | — |
| `design/walls/*.py` (resto) | — | — | — | **EDIT** | — |
| `design/walls/style.py` | — | — | — | **CREATE** | — |
| `design/walls/optimize.py` | — | — | — | **CREATE** | — |
| `design/__init__.py` | — | — | — | — | **EDIT** |
| `tests/test_common_*.py` | **CREATE** | — | — | — | — |
| `tests/test_contract.py` | — | — | — | — | **CREATE** |

**Regla**: si un archivo no aparece en tu columna, NO LO TOCÁS. Si necesitás un cambio cruzado, reportalo en el output del agente, no lo hagas.

---

## 2. FASE 0 — Foundation (secuencial, bloqueante)

### 2.1 Goal
Crear el cimiento común (`design/common/*`) y el contrato escrito (`design/AGENTS.md`). NO tocar lógica de Beam/Column/Wall — solo shims de re-export para que sigan funcionando.

### 2.2 Tareas

**T0.1 — `design/AGENTS.md`** (escribir desde cero, ~150 líneas, en español). Contenido obligatorio:

```markdown
# design/AGENTS.md — Contrato Beam / Column / Wall

## 1. Unidades internas
- Stress: MPa
- Length: mm (INCLUYE lu, ln, hw, cover — NO metros)
- Force: kN
- Moment: kN·m
- Area: mm²

Los factories .rectangular() PUEDEN aceptar metros como conveniencia,
pero deben convertir explícitamente. NO hay *1000 o /1000 flotantes
dispersos en wall.py / column.py / beam.py.

## 2. Naming canónico
| Concepto | Nombre |
|---|---|
| Recubrimiento al exterior del estribo | cover (alias: clear_cover) |
| Acero transversal | steel_transverse: Steel; fyt es property |
| Longitud confinada | lo |
| Espaciamiento confinado | spacing_confined |
| Espaciamiento medio | spacing_middle |
| Momento probable | Mpr (o Mpr_x, Mpr_y, Mpr_pos, Mpr_neg, Mpr_in_plane); NUNCA Mn_pr |
| Shear capacidad-design | Ve_capacity, Ve_used, amplification |

## 3. Contrato (design.common.contracts.Element)
Cada elemento implementa:
    run() -> Self                                (lazy + cache)
    capacity() -> XCapacity
    check(demands) -> XCheck
    design(demands=None) -> XDesignResults       (minimum/demand/capacity/envelope)
    evolve(proposal) -> Self                     (INMUTABLE: nuevo objeto)
    apply(proposal) -> Self                      (alias de evolve)
    summary() -> None
    report() -> dict
    set_units(code_or_name) -> Self
    .plot                                         (lazy plotter property)

## 4. Mutabilidad: SIEMPRE inmutable via evolve()
- evolve(p) devuelve NUEVO elemento.
- apply(p) es alias de evolve().
- design() NO muta self.design_results. El caller guarda el retorno.
- run() SÍ muta caches internos (_surface, _phi_Vn, etc.). Devuelve self.

## 5. Lista negra de hardcodes
NINGÚN archivo en beams/, columns/, walls/ puede contener literal:
    0.17 * lam * sqrt_fc * b * d / 1000      → vc_simplified(b, d, concrete)
    Av * fyt * d / (s * 1000)                 → vs_capacity(Av, fyt, d, s)
    (Vu/phi - Vc) * 1000 / (fyt * d)          → av_s_required(...)
    0.85 - 0.05*(fc-28)/7                     → beta1(fc)
    phi = 0.90 if tc else 0.65                → phi_axial_flexure(eps_t, eps_ty)
    0.83 * sqrt(fc) * Acv / 1000              → vn_wall_max(Acv, fc)
    phi = 0.75 (shear)                        → phi_shear()
    Pu < Ag*fc/20 (§18.7.6.2.1 trigger)       → vc_zero_seismic(...)

## 6. Vc=0 sísmico
Tanto vigas (§18.6.5.2) como columnas (§18.7.6.2.1) deben evaluar
Vc=0 en lo cuando aplica. Centralizado en common.shear.vc_zero_seismic.
Muros NO aplican.

## 7. Mapeo Wall → BE Column
common.contracts.wall_demands_to_be(WallDemands, side) -> ColumnDemands
Mapping:
    Pu  -> Pu          Mu     -> Mux (in-plane)
    Mu_out -> Muy (out-of-plane)
    Vu  -> Vuy         Vu_out -> Vux

## 8. PlotStyle
UNA sola PlotStyle en common.plot_style. Cada elemento expone
element.style. Plotters leen de self.element.style.

## 9. OptimizeAlternative
UNA base en common.optimize. Cada elemento puede subclasear o usar
con dict 'detailing'. Los tres exponen element.run_optimize(demands).

## 10. BarSchedule
UN BarSchedule en common.bar_schedule. Los elementos lo aceptan por
kwarg bar_schedule=None y crean default.

## 11. Tests
tests/test_contract.py itera [beam, col, wall] y verifica todo lo de arriba.
```

**T0.2 — `design/common/shear.py`** (CREATE). API exacta:

```python
from math import sqrt
from design.common.materials import Concrete
from design.common.factors import phi_shear

def vc_simplified(*, b: float, d: float, concrete: Concrete) -> float:
    """Vc simplificado §22.5.5.1. Devuelve kN."""
    return 0.17 * concrete.lam * concrete.sqrt_fc * b * d / 1000.0

def vc_detailed(*, b, d, Nu, Ag, rho_w, concrete) -> float: ...
def vs_capacity(*, Av: float, fyt: float, d: float, s: float) -> float:
    """Vs §22.5.10. Devuelve kN. Av en mm², fyt MPa, d/s mm."""
    if s <= 0: return 0.0
    return Av * fyt * d / s / 1000.0

def vs_max(*, b: float, d: float, concrete) -> float:
    """Cap §22.5.1.2. Devuelve kN."""
    return 0.66 * concrete.sqrt_fc * b * d / 1000.0

def av_s_required(*, Vu: float, Vc: float, fyt: float, d: float,
                  phi: float | None = None) -> float | None:
    """Av/s [mm²/mm] para Vu [kN]. Devuelve 0 si Vc alcanza, None si Vu<=0."""
    if Vu is None or Vu <= 0: return None
    if phi is None: phi = phi_shear()
    Vs_req = Vu / phi - Vc
    if Vs_req <= 0: return 0.0
    return (Vs_req * 1000.0) / (fyt * d)

def av_s_minimum(*, b: float, concrete, fyt: float) -> float:
    """§9.6.3.4. Devuelve mm²/mm."""
    t1 = 0.062 * sqrt(concrete.fc) * b / fyt
    t2 = 0.35 * b / fyt
    return max(t1, t2)

def vc_zero_seismic(*, V_seismic: float, V_total: float,
                    Pu: float, Ag: float, fc: float) -> bool:
    """True si Vc debe tomarse como 0 en lo:
       - §18.6.5.2 (vigas): V_seismic >= 0.5 V_total AND Pu < Ag·fc/20
       - §18.7.6.2.1 (columnas): misma condición
    """
    if V_total <= 0: return False
    seismic_share = V_seismic / V_total
    cond_a = seismic_share >= 0.5
    cond_b = Pu < (Ag * fc / 20.0) / 1000.0  # Pu en kN, Ag·fc en N
    return cond_a and cond_b

# Walls
def alpha_c(hw_over_lw: float) -> float: ...   # §18.10.4.1
def vn_wall(*, Acv, fc, rho_t, fyt, hw_over_lw, lam=1.0) -> float: ...   # §18.10.4.1
def vn_wall_max(*, Acv, fc) -> float: ...   # §18.10.4.4

# Capacity-design shear
def ve_in_plane_capacity_wall(*, Mpr, Mu=None, Vu=None, omega_v_factor=1.5) -> float: ...
```

Copiar implementaciones desde `design/beams/shear.py` y `design/walls/shear.py`. Verificar contra docstrings ACI.

**T0.3 — `design/common/probable.py`** (CREATE). Mover desde `design/columns/probable.py`:
- `probable_interaction_diagram(section, *, n_points=80, angle_deg=0.0, spiral=False, fy_factor=1.25)`
- `mpr_envelope(section, *, angle_deg=0.0, spiral=False, Pu_range=None, n_points=80)`
- Helper interno `_scale_layout_fy` y `_SectionWithRebar`

Después, `design/columns/probable.py` queda como 3 líneas:
```python
"""Re-export — moved to design.common.probable."""
from design.common.probable import probable_interaction_diagram, mpr_envelope
__all__ = ["probable_interaction_diagram", "mpr_envelope"]
```

**T0.4 — `design/common/optimize.py`** (CREATE):

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class OptimizeAlternative:
    proposal: object             # element-specific DesignProposal
    detailing: dict              # element-specific keys
    rho_transverse: float        # kg/m³ concreto
    rho_longitudinal: float
    rho_total: float
    feasible: bool
    is_baseline: bool = False
    is_provided: bool = False
    notes: tuple[str, ...] = ()

def format_optimize_table(alternatives: list[OptimizeAlternative],
                          *, top_n: int = 12,
                          detail_fmt: callable = None) -> str:
    """Tabla pretty-printed. detail_fmt(alt) -> str para la columna 'detalle'."""
    ...
```

**T0.5 — `design/common/plot_style.py`** (CREATE):

```python
from dataclasses import dataclass

@dataclass(slots=True)
class PlotStyle:
    color_nominal: str = "k"
    color_reduced: str = "tab:blue"
    color_demand: str = "tab:red"
    color_pos: str = "tab:blue"
    color_neg: str = "tab:red"
    color_concrete: str = "#D8D8D8"
    color_rebar: str = "tab:blue"
    color_envelope_cap: str = "red"
    linewidth: float = 1.5
    linewidth_cap: float = 0.8
    linestyle_cap: str = "--"
    fill_alpha: float = 0.15
    bar_scale: float = 1.0
    grid_alpha: float = 0.3
    figsize_2d: tuple[float, float] = (6.0, 6.0)
    figsize_wide: tuple[float, float] = (10.0, 4.0)
    figsize_3d: tuple[float, float] = (8.0, 7.0)
    figsize_dashboard: tuple[float, float] = (14.0, 10.0)
```

Shims:
```python
# design/columns/style.py
"""Re-export."""
from design.common.plot_style import PlotStyle
__all__ = ["PlotStyle"]
```
```python
# design/beams/style.py
"""Re-export. BeamPlotStyle es alias de PlotStyle."""
from design.common.plot_style import PlotStyle
BeamPlotStyle = PlotStyle
__all__ = ["PlotStyle", "BeamPlotStyle"]
```

**T0.6 — `design/common/spacing.py`** (CREATE). Mover funciones de:
- `design/columns/confinement.py`: `s_max_confined`, `s_max_middle_zone`, `lo_length`, `hx_check`, `ash_required`
- `design/beams/shear.py`: `s_max_seismic_smf`, `s_max_seismic_outside`

Renombrar para claridad:
```python
def lo_length_beam(*, h_member: float) -> float:
    return 2.0 * h_member

def lo_length_column(*, h_member: float, lu_clear_mm: float) -> float: ...

def s_max_seismic_smf_beam(*, d, db_long_min, db_hoop, grade=60) -> float: ...
def s_max_seismic_outside_beam(*, d) -> float: ...
def s_max_confined_column(*, h_min, db_long_min, hx, grade=60) -> float: ...
def s_max_middle_zone_column(*, db_long_min, grade=60) -> float: ...
def hx_check(*, b_core, n_legs, limit=350.0) -> tuple[float, bool]: ...
def ash_required(*, s, bc, Ag, Ach, fc, fyt, Pu_over_Ag_fc=0.0) -> float: ...
```

Los módulos viejos quedan como shims que reimportan.

**T0.7 — `design/common/factors.py`** (EDIT, agregar):

```python
# Constants para capacity-design shear amplification
OMEGA_0_COLUMN_DEFAULT = 3.0   # §18.7.6.1.1
OMEGA_V_WALL_DEFAULT   = 1.5   # §18.10.3.1.2 simplified

def omega_v_wall(*, n_stories=None, hw_over_lw=None) -> float:
    """§18.10.3.1.2. Default 1.5; baja a 1.0 si n_stories<=2 y hw/lw<=2."""
    if n_stories is not None and hw_over_lw is not None:
        if n_stories <= 2 and hw_over_lw <= 2.0:
            return 1.0
    return OMEGA_V_WALL_DEFAULT
```

**T0.8 — `design/common/contracts.py`** (CREATE):

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Demands(Protocol):
    pass

@runtime_checkable
class Capacity(Protocol):
    phi_Vn: float

@runtime_checkable
class Check(Protocol):
    passed: bool

@runtime_checkable
class DesignProposal(Protocol):
    mode: str
    notes: tuple

@runtime_checkable
class DesignResults(Protocol):
    minimum: DesignProposal
    capacity: DesignProposal
    envelope: DesignProposal

@runtime_checkable
class Element(Protocol):
    label: str
    def run(self): ...
    def capacity(self): ...
    def check(self, demands): ...
    def design(self, demands=None): ...
    def evolve(self, proposal): ...
    def summary(self): ...
    def report(self) -> dict: ...

def wall_demands_to_be(demands) -> "ColumnDemands":
    """Mapeo WallDemands → ColumnDemands para diseño de boundary element.
    
    En un wall:
      In-plane Mu rota la sección sobre el eje fuerte de la BE.
      Out-of-plane Mu_out rota sobre el eje débil.
      In-plane Vu actúa sobre la sección a lo largo de lw (eje fuerte).
      Out-of-plane Vu_out es transversal al wall (eje débil).
    
    Mapping:
        ColumnDemands(
            Pu  = demands.Pu,
            Mux = demands.Mu,         # in-plane bending → strong axis
            Muy = demands.Mu_out,     # out-of-plane    → weak axis
            Vux = demands.Vu_out,     # out-of-plane shear
            Vuy = demands.Vu,         # in-plane shear
        )
    """
    from design.columns.column import ColumnDemands
    return ColumnDemands(
        Pu  = demands.Pu,
        Mux = demands.Mu,
        Muy = demands.Mu_out,
        Vux = demands.Vu_out,
        Vuy = demands.Vu,
    )
```

**T0.9 — `design/common/__init__.py`** (EDIT). Re-exportar nuevas APIs.

**T0.10 — Tests**:
- `tests/test_common_shear.py`: vc_simplified contra ACI fc=28 b=300 d=540 → conocer el valor.
- `tests/test_common_probable.py`: mpr_envelope de una sección rectangular conocida.
- `tests/test_common_contracts.py`: importa Protocols, verifica runtime_checkable.

### 2.3 Acceptance criteria Fase 0
- [ ] `pytest tests/test_common_*.py` pasa.
- [ ] El smoke test existente (`design/_smoke_test.py`) sigue pasando sin cambios.
- [ ] `python -c "from design.common import vc_simplified, OptimizeAlternative, PlotStyle, Element; from design.common.contracts import wall_demands_to_be"` no falla.
- [ ] `design/AGENTS.md` existe con las 11 secciones.
- [ ] Lógica de beam.py / column.py / wall.py NO cambió (solo los 4 shims).
- [ ] `grep -rn "0.17 \* " design/common/` retorna las apariciones esperadas (en shear.py).

### 2.4 Forbidden Fase 0
- Tocar lógica de `beam.py`, `column.py`, `wall.py`.
- Cambiar firmas públicas existentes.
- Borrar archivos viejos (solo convertirlos a shims).

---

## 3. FASE 1A — Column (paralelo)

### 3.A.1 Goal
Adaptar `design/columns/` al contrato. Eliminar shear hardcoded. Implementar Vc=0 §18.7.6.2.1. Migrar `lu` de metros a mm. Adoptar `evolve()` inmutable.

### 3.A.2 Owned files
SOLO archivos bajo `design/columns/*.py`. Permitido también `tests/test_columns*.py`.

### 3.A.3 Tareas

**TC.1 — Eliminar hardcodes en `column.py`**:
- Líneas 470, 507, 735-736, 803-804, 959: `0.17 * lam * sqrt_fc * b * d / 1000.0` → `vc_simplified(b=..., d=..., concrete=...)` desde `design.common.shear`.
- Líneas 471, 510, 807, 808: `Av * fyt * d / (s * 1000.0)` → `vs_capacity(Av=..., fyt=..., d=..., s=...)`.
- Líneas 939-951 `_av_s_required_for_direction`: reemplazar cuerpo por llamada a `av_s_required(Vu=Vu, Vc=Vc, fyt=self.fyt, d=d)` desde common. Mantener método como wrapper.
- Línea 947: `phi = 0.75` → `phi_shear()`.

**TC.2 — Vc=0 §18.7.6.2.1**:
En `_design_one` modo `"capacity"` y `"demand"`, antes de calcular `av_s_req_x`/`av_s_req_y`, evaluar:
```python
from design.common.shear import vc_zero_seismic
# Para cada dirección:
vc_zero_x = vc_zero_seismic(
    V_seismic=Ve_x_capacity,  # o demands.Vux dependiendo del modo
    V_total=Ve_x_used or demands.Vux,
    Pu=Pu_used or 0.0,
    Ag=Ag,
    fc=c.fc,
)
if vc_zero_x:
    Vc_x = 0.0
    notes.append("§18.7.6.2.1 — Vc=0 en X (Pu<Ag·fc/20).")
# ídem Vc_y
```
Agregar campos `vc_zero_x: bool`, `vc_zero_y: bool` a `ColumnDesignProposal`.

**TC.3 — Unidades `lu` → mm**:
Hoy `lu` se pasa en metros (`column.py:264`). Cambiar a:
- Internal: siempre mm.
- Factory `Column.rectangular(lu=...)`: aceptar metros por conveniencia. Detectar con heurística: si `lu < 30` asumir metros y convertir; loguear DeprecationWarning explícito.
- `__init__(lu=...)`: documentar que es mm. No hacer conversión automática (back-compat: si alguien construye Column directamente, debe pasar mm o usar la factory).
- Exponer `lu_m` property que devuelva metros (legacy).
- Línea 789 `lo = lo_length(h_member=..., lu_clear=self.lu * 1000.0)`: eliminar el `*1000`.

**TC.4 — `evolve()` inmutable**:
```python
def evolve(self, proposal: ColumnDesignProposal) -> "Column":
    """Devuelve NUEVA Column con el proposal aplicado."""
    new_section = self.section  # el proposal NO cambia la sección
    new = Column(
        section=new_section,
        lu=self.lu, k=self.k, spiral=self.spiral,
        Av=min(proposal.n_legs_x, proposal.n_legs_y) * (pi * proposal.db_hoop**2 / 4),
        fyt=self.fyt,
        s=proposal.spacing_confined,
        seismic=self.seismic, hx=self.hx,
        transverse_bar_diameter=proposal.db_hoop,
        units=self.units, bar_schedule=self.bar_schedule, label=self.label,
    )
    new.n_legs_x = proposal.n_legs_x
    new.n_legs_y = proposal.n_legs_y
    return new

def apply(self, proposal): return self.evolve(proposal)
```
El `apply` actual (que muta) queda como `evolve`. El nuevo `apply` es alias.

**TC.5 — `omega_0` desde common**:
Línea 311 `self.omega_0: float = 3.0` → `self.omega_0: float = OMEGA_0_COLUMN_DEFAULT` (importar de `design.common.factors`).

**TC.6 — `optimize.py`**:
`OptimizeAlternative` actual de columns/optimize.py: que sea ahora un `dataclass(frozen=True, slots=True)` que herede o adapte el de `common.optimize.OptimizeAlternative`. Más simple: mantener la dataclass pero hacer que `detailing = {"db_hoop": db_hoop, "n_legs_x": n_legs_x, "n_legs_y": n_legs_y}`. Asegurar que `format_optimize_table` use el de common.

**TC.7 — Adoptar Protocol**:
Verificar que `Column` satisface `design.common.contracts.Element`. Agregar typing si falta. Si `column.run()` no devuelve self, ajustar.

**TC.8 — `column.design()` NO muta `self.design_results`**:
Línea 663 `self.design_results = results`: borrar. El caller guarda el resultado si quiere. Esto es coordinado con TW.7.

**TC.9 — `cover` alias**:
Aceptar `cover=...` como alias de `clear_cover=...` en `Column.rectangular()` y en `.design()`. Documentar `cover` como canónico.

### 3.A.4 Acceptance criteria
- [ ] `grep -n "0\.17 \* " design/columns/*.py` solo en archivos de plotting/docstrings (no en fórmulas activas).
- [ ] `grep -n "Av \* .* fyt \* d / .* s" design/columns/column.py` retorna 0.
- [ ] `isinstance(col, design.common.contracts.Element)` es True.
- [ ] `col.evolve(proposal) is not col`.
- [ ] Existe `tests/test_columns_vc_zero.py` (o agregado a tests existentes) que cubre §18.7.6.2.1.
- [ ] El smoke test `design/_smoke_test.py` sigue pasando.

### 3.A.5 Forbidden
- Tocar `design/beams/`, `design/walls/`, `design/common/`, `design/sections/`, `design/__init__.py`.
- Cambiar firmas públicas de `Column.run()`, `.capacity()`, `.check()`, `.design()` — solo extender.

---

## 4. FASE 1B — Beam (paralelo)

### 4.B.1 Goal
Adaptar `design/beams/` al contrato. Agregar `Beam.run()` lazy con cache. Reemplazar `phi=0.90/0.65` por `phi_axial_flexure`. Eliminar shear hardcoded. `evolve()` inmutable.

### 4.B.2 Owned files
SOLO archivos bajo `design/beams/*.py`. Permitido también `tests/test_beams*.py`.

### 4.B.3 Tareas

**TB.1 — `phi_axial_flexure` en `_phi_Mn`**:
Línea 489 `phi = 0.90 if tc else 0.65` → usar `design.common.factors.phi_axial_flexure(eps_t, eps_ty)`. Calcular `eps_t` explícitamente (ya se computa en `is_tension_controlled`).

```python
from design.common.factors import phi_axial_flexure

# en _phi_Mn:
a = (As * s_steel.fy) / (0.85 * c.fc * bw)
c_neutral = a / c.beta1
eps_t = c.eps_cu * (d - c_neutral) / c_neutral if c_neutral > 0 else 0.0
phi = phi_axial_flexure(eps_t=eps_t, eps_ty=s_steel.eps_ty)
```

**TB.2 — Eliminar hardcodes**:
- Línea 500 `Vs = self.Av * self.fyt * d / (self.s * 1000.0)`: reemplazar por `vs_capacity(...)`.
- Línea 729 `Vc = vc_simplified(...)`: verificar que importa de `design.common.shear`.
- Líneas 722-734 `_av_s_required_for`: reemplazar cuerpo por llamada a `av_s_required(...)`.

**TB.3 — `Beam.run()` + cache + proxies**:
```python
def run(self) -> "Beam":
    """Lazy compute, cache. Devuelve self por chaining."""
    self._phi_Mn_pos = self._compute_phi_Mn("pos")
    self._phi_Mn_neg = self._compute_phi_Mn("neg")
    self._phi_Vn_cache = self._compute_phi_Vn()
    if self.seismic:
        self._mpr_pos_cache = self._mpr(direction="pos")
        self._mpr_neg_cache = self._mpr(direction="neg")
    return self

def _ensure_run(self):
    if not hasattr(self, "_phi_Mn_pos"): self.run()

@property
def phi_Mn_pos(self): self._ensure_run(); return self._phi_Mn_pos
@property
def phi_Mn_neg(self): self._ensure_run(); return self._phi_Mn_neg
@property
def phi_Vn(self): self._ensure_run(); return self._phi_Vn_cache

@property
def Mn(self) -> dict:
    """{'pos': Mn_pos, 'neg': Mn_neg} — dict, no DiagramFieldView (Beam no es PMM)."""
    return {"pos": self.phi_Mn_pos / 0.9, "neg": self.phi_Mn_neg / 0.9}  # ajuste si phi varía

@property
def phi_Mn(self) -> dict:
    return {"pos": self.phi_Mn_pos, "neg": self.phi_Mn_neg}
```

Renombrar el método interno actual `_phi_Mn(direction=...)` a `_compute_phi_Mn(direction=...)` (o similar) para no chocar con la property.

**TB.4 — Mpr renaming**:
En `BeamCapacity`, renombrar `Mn_pr_pos` → `Mpr_pos`, `Mn_pr_neg` → `Mpr_neg`. Mantener `Mn_pr_pos` como property alias por back-compat:
```python
@property
def Mn_pr_pos(self): return self.Mpr_pos
@property
def Mn_pr_neg(self): return self.Mpr_neg
```
Esto requiere convertir el dataclass a una clase regular o usar field aliasing.

**TB.5 — `evolve()` inmutable**:
```python
def evolve(self, proposal: BeamDesignProposal) -> "Beam":
    """Devuelve NUEVA Beam."""
    x_min, y_min, x_max, y_max = self.section.bounding_box()
    bw = x_max - x_min
    h = y_max - y_min
    new_rebar = _build_beam_rebar(
        bw=bw, h=h, cover=self.cover,
        db_stirrup=proposal.db_stirrup,
        n_top=proposal.n_top, db_top=proposal.db_top,
        n_bot=proposal.n_bot, db_bot=proposal.db_bot,
        steel=self.steel,
    )
    new_section = RectangularSection(b=bw, h=h, concrete=self.section.concrete, rebar=new_rebar)
    a_per_leg = pi * proposal.db_stirrup ** 2 / 4.0
    new_Av = proposal.n_legs * a_per_leg
    return Beam(
        section=new_section, ln=self.ln, cover=self.cover,
        Av=new_Av, s=proposal.spacing_confined,
        seismic=self.seismic, db_long_min=self.db_long_min,
        db_stirrup=proposal.db_stirrup,
        Vg=self.Vg, steel_transverse=self._steel_transverse,
        units=self.units, bar_schedule=self.bar_schedule, label=self.label,
    )

def apply(self, proposal): return self.evolve(proposal)
```

**TB.6 — `beam.design()` NO muta `self.design_results`**:
Línea 627: borrar `self.design_results = results`.

**TB.7 — `Beam.ln`**:
Mantener en mm internamente (acompañar con column/wall). Factory `Beam.rectangular(ln=...)` acepta metros si ln<30, convierte. Property `ln_m` para legacy.

**TB.8 — `optimize.py`**:
Adaptar `OptimizeAlternative` para que use `detailing` dict del de common, o mantener la dataclass específica de beam pero importar el `format_optimize_table` de common.

**TB.9 — Adoptar Protocol**:
Verificar que `Beam` cumple `design.common.contracts.Element`.

### 4.B.4 Acceptance criteria
- [ ] `grep -n "0\.17 \* " design/beams/*.py` solo en docstrings.
- [ ] `grep -n "0.90 if .* else 0.65" design/beams/*.py` retorna 0.
- [ ] `beam.run()` existe y devuelve self.
- [ ] `beam.phi_Mn` es property (dict).
- [ ] `beam.evolve(proposal) is not beam` y `beam.section` queda intacta.
- [ ] `isinstance(beam, Element)` es True.
- [ ] Smoke test pasa.

### 4.B.5 Forbidden
- Tocar `design/columns/`, `design/walls/`, `design/common/`, `design/sections/`, `design/__init__.py`.

---

## 5. FASE 1C — Wall (paralelo)

### 5.C.1 Goal
Eliminar `__getattr__` proxy peligroso. Crear `Wall.rectangular()` factory. Crear `walls/optimize.py`. Crear `walls/style.py`. Fix unidades `lu` (ya no `/1000`). Externalizar mapeo `WallDemands → ColumnDemands`.

### 5.C.2 Owned files
SOLO archivos bajo `design/walls/*.py`. Permitido también `tests/test_walls*.py`.

### 5.C.3 Tareas

**TW.1 — Eliminar `__getattr__`**:
Líneas 188-190 de `wall.py`: borrar el `__getattr__`. Buscar callers que hagan `result.X` (no `result.envelope.X`). Probable lista:
- `design/_smoke_test.py` — actualizar.
- `design/example_wall*.ipynb` — buscar y actualizar.
- `design/example_wall*.py` si existen.

**TW.2 — `Wall.rectangular()` factory**:
```python
@classmethod
def rectangular(
    cls, *,
    lw: float, tw: float, hw: float, lu: float | None = None,
    concrete: Concrete, steel: Steel,
    # web
    db_web: float = 10.0, web_spacing: float = 200.0, web_layers: int = 2,
    # boundary elements (opcional)
    be_top_length: float = 0.0, be_top_thickness: float = 0.0,
    be_bot_length: float = 0.0, be_bot_thickness: float = 0.0,
    n_be_bars_per_face: int = 4, db_be: float = 20.0,
    # detallado
    db_stirrup: float = 10.0,
    fyt: float = 420.0,
    cover: float = 25.0,
    seismic: bool = True, hx: float = 200.0,
    omega_v_factor: float = 1.5,
    units=None, bar_schedule=None, label: str = "Wall",
) -> "Wall":
    """Construye un Wall en una sola llamada."""
    # 1. Construir WallSection (web + BE si aplica)
    # 2. Construir el Wall
    # 3. Retornar
    ...
```

**TW.3 — `lu` en mm**:
Cuando Fase 1A migre `Column.lu` a mm, **borrar el `/1000.0`** en `wall.py:406`. Para evitar race entre fases: hacer en TW.3 una verificación condicional inicial: si `Column.__init__` documenta mm, sacar la conversión. Si todavía espera metros, mantener temporalmente con comentario `# TODO: borrar cuando Column migre a mm`. (En la práctica, asumir que Fase 1A ya migró y dejarlo limpio.)

**TW.4 — Mapeo `WallDemands → ColumnDemands`**:
Líneas 764-768 de `wall.py`: reemplazar el mapeo inline por:
```python
from design.common.contracts import wall_demands_to_be

col_demands = (
    wall_demands_to_be(demands) if demands is not None else None
)
```

**TW.5 — `walls/style.py`** (CREATE):
```python
"""Wall plot style. Alias de common.PlotStyle por uniformidad."""
from design.common.plot_style import PlotStyle
WallPlotStyle = PlotStyle
__all__ = ["PlotStyle", "WallPlotStyle"]
```
Y modificar `walls/plotting.py` para leer de `self.wall.style` si existe, o usar default.

**TW.6 — `walls/optimize.py`** (CREATE):
Estructura paralela a `beams/optimize.py` y `columns/optimize.py`:
```python
"""Optimize wall reinforcement.

Explora (db_web, spacing, layers) para el web y delega BE optimize a columns.
"""
from design.common.optimize import OptimizeAlternative, format_optimize_table
# ... run_optimize(wall, demands) -> list[OptimizeAlternative]
```

API mínima: `run_optimize(wall, demands=None, *, sort_by="rho_total", include_infeasible=False) -> list[OptimizeAlternative]`. Mass density via `web_rho_provided` ya existente.

Si esto demanda demasiado tiempo, implementar una versión simplificada que sólo enumere (db_web, web_spacing) y reporte. Documentar la limitación.

**TW.7 — `wall.design()` NO muta `self.design_results`**:
Líneas 615-617: borrar `final_wall.design_results = out` y `self.design_results = out`. El caller guarda.

**TW.8 — Eliminar hardcodes**:
- `wall.py:666` `Vc_term = a_c * c.lam * c.sqrt_fc * s.Acv / 1000.0`: reemplazar por una llamada a `design.common.shear` (probablemente extraer una función `vc_wall(Acv, fc, hw_over_lw, lam)` allí).
- `wall.py:667` `Vu/0.75`: reemplazar `0.75` por `phi_shear()`.

**TW.9 — Adoptar Protocol**:
Verificar que `Wall` cumple `design.common.contracts.Element`.

**TW.10 — Smoke test fix**:
`design/_smoke_test.py` probablemente accede a `result.X` directamente (gracias al `__getattr__` que vamos a borrar). Actualizar a `result.envelope.X`.

### 5.C.4 Acceptance criteria
- [ ] `WallDesignResults.__getattr__` NO existe.
- [ ] `Wall.rectangular(...)` construye en una llamada.
- [ ] `wall.run_optimize(demands)` retorna `list[OptimizeAlternative]`.
- [ ] `grep -n "lu / 1000" design/walls/*.py` retorna 0.
- [ ] `from design.common.contracts import wall_demands_to_be` se usa en `wall.py`.
- [ ] `walls/style.py` existe.
- [ ] `walls/optimize.py` existe.
- [ ] `isinstance(wall, Element)` es True.
- [ ] Smoke test pasa.

### 5.C.5 Forbidden
- Tocar `design/columns/`, `design/beams/`, `design/common/`, `design/__init__.py`.

---

## 6. FASE 2 — Integration (secuencial, post-1A/B/C)

### 6.1 Owned files
- `design/__init__.py` (EDIT)
- `design/AGENTS.md` (EDIT — agregar lecciones)
- `design/README.md` (EDIT)
- `tests/test_contract.py` (CREATE)
- `tests/test_cross_module.py` (CREATE)
- `design/example_*.ipynb` (re-run para regenerar outputs)

### 6.2 Tareas
- **TI.1** Tests parametrizados por elemento `[beam, col, wall]` verificando Protocol, run, capacity, check, design, evolve, units mm.
- **TI.2** Tests cross-module: `wall.be_top` round-trip; `grep` que confirme cero hardcodes.
- **TI.3** Actualizar `design/__init__.py` para que importe de `common` (no de `columns/bar_schedule.py` etc.).
- **TI.4** Borrar shims back-compat solo si no hay imports externos a ellos.
- **TI.5** Re-correr `example_column.ipynb`, `example_beam.ipynb`, `example_wall.ipynb` end-to-end.
- **TI.6** Actualizar `design/AGENTS.md` con cualquier desviación o lección aprendida en Fase 1.

### 6.3 Acceptance criteria
- [ ] `pytest` pasa entero.
- [ ] Notebooks corren limpios.
- [ ] `grep -rn "0\.17 \* " design/{beams,columns,walls}/` solo en docstrings.

---

## 7. Comunicación entre agentes

- Si un agente paralelo descubre que necesita algo en `common/` que Fase 0 no creó: termina su trabajo limpio (con notes en el output), reporta la necesidad, y se hace una "Fase 0.5" antes de continuar.
- Si un agente paralelo descubre un bug en otro módulo (no en el suyo): NO lo arregla. Lo reporta en el output.
- Cada agente debe correr el smoke test al final (`python design/_smoke_test.py` o equivalente) y reportar pass/fail.
