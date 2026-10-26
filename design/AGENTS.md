# design/AGENTS.md — Contrato Beam / Column / Wall

Este documento es el contrato vinculante entre los tres elementos de
`design/` (Beam, Column, Wall) y la capa común (`design/common/`).
Cualquier agente que toque código bajo `design/` debe leerlo antes.

---

## 1. Unidades internas

- **Stress**: MPa
- **Length**: mm (INCLUYE `lu`, `ln`, `hw`, `cover` — NO metros)
- **Force**: kN
- **Moment**: kN·m
- **Area**: mm²

Los cálculos internos NUNCA cambian: siempre MPa / mm / kN / kN·m.
NO hay heurísticas de magnitud (`lu < 30 ⇒ metros`); el sistema de
unidades es **explícito**.

**Entrada de data** (interpretar lo que el usuario mete):
- La GEOMETRÍA siempre es canónica **mm** — en los factories
  `.rectangular(...)` y en los constructores desnudos por igual. El
  argumento `units=` del factory es **SOLO presentación**: fija `self.units`
  pero NUNCA reinterpreta la geometría (nada de `to_internal` sobre dims o
  diámetros de barra). Coherente con las ecuaciones de `common/`.
- Los constructores desnudos (`Column.__init__`, `RectangularSection`,
  `Concrete`, `perimeter_bars`) son **INTERNAL-only**: asumen mm/kN/MPa
  sin convertir.
- Las CARGAS (no la geometría) sí son unit-aware: pasan por
  `elem.demands(Pu=..., Mux=..., units=)`, que convierte usando `units` o,
  si se omite, `self.units`. Por eso `set_units(...)` cambia tanto el display
  como cómo `demands()` lee fuerzas/momentos/stress posteriores.

**Salida de data** (presentación): `self.units` gobierna lo que muestran
`summary()`, `report()` y los plots. Se fija con `set_units(...)`.

NO existe global mutable de unidades (se eliminó `_ACTIVE_UNITS` /
`get_active_units` / `set_active_units`). Las unidades viven en cada
objeto (`self.units`). El default cuando no se pasa nada es
`DEFAULT_UNITS = units_from(9)` (N_mm_C), donde N/mm² = MPa y mm = mm,
así la conversión es identidad.

`to_internal(kind, value, *, units)` exige `units` (keyword obligatorio).
NO hay `*1000` o `/1000` flotantes dispersos en `wall.py` / `column.py`
/ `beam.py`; la única conversión de unidades vive en `demands()` (cargas).
La geometría entra ya en mm, no se convierte.

Las properties `lu_m`, `ln_m`, `hw_m` están permitidas como vista
legacy en metros sobre el dato canónico en mm.

---

## 2. Naming canónico

| Concepto | Nombre canónico |
|---|---|
| Recubrimiento al exterior del estribo | `cover` (alias: `clear_cover`) |
| Acero transversal | `steel_transverse: Steel`; `fyt` es property |
| Longitud confinada | `lo` |
| Espaciamiento confinado | `spacing_confined` |
| Espaciamiento medio | `spacing_middle` |
| Momento probable | `Mpr` (o `Mpr_x`, `Mpr_y`, `Mpr_pos`, `Mpr_neg`, `Mpr_in_plane`); NUNCA `Mn_pr` |
| Shear capacidad-design | `Ve_capacity`, `Ve_used`, `amplification` |

Los alias legacy (`Mn_pr_pos`, `clear_cover`, etc.) pueden mantenerse
como property o kwarg-alias durante una fase de transición, pero el
docstring debe declarar el nombre canónico.

---

## 3. Contrato (`design.common.contracts.Element`)

Cada elemento implementa:

```python
def run(self) -> Self                  # lazy + cache
def capacity(self) -> XCapacity
def check(self, demands) -> XCheck
def design(self, demands=None) -> XDesignResults
                                       # minimum / demand / capacity / envelope
def evolve(self, proposal) -> Self     # INMUTABLE: devuelve NUEVO objeto
def apply(self, proposal) -> Self      # alias de evolve
def summary(self) -> None
def report(self) -> dict
def set_units(self, code_or_name) -> Self
# property:
.plot                                  # lazy plotter property
```

`isinstance(elem, design.common.contracts.Element)` debe ser True
para Beam, Column, Wall.

---

## 4. Mutabilidad: SIEMPRE inmutable vía `evolve()`

- `evolve(p)` devuelve **NUEVO** elemento. NO muta `self`.
- `apply(p)` es alias de `evolve()`.
- `design()` NO muta `self.design_results`. El caller guarda el retorno.
- `run()` SÍ muta caches internos (`_surface`, `_phi_Vn`, etc.). Devuelve `self`.

Cualquier `self.design_results = results` en `.design()` es un bug.

---

## 5. Lista negra de hardcodes

NINGÚN archivo en `beams/`, `columns/`, `walls/` puede contener literal:

| Hardcode | Reemplazo |
|---|---|
| `0.17 * lam * sqrt_fc * b * d / 1000` | `vc_simplified(b, d, concrete)` |
| one-way Table 22.5.5.1 Vc logic | `lambda_s(d)`, `vc_one_way(...)` |
| two-way §22.6 Vc logic | `bo_rectangular(...)`, `vc_two_way(...)`, `vc_two_way_force(...)` |
| `Av * fyt * d / (s * 1000)` | `vs_capacity(Av, fyt, d, s)` |
| `(Vu/phi - Vc) * 1000 / (fyt * d)` | `av_s_required(...)` |
| `0.85 - 0.05*(fc-28)/7` | `beta1(fc)` |
| `phi = 0.90 if tc else 0.65` | `phi_axial_flexure(eps_t, eps_ty)` |
| `0.83 * sqrt(fc) * Acv / 1000` | `vn_wall_max(Acv, fc)` |
| wall flange/group shear limits | `alpha_sh(...)`, `vn_wall_group_max(...)` |
| `phi = 0.75` (shear) | `phi_shear()` |
| `Pu < Ag*fc/20` (§18.7.6.2.1 trigger) | `vc_zero_seismic(...)` |

Las únicas excepciones permitidas son strings en docstrings explicando
la ecuación de origen, NO la implementación activa.

---

## 6. Vc=0 sísmico

Tanto vigas (§18.6.5.2) como columnas (§18.7.6.2.1) deben evaluar
`Vc=0` en `lo` cuando aplica:

- §18.6.5.2 (vigas): `V_seismic >= 0.5 V_total` Y `Pu < Ag·fc/20`.
- §18.7.6.2.1 (columnas): misma condición.

Centralizado en `design.common.shear.vc_zero_seismic`.

Muros NO aplican (§18.10 tiene su propio mecanismo de capacidad).

---

## 7. Mapeo Wall → BE Column

`design.common.contracts.wall_demands_to_be(WallDemands) -> ColumnDemands`

Mapeo canónico:

```
Pu      -> Pu
Mu      -> Mux   (in-plane bending del muro = eje fuerte de la BE)
Mu_out  -> Muy   (out-of-plane del muro = eje débil de la BE)
Vu      -> Vuy   (in-plane shear → corta perpendicular al eje fuerte)
Vu_out  -> Vux   (out-of-plane shear → corta perpendicular al eje débil)
```

Cualquier `wall.py` que construya `ColumnDemands` debe usar este helper.

---

## 8. PlotStyle

UNA sola `PlotStyle` en `design.common.plot_style`. Cada elemento
expone `element.style`. Plotters leen de `self.element.style`.

`design/columns/style.py`, `design/beams/style.py` y
`design/walls/style.py` son shims de re-export por compatibilidad.

`BeamPlotStyle` y `WallPlotStyle` son alias de `PlotStyle`.

---

## 9. OptimizeAlternative

UNA base en `design.common.optimize.OptimizeAlternative`
(`dataclass(frozen=True, slots=True)`):

```
proposal: object             # element-specific DesignProposal
detailing: dict              # element-specific keys
rho_transverse: float        # kg/m³ concreto
rho_longitudinal: float
rho_total: float
feasible: bool
is_baseline: bool = False
is_provided: bool = False
notes: tuple[str, ...] = ()
```

Cada elemento puede subclasear o usar la base con `detailing` dict.
Los tres exponen `element.run_optimize(demands)`.

`format_optimize_table` vive en common y acepta un `detail_fmt(alt) -> str`
para customizar la columna "detalle" del print.

---

## 10. BarSchedule

UN `BarSchedule` en `design.common.bar_schedule`. Los elementos lo
aceptan por kwarg `bar_schedule=None` y crean default si no se pasa.

---

## 11. Tests

`tests/test_contract.py` itera `[beam, col, wall]` y verifica todo lo
anterior. Cada Fase agrega tests propios bajo `tests/test_<elemento>*.py`.

`tests/test_common_*.py` cubre la capa común en aislación.

---

## 12. Reglas de oro para agentes

1. **Si un archivo no aparece en tu columna de la matriz de propiedad
   (REFACTOR_GUIDE.md sección 1), NO lo tocás.** Si necesitás un cambio
   cruzado, lo reportás en el output, no lo hacés.

2. **Si necesitás algo nuevo en `common/`, NO lo agregues solo.**
   Solo Fase 0 puede crecer `common/`. Fases 1A/1B/1C consumen `common/`,
   no lo extienden.

3. **Si descubrís un bug en otro módulo, NO lo arregles.** Lo reportás.

4. **Smoke test al final SIEMPRE.** `python design/_smoke_test.py` debe
   pasar al cerrar tu fase.
