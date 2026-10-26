# EXAMPLE_REVIEW — Revisión y diseño ACI 318-25

Tres notebooks que ejercitan el módulo `design/` sobre vigas, columnas y muros,
siguiendo el procedimiento secuencial de análisis del ACI 318-25 y verificando
cada resultado contra su cláusula.

## Notebooks

| Notebook | Elemento | Cláusulas principales |
|---|---|---|
| `01_vigas_review.ipynb`    | Viga (flexión + corte, SMF)         | §9, §18.6, §22.2, §22.5 |
| `02_columnas_review.ipynb` | Columna (P-M-M + corte + confin.)   | §10, §18.7, §22.4, §25.7 |
| `03_muros_review.ipynb`    | Muro especial + elementos de borde  | §11, §18.10, §22.2 |

## Convención de unidades

- **Resultados** (fuerzas, momentos, longitudes): **m** y **tonf** (`Tonf_m_C`).
  1 tonf = 9.80665 kN.
- **Materiales** ($f'_c$, $f_y$, $E_s$): **MPa** (N/mm²).
- Internamente el código calcula en MPa / mm / kN / kN·m; la presentación en
  tonf·m es solo de salida.

## Estructura de cada notebook (dos ejercicios)

### Ejercicio 1 — Evaluar y verificar un armado típico

1. **Sección con armadura mínima de norma.** Se parte de una sección con el
   armado que apenas cumple los mínimos (As,min de flexión §9.6.1 / ρg,min
   §10.6.1 / ρ,min distribuido §18.10.2). Es la "línea base".
2. **Capacidad.** Se calcula `element.capacity()`: diagrama de interacción P-M
   (racional, con tope φPn,max), Mpr (incluido Mpr a P=0), φVn, confinamiento.
3. **Demandas.** Se aplican demandas mayoradas (Pu, Mu, Vu, δu) en tonf·m.
4. **Verificación.** `element.check(demands)` reporta razones demanda/capacidad
   y qué cláusulas gobiernan. Si no cumple, se identifica el modo de falla.
5. **Ajuste.** Se incrementa el armado desde la línea base hasta que el diseño
   cumple las demandas, mostrando el camino.

### Ejercicio 2 — Diseñar desde la sección y las demandas

1. **Sección + demandas.** Geometría fija y demandas dadas.
2. **Diseño automático.** `element.design(demands)` propone el armado
   (longitudinal + transversal + elementos de borde en muros) que cumple la
   norma, partiendo de la armadura mínima.
3. **Resumen.** Tabla ordenada del diseño con cada cantidad trazada a su
   cláusula, diagramas de interacción y Mpr.

## Secciones de partida (parametrizadas, cumplen mínimos)

Definidas al inicio de cada notebook para que se puedan cambiar:

- **Viga:** 300×600 mm, fc = 28 MPa, fy = 420 MPa (Gr 60), armadura mínima.
- **Columna:** 600×600 mm, ρg ≈ 1 % (mínimo §10.6.1), fc = 28 MPa.
- **Muro:** 300×4000 mm (tw×lw), hw alto, ρ distribuido 0.0025 (§18.10.2.1),
  elementos de borde según §18.10.6.

## Documentación asociada

Procedimientos y fórmulas documentados en el vault:
`pxpalacios/ACI/SCRIPTS/` (procedimiento seccional, correcciones, mapa
fórmula↔módulo, ejercicios).
