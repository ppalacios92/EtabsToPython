# design.columns

Reinforced-concrete column: biaxial P-M-M surface, shear, and confinement
(§18.7). See the guide: [Columns](../guides/columns.md).

`Surface.volume(..., close_top=True)` closes the P-M-M plot mesh with the
flat `φPn,max` plateau. Column Vc follows Table 22.5.5.1 through
`design.common.shear.vc_one_way`.

## Column

::: design.columns.column

## Interaction diagram

::: design.columns.interaction

## Interaction surface (3D)

::: design.columns.interaction_surface

## Surface facade

::: design.columns.surface

## Optimize

::: design.columns.optimize

## Plotting

::: design.columns.plotting
