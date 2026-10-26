# design.walls

Structural wall: web + boundary elements, iterative design (§18.10). See the
guide: [Walls](../guides/walls.md).

`Wall(..., n_walls_sharing=1)` exposes individual and common-lateral-force
shear limits through `WallCapacity.Vn_max_individual`,
`Vn_max_group_per_wall`, and governing `Vn_max`. Boundary-element bar curtains
require `n_x_be >= 2`.

## Wall

::: design.walls.wall

## Wall pier classification

::: design.walls.wall_pier

## Boundary elements

::: design.walls.boundary

## Coupling beams

::: design.walls.coupling_beam

## Plotting

::: design.walls.plotting
