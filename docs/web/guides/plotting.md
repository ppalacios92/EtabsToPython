# Plotting

Every design element exposes a lazy `plot` property — a bound plotter that draws
sections, interaction diagrams, demand envelopes, and per-element dashboards on
`matplotlib` axes. A standalone package ([`design.plotting`](../api/plotting.md))
draws diagrams independent of any element.

All plotters share one [`PlotStyle`](common.md) (colours, line widths, `figsize_*`)
reachable through `element.style`, render in the element's presentation `units`,
and accept `ax=` to draw into an existing axes (omit it to create a figure). They
return the `matplotlib` axes (or figure, for dashboards).

## Beam plotter

| Method | Signature |
|---|---|
| section | `section(*, ax=None, show_bar_labels=False, bar_scale=None)` |
| moment envelope | `moment_envelope(demands, *, ax=None)` — `Mu±` per station vs `φMn±` |
| shear envelope | `shear_envelope(demands, *, ax=None)` — `Vu` per station vs `φVn` / `Ve` |
| demand | `demand(demands, *, ax=None)` — alias of `moment_envelope` |
| P-M curve | `pm(*, direction="both", ax=None)` — `'pos'`·`'neg'`·`'both'` |
| dashboard | `dashboard(*, figsize=None, demands=None)` — 2×2: section, capacity, moment, shear |

## Column plotter

| Method | Signature |
|---|---|
| section | `section(*, ax=None, show_bar_labels=False, bar_scale=None)` |
| P-M | `pm(*, angle=0.0, ax=None)` — one neutral-axis angle (deg) |
| compare | `pm_compare(*, angles=(0.0, 45.0, 90.0), ax=None)` |
| 3D volume | `pmm_volume(*, ax=None, n_Pu=25, wireframe=True)` |
| biaxial slice | `pmm_slice(*, Pu, demand=None, ax=None, fill=True)` |
| slice + demand | `pmm_with_demand(*, Pu, Mux, Muy, ax=None)` |
| demand | `demand(demands, *, ax=None)` |
| dashboard | `dashboard(*, figsize=None, demands=None)` |

## Wall plotter

| Method | Signature |
|---|---|
| section | `section(*, ax=None, show_bar_labels=False, bar_scale=None)` |
| BE section | `be_section(*, which="top", ax=None)` — `"top"`·`"bot"` |
| P-M | `pm(*, plane="in-plane", ax=None)` — `"in-plane"`·`"out-of-plane"` |
| compare | `pm_compare(*, ax=None)` — in-plane vs out-of-plane |
| 3D volume | `pmm_volume(*, ax=None, n_Pu=25, wireframe=True)` |
| biaxial slice | `pmm_slice(*, Pu, demand=None, ax=None)` |
| demand | `demand(demands, *, ax=None)` |
| dashboard | `dashboard(*, demands=None, figsize=None)` |
| iteration history | `iteration_history(results, *, ax=None)` — BE-length convergence |

## Joint plotter

| Method | Signature |
|---|---|
| section | `section(*, ax=None)` — plan: column core + framing-beam footprints |
| demand | `demand(*, ax=None)` — `Vu` vs `φVn` for X and Y |
| dashboard | `dashboard(*, figsize=None)` — section + demand |

## Standalone diagram functions

`design.plotting.interaction` plots from raw analysis objects — handy for custom
comparisons:

| Function |
|---|
| `plot_pm_diagram(diagram, *, ax=None, show_nominal=True, show_reduced=True, nominal_kwargs=None, reduced_kwargs=None, units=None)` |
| `plot_pmm_slice(surface, *, Pu, ax=None, demand=None, fill=True, units=None)` |
| `plot_pmm_volume(surface, *, ax=None, n_Pu=25, Pu_min=None, Pu_max=None, surface_alpha=0.25, wireframe=True, cmap="viridis", units=None)` |
| `plot_pmm_with_demand(surface, *, Pu, Mux, Muy, ax=None, n_Pu=25, units=None)` |

`design.plotting.sections.plot_section(section, *, ax=None, show_bar_labels=False,
bar_scale=1.0, concrete_color="#D8D8D8", rebar_color="tab:blue", units=None)` draws
any `Section`'s concrete polygons + rebar with a totals title.

## Reference

[`design.plotting` API](../api/plotting.md). For ETABS model and story-envelope
plots, see [ETABS extraction](etabs_extraction.md).
