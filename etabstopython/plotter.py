import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from etabstopython.tools import extrude_frame
from etabstopython.tools import compute_story_displacement_bounds
from etabstopython.tools import compute_story_drift_bounds
from etabstopython.tools import compute_story_force_bounds
from etabstopython.tools import compute_story_shear_from_elements
from etabstopython.tools import story_shear_envelope_from_elements
from etabstopython.tools import story_max_over_avg_ratios


def plot_structure_3d(model):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # === GRAFICAR VIGAS Y COLUMNAS EXTRUIDAS ===
    for _, row in model.frames_df.iterrows():
        try:
            color = model.section_color_dict.get(row['SectProp'], (0.6, 0.6, 0.6))

            p1 = [row['PointIX'], row['PointIY'], row['PointIZ']]
            p2 = [row['PointJX'], row['PointJY'], row['PointJZ']]

            dims = model.section_dim_dict.get(row['SectProp'], {'t2': 0.3, 't3': 0.5})
            width = dims['t2']
            height = dims['t3']

            extrude_frame(ax, p1, p2, width, height, color)
        except Exception as e:
            print(f"⚠️ Error plotting frame {row['UniqueName']}: {e}")

    # === GRAFICAR MUROS COMO SHELLS ===
    if hasattr(model, 'wall_object_connectivity') and model.wall_object_connectivity is not None:
        df = model.wall_object_connectivity
        if not df.empty:
            poc = model.point_object_connectivity.set_index('UniqueName')
            for _, wall in df.iterrows():
                try:
                    points = [wall['UniquePt1'], wall['UniquePt2'], wall['UniquePt3'], wall['UniquePt4']]
                    coords = poc.loc[points, ['X', 'Y', 'Z']].values.astype(float)
                    verts_wall = [list(zip(coords[:, 0], coords[:, 1], coords[:, 2]))]
                    ax.add_collection3d(Poly3DCollection(verts_wall, alpha=0.5, edgecolor='k', facecolor='orange'))
                except Exception as e:
                    print(f"⚠️ Error plotting wall: {e}")

    # === GRAFICAR LOSAS COMO SHELLS ===
    if hasattr(model, 'floor_object_connectivity') and model.floor_object_connectivity is not None:
        df = model.floor_object_connectivity
        if not df.empty:
            poc = model.point_object_connectivity.set_index('UniqueName')
            pt_cols = ['UniquePt1', 'UniquePt2', 'UniquePt3', 'UniquePt4']
            for _, group in df.groupby('UniqueName', sort=False):
                try:
                    point_ids = []
                    for value in group[pt_cols].values.flatten():
                        key = str(value)
                        if key not in ["", "nan", "None", "__E2P_NULL__"] and key not in point_ids:
                            point_ids.append(key)
                    coords = poc.loc[point_ids, ['X', 'Y', 'Z']].values.astype(float)
                    if len(coords) >= 3:
                        verts = [list(zip(coords[:, 0], coords[:, 1], coords[:, 2]))]
                        ax.add_collection3d(Poly3DCollection(verts, alpha=0.4, edgecolor='k', facecolor='gray'))
                except Exception as e:
                    print(f"⚠️ Error plotting floor: {e}")

    # === GRAFICAR APOYOS ===
    z_zero = model.frames_df[(model.frames_df['PointIZ'] == 0) | (model.frames_df['PointJZ'] == 0)]
    for _, row in z_zero.iterrows():
        if row['PointIZ'] == 0:
            ax.scatter(row['PointIX'], row['PointIY'], row['PointIZ'], color='purple', marker='s', s=100)
        if row['PointJZ'] == 0:
            ax.scatter(row['PointJX'], row['PointJY'], row['PointJZ'], color='purple', marker='s', s=100)

    # === ESCALADO ===
    # all_x = np.concatenate([model.frames_df['PointIX'], model.frames_df['PointJX']]).astype(float)
    # all_y = np.concatenate([model.frames_df['PointIY'], model.frames_df['PointJY']]).astype(float)
    # all_z = np.concatenate([model.frames_df['PointIZ'], model.frames_df['PointJZ']]).astype(float)


    all_x = model.point_object_connectivity['X'].to_numpy(dtype=float)
    all_y = model.point_object_connectivity['Y'].to_numpy(dtype=float)
    all_z = model.point_object_connectivity['Z'].to_numpy(dtype=float)


    mid_x = (all_x.min() + all_x.max()) / 2
    mid_y = (all_y.min() + all_y.max()) / 2
    mid_z = (all_z.min() + all_z.max()) / 2
    max_range = max(all_x.max() - all_x.min(),
                    all_y.max() - all_y.min(),
                    all_z.max() - all_z.min()) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('Eje X (m)')
    ax.set_ylabel('Eje Y (m)')
    ax.set_zlabel('Eje Z (m)')
    ax.set_title(f"3D View Structure - PRY: {model.name}", fontweight='bold', fontsize=10)
    ax.view_init(elev=10, azim=50)

    fig.text(0.99, -0.01, '© 2025 - Patricio Palacios B. - Torrefuerte', 
            ha='right', va='bottom', fontsize=9, color='gray', style='italic')


    plt.show()


def plot_story_displacement_bounds(model, combos_comp, color=[0.7, 0.7, 0.7], lw=1.0, highlight_combo=None , factor=1.0):

    # === Calcular desplazamientos para combos principales ===
    df_plot = compute_story_displacement_bounds(model, combos_comp , factor)

    # === Calcular desplazamientos solo para el combo destacado si no está en la lista ===
    df_plot_highlight = None
    if highlight_combo and highlight_combo not in combos_comp:
        df_plot_highlight = compute_story_displacement_bounds(model, [highlight_combo] , factor)

    # === Crear figura y subplots ===
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharex=False)

    # === Plot para combinaciones normales ===
    for combo in combos_comp:
        data = df_plot[df_plot['OutputCase'] == combo]
        height = data['Height'].values

        disp_min_x = data['Ux_min'].values
        disp_max_x = data['Ux_max'].values
        disp_min_y = data['Uy_min'].values
        disp_max_y = data['Uy_max'].values

        ax1.plot(disp_min_x, height, 'o--', label=f'{combo} - min', color=color, alpha=0.6)
        ax1.plot(disp_max_x, height, 'o-',  label=f'{combo} - max', color=color, linewidth=lw)

        ax2.plot(disp_min_y, height, 'o--', label=f'{combo} - min', color=color, alpha=0.6)
        ax2.plot(disp_max_y, height, 'o-',  label=f'{combo} - max', color=color, linewidth=lw)

    # === Plot para highlight_combo si existe ===
    if highlight_combo:
        # Si ya se calculó en df_plot principal
        if highlight_combo in combos_comp:
            data_h = df_plot[df_plot['OutputCase'] == highlight_combo]
        else:
            data_h = df_plot_highlight[df_plot_highlight['OutputCase'] == highlight_combo]

        height = data_h['Height'].values
        disp_min_x = data_h['Ux_min'].values
        disp_max_x = data_h['Ux_max'].values
        disp_min_y = data_h['Uy_min'].values
        disp_max_y = data_h['Uy_max'].values

        ax1.plot(disp_min_x, height, 'o--', label=f'{highlight_combo} - min', color='tab:red', alpha=0.8)
        ax1.plot(disp_max_x, height, 'o-',  label=f'{highlight_combo} - max', color='tab:blue', linewidth=2.5)

        ax2.plot(disp_min_y, height, 'o--', label=f'{highlight_combo} - min', color='tab:red', alpha=0.8)
        ax2.plot(disp_max_y, height, 'o-',  label=f'{highlight_combo} - max', color='tab:blue', linewidth=2.5)

    # === Formato subplot Ux ===
    ax1.set_xlabel('Displacement Ux [m]', fontweight='bold')
    ax1.set_ylabel('Accumulated Height [m]', fontweight='bold')
    ax1.set_title(f"Story Displacement Ux - Model: {model.name}", fontweight='bold', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # === Formato subplot Uy ===
    ax2.set_xlabel('Displacement Uy [m]', fontweight='bold')
    ax2.set_ylabel('Accumulated Height [m]', fontweight='bold')
    ax2.set_title(f"Story Displacement Uy - Model: {model.name}", fontweight='bold', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # === Etiquetas de pisos al lado derecho ===
    story_names = model.story_definitions['Story'].tolist()
    story_heights = model.story_definitions['Accumulated_Height'].tolist()

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylim(ax1.get_ylim())
    ax1_twin.set_yticks(story_heights)
    ax1_twin.set_yticklabels(story_names, fontsize=6)
    ax1_twin.set_ylabel("Story")

    ax2_twin = ax2.twinx()
    ax2_twin.set_ylim(ax2.get_ylim())
    ax2_twin.set_yticks(story_heights)
    ax2_twin.set_yticklabels(story_names, fontsize=6)
    ax2_twin.set_ylabel("Story")

    fig.text(0.99, -0.01, '© 2025 - Patricio Palacios B. - Torrefuerte', 
                ha='right', va='bottom', fontsize=9, color='gray', style='italic')


    plt.tight_layout()
    plt.show()


def plot_story_drift_bounds(model, combos_comp, color=[0.7, 0.7, 0.7], lw=1.0, highlight_combo=None, factor=1.0):

    # === Calcular desplazamientos ===
    df_plot = compute_story_drift_bounds(model, combos_comp, factor)
    # === Crear figura ===
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharex=False)

    # === Función auxiliar para calcular derivas ===
    def compute_drift(heights, disp):
        drift = []
        for i in range(1, len(disp)):
            h = heights[i] - heights[i - 1]
            d = abs(disp[i] - disp[i - 1]) / h * 100 
            drift.append(d)
        return np.insert(drift, 0, 0)

    # === Combos normales ===
    for combo in combos_comp:
        data = df_plot[df_plot['OutputCase'] == combo].copy()
        data = data.sort_values(by='Height')

        height = data['Height'].values
        drift_x = data['DriftX'].values
        drift_y = data['DriftY'].values

        ax1.plot(drift_x, height, 'o-', color=color, linewidth=lw, alpha=0.9, label=combo)
        ax2.plot(drift_y, height, 'o-', color=color, linewidth=lw, alpha=0.9, label=combo)

    # === Combo destacado ===
    if highlight_combo:
        if highlight_combo in combos_comp:
            data_h = df_plot[df_plot['OutputCase'] == highlight_combo]
        else:
            data_h = (
                df_plot.groupby(['Story', 'Height'], as_index=False)[['DriftX', 'DriftY']]
                .max()
            )
        data_h = data_h.sort_values(by='Height')

        height = data_h['Height'].values
        drift_x = data_h['DriftX'].values
        drift_y = data_h['DriftY'].values

        ax1.plot(drift_x, height, 'o-', color='tab:red', linewidth=2.5, label=f'{highlight_combo}')
        ax2.plot(drift_y, height, 'o-', color='tab:blue', linewidth=2.5, label=f'{highlight_combo}')

    # === Formato subplot Ux ===
    ax1.set_xlabel('Drift X [%]', fontweight='bold')
    ax1.set_ylabel('Accumulated Height [m]', fontweight='bold')
    ax1.set_title(f"Story Drift Ux - Model: {model.name}", fontweight='bold', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # === Formato subplot Uy ===
    ax2.set_xlabel('Drift Y [%]', fontweight='bold')
    ax2.set_ylabel('Accumulated Height [m]', fontweight='bold')
    ax2.set_title(f"Story Drift Uy - Model: {model.name}", fontweight='bold', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # === Etiquetas de pisos al lado derecho ===
    story_names = model.story_definitions['Story'].tolist()
    story_levels = model.story_definitions['Accumulated_Height'].tolist()

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylim(ax1.get_ylim())
    ax1_twin.set_yticks(story_levels)
    ax1_twin.set_yticklabels(story_names, fontsize=6)
    ax1_twin.set_ylabel("Story")

    ax2_twin = ax2.twinx()
    ax2_twin.set_ylim(ax2.get_ylim())
    ax2_twin.set_yticks(story_levels)
    ax2_twin.set_yticklabels(story_names, fontsize=6)
    ax2_twin.set_ylabel("Story")

    fig.text(0.99, -0.01, '© 2025 - Patricio Palacios B. - Torrefuerte', 
            ha='right', va='bottom', fontsize=9, color='gray', style='italic')


    plt.tight_layout()
    plt.show()




def plot_story_shear_bounds(model, combos_comp, color=[0.7, 0.7, 0.7], lw=1.0, highlight_combo=None):
    # === Obtener alturas de los pisos ===
    altura_dict = model.story_definitions.set_index('Story')['Accumulated_Height'].to_dict()
    altura_pisos = model.story_definitions.set_index('Story')['Height'].to_dict()

    # === Calcular resumen de combos normales ===
    resumen_max = compute_story_force_bounds(model, combos_comp)

    # === Crear figura ===
    fig, ax = plt.subplots(figsize=(9, 6))

    # === Graficar combos comunes ===
    for combo in combos_comp:
        if highlight_combo and combo == highlight_combo:
            continue  # Se grafica después
        df_plot = resumen_max[resumen_max['OutputCase'] == combo].copy()
        vx_vals, vy_vals, h_infs = [], [], []

        for _, row in df_plot.iterrows():
            piso = row['Story']
            if pd.isna(piso) or piso not in altura_dict:
                continue

            h_sup = altura_dict[piso]
            h_inf = h_sup - altura_pisos[piso]
            vx = row['VX']
            vy = -row['VY']  # Vy a la izquierda

            ax.plot([vx, vx], [h_inf, h_sup], color=color, linestyle='-', alpha=0.8, linewidth=lw)
            ax.plot([vy, vy], [h_inf, h_sup], color=color, linestyle='-', alpha=0.8, linewidth=lw)

            vx_vals.append(vx)
            vy_vals.append(vy)
            h_infs.append(h_inf)

        for j in range(len(h_infs) - 1):
            ax.plot([vx_vals[j], vx_vals[j+1]], [h_infs[j], h_infs[j]], color=color, linestyle='-', alpha=0.8, linewidth=lw)
            ax.plot([vy_vals[j], vy_vals[j+1]], [h_infs[j], h_infs[j]], color=color, linestyle='-', alpha=0.8, linewidth=lw)

    # === Graficar combo destacado ===
    if highlight_combo:
        available_cases = set(model.story_forces['OutputCase'].dropna().unique())
        if highlight_combo in available_cases:
            resumen_high = compute_story_force_bounds(model, [highlight_combo])
            df_high = resumen_high[resumen_high['OutputCase'] == highlight_combo].copy()
        else:
            force_cols = ['P', 'VX', 'VY', 'MX', 'MY']
            df_high = (
                resumen_max.groupby('Story', as_index=False, observed=False)[force_cols]
                .max()
            )
            df_high['OutputCase'] = highlight_combo
            orden_pisos = model.story_definitions['Story'].tolist()
            df_high['Story'] = pd.Categorical(df_high['Story'], categories=orden_pisos, ordered=True)
            df_high.sort_values('Story', inplace=True)

        vx_vals, vy_vals, h_infs = [], [], []

        for _, row in df_high.iterrows():
            piso = row['Story']
            if pd.isna(piso) or piso not in altura_dict:
                continue

            h_sup = altura_dict[piso]
            h_inf = h_sup - altura_pisos[piso]
            vx = row['VX']
            vy = -row['VY']

            ax.plot([vx, vx], [h_inf, h_sup], color='tab:blue', linestyle='-', linewidth=2.5)
            ax.plot([vy, vy], [h_inf, h_sup], color='tab:red', linestyle='-', linewidth=2.5)

            vx_vals.append(vx)
            vy_vals.append(vy)
            h_infs.append(h_inf)

        for j in range(len(h_infs) - 1):
            ax.plot([vx_vals[j], vx_vals[j+1]], [h_infs[j], h_infs[j]], color='tab:blue', linestyle='-', linewidth=2.5)
            ax.plot([vy_vals[j], vy_vals[j+1]], [h_infs[j], h_infs[j]], color='tab:red', linestyle='-', linewidth=2.5)

    # === Formato ===
    # ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Shear Force [Tonf]', fontweight='bold')
    ax.set_ylabel('Height [m]', fontweight='bold')
    ax.set_title(f'Story Shear - Model: {model.name}', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

    # === Etiquetas de pisos al lado derecho ===
    story_names = model.story_definitions['Story'].tolist()
    story_heights = model.story_definitions['Accumulated_Height'].tolist()

    ax_twin = ax.twinx()
    ax_twin.set_ylim(ax.get_ylim())
    ax_twin.set_yticks(story_heights)
    ax_twin.set_yticklabels(story_names, fontsize=6)
    ax_twin.set_ylabel("Story")

    fig.text(0.99, -0.01, '© 2025 - Patricio Palacios B. - Torrefuerte', 
             ha='right', va='bottom', fontsize=9, color='gray', style='italic')


    # === Leyenda manual ===
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=[0.7, 0.7, 0.7], lw=1.0, linestyle='-', label='All Rlz'),
        Line2D([0], [0], color='tab:blue', lw=2.5, linestyle='-', label='Highlight - VX'),
        Line2D([0], [0], color='tab:red', lw=2.5, linestyle='-', label='Highlight - VY'),
    ]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)


    plt.tight_layout()
    plt.show()





def _draw_story_shear_steps(ax, df, altura_dict, altura_pisos,
                            color_vx, color_vy, lw, linestyle='-', alpha=1.0):
    """Draw VX (right, +) and VY (left, -) as constant steps over each story band."""
    vx_vals, vy_vals, h_infs = [], [], []
    for _, row in df.iterrows():
        piso = row['Story']
        if pd.isna(piso) or piso not in altura_dict:
            continue

        h_sup = altura_dict[piso]
        h_inf = h_sup - altura_pisos[piso]
        vx = row['VX']
        vy = -row['VY']  # Vy a la izquierda

        ax.plot([vx, vx], [h_inf, h_sup], color=color_vx, linestyle=linestyle, linewidth=lw, alpha=alpha)
        ax.plot([vy, vy], [h_inf, h_sup], color=color_vy, linestyle=linestyle, linewidth=lw, alpha=alpha)

        vx_vals.append(vx)
        vy_vals.append(vy)
        h_infs.append(h_inf)

    for j in range(len(h_infs) - 1):
        ax.plot([vx_vals[j], vx_vals[j+1]], [h_infs[j], h_infs[j]], color=color_vx, linestyle=linestyle, linewidth=lw, alpha=alpha)
        ax.plot([vy_vals[j], vy_vals[j+1]], [h_infs[j], h_infs[j]], color=color_vy, linestyle=linestyle, linewidth=lw, alpha=alpha)


def plot_story_shear_bounds_elements(model, combos_comp, color=[0.7, 0.7, 0.7], lw=1.0,
                                     highlight_combo=None, elements='columns',
                                     rebuild_envelope=False,
                                     transition_tol=None, exclude_piers=None):
    """Story shear obtained by summing the vertical elements per story.

    Same look as :func:`plot_story_shear_bounds`, but instead of reading
    'Story Forces' it sums the shear of the vertical elements at their base
    (column I-node / pier ``Bottom``).

    ``elements``:
      - ``'columns'`` : frame columns only.
      - ``'walls'``   : piers only.
      - ``'all'``     : columns + piers. The highlight combo is drawn decomposed:
        the total in tab:blue (VX) / tab:red (VY), the column part in solid black
        and the pier part in dashed black.

    Extra options:
      - ``rebuild_envelope`` : build the highlight as the correct design envelope
        ``env(sum over combos_comp)`` instead of a pre-enveloped ``highlight_combo``
        (which inflates as ``sum(env)``). Use when ``highlight_combo`` is an ETABS
        envelope case like 'EnveDiseño'.
      - ``transition_tol`` : if set (e.g. 0.5), floors where the element sum
        deviates from Story Forces by more than this fraction (transition floors
        where the pier reports local transfer, not story shear) are warned about
        and omitted from the highlight trace.
      - ``exclude_piers`` : pier names to drop (avoid double counting an aggregate
        pier coexisting with its components).
    """
    # === Obtener alturas de los pisos ===
    altura_dict = model.story_definitions.set_index('Story')['Accumulated_Height'].to_dict()
    altura_pisos = model.story_definitions.set_index('Story')['Height'].to_dict()

    # === Calcular cortante de piso sumando elementos verticales ===
    resumen_max = compute_story_shear_from_elements(model, combos_comp, elements, exclude_piers)

    # === Crear figura ===
    fig, ax = plt.subplots(figsize=(9, 6))

    # === Graficar combos comunes (gris) ===
    for combo in combos_comp:
        if highlight_combo and combo == highlight_combo:
            continue  # Se grafica después
        df_plot = resumen_max[resumen_max['OutputCase'] == combo]
        _draw_story_shear_steps(ax, df_plot, altura_dict, altura_pisos,
                                color, color, lw, '-', 0.8)

    # === Graficar combo destacado ===
    if highlight_combo:
        # serie del highlight para un 'elements' dado (combo directo, o env(sum))
        def _series(el):
            if rebuild_envelope:
                return story_shear_envelope_from_elements(
                    model, combos_comp, el, exclude_piers, label=highlight_combo)
            return compute_story_shear_from_elements(
                model, [highlight_combo], el, exclude_piers)

        # referencia Story Forces (solo para la detección de transición)
        sf_ref = None
        if transition_tol is not None:
            sf_src = combos_comp if rebuild_envelope else [highlight_combo]
            try:
                sfb = compute_story_force_bounds(model, sf_src)
                sf_ref = sfb.groupby('Story', as_index=False, observed=True)[['VX', 'VY']].max()
            except Exception:
                sf_ref = None

        # pisos de transición a omitir (el pier reporta transferencia local, no corte de piso)
        mask = set()
        if transition_tol is not None and sf_ref is not None:
            cmp = _series('all').merge(sf_ref, on='Story', suffixes=('', '_sf'))
            # compare the combined shear magnitude (robust to the seismic
            # direction: for SpecY the dominant shear is VY, not VX)
            all_mag = np.hypot(cmp['VX'].astype(float), cmp['VY'].astype(float))
            sf_mag = np.hypot(cmp['VX_sf'].astype(float), cmp['VY_sf'].astype(float))
            dev = (all_mag - sf_mag).abs() / sf_mag.replace(0, np.nan)
            mask = set(cmp.loc[dev > transition_tol, 'Story'].astype(str))
            if mask:
                print(f"[aviso] Pisos de transicion (|all-SF|/SF > {transition_tol:.0%}, "
                      f"omitidos del trazo): {sorted(mask)}")

        def _mask(df):
            return df[~df['Story'].astype(str).isin(mask)] if mask else df

        if elements == 'all':
            _draw_story_shear_steps(ax, _mask(_series('columns')),
                                    altura_dict, altura_pisos, 'black', 'black', 1.5, '-', 0.9)
            _draw_story_shear_steps(ax, _mask(_series('walls')),
                                    altura_dict, altura_pisos, 'black', 'black', 1.5, '--', 0.9)
            _draw_story_shear_steps(ax, _mask(_series('all')),
                                    altura_dict, altura_pisos, 'tab:blue', 'tab:red', 2.5, '-', 1.0)
        else:
            _draw_story_shear_steps(ax, _mask(_series(elements)),
                                    altura_dict, altura_pisos, 'tab:blue', 'tab:red', 2.5, '-', 1.0)

    # === Formato ===
    ax.set_xlabel('Shear Force [Tonf]', fontweight='bold')
    ax.set_ylabel('Height [m]', fontweight='bold')
    ax.set_title(f'Story Shear from {elements} - Model: {model.name}', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

    # === Etiquetas de pisos al lado derecho ===
    story_names = model.story_definitions['Story'].tolist()
    story_heights = model.story_definitions['Accumulated_Height'].tolist()

    ax_twin = ax.twinx()
    ax_twin.set_ylim(ax.get_ylim())
    ax_twin.set_yticks(story_heights)
    ax_twin.set_yticklabels(story_names, fontsize=6)
    ax_twin.set_ylabel("Story")

    fig.text(0.99, -0.01, '© 2025 - Patricio Palacios B. - Torrefuerte',
             ha='right', va='bottom', fontsize=9, color='gray', style='italic')

    # === Leyenda manual ===
    from matplotlib.lines import Line2D

    if elements == 'all':
        legend_elements = [
            Line2D([0], [0], color=color, lw=1.0, linestyle='-', label='All Rlz'),
            Line2D([0], [0], color='tab:blue', lw=2.5, linestyle='-', label='Total - VX'),
            Line2D([0], [0], color='tab:red', lw=2.5, linestyle='-', label='Total - VY'),
            Line2D([0], [0], color='black', lw=1.5, linestyle='-', label='Columns'),
            Line2D([0], [0], color='black', lw=1.5, linestyle='--', label='Walls'),
        ]
    else:
        legend_elements = [
            Line2D([0], [0], color=color, lw=1.0, linestyle='-', label='All Rlz'),
            Line2D([0], [0], color='tab:blue', lw=2.5, linestyle='-', label='Highlight - VX'),
            Line2D([0], [0], color='tab:red', lw=2.5, linestyle='-', label='Highlight - VY'),
        ]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()




def plot_story_max_over_avg_ratios(model, combos_comp, show=True, save_path=None):
    """
    Grafica el ratio Max Drift / Avg Drift por Story para múltiples combinaciones.
    Dirección Y se muestra a la izquierda (negativo) y dirección X a la derecha (positivo).
    """

    # === Obtener datos desde tools
    stories, alturas, data_dict = story_max_over_avg_ratios(model, combos_comp)

    altura_dict = model.story_definitions.set_index('Story')['Accumulated_Height'].to_dict()

    # === Posiciones para eje Y
    y_pos = np.arange(len(stories))

    # === Crear figura
    fig, ax = plt.subplots(figsize=(9, 6))
    bar_width = 0.8 / len(combos_comp)
    offsets = np.linspace(-bar_width * (len(combos_comp)-1)/2, bar_width * (len(combos_comp)-1)/2, len(combos_comp))

    legend_handles = []

    for i, combo in enumerate(combos_comp):
        valores_X = data_dict[combo]['X']
        valores_Y = [-v if not np.isnan(v) else np.nan for v in data_dict[combo]['Y']]  # Dirección Y negativa

        # Plot X positivo
        bar_X = ax.barh(y_pos + offsets[i], valores_X, height=bar_width, label=combo)
        # Plot Y negativo con mismo color
        ax.barh(y_pos + offsets[i], valores_Y, height=bar_width, color=bar_X[0].get_facecolor())

        legend_handles.append(bar_X[0])

    # === Ejes
    ax.set_xlim(-2, 2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{altura_dict[s]:.2f}" for s in stories], fontsize=10)
    ax.set_ylabel('Height [m]', fontsize=12, fontweight='bold')

    ax_twin = ax.twinx()
    ax_twin.set_ylim(ax.get_ylim())
    ax_twin.set_yticks(y_pos)
    ax_twin.set_yticklabels(stories, fontsize=9)
    ax_twin.set_ylabel('Story', fontsize=11)

    # === Estética
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Max Drift / Avg Drift (Ratio)', fontsize=12, fontweight='bold')
    ax.set_title(f'Max Drift / Avg Drift (Ratio) - Model: {model.name}', fontweight='bold')
    ax.tick_params(axis='x', labelsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    # === Leyenda
    fig.legend(handles=legend_handles, labels=combos_comp,
               loc='lower center', bbox_to_anchor=(0.5, -0.01),
               ncol=4, fontsize=9, title='Combinación', title_fontsize=10, frameon=False)

    # === Firma inferior
    fig.text(0.99, -0.01, '© 2025 - Patricio Palacios B. - Torrefuerte',
             ha='right', va='bottom', fontsize=9, color='gray', style='italic')

    fig.subplots_adjust(bottom=0.2)

    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()


# ==============================================================
# Story demand from vertical elements (one demand, its Max / Min)
# ==============================================================

# demandas disponibles en las tablas de fuerzas
_DEMAND_COLS = ('P', 'V2', 'V3', 'T', 'M2', 'M3')


def _story_demand_columns(model, combo, demand):
    """Per-story Max/Min of one ``demand`` summed over the columns at their I-node.

    Sums ``demand`` over the columns that share a story (per ``StepType``) and
    keeps the story Max and Min of that sum. Returns ``['Story', 'vmax', 'vmin']``
    for one ``combo``. Values as reported by ETABS (no rotation).
    """
    df = model.element_forces_columns.copy()
    df['OutputCase'] = df['OutputCase'].astype(str)
    df['Station'] = pd.to_numeric(df['Station'], errors='coerce')
    df[demand] = pd.to_numeric(df[demand], errors='coerce')
    df = df[df['OutputCase'] == str(combo)]
    if df.empty:
        return pd.DataFrame(columns=['Story', 'vmax', 'vmin'])

    # nodo i (Station minima) de cada columna por step, sumar las columnas del piso
    inode = df.loc[df.groupby(['UniqueName', 'StepType'], dropna=False)['Station'].idxmin()]
    signed = inode.groupby(['Story', 'StepType'], dropna=False)[demand].sum().reset_index()
    return signed.groupby('Story')[demand].agg(vmax='max', vmin='min').reset_index()


def _story_demand_piers(model, combo, demand, exclude_piers=None):
    """Per-story Max/Min of one ``demand`` summed over the piers at their Bottom.

    Reads 'Pier Forces' at ``Location == 'Bottom'`` and sums ``demand`` over the
    piers that share a story (per ``StepType``), keeping the story Max and Min.
    ``V2``/``V3`` are used as reported by ETABS (local axes, NOT rotated).
    Returns ``['Story', 'vmax', 'vmin']``.
    """
    try:
        pf = model._get_table_as_dataframe('Pier Forces')
    except Exception:
        return pd.DataFrame(columns=['Story', 'vmax', 'vmin'])
    if pf is None or pf.empty or 'Location' not in pf.columns:
        return pd.DataFrame(columns=['Story', 'vmax', 'vmin'])
    pf = pf.copy()
    pf['OutputCase'] = pf['OutputCase'].astype(str)
    pf[demand] = pd.to_numeric(pf[demand], errors='coerce')
    pf = pf[(pf['OutputCase'] == str(combo)) & (pf['Location'] == 'Bottom')]
    if exclude_piers:
        pf = pf[~pf['Pier'].astype(str).str.strip().isin(set(exclude_piers))]
    if pf.empty:
        return pd.DataFrame(columns=['Story', 'vmax', 'vmin'])

    signed = pf.groupby(['Story', 'StepType'], dropna=False)[demand].sum().reset_index()
    return signed.groupby('Story')[demand].agg(vmax='max', vmin='min').reset_index()


def _story_demand_from_elements(model, combo, demand, elements):
    """Dispatcher: per-story Max/Min of ``demand`` for one ``combo``."""
    if elements == 'columns':
        return _story_demand_columns(model, combo, demand)
    if elements == 'walls':
        return _story_demand_piers(model, combo, demand)
    raise NotImplementedError(
        f"elements={elements!r} aun no implementado (columns / walls)")


def _draw_minmax_steps(ax, df, altura_dict, altura_pisos, color_max, color_min, lw, linestyle='-', alpha=1.0):
    """Draw the Max and the Min of a demand as constant steps over each story band."""
    df = df.copy()
    df['_h'] = df['Story'].map(altura_dict)
    df = df.dropna(subset=['_h']).sort_values('_h', ascending=False)  # tope -> base

    vmax_vals, vmin_vals, h_infs = [], [], []
    for _, row in df.iterrows():
        piso = row['Story']
        h_sup = altura_dict[piso]
        h_inf = h_sup - altura_pisos[piso]
        ax.plot([row['vmax'], row['vmax']], [h_inf, h_sup], color=color_max, linestyle=linestyle, linewidth=lw, alpha=alpha)
        ax.plot([row['vmin'], row['vmin']], [h_inf, h_sup], color=color_min, linestyle=linestyle, linewidth=lw, alpha=alpha)
        vmax_vals.append(row['vmax'])
        vmin_vals.append(row['vmin'])
        h_infs.append(h_inf)

    for j in range(len(h_infs) - 1):
        ax.plot([vmax_vals[j], vmax_vals[j + 1]], [h_infs[j], h_infs[j]], color=color_max, linestyle=linestyle, linewidth=lw, alpha=alpha)
        ax.plot([vmin_vals[j], vmin_vals[j + 1]], [h_infs[j], h_infs[j]], color=color_min, linestyle=linestyle, linewidth=lw, alpha=alpha)


def plot_story_bounds_elements(model, combos_comp, color=[0.7, 0.7, 0.7], lw=1.0,
                               highlight_combo=None, demand='V2', elements='columns'):
    """Story profile of ONE demand: its Max and its Min, from the vertical elements.

    Plots a single ``demand`` (e.g. ``V2``): the story **Max** and the story
    **Min** of that demand, as constant steps over each story band. Combos in
    grey, highlight thick (Max ``tab:blue``, Min ``tab:red``). The per-story
    value is the sum of ``demand`` over the elements of the story (columns at
    their I-node / piers at their Bottom). Values as reported by ETABS
    (``V2``/``V3`` local axes, no rotation).

    - ``elements`` : ``'columns'`` or ``'walls'``.
    - ``demand``   : one of ``P, V2, V3, T, M2, M3``.
    """
    if demand not in _DEMAND_COLS:
        raise ValueError(f"demand must be one of {_DEMAND_COLS}, got {demand!r}")

    # === Alturas de los pisos ===
    altura_dict = model.story_definitions.set_index('Story')['Accumulated_Height'].to_dict()
    altura_pisos = model.story_definitions.set_index('Story')['Height'].to_dict()

    # === Crear figura ===
    fig, ax = plt.subplots(figsize=(9, 6))

    # === Combos comunes (gris): Max y Min de la demanda ===
    for combo in combos_comp:
        if highlight_combo and combo == highlight_combo:
            continue  # se grafica despues
        df_plot = _story_demand_from_elements(model, combo, demand, elements)
        _draw_minmax_steps(ax, df_plot, altura_dict, altura_pisos, color, color, lw, '-', 0.8)

    # === Combo destacado: Max tab:blue, Min tab:red ===
    if highlight_combo:
        df_high = _story_demand_from_elements(model, highlight_combo, demand, elements)
        if df_high.empty:
            # el highlight no es un combo directo -> envolvente sobre combos_comp
            parts = [_story_demand_from_elements(model, c, demand, elements) for c in combos_comp]
            parts = [p for p in parts if not p.empty]
            df_high = (pd.concat(parts, ignore_index=True)
                       .groupby('Story', as_index=False).agg(vmax=('vmax', 'max'), vmin=('vmin', 'min'))
                       if parts else pd.DataFrame(columns=['Story', 'vmax', 'vmin']))
        _draw_minmax_steps(ax, df_high, altura_dict, altura_pisos, 'tab:blue', 'tab:red', 2.5, '-', 1.0)

    # === Formato ===
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel(f'{demand} from {elements}', fontweight='bold')
    ax.set_ylabel('Height [m]', fontweight='bold')
    ax.set_title(f'Story {demand} from {elements} - Model: {model.name}', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

    # === Etiquetas de pisos al lado derecho ===
    story_names = model.story_definitions['Story'].tolist()
    story_heights = model.story_definitions['Accumulated_Height'].tolist()

    ax_twin = ax.twinx()
    ax_twin.set_ylim(ax.get_ylim())
    ax_twin.set_yticks(story_heights)
    ax_twin.set_yticklabels(story_names, fontsize=6)
    ax_twin.set_ylabel("Story")

    fig.text(0.99, -0.01, '© 2025 - Patricio Palacios B. - Torrefuerte',
             ha='right', va='bottom', fontsize=9, color='gray', style='italic')

    # === Leyenda manual ===
    from matplotlib.lines import Line2D

    legend_elements = [Line2D([0], [0], color=color, lw=1.0, linestyle='-', label='All Rlz'),
                       Line2D([0], [0], color='tab:blue', lw=2.5, linestyle='-', label=f'{demand} Max'),
                       Line2D([0], [0], color='tab:red', lw=2.5, linestyle='-', label=f'{demand} Min')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()
