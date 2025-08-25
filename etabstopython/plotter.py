import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from etabstopython.tools import extrude_frame
from etabstopython.tools import compute_story_displacement_bounds
from etabstopython.tools import compute_story_force_bounds


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
    if hasattr(model, 'floor_points_by_story'):
        poc = model.point_object_connectivity
        for puntos in model.floor_points_by_story:
            coordenadas = []
            for punto in puntos:
                fila = poc[poc['UniqueName'] == str(int(punto))]
                if not fila.empty:
                    x, y, z = fila.iloc[0][['X', 'Y', 'Z']]
                    coordenadas.append([x, y, z])
            if len(coordenadas) >= 3:
                coords_array = np.array(coordenadas, dtype=float)
                verts = [list(zip(coords_array[:, 0], coords_array[:, 1], coords_array[:, 2]))]
                ax.add_collection3d(Poly3DCollection(verts, alpha=0.4, edgecolor='k', facecolor='gray'))

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
    df_plot = compute_story_displacement_bounds(model, combos_comp, factor)
    df_plot_highlight = None
    if highlight_combo and highlight_combo not in combos_comp:
        df_plot_highlight = compute_story_displacement_bounds(model, [highlight_combo], factor)

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
        dx = data['Ux_max'].values
        dy = data['Uy_max'].values

        drift_x = compute_drift(height, dx)
        drift_y = compute_drift(height, dy)

        ax1.plot(drift_x, height, 'o-', color=color, linewidth=lw, alpha=0.9, label=combo)
        ax2.plot(drift_y, height, 'o-', color=color, linewidth=lw, alpha=0.9, label=combo)

    # === Combo destacado ===
    if highlight_combo:
        if highlight_combo in combos_comp:
            data_h = df_plot[df_plot['OutputCase'] == highlight_combo]
        else:
            data_h = df_plot_highlight[df_plot_highlight['OutputCase'] == highlight_combo]
        data_h = data_h.sort_values(by='Height')

        height = data_h['Height'].values
        dx = data_h['Ux_max'].values
        dy = data_h['Uy_max'].values

        drift_x = compute_drift(height, dx)
        drift_y = compute_drift(height, dy)

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
        resumen_high = compute_story_force_bounds(model, [highlight_combo])
        df_high = resumen_high[resumen_high['OutputCase'] == highlight_combo].copy()
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
