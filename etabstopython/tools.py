import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd

def extrude_frame(ax, p1, p2, width, height, color):
    """
    Draws a rectangular prism between two 3D points representing a frame element.

    Parameters:
        ax: Matplotlib 3D axis object
        p1, p2: Endpoints (x, y, z) of the frame element
        width (float): Width of the cross section (in meters)
        height (float): Height of the cross section (in meters)
        color: RGB tuple or color string
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    v = p2 - p1
    v = v / np.linalg.norm(v)

    if np.allclose(v, [0, 0, 1]):
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = np.cross(v, [0.0, 0.0, 1.0])
        u = u / np.linalg.norm(u)

    w = np.cross(v, u)

    u *= width / 2
    w *= height / 2

    corners = []
    for sign_u in [-1, 1]:
        for sign_w in [-1, 1]:
            offset = sign_u * u + sign_w * w
            corners.append(p1 + offset)
    for sign_u in [-1, 1]:
        for sign_w in [-1, 1]:
            offset = sign_u * u + sign_w * w
            corners.append(p2 + offset)

    faces = [
        [corners[0], corners[1], corners[3], corners[2]],
        [corners[4], corners[5], corners[7], corners[6]],
        [corners[0], corners[1], corners[5], corners[4]],
        [corners[2], corners[3], corners[7], corners[6]],
        [corners[1], corners[3], corners[7], corners[5]],
        [corners[0], corners[2], corners[6], corners[4]],
    ]

    ax.add_collection3d(Poly3DCollection(faces, color=color, alpha=0.9, edgecolor='k'))

# ==============================================================
# Drift and Displacement Calculation Tools
# ==============================================================

# def derivas_combos(model, combo, Cd, Ie, R):
#     df = model.joint_displacements
#     heights = model.floor_heights

#     df = df[(df['OutputCase'] == combo) & (df['StepType'] == 'Max')].copy()
#     df[['Ux', 'Uy']] = df[['Ux', 'Uy']].apply(pd.to_numeric, errors='coerce')
#     max_disp = df.groupby('Story')[['Ux', 'Uy']].max().reset_index()

#     orden_original = df['Story'].unique()
#     max_disp['Story'] = pd.Categorical(max_disp['Story'], categories=orden_original, ordered=True)
#     max_disp = max_disp.sort_values('Story').reset_index(drop=True)

#     dx = max_disp['Ux'][::-1].values
#     dy = max_disp['Uy'][::-1].values
    
#     drift_x, drift_y = [], []
#     for i in range(1, len(heights)):
#         h = heights[i] - heights[i - 1]
#         drift_x.append(abs((dx[i] - dx[i - 1]) / h * Cd / Ie * 100 / R))
#         drift_y.append(abs((dy[i] - dy[i - 1]) / h * Cd / Ie * 100 / R))

#     drift_x = np.insert(drift_x, 0, 0)
#     drift_y = np.insert(drift_y, 0, 0)

#     return dx, dy, drift_x, drift_y


def compute_story_displacement_bounds(model, combos_comp , factor=1.0):

    # Copy displacements and ensure numerical
    df_disp = model.joint_displacements.copy()
    for col in ['Ux', 'Uy']:
        df_disp[col] = pd.to_numeric(df_disp[col], errors='coerce')

    df_disp['Ux'] *= factor
    df_disp['Uy'] *= factor

    # Filter desired combinations
    df_filt = df_disp[df_disp['OutputCase'].isin(combos_comp)].copy()

    # Get min and max per node
    df_min = df_filt.groupby(['OutputCase', 'UniqueName'])[['Ux', 'Uy']].min().reset_index()
    df_max = df_filt.groupby(['OutputCase', 'UniqueName'])[['Ux', 'Uy']].max().reset_index()

    df_min = df_min.rename(columns={'Ux': 'Ux_min', 'Uy': 'Uy_min'})
    df_max = df_max.rename(columns={'Ux': 'Ux_max', 'Uy': 'Uy_max'})

    df_bounds = pd.merge(df_min, df_max, on=['OutputCase', 'UniqueName'])

    # Add story info
    df_info = df_filt[['OutputCase', 'UniqueName', 'Story']].drop_duplicates()
    df_bounds = pd.merge(df_bounds, df_info, on=['OutputCase', 'UniqueName'], how='left')

    # Group by story to get story-level max/min
    df_plot = df_bounds.groupby(['OutputCase', 'Story']).agg({
        'Ux_min': 'min', 'Ux_max': 'max',
        'Uy_min': 'min', 'Uy_max': 'max'
    }).reset_index()

    # Map to accumulated height
    story_map = model.story_definitions.set_index('Story')['Accumulated_Height'].to_dict()
    df_plot['Height'] = df_plot['Story'].map(story_map)

    # Sort by height
    df_plot = df_plot.sort_values(by='Height')

    return df_plot

def compute_story_force_bounds(model, combos_comp):
   
    df = model.story_forces.copy()

    # Asegurar que sean numéricos
    cols = ['P', 'VX', 'VY', 'MX', 'MY']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

    # Filtrar combinaciones
    df = df[df['OutputCase'].isin(combos_comp)]

    # Agrupar por Story y OutputCase tomando el máximo absoluto
    resumen = df.groupby(['Story', 'OutputCase'])[cols].agg(lambda x: x.abs().max()).reset_index()

    # Ordenar pisos según el modelo
    orden_pisos = model.story_definitions['Story'].tolist()
    resumen['Story'] = pd.Categorical(resumen['Story'], categories=orden_pisos, ordered=True)
    resumen.sort_values(['OutputCase', 'Story'], inplace=True)

    return resumen
