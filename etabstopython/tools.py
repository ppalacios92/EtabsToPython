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

def derivas_combos(model, combo, Cd, Ie, R):
    df = model.joint_displacements
    heights = model.floor_heights

    df = df[(df['OutputCase'] == combo) & (df['StepType'] == 'Max')].copy()
    df[['Ux', 'Uy']] = df[['Ux', 'Uy']].apply(pd.to_numeric, errors='coerce')
    max_disp = df.groupby('Story')[['Ux', 'Uy']].max().reset_index()

    orden_original = df['Story'].unique()
    max_disp['Story'] = pd.Categorical(max_disp['Story'], categories=orden_original, ordered=True)
    max_disp = max_disp.sort_values('Story').reset_index(drop=True)

    dx = max_disp['Ux'][::-1].values
    dy = max_disp['Uy'][::-1].values
    
    drift_x, drift_y = [], []
    for i in range(1, len(heights)):
        h = heights[i] - heights[i - 1]
        drift_x.append(abs((dx[i] - dx[i - 1]) / h * Cd / Ie * 100 / R))
        drift_y.append(abs((dy[i] - dy[i - 1]) / h * Cd / Ie * 100 / R))

    drift_x = np.insert(drift_x, 0, 0)
    drift_y = np.insert(drift_y, 0, 0)

    return dx, dy, drift_x, drift_y
