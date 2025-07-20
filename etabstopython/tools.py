import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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
