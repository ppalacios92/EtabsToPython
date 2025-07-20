import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from etabstopython.tools import extrude_frame


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
    all_x = np.concatenate([model.frames_df['PointIX'], model.frames_df['PointJX']]).astype(float)
    all_y = np.concatenate([model.frames_df['PointIY'], model.frames_df['PointJY']]).astype(float)
    all_z = np.concatenate([model.frames_df['PointIZ'], model.frames_df['PointJZ']]).astype(float)

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

    plt.show()
