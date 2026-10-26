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

    # df_disp['Ux'] *= factor
    # df_disp['Uy'] *= factor

    df_disp['Ux'] = np.array(df_disp['Ux'], dtype=float) * factor
    df_disp['Uy'] = np.array(df_disp['Uy'], dtype=float) * factor

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


def compute_story_drift_bounds(model, combos_comp, factor=1.0):
    """Compute story drift by matching nodes with the same plan coordinates."""
    out_cols = ["OutputCase", "Story", "Height", "DriftX", "DriftY"]

    df = model.joint_displacements.copy()
    df = df[df["OutputCase"].isin(combos_comp)].copy()
    if df.empty:
        return pd.DataFrame(columns=out_cols)

    for col in ["Ux", "Uy"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") * factor

    points = model.point_object_connectivity[["UniqueName", "X", "Y"]].copy()
    points["X"] = pd.to_numeric(points["X"], errors="coerce")
    points["Y"] = pd.to_numeric(points["Y"], errors="coerce")
    points["xy_key"] = points["X"].round(6).astype(str) + "|" + points["Y"].round(6).astype(str)

    df = df.merge(points[["UniqueName", "xy_key"]], on="UniqueName", how="left")
    df = df.dropna(subset=["Ux", "Uy", "xy_key"])

    story_def = model.story_definitions[["Story", "Accumulated_Height"]].copy()
    story_def = story_def.rename(columns={"Accumulated_Height": "Height"})
    story_def["Height"] = pd.to_numeric(story_def["Height"], errors="coerce")
    story_def = story_def.sort_values("Height")
    story_order = story_def["Story"].tolist()
    height_map = story_def.set_index("Story")["Height"].to_dict()

    group_cols = ["OutputCase"]
    if "StepType" in df.columns:
        group_cols.append("StepType")
    if "StepNumber" in df.columns:
        group_cols.append("StepNumber")

    rows = []
    for keys, case_df in df.groupby(group_cols, dropna=False, sort=False):
        output_case = keys[0] if isinstance(keys, tuple) else keys
        first_story = story_order[0]
        rows.append({
            "OutputCase": output_case,
            "Story": first_story,
            "Height": height_map[first_story],
            "DriftX": 0.0,
            "DriftY": 0.0,
        })

        for i in range(1, len(story_order)):
            bot_story = story_order[i - 1]
            top_story = story_order[i]
            dz = height_map[top_story] - height_map[bot_story]
            if dz <= 0:
                continue

            bot = case_df[case_df["Story"] == bot_story][["xy_key", "Ux", "Uy"]]
            top = case_df[case_df["Story"] == top_story][["xy_key", "Ux", "Uy"]]
            if bot.empty or top.empty:
                continue

            bot = bot.groupby("xy_key", as_index=False)[["Ux", "Uy"]].mean()
            top = top.groupby("xy_key", as_index=False)[["Ux", "Uy"]].mean()
            pairs = top.merge(bot, on="xy_key", suffixes=("_top", "_bot"))
            if pairs.empty:
                continue

            drift_x = (pairs["Ux_top"] - pairs["Ux_bot"]).abs().max() / dz * 100.0
            drift_y = (pairs["Uy_top"] - pairs["Uy_bot"]).abs().max() / dz * 100.0
            rows.append({
                "OutputCase": output_case,
                "Story": top_story,
                "Height": height_map[top_story],
                "DriftX": drift_x,
                "DriftY": drift_y,
            })

    if not rows:
        return pd.DataFrame(columns=out_cols)

    result = pd.DataFrame(rows)
    result = (
        result.groupby(["OutputCase", "Story", "Height"], as_index=False)[["DriftX", "DriftY"]]
        .max()
        .sort_values(["OutputCase", "Height"])
    )
    return result[out_cols]

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


_SIGNED_COLS = ['OutputCase', 'Story', 'StepType', 'VXs', 'VYs']


def _columns_signed(model, combos_comp):
    """Signed story-shear contribution of columns, per (combo, story, StepType).

    Sums 'Element Forces - Columns' at the I-node (minimum station = bottom).
    Local column shears map to global axes as ``V2 -> VX`` and ``V3 -> VY``
    (default column local axes, angle 0). ``VXs``/``VYs`` keep the sign so the
    'all' mode can add columns and piers coherently.
    """
    df = getattr(model, 'element_forces_columns', None)
    if df is None or df.empty:
        return pd.DataFrame(columns=_SIGNED_COLS)
    df = df.copy()

    name_col = 'UniqueName' if 'UniqueName' in df.columns else 'Column'
    for col in ['Station', 'V2', 'V3']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[df['OutputCase'].isin(combos_comp)]
    if df.empty:
        return pd.DataFrame(columns=_SIGNED_COLS)

    # I-node = minimum station for each element / load step. dropna=False is
    # required: linear design combos have StepType/StepNumber = None, and a
    # groupby drops None keys by default, which would make the transform return
    # NaN and leave the I-node filter empty.
    step_cols = [c for c in ['StepType', 'StepNumber'] if c in df.columns]
    df['Station'] = df['Station'].round(6)
    df['_min_station'] = df.groupby(['OutputCase', name_col] + step_cols,
                                    dropna=False)['Station'].transform('min')
    inode = df[np.isclose(df['Station'], df['_min_station'])]

    signed = inode.groupby(['OutputCase', 'Story', 'StepType'], dropna=False,
                           as_index=False)[['V2', 'V3']].sum()
    return signed.rename(columns={'V2': 'VXs', 'V3': 'VYs'})[_SIGNED_COLS]


def _piers_signed(model, combos_comp, exclude_piers=None):
    """Signed story-shear contribution of piers/walls, per (combo, story, StepType).

    Reads 'Pier Forces' at ``Location == 'Bottom'`` (the base of the pier at
    each story, analogous to the column I-node) and projects the local pier
    shears ``V2``/``V3`` onto global ``X``/``Y`` using the pier ``AxisAngle``
    from 'Pier Section Properties' (falls back to 0 if unavailable).

    ``exclude_piers`` drops piers by name. This is the guard against double
    counting when a model defines an *aggregate* pier (e.g. ``P_Total``) that
    shares a story with its components (``w1..w5``): summing both would count
    the wall shear twice. Such coexistence is auto-detected and warned about.
    """
    try:
        pf = model._get_table_as_dataframe('Pier Forces')
    except Exception:
        return pd.DataFrame(columns=_SIGNED_COLS)
    if pf is None or pf.empty or 'Location' not in pf.columns:
        return pd.DataFrame(columns=_SIGNED_COLS)
    pf = pf.copy()

    for col in ['V2', 'V3']:
        pf[col] = pd.to_numeric(pf[col], errors='coerce')
    pf = pf[(pf['OutputCase'].isin(combos_comp)) & (pf['Location'] == 'Bottom')]
    if pf.empty:
        return pd.DataFrame(columns=_SIGNED_COLS)

    if exclude_piers:
        pf = pf[~pf['Pier'].astype(str).str.strip().isin(set(exclude_piers))]
        if pf.empty:
            return pd.DataFrame(columns=_SIGNED_COLS)

    # double-counting guard: warn if a pier looks like the aggregate of the
    # others in the same (combo, story, step) -- |V2_p| ~ |sum of the others|.
    grp = pf.groupby(['OutputCase', 'Story', 'StepType'], dropna=False)
    others = grp['V2'].transform('sum') - pf['V2']
    npiers = grp['Pier'].transform('nunique')
    susp = pf[(npiers >= 2) & (pf['V2'].abs() > 1.0)
              & ((pf['V2'].abs() - others.abs()).abs() <= 0.08 * pf['V2'].abs())]
    if not susp.empty:
        pairs = sorted({(str(s), str(p)) for s, p in zip(susp['Story'], susp['Pier'])})
        print(f"[aviso] Posible pier agregado (riesgo de doble conteo) en {len(pairs)} "
              f"piso/pier: {pairs[:6]}{' ...' if len(pairs) > 6 else ''}. "
              f"Si corresponde, pasa exclude_piers=[...].")

    # local V2/V3 -> global X/Y via the pier axis angle
    angle = pd.Series(0.0, index=pf.index)
    try:
        ps = model._get_table_as_dataframe('Pier Section Properties')
        if ps is not None and {'Story', 'Pier', 'AxisAngle'}.issubset(ps.columns):
            ps = ps[['Story', 'Pier', 'AxisAngle']].drop_duplicates(['Story', 'Pier']).copy()
            ps['AxisAngle'] = pd.to_numeric(ps['AxisAngle'], errors='coerce')
            pf = pf.merge(ps, on=['Story', 'Pier'], how='left')
            angle = pf['AxisAngle'].fillna(0.0)
    except Exception:
        pass

    th = np.radians(angle)
    pf['VXs'] = pf['V2'] * np.cos(th) - pf['V3'] * np.sin(th)
    pf['VYs'] = pf['V2'] * np.sin(th) + pf['V3'] * np.cos(th)
    signed = pf.groupby(['OutputCase', 'Story', 'StepType'], dropna=False,
                        as_index=False)[['VXs', 'VYs']].sum()
    return signed[_SIGNED_COLS]


def compute_story_shear_from_elements(model, combos_comp, elements="columns",
                                      exclude_piers=None):
    """Story shear built by summing the shear of the vertical elements per story.

    Unlike :func:`compute_story_force_bounds` (which reads the 'Story Forces'
    table), this adds up the shear carried by the vertical elements at their
    base (column I-node / pier ``Bottom``) and groups it by story.

    Validated against 'Story Forces' on a real model (response-spectrum case):
    in the regular height band the agreement is within ~1-2 %. It can diverge at
    transition floors (the pier reports local transfer, not story shear) and at
    the base; use :func:`compute_story_force_bounds` as the reference there.

    Parameters
    ----------
    model : EtabsModel
        Live model exposing ``element_forces_columns`` and ``story_definitions``
        (and, for piers, the 'Pier Forces' / 'Pier Section Properties' tables via
        the ETABS COM API ``GetTableForDisplayArray``).
    combos_comp : list[str]
        Output cases to keep.
    elements : {'columns', 'walls', 'all'}
        ``'columns'`` sums frame columns, ``'walls'`` sums piers, ``'all'`` adds
        both. Contributions are summed with sign per (combo, story, step), so for
        a linear combo ``all`` equals the true story shear; for an envelope it is
        conservative (the per-source Max/Min peaks occur in different base cases).
    exclude_piers : list[str], optional
        Pier names to drop (avoid double counting an aggregate pier that
        coexists with its components). Passed to :func:`_piers_signed`.

    Returns
    -------
    pandas.DataFrame
        Columns ``['OutputCase', 'Story', 'VX', 'VY']`` (magnitudes), story
        ordered like the model.
    """
    out_cols = ['OutputCase', 'Story', 'VX', 'VY']
    valid = {'columns', 'walls', 'all'}
    if elements not in valid:
        raise ValueError(f"elements must be one of {sorted(valid)}, got {elements!r}")

    parts = []
    if elements in ('columns', 'all'):
        parts.append(_columns_signed(model, combos_comp))
    if elements in ('walls', 'all'):
        parts.append(_piers_signed(model, combos_comp, exclude_piers=exclude_piers))
    parts = [p for p in parts if not p.empty]

    if not parts:
        try:
            available = sorted(model.element_forces_columns['OutputCase'].dropna().unique())
        except Exception:
            available = []
        print(f"[aviso] No hay aporte de '{elements}' para los combos pedidos.")
        print(f"        Combos disponibles (Element Forces - Columns): {available}")
        return pd.DataFrame(columns=out_cols)

    signed = pd.concat(parts, ignore_index=True)
    # add up the contributions of the requested sources per (combo, story, step)
    signed = signed.groupby(['OutputCase', 'Story', 'StepType'], dropna=False,
                            as_index=False)[['VXs', 'VYs']].sum()

    # collapse load steps (e.g. envelope Max/Min) to the governing magnitude
    signed['VX'] = signed['VXs'].abs()
    signed['VY'] = signed['VYs'].abs()
    resumen = signed.groupby(['OutputCase', 'Story'], as_index=False)[['VX', 'VY']].max()

    # order stories like the model
    orden_pisos = model.story_definitions['Story'].tolist()
    resumen['Story'] = pd.Categorical(resumen['Story'], categories=orden_pisos, ordered=True)
    resumen.sort_values(['OutputCase', 'Story'], inplace=True)

    return resumen[out_cols]


def story_shear_envelope_from_elements(model, combos_comp, elements="columns",
                                       exclude_piers=None, label="Envelope"):
    """Envelope of the element-summed story shear over individual combos.

    This is the *correct* design envelope: it sums the elements **within each
    combo** and then takes the per-story maximum over combos -- ``env(sum)``. It
    matches the 'Story Forces' envelope, unlike feeding a pre-enveloped case
    (e.g. ETABS 'EnveDiseño'), whose per-element Max/Min already lost the base
    combos and would give the inflated ``sum(env)``.

    Returns a DataFrame ``['OutputCase', 'Story', 'VX', 'VY']`` with a single
    ``OutputCase == label``.
    """
    out_cols = ['OutputCase', 'Story', 'VX', 'VY']
    per = compute_story_shear_from_elements(model, combos_comp, elements,
                                            exclude_piers=exclude_piers)
    if per.empty:
        return pd.DataFrame(columns=out_cols)
    env = per.groupby('Story', as_index=False, observed=True)[['VX', 'VY']].max()
    env['OutputCase'] = label

    orden_pisos = model.story_definitions['Story'].tolist()
    env['Story'] = pd.Categorical(env['Story'], categories=orden_pisos, ordered=True)
    env.sort_values('Story', inplace=True)
    return env[out_cols]


def story_max_over_avg_ratios(model, combos_comp):
    """
    Extrae y organiza datos del ratio Max Drift / Avg Drift por Story, para cada combinación.
    
    Retorna:
    - stories: lista ordenada de nombres de pisos
    - alturas: lista de alturas acumuladas
    - data_dict: diccionario {combo: {'X': [...], 'Y': [...]}} con listas alineadas a `stories`
    """

    # === Leer tabla desde ETABS ===
    df_ratios = model._get_table_as_dataframe('Story Max Over Avg Drifts').copy()
    df_ratios = df_ratios[df_ratios['OutputCase'].isin(combos_comp)]
    df_ratios['Ratio'] = pd.to_numeric(df_ratios['Ratio'], errors='coerce')

    # === Mapear alturas desde definición de pisos ===
    story_def = model.story_definitions
    altura_dict = story_def.set_index('Story')['Accumulated_Height'].to_dict()
    df_ratios['Height'] = df_ratios['Story'].map(altura_dict)
    df_ratios = df_ratios.dropna(subset=['Height', 'Ratio']).sort_values(by='Height')

    # === Lista de pisos ordenada por altura
    stories = sorted(df_ratios['Story'].unique(), key=lambda s: altura_dict.get(s, 0))
    alturas = [altura_dict[s] for s in stories]

    # === Extraer valores por combo y dirección
    data_dict = {}
    for combo in combos_comp:
        ratio_X = df_ratios[(df_ratios['OutputCase'] == combo) & (df_ratios['Direction'] == 'X')].groupby('Story')['Ratio'].mean()
        ratio_Y = df_ratios[(df_ratios['OutputCase'] == combo) & (df_ratios['Direction'] == 'Y')].groupby('Story')['Ratio'].mean()
        valores_X = [ratio_X.get(s, np.nan) for s in stories]
        valores_Y = [ratio_Y.get(s, np.nan) for s in stories]
        data_dict[combo] = {'X': valores_X, 'Y': valores_Y}

    return stories, alturas, data_dict

