import comtypes.client
import numpy as np
import pandas as pd


class EtabsModel:
    def __init__(self, units: int = 12 , 
                 name: str = 'PRY00',
                 
                 ):
        self.units = units
        self.name=  name    


        self.model = None

# --------------------------------------------------------------
        
            
        # --- Conexión y unidades ---
        self._connect_to_etabs()
        self._set_units()

        # --- Tablas base ---
        self.load_story_definitions()
        self.load_point_object_connectivity()

        # --- Propiedades de secciones ---
        self.load_frame_section_property_definitions_summary()
        self.load_frame_section_property_definitions_concrete_rectangular()

        # --- Asignaciones y geometría ---
        self.load_frame_assignments_section_properties()
        self.load_beam_object_connectivity()
        self.load_column_object_connectivity()
        self.load_brace_object_connectivity()
        self.build_linear_elements_dataframe()

        # --- Shells y planos ---
        self.load_wall_object_connectivity()
        self.load_floor_object_connectivity()

        # --- Gráficos y colores ---
        self.build_section_color_dict()


        # --- Cargar fuerzas de elementos ---
        self.load_element_forces_columns()

        # --- Cargar masas y fuerzas ---
        self.load_modal_participating_mass_ratios()

        # --- Cargar story forces ---
                
        self.load_story_forces()
        

        # --- Cargar joint displacements ---
        self.load_joint_displacements()
        self.joint_reactions()


    def _connect_to_etabs(self):
        etabs = comtypes.client.GetActiveObject('CSI.ETABS.API.ETABSObject')
        self.model = etabs.SapModel
        print("✅ Connected to ETABS.")

    def _set_units(self):
        self.model.SetPresentUnits(self.units)
        print(f"📏 Units set to {self.units}.")

    def _get_table_as_dataframe(self, table_title: str) -> pd.DataFrame:
        ret = self.model.DatabaseTables.GetTableForDisplayArray(table_title, '', '')

        headers = list(ret[2])
        data = np.array(ret[4])
        data = data.reshape(len(data)//len(headers), len(headers))
        df = pd.DataFrame(data)
        cols_dict = dict(zip(df.columns.tolist(), headers))
        df.rename(cols_dict, axis = 'columns', inplace = True)
        return df


# --------------------------------------------------------------
    def load_story_definitions(self):
        table_title = 'Story Definitions'
        df = self._get_table_as_dataframe(table_title)
        
        # Asegurar que la columna de altura sea numérica
        df['Height'] = pd.to_numeric(df['Height'], errors='coerce')
        
        # Calcular altura acumulada desde la base hacia arriba
        df['Accumulated_Height'] = df.iloc[::-1]['Height'].cumsum()[::-1]
        
        # Verificar si 'BASE' ya existe
        if 'BASE' not in df['Story'].values:
            # Tomar la última fila y duplicarla
            base_row = df.iloc[[-1]].copy()
            base_row['Story'] = 'Base'
            base_row['Height'] = 0.0
            base_row['Accumulated_Height'] = 0.0
            df = pd.concat([df, base_row], ignore_index=True)

        # Guardar el DataFrame completo
        self.story_definitions = df
        # print(f"✅ Loaded Story Definitions: {len(df)} stories.")
        
        # Calcular vector de alturas (agregando 0 al final para tener todas las cotas de niveles)
        heights = df['Accumulated_Height'].to_numpy()
        # heights = np.append(heights, 0)  # piso inferior
        heights = heights[::-1]          # ordenar de base hacia arriba
        heights = np.round(heights, 3)
        
        self.floor_heights = heights

    # @property
    # def story_names_full(self):
    #     return ["BASE"] + self.story_definitions['Story'].tolist()

    def load_point_object_connectivity(self):
        table_title = 'Point Object Connectivity'
        df = self._get_table_as_dataframe(table_title)
        # Obtener alturas únicas y ordenarlas
        alturas_unicas = sorted(df['Z'].astype(float).unique())
        label_dict = {z: f"F{i}" for i, z in enumerate(alturas_unicas)}
        # Asignar etiquetas en función de Z
        df['Z'] = df['Z'].astype(float)
        df['Label'] = df['Z'].map(label_dict)
        self.point_object_connectivity = df
        # print(f"✅ Loaded Point Object Connectivity: {len(df)} connections.")

# --------------------------------------------------------------
    def load_frame_section_property_definitions_summary(self):
        table_title = 'Frame Section Property Definitions - Summary'
        self.frame_section_property_definitions_summary = self._get_table_as_dataframe(table_title)
        # print(f"✅ Loaded Frame Section Property Definitions: {len(self.frame_section_property_definitions_summary)} properties.")



    
    def load_frame_section_property_definitions_concrete_rectangular(self):
        table_title='Frame Section Property Definitions - Concrete Rectangular'
        self.frame_section_property_definitions_concrete_rectangular = self._get_table_as_dataframe(table_title)
        # print(f"✅ Loaded Frame Section Property Definitions - Concrete Rectangular: {len(self.frame_section_property_definitions_concrete_rectangular)} properties.")



    def load_frame_assignments_section_properties(self):
        table_title='Frame Assignments - Section Properties'
        self.frame_assignments_section_properties = self._get_table_as_dataframe(table_title)
        # print(f"✅ Loaded Frame Assignments - Section Properties: {len(self.frame_assignments_section_properties)} properties.")


# --------------------------------------------------------------
    def load_beam_object_connectivity(self):
        table_title = 'Beam Object Connectivity'
        self.beam_object_connectivity = self._get_table_as_dataframe(table_title)
        # print(f"✅ Loaded Beam Object Connectivity: {len(self.beam_object_connectivity)} entries.")

    def load_column_object_connectivity(self):
        table_title = 'Column Object Connectivity'
        self.column_object_connectivity = self._get_table_as_dataframe(table_title)
        # print(f"✅ Loaded Column Object Connectivity: {len(self.column_object_connectivity)} entries.")

    def load_brace_object_connectivity(self):
        table_title = 'Brace Object Connectivity'
        self.brace_object_connectivity = self._get_table_as_dataframe(table_title)
        # print(f"✅ Loaded Brace Object Connectivity: {len(self.brace_object_connectivity)} entries.")

# --------------------------------------------------------------
    def build_linear_elements_dataframe(self):
        cols_requeridas = ['UniqueName', 'Story', 'UniquePtI', 'UniquePtJ']
        frame_parts = []

        # Mapeo: nombre interno → atributo
        tablas = {
            'beam_object_connectivity': self.beam_object_connectivity,
            'column_object_connectivity': self.column_object_connectivity,
            'brace_object_connectivity': self.brace_object_connectivity
        }

        for nombre, tabla in tablas.items():
            if tabla is not None:
                if all(col in tabla.columns for col in cols_requeridas):
                    frame_parts.append(tabla[cols_requeridas].copy())
                else:
                    print(f"⚠️ {nombre} cargada pero le faltan columnas requeridas.")
            else:
                print(f"⚠️ {nombre} no fue cargada.")

        if not frame_parts:
            raise ValueError("No valid frame connectivity tables were found.")

        df = pd.concat(frame_parts, ignore_index=True)

        # Obtener coordenadas
        point_coords = self.point_object_connectivity.set_index('UniqueName')[['X', 'Y', 'Z']].astype(float)
        coords_I = point_coords.loc[df['UniquePtI']].reset_index(drop=True)
        coords_J = point_coords.loc[df['UniquePtJ']].reset_index(drop=True)

        df['PointIX'] = coords_I['X']
        df['PointIY'] = coords_I['Y']
        df['PointIZ'] = coords_I['Z']
        df['PointJX'] = coords_J['X']
        df['PointJY'] = coords_J['Y']
        df['PointJZ'] = coords_J['Z']

        # Agregar propiedades de sección si existen
        if hasattr(self, 'frame_assignments_section_properties'):
            props = self.frame_assignments_section_properties
            if 'SectProp' in props.columns:
                df = df.merge(props[['UniqueName', 'SectProp']], on='UniqueName', how='left')
            else:
                df['SectProp'] = None
        else:
            df['SectProp'] = None

        self.frames_df = df
        # print(f"✅ Combined linear elements: {len(df)} total.")


# --------------------------------------------------------------

    def load_wall_object_connectivity(self):
        table_title = 'Wall Object Connectivity'
        self.wall_object_connectivity = self._get_table_as_dataframe(table_title)
        # print(f"✅ Loaded Wall Object Connectivity: {len(self.wall_object_connectivity)} entries.")

    def load_floor_object_connectivity(self):
        table_title = 'Floor Object Connectivity'
        df = self._get_table_as_dataframe(table_title)
        self.floor_object_connectivity = df

        # Caso 1: DataFrame vacío
        if df.empty:
            print("⚠️ No floor objects found. Creating empty floor_points_by_story.")
            self.floor_points_by_story = np.array([], dtype=object)
            return

        # Caso 2: Falta la columna "Story"
        if 'Story' not in df.columns:
            print(f"⚠️ Column 'Story' not found in {table_title}. Available columns: {list(df.columns)}")
            self.floor_points_by_story = np.array([], dtype=object)
            return

        # Procesar puntos de losas por piso (flujo normal)
        filtered_df = df.dropna(subset=['Story'])
        cambios_story = filtered_df.index[filtered_df['Story'].ne(filtered_df['Story'].shift())].tolist()

        vector_final = []
        for i in range(len(cambios_story) - 1):
            fila_inicio = cambios_story[i]
            fila_fin = cambios_story[i + 1] - 1
            valores = df.loc[fila_inicio:fila_fin, ['UniquePt1', 'UniquePt2', 'UniquePt3', 'UniquePt4']].values.flatten()
            valores_numericos = pd.to_numeric(valores, errors='coerce')
            valores_numericos = valores_numericos[~np.isnan(valores_numericos)]
            vector_final.append(valores_numericos)

        self.floor_points_by_story = np.array(vector_final, dtype=object)

        # print(f"✅ Processed floor point groups by story: {len(self.floor_points_by_story)} levels.")


# --------------------------------------------------------------

    def build_section_color_dict(self):
        colores_vigas_col = [
            (0.7, 0.7, 0.7),
            "#8080FF",
            "#80FF80",
            "#80FF80",
            "#0080FF"
        ]
        secciones = self.frames_df['SectProp'].unique()
        self.section_color_dict = {
            sec: colores_vigas_col[i % len(colores_vigas_col)] for i, sec in enumerate(secciones)
        }

        # Añadir dimensiones por sección
        if hasattr(self, 'frame_section_property_definitions_concrete_rectangular'):
            self.section_dim_dict = (
                self.frame_section_property_definitions_concrete_rectangular
                .set_index('Name')[['t2', 't3']]
                .astype(float)
                .to_dict('index')
            )
            
        # print(f"✅ Assigned colors to {len(self.section_color_dict)} frame sections.")


# --------------------------------------------------------------

    def load_element_forces_columns(self):
        table_title = 'Element Forces - Columns'
        self.element_forces_columns = self._get_table_as_dataframe(table_title)
        # print(f"✅ Loaded Element Forces - Columns: {len(self.element_forces_columns)} entries.")



# --------------------------------------------------------------
    def load_modal_participating_mass_ratios(self):
        table_title = 'Modal Participating Mass Ratios'
        self.modal_participating_mass_ratios = self._get_table_as_dataframe(table_title)
        # print(f"✅ Loaded Modal Participating Mass Ratios: {len(self.modal_participating_mass_ratios)} entries.")


    def load_story_forces(self):
        table_title = 'Story Forces'
        self.story_forces = self._get_table_as_dataframe(table_title)
        self.combos=self.story_forces['OutputCase'].unique()
        # print(f"✅ Loaded Story Forces: {len(self.story_forces)} entries.")
        # print("---"*50)
        # print(f"Unique combos: {self.combos}")
        # print("---"*50)

    def load_joint_displacements(self):
        table_title = 'Joint Displacements'
        self.joint_displacements = self._get_table_as_dataframe(table_title)
        # print(f"✅ Loaded Joint Displacements: {len(self.joint_displacements)} entries.")

    def joint_reactions(self):
        table_title = 'Joint Reactions'
        self.joint_reactions = self._get_table_as_dataframe(table_title)
        # print(f"✅ Loaded Joint Reactions: {len(self.joint_reactions)} entries.")


    def summary(self):
        print("📦 ETABS Model Summary")
        print(f"🧾 Model name: {self.name}")
        print(f"📏 Unit system code: {self.units}")
        print("──────────────────────────────────────────────")
        print(f"🏢 Story definitions loaded: {len(self.story_definitions)}")
        print(f"📐 Floor heights vector: {self.floor_heights.tolist()}")
        print(f"🔗 Point connections loaded: {len(self.point_object_connectivity)}")
        print(f"📦 Frame section summary: {len(self.frame_section_property_definitions_summary)}")
        print(f"🧱 Rectangular concrete sections: {len(self.frame_section_property_definitions_concrete_rectangular)}")
        print(f"🧩 Frame assignments with sections: {len(self.frame_assignments_section_properties)}")
        print(f"🧱 Beams loaded: {len(self.beam_object_connectivity)}")
        print(f"🧱 Columns loaded: {len(self.column_object_connectivity)}")
        print(f"🧱 Braces loaded: {len(self.brace_object_connectivity)}")
        print(f"📊 Linear elements combined: {len(self.frames_df)}")
        print(f"🧱 Walls loaded: {len(self.wall_object_connectivity)}")
        print(f"🪵 Floors loaded: {len(self.floor_object_connectivity)}")
        print(f"🧩 Floor point groups: {len(self.floor_points_by_story)} levels")
        print(f"🎨 Frame section colors assigned: {len(self.section_color_dict)}")
        print(f"📉 Element column forces: {len(self.element_forces_columns)}")
        print(f"🌀 Modal mass ratios: {len(self.modal_participating_mass_ratios)}")
        print(f"🧮 Story forces: {len(self.story_forces)}")
        print("---"*50)
        print(f"📊 Load combos: {self.combos}")
        print("---"*50)
        print(f"📍 Joint displacements: {len(self.joint_displacements)}")
        print("──────────────────────────────────────────────")
