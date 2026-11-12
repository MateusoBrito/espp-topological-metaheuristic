import pandas as pd
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DEFAULT_COLUMNS = [
    "Número Estação", "Latitude decimal", "Longitude decimal",
    "Empresa Estação", "Geração", "Tecnologia", "Município-UF"
]

def load_raw_data(file_path: str, columns: list = None):
    """Carrega os dados brutos de um arquivo CSV."""
    if columns is None:
        columns = DEFAULT_COLUMNS
    df = pd.read_csv(file_path, sep = ";", encoding="utf-8-sig", usecols=columns)

    n = len(df)
    df = df[
        df["Latitude decimal"].between(-34.0, 5.5) &
        df["Longitude decimal"].between(-74.5, -32.0)
    ]
    print(f"[Data Processing] {n - len(df)} registros removidos por coordenadas fora do Brasil.")

    return df

def assign_spatial_clusters(df: pd.DataFrame, radius_meters: float = 30.0) -> pd.DataFrame:
    """Agrupa coordenadas próximas dentro de um raio em metros sob um mesmo 'site_id'."""

    R = 6371.0  # Raio da Terra em km

    # 1. Trabalha apenas com os pares únicos (lat, lon)
    coords_df = df[["Latitude decimal", "Longitude decimal"]].drop_duplicates().reset_index(drop=True)

    # 2. Projeta na esfera 3D em km
    lat_r = np.radians(coords_df["Latitude decimal"].values)
    lon_r = np.radians(coords_df["Longitude decimal"].values)
    coords_3d = np.column_stack([
        R * np.cos(lat_r) * np.cos(lon_r),
        R * np.cos(lat_r) * np.sin(lon_r),
        R * np.sin(lat_r)
    ])

    # 3. Encontra todos os vizinhos dentro do raio e cria os clusters (connected_components)
    tree = cKDTree(coords_3d)
    pairs = tree.query_pairs(r=radius_meters / 1000.0, output_type="ndarray")
    
    n = len(coords_df)
    adj = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    n_sites, labels = connected_components(adj, directed=False)
    
    coords_df["site_id"] = labels
    print(f"[Spatial Clustering] {n:,} coords exatas -> {n_sites:,} sites físicos (raio={radius_meters}m)")
    # 4. Devolve o site_id para o DataFrame original
    return df.merge(coords_df, on=["Latitude decimal", "Longitude decimal"], how="left")

def group_unique_station(df: pd.DataFrame):
    """Agrupa por latitude/longitude consolidando operadoras e tecnologias presentes em cada ponto."""
    df_unicas = (
        df.dropna(subset=["site_id"])
        .groupby("site_id", as_index=False)
        .agg(
            id_local=("Número Estação", "first"),
            qtd_antenas=("Número Estação", "count"),
            operadoras=("Empresa Estação", lambda x: ", ".join(sorted(set(x.dropna())))),
            tecnologias=("Tecnologia", lambda x: ", ".join(sorted(set(x.dropna())))),
            latitude=("Latitude decimal", "mean"),
            longitude=("Longitude decimal", "mean"),
            municipio=("Município-UF", "first"),
            tem_2g=("Geração", lambda x: "2G" in set(x)),
            tem_3g=("Geração", lambda x: "3G" in set(x)),
            tem_4g=("Geração", lambda x: "4G" in set(x)),
            tem_5g=("Geração", lambda x: "5G" in set(x)),
        )
    )
    df_unicas["node_id"] = df_unicas.index
    return df_unicas

def save_data(df: pd.DataFrame, output_path: str):
    """Salva o DataFrame em um arquivo CSV no caminho especificado."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    df.to_csv(output_path, index=False)
    print(f"DataFrame salvo em: {output_path}")