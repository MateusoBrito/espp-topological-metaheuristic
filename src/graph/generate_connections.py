import numpy as np
import networkx as nx
from pathlib import Path
from scipy.spatial import cKDTree, ConvexHull, QhullError

R = 6371.0  # Raio médio da Terra em km


def calculate_haversine_matrix(lats, lons):
    """
    Calcula a matriz de distâncias par a par em quilômetros usando Haversine vetorizado.
    Recomendada apenas para conjuntos pequenos (N < 5.000) devido ao consumo de memória O(N^2).
    """
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)

    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) * np.sin(dlon / 2) ** 2
    )
    a = np.clip(a, 0.0, 1.0)

    return 2.0 * R * np.arcsin(np.sqrt(a))


def _latlon_to_cartesian_3d(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Projeta coordenadas geográficas (graus) na esfera terrestre 3D em km."""
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    return np.column_stack([
        R * np.cos(lat_rad) * np.cos(lon_rad),
        R * np.cos(lat_rad) * np.sin(lon_rad),
        R * np.sin(lat_rad),
    ])


def _chord_to_haversine_km(chord_dist: np.ndarray) -> np.ndarray:
    """Converte distância euclidiana da corda 3D (em km) para distância geodésica na superfície."""
    return 2.0 * R * np.arcsin(np.clip(chord_dist / (2.0 * R), 0.0, 1.0))


def calculate_max_diameter_km(lats: np.ndarray, lons: np.ndarray, coords_3d: np.ndarray) -> float:
    """
    Calcula o diâmetro geográfico máximo L (em km) usando o Fecho Convexo 2D.
    Evita alocar matrizes N x N para dezenas de milhares de pontos.
    """
    n = len(lats)
    if n <= 1:
        return 0.0

    try:
        if n >= 3:
            hull = ConvexHull(np.column_stack([lons, lats]))
            idx = hull.vertices
        else:
            idx = np.arange(n)
    except QhullError:
        idx = np.arange(n)

    pts = coords_3d[idx]
    diffs = pts[:, None, :] - pts[None, :, :]
    chord_dists = np.linalg.norm(diffs, axis=-1)
    return float(np.max(_chord_to_haversine_km(chord_dists)))


def generate_waxman_graph(df_stations, alpha=0.4, beta=0.0003, max_dist_km=15.0, seed=42) -> nx.Graph:
    """
    Gera um grafo espacial usando o Modelo de Waxman otimizado com cKDTree (Opção A).
    Escalável para o Brasil inteiro sem estourar memória RAM.

    Args:
        df_stations (pd.DataFrame): DataFrame com as estações únicas.
        alpha (float): Controla a densidade geral de enlaces (0 < alpha <= 1).
        beta (float): Controla o alcance dos enlaces relativo ao diâmetro L.
                      - Para o Brasil todo (L ~ 4.500 km): use valores entre 0.0003 e 0.0005 (beta*L ~ 1.3 a 2.2 km).
                      - Para um município isolado (L ~ 40 km): use valores entre 0.05 e 0.15.
        max_dist_km (float): Raio máximo de busca para enlaces diretos do modelo de Waxman (km).
        seed (int): Semente para geração de números aleatórios (reprodutibilidade).

    Returns:
        nx.Graph: Grafo com nós georreferenciados e arestas ponderadas pela distância em km.
    """
    np.random.seed(seed)
    print(f"[Waxman] Gerando grafo Waxman com alpha={alpha}, beta={beta}, max_dist_km={max_dist_km} km, seed={seed}")
    n = len(df_stations)
    if n == 0:
        raise ValueError("O DataFrame df_stations está vazio. Verifique o filtro aplicado.")

    lats = df_stations["latitude"].values
    lons = df_stations["longitude"].values
    ids = df_stations["node_id"].values.astype(int)

    # 1. Projeta para 3D na esfera e constrói a árvore espacial (cKDTree)
    coords_3d = _latlon_to_cartesian_3d(lats, lons)
    tree = cKDTree(coords_3d)

    # 2. Diâmetro máximo L calculado apenas sobre os vértices da borda (Convex Hull)
    L = calculate_max_diameter_km(lats, lons, coords_3d)
    print(f"[Waxman] Diâmetro geográfico máximo (L): {L:.2f} km | beta*L = {beta * L:.2f} km")

    # 3. Busca esparsa apenas de pares dentro do raio max_dist_km
    pairs = tree.query_pairs(r=max_dist_km, output_type="ndarray")
    if len(pairs) > 0:
        u_idx = pairs[:, 0]
        v_idx = pairs[:, 1]

        d_chord = np.linalg.norm(coords_3d[u_idx] - coords_3d[v_idx], axis=1)
        d_km = _chord_to_haversine_km(d_chord)

        # Filtra exatamente por distância geodésica <= max_dist_km
        valid_mask = d_km <= max_dist_km
        u_idx = u_idx[valid_mask]
        v_idx = v_idx[valid_mask]
        d_km = d_km[valid_mask]

        # 4. Probabilidade de Waxman apenas nos pares candidatos
        probs = alpha * np.exp(-d_km / (beta * L)) if L > 0 else np.zeros(len(d_km))
        keep_mask = np.random.rand(len(d_km)) < probs

        u_sel = u_idx[keep_mask]
        v_sel = v_idx[keep_mask]
        d_sel = d_km[keep_mask]
    else:
        u_sel = np.array([], dtype=int)
        v_sel = np.array([], dtype=int)
        d_sel = np.array([], dtype=float)

    # 5. Criar o grafo no NetworkX (inserção vetorizada para máxima velocidade)
    G = nx.Graph()

    id_locais = df_stations["id_local"].values if "id_local" in df_stations else np.full(n, -1)
    operadoras = df_stations["operadoras"].fillna("").astype(str).values if "operadoras" in df_stations else np.full(n, "")
    tecnologias = df_stations["tecnologias"].fillna("").astype(str).values if "tecnologias" in df_stations else np.full(n, "")
    qtd_antenas = df_stations["qtd_antenas"].fillna(1).astype(int).values if "qtd_antenas" in df_stations else np.ones(n, dtype=int)
    tem_5g = df_stations["tem_5g"].fillna(False).astype(bool).values if "tem_5g" in df_stations else np.zeros(n, dtype=bool)

    nodes_to_add = [
        (
            int(ids[i]),
            {
                "pos": (float(lons[i]), float(lats[i])),
                "latitude": float(lats[i]),
                "longitude": float(lons[i]),
                "id_local": int(id_locais[i]) if id_locais[i] == id_locais[i] else -1,
                "operadoras": operadoras[i],
                "tecnologias": tecnologias[i],
                "qtd_antenas": int(qtd_antenas[i]),
                "tem_5g": bool(tem_5g[i]),
            },
        )
        for i in range(n)
    ]
    G.add_nodes_from(nodes_to_add)

    edges_to_add = [
        (int(ids[u]), int(ids[v]), {"weight": float(d), "distance_km": float(d), "tipo": "waxman"})
        for u, v, d in zip(u_sel, v_sel, d_sel)
    ]
    G.add_edges_from(edges_to_add)

    # 6. Fallback: conecta estações isoladas (inclusive > max_dist_km) à estação mais próxima
    cabos_fallback = 0
    if n > 1:
        isolated_indices = [i for i in range(n) if G.degree(int(ids[i])) == 0]
        if isolated_indices:
            dists_chord, nearest_idx = tree.query(coords_3d[isolated_indices], k=2)
            nearest_dists_km = _chord_to_haversine_km(dists_chord[:, 1])
            nearest_neighbors = nearest_idx[:, 1]

            for idx_pos, u_i in enumerate(isolated_indices):
                u_id = int(ids[u_i])
                v_i = int(nearest_neighbors[idx_pos])
                v_id = int(ids[v_i])
                if not G.has_edge(u_id, v_id):
                    dist_km = float(nearest_dists_km[idx_pos])
                    G.add_edge(u_id, v_id, weight=dist_km, distance_km=dist_km, tipo="fallback_mais_proxima")
                    cabos_fallback += 1

    print(f"[Waxman] Cabos de fallback adicionados (nós isolados / > {max_dist_km} km): {cabos_fallback:,}")
    print(f"[Waxman] Grafo gerado: {G.number_of_nodes():,} nós e {G.number_of_edges():,} arestas.")
    return G


def save_graph(G, output_path):
    """
    Salva o grafo em disco no formato GraphML ou Pickle (.pkl).

    Args:
        G (nx.Graph): Grafo NetworkX a ser salvo.
        output_path (str): Caminho do arquivo de destino (ex: 'data/processed/grafo_waxman.graphml').
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix in [".pkl", ".pickle"]:
        import pickle
        with open(output_path, "wb") as f:
            pickle.dump(G, f)
    else:
        # GraphML não suporta tuplas (pos=(lon, lat)). Removemos na cópia de exportação.
        G_export = G.copy()
        for node in G_export.nodes:
            G_export.nodes[node].pop("pos", None)
        nx.write_graphml(G_export, str(output_path))

    print(f"[Waxman] Grafo salvo com sucesso em: {output_path}")
