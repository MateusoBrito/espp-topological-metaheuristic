import numpy as np
import networkx as nx
from pathlib import Path
from scipy.sparse import csgraph, csr_matrix

# Velocidade aproximada de propagação da luz em fibra óptica (~200 km/ms).
# Para RTT (ida e volta): rtt_ms = (2 * dist_km) / 200.0 = dist_km / 100.0
FIBER_PROPAGATION_SPEED_KM_PER_MS = 200.0


def km_to_rtt_ms(dist_km: float | np.ndarray) -> float | np.ndarray:
    """Converte distância geodésica em km para latência RTT (ida e volta) em ms."""
    return (2.0 * dist_km) / FIBER_PROPAGATION_SPEED_KM_PER_MS


def evaluate(S: list[int] | set[int], dist_matrix: np.ndarray, demands: np.ndarray) -> float:
    """
    Calcula a função objetivo do problema da p-Mediana (latência populacional total):
    f(S) = sum_{i in V} w_i * min_{s in S} d(i, s)

    Args:
        S: Lista de índices dos servidores escolhidos (ex: [0, 5, 23]).
        dist_matrix: Matriz (N x N) de distâncias de caminhos mínimos.
        demands: Vetor de dimensão N com a demanda populacional de cada nó.

    Returns:
        Custo total da solução f(S).
    """
    if len(S) == 0:
        return float("inf")
    # Para cada um dos N nós, pega a distância mínima até o servidor mais próximo em S
    min_dists = np.min(dist_matrix[:, list(S)], axis=1)
    # Multiplica pela demanda populacional e soma
    return float(np.sum(demands * min_dists))


def load_sergipe_data(
    graph_path: str = "data/processed/grafo_se.graphml",
    demands_path: str = "data/processed/vetor_demandas_se.bin",
    matrix_cache_path: str = "data/processed/distance_matrix_se.bin",
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Carrega o grafo de Sergipe e o vetor de demandas, calculando ou recuperando
    a matriz de distâncias de caminho mínimo no grafo.

    Returns:
        (dist_matrix, demands, node_ids):
        - dist_matrix: Matriz numpy (N x N) de distâncias geodésicas (km).
        - demands: Vetor numpy (N,) de demandas populacionais.
        - node_ids: Lista com os IDs originais das torres correspondentes aos índices 0..N-1.
    """
    G = nx.read_graphml(graph_path, node_type=int)
    # Garante maior componente conexo
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    node_ids = list(G.nodes())
    N = len(node_ids)

    # 1. Matriz de distâncias (tenta ler cache ou calcula via Dijkstra)
    cache_file = Path(matrix_cache_path)
    if cache_file.exists() and cache_file.stat().st_size == N * N * 8:
        print(f"[DataLoader] Carregando matriz de distâncias em cache ({N}x{N}): {cache_file}")
        with open(cache_file, "rb") as f:
            dist_matrix = np.frombuffer(f.read(), dtype=np.float64).reshape((N, N))
    else:
        print(f"[DataLoader] Calculando caminhos mínimos para {N} nós...")
        sparse_adj = nx.to_scipy_sparse_array(G, weight="weight", dtype=np.float64)
        sparse_adj = csr_matrix(
            (sparse_adj.data, sparse_adj.indices.astype(np.int32), sparse_adj.indptr.astype(np.int32)),
            shape=(N, N),
        )
        dist_matrix = csgraph.shortest_path(sparse_adj, directed=False)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            f.write(dist_matrix.astype(np.float64).tobytes())
        print(f"[DataLoader] Matriz de distâncias salva em cache: {cache_file}")

    # 2. Vetor de demandas alinhado aos nós do maior componente conexo
    with open(demands_path, "rb") as f:
        _total = np.frombuffer(f.read(4), dtype=np.int32)[0]
        raw_demandas = np.frombuffer(f.read(), dtype=np.float64)

    demands = np.array([raw_demandas[node] for node in node_ids], dtype=np.float64)

    return dist_matrix, demands, node_ids


def generate_mock_data(n: int = 20, seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Gera dados sintéticos pré-definidos para testes rápidos e validação do algoritmo.
    """
    np.random.seed(seed)
    # Coordenadas aleatórias em uma área de 50x50 km
    coords = np.random.uniform(0, 50, size=(n, 2))
    # Matriz de distâncias euclidianas
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    # Demandas aleatórias entre 10 e 100 habitantes
    demands = np.random.uniform(10, 100, size=n)
    node_ids = list(range(n))
    return dist_matrix, demands, node_ids
