from dataclasses import dataclass, replace
from pathlib import Path
import networkx as nx
import numpy as np
from scipy.sparse import csgraph, csr_matrix

from src.metaheuristic.evaluator import km_to_rtt_ms


@dataclass
class ProblemInstance:
    """Representa uma instância do problema da p-Mediana (Edge Server Placement)."""

    name: str
    dist_matrix_km: np.ndarray
    latency_matrix_ms: np.ndarray
    demands: np.ndarray
    node_ids: list[int]
    candidates: list[int]
    graph: nx.Graph | None = None

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_candidates(self) -> int:
        return len(self.candidates)


def load_sergipe_instance(
    graph_path: str = "data/processed/grafo_se.graphml",
    demands_path: str = "data/processed/vetor_demandas_se.bin",
    matrix_cache_path: str = "data/processed/distance_matrix_se.bin",
) -> ProblemInstance:
    """
    Carrega a instância real de Sergipe (grafo Waxman + vetor de demandas do IBGE),
    calculando (ou recuperando do cache binário) a matriz de caminhos mínimos via Dijkstra.
    """
    G = nx.read_graphml(graph_path, node_type=int)
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    node_ids = list(G.nodes())
    n = len(node_ids)

    cache_file = Path(matrix_cache_path)
    if cache_file.exists() and cache_file.stat().st_size == n * n * 8:
        with open(cache_file, "rb") as f:
            dist_matrix = np.frombuffer(f.read(), dtype=np.float64).reshape((n, n))
    else:
        sparse_adj = nx.to_scipy_sparse_array(G, weight="weight", dtype=np.float64)
        sparse_adj = csr_matrix(
            (
                sparse_adj.data,
                sparse_adj.indices.astype(np.int32),
                sparse_adj.indptr.astype(np.int32),
            ),
            shape=(n, n),
        )
        dist_matrix = csgraph.shortest_path(sparse_adj, directed=False, method="D")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            f.write(dist_matrix.astype(np.float64).tobytes())

    with open(demands_path, "rb") as f:
        _total = np.frombuffer(f.read(4), dtype=np.int32)[0]
        raw_demandas = np.frombuffer(f.read(), dtype=np.float64)

    demands = np.array([raw_demandas[node] for node in node_ids], dtype=np.float64)
    latency_matrix = km_to_rtt_ms(dist_matrix)

    return ProblemInstance(
        name="Sergipe (SE)",
        dist_matrix_km=dist_matrix,
        latency_matrix_ms=latency_matrix,
        demands=demands,
        node_ids=node_ids,
        candidates=list(range(n)),
        graph=G,
    )


def generate_mock_instance(n: int = 30, seed: int = 42) -> ProblemInstance:
    """Gera uma instância sintética pré-definida para testes rápidos e validação."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 50, size=(n, 2))
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    demands = rng.uniform(10, 100, size=n)
    latency_matrix = km_to_rtt_ms(dist_matrix)

    return ProblemInstance(
        name=f"Mock Sintético (N={n})",
        dist_matrix_km=dist_matrix,
        latency_matrix_ms=latency_matrix,
        demands=demands,
        node_ids=list(range(n)),
        candidates=list(range(n)),
        graph=None,
    )


def filter_candidates_by_centrality(
    instance: ProblemInstance,
    metric: str = "betweenness",
    top_k: int = 50,
) -> ProblemInstance:
    """
    Aplica a filtragem topológica de Redes Complexas sobre o espaço de busca,
    retornando uma nova instância restrita aos top_k nós mais centrais.

    Args:
        instance: Instância original (deve conter o grafo NetworkX).
        metric: Métrica de centralidade ('betweenness', 'degree', 'closeness').
        top_k: Quantidade de nós candidatos a manter.
    """
    if instance.graph is None:
        raise ValueError("A filtragem por centralidade requer o grafo carregado na instância.")

    G = instance.graph
    metric = metric.strip().lower()

    if metric == "betweenness":
        scores = nx.betweenness_centrality(G, weight="weight")
    elif metric == "degree":
        scores = nx.degree_centrality(G)
    elif metric == "closeness":
        scores = nx.closeness_centrality(G, distance="weight")
    else:
        raise ValueError(f"Métrica desconhecida: '{metric}'. Use 'betweenness', 'degree' ou 'closeness'.")

    k = min(top_k, instance.num_nodes)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(instance.node_ids)}
    top_nodes = sorted(scores.keys(), key=lambda n: scores[n], reverse=True)[:k]
    filtered_candidates = [node_to_idx[n] for n in top_nodes]

    return replace(
        instance,
        name=f"{instance.name} [Top-{k} {metric}]",
        candidates=filtered_candidates,
    )
