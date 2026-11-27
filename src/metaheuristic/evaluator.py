import numpy as np

# Velocidade aproximada de propagação da luz em fibra óptica (~200 km/ms).
# Para RTT (ida e volta): rtt_ms = (2 * dist_km) / 200.0 = dist_km / 100.0
FIBER_PROPAGATION_SPEED_KM_PER_MS = 200.0


def km_to_rtt_ms(dist_km: float | np.ndarray) -> float | np.ndarray:
    """Converte distância geodésica em km para latência RTT (ida e volta) em ms."""
    return (2.0 * dist_km) / FIBER_PROPAGATION_SPEED_KM_PER_MS


def evaluate(
    solution: list[int] | set[int],
    cost_matrix: np.ndarray,
    demands: np.ndarray,
) -> float:
    """
    Calcula a função objetivo do problema da p-Mediana com demanda ponderada:
        f(S) = sum_{i in V} w_i * min_{s in S} c(i, s)

    Args:
        solution: Índices internos dos servidores escolhidos (ex.: [0, 5, 23]).
        cost_matrix: Matriz (N x N) de custos entre nós (latência em ms ou distância em km).
        demands: Vetor de dimensão N com a demanda populacional w_i de cada nó.

    Returns:
        Custo total ponderado f(S).
    """
    if len(solution) == 0:
        return float("inf")
    min_costs = np.min(cost_matrix[:, list(solution)], axis=1)
    return float(np.sum(demands * min_costs))


def weighted_average(total_cost: float, demands: np.ndarray) -> float:
    """Calcula a média ponderada por unidade de demanda: f(S) / sum(w_i)."""
    total_demand = float(np.sum(demands))
    return total_cost / total_demand if total_demand > 0 else 0.0
