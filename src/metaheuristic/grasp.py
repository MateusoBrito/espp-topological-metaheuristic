from dataclasses import dataclass
import random
import time
import numpy as np

from src.metaheuristic.construction import construcao_gulosa_aleatoria
from src.metaheuristic.evaluator import evaluate, weighted_average
from src.metaheuristic.instance import ProblemInstance
from src.metaheuristic.local_search import LOCAL_SEARCH_STRATEGIES


@dataclass
class GRASPResult:
    """Resultado consolidado de uma execução do GRASP."""

    best_solution: list[int]
    server_node_ids: list[int]
    best_cost_ms: float
    avg_latency_ms: float
    avg_distance_km: float
    history: list[float]
    elapsed_seconds: float


def grasp(
    instance: ProblemInstance,
    p: int,
    grasp_max: int = 30,
    alpha: float = 0.2,
    local_search_strategy: str = "best",
    seed: int | None = None,
    verbose: bool = True,
) -> GRASPResult:
    """
    Orquestrador do Algoritmo GRASP Básico (Algorithm 4):
      1. sbest <- inf
      2. iter <- 0
      3. while iter < GRASPMax do
      4.     s <- ConstrucaoGulosaAleatoria(alpha)
      5.     s <- BuscaLocal(s)
      6.     if f(s) < f(sbest) then sbest <- s
      7.     iter <- iter + 1
      8. return sbest

    Args:
        instance: Instância do problema (matrizes, demandas e candidatos).
        p: Número de servidores a alocar.
        grasp_max: Número máximo de iterações do GRASP.
        alpha: Parâmetro da LRC (0 = guloso puro, 1 = aleatório).
        local_search_strategy: Estratégia de busca local ('best' ou 'first').
        seed: Semente opcional para reprodutibilidade.
        verbose: Se True, imprime cada melhoria encontrada.
    """
    if local_search_strategy not in LOCAL_SEARCH_STRATEGIES:
        valid = ", ".join(LOCAL_SEARCH_STRATEGIES.keys())
        raise ValueError(f"Estratégia de busca local inválida: '{local_search_strategy}'. Opções: {valid}.")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    busca_local = LOCAL_SEARCH_STRATEGIES[local_search_strategy]
    cost_matrix = instance.latency_matrix_ms
    demands = instance.demands
    candidatos = instance.candidates

    sbest: list[int] = []
    f_sbest = float("inf")
    history: list[float] = []

    inicio = time.time()
    for iteracao in range(grasp_max):
        s = construcao_gulosa_aleatoria(candidatos, p, alpha, cost_matrix, demands)
        s = busca_local(s, candidatos, cost_matrix, demands)
        f_s = evaluate(s, cost_matrix, demands)

        if f_s < f_sbest:
            f_sbest = f_s
            sbest = list(s)
            if verbose:
                print(
                    f"[GRASP] Iteração {iteracao + 1:3d}/{grasp_max} | "
                    f"Novo melhor custo: {f_sbest:,.2f} ms·usuário"
                )

        history.append(f_sbest)
    elapsed = time.time() - inicio

    avg_latency_ms = weighted_average(f_sbest, demands)
    total_dist_km = evaluate(sbest, instance.dist_matrix_km, demands)
    avg_distance_km = weighted_average(total_dist_km, demands)
    server_node_ids = [instance.node_ids[idx] for idx in sbest]

    return GRASPResult(
        best_solution=sbest,
        server_node_ids=server_node_ids,
        best_cost_ms=f_sbest,
        avg_latency_ms=avg_latency_ms,
        avg_distance_km=avg_distance_km,
        history=history,
        elapsed_seconds=elapsed,
    )


if __name__ == "__main__":
    from src.metaheuristic.run_experiments import main

    main()
