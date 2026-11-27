import numpy as np

from src.metaheuristic.evaluator import evaluate


def busca_local_best_improvement(
    s: list[int],
    candidatos: list[int],
    cost_matrix: np.ndarray,
    demands: np.ndarray,
) -> list[int]:
    """
    Fase 2 do GRASP: Busca Local com Best Improvement na vizinhança 1-swap.
    Avalia todas as trocas possíveis entre um servidor ativo e um candidato inativo
    e adota a melhor troca global de cada iteração até atingir um ótimo local.
    """
    melhoria = True
    melhor_custo = evaluate(s, cost_matrix, demands)
    candidatos_set = set(candidatos)

    while melhoria:
        melhoria = False
        melhor_vizinho = list(s)
        inativos = list(candidatos_set - set(s))

        for i in range(len(s)):
            for novo_cand in inativos:
                vizinho = list(s)
                vizinho[i] = novo_cand

                custo_vizinho = evaluate(vizinho, cost_matrix, demands)
                if custo_vizinho < melhor_custo:
                    melhor_custo = custo_vizinho
                    melhor_vizinho = vizinho
                    melhoria = True

        if melhoria:
            s = melhor_vizinho

    return s


def busca_local_first_improvement(
    s: list[int],
    candidatos: list[int],
    cost_matrix: np.ndarray,
    demands: np.ndarray,
) -> list[int]:
    """
    Variação da Fase 2 do GRASP: Busca Local com First Improvement na vizinhança 1-swap.
    Aceita imediatamente a primeira troca que reduzir o custo total, acelerando a
    convergência em instâncias maiores.
    """
    melhoria = True
    melhor_custo = evaluate(s, cost_matrix, demands)
    candidatos_set = set(candidatos)

    while melhoria:
        melhoria = False
        inativos = list(candidatos_set - set(s))

        for i in range(len(s)):
            for novo_cand in inativos:
                vizinho = list(s)
                vizinho[i] = novo_cand

                custo_vizinho = evaluate(vizinho, cost_matrix, demands)
                if custo_vizinho < melhor_custo:
                    melhor_custo = custo_vizinho
                    s = vizinho
                    melhoria = True
                    break
            if melhoria:
                break

    return s


LOCAL_SEARCH_STRATEGIES = {
    "best": busca_local_best_improvement,
    "first": busca_local_first_improvement,
}
