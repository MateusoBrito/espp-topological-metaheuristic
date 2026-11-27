import random
import numpy as np

from src.metaheuristic.evaluator import evaluate


def construcao_gulosa_aleatoria(
    candidatos: list[int],
    p: int,
    alpha: float,
    cost_matrix: np.ndarray,
    demands: np.ndarray,
) -> list[int]:
    """
    Fase 1 do GRASP: Construção Gulosa Aleatória usando LRC (Lista Restrita de Candidatos) por valor.

    A cada passo, avalia o custo incremental de adicionar cada candidato disponível c:
        limite = c_min + alpha * (c_max - c_min)
        LRC = { c | custo(S U {c}) <= limite }
    Em seguida, sorteia aleatoriamente um candidato da LRC até atingir p servidores.

    Args:
        candidatos: Índices dos nós candidatos a receber servidores.
        p: Quantidade de servidores a alocar.
        alpha: Parâmetro de aleatoriedade da LRC (0 = guloso puro, 1 = totalmente aleatório).
        cost_matrix: Matriz (N x N) de custos (latência em ms).
        demands: Vetor de demandas populacionais (N,).

    Returns:
        Lista com os p índices dos servidores escolhidos.
    """
    if p > len(candidatos):
        raise ValueError(f"p={p} é maior que o número de candidatos disponíveis ({len(candidatos)}).")

    s: list[int] = []
    candidatos_disponiveis = set(candidatos)

    while len(s) < p:
        custos = []
        for cand in candidatos_disponiveis:
            custo = evaluate(s + [cand], cost_matrix, demands)
            custos.append((cand, custo))

        valores_custo = [c[1] for c in custos]
        c_min = min(valores_custo)
        c_max = max(valores_custo)

        limite = c_min + alpha * (c_max - c_min)
        lrc = [cand for cand, custo in custos if custo <= limite]

        escolhido = random.choice(lrc)
        s.append(escolhido)
        candidatos_disponiveis.remove(escolhido)

    return s
