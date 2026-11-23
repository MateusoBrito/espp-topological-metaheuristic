import time
import random
import argparse
import numpy as np
from src.metaheuristic.evaluator import evaluate, load_sergipe_data, generate_mock_data


def construcao_gulosa_aleatoria(
    candidatos: list[int],
    p: int,
    alpha: float,
    dist_matrix: np.ndarray,
    demands: np.ndarray,
) -> list[int]:
    """
    Fase de Construção Gulosa Aleatória usando LRC (Lista Restrita de Candidatos) por valor.
    Adaptado para o problema da p-Mediana .

    Args:
        candidatos: Lista de índices dos nós candidatos a receber servidores.
        p: Quantidade de servidores a instalar.
        alpha: Parâmetro da LRC (0 = guloso puro, 1 = totalmente aleatório).
        dist_matrix: Matriz de distâncias (N x N).
        demands: Vetor de demandas populacionais (N,).

    Returns:
        s: Lista com os p servidores escolhidos.
    """
    s: list[int] = []
    candidatos_disponiveis = set(candidatos)

    while len(s) < p:
        # Avalia o custo incremental de cada candidato disponível
        custos = []
        for cand in candidatos_disponiveis:
            s_cand = s + [cand]
            custo = evaluate(s_cand, dist_matrix, demands)
            custos.append((cand, custo))

        # Encontra c_min e c_max
        valores_custo = [c[1] for c in custos]
        c_min = min(valores_custo)
        c_max = max(valores_custo)

        # Monta a LRC por valor: custo <= c_min + alpha * (c_max - c_min)
        limite = c_min + alpha * (c_max - c_min)
        lrc = [cand for cand, custo in custos if custo <= limite]

        # Escolhe aleatoriamente um elemento da LRC
        escolhido = random.choice(lrc)
        s.append(escolhido)
        candidatos_disponiveis.remove(escolhido)

    return s


def busca_local(
    s: list[int],
    candidatos: list[int],
    dist_matrix: np.ndarray,
    demands: np.ndarray,
) -> list[int]:
    """
    Busca Local com Best Improvement utilizando a vizinhança 1-swap:
    Testa a troca de cada servidor ativo s[i] por um candidato inativo.
    """
    melhoria = True
    melhor_custo = evaluate(s, dist_matrix, demands)
    candidatos_set = set(candidatos)

    while melhoria:
        melhoria = False
        melhor_vizinho = list(s)
        inativos = list(candidatos_set - set(s))

        # Testa todas as duplas: (servidor ativo, candidato inativo)
        for i in range(len(s)):
            for novo_cand in inativos:
                vizinho = list(s)
                vizinho[i] = novo_cand

                custo_vizinho = evaluate(vizinho, dist_matrix, demands)

                if custo_vizinho < melhor_custo:
                    melhor_custo = custo_vizinho
                    melhor_vizinho = vizinho
                    melhoria = True

        if melhoria:
            s = melhor_vizinho

    return s


def grasp(
    candidatos: list[int],
    p: int,
    grasp_max: int,
    alpha: float,
    dist_matrix: np.ndarray,
    demands: np.ndarray,
    verbose: bool = True,
) -> tuple[list[int], float, list[float]]:
   
    sbest: list[int] = []
    f_sbest = float("inf")
    historico = []

    for iteracao in range(grasp_max):
        s = construcao_gulosa_aleatoria(candidatos, p, alpha, dist_matrix, demands)
        s = busca_local(s, candidatos, dist_matrix, demands)

        f_s = evaluate(s, dist_matrix, demands)

        if f_s < f_sbest:
            f_sbest = f_s
            sbest = list(s)
            if verbose:
                print(f"[GRASP] Iteração {iteracao + 1:3d}/{grasp_max} | Novo melhor custo: {f_sbest:,.2f}")

        historico.append(f_sbest)

    return sbest, f_sbest, historico


def main():
    parser = argparse.ArgumentParser(description="GRASP para o problema da p-Mediana (Edge Server Placement)")
    parser.add_argument("--mock", action="store_true", help="Usa dados sintéticos pré-definidos para teste rápido")
    parser.add_argument("--p", type=int, default=5, help="Número de servidores a alocar (padrão: 5)")
    parser.add_argument("--grasp_max", type=int, default=30, help="Número de iterações do GRASP (padrão: 30)")
    parser.add_argument("--alpha", type=float, default=0.2, help="Parâmetro alpha da LRC (padrão: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Semente de aleatoriedade")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("  GRASP — Otimização de Alocação de Servidores (p-Mediana)")
    print("=" * 60)

    if args.mock:
        print("[Modo] Utilizando dados sintéticos pré-definidos (N = 30)...")
        dist_matrix, demands, node_ids = generate_mock_data(n=30, seed=args.seed)
    else:
        print("[Modo] Carregando dados reais de Sergipe...")
        dist_matrix, demands, node_ids = load_sergipe_data()

    N = len(node_ids)
    candidatos = list(range(N))  # Por padrão, todos os nós são candidatos

    print(f"Total de nós (torres): {N}")
    print(f"Candidatos disponíveis: {len(candidatos)}")
    print(f"Servidores a alocar (p): {args.p}")
    print(f"Alpha (LRC): {args.alpha}")
    print(f"Iterações (GRASPMax): {args.grasp_max}")
    print("-" * 60)

    inicio = time.time()
    sbest, f_sbest, _historico = grasp(
        candidatos=candidatos,
        p=args.p,
        grasp_max=args.grasp_max,
        alpha=args.alpha,
        dist_matrix=dist_matrix,
        demands=demands,
        verbose=True,
    )
    fim = time.time()

    # Mapeia os índices internos de volta para os IDs reais das estações
    servidores_reais = [node_ids[idx] for idx in sbest]

    # Latência média por habitante
    demanda_total = np.sum(demands)
    latencia_media_km = f_sbest / demanda_total if demanda_total > 0 else 0.0

    print("=" * 60)
    print("=== RESULTADO GRASP (p-Mediana) ===")
    print(f"Melhor custo total (Latência x População): {f_sbest:,.2f}")
    print(f"Latência média ponderada: {latencia_media_km:.2f} km por usuário")
    print(f"Índices internos dos servidores: {sbest}")
    print(f"IDs reais das torres escolhidas: {servidores_reais}")
    print(f"Tempo total de execução: {fim - inicio:.2f} segundos")
    print("=" * 60)


if __name__ == "__main__":
    main()
