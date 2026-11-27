import argparse

from src.metaheuristic.grasp import grasp
from src.metaheuristic.instance import (
    ProblemInstance,
    filter_candidates_by_centrality,
    generate_mock_instance,
    load_sergipe_instance,
)
from src.metaheuristic.local_search import LOCAL_SEARCH_STRATEGIES


def run_single(
    instance: ProblemInstance,
    p: int,
    grasp_max: int,
    alpha: float,
    local_search: str,
    seed: int,
) -> None:
    print("=" * 64)
    print("  GRASP — Otimização de Alocação de Servidores (p-Mediana)")
    print("=" * 64)
    print(f"Instância:               {instance.name}")
    print(f"Total de nós (torres):   {instance.num_nodes}")
    print(f"Candidatos disponíveis:  {instance.num_candidates}")
    print(f"Servidores a alocar (p): {p}")
    print(f"Alpha (LRC):             {alpha}")
    print(f"Busca Local (1-swap):    {local_search}")
    print(f"Iterações (GRASPMax):    {grasp_max}")
    print("-" * 64)

    result = grasp(
        instance=instance,
        p=p,
        grasp_max=grasp_max,
        alpha=alpha,
        local_search_strategy=local_search,
        seed=seed,
        verbose=True,
    )

    print("=" * 64)
    print("=== RESULTADO GRASP (p-Mediana) ===")
    print(f"Melhor custo total (Latência RTT x População): {result.best_cost_ms:,.2f} ms·usuário")
    print(f"Latência média ponderada (RTT): {result.avg_latency_ms:.4f} ms ({result.avg_distance_km:.2f} km)")
    print(f"Índices internos dos servidores: {result.best_solution}")
    print(f"IDs reais das torres escolhidas: {result.server_node_ids}")
    print(f"Tempo total de execução: {result.elapsed_seconds:.2f} segundos")
    print("=" * 64)


def run_benchmark(base_instance: ProblemInstance, grasp_max: int, alpha: float, seed: int) -> None:
    """
    Executa uma bateria comparativa de testes para o relatório da Etapa 2:
    compara diferentes valores de p, estratégias de busca local (best vs first)
    e o impacto da filtragem de candidatos por Redes Complexas (Betweenness).
    """
    print("=" * 92)
    print("  BATERIA DE TESTES COMPARATIVOS — GRASP (p-Mediana / Sergipe)")
    print(f"  Parâmetros fixos: GRASPMax={grasp_max} | alpha={alpha} | seed={seed}")
    print("=" * 92)

    scenarios: list[tuple[str, ProblemInstance, int, str]] = [
        ("Grafo Completo", base_instance, 5, "best"),
        ("Grafo Completo", base_instance, 5, "first"),
        ("Grafo Completo", base_instance, 10, "best"),
        ("Grafo Completo", base_instance, 10, "first"),
    ]

    if base_instance.graph is not None:
        inst_bet50 = filter_candidates_by_centrality(base_instance, metric="betweenness", top_k=50)
        scenarios.extend([
            ("Top-50 Betweenness", inst_bet50, 5, "best"),
            ("Top-50 Betweenness", inst_bet50, 10, "best"),
            ("Top-50 Betweenness", inst_bet50, 10, "first"),
        ])

    header = (
        f"{'Cenário':<20} | {'Cand.':>5} | {'p':>2} | {'Busca':<5} | "
        f"{'Custo (ms·usu)':>15} | {'RTT (ms)':>9} | {'Dist (km)':>9} | {'Tempo (s)':>9}"
    )
    print(header)
    print("-" * len(header))

    for label, inst, p_val, ls_strategy in scenarios:
        res = grasp(
            instance=inst,
            p=p_val,
            grasp_max=grasp_max,
            alpha=alpha,
            local_search_strategy=ls_strategy,
            seed=seed,
            verbose=False,
        )
        print(
            f"{label:<20} | {inst.num_candidates:>5} | {p_val:>2} | {ls_strategy:<5} | "
            f"{res.best_cost_ms:>15,.2f} | {res.avg_latency_ms:>9.4f} | "
            f"{res.avg_distance_km:>9.2f} | {res.elapsed_seconds:>9.2f}"
        )

    print("=" * 92)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execução e testes do GRASP modularizado para o problema da p-Mediana."
    )
    parser.add_argument("--mock", action="store_true", help="Usa instância sintética pré-definida (N=30)")
    parser.add_argument("--benchmark", action="store_true", help="Roda bateria comparativa de testes")
    parser.add_argument("--p", type=int, default=5, help="Número de servidores a alocar (padrão: 5)")
    parser.add_argument("--grasp_max", type=int, default=10, help="Iterações do GRASP (padrão: 10)")
    parser.add_argument("--alpha", type=float, default=0.2, help="Parâmetro alpha da LRC (padrão: 0.2)")
    parser.add_argument(
        "--local_search",
        type=str,
        default="best",
        choices=list(LOCAL_SEARCH_STRATEGIES.keys()),
        help="Estratégia de busca local 1-swap: 'best' ou 'first' (padrão: best)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="none",
        choices=["none", "betweenness", "degree", "closeness"],
        help="Filtro opcional de candidatos por centralidade de Redes Complexas",
    )
    parser.add_argument("--top_k", type=int, default=50, help="Quantidade de candidatos quando --filter != none")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória (padrão: 42)")
    args = parser.parse_args()

    instance = generate_mock_instance(n=30, seed=args.seed) if args.mock else load_sergipe_instance()

    if args.benchmark:
        run_benchmark(instance, grasp_max=args.grasp_max, alpha=args.alpha, seed=args.seed)
        return

    if args.filter != "none":
        instance = filter_candidates_by_centrality(instance, metric=args.filter, top_k=args.top_k)

    run_single(
        instance=instance,
        p=args.p,
        grasp_max=args.grasp_max,
        alpha=args.alpha,
        local_search=args.local_search,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
