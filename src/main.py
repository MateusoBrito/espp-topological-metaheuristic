import argparse
import pandas as pd
from pathlib import Path
from graph.data_processing import load_raw_data, assign_spatial_clusters, group_unique_station, save_data
from graph.generate_connections import generate_waxman_graph, save_graph
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def argument_parser():
    parser = argparse.ArgumentParser(description="Processa estações, gera conexões e plota o mapa.")
    parser.add_argument("--input_file", type=str, default="data/raw/Estacoes_SMP.csv")
    parser.add_argument("--stations_file", type=str, default="data/processed/estacoes_unicas.csv")
    parser.add_argument("--graph_file", type=str, default="data/processed/grafo.graphml")
    parser.add_argument("--map_file", type=str, default="reports/mapa_conexoes.png")
    return parser

def plot_network_map(G, output_path: str, title: str = "Topologia Waxman"):
    """Plota os vértices posicionados geograficamente e diferencia enlaces Waxman vs Fallback."""
    fig, ax = plt.subplots(figsize=(12, 12))

    # 1. Extrai a posição (lon, lat) de cada nó do grafo
    pos = {node: data["pos"] for node, data in G.nodes(data=True)}

    # 2. Separa segmentos de linha para Waxman (azul) e Fallback (vermelho tracejado)
    waxman_segs = [
        [pos[u], pos[v]]
        for u, v, d in G.edges(data=True)
        if d.get("tipo") != "fallback_mais_proxima"
    ]
    fallback_segs = [
        [pos[u], pos[v]]
        for u, v, d in G.edges(data=True)
        if d.get("tipo") == "fallback_mais_proxima"
    ]

    # 3. Ajuste automático de tamanho visual (cidade pequena vs Brasil todo)
    n_nos = G.number_of_nodes()
    is_large = n_nos > 5000
    node_size = 0.5 if is_large else 25
    edge_width = 0.2 if is_large else 0.8
    edge_alpha = 0.15 if is_large else 0.35

    # 4. Desenha as arestas rapidamente usando LineCollection
    if waxman_segs:
        ax.add_collection(
            LineCollection(
                waxman_segs,
                colors="#1f77b4",
                linewidths=edge_width,
                alpha=edge_alpha,
                rasterized=True,
                label="Enlace Waxman",
            )
        )
    if fallback_segs:
        ax.add_collection(
            LineCollection(
                fallback_segs,
                colors="#d62728",
                linewidths=edge_width * 2,
                linestyles="--",
                alpha=0.85,
                rasterized=True,
                label="Fallback (mais próxima)",
            )
        )

    # 5. Desenha os vértices (torres) sobre as linhas
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.scatter(
        xs, ys,
        s=node_size,
        c="#2ca02c",
        alpha=0.85,
        zorder=5,
        rasterized=True,
        label=f"Torres ({n_nos:,})",
    )

    ax.set_title(f"{title}\n({n_nos:,} nós | {G.number_of_edges():,} arestas)", fontsize=14, pad=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", markerscale=1 if not is_large else 5)

    # 6. Cria a pasta de saída e salva em alta resolução
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualização] Mapa salvo em: {out}")

def main():
    args = argument_parser().parse_args()

    # 1. Processamento (usa cache se já existir, senão processa do zero)
    if Path(args.stations_file).exists():
        print(f"Carregando estações já processadas de: {args.stations_file}")
        df_unique = pd.read_csv(args.stations_file)
    else:
        df_raw = load_raw_data(args.input_file)
        df_clustered = assign_spatial_clusters(df_raw, radius_meters=30.0)
        df_unique = group_unique_station(df_clustered)
        save_data(df_unique, args.stations_file)

    # 3. Gerar o Grafo (Modelo de Waxman + Fallback)
    G = generate_waxman_graph(df_unique, seed=42)
    save_graph(G, args.graph_file)

    # 4. Gerar e salvar o mapa conectado
    plot_network_map(G, args.map_file, title="Topologia Waxman")

if __name__ == "__main__":
    main()
