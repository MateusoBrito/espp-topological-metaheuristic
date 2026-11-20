import argparse
import pandas as pd
from pathlib import Path
from src.graph.data_processing import load_raw_data, assign_spatial_clusters, group_unique_station, save_data, export_distance_matrix
from src.graph.generate_connections import generate_waxman_graph, save_graph
from src.graph.generate_demands import generate_demands, save_demands
from src.graph.plots import generate_all_plots

def argument_parser():
    parser = argparse.ArgumentParser(description="Processa estações, gera conexões e plota o mapa.")
    parser.add_argument("--input_file", type=str, default="data/raw/Estacoes_SMP.csv")
    parser.add_argument("--pop_file", type=str, default="data/raw/estimativa_dou_2026.xlsx", help="Arquivo Excel com estimativa populacional do IBGE")
    parser.add_argument("--stations_file", type=str, default="data/processed/estacoes_unicas.csv")
    parser.add_argument("--uf",type=str,default="MG",help="Sigla da UF para filtrar (ex: MG, SP) ou 'BR' para considerar o Brasil todo (padrão: MG).",)
    parser.add_argument("--graph_file", type=str, default=None, help="Caminho de saída do grafo (gerado dinamicamente por UF se omitido)")
    parser.add_argument("--map_file", type=str, default=None, help="Caminho de saída do mapa (gerado dinamicamente por UF se omitido)")
    parser.add_argument("--matrix_file", type=str, default=None, help="Caminho de saída da matriz de distâncias (gerado dinamicamente por UF se omitido)")
    parser.add_argument("--demands_file", type=str, default=None, help="Caminho de saída do vetor de demandas (gerado dinamicamente por UF se omitido)")
    parser.add_argument("--alpha", type=float, default=0.4, help="Parâmetro alpha do modelo de Waxman")
    parser.add_argument("--beta", type=float, default=None, help="Parâmetro beta do modelo de Waxman (padrão automático: 0.015 para UF ou 0.0003 para BR)")
    parser.add_argument("--max_dist_km", type=float, default=None, help="Distância máxima de busca (km). Se omitido, é calculada dinamicamente.")
    parser.add_argument("--min_prob_ratio", type=float, default=0.0025, help="Limiar de probabilidade residual epsilon")
    return parser

def main():
    args = argument_parser().parse_args()

    # 1. Processamento (usa cache se já existir, senão processa do zero)
    if Path(args.stations_file).exists():
        print(f"Carregando estações já processadas de: {args.stations_file}")
        df_unique = pd.read_csv(args.stations_file, skipinitialspace=True)
        df_unique.columns = df_unique.columns.str.strip()
        for col in df_unique.select_dtypes(include="object").columns:
            df_unique[col] = df_unique[col].astype(str).str.strip()
    else:
        df_raw = load_raw_data(args.input_file)
        df_clustered = assign_spatial_clusters(df_raw, radius_meters=30.0)
        df_unique = group_unique_station(df_clustered)
        save_data(df_unique, args.stations_file)

    # 2. Filtra por UF (se diferente de 'BR') ou considera o Brasil inteiro
    uf_param = args.uf.strip().upper()
    if uf_param != "BR":
        df_unique = df_unique[df_unique["uf"].str.upper() == uf_param].copy()
        df_unique = df_unique.reset_index(drop=True)
        df_unique["node_id"] = df_unique.index
        scope_tag = uf_param.lower()
        scope_label = uf_param
        print(f"[Filter] UF={uf_param} selecionada: {len(df_unique):,} estações únicas.")
    else:
        df_unique = df_unique.reset_index(drop=True)
        df_unique["node_id"] = df_unique.index
        scope_tag = "br"
        scope_label = "Brasil"
        print(f"[Filter] Sem filtro de UF (BR - Brasil completo): {len(df_unique):,} estações únicas.")

    graph_file = args.graph_file or f"data/processed/grafo_{scope_tag}.graphml"
    matrix_file = args.matrix_file or f"data/processed/distance_matrix_{scope_tag}.bin"
    demands_file = args.demands_file or f"data/processed/vetor_demandas_{scope_tag}.bin"
    beta = args.beta if args.beta is not None else (0.0003 if uf_param == "BR" else 0.015)

    # 3. Gerar o Grafo (Modelo de Waxman + Fallback)
    G = generate_waxman_graph(
        df_unique,
        alpha=args.alpha,
        beta=beta,
        max_dist_km=args.max_dist_km,
        min_prob_ratio=args.min_prob_ratio,
        biggest_component_only=True,
        seed=42,
    )

    save_graph(G, graph_file)

    # 4. Gerar e exportar o vetor de demandas populacionais
    df_populacao = pd.read_excel(args.pop_file, sheet_name="municípios", skiprows=1)
    vetor_demandas = generate_demands(df_unique, df_populacao)
    save_demands(vetor_demandas, output_path=demands_file)

    # 5. Gerar e salvar os mapas (conexões, população, demandas e combinado)
    generate_all_plots(
        G,
        df_unique,
        vetor_demandas,
        scope_tag=scope_tag,
        scope_label=scope_label,
        map_file=args.map_file,
    )

    # 6. Exportar a matriz de distâncias
    #export_distance_matrix(G, output_path=matrix_file)

if __name__ == "__main__":
    main()
