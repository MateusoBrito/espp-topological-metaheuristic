import argparse
import pandas as pd
from pathlib import Path
from graph.data_processing import load_raw_data, group_unique_station, save_data

def argument_parser():
    parser = argparse.ArgumentParser(description="Processa dados de estações e gera conexões.")
    parser.add_argument("--input_file", type=str, help="Caminho para o arquivo CSV de entrada.", default="data/Estacoes_SMP.csv")
    parser.add_argument("--output_file", type=str, help="Caminho para o arquivo CSV de saída.", default="data/estacoes_unicas.csv")
    return parser

def main():
    parser = argument_parser()
    args = parser.parse_args()

    df_raw = load_raw_data(args.input_file)
    df_unique_stations = group_unique_station(df_raw)
    save_data(df_unique_stations, args.output_file)

if __name__ == "__main__":
    main()
