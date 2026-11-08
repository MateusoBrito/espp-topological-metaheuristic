import pandas as pd
from pathlib import Path

DEFAULT_COLUMNS = [
    "Número Estação", "Latitude decimal", "Longitude decimal",
    "Empresa Estação", "Geração", "Tecnologia", "Município-UF"
]

def load_raw_data(file_path: str, columns: list = None):
    """Carrega os dados brutos de um arquivo CSV."""
    if columns is None:
        columns = DEFAULT_COLUMNS
    df = pd.read_csv(file_path, sep = ";", encoding="utf-8-sig", usecols=columns)
    return df

def group_unique_station(df: pd.DataFrame):
    """Agrupa por latitude/longitude consolidando operadoras e tecnologias presentes em cada ponto."""
    df_unicas = (
        df.dropna(subset=["Latitude decimal", "Longitude decimal"])
        .groupby(["Latitude decimal", "Longitude decimal"])
        .agg(
            id_local=("Número Estação", "first"),
            qtd_antenas=("Número Estação", "count"),
            operadoras=("Empresa Estação", lambda x: ", ".join(sorted(set(x.dropna())))),
            tecnologias=("Tecnologia", lambda x: ", ".join(sorted(set(x.dropna())))),
            tem_2g=("Geração", lambda x: "2G" in set(x)),
            tem_3g=("Geração", lambda x: "3G" in set(x)),
            tem_4g=("Geração", lambda x: "4G" in set(x)),
            tem_5g=("Geração", lambda x: "5G" in set(x)),
        )
        .reset_index()
    )
    return df_unicas

def save_data(df: pd.DataFrame, output_path: str):
    """Salva o DataFrame em um arquivo CSV no caminho especificado."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    df.to_csv(output_path, index=False)
    print(f"DataFrame salvo em: {output_path}")