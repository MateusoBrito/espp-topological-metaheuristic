import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path

def text_normalize(texto):
    """Remove acentos, hifens, espaços extras e converte para maiúsculo."""
    if not isinstance(texto, str):
        return ""
    s = unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("utf-8").strip().upper()
    return " ".join(s.replace("-", " ").replace("'", " ").split())

def generate_demands(
    df_estacoes: pd.DataFrame,
    df_populacao: pd.DataFrame,
) -> np.ndarray:
    print("[Demands] Iniciando cálculo populacional por município...")

    # 1. Trata a coluna de população estimada e código IBGE de 7 dígitos
    df_populacao = df_populacao.dropna(subset=["POPULAÇÃO ESTIMADA"]).copy()
    df_populacao["populacao"] = df_populacao["POPULAÇÃO ESTIMADA"].astype(int)
    df_populacao["cod_ibge"] = (
        df_populacao["COD. UF"].astype(int).astype(str).str.zfill(2)
        + df_populacao["COD. MUNIC"].astype(int).astype(str).str.zfill(5)
    )

    # 2. Agrega antenas por município e UF (evita misturar cidades homônimas de estados diferentes)
    df_antenas_mun = (
        df_estacoes.groupby(["municipio", "uf"], as_index=False)
        .agg(total_antenas=("qtd_antenas", "sum"))
    )

    # 3. Cria as chaves de busca padronizadas e faz o Merge
    df_antenas_mun["chave_busca"] = (
        df_antenas_mun["municipio"].apply(text_normalize)
        + "_"
        + df_antenas_mun["uf"].apply(text_normalize)
    )

    df_populacao["chave_busca"] = (
        df_populacao["NOME DO MUNICÍPIO"].apply(text_normalize)
        + "_"
        + df_populacao["UF"].apply(text_normalize)
    )

    df_merged = pd.merge(
        df_antenas_mun,
        df_populacao[["chave_busca", "cod_ibge", "populacao"]],
        on="chave_busca",
        how="inner",
    )

    # 4. Calcula a heurística (População / Total de Antenas)
    df_merged["habitantes_por_antena"] = df_merged["populacao"] / df_merged["total_antenas"]

    # 5. Mescla de volta para o nível do vértice (node_id) usando municipio + uf
    df_final = pd.merge(
        df_estacoes,
        df_merged[["municipio", "uf", "cod_ibge", "populacao", "habitantes_por_antena"]],
        on=["municipio", "uf"],
        how="left",
    )

    # Verifica quantas estações ficaram sem cobertura populacional antes de preencher com 0
    sem_cobertura = int(df_final["habitantes_por_antena"].isna().sum())
    print(f"[Demands] Estações sem cobertura populacional (preenchidas com 0): {sem_cobertura:,} de {len(df_final):,}")

    # Preenche casos sem cruzamento com 0 e garante a ordem dos nós do grafo
    df_final["cod_ibge"] = df_final["cod_ibge"].fillna("")
    df_final["populacao"] = df_final["populacao"].fillna(0).astype(int)
    df_final["habitantes_por_antena"] = df_final["habitantes_por_antena"].fillna(0.0)
    df_final = df_final.sort_values("node_id")

    # Anexa as colunas calculadas no DataFrame de estações (alinhado por node_id)
    df_estacoes["cod_ibge"] = df_final["cod_ibge"].values
    df_estacoes["populacao"] = df_final["populacao"].values
    df_estacoes["habitantes_por_antena"] = df_final["habitantes_por_antena"].values

    return df_final["habitantes_por_antena"].values


def save_demands(
    vetor_demandas: np.ndarray,
    output_path: str = "data/processed/vetor_demandas.bin",
):
    """Salva o vetor de demandas em arquivo binário (int32 N seguido de N float64)."""
    N = len(vetor_demandas)

    caminho_saida = Path(output_path)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    with open(caminho_saida, "wb") as f:
        # Escreve dimensão (N) e depois os floats sequenciais
        f.write(np.array([N], dtype=np.int32).tobytes())
        f.write(vetor_demandas.astype(np.float64).tobytes())

    print(f"[Demands] Sucesso! Vetor exportado em {caminho_saida} com {N:,} nós ({caminho_saida.stat().st_size / 1024:.2f} KB).")
