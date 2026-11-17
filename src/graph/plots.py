import json
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LogNorm

# Mapeamento oficial UF -> prefixo de 2 dígitos do código IBGE
UF_TO_IBGE_PREFIX = {
    "RO": "11", "AC": "12", "AM": "13", "RR": "14", "PA": "15", "AP": "16", "TO": "17",
    "MA": "21", "PI": "22", "CE": "23", "RN": "24", "PB": "25", "PE": "26", "AL": "27",
    "SE": "28", "BA": "29", "MG": "31", "ES": "32", "RJ": "33", "SP": "35", "PR": "41",
    "SC": "42", "RS": "43", "MS": "50", "MT": "51", "GO": "52", "DF": "53",
}


def _save_figure(output_path: str):
    """Cria a pasta de destino e salva a figura atual em alta resolução."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Mapa salvo em: {out}")


def _load_ibge_geojson(geojson_path: str = "data/raw/malhas_municipios_br.geojson") -> dict:
    """Carrega a malha municipal do IBGE (GeoJSON) do disco ou baixa da API oficial se não existir."""
    path = Path(geojson_path)
    if not path.exists():
        print("[Visualization] Baixando malha municipal do IBGE...")
        path.parent.mkdir(parents=True, exist_ok=True)
        url = (
            "https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR"
            "?formato=application/vnd.geo+json&qualidade=intermediaria&intrarregiao=municipio"
        )
        urllib.request.urlretrieve(url, path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_municipality_polygons(geojson: dict, uf_filter: str | None = None):
    """
    Extrai os anéis exteriores de cada município do GeoJSON.
    Retorna lista de tuplas (cod_ibge, coords_array).
    """
    prefix = UF_TO_IBGE_PREFIX.get(uf_filter.upper()) if uf_filter and uf_filter.upper() != "BR" else None
    polys = []

    for feat in geojson.get("features", []):
        cod = str(feat.get("properties", {}).get("codarea", ""))
        if prefix and not cod.startswith(prefix):
            continue

        geom = feat.get("geometry")
        if not geom:
            continue

        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if gtype == "Polygon":
            for ring in coords[:1]:
                polys.append((cod, np.asarray(ring, dtype=float)))
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly[:1]:
                    polys.append((cod, np.asarray(ring, dtype=float)))

    return polys


def _draw_choropleth(
    ax,
    fig,
    polys: list[tuple[str, np.ndarray]],
    value_by_cod: dict[str, float],
    cmap_name: str,
    cbar_label: str,
    poly_alpha: float = 0.9,
):
    """Desenha os polígonos dos municípios coloridos em escala logarítmica."""
    positive_vals = [v for v in value_by_cod.values() if v is not None and v > 0]
    vmin = max(min(positive_vals), 1.0) if positive_vals else 1.0
    vmax = max(max(positive_vals), vmin * 10) if positive_vals else 10.0

    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    verts_with_val = []
    vals = []
    verts_missing = []

    for cod, ring in polys:
        val = value_by_cod.get(cod, 0.0)
        if val is not None and val > 0:
            verts_with_val.append(ring)
            vals.append(np.clip(val, vmin, vmax))
        else:
            verts_missing.append(ring)

    lw = 0.08 if len(polys) > 1500 else 0.25

    if verts_missing:
        missing_col = PolyCollection(
            verts_missing,
            facecolors="#e0e0e0",
            edgecolors="#888888",
            linewidths=lw,
            alpha=poly_alpha,
            zorder=1,
            rasterized=True,
        )
        ax.add_collection(missing_col)

    if verts_with_val:
        col = PolyCollection(
            verts_with_val,
            array=np.asarray(vals, dtype=float),
            cmap=cmap,
            norm=norm,
            edgecolors="#444444",
            linewidths=lw,
            alpha=poly_alpha,
            zorder=2,
            rasterized=True,
        )
        ax.add_collection(col)
        cbar = fig.colorbar(col, ax=ax, fraction=0.035, pad=0.03)
        cbar.set_label(cbar_label, fontsize=11)

    ax.autoscale_view()


def _draw_graph_edges(ax, G: nx.Graph, is_large: bool):
    """Desenha as arestas Waxman e Fallback no eixo fornecido."""
    pos = {node: data["pos"] for node, data in G.nodes(data=True)}

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

    edge_width = 0.2 if is_large else 0.8
    edge_alpha = 0.15 if is_large else 0.35

    if waxman_segs:
        ax.add_collection(
            LineCollection(
                waxman_segs,
                colors="#1f77b4",
                linewidths=edge_width,
                alpha=edge_alpha,
                zorder=3,
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
                zorder=4,
                rasterized=True,
                label="Fallback (mais próxima)",
            )
        )
    return pos


def plot_network_map(G: nx.Graph, output_path: str, title: str = "Topologia Waxman"):
    """Plota os vértices posicionados geograficamente e diferencia enlaces Waxman vs Fallback."""
    fig, ax = plt.subplots(figsize=(12, 12))

    n_nos = G.number_of_nodes()
    is_large = n_nos > 5000
    node_size = 0.5 if is_large else 25

    pos = _draw_graph_edges(ax, G, is_large)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.scatter(
        xs,
        ys,
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

    _save_figure(output_path)


def plot_population_map(
    df_stations: pd.DataFrame,
    output_path: str,
    uf_filter: str = "MG",
    title: str = "População por Município",
):
    """Plota o formato dos municípios coloridos pela sua população estimada (escala logarítmica)."""
    fig, ax = plt.subplots(figsize=(12, 12))

    geojson = _load_ibge_geojson()
    polys = _extract_municipality_polygons(geojson, uf_filter=uf_filter)

    df_mun = (
        df_stations[df_stations["cod_ibge"] != ""]
        .groupby("cod_ibge", as_index=False)["populacao"]
        .first()
    )
    pop_by_cod = dict(zip(df_mun["cod_ibge"].astype(str), df_mun["populacao"].astype(float)))

    _draw_choropleth(
        ax,
        fig,
        polys,
        pop_by_cod,
        cmap_name="viridis",
        cbar_label="População do Município (habitantes - escala log)",
        poly_alpha=0.92,
    )

    n_mun = len(pop_by_cod)
    ax.set_title(f"{title}\n({n_mun:,} municípios)", fontsize=14, pad=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)

    _save_figure(output_path)


def plot_demands_map(
    df_stations: pd.DataFrame,
    vetor_demandas: np.ndarray,
    output_path: str,
    uf_filter: str = "MG",
    title: str = "Demanda por Município (Habitantes por Antena)",
):
    """Plota o formato dos municípios coloridos pela demanda (habitantes por antena)."""
    fig, ax = plt.subplots(figsize=(12, 12))

    geojson = _load_ibge_geojson()
    polys = _extract_municipality_polygons(geojson, uf_filter=uf_filter)

    df_plot = df_stations.sort_values("node_id").copy()
    df_plot["demanda"] = vetor_demandas
    df_mun = (
        df_plot[df_plot["cod_ibge"] != ""]
        .groupby("cod_ibge", as_index=False)["demanda"]
        .first()
    )
    demand_by_cod = dict(zip(df_mun["cod_ibge"].astype(str), df_mun["demanda"].astype(float)))

    _draw_choropleth(
        ax,
        fig,
        polys,
        demand_by_cod,
        cmap_name="plasma",
        cbar_label="Demanda: Habitantes por Antena (escala log)",
        poly_alpha=0.92,
    )

    n_mun = len(demand_by_cod)
    ax.set_title(f"{title}\n({n_mun:,} municípios | {len(df_stations):,} torres)", fontsize=14, pad=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)

    _save_figure(output_path)


def plot_demands_and_connections_map(
    G: nx.Graph,
    df_stations: pd.DataFrame,
    vetor_demandas: np.ndarray,
    output_path: str,
    uf_filter: str = "MG",
    title: str = "Topologia Waxman sobre Demanda por Município",
):
    """Combina o mapa coroplético de demandas por município ao fundo com as conexões e torres por cima."""
    fig, ax = plt.subplots(figsize=(12, 12))

    geojson = _load_ibge_geojson()
    polys = _extract_municipality_polygons(geojson, uf_filter=uf_filter)

    df_plot = df_stations.sort_values("node_id").copy()
    df_plot["demanda"] = vetor_demandas
    df_mun = (
        df_plot[df_plot["cod_ibge"] != ""]
        .groupby("cod_ibge", as_index=False)["demanda"]
        .first()
    )
    demand_by_cod = dict(zip(df_mun["cod_ibge"].astype(str), df_mun["demanda"].astype(float)))

    # 1. Desenha os polígonos dos municípios ao fundo com transparência suave
    _draw_choropleth(
        ax,
        fig,
        polys,
        demand_by_cod,
        cmap_name="YlOrRd",
        cbar_label="Demanda do Município: Habitantes por Antena (escala log)",
        poly_alpha=0.65,
    )

    # 2. Desenha as conexões e torres por cima dos municípios
    n_nos = G.number_of_nodes()
    is_large = n_nos > 5000
    node_size = 0.5 if is_large else 20

    pos = _draw_graph_edges(ax, G, is_large)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.scatter(
        xs,
        ys,
        s=node_size,
        c="#1b7837",
        alpha=0.85,
        zorder=5,
        rasterized=True,
        label=f"Torres ({n_nos:,})",
    )

    ax.set_title(
        f"{title}\n({n_nos:,} nós | {G.number_of_edges():,} arestas)",
        fontsize=14,
        pad=12,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", markerscale=1 if not is_large else 5)

    _save_figure(output_path)


def generate_all_plots(
    G: nx.Graph,
    df_stations: pd.DataFrame,
    vetor_demandas: np.ndarray,
    scope_tag: str,
    scope_label: str,
    map_file: str | None = None,
):
    """Gera os 4 mapas analíticos na pasta reports/."""
    conexoes_path = map_file or f"reports/mapa_conexoes_{scope_tag}.png"
    populacao_path = f"reports/mapa_populacao_{scope_tag}.png"
    demandas_path = f"reports/mapa_demandas_{scope_tag}.png"
    combinado_path = f"reports/mapa_conexoes_demandas_{scope_tag}.png"

    uf_filter = scope_tag.upper()

    plot_network_map(G, conexoes_path, title=f"Topologia Waxman - {scope_label}")
    plot_population_map(
        df_stations,
        populacao_path,
        uf_filter=uf_filter,
        title=f"População por Município - {scope_label}",
    )
    plot_demands_map(
        df_stations,
        vetor_demandas,
        demandas_path,
        uf_filter=uf_filter,
        title=f"Demanda (Habitantes/Antena) por Município - {scope_label}",
    )
    plot_demands_and_connections_map(
        G,
        df_stations,
        vetor_demandas,
        combinado_path,
        uf_filter=uf_filter,
        title=f"Topologia Waxman + Demanda por Município - {scope_label}",
    )
