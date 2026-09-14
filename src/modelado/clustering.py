"""
Submódulo de estratificación: clustering para descubrir subgrupos de pacientes.
La idea es ver si las clases están ya divididas antes incluso de entrenar el 
modelo.
Punto del índice que cubre: 4.3.1.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin ventana -- solo guardamos a fichero, ver eda.py
import matplotlib.pyplot as plt
import pandas as pd

CARPETA_RESULTADOS_DEFECTO = str(Path(__file__).resolve().parent.parent.parent / "resultados")
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

VARIABLES_CLUSTERING = ["edad", "mmse", "tau_pg_ml", "abeta42_pg_ml", "hipocampo_mm3"]


def ejecutar_clustering(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Aplica KMeans sobre las variables numéricas (estandarizadas) y
    devuelve el DataFrame con una columna 'cluster' añadida.
    """
    datos = df[VARIABLES_CLUSTERING].dropna()
    X = StandardScaler().fit_transform(datos)

    modelo = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    etiquetas = modelo.fit_predict(X)

    df_resultado = df.loc[datos.index].copy()
    df_resultado["cluster"] = etiquetas
    return df_resultado


def proyectar_2d(df: pd.DataFrame):
    """Proyección a 2D de las variables de clustering: UMAP si está
    instalado (como en el mockup original), si no PCA. Devuelve
    (coordenadas, etiqueta_eje). Función compartida entre el gráfico
    estático de esta sección y el panel interactivo del dashboard (5.1),
    para no mantener la misma lógica en dos sitios.
    """
    X = StandardScaler().fit_transform(df[VARIABLES_CLUSTERING])
    try:
        import umap
        return umap.UMAP(random_state=42).fit_transform(X), "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(X), "PCA"


def graficar_clusters(df_con_clusters: pd.DataFrame, carpeta_salida: str = CARPETA_RESULTADOS_DEFECTO) -> None:
    """Proyección 2D de los clusters. Usa UMAP si está instalado (como en
    el mockup original del dashboard); si no, cae a PCA para no bloquear
    el pipeline por una dependencia opcional.
    """
    Path(carpeta_salida).mkdir(exist_ok=True)
    proyeccion, etiqueta_eje = proyectar_2d(df_con_clusters)

    fig, ax = plt.subplots(figsize=(7, 6))
    dispersion = ax.scatter(proyeccion[:, 0], proyeccion[:, 1], c=df_con_clusters["cluster"], cmap="tab10", s=25, alpha=0.8)
    ax.set_xlabel(f"{etiqueta_eje} 1")
    ax.set_ylabel(f"{etiqueta_eje} 2")
    ax.set_title("Subgrupos descubiertos por clustering")
    fig.colorbar(dispersion, ax=ax, label="Cluster")
    fig.tight_layout()
    ruta = Path(carpeta_salida) / "clustering_proyeccion.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"Gráfico guardado en {ruta}")


if __name__ == "__main__":
    from datos import generar_datos_sinteticos
    df = generar_datos_sinteticos()
    df_clusters = ejecutar_clustering(df)
    print(df_clusters.groupby("cluster")["diagnostico"].value_counts())
    graficar_clusters(df_clusters)
