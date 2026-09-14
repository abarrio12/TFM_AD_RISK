"""
Análisis Exploratorio de Datos (EDA).

Punto del índice que cubre: 4.2.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin esto, en Windows puede intentar abrir una ventana con Tk y fallar si Tcl/Tk no está bien instalado -- nunca hace falta ventana, solo guardamos a fichero
import matplotlib.pyplot as plt
import pandas as pd

COLUMNAS_MODELO_EDA = [
    "edad", "mmse", "hipocampo_mm3",
    "tau_pg_ml", "abeta42_pg_ml",
    "ptau217_plasma_pg_ml", "ratio_ab42_ab40_plasma",
]

# Siempre <raiz_del_proyecto>/resultados, sin importar desde qué carpeta se
# lance el script -- antes era una ruta relativa ("resultados" a secas),
# así que aparecía en un sitio distinto según desde dónde se ejecutara.
CARPETA_RESULTADOS_DEFECTO = str(Path(__file__).resolve().parent.parent.parent / "resultados")


def resumen_estadistico(df: pd.DataFrame) -> pd.DataFrame:
    """Distribución de diagnósticos y estadísticos descriptivos por variable."""
    print("--- Distribución de diagnósticos ---")
    conteo = df["diagnostico"].value_counts()
    porcentaje = (conteo / len(df) * 100).round(1)
    print(pd.DataFrame({"pacientes": conteo, "%": porcentaje}))
    print("\n--- Estadísticos numéricos ---")
    resumen = df.describe()
    print(resumen)
    return resumen


def completitud_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Qué porcentaje de pacientes tiene cada variable disponible (no NaN).

    Es la tabla que justifica la decisión de modelado en 4.3.2: con huecos
    tan dispares entre variables (edad/MMSE casi al 100%, biomarcadores de
    plasma por debajo del 25%), exigir todas las variables completas
    descartaría a la mayoría de los pacientes.
    """
    filas = []
    for col in COLUMNAS_MODELO_EDA:
        if col in df.columns:
            filas.append({"variable": col, "completitud_%": round(df[col].notna().mean() * 100, 1)})
    tabla = pd.DataFrame(filas).sort_values("completitud_%", ascending=False)
    print("--- Completitud de variables ---")
    print(tabla.to_string(index=False))
    return tabla


def graficar_distribuciones(df: pd.DataFrame, carpeta_salida: str = CARPETA_RESULTADOS_DEFECTO) -> None:
    """Guarda histogramas de las variables clave, separados por diagnóstico."""
    Path(carpeta_salida).mkdir(exist_ok=True)
    variables = [v for v in ["mmse", "tau_pg_ml", "abeta42_pg_ml", "hipocampo_mm3"] if v in df.columns]
    colores = {"CN": "#2E7D32", "MCI": "#F9A825", "AD": "#C62828"}

    fig, ejes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, var in zip(ejes.flat, variables):
        for grupo, color in colores.items():
            subset = df[df["diagnostico"] == grupo][var].dropna()
            if len(subset):
                ax.hist(subset, bins=15, alpha=0.5, label=grupo, color=color)
        ax.set_title(var)
        ax.legend(fontsize=8)
    fig.tight_layout()
    ruta = Path(carpeta_salida) / "eda_distribuciones.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"Gráfico guardado en {ruta}")


def graficar_completitud(df: pd.DataFrame, carpeta_salida: str = CARPETA_RESULTADOS_DEFECTO) -> None:
    """Gráfico de barras horizontal con el % de completitud de cada variable."""
    Path(carpeta_salida).mkdir(exist_ok=True)
    tabla = completitud_variables(df).sort_values("completitud_%")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(tabla["variable"], tabla["completitud_%"], color="#1565C0")
    ax.set_xlabel("% de pacientes con dato disponible")
    ax.set_xlim(0, 100)
    for i, valor in enumerate(tabla["completitud_%"]):
        ax.text(valor + 1, i, f"{valor}%", va="center", fontsize=9)
    fig.tight_layout()
    ruta = Path(carpeta_salida) / "eda_completitud.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"Gráfico guardado en {ruta}")


def graficar_tendencia_mmse(ruta_adni_completo: str, carpeta_salida: str = CARPETA_RESULTADOS_DEFECTO, minimo_pacientes: int = 10) -> None:
    """Evolución del MMSE medio por grupo diagnóstico basal, a lo largo de
    las visitas de seguimiento reales de ADNI -- descriptivo, no una
    proyección individual (esa distinción se explicita también en el
    dashboard, sección 5.3). Puntos con menos de minimo_pacientes se
    descartan: con muy pocos pacientes, la media es ruido, no tendencia.
    """
    from datos import cargar_datos_longitudinales

    Path(carpeta_salida).mkdir(parents=True, exist_ok=True)
    df_long = cargar_datos_longitudinales(ruta_adni_completo)
    df_long = df_long[df_long["diagnostico_grupo"] != "Desconocido"]
    agregado = (
        df_long.dropna(subset=["mmse"])
        .groupby(["diagnostico_grupo", "meses"])
        .agg(mmse_medio=("mmse", "mean"), n_pacientes=("paciente_id", "nunique"))
        .reset_index()
    )
    agregado = agregado[agregado["n_pacientes"] >= minimo_pacientes].sort_values(["diagnostico_grupo", "meses"])

    colores = {"CN": "#2E7D32", "MCI": "#F9A825", "AD": "#C62828"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for grupo in ["CN", "MCI", "AD"]:
        sub = agregado[agregado["diagnostico_grupo"] == grupo]
        if len(sub):
            ax.plot(sub["meses"], sub["mmse_medio"], marker="o", label=grupo, color=colores[grupo])
    ax.set_xlabel("Meses desde la visita basal")
    ax.set_ylabel("MMSE medio")
    ax.set_title("Evolución del MMSE por grupo diagnóstico basal (visitas reales de seguimiento)")
    ax.legend(title="Diagnóstico basal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ruta = Path(carpeta_salida) / "tendencia_mmse.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"Gráfico guardado en {ruta}")


if __name__ == "__main__":
    from datos import generar_datos_sinteticos
    df = generar_datos_sinteticos()
    resumen_estadistico(df)
    completitud_variables(df)
    graficar_distribuciones(df)
    graficar_completitud(df)
