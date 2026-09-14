"""
Carga y estandarización de datos para el modelo predictivo (sección 4).

Punto del índice que cubre: 4.1 (selección y carga del dataset).

Dos modos de uso:
- generar_datos_sinteticos(): datos ficticios con la MISMA estructura que
  tendrán los reales, para poder construir y probar todo el pipeline de
  4.2 a 4.4 ANTES de tener acceso a los datos reales.
- cargar_datos_reales(ruta_csv, fuente="adni"): carga el CSV de ADNI
  (ADNIMERGE.csv, o el resultado de unir las tablas sueltas por RID).

Ambos casos dejan el resultado con los mismos nombres de columna que usa
el resto del pipeline y que ya coinciden con los campos de
FichaPaciente (sección 3.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Adaptadores de columnas: nombre real (izquierda) -> nombre estándar (derecha)
# --------------------------------------------------------------------------

# ADNI (ADNIMERGE.csv): TAU, PTAU y ABETA están en ADNIMERGE 
# --> no hace falta ningún fichero de biomarcadores aparte.
#   - AGE, MMSE, Hippocampus, DX_bl: alta completitud en
#     visita basal (>99% salvo Hippocampus, 85.6%).
#   - APOE4: conteo de alelos e4 (0/1/2), 91% disponible.
#   - TAU/PTAU/ABETA: 50% disponible en basal (1215 de 2430 sujetos)
#     --> bastante menos que el resto, pero por encima del umbral de
#     riesgo (300-400) que marca la guía del máster. Vienen a veces como
#     texto (">1700", "<200" = límite del ensayo) en vez de número; se
#     limpian con _limpiar_valor_censurado() más abajo.
MAPEO_COLUMNAS_ADNI = {
    "RID": "paciente_id",
    "AGE": "edad",
    "PTGENDER": "sexo",
    "DX_bl": "diagnostico_texto",
    "MMSE": "mmse",
    "APOE4": "apoe4_conteo",
    "Hippocampus": "hipocampo_mm3",
    "TAU": "tau_pg_ml",
    "PTAU": "p_tau_pg_ml",
    "ABETA": "abeta42_pg_ml",
}


def _limpiar_valor_censurado(valor):
    """TAU/PTAU/ABETA en ADNI a veces vienen como '>1700' o '<200' (límite
    de detección del ensayo de laboratorio) en vez de un número. Se quita
    el símbolo y se usa el valor límite como aproximación  es la
    convención estándar, no la medida exacta (ver 4.2/6.2).
    """
    if pd.isna(valor):
        return np.nan
    texto = str(valor).strip()
    if texto.startswith((">", "<")):
        texto = texto[1:]
    try:
        return float(texto)
    except ValueError:
        return np.nan


def apoe4_conteo_a_texto(conteo) -> str:
    """En ADNIMERGE, APOE4 es el conteo de alelos e4 (0, 1 o 2), no texto."""
    if pd.isna(conteo):
        return "Desconocido"
    return "Positivo" if conteo >= 1 else "Negativo"


# Biomarcadores en PLASMA (UPENN_PLASMA_FUJIREBIO_QUANTERIX): un fichero
# aparte de ADNIMERGE, con muy poco solape de sujetos respecto al LCR
# (63 en común, de 1215 con LCR y 558 con plasma en visita basal) -- por
# eso se añaden como columnas nuevas, no sustituyendo a tau/abeta de LCR.
MAPEO_COLUMNAS_ADNI_PLASMA = {
    "RID": "paciente_id",
    "pT217_F": "ptau217_plasma_pg_ml",
    "AB42_AB40_F": "ratio_ab42_ab40_plasma",
}


def amiloide_plasma_estado(ratio_ab42_ab40) -> str | None:
    """Clasifica el estado amiloide a partir del ratio Aβ42/Aβ40 en
    plasma, con los puntos de corte validados por el UPENN Biomarker Core
    (Cousins et al. 2025, cohorte ADRC de Penn): negativo >=0.1053,
    positivo <=0.0820, intermedio en medio. Útil como hallazgo adicional
    o para contrastar con el diagnóstico -- citable en 4.1/6.1.
    """
    if pd.isna(ratio_ab42_ab40):
        return None
    if ratio_ab42_ab40 >= 0.1053:
        return "Negativo"
    if ratio_ab42_ab40 <= 0.0820:
        return "Positivo"
    return "Intermedio"


def cdr_a_diagnostico(cdr: float) -> str:
    """CDR es la escala estándar de estadificación en ADNI.
    Mapeo convencional en la literatura: 0 = CN, 0.5 = MCI, >=1 = AD.
    """
    if pd.isna(cdr):
        return "Desconocido"
    if cdr == 0:
        return "CN"
    if cdr == 0.5:
        return "MCI"
    return "AD"


def diagnostico_desde_adni(valor_dx) -> str:
    """DX_bl en ADNIMERGE tiene 5 categorías reales: CN, SMC, EMCI, LMCI, AD.
    Nuestro esquema es de 3 clases:
      - EMCI, LMCI -> MCI (deterioro leve, con o sin más detalle de fase)
      - CN, AD -> igual
      - SMC (queja subjetiva de memoria, cognición objetivamente normal)
        -> CN. Decisión revisada: SMC se define por rendimiento objetivo
        normal --> lo que mmse/hipocampo/biomarcadores pueden
        medir. La "queja subjetiva" no es una variable de este esquema,
        así que el modelo no puede sesgarse por ello. 
    """
    if pd.isna(valor_dx):
        return "Desconocido"
    texto = str(valor_dx).upper()
    if texto in ("CN", "SMC"):
        return "CN"
    if "MCI" in texto:  # cubre EMCI, LMCI
        return "MCI"
    if "DEMENTIA" in texto or texto == "AD":
        return "AD"
    return "Desconocido"


def viscode_a_meses(viscode: str):
    """Convierte el código de visita de ADNI ('bl', 'm06', 'm12'...) a
    meses desde la basal, para poder ordenar y graficar cronológicamente.
    Devuelve None para códigos no reconocidos (p. ej. 'sc' de screening).
    """
    if viscode == "bl":
        return 0
    if isinstance(viscode, str) and viscode.startswith("m") and viscode[1:].isdigit():
        return int(viscode[1:])
    return None


def cargar_datos_longitudinales(ruta_csv: str) -> pd.DataFrame:
    """Carga TODAS las visitas de ADNIMERGE (no solo la basal) -- para
    ver cómo evoluciona el MMSE a lo largo del tiempo por grupo
    diagnóstico. Distinto de cargar_datos_reales(), que se queda solo
    con la visita basal para entrenar el clasificador (4.3.2): esto es
    para un gráfico descriptivo de tendencia poblacional (5.1), no para
    entrenar nada -- no confundir con una predicción por paciente.
    """
    df = pd.read_csv(ruta_csv, low_memory=False)
    df = df.rename(columns={"RID": "paciente_id", "VISCODE": "viscode", "DX_bl": "diagnostico_texto"})
    df["diagnostico_grupo"] = df["diagnostico_texto"].apply(diagnostico_desde_adni)
    df["mmse"] = pd.to_numeric(df["MMSE"], errors="coerce")
    df["meses"] = df["viscode"].apply(viscode_a_meses)
    return df[["paciente_id", "viscode", "meses", "diagnostico_grupo", "mmse"]].dropna(subset=["meses"])


def cargar_datos_reales(ruta_csv: str, fuente: str = "adni", ruta_plasma: str | None = None) -> pd.DataFrame:
    """Carga el CSV real y lo estandariza.

    fuente: "adni" (única fuente soportada -- ver 4.1 de la memoria sobre
        por qué se abandonó OASIS-3 como alternativa).
    ruta_plasma: ruta al CSV de biomarcadores en plasma
        (UPENN_PLASMA_FUJIREBIO_QUANTERIX). Se une por RID -- amplía la
        cobertura de biomarcadores a sujetos sin punción lumbar.
    """
    mapeo = MAPEO_COLUMNAS_ADNI

    df_crudo = pd.read_csv(ruta_csv, low_memory=False)
    columnas_presentes = {k: v for k, v in mapeo.items() if k in df_crudo.columns}
    faltantes = set(mapeo) - set(columnas_presentes)
    if faltantes:
        print(f"Aviso: columnas esperadas no encontradas en el CSV: {faltantes}")
        print("Revisa el mapeo correspondiente arriba y ajusta a los nombres de tu fichero real.")

    df = df_crudo.rename(columns=columnas_presentes)

    if "VISCODE" in df.columns:
        df = df[df["VISCODE"] == "bl"].copy()  # una fila por paciente: visita basal

    for columna in ["tau_pg_ml", "p_tau_pg_ml", "abeta42_pg_ml"]:
        if columna in df.columns:
            df[columna] = df[columna].apply(_limpiar_valor_censurado)

    if ruta_plasma:
        df_plasma = pd.read_csv(ruta_plasma, low_memory=False)
        df_plasma = df_plasma[df_plasma["VISCODE"] == "bl"]
        columnas_plasma = {k: v for k, v in MAPEO_COLUMNAS_ADNI_PLASMA.items() if k in df_plasma.columns}
        df_plasma = df_plasma.rename(columns=columnas_plasma)
        columnas_a_unir = [c for c in ["ptau217_plasma_pg_ml", "ratio_ab42_ab40_plasma"] if c in df_plasma.columns]
        for columna in columnas_a_unir:
            df_plasma.loc[df_plasma[columna] < -1, columna] = np.nan  # -4 = código de perdido
        df = df.merge(df_plasma[["paciente_id"] + columnas_a_unir], on="paciente_id", how="left")
        if "ratio_ab42_ab40_plasma" in df.columns:
            df["amiloide_plasma"] = df["ratio_ab42_ab40_plasma"].apply(amiloide_plasma_estado)

    if "diagnostico" not in df.columns and "diagnostico_texto" in df.columns:
        df["diagnostico"] = df["diagnostico_texto"].apply(diagnostico_desde_adni)

    if "apoe4_conteo" in df.columns:
        df["apoe4"] = df["apoe4_conteo"].apply(apoe4_conteo_a_texto)

    return df


def generar_datos_sinteticos(n: int = 400, semilla: int = 42) -> pd.DataFrame:
    """Genera un dataset ficticio con la misma estructura y nombres de
    columna que tendrán los datos reales, para poder construir y probar
    todo el pipeline (4.2-4.4).
    Los valores por grupo diagnóstico son aproximaciones razonables, NO
    estadísticas reales -- solo sirven para validar que el código funciona.
    """
    rng = np.random.default_rng(semilla)
    grupos = rng.choice(["CN", "MCI", "AD"], size=n, p=[0.45, 0.30, 0.25])

    perfiles = {
        "CN": dict(mmse=(29, 1.2), tau=(200, 40), abeta=(900, 100), hipo=(3800, 300), apoe_pos=0.25),
        "MCI": dict(mmse=(26, 2.0), tau=(320, 60), abeta=(700, 120), hipo=(3400, 350), apoe_pos=0.45),
        "AD": dict(mmse=(19, 3.0), tau=(550, 90), abeta=(450, 100), hipo=(2900, 400), apoe_pos=0.65),
    }

    filas = []
    for i, grupo in enumerate(grupos):
        p = perfiles[grupo]
        filas.append({
            "paciente_id": f"SIM-{i:04d}",
            "edad": int(np.clip(rng.normal(72, 8), 55, 95)),
            "sexo": rng.choice(["M", "F"]),
            "diagnostico": grupo,
            "mmse": int(np.clip(rng.normal(*p["mmse"]), 0, 30)),
            "apoe4": "Positivo" if rng.random() < p["apoe_pos"] else "Negativo",
            "tau_pg_ml": round(max(rng.normal(*p["tau"]), 50), 1),
            "abeta42_pg_ml": round(max(rng.normal(*p["abeta"]), 100), 1),
            "hipocampo_mm3": round(max(rng.normal(*p["hipo"]), 1500), 1),
        })

    return pd.DataFrame(filas)


if __name__ == "__main__":
    df = generar_datos_sinteticos()
    print(df.head())
    print(f"\n{len(df)} pacientes sintéticos generados.")
    print(df["diagnostico"].value_counts())