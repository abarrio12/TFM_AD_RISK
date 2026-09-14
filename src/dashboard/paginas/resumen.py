"""
Resumen del dataset de entrenamiento -- estadísticas del conjunto usado
para entrenar el modelo (distribución de diagnósticos, subgrupos,
importancia de variables). Separado de la página de predicción a
propósito: un médico usando la herramienta para un paciente concreto no
necesita ver esto antes de poder trabajar -- esta página es para quien
quiera entender el modelo, no para el flujo clínico del día a día.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from carga import COLOR_DIAGNOSTICO, aplicar_estilo_grafico, cargar_todo, calcular_proyeccion, mostrar_cabecera_sidebar

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "modelado"))
from datos import cargar_datos_longitudinales

mostrar_cabecera_sidebar()

df, modelo, codificador, explicador, columnas_modelo, fuente_datos = None, None, None, None, None, None
with st.spinner("Cargando NeuroInsight..."):
    df, modelo, codificador, explicador, columnas_modelo, fuente_datos = cargar_todo()

st.title("Resumen del dataset")
st.caption(f"Datos usados para entrenar el modelo: {fuente_datos}")

with st.expander("Sobre el conjunto de datos (ADNI)", expanded=True):
    st.markdown(
        "La **Alzheimer's Disease Neuroimaging Initiative (ADNI)** es un estudio longitudinal "
        "multicéntrico, en marcha desde 2004, que reúne datos clínicos, cognitivos, de "
        "neuroimagen y biomarcadores de más de 2000 participantes en distintos estadios "
        "(cognitivamente normales, deterioro cognitivo leve y enfermedad de Alzheimer), con el "
        "objetivo de desarrollar marcadores para la detección y el seguimiento tempranos de la "
        "enfermedad."
    )
    st.markdown("🔗 [adni.loni.usc.edu](https://adni.loni.usc.edu)")
    st.markdown(
        "**Cita (formato APA):** Alzheimer's Disease Neuroimaging Initiative. (2004–2026). "
        "*ADNI database* [Conjunto de datos]. University of Southern California, "
        "Laboratory of Neuro Imaging. https://adni.loni.usc.edu"
    )
    st.caption(
        "ADNI no tiene un único artículo canónico que citar (es un consorcio, no una publicación) -- "
        "esta es la forma recomendada de citar la base de datos en sí. El texto de reconocimiento "
        "oficial en inglés, exigido para cualquier uso de estos datos, es el siguiente:"
    )
    st.caption(
        "Data used in preparation of this article were obtained from the Alzheimer's Disease "
        "Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators "
        "within ADNI contributed to the design and implementation of ADNI and/or provided data "
        "but did not participate in analysis or writing of this report."
    )

# --------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
with c1.container(border=True):
    st.metric("Pacientes", len(df))
with c2.container(border=True):
    st.metric("CN", int((df["diagnostico"] == "CN").sum()))
with c3.container(border=True):
    st.metric("MCI", int((df["diagnostico"] == "MCI").sum()))
with c4.container(border=True):
    st.metric("AD", int((df["diagnostico"] == "AD").sum()))

st.divider()

# --------------------------------------------------------------------------
# Distribución de diagnósticos + Clustering
# --------------------------------------------------------------------------

col_izq, col_der = st.columns(2)

with col_izq:
    st.subheader("Distribución de diagnósticos")
    fig_donut = px.pie(
        df, names="diagnostico", hole=0.5, color="diagnostico",
        color_discrete_map=COLOR_DIAGNOSTICO,
    )
    fig_donut = aplicar_estilo_grafico(fig_donut)
    fig_donut.update_layout(margin=dict(t=30, b=40, l=40, r=20))
    st.plotly_chart(fig_donut, use_container_width=True)

with col_der:
    st.subheader("Subgrupos (clustering)")
    st.caption(
        "Cada punto es un paciente, agrupado automáticamente (KMeans) según lo parecido de su perfil "
        "clínico completo. Los ejes (UMAP 1 / UMAP 2) no tienen un significado propio -- son una "
        "proyección a 2 dimensiones de muchas variables a la vez, solo para poder dibujarlo; lo que "
        "importa es qué tan cerca o lejos caen los puntos entre sí, no la posición exacta de cada eje."
    )
    proyeccion, etiqueta_eje = calcular_proyeccion(df)
    df_plot = df.loc[df[["edad", "mmse", "tau_pg_ml", "abeta42_pg_ml", "hipocampo_mm3"]].dropna().index].copy()
    df_plot[f"{etiqueta_eje} 1"] = proyeccion[:, 0]
    df_plot[f"{etiqueta_eje} 2"] = proyeccion[:, 1]
    fig_cluster = px.scatter(
        df_plot, x=f"{etiqueta_eje} 1", y=f"{etiqueta_eje} 2",
        color=df_plot["cluster"].astype(str),
        labels={"color": "Cluster"},
    )
    fig_cluster = aplicar_estilo_grafico(fig_cluster)
    fig_cluster.update_layout(margin=dict(t=30, b=60, l=70, r=20))
    st.plotly_chart(fig_cluster, use_container_width=True)

    resumen_cluster = (
        df_plot.groupby("cluster")[["edad", "mmse", "tau_pg_ml", "abeta42_pg_ml", "hipocampo_mm3"]]
        .mean().round(1)
    )
    resumen_cluster.insert(0, "Pacientes", df_plot.groupby("cluster").size())
    resumen_cluster.index = [f"Cluster {i}" for i in resumen_cluster.index]
    resumen_cluster.columns = ["Pacientes", "Edad media", "MMSE medio", "Tau medio", "Abeta42 medio", "Hipocampo medio (mm³)"]
    st.caption("Qué caracteriza a cada grupo (valores medios):")
    st.dataframe(resumen_cluster, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# Importancia de variables (global, XGBoost)
# --------------------------------------------------------------------------

st.subheader("Importancia de variables (modelo XGBoost)")
st.caption(
    "Cuánto pesa cada variable, en promedio, en las predicciones del modelo entrenado -- no en un "
    "paciente concreto (para eso, ver el gráfico SHAP en Predicción individual), sino en general."
)
df_importancia = pd.DataFrame({
    "variable": columnas_modelo,
    "importancia": modelo.feature_importances_,
}).sort_values("importancia")
fig_importancia = aplicar_estilo_grafico(px.bar(df_importancia, x="importancia", y="variable", orientation="h"))
fig_importancia.update_layout(margin=dict(t=30, b=60, l=140, r=20), height=340)
st.plotly_chart(fig_importancia, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# Tendencia poblacional de MMSE por grupo diagnóstico (visitas longitudinales)
# --------------------------------------------------------------------------

st.subheader("Evolución del MMSE por grupo diagnóstico")
st.caption(
    "Descriptivo, no predictivo: media observada en las visitas de seguimiento reales de ADNI, "
    "agrupada por el diagnóstico que tenía cada paciente en su visita basal. No es una proyección "
    "para un paciente individual -- para eso haría falta un modelo distinto, entrenado "
    "específicamente para pronosticar trayectorias (ver 6.3, líneas futuras)."
)

raiz_datos = Path(__file__).parent.parent.parent.parent / "data" / "raw"
ruta_adni_completo = raiz_datos / "ADNIMERGE_11Aug2026.csv"

if ruta_adni_completo.exists():
    @st.cache_data
    def cargar_tendencia(ruta):
        df_long = cargar_datos_longitudinales(str(ruta))
        df_long = df_long[df_long["diagnostico_grupo"] != "Desconocido"]
        agregado = (
            df_long.dropna(subset=["mmse"])
            .groupby(["diagnostico_grupo", "meses"])
            .agg(mmse_medio=("mmse", "mean"), n_pacientes=("paciente_id", "nunique"))
            .reset_index()
        )
        # Puntos con muestra muy pequeña se descartan -- con pocos pacientes,
        # la media es ruido, no una tendencia real (ver ejemplo real: AD a
        # los 18 meses con solo 4 pacientes daba una media de 12.5, un salto
        # que no refleja progresión real, solo casualidad de quién tenía visita).
        return agregado[agregado["n_pacientes"] >= 10].sort_values(["diagnostico_grupo", "meses"])

    df_tendencia = cargar_tendencia(ruta_adni_completo)

    fig_tendencia = px.line(
        df_tendencia, x="meses", y="mmse_medio", color="diagnostico_grupo",
        color_discrete_map=COLOR_DIAGNOSTICO, markers=True,
        labels={"meses": "Meses desde la visita basal", "mmse_medio": "MMSE medio", "diagnostico_grupo": "Diagnóstico basal"},
        hover_data={"n_pacientes": True},
    )
    fig_tendencia = aplicar_estilo_grafico(fig_tendencia)
    fig_tendencia.update_layout(margin=dict(t=30, b=60, l=70, r=20), height=440)
    st.plotly_chart(fig_tendencia, use_container_width=True)
    st.caption("El número de pacientes por punto disminuye con el tiempo (abandonos del seguimiento) -- pasa el ratón sobre cada punto para verlo.")
else:
    st.info(f"No se encontró {ruta_adni_completo.name} en data/raw/ -- esta sección necesita el fichero completo de ADNIMERGE.")

st.divider()
with st.expander("Glosario de siglas usadas en esta página"):
    st.markdown(
        "- **ADNI**: Alzheimer's Disease Neuroimaging Initiative\n"
        "- **CN**: Cognitively Normal (cognitivamente normal, sin deterioro)\n"
        "- **MCI**: Mild Cognitive Impairment (deterioro cognitivo leve)\n"
        "- **AD**: Alzheimer's Disease (enfermedad de Alzheimer)\n"
        "- **MMSE**: Mini-Mental State Examination (prueba cognitiva breve, 0-30 puntos)\n"
        "- **UMAP**: Uniform Manifold Approximation and Projection (técnica para reducir muchas "
        "variables a 2 dimensiones y poder dibujarlas)\n"
        "- **SHAP**: SHapley Additive exPlanations (método para explicar cuánto pesa cada variable "
        "en una predicción concreta)"
    )