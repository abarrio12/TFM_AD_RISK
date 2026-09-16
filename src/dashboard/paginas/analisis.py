"""Página de Análisis e Interpretabilidad para el paciente activo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "modelado"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from carga import aplicar_estilo_grafico, cargar_todo, mostrar_cabecera_sidebar
from clasificacion import probabilidades_escenario_mmse
from datos import cargar_datos_longitudinales

mostrar_cabecera_sidebar()

historial = st.session_state.get("historial_pacientes", [])

if not historial and ("ultima_prediccion" not in st.session_state or not st.session_state["ultima_prediccion"]):
    st.warning("No hay ningún paciente en el historial. Ve a Predicción individual y calcula una predicción para empezar.")
    st.stop()

if historial:
    opciones = {f"{item['fecha']} - {item['prediccion']['clase_predicha']}": item for item in historial}
    claves_opciones = list(opciones.keys())
    paciente_activo = st.session_state.get("paciente_activo")
    indice_activo = next(
        (indice for indice, item in enumerate(opciones.values()) if item.get("id") == paciente_activo),
        len(claves_opciones) - 1,
    )
    seleccion = st.selectbox("Paciente activo", claves_opciones, index=indice_activo)
    item_seleccionado = opciones[seleccion]
    st.session_state["paciente_activo"] = item_seleccionado["id"]
    st.session_state["ultima_prediccion"] = item_seleccionado["prediccion"]
    st.session_state["ultima_fila"] = pd.DataFrame([item_seleccionado["fila"]]) if item_seleccionado.get("fila") else None
    st.session_state["ultima_ficha"] = item_seleccionado.get("ficha")
    st.session_state["notas_clinicas"] = item_seleccionado.get("notas", "")
    pred = item_seleccionado["prediccion"]
else:
    pred = st.session_state["ultima_prediccion"]

st.title("Análisis e Interpretabilidad")
st.caption("Se evalúa el paciente activo con un escenario clínico basado en MMSE")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Diagnóstico", pred["clase_predicha"])
with col2:
    st.metric("Probabilidad", f"{pred['probabilidad'] * 100:.1f}%")
with col3:
    st.metric("Fecha", pred.get("fecha", "-"))

st.markdown("---")
st.subheader("Qué está impulsando la decisión")

if st.session_state.get("ultima_fila") is not None:
    fila = st.session_state["ultima_fila"]
    row = fila.iloc[0].to_dict()
    st.dataframe(
        {
            "Variable": list(row.keys()),
            "Valor": [round(float(v), 3) if isinstance(v, (int, float)) and pd.notna(v) else v for v in row.values()],
        },
        use_container_width=True,
    )
else:
    st.info("Todavía no hay datos del paciente guardados en la sesión.")

# ----------------------------------------------------------------
# Escenario clínico -- usando el modelo real, no una fórmula ajena a él
# ----------------------------------------------------------------
st.markdown("---")
st.subheader("Escenario clínico")
st.caption("Cambia el MMSE y vuelve a calcularse la predicción real con el modelo entrenado.")

with st.spinner("Cargando NeuroInsight..."):
    df_pob, modelo, codificador, explicador, columnas_modelo, fuente_datos = cargar_todo()

if st.session_state.get("ultima_fila") is not None:
    fila_original = st.session_state["ultima_fila"]
    mmse_actual = float(fila_original.iloc[0].get("mmse", 0.0) or 0.0)

    mmse_hipotetico = st.slider("MMSE hipotético", 0, 30, int(mmse_actual) if mmse_actual else 20)

    fila_escenario = fila_original.copy()
    fila_escenario["mmse"] = float(mmse_hipotetico)

    proba_escenario = probabilidades_escenario_mmse(
        modelo, fila_escenario, columnas_modelo, codificador, mmse_hipotetico
    )
    idx_escenario = int(np.argmax(proba_escenario))
    clase_escenario = codificador.inverse_transform([idx_escenario])[0]
    prob_clase_actual_escenario = float(proba_escenario[list(codificador.classes_).index(pred["clase_predicha"])])
    prob_clase_actual_real = float(pred["probabilidades"].get(pred["clase_predicha"], pred["probabilidad"]))

    curva_mmse = []
    for valor_mmse in range(31):
        probabilidades_mmse = probabilidades_escenario_mmse(
            modelo, fila_escenario, columnas_modelo, codificador, valor_mmse
        )
        curva_mmse.append({
            "MMSE": valor_mmse,
            **{str(clase): float(probabilidad) * 100 for clase, probabilidad in zip(codificador.classes_, probabilidades_mmse)},
        })
    df_curva_mmse = pd.DataFrame(curva_mmse)

    fig_sensibilidad = go.Figure()
    colores_clase = {"CN": "#2E7D32", "MCI": "#F9A825", "AD": "#C62828"}
    for clase in codificador.classes_:
        fig_sensibilidad.add_trace(go.Scatter(
            x=df_curva_mmse["MMSE"], y=df_curva_mmse[str(clase)], mode="lines+markers",
            name=str(clase), line=dict(color=colores_clase.get(str(clase), "#37474F"), width=3),
            hovertemplate=f"MMSE: %{{x}}<br>{clase}: %{{y:.1f}}%<extra></extra>",
        ))
    fig_sensibilidad.add_vline(x=mmse_hipotetico, line_dash="dash", line_color="#1A1A2E", annotation_text="Hipotético")
    if mmse_actual > 0:
        fig_sensibilidad.add_vline(x=mmse_actual, line_dash="dot", line_color="#546E7A", annotation_text="Real")
    fig_sensibilidad.update_layout(
        title="Cómo cambia la salida del modelo según el MMSE",
        xaxis_title="MMSE", yaxis_title="Probabilidad del modelo (%)",
        xaxis=dict(range=[0, 30], dtick=5), yaxis=dict(range=[0, 100]),
        hovermode="x unified", height=430,
    )
    fig_sensibilidad = aplicar_estilo_grafico(fig_sensibilidad)
    st.plotly_chart(fig_sensibilidad, use_container_width=True)
    st.caption(
        "Cada línea muestra la probabilidad estimada para una clase al variar mmse. "
        "La línea punteada es el valor real y la discontinua el escenario elegido."
    )

    cambio = prob_clase_actual_escenario - prob_clase_actual_real
    if mmse_hipotetico == int(mmse_actual):
        st.info(f"MMSE = {mmse_hipotetico} (igual al valor actual del paciente)")
    else:
        direccion = "sube" if cambio > 0 else "baja"
        st.success(
            f"Con MMSE = {mmse_hipotetico} (valor real del paciente: {mmse_actual:.0f}), el modelo entrenado predice "
            f"**{clase_escenario}** como clase más probable. La probabilidad de la clase actualmente predicha "
            f"({pred['clase_predicha']}) {direccion} de {prob_clase_actual_real*100:.1f}% a {prob_clase_actual_escenario*100:.1f}%."
        )
else:
    st.info("No hay datos estructurados del paciente para simular un escenario.")

# ----------------------------------------------------------------
# Trayectoria comparada con pacientes similares reales de ADNI
# ----------------------------------------------------------------
st.markdown("---")
st.subheader("Trayectoria comparada con pacientes similares")
st.caption(
    "No es una predicción para este paciente concreto. Es la evolución real, observada en las visitas, "
    "de pacientes de ADNI que empezaron con un perfil similar al de este paciente. " \
    "La banda sombreada es la variación real observada entre esos pacientes."
)

raiz_datos = Path(__file__).parent.parent.parent.parent / "data" / "raw"
ruta_adni_completo = raiz_datos / "ADNIMERGE_11Aug2026.csv"

if st.session_state.get("ultima_fila") is not None and ruta_adni_completo.exists():
    fila_actual = st.session_state["ultima_fila"]
    mmse_paciente = float(fila_actual.iloc[0].get("mmse", 0.0) or 0.0)
    diagnostico_paciente = pred["clase_predicha"]

    if mmse_paciente > 0 and diagnostico_paciente in ("CN", "MCI", "AD"):
        @st.cache_data
        def cargar_trayectoria_similar(ruta, diagnostico, mmse_centro, margen=3, minimo_n=8):
            df_long = cargar_datos_longitudinales(str(ruta))
            df_long = df_long[df_long["diagnostico_grupo"] == diagnostico]
            basal = df_long[df_long["meses"] == 0].set_index("paciente_id")["mmse"]
            similares = basal[(basal >= mmse_centro - margen) & (basal <= mmse_centro + margen)].index
            df_similar = df_long[df_long["paciente_id"].isin(similares)]
            agregado = (
                df_similar.dropna(subset=["mmse"])
                .groupby("meses")
                .agg(mmse_medio=("mmse", "mean"), mmse_std=("mmse", "std"), n=("paciente_id", "nunique"))
                .reset_index()
            )
            agregado["mmse_std"] = agregado["mmse_std"].fillna(0)
            return agregado[agregado["n"] >= minimo_n].sort_values("meses"), len(similares)

        df_traj, n_similares = cargar_trayectoria_similar(ruta_adni_completo, diagnostico_paciente, mmse_paciente)

        if len(df_traj) >= 2:
            banda_arriba = (df_traj["mmse_medio"] + df_traj["mmse_std"]).clip(upper=30)
            banda_abajo = (df_traj["mmse_medio"] - df_traj["mmse_std"]).clip(lower=0)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(df_traj["meses"]) + list(df_traj["meses"])[::-1],
                y=list(banda_arriba) + list(banda_abajo)[::-1],
                fill="toself", fillcolor="rgba(21,101,192,0.15)",
                line=dict(color="rgba(255,255,255,0)"), name="Variación observada", showlegend=True,
            ))
            fig.add_trace(go.Scatter(
                x=df_traj["meses"], y=df_traj["mmse_medio"], mode="lines+markers",
                line=dict(color="#1565C0", width=3), name="MMSE medio (pacientes similares)",
            ))
            fig.add_trace(go.Scatter(
                x=[0], y=[mmse_paciente], mode="markers",
                marker=dict(size=16, color="#C62828", symbol="diamond", line=dict(width=2, color="white")),
                name="Este paciente (hoy)",
            ))
            fig.update_layout(
                xaxis_title="Meses desde la visita basal", yaxis_title="MMSE",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            fig = aplicar_estilo_grafico(fig)
            fig.update_layout(margin=dict(t=60, b=10, l=10, r=10), height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Comparado con {n_similares} pacientes reales de ADNI con diagnóstico basal {diagnostico_paciente} y MMSE inicial entre {mmse_paciente-3:.0f} y {mmse_paciente+3:.0f}.")
        else:
            st.info("No hay suficientes pacientes similares en el dataset con seguimiento a largo plazo para mostrar esta comparación de forma fiable.")
    else:
        st.info("Falta el MMSE del paciente, o el diagnóstico no es CN/MCI/AD.")
else:
    st.info("No se encontró el fichero completo de ADNI, o no hay datos del paciente.")