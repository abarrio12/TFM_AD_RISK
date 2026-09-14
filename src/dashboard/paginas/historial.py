"""Página de historial de pacientes con persistencia por sesión."""

import pandas as pd
import streamlit as st

from carga import mostrar_cabecera_sidebar

mostrar_cabecera_sidebar()

st.title("Historial de pacientes")
st.caption("Todos los pacientes que has analizado quedan guardados en esta sesión")

historial = st.session_state.get("historial_pacientes", [])

if not historial:
    st.info("Todavía no has guardado ningún paciente en el historial.")
    st.stop()

for idx, item in enumerate(reversed(historial), 1):
    with st.container(border=True):
        st.markdown(f"### {idx}. {item['prediccion']['clase_predicha']} · {item['prediccion']['probabilidad'] * 100:.1f}%")
        st.caption(item.get("fecha", "Fecha no disponible"))

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Probabilidades**")
            for nombre, valor in item["prediccion"]["probabilidades"].items():
                st.write(f"- {nombre}: {valor * 100:.1f}%")
        with col2:
            st.write("**Datos del paciente**")
            if item.get("fila"):
                for clave, valor in item["fila"].items():
                    if valor is not None:
                        st.write(f"- {clave}: {valor}")
            else:
                st.write("Sin datos estructurados")

        if st.button(f"Abrir paciente {idx}", key=f"abrir_{item['id']}"):
            st.session_state["ultima_fila"] = pd.DataFrame([item["fila"]]) if item.get("fila") else None
            st.session_state["ultima_prediccion"] = item["prediccion"]
            st.session_state["ultima_ficha"] = item.get("ficha")
            st.session_state["notas_clinicas"] = item.get("notas", "")
            st.session_state["paciente_activo"] = item["id"]
            st.success("Paciente cargado en la sesión actual.")

        if item.get("notas"):
            with st.expander("Ver notas clínicas"):
                st.write(item["notas"])