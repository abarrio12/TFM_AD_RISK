"""Página de notas clínicas del paciente activo."""

import streamlit as st

from carga import mostrar_cabecera_sidebar

mostrar_cabecera_sidebar()

if "ultima_prediccion" not in st.session_state or not st.session_state["ultima_prediccion"]:
    st.warning("No hay un paciente activo. Ve a la página de Predicción individual y ejecuta la predicción antes de añadir notas.")
    st.stop()

st.title("Notas Clínicas y Documentación")
st.caption("La información está asociada al paciente activo y no a un ejemplo genérico")

paciente = st.session_state["ultima_prediccion"]
st.subheader(f"Paciente activo: {paciente['clase_predicha']} · {paciente['probabilidad'] * 100:.1f}%")

st.session_state.setdefault("notas_clinicas", "")
notas = st.text_area(
    "Observaciones clínicas",
    value=st.session_state["notas_clinicas"],
    height=220,
    placeholder="Escriba aquí observaciones, cambios de tratamiento, valoración funcional, etc...",
)
st.session_state["notas_clinicas"] = notas

if st.button("Guardar notas", type="primary"):
    st.success("Notas guardadas en la sesión del paciente activo.")

if notas.strip():
    with st.expander("Vista previa de las notas guardadas"):
        st.write(notas)
else:
    st.info("Todavía no ha escrito ninguna nota para este paciente.")