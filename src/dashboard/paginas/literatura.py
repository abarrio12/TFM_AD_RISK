"""Página de literatura relevante para el paciente activo."""

import streamlit as st

from carga import mostrar_cabecera_sidebar
from reportes import obtener_papers_relevantes

mostrar_cabecera_sidebar()

if "ultima_prediccion" not in st.session_state or not st.session_state["ultima_prediccion"]:
    st.warning("No hay un paciente activo todavía. Ve a Predicción individual y calcula una predicción para ver literatura relevante.")
    st.stop()

pred = st.session_state["ultima_prediccion"]

st.title("Literatura Relevante")
st.caption("Búsqueda en vivo en PubMed, filtrada por el paciente activo y la predicción actual")

NOMBRES_LEGIBLES = {
    "tau_pg_ml": "Tau",
    "abeta42_pg_ml": "Abeta42",
    "ptau217_plasma_pg_ml": "p-Tau217 (plasma)",
    "ratio_ab42_ab40_plasma": "Ratio Aβ42/Aβ40 (plasma)",
    "hipocampo_mm3": "Volumen hipocampal",
}

fila = st.session_state.get("ultima_fila")
biomarcadores = []
if fila is not None and not fila.empty:
    for key in ["tau_pg_ml", "abeta42_pg_ml", "ptau217_plasma_pg_ml", "ratio_ab42_ab40_plasma", "hipocampo_mm3"]:
        if key in fila.columns and fila[key].notna().any():
            biomarcadores.append(key)

etiquetas = [NOMBRES_LEGIBLES.get(b, b) for b in biomarcadores] or ["Solo diagnóstico predicho"]
chips_html = " ".join(
    f'<span style="background:#E3F2FD;color:#1565C0;padding:5px 14px;border-radius:16px;'
    f'margin-right:8px;display:inline-block;font-size:0.85em;font-weight:600;">{etq}</span>'
    for etq in etiquetas
)
st.markdown(
    f'<div style="margin-bottom:8px;"> <strong>Buscando literatura sobre:</strong> {chips_html} '
    f'<span style="background:#FFF3E0;color:#E65100;padding:5px 14px;border-radius:16px;'
    f'display:inline-block;font-size:0.85em;font-weight:600;">{pred["clase_predicha"]}</span></div>',
    unsafe_allow_html=True,
)

with st.spinner("Buscando en PubMed (puede tardar unos segundos)..."):
    papers = obtener_papers_relevantes(biomarcadores, pred["clase_predicha"])

if not papers:
    st.info("No se encontraron artículos que coincidan con este perfil concreto en PubMed ahora mismo.")
    st.stop()

for idx, paper in enumerate(papers, 1):
    with st.container(border=True):
        st.markdown(f"**{idx}. {paper['titulo']}**")
        st.caption(f"Autores: {', '.join(paper['autores'][:3])} | {paper['año']} | {paper['journal']}")
        st.link_button("Ver en PubMed", f"https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/")