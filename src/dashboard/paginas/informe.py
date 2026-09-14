"""Página de generación de informe -- el usuario elige qué secciones incluir."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from carga import mostrar_cabecera_sidebar
from reportes import generar_reporte_docx, obtener_papers_relevantes

mostrar_cabecera_sidebar()

if "ultima_prediccion" not in st.session_state or not st.session_state["ultima_prediccion"]:
    st.warning("No hay un paciente activo. Ve a Predicción individual (o elige uno en Historial) antes de generar un informe.")
    st.stop()

pred = st.session_state["ultima_prediccion"]
fila = st.session_state.get("ultima_fila")
notas = st.session_state.get("notas_clinicas", "")

st.title("Generar informe")
st.caption(f"Paciente activo: {pred['clase_predicha']} · {pred['probabilidad'] * 100:.1f}%")

st.subheader("¿Qué quieres incluir en el informe?")
incluir_prediccion = st.checkbox("Predicción y datos del paciente", value=True)
factores_shap_disponibles = st.session_state.get("ultimos_factores_shap")
incluir_porque = st.checkbox(
    "Por qué (factores que más pesaron en la predicción)",
    value=bool(factores_shap_disponibles),
    disabled=not bool(factores_shap_disponibles),
    help="No hay datos de esta explicación en la sesión actual -- vuelve a Predicción individual" if not factores_shap_disponibles else None,
)
incluir_literatura = st.checkbox("Literatura relevante (búsqueda en PubMed)", value=False)
incluir_notas = st.checkbox(
    "Notas clínicas",
    value=bool(notas and notas.strip()),
    disabled=not bool(notas and notas.strip()),
    help="No hay notas guardadas para este paciente" if not (notas and notas.strip()) else None,
)

if not (incluir_prediccion or incluir_literatura or incluir_notas or incluir_porque):
    st.info("Selecciona al menos una sección para poder generar el informe.")
elif st.button("Generar informe DOCX", type="primary"):
    with st.spinner("Generando el documento..."):
        ficha_para_reporte = None
        if incluir_prediccion and fila is not None:
            ficha_para_reporte = {k: v for k, v in fila.iloc[0].to_dict().items() if pd.notna(v)}

        factores_riesgo = None
        if incluir_porque and factores_shap_disponibles:
            clase = pred["clase_predicha"]
            factores_riesgo = []
            for f in factores_shap_disponibles:
                sentido = "hacia" if f["impacto"] > 0 else "en contra de"
                factores_riesgo.append(f"{f['variable']}: empuja {sentido} {clase} (peso {abs(f['impacto']):.2f})")

        papers = None
        if incluir_literatura:
            biomarcadores = []
            if fila is not None:
                for col in ["tau_pg_ml", "abeta42_pg_ml", "ptau217_plasma_pg_ml", "ratio_ab42_ab40_plasma"]:
                    if col in fila.columns and pd.notna(fila[col].iloc[0]):
                        biomarcadores.append(col)
            papers = obtener_papers_relevantes(biomarcadores, pred["clase_predicha"])
            if not papers:
                st.info("No se encontraron artículos en PubMed para este perfil concreto -- el informe se genera igualmente, con esa sección indicándolo así, no con resultados inventados.")

        doc_bytes = generar_reporte_docx(
            ficha_paciente=ficha_para_reporte,
            probabilidades=pred.get("probabilidades"),
            clase_predicha=pred["clase_predicha"] if incluir_prediccion else None,
            factores_riesgo=factores_riesgo,
            papers=papers,
            notas=notas if incluir_notas else None,
            incluir_prediccion=incluir_prediccion,
            incluir_literatura=incluir_literatura,
            incluir_notas=incluir_notas,
        )

    st.success("Informe generado.")
    st.download_button(
        "📥 Descargar informe (DOCX)",
        data=doc_bytes,
        file_name=f"neuroinsight_informe_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
    )