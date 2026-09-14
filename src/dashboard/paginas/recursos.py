"""
Recursos -- enlaces externos de interés para pacientes, familiares y
cuidadores. Contenido de apoyo, no clínico -- no sustituye a la
valoración de un profesional.
"""

import streamlit as st

from carga import mostrar_cabecera_sidebar

mostrar_cabecera_sidebar()

st.title("Recursos")
st.caption("Herramientas y enlaces externos de interés -- contenido informativo, no un diagnóstico")

st.markdown("### Actividades y apps para personas con demencia")
st.markdown(
    "Recopilación del Centro de Referencia Estatal de Atención a Personas con "
    "Enfermedad de Alzheimer (CREA-IMSERSO): aplicaciones móviles con juegos y "
    "actividades de estimulación cognitiva.\n\n"
    "🔗 [crealzheimer.imserso.es](https://crealzheimer.imserso.es/documentacion/recursos/aplicaciones-moviles/aplicaciones-personas-demencia)"
)

st.divider()

st.markdown("### Calculadora de riesgo de demencia")
st.markdown(
    "Herramienta web para adultos de 55 años o más: a partir de edad, estilo de "
    "vida, salud general y actividad física, estima la \"edad cerebral\" y el "
    "riesgo de diagnóstico de demencia a 5 años, con recomendaciones para "
    "reducirlo.\n\n"
    "🔗 [projectbiglife.ca/calculators/dementia](https://www.projectbiglife.ca/calculators/dementia)"
)

st.divider()

st.markdown("### Test de memoria online (MindCrowd)")
st.markdown(
    "Test de memoria y atención de 10 minutos, parte de un estudio abierto sobre "
    "el cerebro y la memoria.\n\n"
    "🔗 [mindcrowd.org/es](https://mindcrowd.org/es/)"
)

st.divider()
st.caption(
    "Estos recursos son externos a NeuroInsight y de carácter informativo -- "
    "ninguno sustituye una valoración médica."
)