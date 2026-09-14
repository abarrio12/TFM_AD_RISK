"""
Dashboard -- sección 5 del TFM. Punto de entrada: declara las páginas y
sus iconos (Material Symbols, monocromos) y delega en st.navigation().

Uso: streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="NeuroInsight", page_icon="assets/favicon.svg", layout="wide")

pagina_inicio = st.Page("paginas/inicio.py", title="Inicio", icon=":material/home:", default=True)
pagina_prediccion = st.Page("paginas/prediccion.py", title="Predicción individual", icon=":material/stethoscope:")
pagina_analisis = st.Page("paginas/analisis.py", title="Análisis e Interpretabilidad", icon=":material/trending_up:")
pagina_notas = st.Page("paginas/notas.py", title="Notas Clínicas", icon=":material/description:")
pagina_informe = st.Page("paginas/informe.py", title="Generar informe", icon=":material/summarize:")
pagina_literatura = st.Page("paginas/literatura.py", title="Literatura", icon=":material/library_books:")
pagina_historial = st.Page("paginas/historial.py", title="Historial de pacientes", icon=":material/history:")
pagina_resumen = st.Page("paginas/resumen.py", title="Resumen del dataset", icon=":material/bar_chart:")
pagina_recursos = st.Page("paginas/recursos.py", title="Recursos", icon=":material/menu_book:")
pagina_contacto = st.Page("paginas/contacto.py", title="Contacto", icon=":material/mail:")

pg = st.navigation([pagina_inicio, pagina_prediccion, pagina_analisis, pagina_notas, pagina_informe,
                     pagina_literatura, pagina_historial, pagina_resumen, pagina_recursos, pagina_contacto])
pg.run()