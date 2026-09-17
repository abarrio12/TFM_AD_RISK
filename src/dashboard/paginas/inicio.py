"""
Página de inicio (Home): bienvenida y orientación. La predicción en sí
vive en su propia página -- separada a propósito para que un médico no
tenga que atravesar nada más para llegar a lo que necesita.
"""

import streamlit as st
from pathlib import Path

from carga import cargar_todo, mostrar_cabecera_sidebar

mostrar_cabecera_sidebar()

with st.spinner("Cargando NeuroInsight..."):
    df, modelo, codificador, explicador, columnas_modelo, fuente_datos = cargar_todo()

# Estilos personalizados profesionales
st.markdown(
    """
    <style>
        .hero-header {
            text-align: center;
            padding: 20px 20px 30px 20px;
            margin-bottom: 30px;
        }
        .hero-subtitle {
            font-size: 1.7em;
            color: #666;
            margin-top: 15px;
            font-weight: 500;
        }
        .hero-tagline {
            font-size: 1.15em;
            color: #999;
            margin-top: 10px;
            font-style: italic;
        }
        
        /* Contenedor y tamaño ampliado del logo */
        .logo-container {
            text-align: center;
            margin: 0 auto 20px auto;
            width: 100%;
        }
        .logo-container svg {
            width: 100% !important;
            max-width: 750px !important; /* Ajusta aquí el tamaño del logo */
            height: auto !important;
        }
        
        .info-box {
            background: #e3f2fd;
            border-left: 4px solid #1565C0;
            padding: 15px 20px;
            border-radius: 4px;
            margin: 20px 0;
            color: #0d47a1;
        }
        
        .tfm-note {
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 15px 20px;
            border-radius: 4px;
            margin-top: 30px;
            color: #e65100;
            font-size: 0.95em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sección Hero construida en un único bloque HTML para evitar artefactos
ruta_logo = Path(__file__).parent.parent / "assets" / "logo_sidebar.svg"

logo_html = ""
if ruta_logo.exists():
    contenido_svg = ruta_logo.read_text(encoding="utf-8")
    logo_html = f'<div class="logo-container">{contenido_svg}</div>'

hero_html = f"""
<div class="hero-header">
    {logo_html}
    <div class="hero-subtitle">Herramienta de Apoyo al Diagnóstico de Alzheimer</div>
    <div class="hero-tagline">Para profesionales y estudiantes de ciencias de la salud</div>
</div>
"""

st.markdown(hero_html, unsafe_allow_html=True)

# Información introductoria
st.markdown(
    """
    <div class="info-box">
    <strong>¿Qué es NeuroInsight?</strong><br><br>
    NeuroInsight es una herramienta de soporte clínico que combina:
    <ul>
    <li><strong>Extracción inteligente</strong>: Procesa informes PDF con IA local para extraer datos clínicos automáticamente</li>
    <li><strong>Predicción personalizada</strong>: Estima el riesgo de deterioro cognitivo del paciente basándose en biomarcadores y datos clínicos</li>
    <li><strong>Explicabilidad</strong>: Muestra qué factores influyen en cada predicción (ideal para la toma de decisiones)</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# Secciones con acordeón usando st.expander (más limpio)
st.markdown("## Explore las funcionalidades")

with st.expander("**Predicción Individual**", expanded=False):
    st.markdown("""
    Obtenga predicciones personalizadas para sus pacientes. Introduzca los datos de un paciente 
    (manualmente o subiendo su informe PDF) para ver:
    
    - **Diagnóstico predicho**:
        - **CN** = Cognitivamente Normal (sin deterioro)
        - **MCI** = Deterioro Cognitivo Leve (deterioro leve, sin demencia)
        - **AD** = Enfermedad de Alzheimer (demencia)
    - **Probabilidades**: Confianza del modelo en cada categoría diagnóstica
    - **Análisis de importancia**: Visualización de qué variables influyen más en la predicción
    - **Contexto poblacional**: Cómo se posiciona este paciente respecto a la población de entrenamiento
    
    **Datos soportados**: Edad, MMSE, biomarcadores en plasma (Tau, P-tau, Abeta), volumen hipocampal, APOE4
    """)

with st.expander("**Análisis e Interpretabilidad**", expanded=False):
    st.markdown("""
    Entienda cómo evolucionarían los pacientes y qué cambios influirían más.
    
    - **Trayectoria comparada**: evolución real de pacientes de ADNI con un perfil inicial similar al del paciente activo, no una proyección individual
    - **Análisis de sensibilidad**: "¿Qué pasaría si...?" (Ej: si el MMSE bajara de 25 a 20)
    - **Rango de variación real**: banda que muestra cuánto varió esa evolución entre esos pacientes similares
    - **Factores clave**: Identifica qué biomarcadores son más relevantes para este paciente
    """)

with st.expander("**Notas e Informes Clínicos**", expanded=False):
    st.markdown("""
    Mantenga un registro organizado de cada paciente.
    
    - **Notas personalizadas**: Escriba observaciones clínicas adjuntas a cada predicción
    - **Generador de informes** (página aparte "Generar informe"): descargue un DOCX con las
        secciones que elijas.
    - **Historial**: Guarde los pacientes estudiados en cada sesión. No se mantiene al cerrar.
    """)

with st.expander("**Búsqueda de Literatura**", expanded=False):
    st.markdown("""
    Acceda a evidencia científica relacionada con sus casos.
    
    - **Búsqueda automática de papers**: Busca publicaciones en PubMed relacionadas con:
        - El perfil específico del paciente
        - El tipo de deterioro cognitivo predicho
    """)

with st.expander("**Resumen del Dataset**", expanded=False):
    st.markdown(f"""
    Explore las estadísticas del conjunto de datos de entrenamiento.
    **Dataset utilizado:** {fuente_datos}
    """)

with st.expander("**Recursos y Enlaces**", expanded=False):
    st.markdown("""
    Referencias externas que puede resultar de interés.

    """)

# Nota sobre el TFM
st.markdown(
    """
    <div class="tfm-note">
    <strong> Nota académica:</strong> Esta herramienta se enmarca en un Trabajo de Fin de Máster 
    en el ámbito de ciencias de datos e inteligencia artificial aplicada a la medicina. 
    Es una herramienta de <strong>apoyo a la valoración clínica</strong> y no reemplaza el juicio médico profesional.
    </div>
    """,
    unsafe_allow_html=True,
)