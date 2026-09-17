"""
Carga y entrenamiento compartidos entre las páginas del dashboard.

Aislado en su propio módulo para que app.py (la página de predicción,
la que usa el médico) y pages/1_Resumen_del_dataset.py (la vista con
las estadísticas del conjunto de entrenamiento) compartan exactamente
el mismo modelo ya entrenado y cacheado, en vez de cargarlo/entrenarlo
dos veces por separado.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "modelado"))

import shap
import streamlit as st
import plotly.graph_objects as go

from clasificacion import entrenar_modelo, preparar_features
from clustering import ejecutar_clustering, proyectar_2d
from datos import cargar_datos_reales, generar_datos_sinteticos

COLOR_DIAGNOSTICO = {"CN": "#2E7D32", "MCI": "#F9A825", "AD": "#C62828"}


def mostrar_cargando(mensaje: str):
    """Indicador de carga con icono girando de verdad (animación CSS,
    no el spinner nativo de Streamlit) -- para que se note que
    algo se está ejecutando --> da seguridad al usuario

    Devuelve el contenedor (st.empty()) para poder limpiarlo con
    .empty() cuando termine el trabajo.
    """
    contenedor = st.empty()
    contenedor.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:10px; padding:10px 14px;
                    background:#E3F2FD; border-radius:8px; margin-bottom:8px;">
            <span style="display:inline-block; font-size:1.4em;
                         animation: neuroinsight_girar 1s linear infinite;">⏳</span>
            <span style="color:#1A1A2E;">{mensaje}</span>
        </div>
        <style>
        @keyframes neuroinsight_girar {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return contenedor


def mostrar_cabecera_sidebar():
    """Logo de la app fijo arriba de la barra lateral, ANTES de la lista
    de páginas. st.logo() es la única forma de conseguir esto en
    Streamlit -- cualquier cosa dentro de with st.sidebar se dibuja
    siempre DESPUÉS de la navegación automática entre páginas.

    Los colores (fondo blanco, texto oscuro) vienen de
    .streamlit/config.toml -- el sistema de temas oficial de Streamlit,
    estable entre versiones. Aquí solo se sube el tamaño de letra base,
    con un selector amplio (html) que no depende de nombres internos de
    Streamlit que puedan cambiar de una versión a otra.
    """
    ruta_logo = Path(__file__).parent / "assets" / "logo_sidebar.svg"
    if ruta_logo.exists():
        st.logo(str(ruta_logo), size="large")

    st.markdown(
        "<style>html { font-size: 27px; } h1 { font-size: 2.5em !important; } "
        "h2, h3 { font-size: 1.8em !important; }</style>",
        unsafe_allow_html=True,
    )


def aplicar_estilo_grafico(fig):
    """Tamaño de letra consistente y algo mayor para ejes, leyenda y
    título en los gráficos de Plotly -- aplicar a cada figura antes de
    st.plotly_chart(...). Color de letra oscuro explícito: con fondo
    blanco (tema en .streamlit/config.toml), depender del color por
    defecto de Plotly puede dejar texto claro ilegible sobre blanco.
    """
    fig.update_layout(
        font=dict(size=14, color="#1A1A2E"),
        legend=dict(font=dict(size=13, color="#1A1A2E")),
        title_font=dict(size=16, color="#1A1A2E"),
    )
    return fig


def grafico_degradado(valor: float, minimo: float, maximo: float, titulo: str):
    """Barra de degradado continuo (rojo -> amarillo -> verde) con una
    línea marcando dónde cae un valor -- como un gráfico de espectro/
    longitud de onda. Plotly solo admite tramos de color
    sólido (steps), no un degradado real, así que se simula con muchos
    tramos finos.
    """
    rojo, amarillo, verde = (198, 40, 40), (249, 168, 37), (46, 125, 50)
    n_pasos = 30
    ancho = (maximo - minimo) / n_pasos
    pasos = []
    for i in range(n_pasos):
        t = i / (n_pasos - 1)
        if t < 0.5:
            t2 = t / 0.5
            rgb = tuple(int(rojo[j] + (amarillo[j] - rojo[j]) * t2) for j in range(3))
        else:
            t2 = (t - 0.5) / 0.5
            rgb = tuple(int(amarillo[j] + (verde[j] - amarillo[j]) * t2) for j in range(3))
        pasos.append({
            "range": [minimo + i * ancho, minimo + (i + 1) * ancho],
            "color": f"rgb({rgb[0]},{rgb[1]},{rgb[2]})",
        })

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=valor,
        title={"text": titulo},
        gauge={
            "axis": {"range": [minimo, maximo]},
            "bar": {"color": "rgba(0,0,0,0)", "thickness": 0},
            "steps": pasos,
            "threshold": {"line": {"color": "#1A1A2E", "width": 5}, "thickness": 0.85, "value": valor},
        },
    ))
    fig.update_layout(height=260, margin=dict(t=60, b=20, l=30, r=30))
    return fig


@st.cache_resource(show_spinner=False)
def cargar_todo():
    """Carga los datos y entrena el modelo una sola vez por sesión.

    Usa los datos reales de ADNI (+ plasma) si encuentra los ficheros en
    data/raw/; si no, cae a datos sintéticos para que el dashboard nunca
    se rompa por un fichero movido de sitio.
    """
    raiz_datos = Path(__file__).parent.parent.parent / "data" / "raw"
    ruta_adni = raiz_datos / "ADNIMERGE_11Aug2026.csv"
    ruta_plasma = raiz_datos / "UPENN_PLASMA_FUJIREBIO_QUANTERIX_11Aug2026.csv"

    if ruta_adni.exists():
        df = cargar_datos_reales(
            str(ruta_adni), fuente="adni",
            ruta_plasma=str(ruta_plasma) if ruta_plasma.exists() else None,
        )
        df = df[df["diagnostico"] != "Desconocido"].copy()
        fuente_datos = f"ADNI real — {len(df)} pacientes"
    else:
        df = generar_datos_sinteticos(n=400)
        fuente_datos = f"sintéticos — {len(df)} pacientes (no se encontró {ruta_adni.name} en data/raw/)"

    df = ejecutar_clustering(df)
    X, y, codificador = preparar_features(df)
    modelo, X_train, X_test, y_train, y_test = entrenar_modelo(X, y)
    explicador = shap.TreeExplainer(getattr(modelo, "modelo_base", modelo))

    return df, modelo, codificador, explicador, list(X.columns), fuente_datos


@st.cache_data
def calcular_proyeccion(_df):
    """Envuelve proyectar_2d en caché: sin esto, Streamlit lo recalculaba
    entero (UMAP incluido) en CADA interacción del usuario. El guion bajo
    en _df le dice a Streamlit que no intente hashear el DataFrame (ya es
    estable, viene de cargar_todo(), que también está cacheado).
    """
    return proyectar_2d(_df)