"""
Predicción individual -- sección 5.2 del TFM. Introduce los datos de un
paciente (a mano o subiendo su informe en PDF) y obtén una predicción
explicada: probabilidad, qué variables la empujaron (SHAP), y dónde cae
este paciente respecto a la población de entrenamiento.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "extraccion"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from carga import (
    COLOR_DIAGNOSTICO,
    aplicar_estilo_grafico,
    cargar_todo,
    mostrar_cabecera_sidebar,
)
from clasificacion import apoe4_a_feature
from extraccion_paciente import extraer_ficha_paciente
from inferencia import campos_faltantes, ficha_a_features
from leer_pdf import extraer_texto_pdf

mostrar_cabecera_sidebar()

with st.sidebar:
    st.subheader("Configuración de Ollama")
    host_ollama = st.text_input(
        "URL de Ollama remoto (vacío = local)",
        key="host_ollama",
        placeholder="https://xxxx.ngrok-free.app",
        help="Deja vacío para usar Ollama en este ordenador. Pega aquí la URL del "
             "túnel de ngrok si estás usando el notebook de Colab con GPU.",
    )
    if host_ollama:
        try:
            import ollama as _ollama_check
            host_ollama = host_ollama.strip().rstrip("/")
            _ollama_check.Client(
                host=host_ollama,
                timeout=30,
                headers={"ngrok-skip-browser-warning": "true"},
            ).list()
            st.success("✅ Conectado -- usando Ollama remoto (Colab)")
        except Exception as error:
            st.error(
                "❌ No responde la URL de Ollama. Comprueba que sea la `public_url` recién generada "
                f"por ngrok y que Colab siga ejecutando Ollama. Detalle: {error}"
            )
    else:
        st.caption("Usando Ollama local (CPU)")

df, modelo, codificador, explicador, columnas_modelo, fuente_datos = None, None, None, None, None, None
with st.spinner("Cargando NeuroInsight..."):
    df, modelo, codificador, explicador, columnas_modelo, fuente_datos = cargar_todo()

st.session_state.setdefault("ultima_fila", None)
st.session_state.setdefault("ultima_ficha", None)
st.session_state.setdefault("ultima_prediccion", {})
st.session_state.setdefault("notas_clinicas", "")
st.session_state.setdefault("historial_pacientes", [])


def _valor_formulario(fila_activa, nombre: str, valor_vacio=0.0):
    """Devuelve un valor guardable en un widget, convirtiendo NaN en vacío."""
    if fila_activa is None or fila_activa.empty or nombre not in fila_activa.columns:
        return valor_vacio
    valor = fila_activa.iloc[0][nombre]
    return valor_vacio if pd.isna(valor) else float(valor)


def _apoe_formulario(fila_activa):
    if fila_activa is None or fila_activa.empty or "apoe4_positivo" not in fila_activa.columns:
        return "Desconocido"
    valor = fila_activa.iloc[0]["apoe4_positivo"]
    if pd.isna(valor):
        return "Desconocido"
    return "Positivo" if float(valor) == 1 else "Negativo"


fila_activa = st.session_state.get("ultima_fila")
identidad_activa = st.session_state.get("paciente_activo", "sin_paciente")
if identidad_activa != st.session_state.get("_prediccion_formulario_paciente"):
    st.session_state["pred_edad"] = _valor_formulario(fila_activa, "edad")
    st.session_state["pred_mmse"] = int(_valor_formulario(fila_activa, "mmse"))
    st.session_state["pred_apoe4"] = _apoe_formulario(fila_activa)
    st.session_state["pred_tau"] = _valor_formulario(fila_activa, "tau_pg_ml")
    st.session_state["pred_abeta42"] = _valor_formulario(fila_activa, "abeta42_pg_ml")
    st.session_state["pred_ptau217"] = _valor_formulario(fila_activa, "ptau217_plasma_pg_ml")
    st.session_state["pred_ratio_abeta"] = _valor_formulario(fila_activa, "ratio_ab42_ab40_plasma")
    st.session_state["pred_hipocampo"] = _valor_formulario(fila_activa, "hipocampo_mm3")
    st.session_state["_prediccion_formulario_paciente"] = identidad_activa

st.title("Predicción individual")
st.caption("Introduce los datos de un paciente y obtén una predicción explicada (sección 5.2)")

if st.session_state.get("ultima_prediccion"):
    _p = st.session_state["ultima_prediccion"]
    st.info(
        f"Ya tienes un paciente activo cargado ({_p['clase_predicha']} · {_p['probabilidad']*100:.1f}%, "
        f"{_p.get('fecha', '')}) -- puedes verlo en Análisis, Notas o Literatura. Esta página siempre "
        "empieza en blanco para calcular un paciente **nuevo**; no sobrescribe el activo hasta que pulses "
        "\"Predecir riesgo\" o \"Extraer y predecir\" otra vez."
    )

modo = st.radio(
    "¿Cómo quieres introducir los datos del paciente?",
    ["Rellenar a mano", "Subir informe en PDF"],
    horizontal=True,
)

fila = None
ficha_extraida = None

if modo == "Rellenar a mano":
    st.caption(
        "Estos son exactamente los campos que usa el modelo entrenado (ver Resumen del dataset → "
        "Importancia de variables) -- no una selección arbitraria; el modelo no tiene ninguna otra entrada."
    )
    col_form, _ = st.columns([1, 1])
    with col_form:
        edad_m = st.number_input("Edad", min_value=0, max_value=110, key="pred_edad")
        mmse_m = st.slider("MMSE", 0, 30, key="pred_mmse")
        apoe4_m = st.selectbox("APOE4", ["Desconocido", "Positivo", "Negativo"], key="pred_apoe4")
        tau_m = st.number_input("Tau (pg/mL)", min_value=0.0, step=10.0, key="pred_tau")
        abeta42_m = st.number_input("Abeta42 (pg/mL)", min_value=0.0, step=10.0, key="pred_abeta42")
        ptau217_m = st.number_input("p-Tau217 plasma (pg/mL)", min_value=0.0, step=1.0, key="pred_ptau217")
        ratio_abeta_m = st.number_input("Ratio Aβ42/Aβ40 plasma", min_value=0.0, step=0.001, format="%.3f", key="pred_ratio_abeta")
        hipocampo_m = st.number_input("Volumen hipocampal (mm³)", min_value=0.0, step=50.0, key="pred_hipocampo")
        if st.button("Predecir riesgo", type="primary", use_container_width=True):
            # 0.0 es el valor "sin tocar" de estos campos (para no precargar un perfil falso,
            # ver más arriba) -- pero un 0 real (p. ej. hipocampo_mm3=0) es biológicamente
            # imposible y el modelo lo trataba como un dato real y extremo, no como "sin dato".
            # Se convierte a NaN aquí, igual que ya se hace con los biomarcadores de plasma.
            def _o_nan(valor):
                return np.nan if valor == 0 else float(valor)

            fila = pd.DataFrame([{
                "edad": _o_nan(edad_m), "mmse": _o_nan(mmse_m), "hipocampo_mm3": _o_nan(hipocampo_m),
                "tau_pg_ml": _o_nan(tau_m), "abeta42_pg_ml": _o_nan(abeta42_m),
                "ptau217_plasma_pg_ml": _o_nan(ptau217_m), "ratio_ab42_ab40_plasma": _o_nan(ratio_abeta_m),
                "apoe4_positivo": apoe4_a_feature(apoe4_m),
            }])[columnas_modelo].astype(float)

else:
    archivo = st.file_uploader("Sube el informe clínico (PDF)", type="pdf")
    if archivo is not None and st.button("Extraer y predecir", type="primary"):
        # Validar tamaño del archivo (máximo 100MB - la extracción de palabras clave lo optimiza)
        tamaño_mb = archivo.size / (1024 * 1024)
        if tamaño_mb > 100:
            st.error(f"❌ El archivo es demasiado grande ({tamaño_mb:.1f} MB). Máximo permitido: 100 MB.")
        else:
            ruta_temp = Path("_temp_informe_dashboard.pdf")
            ruta_temp.write_bytes(archivo.getvalue())
            _animacion = mostrar_cargando("Leyendo el contenido del documento PDF...")
            try:
                texto = extraer_texto_pdf(str(ruta_temp))
                if texto is None:
                    _animacion.empty()
                    st.error("No se pudo extraer texto de este PDF. Compruebe si se trata de un documento escaneado sin capa de texto.")
                else:
                    _animacion = mostrar_cargando(
                        "Extrayendo información estructurada mediante el modelo de lenguaje local. "
                        "Esta operación puede requerir varios minutos, en particular en la primera "
                        "solicitud tras iniciar el servicio, ya que el modelo debe cargarse en memoria."
                    )
                    ficha, intentos = extraer_ficha_paciente(texto, host=host_ollama if host_ollama else None)
                    _animacion.empty()
                    if ficha is None:
                        st.warning("El modelo no devolvió una ficha válida tras varios intentos. Se recomienda la introducción manual de los datos.")
                    else:
                        ficha_extraida = ficha
                        fila = ficha_a_features(ficha)
                        st.success(f"Extracción completada ({intentos} intento(s) requerido(s)).")
            except Exception as error:
                _animacion.empty()
                mensaje = str(error)
                if "timed out" in mensaje.lower() or "timeout" in mensaje.lower():
                    st.error(
                        "El modelo de lenguaje no respondió dentro del tiempo previsto (más de 5 minutos). "
                        "Si utiliza el servicio local (CPU), esta situación puede producirse con informes extensos; "
                        "se recomienda repetir la solicitud o utilizar la conexión remota mediante GPU (barra lateral)."
                    )
                else:
                    st.error(f"No se pudo completar la extracción. Compruebe que el servicio de Ollama esté activo. Detalle: {error}")
            finally:
                _animacion.empty()
                ruta_temp.unlink(missing_ok=True)

if ficha_extraida is not None:
    faltan = campos_faltantes(fila)
    if faltan:
        st.info(f"El informe no mencionaba estos campos (la predicción sigue adelante sin ellos): {', '.join(faltan)}")
    with st.expander("Ver la ficha extraída completa"):
        st.json(ficha_extraida.model_dump())

if fila is not None:
    with st.spinner("Calculando la predicción..."):
        probabilidades = modelo.predict_proba(fila)[0]
        indice_predicho = int(np.argmax(probabilidades))
        clase_predicha = codificador.inverse_transform([indice_predicho])[0]
        probabilidad = float(probabilidades[indice_predicho])

    st.session_state["ultima_fila"] = fila
    st.session_state["ultima_ficha"] = ficha_extraida
    st.session_state["ultima_prediccion"] = {
        "probabilidades": {str(clase): float(prob) for clase, prob in zip(codificador.classes_, probabilidades)},
        "clase_predicha": str(clase_predicha),
        "probabilidad": float(probabilidad),
        "columnas_modelo": list(columnas_modelo),
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    historial = st.session_state.get("historial_pacientes", [])
    version = {}
    if ficha_extraida is not None:
        version["ficha"] = ficha_extraida.model_dump()
    if fila is not None:
        version["fila"] = fila.iloc[0].to_dict()
    version["prediccion"] = {
        "clase_predicha": str(clase_predicha),
        "probabilidad": float(probabilidad),
        "probabilidades": {str(clase): float(prob) for clase, prob in zip(codificador.classes_, probabilidades)},
    }
    version["fecha"] = st.session_state["ultima_prediccion"]["fecha"]
    version["notas"] = st.session_state.get("notas_clinicas", "")
    version["signature"] = json.dumps(version, sort_keys=True, default=str)

    existente = next((item for item in historial if item.get("signature") == version["signature"]), None)
    if existente is None:
        item = {
            "id": f"paciente_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "fecha": version["fecha"],
            "fila": version.get("fila", {}),
            "ficha": version.get("ficha", {}),
            "prediccion": version["prediccion"],
            "notas": version["notas"],
            "signature": version["signature"],
        }
        historial.append(item)
        st.session_state["historial_pacientes"] = historial
    else:
        item = existente
    st.session_state["paciente_activo"] = item["id"]

    col_a, col_b = st.columns(2)

    with col_a:
        # Diagnóstico bien grande y claro -- antes solo era el título pequeño del gráfico,
        # fácil de pasar por alto o de leer mal ("¿ponía Alzheimer?" cuando ponía MCI).
        _color_dx = {"CN": "#2E7D32", "MCI": "#F9A825", "AD": "#C62828"}
        st.markdown(
            f'<div style="text-align:center; margin-bottom:-10px;">'
            f'<span style="font-size:1.1em; color:#666;">Diagnóstico predicho</span><br>'
            f'<span style="font-size:2.4em; font-weight:800; color:{_color_dx[clase_predicha]};">{clase_predicha}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # Aviso de fiabilidad: cuántas de las 8 variables tienen dato real. Con muy pocas,
        # el modelo puede mostrar un porcentaje alto igualmente -- eso no es un error del
        # modelo (ver 6.2: usa la ausencia de dato como señal), pero SÍ hay que avisar de
        # que la predicción se apoya en poca información real, no ocultarlo tras un número.
        n_disponibles = int(fila.notna().sum(axis=1).iloc[0])
        n_total = len(columnas_modelo)

        NOMBRES_LEGIBLES = {
            "edad": "Edad", "mmse": "MMSE", "hipocampo_mm3": "Volumen hipocampal",
            "tau_pg_ml": "Tau", "abeta42_pg_ml": "Abeta42",
            "ptau217_plasma_pg_ml": "p-Tau217 (plasma)", "ratio_ab42_ab40_plasma": "Ratio Aβ42/Aβ40 (plasma)",
            "apoe4_positivo": "APOE4",
        }
        usadas = [NOMBRES_LEGIBLES.get(c, c) for c in columnas_modelo if pd.notna(fila[c].iloc[0])]
        no_usadas = [NOMBRES_LEGIBLES.get(c, c) for c in columnas_modelo if pd.isna(fila[c].iloc[0])]
        with st.expander(f"Datos usados en esta predicción ({n_disponibles} de {n_total})"):
            st.markdown(f"✅ **Con dato real:** {', '.join(usadas) if usadas else '(ninguna)'}")
            st.markdown(f"◻️ **Sin dato (el modelo trata la ausencia como información, no como cero):** {', '.join(no_usadas) if no_usadas else '(ninguna)'}")

        if n_disponibles <= 3:
            st.warning(
                f"⚠️ Solo {n_disponibles} de {n_total} variables tienen dato real para este paciente. "
                "Con tan poca información, este porcentaje no debe leerse como un diagnóstico fiable, "
                "aunque el modelo lo muestre con aparente seguridad."
            )

        # Un único color de fondo, no 3 zonas: el valor mostrado es SIEMPRE la confianza en
        # la clase ganadora (nunca "CN al 50%" mostrado como tal, porque entonces otra clase
        # sería la ganadora) -- así que dentro de un mismo diagnóstico, más alto siempre es
        # "más seguro de eso", y si "eso" es preocupante (MCI/AD), más alto es peor noticia,
        # no mejor. Una zona "buena" dentro del propio rosco de MCI/AD no tiene sentido.
        _color_fondo = "#C8E6C9" if clase_predicha == "CN" else "#FFCDD2"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probabilidad * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#37474F"},
                "steps": [{"range": [0, 100], "color": _color_fondo}],
            },
        ))
        fig_gauge = aplicar_estilo_grafico(fig_gauge)
        fig_gauge.update_layout(height=240, margin=dict(t=20, b=10, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)
        if clase_predicha == "CN":
            st.caption("Fondo verde: CN es un resultado tranquilizador. El porcentaje mostrado está calibrado y no equivale por sí solo a una certeza clínica.")
        else:
            st.caption(f"Fondo rojo: {clase_predicha} implica algún grado de deterioro. El porcentaje está calibrado y no equivale por sí solo a una certeza clínica.")

    with col_b:
        st.caption("Específico de **este** paciente (no el promedio general -- para eso, ver Resumen del dataset → Importancia de variables).")
        valores_shap = explicador.shap_values(fila)
        if isinstance(valores_shap, list):
            fila_shap = valores_shap[indice_predicho][0]
        else:
            fila_shap = valores_shap[0, :, indice_predicho]

        df_shap = pd.DataFrame({"variable": columnas_modelo, "impacto": fila_shap})
        faltantes_actuales = campos_faltantes(fila)
        df_shap["variable"] = df_shap["variable"].apply(
            lambda v: f"{v} (no disponible)" if v in faltantes_actuales else v
        )
        df_shap = df_shap.sort_values("impacto")

        # Se guarda para que la página "Generar informe" pueda incluir el "por qué",
        # no solo el diagnóstico -- sin esto, el informe dice AD pero no explica nada.
        st.session_state["ultimos_factores_shap"] = [
            {"variable": fila_v["variable"], "impacto": round(float(fila_v["impacto"]), 3)}
            for _, fila_v in df_shap.sort_values("impacto", key=abs, ascending=False).iterrows()
        ]

        # El color refleja el sentido clínico (rojo = hacia más deterioro, verde = hacia más
        # normalidad), no solo "a favor/en contra de la clase predicha" -- con eso último, las
        # barras que empujan hacia CN saldrían en rojo igual que las que empujan hacia AD, lo cual
        # es contraintuitivo (CN es un resultado tranquilizador, no debería leerse como alarma).
        if clase_predicha == "CN":
            escala_color = ["#C62828", "#2E7D32"]  # negativo (aleja de CN) = rojo; positivo (hacia CN) = verde
        else:  # AD o MCI: ambas implican algún grado de deterioro frente a CN
            escala_color = ["#2E7D32", "#C62828"]  # negativo (aleja) = verde; positivo (hacia AD/MCI) = rojo

        fig_shap = px.bar(
            df_shap, x="impacto", y="variable", orientation="h",
            color="impacto", color_continuous_scale=escala_color,
            title=f"Por qué (SHAP): qué empujó hacia {clase_predicha}",
        )
        fig_shap = aplicar_estilo_grafico(fig_shap)
        fig_shap.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=280, coloraxis_showscale=False)
        st.plotly_chart(fig_shap, use_container_width=True)
        _direccion_roja = f"hacia {clase_predicha}" if clase_predicha != "CN" else "en contra de CN (hacia más deterioro)"
        _direccion_verde = f"en contra de {clase_predicha}" if clase_predicha != "CN" else "hacia CN (más normalidad)"
        st.caption(
            f"Cada barra es una variable, ordenada por peso. **Rojo**: empuja {_direccion_roja}. "
            f"**Verde**: empuja {_direccion_verde}. La longitud de la barra es cuánto pesa esa variable."
        )
        if faltantes_actuales:
            st.caption(
                "Las variables marcadas \"(no disponible)\" no se dieron en el informe -- "
                "el modelo puede usar la propia ausencia del dato como señal (p. ej., si en los "
                "datos de entrenamiento no tener un biomarcador medido está asociado a un tipo de "
                "diagnóstico), no un valor concreto de esa variable."
            )

    # ----------------------------------------------------------------
    # Posicionamiento frente a la población de entrenamiento
    # ----------------------------------------------------------------
    st.divider()
    st.subheader("Dónde se sitúa este paciente")
    st.caption(
        "Comparamos cada valor disponible con los pacientes reales del conjunto de entrenamiento. "
        "Cada caja representa el 50% central de un grupo diagnóstico y los puntos muestran valores "
        "atípicos; el rombo negro es este paciente. Así puedes ver si su MMSE o su hipocampo se "
        "parecen más a CN, MCI o AD. Esta comparación es descriptiva y no sustituye a la predicción."
    )

    mmse_paciente = fila["mmse"].iloc[0]
    hipo_paciente = fila["hipocampo_mm3"].iloc[0]
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        if pd.notna(mmse_paciente) and mmse_paciente > 0:
            fig_mmse = go.Figure()
            for diagnostico, color in COLOR_DIAGNOSTICO.items():
                valores = df.loc[df["diagnostico"] == diagnostico, "mmse"].dropna()
                if len(valores):
                    fig_mmse.add_trace(go.Box(
                        x=valores, y=[diagnostico] * len(valores), name=diagnostico,
                        orientation="h", marker_color=color, line_color=color,
                        boxpoints="outliers", hovertemplate="%{x:.1f} puntos<extra>%{fullData.name}</extra>",
                    ))
            fig_mmse.add_trace(go.Scatter(
                x=[mmse_paciente], y=["Este paciente"], mode="markers", name="Este paciente",
                marker=dict(symbol="diamond", size=13, color="#1A1A2E", line=dict(color="white", width=1)),
                hovertemplate="MMSE: %{x:.1f}<extra>Este paciente</extra>",
            ))
            fig_mmse.update_layout(
                title="MMSE: comparación con cada diagnóstico", xaxis_title="Puntuación MMSE (0-30)",
                yaxis_title="Grupo", showlegend=False, height=310,
            )
            fig_mmse = aplicar_estilo_grafico(fig_mmse)
            st.plotly_chart(fig_mmse, use_container_width=True)
        else:
            st.info("MMSE no disponible en este informe")

    with col_g2:
        if pd.notna(hipo_paciente) and hipo_paciente > 0:
            fig_hipo = go.Figure()
            for diagnostico, color in COLOR_DIAGNOSTICO.items():
                valores = df.loc[df["diagnostico"] == diagnostico, "hipocampo_mm3"].dropna()
                if len(valores):
                    fig_hipo.add_trace(go.Box(
                        x=valores, y=[diagnostico] * len(valores), name=diagnostico,
                        orientation="h", marker_color=color, line_color=color,
                        boxpoints="outliers", hovertemplate="%{x:.0f} mm³<extra>%{fullData.name}</extra>",
                    ))
            fig_hipo.add_trace(go.Scatter(
                x=[hipo_paciente], y=["Este paciente"], mode="markers", name="Este paciente",
                marker=dict(symbol="diamond", size=13, color="#1A1A2E", line=dict(color="white", width=1)),
                hovertemplate="Hipocampo: %{x:.0f} mm³<extra>Este paciente</extra>",
            ))
            fig_hipo.update_layout(
                title="Hipocampo: comparación con cada diagnóstico", xaxis_title="Volumen (mm³)",
                yaxis_title="Grupo", showlegend=False, height=310,
            )
            fig_hipo = aplicar_estilo_grafico(fig_hipo)
            st.plotly_chart(fig_hipo, use_container_width=True)
        else:
            st.info("Volumen hipocampal no disponible en este informe")

else:
    st.info("Rellena los datos, o sube un PDF y pulsa el botón correspondiente, para ver una predicción.")

st.divider()
with st.expander("Glosario de siglas usadas en esta página"):
    st.markdown(
        "- **CN / MCI / AD**: Cognitivamente Normal / Deterioro Cognitivo Leve / Enfermedad de Alzheimer\n"
        "- **MMSE**: Mini-Mental State Examination (prueba cognitiva breve, 0-30 puntos)\n"
        "- **APOE4**: variante del gen APOE asociada a mayor riesgo de Alzheimer\n"
        "- **SHAP**: método que explica cuánto pesó cada variable en esta predicción concreta"
    )