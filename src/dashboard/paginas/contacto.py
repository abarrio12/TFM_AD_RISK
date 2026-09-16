"""
Página de contacto --> permite a usuarios enviar consultas o feedback.
Requiere configurar SMTP con una cuenta de Gmail personal.
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import streamlit as st

from carga import mostrar_cabecera_sidebar

mostrar_cabecera_sidebar()

st.title("Contacto")
st.caption("¿Preguntas o sugerencias? Ponte en contacto con nosotros")

st.markdown(
    """
    <style>
        .success-message {
            background: #c8e6c9;
            border-left: 4px solid #2e7d32;
            padding: 15px 20px;
            border-radius: 4px;
            color: #1b5e20;
            margin-bottom: 20px;
        }
        .error-message {
            background: #ffcdd2;
            border-left: 4px solid #c62828;
            padding: 15px 20px;
            border-radius: 4px;
            color: #b71c1c;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div style="margin-bottom: 30px;"></div>', unsafe_allow_html=True)

with st.form("contact_form", clear_on_submit=True):
    nombre = st.text_input(
        "Tu nombre",
        placeholder="Juan Pérez",
        help="Nombre completo o profesión"
    )
    
    email = st.text_input(
        "Tu email",
        placeholder="juan@ejemplo.com",
        help="Tu correo electrónico (para que podamos responder)"
    )
    
    asunto = st.selectbox(
        "Tipo de consulta",
        [
            "Selecciona una opción",
            "Pregunta técnica",
            "Sugerencia de mejora",
            "Reporte de error",
            "Colaboración / Investigación",
            "Otros"
        ]
    )
    
    mensaje = st.text_area(
        "Mensaje",
        placeholder="Cuéntanos qué te gustaría saber o sugerir...",
        height=150,
        help="Sé lo más específico posible para ayudarnos a entenderte mejor"
    )
    
    submitted = st.form_submit_button(
        "Enviar mensaje",
        use_container_width=True,
        type="primary"
    )

if submitted:
    # Validaciones básicas
    errores = []
    if not nombre.strip():
        errores.append("Por favor ingresa tu nombre")
    if not email.strip() or "@" not in email:
        errores.append("Por favor ingresa un email válido")
    if not mensaje.strip():
        errores.append("Por favor escribe un mensaje")
    if asunto == "Selecciona una opción":
        errores.append("Por favor selecciona el tipo de consulta")

    if errores:
        for error in errores:
            st.markdown(f'<div class="error-message">❌ {error}</div>', unsafe_allow_html=True)
    else:
        remitente = st.secrets.get("EMAIL_ADDRESS", "")
        contraseña = st.secrets.get("EMAIL_PASSWORD", "")
        destinatario = st.secrets.get("EMAIL_ADDRESS", "")

        if not remitente or not contraseña:
            st.markdown(
                '<div class="error-message">⚠️ El servicio de email no está configurado. '
                'Contacta directamente a: abarriol20@gmail.com</div>',
                unsafe_allow_html=True,
            )
        else:
            # Crear mensaje
            mensaje_email = MIMEMultipart()
            mensaje_email["From"] = remitente
            mensaje_email["To"] = destinatario
            mensaje_email["Subject"] = f"[NeuroInsight] {asunto} - {nombre}"

            cuerpo = f"""
Nuevo mensaje de contacto en NeuroInsight:

REMITENTE: {nombre}
EMAIL: {email}
TIPO: {asunto}

MENSAJE:
{mensaje}

---
Este mensaje fue enviado desde el formulario de contacto de NeuroInsight
            """

            mensaje_email.attach(MIMEText(cuerpo, "plain"))

            try:
                servidor = smtplib.SMTP("smtp.gmail.com", 587)
                servidor.starttls()
                servidor.login(remitente, contraseña)
                servidor.send_message(mensaje_email)
                servidor.quit()

                st.markdown(
                    '<div class="success-message">✅ Mensaje enviado correctamente. '
                    'Te responderemos en breve.</div>',
                    unsafe_allow_html=True,
                )
            except smtplib.SMTPException as e:
                st.markdown(
                    f'<div class="error-message">❌ Error al enviar: {e!s}</div>',
                    unsafe_allow_html=True,
                )
            except (OSError, ValueError) as e:
                st.markdown(
                    f'<div class="error-message">❌ Error inesperado: {e!s}</div>',
                    unsafe_allow_html=True,
                )

# Información adicional
st.markdown("---")

st.markdown(
    """
    **Nota de privacidad**: Su información será usada únicamente para responder 
    a su consulta y no será compartida con terceros.
    """
)

st.markdown("---")

st.subheader("Información del Proyecto")
st.markdown("""
- **Tipo**: Trabajo de Fin de Máster. Máster en IA, Big Data y Data Science
- **Centro**: Universidad Complutense de Madrid 
- **Enfoque**: Herramienta de apoyo a la valoración clínica de Alzheimer asistido por IA
- **Dataset**: ADNI (Alzheimer's Disease Neuroimaging Initiative)
- **Año**: 2026
""")