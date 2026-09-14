"""
Genera el embedding de una ficha de paciente y la guarda en PostgreSQL
(pgvector). También resuelve la búsqueda de "pacientes similares".

Punto del índice que cubre: 3.3 (almacenamiento de embeddings y búsqueda
semántica) y 3.4 (mecanismos de depuración: aquí, evitar reprocesar el
mismo documento dos veces). Da soporte al panel opcional de pacientes
similares (6.3).

Requiere: el contenedor Docker corriendo y la tabla `pacientes` ya creada,
con la columna hash_documento (ver tablas.sql), más el modelo de
embeddings descargado:
    ollama pull nomic-embed-text

Instala dependencias nuevas:
    pip install psycopg2-binary pgvector python-dotenv
"""

from __future__ import annotations

import hashlib
import os
import uuid

import ollama
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

from extraccion_paciente import FichaPaciente  # módulo de la sección 3.2

load_dotenv()

MODELO_EMBEDDING = "nomic-embed-text"


def ficha_a_texto(ficha: FichaPaciente) -> str:
    """Convierte la ficha estructurada en una frase normalizada para embeber.

    Incluye hallazgos_adicionales -- aunque no alimenta al clasificador
    (4.3.2), sí enriquece la búsqueda por similitud (3.3): dos pacientes
    con hallazgos raros parecidos quedan más cerca en el espacio semántico.
    """
    d, b = ficha.diagnostico, ficha.biomarcadores
    partes = [f"Diagnóstico: {d.diagnostico_actual.value}.", f"APOE4: {d.apoe4.value}."]

    if ficha.edad is not None:
        partes.append(f"Edad: {ficha.edad} años.")
    if ficha.mmse is not None:
        partes.append(f"MMSE: {ficha.mmse:g}/30.")
    if b.tau_pg_ml is not None:
        partes.append(f"Tau: {b.tau_pg_ml} pg/mL.")
    if b.p_tau_pg_ml is not None:
        partes.append(f"P-Tau: {b.p_tau_pg_ml} pg/mL.")
    if b.abeta42_pg_ml is not None:
        partes.append(f"Abeta42: {b.abeta42_pg_ml} pg/mL.")
    if b.hipocampo_mm3 is not None:
        partes.append(f"Volumen hipocampal: {b.hipocampo_mm3} mm3.")
    if b.atrofia_grado is not None:
        partes.append(f"Atrofia: {b.atrofia_grado}.")

    for hallazgo in ficha.hallazgos_adicionales:
        partes.append(f"{hallazgo}.")

    return " ".join(partes)


def generar_embedding(texto: str) -> list[float]:
    respuesta = ollama.embed(model=MODELO_EMBEDDING, input=texto)
    return respuesta["embeddings"][0]


def hash_texto(texto: str) -> str:
    """Huella única del documento de entrada. Detecta que procesamos
    EXACTAMENTE este texto antes (p. ej. al relanzar el script en pruebas).
    No detecta que dos informes distintos son del mismo paciente real en
    dos visitas -- eso exigiría un identificador de paciente real, que hoy
    no viene en el documento.
    """
    return hashlib.sha256(texto.strip().encode("utf-8")).hexdigest()


def _conectar():
    conexion = psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="tfm_pacientes",
        user="tfm_user",
        password=os.environ["POSTGRES_PASSWORD"],
    )
    register_vector(conexion)
    return conexion


def ya_procesado(hash_doc: str) -> str | None:
    """Si este documento exacto ya está guardado, devuelve su paciente_id."""
    conexion = _conectar()
    with conexion, conexion.cursor() as cur:
        cur.execute(
            "SELECT paciente_id FROM pacientes WHERE hash_documento = %s",
            (hash_doc,),
        )
        fila = cur.fetchone()
    conexion.close()
    return fila[0] if fila else None


def obtener_ficha(paciente_id: str) -> FichaPaciente | None:
    """Recupera la ficha ya guardada de un paciente sin volver a llamar al
    LLM -- se usa cuando procesar_informe() detecta que el documento ya
    existía, para no gastar una extracción de más."""
    conexion = _conectar()
    with conexion, conexion.cursor() as cur:
        cur.execute('SELECT ficha FROM pacientes WHERE paciente_id = %s', (paciente_id,))
        fila = cur.fetchone()
    conexion.close()
    return FichaPaciente.model_validate(fila[0]) if fila else None


def guardar_paciente(ficha: FichaPaciente, hash_doc: str, paciente_id: str | None = None) -> str:
    """Embebe la ficha y la guarda en la tabla `pacientes`. Devuelve el id usado.

    hash_doc tiene restricción UNIQUE en la base de datos: es la red de
    seguridad real. Aunque un bug se saltara ya_procesado(), Postgres
    rechazaría el duplicado igualmente.
    """
    paciente_id = paciente_id or str(uuid.uuid4())
    resumen = ficha_a_texto(ficha)
    embedding = generar_embedding(resumen)

    conexion = _conectar()
    with conexion, conexion.cursor() as cur:
        cur.execute(
            """
            INSERT INTO pacientes (paciente_id, ficha, resumen_texto, embedding, hash_documento)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (paciente_id) DO NOTHING
            """,
            (paciente_id, ficha.model_dump_json(), resumen, embedding, hash_doc),
        )
    conexion.close()
    return paciente_id


def pacientes_similares(ficha: FichaPaciente, k: int = 5) -> list[tuple]:
    """Devuelve los k pacientes ya guardados más parecidos a esta ficha
    (paciente_id, resumen_texto, distancia). Distancia menor = más parecido.
    """
    embedding = generar_embedding(ficha_a_texto(ficha))

    conexion = _conectar()
    with conexion, conexion.cursor() as cur:
        cur.execute(
            """
            SELECT paciente_id, resumen_texto, embedding <=> %s::vector AS distancia
            FROM pacientes
            ORDER BY distancia
            LIMIT %s
            """,
            (embedding, k),
        )
        resultados = cur.fetchall()
    conexion.close()
    return resultados


def procesar_informe(texto_informe: str) -> tuple[str | None, FichaPaciente | None]:
    """Punto de entrada único para un documento: comprueba si ya se procesó
    (por hash del texto) antes de gastar una llamada al LLM; si ya existía,
    recupera la ficha guardada en vez de volver a extraerla. Devuelve
    siempre (paciente_id, ficha) listos para usar, por ejemplo con
    pacientes_similares(), sin que quien llama tenga que extraer nada por
    su cuenta.
    """
    from extraccion_paciente import extraer_ficha_paciente

    hash_doc = hash_texto(texto_informe)
    existente = ya_procesado(hash_doc)
    if existente:
        print(f"Este documento ya se había procesado (paciente_id={existente}); no se repite.")
        return existente, obtener_ficha(existente)

    ficha, intentos = extraer_ficha_paciente(texto_informe)
    if not ficha:
        print("No se pudo extraer la ficha; no se guarda nada.")
        return None, None

    paciente_id = guardar_paciente(ficha, hash_doc)
    return paciente_id, ficha


def procesar_pdf(ruta_pdf: str) -> tuple[str | None, FichaPaciente | None]:
    """Punto de entrada para un informe en PDF real -- el equivalente de
    'el médico sube su PDF'. Extrae el texto (leer_pdf.py) y delega en
    procesar_informe(), que ya sabe hacer todo lo demás: hash, extracción,
    embedding y guardado.
    """
    from leer_pdf import extraer_texto_pdf

    texto = extraer_texto_pdf(ruta_pdf)
    if texto is None:
        return None, None  # ya se avisó del motivo dentro de extraer_texto_pdf

    return procesar_informe(texto)


if __name__ == "__main__":
    informe_ejemplo = """
    Paciente mujer, diagnóstico AD. APOE4 positivo. Tau en LCR: 610 pg/mL.
    Atrofia hipocampal severa.
    """
    pid, ficha = procesar_informe(informe_ejemplo)
    if pid and ficha:
        print(f"paciente_id: {pid}")
        print("Pacientes similares:", pacientes_similares(ficha))
    else:
        print("No se pudo procesar el informe de ejemplo")