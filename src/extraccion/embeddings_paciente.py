"""
Genera el embedding de una ficha de paciente y la guarda en PostgreSQL
(pgvector). También resuelve la búsqueda de "pacientes similares".

Punto del índice que cubre: 3.3 (almacenamiento de embeddings y búsqueda
semántica), y da soporte al panel opcional de pacientes similares (6.3).

Requiere: el contenedor Docker corriendo y la tabla `pacientes` ya creada
(ver tablas.sql), más el modelo de embeddings descargado:
    ollama pull nomic-embed-text

Instala dependencias nuevas:
    pip install psycopg2-binary pgvector python-dotenv
"""

from __future__ import annotations

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

    Se embebe este resumen, no el JSON crudo: los embeddings capturan mejor
    el significado en lenguaje natural, y un texto siempre generado con el
    mismo patrón da comparaciones más consistentes entre pacientes que
    comparar JSON directamente.
    """
    d, b = ficha.diagnostico, ficha.biomarcadores
    partes = [f"Diagnóstico: {d.diagnostico_actual.value}.", f"APOE4: {d.apoe4.value}."]
    if b.tau_pg_ml is not None:
        partes.append(f"Tau: {b.tau_pg_ml} pg/mL.")
    if b.abeta42_pg_ml is not None:
        partes.append(f"Abeta42: {b.abeta42_pg_ml} pg/mL.")
    if b.hipocampo_mm3 is not None:
        partes.append(f"Volumen hipocampal: {b.hipocampo_mm3} mm3.")
    return " ".join(partes)


def generar_embedding(texto: str) -> list[float]:
    respuesta = ollama.embed(model=MODELO_EMBEDDING, input=texto)
    return respuesta["embeddings"][0]


def _conectar():
    conexion = psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="tfm_pacientes",
        user="tfm_user",
        password=os.environ["POSTGRES_PASSWORD"],
    )
    register_vector(conexion)  # enseña a psycopg2 a traducir listas <-> vector
    return conexion


def guardar_paciente(ficha: FichaPaciente, paciente_id: str | None = None) -> str:
    """Embebe la ficha y la guarda en la tabla `pacientes`. Devuelve el id usado."""
    paciente_id = paciente_id or str(uuid.uuid4())
    resumen = ficha_a_texto(ficha)
    embedding = generar_embedding(resumen)

    conexion = _conectar()
    with conexion, conexion.cursor() as cur:
        cur.execute(
            """
            INSERT INTO pacientes (paciente_id, ficha, resumen_texto, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (paciente_id) DO NOTHING
            """,
            (paciente_id, ficha.model_dump_json(), resumen, embedding),
        )
    conexion.close()
    return paciente_id


def pacientes_similares(ficha: FichaPaciente, k: int = 5) -> list[tuple]:
    """Devuelve los k pacientes ya guardados más parecidos a esta ficha
    (paciente_id, resumen_texto, distancia). Distancia menor = más parecido.
    Es la base del panel "pacientes similares" del dashboard.
    """
    embedding = generar_embedding(ficha_a_texto(ficha))

    conexion = _conectar()
    with conexion, conexion.cursor() as cur:
        cur.execute(
            """
            SELECT paciente_id, resumen_texto, embedding <=> %s AS distancia
            FROM pacientes
            ORDER BY distancia
            LIMIT %s
            """,
            (embedding, k),
        )
        resultados = cur.fetchall()
    conexion.close()
    return resultados


if __name__ == "__main__":
    from extraccion_paciente import extraer_ficha_paciente

    informe_ejemplo = """
    Paciente varón, diagnóstico MCI. APOE4 positivo. Tau en LCR: 320 pg/mL.
    Volumen hipocampal reducido, compatible con atrofia leve.
    """
    ficha, _ = extraer_ficha_paciente(informe_ejemplo)
    if ficha:
        pid = guardar_paciente(ficha)
        print(f"Paciente guardado con id {pid}")
        print("Pacientes similares:", pacientes_similares(ficha))
    else:
        print("No se pudo extraer la ficha de ejemplo")
