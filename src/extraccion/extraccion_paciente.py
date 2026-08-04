"""
Módulo de extracción estructurada de informes clínicos mediante LLM local (Ollama).

Punto del índice del TFM que cubre: 3.2 (Estrategia de Prompt Engineering)
y sienta la base de 3.4 (monitorización/depuración ETL) y 6.1 (evaluación
frente a gold set).

Requisitos:
    pip install ollama pydantic
    ollama pull qwen3:8b     # o llama3.2:3b para pruebas rápidas en hardware modesto
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import ollama
from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extraccion_etl")  # alimenta el 3.4 (monitorización ETL)


# --------------------------------------------------------------------------
# Esquema: mismo diseño del JSON de la sección 3.1, ahora en Pydantic.
# Cada bloque lleva su propio campo de confianza (0-1). Faltan aquí
# "demograficos" y "cognitivo" por brevedad, pero siguen el mismo patrón
# exacto que "diagnostico" y "biomarcadores": campos + confianza por bloque.
# --------------------------------------------------------------------------

class DiagnosticoEnum(str, Enum):
    CN = "CN"
    MCI = "MCI"
    AD = "AD"


class APOE4Enum(str, Enum):
    POSITIVO = "Positivo"
    NEGATIVO = "Negativo"
    DESCONOCIDO = "Desconocido"


class Diagnostico(BaseModel):
    diagnostico_actual: DiagnosticoEnum
    apoe4: APOE4Enum
    confianza: float = Field(ge=0, le=1)


class Biomarcadores(BaseModel):
    tau_pg_ml: Optional[float] = None
    abeta42_pg_ml: Optional[float] = None
    hipocampo_mm3: Optional[float] = None
    confianza: float = Field(ge=0, le=1)


class FichaPaciente(BaseModel):
    diagnostico: Diagnostico
    biomarcadores: Biomarcadores
    revision_humana_requerida: bool


# --------------------------------------------------------------------------
# Prompt: reglas explícitas para minimizar alucinación. Ver 3.2 en la
# memoria para la justificación de cada regla.
# --------------------------------------------------------------------------

PROMPT_SISTEMA = """Eres un sistema de extracción de datos clínicos. Tu única tarea \
es leer el texto de un informe médico y devolver los campos del esquema JSON dado.

Reglas estrictas:
1. Extrae solo lo que esté escrito explícitamente en el texto. No infieras ni \
completes con conocimiento médico general.
2. Si un dato no aparece en el texto, el campo queda en null. Nunca inventes \
un valor.
3. Para cada bloque, estima tu confianza de 0 a 1 según la claridad del texto.
4. Si el texto es ambiguo o hay dos valores posibles para un mismo campo, marca \
revision_humana_requerida como true.
5. No añadas texto fuera del JSON. No expliques tu razonamiento.
"""


# --------------------------------------------------------------------------
# Extracción con validación y reintentos.
# Ollama no garantiza que la respuesta cumpla el schema al 100% (puede
# cortar la generación a mitad de un campo), así que el reintento con el
# error inyectado de vuelta al modelo es la pieza que da fiabilidad real.
# --------------------------------------------------------------------------

def extraer_ficha_paciente(
    texto_informe: str,
    modelo: str = "qwen3:8b",
    max_intentos: int = 3,
) -> tuple[Optional[FichaPaciente], int]:
    """Extrae una FichaPaciente a partir del texto libre de un informe.

    Devuelve (ficha, intentos_usados). Si ficha es None tras max_intentos,
    el documento queda para revisión manual: ese conteo es exactamente el
    dato que alimenta la evaluación frente al gold set (sección 6.1).
    """
    mensajes = [
        {"role": "system", "content": PROMPT_SISTEMA},
        {"role": "user", "content": texto_informe},
    ]

    for intento in range(1, max_intentos + 1):
        respuesta = ollama.chat(
            model=modelo,
            messages=mensajes,
            format=FichaPaciente.model_json_schema(),
            options={"temperature": 0},  # determinismo: mismo informe -> misma salida
        )
        contenido = respuesta["message"]["content"]

        try:
            ficha = FichaPaciente.model_validate_json(contenido)
            logger.info("Extracción OK en intento %d/%d", intento, max_intentos)
            return ficha, intento
        except ValidationError as error:
            logger.warning("Intento %d/%d inválido: %s", intento, max_intentos, error)
            mensajes.append({"role": "assistant", "content": contenido})
            mensajes.append({
                "role": "user",
                "content": (
                    f"La respuesta no cumple el esquema: {error}. "
                    f"Corrige y devuelve solo el JSON corregido."
                ),
            })

    logger.error("Extracción fallida tras %d intentos; marcar para revisión manual", max_intentos)
    return None, max_intentos


if __name__ == "__main__":
    informe_ejemplo = """
    Paciente varón, diagnóstico MCI. APOE4 positivo. Tau en LCR: 320 pg/mL.
    Volumen hipocampal reducido, compatible con atrofia leve.
    """
    ficha, intentos = extraer_ficha_paciente(informe_ejemplo)
    if ficha:
        print(ficha.model_dump_json(indent=2))
    else:
        print("Revisión manual requerida")
