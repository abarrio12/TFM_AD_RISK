"""
Módulo de extracción estructurada de informes clínicos mediante LLM local (Ollama).

Punto del índice del TFM que cubre: 3.2 (Estrategia de Prompt Engineering)
y sienta la base de 3.4 (monitorización/depuración) y 6.1 (evaluación
frente a gold set).

Principio de diseño del esquema: los campos "core" (diagnóstico,
biomarcadores clásicos) son los que razonablemente coincidirán con
columnas reales de ADNI, porque son los que el modelo de la
sección 4 puede llegar a usar. Todo lo demás que aparezca en un informe
real pero no tenga equivalente en el dataset de entrenamiento va a
hallazgos_adicionales: enriquece el texto que se embebe (3.3) y lo que
ve el médico en el dashboard (5), pero no lo consume el clasificador.

Requisitos:
    pip install ollama pydantic
    ollama pull llama3.2:3b     # o qwen3:8b si el hardware lo permite
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Optional

import ollama
from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extraccion_etl")  # alimenta el 3.4 (monitorización ETL)


# --------------------------------------------------------------------------
# Esquema
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
    p_tau_pg_ml: Optional[float] = None  # tau FOSFORILADA -- distinta de tau total
    abeta42_pg_ml: Optional[float] = None
    hipocampo_mm3: Optional[float] = None
    atrofia_grado: Optional[str] = None  # alternativa cualitativa (p. ej. "grado 2"),
                                          # NO una conversión de hipocampo_mm3
    confianza: float = Field(ge=0, le=1)


class FichaPaciente(BaseModel):
    diagnostico: Diagnostico
    edad: Optional[int] = None  # edad del paciente EN el momento de este informe
    mmse: Optional[float] = None  # específicamente MMSE; otras pruebas van a hallazgos_adicionales
    biomarcadores: Biomarcadores
    hallazgos_adicionales: list[str] = Field(default_factory=list)
    # cajón acotado: cosas clínicamente relevantes sin campo propio
    # (14-3-3, PET amiloide, cortisol...). Alimenta el texto embebido
    # (3.3) y el dashboard (5); NO lo consume el clasificador (4.3.2).
    revision_humana_requerida: bool


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

PROMPT_SISTEMA = """Eres un sistema de extracción de datos clínicos. Tu única tarea \
es leer el texto de un informe médico y devolver los campos del esquema JSON dado.

Reglas estrictas:
1. Extrae solo lo que esté escrito explícitamente en el texto. No infieras ni \
completes con conocimiento médico general.
2. Si un dato numérico o de texto libre no aparece en el texto, el campo \
queda en null. Nunca inventes un valor.
3. Para campos que solo aceptan un conjunto cerrado de valores (como \
APOE4: Positivo/Negativo/Desconocido) y el texto no dice cuál es, usa \
"Desconocido" -- nunca asumas Positivo o Negativo sin que el texto lo \
afirme explícitamente.
4. mmse: si el texto contiene "MMSE" seguido de un número, en cualquier \
formato -- "MMSE: 26/30", "MMSE 26", "(MMSE): 26/30" son todos el mismo \
dato -- pon ese número en mmse.
5. Otras pruebas cognitivas distintas de MMSE (CDR, CDR-SB, MoCA, ACE-R, \
PHQ-9...) van a hallazgos_adicionales con su nombre y puntuación -- \
nunca a mmse.
6. edad es la edad del paciente EN EL MOMENTO de este informe o visita. \
Si el texto distingue edad actual de edad de inicio de los síntomas, usa \
la edad actual -- la de inicio, si se menciona, va a hallazgos_adicionales.
7. La atrofia del hipocampo puede venir como volumen (hipocampo_mm3) o \
como grado/escala cualitativa (atrofia_grado, p. ej. "grado 2"). No \
conviertas de uno a otro: si el informe da un grado, ponlo en \
atrofia_grado y deja hipocampo_mm3 en null.
8. tau_pg_ml es tau TOTAL; p_tau_pg_ml es tau FOSFORILADA -- son dos \
biomarcadores distintos. No pongas el mismo número en ambos si el \
informe solo da uno de los dos.
9. hallazgos_adicionales recoge cualquier otro dato clínicamente \
relevante que no encaje en los campos anteriores -- otras pruebas \
cognitivas, hallazgos genéticos, PET, proteínas no listadas arriba, etc. \
-- como notas breves en texto libre.
10. Para cada bloque (diagnóstico, biomarcadores), estima tu confianza \
de 0 a 1 según la claridad del texto.
11. Si el texto es ambiguo, o algún campo cerrado queda en "Desconocido", \
marca revision_humana_requerida como true.
12. No añadas texto fuera del JSON. No expliques tu razonamiento.
"""


# --------------------------------------------------------------------------
# Extracción con validación y reintentos
# --------------------------------------------------------------------------

def _mmse_por_regex(texto: str) -> Optional[float]:
    """Red de seguridad determinista: el LLM local falla en extraer MMSE
    de forma reproducible (documentado en 6.1/6.2, confirmado incluso con
    GPU y con la forma más simple "MMSE: 26/30"). Si el campo estructurado
    queda en null pero el texto sí contiene el patrón, se recupera con
    una expresión regular directa -- no sustituye a la extracción por
    LLM, es un parche barato para un fallo ya bien documentado.
    """
    coincidencia = re.search(r"MMSE[^\d]{0,15}(\d{1,2})(?:\s*/\s*30)?", texto, re.IGNORECASE)
    if coincidencia:
        valor = float(coincidencia.group(1))
        if 0 <= valor <= 30:
            return valor
    return None


def extraer_ficha_paciente(
    texto_informe: str,
    modelo: str = "llama3.2:3b",
    max_intentos: int = 3,
    timeout_segundos: int = 300,
    host: Optional[str] = None,
) -> tuple[Optional[FichaPaciente], int]:
    """Extrae una FichaPaciente a partir del texto libre de un informe.

    Devuelve (ficha, intentos_usados). Si ficha es None tras max_intentos,
    el documento queda para revisión manual: ese conteo alimenta la
    evaluación frente al gold set (sección 6.1).

    timeout_segundos: 300 por defecto -- corregido tras comprobar en la
    práctica que 180s se quedaba corto. La prueba de referencia (~70s)
    era con texto libre; exigir que la salida encaje en un esquema JSON
    es más lento de generar, y con un informe largo (no un "di hola")
    el margen real necesario es mayor de lo que la primera estimación
    sugería.

    host: None usa Ollama local (http://localhost:11434, por defecto de
    la librería). Pasa la URL de un túnel (p. ej. de ngrok en Colab) para
    usar Ollama con GPU en vez de la CPU local -- ver notebook de Colab.
    """
    mensajes = [
        {"role": "system", "content": PROMPT_SISTEMA},
        {"role": "user", "content": texto_informe},
    ]
    ultimo_error: Optional[str] = None

    for intento in range(1, max_intentos + 1):
        logger.info("Intento %d/%d: llamando a Ollama (modelo=%s)...", intento, max_intentos, modelo)
        if ultimo_error is not None:
            # Solo se añade el error, no la respuesta inválida completa de
            # antes -- si no, cada reintento reprocesa una conversación más
            # larga que la anterior, y un informe largo se vuelve cada vez
            # más lento en vez de tardar más o menos lo mismo cada vez.
            mensajes.append({
                "role": "user",
                "content": f"Tu respuesta anterior no cumplía el esquema: {ultimo_error}. Devuelve solo el JSON corregido.",
            })

        # Cliente (y conexión) nuevos en cada intento -- no se reutiliza el de
        # antes. Con esperas de varios minutos (normal aquí), una conexión ya
        # abierta puede quedar en mal estado y fallar con un error de
        # descifrado SSL en vez de un fallo de red limpio; una conexión
        # nueva evita heredar ese estado.
        # ngrok-skip-browser-warning: sin esto, el plan gratuito de ngrok
        # devuelve una pagina HTML de aviso en vez de la respuesta real de
        # Ollama. No afecta nada cuando host es local.
        if host:
            cliente = ollama.Client(host=host, timeout=timeout_segundos, headers={"ngrok-skip-browser-warning": "true"})
        else:
            cliente = ollama.Client(timeout=timeout_segundos)

        try:
            respuesta = cliente.chat(
                model=modelo,
                messages=mensajes,
                format=FichaPaciente.model_json_schema(),
                keep_alive="30m",  # evita que Ollama descargue el modelo entre llamadas seguidas
                options={"temperature": 0},
            )
        except Exception as error_conexion:
            # Fallo de red/TLS, no de formato -- no tiene sentido pedirle al
            # LLM que "corrija" nada (nunca llegó a responder), así que no se
            # añade mensaje de corrección; solo se reintenta con conexión nueva.
            logger.warning(
                "Intento %d/%d: fallo de conexión (%s). Reintentando con una conexión nueva...",
                intento, max_intentos, error_conexion,
            )
            if intento == max_intentos:
                logger.error("Extracción fallida tras %d intentos por fallo de conexión persistente", max_intentos)
                return None, max_intentos
            continue

        contenido = respuesta["message"]["content"]

        try:
            ficha = FichaPaciente.model_validate_json(contenido)
            if ficha.mmse is None:
                mmse_recuperado = _mmse_por_regex(texto_informe)
                if mmse_recuperado is not None:
                    ficha = ficha.model_copy(update={"mmse": mmse_recuperado})
                    ficha.revision_humana_requerida = True  # el LLM no lo vio; que un humano lo confirme
                    logger.info("mmse recuperado por regex (el LLM lo dejó en null): %s", mmse_recuperado)
            logger.info("Extracción OK en intento %d/%d", intento, max_intentos)
            return ficha, intento
        except ValidationError as error:
            logger.warning("Intento %d/%d inválido: %s", intento, max_intentos, error)
            ultimo_error = str(error)[:500]  # recortado: no hace falta el detalle completo para corregir

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