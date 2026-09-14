import time
import sys

sys.path.insert(0, '.')
from extraccion_paciente import extraer_ficha_paciente

texto = """
INFORME CLINICO - UNIDAD DE MEMORIA
Caso de prueba con datos ficticios, generado para testeo del sistema de prediccion
Sexo: Mujer  Edad: 74 anios

Motivo de consulta
Paciente remitida por quejas subjetivas de memoria de varios meses de evolucion, sin repercusion funcional relevante en las actividades basicas de la vida diaria.

Exploracion neuropsicologica
Mini-Mental State Examination (MMSE): 26/30.

Biomarcadores en LCR
Tau total: 290 pg/mL.

Neuroimagen (RM craneal)
Volumen hipocampal en el limite bajo de la normalidad para la edad de la paciente, sin otros hallazgos relevantes.

Impresion diagnostica
Deterioro cognitivo leve (Mild Cognitive Impairment, MCI) de perfil amnesico
"""

print("Mismo texto exacto, esta vez con qwen3:8b...")
inicio = time.time()
ficha, intentos = extraer_ficha_paciente(texto, modelo="qwen3:8b")
print(f"\nTerminado en {time.time() - inicio:.1f} segundos, {intentos} intento(s)")
print(ficha.model_dump_json(indent=2) if ficha else "FALLO: ninguna ficha valida")