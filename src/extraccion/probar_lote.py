"""
Procesa un lote de informes de ejemplo de una vez: da variedad real a la
base de datos (hasta ahora solo tenía un paciente) y empieza a poner a
prueba cómo se comporta la extracción con datos ambiguos o ausentes --
algo que el prompt de 3.2 contempla mediante revision_humana_requerida,
pero que aún no habíamos comprobado con ningún caso real.

Uso: python probar_lote.py
"""

from embeddings_paciente import procesar_informe

INFORMES_EJEMPLO = [
    # 1. CN claro
    """Paciente varón, 68 años. Evaluación cognitiva dentro de la
    normalidad, MMSE 29/30. Diagnóstico: CN (control sano). APOE4
    negativo. Biomarcadores en rango normal: Tau 180 pg/mL, Abeta42 950
    pg/mL. Resonancia sin signos de atrofia significativa, volumen
    hipocampal conservado.""",

    # 2. MCI con APOE4 poco claro en el texto -- a ver si lo detecta
    """Paciente mujer, 74 años. Quejas subjetivas de memoria, MMSE 26/30.
    Diagnóstico: MCI. Tau ligeramente elevada, 290 pg/mL. Hipocampo en el
    límite bajo de la normalidad para su edad.""",

    # 3. AD claro, distinto del que ya tienes guardado
    """Paciente varón, 79 años. Deterioro cognitivo evidente, MMSE 18/30.
    Diagnóstico: AD. APOE4 positivo (homocigoto e4/e4). Tau muy elevada:
    540 pg/mL. Abeta42 reducido: 410 pg/mL. Atrofia hipocampal marcada en
    ambos hemisferios.""",

    # 4. MCI con biomarcadores explícitamente NO disponibles
    """Paciente mujer, 71 años. MMSE 27/30, rendimiento cognitivo
    levemente por debajo de lo esperado. Diagnóstico: MCI. APOE4
    negativo. Biomarcadores de LCR no disponibles en este informe.
    Volumen hipocampal ligeramente reducido.""",

    # 5. CN con informe corto e informal (distinto estilo de redacción)
    """Varón, 65 años, revisión rutinaria neurológica. Sin quejas
    cognitivas. MMSE 30/30. Todo normal, sin hallazgos relevantes.
    APOE4: negativo.""",
]


if __name__ == "__main__":
    for i, informe in enumerate(INFORMES_EJEMPLO, start=1):
        print(f"\n--- Informe {i}/{len(INFORMES_EJEMPLO)} ---")
        pid, ficha = procesar_informe(informe)
        if ficha:
            print(f"  Diagnóstico extraído: {ficha.diagnostico.diagnostico_actual.value}")
            print(f"  Revisión humana requerida: {ficha.revision_humana_requerida}")
        else:
            print("  No se pudo procesar este informe")

    print("\nListo. Comprueba la tabla completa con la extensión de PostgreSQL, o:")
    print('  docker exec -it tfm_postgres psql -U tfm_user -d tfm_pacientes -c "SELECT paciente_id, resumen_texto FROM pacientes;"')