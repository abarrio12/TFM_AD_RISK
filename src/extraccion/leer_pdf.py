"""
Extrae el texto de un PDF real (el informe que sube el médico) para
pasarlo al pipeline de extracción de la sección 3.2.

Es la pieza que faltaba entre "el médico sube su PDF" y "el LLM extrae
la ficha": hasta ahora, el pipeline solo se había probado con texto
Python escrito a mano (informe_ejemplo). Este módulo cierra ese hueco.

Alcance: PDFs con texto digital (los de las 2-3 plantillas representativas
del TFM, sección 1.3). Los PDFs escaneados (solo imagen, sin capa de
texto) necesitarían OCR -- una pieza distinta y más pesada, fuera del
alcance acotado del trabajo. Este módulo detecta ese caso y avisa en
lugar de fallar en silencio o pasarle texto vacío al LLM.

Requisitos:
    pip install pdfplumber
"""

from __future__ import annotations

import re
import pdfplumber

LONGITUD_MINIMA_TEXTO = 50  # por debajo de esto, probablemente el PDF es escaneado
MAXIMO_TEXTO_LLM = 12000  # evita enviar informes completos de cientos de páginas al modelo

# Palabras clave médicas relevantes para el diagnóstico de Alzheimer
PALABRAS_CLAVE = [
    "mmse", "moca", "mini-cog",  # Tests cognitivos
    "tau", "ptau", "abeta", "ab40", "ab42", "plasma",  # Biomarcadores
    "hipocampo", "ventrícul", "atrofia", "volumen",  # Neuroimagen
    "apoe", "genétipo", "diagnóstico", "impresión",  # Diagnóstico
    "alzheimer", "mci", "demencia", "cognitiv",  # Estados clínicos
    "edad", "sexo", "género", "antecedentes",  # Datos demográficos
    "positivo", "negativo", "normal", "anormal",  # Resultados
]


def extraer_fragmentos_con_palabras_clave(
    texto: str, ventana_caracteres: int = 500, maximo_caracteres: int = MAXIMO_TEXTO_LLM
) -> str:
    """Busca palabras clave médicas en el texto y extrae fragmentos contextuales.
    
    Args:
        texto: Texto completo del PDF
        ventana_caracteres: Caracteres antes/después de cada palabra clave
    
    Returns:
        Fragmentos del texto que contienen palabras clave, o el texto original
        si es corto o no hay palabras clave (para no perder información).
    """
    
    # Si el texto es corto, no vale la pena filtrar
    if len(texto) < 2000:
        return texto
    
    fragmentos_indices = set()
    texto_lower = texto.lower()
    
    # Buscar cada palabra clave y extraer contexto
    for palabra_clave in PALABRAS_CLAVE:
        # Buscar todas las ocurrencias de la palabra clave
        for match in re.finditer(r'\b' + re.escape(palabra_clave) + r'\w*', texto_lower):
            inicio = max(0, match.start() - ventana_caracteres)
            fin = min(len(texto), match.end() + ventana_caracteres)
            fragmentos_indices.add((inicio, fin))
    
    # Si no hay palabras clave, conservar el principio y el final, donde suelen
    # aparecer identificación, impresión diagnóstica y conclusiones.
    if not fragmentos_indices:
        if len(texto) <= maximo_caracteres:
            return texto
        mitad = maximo_caracteres // 2
        return texto[:mitad] + "\n\n[...] CONTENIDO OMITIDO [...]\n\n" + texto[-mitad:]
    
    # Ordenar fragmentos y fusionar los solapados
    fragmentos = sorted(list(fragmentos_indices))
    fragmentos_fusionados = []
    
    for inicio, fin in fragmentos:
        if fragmentos_fusionados and inicio <= fragmentos_fusionados[-1][1]:
            # Solapamiento: fusionar
            fragmentos_fusionados[-1] = (fragmentos_fusionados[-1][0], max(fin, fragmentos_fusionados[-1][1]))
        else:
            fragmentos_fusionados.append((inicio, fin))
    
    # Construir el texto resultado
    resultado_partes = []
    caracteres = 0
    for inicio, fin in fragmentos_fusionados:
        fragmento = texto[inicio:fin]
        separador = "\n\n[...]\n\n" if resultado_partes else ""
        if caracteres + len(separador) + len(fragmento) > maximo_caracteres:
            espacio = maximo_caracteres - caracteres - len(separador)
            if espacio > 100:
                resultado_partes.append(separador + fragmento[:espacio])
            break
        resultado_partes.append(separador + fragmento)
        caracteres += len(separador) + len(fragmento)

    resultado = "".join(resultado_partes)
    
    porcentaje = round(len(resultado) / len(texto) * 100, 1)
    print(
        f"Extracción de palabras clave: {porcentaje}% del texto original "
        f"({len(resultado)}/{len(texto)} caracteres; máximo {maximo_caracteres})."
    )
    
    return resultado


def extraer_texto_pdf(ruta_pdf: str) -> str | None:
    """Devuelve el texto de todas las páginas del PDF, extrayendo fragmentos
    con palabras clave médicas para optimizar el procesamiento posterior.
    
    Devuelve None si no se pudo extraer texto suficiente (probable PDF 
    escaneado, sin capa de texto).
    """
    partes = []
    
    with pdfplumber.open(ruta_pdf) as pdf:
        for pagina in pdf.pages:
            texto_pagina = pagina.extract_text()
            if texto_pagina:
                partes.append(texto_pagina)

    texto_completo = "\n".join(partes).strip()

    if len(texto_completo) < LONGITUD_MINIMA_TEXTO:
        print(
            f"Aviso: '{ruta_pdf}' devolvió muy poco texto ({len(texto_completo)} caracteres). "
            f"Es probable que sea un PDF escaneado (solo imagen, sin capa de texto), "
            f"que este pipeline no soporta -- necesitaría OCR, fuera del alcance del TFM."
        )
        return None

    # Extraer fragmentos con palabras clave para optimizar el procesamiento
    texto_optimizado = extraer_fragmentos_con_palabras_clave(texto_completo)
    
    return texto_optimizado


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python leer_pdf.py ruta/al/informe.pdf")
        sys.exit(1)

    texto = extraer_texto_pdf(sys.argv[1])
    if texto:
        print(f"Extraídos {len(texto)} caracteres:\n")
        print(texto[:500] + ("..." if len(texto) > 500 else ""))