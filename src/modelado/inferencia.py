"""
Conector entre la ficha extraída de un informe (sección 3) y el modelo
entrenado (sección 4): traduce una FichaPaciente a la fila de variables
exacta que espera el clasificador, y avisa de qué falta.

Punto del índice que cubre: 5.2 (simulación del flujo de un paciente
nuevo)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "extraccion"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import numpy as np
import pandas as pd

from extraccion_paciente import FichaPaciente
from clasificacion import VARIABLES_MODELO, apoe4_a_feature

COLUMNAS_MODELO = VARIABLES_MODELO + ["apoe4_positivo"]


def ficha_a_features(ficha: FichaPaciente) -> pd.DataFrame:
    """Convierte una FichaPaciente en la fila de variables que espera el
    modelo entrenado -- mismas columnas y mismo orden que en 4.3.2.

    Los biomarcadores de PLASMA (ptau217_plasma_pg_ml,
    ratio_ab42_ab40_plasma) quedan en null para pacientes extraídos de
    PDF: los informes de casos clínicos que hemos usado de referencia
    casi nunca especifican el ensayo de plasma concreto -- casi siempre
    dan datos de LCR. Es una simplificación de alcance consciente (ver
    4.1), no un descuido; el modelo admite estos huecos de forma nativa.
    """
    b = ficha.biomarcadores
    fila = {
        "edad": ficha.edad,
        "mmse": ficha.mmse,
        "hipocampo_mm3": b.hipocampo_mm3,
        "tau_pg_ml": b.tau_pg_ml,
        "abeta42_pg_ml": b.abeta42_pg_ml,
        "ptau217_plasma_pg_ml": np.nan,
        "ratio_ab42_ab40_plasma": np.nan,
        "apoe4_positivo": apoe4_a_feature(ficha.diagnostico.apoe4.value),
    }
    # None -> NaN no basta por sí solo si una columna queda entera a None
    # (pandas la deja en dtype "object", y XGBoost la rechaza). Se fuerza
    # float explícitamente para las ocho columnas, que son todas numéricas.
    fila_df = pd.DataFrame([fila])[COLUMNAS_MODELO]
    return fila_df.astype(float)


def campos_faltantes(fila: pd.DataFrame) -> list[str]:
    """Qué variables del modelo no se pudieron rellenar desde el PDF --
    para avisar en el dashboard de que esta predicción concreta se apoya
    en menos información de la habitual.
    """
    primera = fila.iloc[0]
    return [c for c in fila.columns if pd.isna(primera[c])]


if __name__ == "__main__":
    from extraccion_paciente import Diagnostico, Biomarcadores

    print("--- Caso 1: ficha completa (informe rico en datos) ---")
    ficha_completa = FichaPaciente(
        diagnostico=Diagnostico(diagnostico_actual="AD", apoe4="Positivo", confianza=0.9),
        edad=79,
        mmse=18,
        biomarcadores=Biomarcadores(
            tau_pg_ml=830.0, p_tau_pg_ml=127.0, abeta42_pg_ml=392.0, confianza=0.9,
        ),
        revision_humana_requerida=False,
    )
    fila1 = ficha_a_features(ficha_completa)
    print(fila1.to_string(index=False))
    print("Campos que faltan:", campos_faltantes(fila1))

    print("\n--- Caso 2: ficha parcial (solo edad, MMSE, hipocampo, APOE4) ---")
    ficha_parcial = FichaPaciente(
        diagnostico=Diagnostico(diagnostico_actual="MCI", apoe4="Desconocido", confianza=0.6),
        edad=71,
        mmse=27,
        biomarcadores=Biomarcadores(hipocampo_mm3=3400.0, confianza=0.5),
        revision_humana_requerida=True,
    )
    fila2 = ficha_a_features(ficha_parcial)
    print(fila2.to_string(index=False))
    print("Campos que faltan:", campos_faltantes(fila2))
