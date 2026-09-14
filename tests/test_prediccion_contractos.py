import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).parents[1]
sys.path.insert(0, str(RAIZ / "src" / "modelado"))
sys.path.insert(0, str(RAIZ / "src" / "dashboard"))

from clasificacion import apoe4_a_feature, preparar_features, probabilidades_escenario_mmse
from reportes import obtener_papers_relevantes


def test_apoe_desconocido_no_se_codifica_como_negativo():
    assert np.isnan(apoe4_a_feature("Desconocido"))
    assert np.isnan(apoe4_a_feature(None))
    assert apoe4_a_feature("Negativo") == 0.0
    assert apoe4_a_feature("Positivo") == 1.0

    datos = pd.DataFrame({
        "edad": [60, 70, 75],
        "mmse": [25, 20, 28],
        "apoe4": ["Desconocido", "Positivo", "Negativo"],
        "diagnostico": ["MCI", "AD", "CN"],
    })
    features, _, _ = preparar_features(datos)

    assert pd.isna(features.iloc[0]["apoe4_positivo"])
    assert features.iloc[1]["apoe4_positivo"] == 1.0
    assert features.iloc[2]["apoe4_positivo"] == 0.0


class _ModeloEscenario:
    def predict_proba(self, filas):
        mmse = filas["mmse"].to_numpy()
        riesgo_ad = np.clip(0.95 - 0.02 * mmse, 0.1, 0.95)
        riesgo_mci = np.clip(0.1 + 0.01 * mmse, 0.05, 0.4)
        normalidad = 1 - riesgo_ad - riesgo_mci
        return np.column_stack([riesgo_ad, normalidad, riesgo_mci])


class _CodificadorEscenario:
    classes_ = np.array(["AD", "CN", "MCI"])


def test_escenario_mmse_no_aumenta_riesgo_deterioro():
    columnas = ["edad", "mmse", "apoe4_positivo"]
    fila = pd.DataFrame([[60.0, 20.0, np.nan]], columns=columnas)
    modelo = _ModeloEscenario()
    codificador = _CodificadorEscenario()
    indices_deterioro = [0, 2]

    riesgos = [
        probabilidades_escenario_mmse(modelo, fila, columnas, codificador, mmse)[indices_deterioro].sum()
        for mmse in range(31)
    ]

    assert all(anterior >= siguiente for anterior, siguiente in zip(riesgos, riesgos[1:]))
    assert all(np.isclose(probabilidades_escenario_mmse(modelo, fila, columnas, codificador, mmse).sum(), 1.0) for mmse in range(31))


def test_literatura_sin_biomarcadores_no_anade_tau(monkeypatch):
    consultas = []

    def fake_get_json(ruta, params):
        consultas.append(params["term"])
        return {"esearchresult": {"idlist": []}}

    monkeypatch.setattr("reportes._get_json", fake_get_json)
    assert obtener_papers_relevantes([], "MCI") == []
    assert consultas == ['"mild cognitive impairment"[Title/Abstract]']