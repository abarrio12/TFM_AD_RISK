"""
Submódulo de clasificación: modelo predictivo de riesgo con XGBoost,
evaluación y explicabilidad con SHAP.

Punto del índice que cubre: 4.3.2 (entrenamiento) y 4.4 (evaluación),
más 5.3 (XAI) en cuanto a la parte de SHAP.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin ventana -- solo guardamos a fichero, ver eda.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder

CARPETA_RESULTADOS_DEFECTO = str(Path(__file__).resolve().parent.parent.parent / "resultados")

VARIABLES_MODELO = [
    "edad", "mmse", "hipocampo_mm3",
    "tau_pg_ml", "abeta42_pg_ml",              # biomarcadores LCR
    "ptau217_plasma_pg_ml", "ratio_ab42_ab40_plasma",  # biomarcadores plasma
]
# apoe4 se añade aparte por ser categórica (Positivo/Negativo/Desconocido)


def apoe4_a_feature(valor) -> float:
    """Codifica APOE4 conservando la ausencia de información como NaN."""
    if pd.isna(valor) or str(valor).strip().lower() in {"desconocido", "unknown", ""}:
        return float("nan")
    return 1.0 if str(valor).strip().lower() == "positivo" else 0.0


def preparar_features(df: pd.DataFrame):
    """Solo exige diagnóstico y edad/mmse (>99% completos). El resto de
    variables -- sobre todo los biomarcadores, con huecos grandes y NO
    solapados entre sí (LCR vs plasma) -- se dejan con sus NaN: XGBoost
    los maneja de forma nativa (aprende hacia qué lado mandar un valor
    ausente en cada división del árbol), lo que aprovecha muchos más
    pacientes que forzar que todos tengan todo relleno.
    """
    columnas_esenciales = [c for c in ["edad", "mmse"] if c in df.columns]
    datos = df.dropna(subset=columnas_esenciales).copy()
    # filtro ya presente en main.py pero se agrega aqui por dejarlo consistente
    # para futuras ejecuciones
    datos = datos[datos["diagnostico"] != "Desconocido"] 
    datos["apoe4_positivo"] = datos["apoe4"].apply(apoe4_a_feature)

    columnas_modelo = [c for c in VARIABLES_MODELO if c in datos.columns]
    X = datos[columnas_modelo + ["apoe4_positivo"]]
    codificador = LabelEncoder()
    y = pd.Series(codificador.fit_transform(datos["diagnostico"]), index=X.index)  # CN/MCI/AD -> 0/1/2
    return X, y, codificador


def entrenar_modelo(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    modelo_base = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        eval_metric="mlogloss", random_state=42,
    )
    modelo_base.fit(X_train, y_train)

    # XGBoost separa bien algunos perfiles, pero sus probabilidades de árbol
    # suelen ser demasiado extremas para interpretarlas como confianza.
    modelo = CalibratedClassifierCV(modelo_base, method="sigmoid", cv=5)
    modelo.fit(X_train, y_train)
    modelo.modelo_base = modelo_base
    modelo.feature_importances_ = modelo_base.feature_importances_
    return modelo, X_train, X_test, y_train, y_test


def probabilidades_escenario_mmse(modelo, fila: pd.DataFrame, columnas_modelo: list[str], codificador, mmse: int):
    """Calcula un escenario MMSE monotónico a partir de la salida del modelo.

    La predicción base sigue siendo la salida directa de XGBoost. Para el
    escenario se ajusta únicamente el riesgo combinado MCI/AD con una
    envolvente isotónica decreciente: al aumentar MMSE no puede aumentar el
    riesgo de deterioro. La proporción entre MCI y AD se conserva desde la
    salida original en cada punto.
    """
    clases = list(codificador.classes_)
    indices_deterioro = [clases.index(clase) for clase in ("MCI", "AD") if clase in clases]
    if not indices_deterioro:
        return modelo.predict_proba(fila[columnas_modelo])[0]

    valores_mmse = np.arange(31, dtype=float)
    filas_escenario = pd.concat(
        [fila.assign(mmse=valor) for valor in valores_mmse],
        ignore_index=True,
    )
    probabilidades_brutas = modelo.predict_proba(filas_escenario[columnas_modelo])
    riesgo_bruto = probabilidades_brutas[:, indices_deterioro].sum(axis=1)
    riesgo_monotono = IsotonicRegression(
        increasing=False, y_min=0.0, y_max=1.0, out_of_bounds="clip"
    ).fit_transform(valores_mmse, riesgo_bruto)

    indice_mmse = int(np.clip(mmse, 0, 30))
    proporcion = probabilidades_brutas[indice_mmse, indices_deterioro]
    suma_proporcion = proporcion.sum()
    if suma_proporcion == 0:
        proporcion = np.full(len(indices_deterioro), 1 / len(indices_deterioro))
    else:
        proporcion = proporcion / suma_proporcion

    resultado = probabilidades_brutas[indice_mmse].copy()
    resultado[indices_deterioro] = riesgo_monotono[indice_mmse] * proporcion
    if len(indices_deterioro) > 1:
        resultado[indices_deterioro[-1]] = (
            riesgo_monotono[indice_mmse] - resultado[indices_deterioro[:-1]].sum()
        )
    indices_normalidad = [indice for indice, clase in enumerate(clases) if clase == "CN"]
    if indices_normalidad:
        resultado[indices_normalidad[0]] = 1.0 - riesgo_monotono[indice_mmse]
    return resultado


def evaluar_modelo(modelo, X_test: pd.DataFrame, y_test: pd.Series, codificador: LabelEncoder, carpeta_salida: str = CARPETA_RESULTADOS_DEFECTO) -> None:
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)

    print("--- Informe de clasificación ---")
    print(classification_report(y_test, y_pred, target_names=codificador.classes_))

    matriz = confusion_matrix(y_test, y_pred)
    print("--- Matriz de confusión ---")
    print(pd.DataFrame(
        matriz,
        index=[f"real_{c}" for c in codificador.classes_],
        columns=[f"pred_{c}" for c in codificador.classes_],
    ))

    auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
    print(f"\nAUC (macro, one-vs-rest): {auc:.3f}")

    Path(carpeta_salida).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(matriz, cmap="Blues")
    clases = list(codificador.classes_)
    ax.set_xticks(range(len(clases))); ax.set_xticklabels(clases)
    ax.set_yticks(range(len(clases))); ax.set_yticklabels(clases)
    ax.set_xlabel("Predicción del modelo"); ax.set_ylabel("Diagnóstico real")
    ax.set_title("Matriz de confusión — modelo final (XGBoost calibrado)")
    for i in range(len(clases)):
        for j in range(len(clases)):
            color = "white" if matriz[i, j] > matriz.max() / 2 else "black"
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center", color=color, fontsize=13)
    fig.colorbar(im, ax=ax, label="Nº de pacientes")
    fig.tight_layout()
    ruta = Path(carpeta_salida) / "matriz_confusion.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"Matriz de confusión guardada en {ruta}")


def explicar_con_shap(modelo, X_test: pd.DataFrame, carpeta_salida: str = CARPETA_RESULTADOS_DEFECTO) -> None:
    """Genera y guarda el gráfico resumen de SHAP: qué variables pesan
    más, en promedio, en las predicciones del modelo (secciones 4.4 y 5.3).
    """
    Path(carpeta_salida).mkdir(parents=True, exist_ok=True)
    explicador = shap.TreeExplainer(getattr(modelo, "modelo_base", modelo))
    valores_shap = explicador.shap_values(X_test)

    plt.figure()
    shap.summary_plot(valores_shap, X_test, show=False, plot_type="bar")
    plt.tight_layout()
    ruta = Path(carpeta_salida) / "shap_importancia.png"
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"Gráfico SHAP guardado en {ruta}")


if __name__ == "__main__":
    from datos import generar_datos_sinteticos
    df = generar_datos_sinteticos()
    X, y, codificador = preparar_features(df)
    modelo, X_train, X_test, y_train, y_test = entrenar_modelo(X, y)
    evaluar_modelo(modelo, X_test, y_test, codificador)
    explicar_con_shap(modelo, X_test)