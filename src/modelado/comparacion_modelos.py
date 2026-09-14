"""
Comparación de modelos y evaluación del efecto de la calibración.

Punto del índice que cubre: 4.3.2 (comparación de modelos) -- justifica
por qué se eligió XGBoost frente a otras 4 alternativas, y documenta qué
cambia (y qué no) al añadir calibración de probabilidades (6.1/6.2).

Uso:
    python comparacion_modelos.py   # para datos sintéticos. Pruebas iniciales
    python comparacion_modelos.py ruta/ADNIMERGE.csv ruta/plasma.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin ventana -- solo guardamos a fichero, ver eda.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from clasificacion import entrenar_modelo, preparar_features

# Siempre <raiz_del_proyecto>/resultados, sin importar desde qué carpeta se
# lance el script.
CARPETA_RESULTADOS_DEFECTO = str(Path(__file__).resolve().parent.parent.parent / "resultados")
COLOR_MODELO = {
    "XGBoost": "#1565C0", "LightGBM": "#F9A825", "CatBoost": "#2E7D32",
    "Random Forest": "#6A1B9A", "Regresión Logística": "#757575",
}


def comparar_modelos(X: pd.DataFrame, y: pd.Series, carpeta_salida: str = CARPETA_RESULTADOS_DEFECTO) -> pd.DataFrame:
    """Entrena 5 modelos distintos (sin calibrar, mismos datos y misma
    partición) y compara accuracy y AUC. XGBoost usa manejo nativo de NaN;
    Random Forest y Regresión Logística, al no soportarlo, usan imputación
    por mediana --> una desventaja real de esos dos frente a XGBoost/LightGBM/
    CatBoost con este dataset, no una elección arbitraria del experimento.

    Guarda un gráfico de barras (exactitud y AUC) en carpeta_salida.
    """
    import lightgbm as lgb
    import catboost as cb
    import xgboost as xgb

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    resultados = []

    def evaluar(nombre, modelo, X_tr, X_te):
        modelo.fit(X_tr, y_train)
        pred = modelo.predict(X_te)
        proba = modelo.predict_proba(X_te)
        resultados.append({
            "modelo": nombre,
            "exactitud": round(accuracy_score(y_test, pred), 4),
            "auc_macro": round(roc_auc_score(y_test, proba, multi_class="ovr"), 4),
        })

    evaluar("XGBoost", xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric="mlogloss", random_state=42), X_train, X_test)
    evaluar("LightGBM", lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1), X_train, X_test)
    evaluar("CatBoost", cb.CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05, random_state=42, verbose=False), X_train, X_test)

    imputer = SimpleImputer(strategy="median")
    X_train_imp, X_test_imp = imputer.fit_transform(X_train), imputer.transform(X_test)
    evaluar("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42), X_train_imp, X_test_imp)

    scaler = StandardScaler()
    X_train_sc, X_test_sc = scaler.fit_transform(X_train_imp), scaler.transform(X_test_imp)
    evaluar("Regresión Logística", LogisticRegression(max_iter=1000, random_state=42), X_train_sc, X_test_sc)

    tabla = pd.DataFrame(resultados).sort_values("auc_macro", ascending=False).reset_index(drop=True)
    print("--- Comparación de modelos (sin calibrar, partición única) ---")
    print(tabla.to_string(index=False))

    Path(carpeta_salida).mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    colores = [COLOR_MODELO.get(m, "#333333") for m in tabla["modelo"]]
    ax1.barh(tabla["modelo"], tabla["exactitud"] * 100, color=colores)
    ax1.set_xlabel("Exactitud (%)")
    ax1.invert_yaxis()
    ax2.barh(tabla["modelo"], tabla["auc_macro"], color=colores)
    ax2.set_xlabel("AUC (macro, one-vs-rest)")
    ax2.set_xlim(0.7, 0.9)
    ax2.invert_yaxis()
    fig.suptitle("Comparación de modelos -- partición única 80/20")
    fig.tight_layout()
    ruta = Path(carpeta_salida) / "comparacion_modelos.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"Gráfico guardado en {ruta}")
    return tabla


def comparar_modelos_cv(X: pd.DataFrame, y: pd.Series, n_pliegues: int = 5, carpeta_salida: str = CARPETA_RESULTADOS_DEFECTO) -> pd.DataFrame:
    """Repite la comparación entre los 3 modelos de gradient boosting con
    validación cruzada estratificada -- una única partición no basta para
    saber si una diferencia de 1-2 puntos es real o es ruido del reparto
    concreto que tocó. Guarda un gráfico de líneas con la exactitud en
    cada pliegue, para ver si el orden entre modelos es estable o no.
    """
    import lightgbm as lgb
    import catboost as cb
    import xgboost as xgb

    cv = StratifiedKFold(n_splits=n_pliegues, shuffle=True, random_state=42)
    modelos = {
        "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric="mlogloss", random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1),
        "CatBoost": cb.CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05, random_state=42, verbose=False),
    }

    print(f"\n--- Comparación con validación cruzada ({n_pliegues} pliegues) ---")
    resultados_por_pliegue = {}
    filas_resumen = []
    for nombre, modelo in modelos.items():
        scores = cross_val_score(modelo, X, y, cv=cv, scoring="accuracy")
        resultados_por_pliegue[nombre] = scores
        filas_resumen.append({"modelo": nombre, "exactitud_media": round(scores.mean(), 4), "desviacion_tipica": round(scores.std(), 4)})
        print(f"{nombre:12s}: {[round(s, 3) for s in scores]}  ->  media={scores.mean():.4f}  desv.típica={scores.std():.4f}")

    tabla_resumen = pd.DataFrame(filas_resumen).sort_values("exactitud_media", ascending=False).reset_index(drop=True)

    Path(carpeta_salida).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for nombre, scores in resultados_por_pliegue.items():
        ax.plot(range(1, n_pliegues + 1), scores * 100, marker="o", label=nombre, color=COLOR_MODELO.get(nombre))
    ax.set_xlabel("Pliegue")
    ax.set_ylabel("Exactitud (%)")
    ax.set_xticks(range(1, n_pliegues + 1))
    ax.set_title(f"Exactitud por pliegue -- validación cruzada ({n_pliegues} pliegues)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ruta = Path(carpeta_salida) / "comparacion_modelos_cv.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"Gráfico guardado en {ruta}")
    return tabla_resumen


def evaluar_efecto_calibracion(X: pd.DataFrame, y: pd.Series, carpeta_salida: str = CARPETA_RESULTADOS_DEFECTO) -> pd.DataFrame:
    """Compara XGBoost sin calibrar vs. calibrado (CalibratedClassifierCV,
    sigmoid, cv=5 -- la configuración real usada en el dashboard). El
    objetivo de calibrar es corregir la sobreconfianza del modelo con
    datos escasos (6.1), no mejorar accuracy/AUC -- esta función comprueba
    precisamente que esas dos métricas no empeoran al calibrar, y guarda
    un gráfico con las tres magnitudes comparadas.
    """
    import xgboost as xgb

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    modelo_base = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric="mlogloss", random_state=42)
    modelo_base.fit(X_train, y_train)
    pred_base = modelo_base.predict(X_test)
    proba_base = modelo_base.predict_proba(X_test)

    modelo_calibrado = CalibratedClassifierCV(modelo_base, method="sigmoid", cv=5)
    modelo_calibrado.fit(X_train, y_train)
    pred_cal = modelo_calibrado.predict(X_test)
    proba_cal = modelo_calibrado.predict_proba(X_test)

    confianza_base = float(proba_base.max(axis=1).mean())
    confianza_cal = float(proba_cal.max(axis=1).mean())

    tabla = pd.DataFrame([
        {"version": "Sin calibrar", "exactitud": round(accuracy_score(y_test, pred_base), 4), "auc_macro": round(roc_auc_score(y_test, proba_base, multi_class="ovr"), 4), "confianza_media": round(confianza_base, 4)},
        {"version": "Calibrado (sigmoid, cv=5)", "exactitud": round(accuracy_score(y_test, pred_cal), 4), "auc_macro": round(roc_auc_score(y_test, proba_cal, multi_class="ovr"), 4), "confianza_media": round(confianza_cal, 4)},
    ])
    print("\n--- Efecto de la calibración sobre XGBoost ---")
    print(tabla.to_string(index=False))

    Path(carpeta_salida).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    etiquetas = ["Exactitud", "AUC", "Confianza media\nen la clase predicha"]
    valores_sin = [tabla.iloc[0]["exactitud"], tabla.iloc[0]["auc_macro"], tabla.iloc[0]["confianza_media"]]
    valores_con = [tabla.iloc[1]["exactitud"], tabla.iloc[1]["auc_macro"], tabla.iloc[1]["confianza_media"]]
    x = np.arange(len(etiquetas))
    ancho = 0.35
    ax.bar(x - ancho / 2, valores_sin, ancho, label="Sin calibrar", color="#90A4AE")
    ax.bar(x + ancho / 2, valores_con, ancho, label="Calibrado", color="#1565C0")
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas)
    ax.set_ylim(0, 1)
    ax.set_title("Efecto de la calibración sobre XGBoost")
    ax.legend()
    for i, (v1, v2) in enumerate(zip(valores_sin, valores_con)):
        ax.text(i - ancho / 2, v1 + 0.02, f"{v1:.2f}", ha="center", fontsize=9)
        ax.text(i + ancho / 2, v2 + 0.02, f"{v2:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    ruta = Path(carpeta_salida) / "efecto_calibracion.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"Gráfico guardado en {ruta}")
    return tabla


if __name__ == "__main__":
    if len(sys.argv) > 2:
        from datos import cargar_datos_reales
        df = cargar_datos_reales(sys.argv[1], fuente="adni", ruta_plasma=sys.argv[2])
        df = df[df["diagnostico"] != "Desconocido"].copy()
        print(f"Datos reales cargados: {len(df)} pacientes")
    else:
        from datos import generar_datos_sinteticos
        df = generar_datos_sinteticos()
        print("Sin CSV indicado: usando datos SINTÉTICOS.")

    X, y, codificador = preparar_features(df)
    comparar_modelos(X, y)
    comparar_modelos_cv(X, y)
    evaluar_efecto_calibracion(X, y)
