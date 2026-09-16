"""
Orquesta el pipeline completo de la fase de Data Science (sección 4):
carga -> EDA -> clustering -> clasificación (calibrada) -> evaluación -> SHAP
-> comparación con otros modelos.

Uso:
    python main.py ruta/ADNIMERGE.csv ruta/plasma.csv    # datos reales de ADNI
"""

import sys
from pathlib import Path


from clasificacion import entrenar_modelo, evaluar_modelo, explicar_con_shap, preparar_features
from clustering import ejecutar_clustering, graficar_clusters
from datos import cargar_datos_reales, generar_datos_sinteticos
from eda import completitud_variables, graficar_completitud, graficar_distribuciones, resumen_estadistico
from comparacion_modelos import comparar_modelos, comparar_modelos_cv, evaluar_efecto_calibracion

# <raiz_del_proyecto>/data/raw para evitar rutas locales 
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent.parent
RUTA_ADNI_DEFECTO = str(RAIZ_PROYECTO / "data" / "raw" / "ADNIMERGE_11Aug2026.csv")
RUTA_PLASMA_DEFECTO = str(RAIZ_PROYECTO / "data" / "raw" / "UPENN_PLASMA_FUJIREBIO_QUANTERIX_11Aug2026.csv") 


def main():
    if sys.argv[1:] == ["--sintetico"]:
        print("Usando datos SINTÉTICOS (--sintetico).")
        df = generar_datos_sinteticos()
    elif len(sys.argv) == 1:
        print("Sin argumentos: usando las rutas por defecto.")
        df = cargar_datos_reales(RUTA_ADNI_DEFECTO, fuente="adni", ruta_plasma=RUTA_PLASMA_DEFECTO)
        df = df[df["diagnostico"] != "Desconocido"].copy()
    elif len(sys.argv) == 3:
        ruta_adni, ruta_plasma = sys.argv[1], sys.argv[2]
        print(f"Cargando datos reales desde {ruta_adni} (+ plasma: {ruta_plasma})")
        df = cargar_datos_reales(ruta_adni, fuente="adni", ruta_plasma=ruta_plasma)
        df = df[df["diagnostico"] != "Desconocido"].copy()
    else:
        sys.exit("Uso: py main.py  |  py main.py ruta_adni ruta_plasma  |  py main.py --sintetico")

    print(f"\n{len(df)} pacientes con diagnóstico válido.")
   

    print("\n========== EDA ==========")
    resumen_estadistico(df)
    completitud_variables(df)
    graficar_distribuciones(df)
    graficar_completitud(df)

    print("\n========== Clustering ==========")
    df_clusters = ejecutar_clustering(df)
    graficar_clusters(df_clusters)

    print("\n========== Clasificación (modelo final: XGBoost calibrado) ==========")
    X, y, codificador = preparar_features(df)
    modelo, X_train, X_test, y_train, y_test = entrenar_modelo(X, y)

    print("\n========== Comparación con otros modelos ==========")
    comparar_modelos(X, y)
    comparar_modelos_cv(X, y)
    evaluar_efecto_calibracion(X, y)


    print("\n========== Evaluación del modelo final (XGBoost) ==========")
    evaluar_modelo(modelo, X_test, y_test, codificador)
    explicar_con_shap(modelo, X_test)

    print("\nPipeline completo. Resultados guardados en la carpeta 'resultados/'.")


if __name__ == "__main__":
    main()
