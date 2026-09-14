"""
Orquesta el pipeline completo de la fase de Data Science (sección 4):
carga -> EDA -> clustering -> clasificación (calibrada) -> evaluación -> SHAP
-> comparación con otros modelos.

Uso:
    python main.py ruta/ADNIMERGE.csv ruta/plasma.csv    # datos reales de ADNI
"""

import sys

from clasificacion import entrenar_modelo, evaluar_modelo, explicar_con_shap, preparar_features
from clustering import ejecutar_clustering, graficar_clusters
from comparacion_modelos import comparar_modelos, evaluar_efecto_calibracion
from datos import cargar_datos_reales, generar_datos_sinteticos
from eda import completitud_variables, graficar_completitud, graficar_distribuciones, resumen_estadistico


def main():
    if len(sys.argv) > 2:
        ruta_adni, ruta_plasma = sys.argv[1], sys.argv[2]
        print(f"Cargando datos reales desde {ruta_adni} (+ plasma: {ruta_plasma})")
        df = cargar_datos_reales(ruta_adni, fuente="adni", ruta_plasma=ruta_plasma)
        df = df[df["diagnostico"] != "Desconocido"].copy()
    else:
        print("Sin CSV indicado: usando datos sintéticos para probar")
        df = generar_datos_sinteticos()

    print(f"\n{len(df)} pacientes con diagnóstico válido.")

    print("\n========== 4.2 EDA ==========")
    resumen_estadistico(df)
    completitud_variables(df)
    graficar_distribuciones(df)
    graficar_completitud(df)

    print("\n========== 4.3.1 Clustering ==========")
    df_clusters = ejecutar_clustering(df)
    graficar_clusters(df_clusters)

    print("\n========== 4.3.2 Clasificación (modelo final: XGBoost calibrado) ==========")
    X, y, codificador = preparar_features(df)
    modelo, X_train, X_test, y_train, y_test = entrenar_modelo(X, y)

    print("\n========== 4.3.2 Comparación con otros modelos ==========")
    comparar_modelos(X, y)
    evaluar_efecto_calibracion(X, y)

    print("\n========== 4.4 Evaluación del modelo final ==========")
    evaluar_modelo(modelo, X_test, y_test, codificador)
    explicar_con_shap(modelo, X_test)

    print("\nPipeline completo. Resultados guardados en la carpeta 'resultados/'.")


if __name__ == "__main__":
    main()
