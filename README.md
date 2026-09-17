# NeuroInsight — Apoyo a la evaluación clínica del Alzheimer

Trabajo de Fin de Máster: un pipeline de extremo a extremo que va de un
informe clínico no estructurado (PDF) a una predicción interpretable de
categoría diagnóstica (CN / MCI / AD), pasando por extracción con un LLM
local, un modelo de clasificación calibrado y explicabilidad con SHAP.

No es una herramienta de diagnóstico ni de pronóstico de riesgo futuro:
clasifica el perfil clínico **actual** de un paciente en una de tres
categorías, como apoyo al criterio del profesional — nunca como sustituto.

## Qué hace

- **Ingesta (LLM en local, Ollama):** lee un informe en PDF y extrae los
  datos clínicos relevantes en un esquema validado (Pydantic), con
  reintentos acotados y una red de seguridad por regex para el campo MMSE.
- **Predicción (ML clásico):** XGBoost calibrado, entrenado y comparado
  empíricamente frente a otros cuatro algoritmos (CatBoost, LightGBM,
  Random Forest, Regresión Logística) sobre datos reales de ADNI.
- **Explicabilidad:** SHAP muestra qué variables pesaron en cada
  predicción concreta; el dashboard distingue siempre entre dato real y
  dato ausente.
- **Dashboard (Streamlit):** predicción individual (manual o por PDF),
  análisis e interpretabilidad, notas clínicas, generación de informe
  descargable, búsqueda de literatura en PubMed, historial de pacientes,
  resumen del dataset y recursos.

## Dataset

**ADNI** (Alzheimer's Disease Neuroimaging Initiative), fase ADNI-4:
2 414 pacientes en el conjunto final de modelado, con datos cognitivos
(MMSE), estructurales (volumen hipocampal por RM), genéticos (APOE4) y
bioquímicos (biomarcadores en LCR y en plasma). Detalle completo de la
selección de variables y la limpieza aplicada en la memoria, sección 4.

## Resultados (sobre 483 pacientes de test)

| Métrica | Valor |
|---|---|
| Exactitud | 66,9% (baseline de clase mayoritaria: 46,0%) |
| AUC macro | 0,852 |
| Recall AD | 62,2% |

El sistema no ha sido optimizado con una búsqueda exhaustiva de
hiperparámetros: la prioridad fue tener un pipeline completo y verificado
de extremo a extremo, no el mejor clasificador aislado posible (memoria,
secciones 4.5 y 6.2).

## Estructura

- `data/` — datos (no versionados, ver `.gitignore`)
- `src/extraccion/` — pipeline de extracción LLM + embeddings (memoria, sección 3)
- `src/modelado/` — carga de datos, EDA, clustering y clasificación (memoria, sección 4)
- `src/dashboard/` — app Streamlit NeuroInsight (memoria, sección 5)
- `notebooks/` — EDA y experimentos, incluido el cuaderno de Colab + ngrok para GPU remota
- `docker/` — PostgreSQL + pgvector (fichas de pacientes y búsqueda por similitud)
- `tests/` — gold set de evaluación de la extracción (memoria, sección 6.1)
- `docs/` — memoria y material de apoyo

## Puesta en marcha

```
py -m pip install -r requirements.txt
py main.py                # pipeline de datos y modelado (src/modelado)
streamlit run src/dashboard/app.py
```

Ollama debe estar en marcha en local (`ollama pull llama3.2:3b`), o bien
configurarse una URL remota (Colab + ngrok) desde la barra lateral del
dashboard — ver Anexo E de la memoria.

## Limitaciones conocidas

Sin corrección del desbalance de clases, sin ajuste de hiperparámetros,
validado sobre una cohorte mayoritariamente estadounidense, uso únicamente de ocho variables. 