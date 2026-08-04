# TFM — Estratificación y predicción de riesgo (Alzheimer/MCI)

Proyecto de Trabajo Fin de Master: ingesta de informes clínicos mediante LLM
local (RAG), modelado predictivo sobre OASIS-3 (clustering + XGBoost), y
dashboard interactivo con explicabilidad (SHAP).

## Estructura

- `data/` — datos (no versionados, ver .gitignore)
- `src/extraccion/` — pipeline de extracción LLM (memoria, sección 3.2)
- `src/modelado/` — clustering y clasificación (memoria, sección 4)
- `src/dashboard/` — app Streamlit (memoria, sección 5)
- `notebooks/` — EDA y experimentos
- `docker/` — PostgreSQL + pgvector
- `tests/` — gold set de evaluación (memoria, sección 6.1)
- `docs/` — memoria y material de apoyo
