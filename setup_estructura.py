"""
Genera la estructura de carpetas del proyecto TFM, con README, requirements.txt
y .gitignore ya listos.

Uso: colócalo dentro de la carpeta vacía del proyecto (ej. tfm-riesgo-alzheimer/)
y ejecútalo UNA vez desde la terminal de VS Code:

    python setup_estructura.py
    (en Mac/Linux puede ser: python3 setup_estructura.py)
"""

from pathlib import Path

CARPETAS = [
    "data/raw", "data/processed",
    "src/extraccion", "src/modelado", "src/dashboard",
    "notebooks", "docker", "tests", "docs",
]

README = """# TFM — Estratificación y predicción de riesgo (Alzheimer/MCI)

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
"""

REQUIREMENTS = """ollama
pydantic
pandas
scikit-learn
xgboost
shap
streamlit
psycopg2-binary
pgvector
python-dotenv
"""

GITIGNORE = """# Datos: NO se suben (términos de uso de OASIS-3 lo prohíben)
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Entornos y cachés de Python
venv/
__pycache__/
*.pyc

# Docker
docker/postgres_data/

# Credenciales / configuración local
.env

# Sistema
.DS_Store
"""


def crear_estructura() -> None:
    for carpeta in CARPETAS:
        ruta = Path(carpeta)
        ruta.mkdir(parents=True, exist_ok=True)
        (ruta / ".gitkeep").touch()

    Path("README.md").write_text(README, encoding="utf-8")
    Path("requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    Path(".gitignore").write_text(GITIGNORE, encoding="utf-8")

    print("Estructura creada:")
    for carpeta in CARPETAS:
        print(f"  {carpeta}/")
    print("Más README.md, requirements.txt, .gitignore")


if __name__ == "__main__":
    crear_estructura()
