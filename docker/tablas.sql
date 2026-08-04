-- Tabla principal: ficha completa (JSON) + resumen embebido para búsqueda
-- por similitud ("pacientes similares"). Ejecutar UNA vez contra el
-- contenedor ya en marcha (no va en init.sql, que solo corre al crear
-- el contenedor por primera vez).

CREATE TABLE IF NOT EXISTS pacientes (
    id SERIAL PRIMARY KEY,
    paciente_id TEXT UNIQUE NOT NULL,
    ficha JSONB NOT NULL,
    resumen_texto TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    creado_en TIMESTAMP DEFAULT now()
);

-- Índice HNSW: búsqueda por similitud rápida según crece la tabla.
-- "vector_cosine_ops" porque comparamos los vectores por distancia coseno.
CREATE INDEX IF NOT EXISTS pacientes_embedding_idx
ON pacientes USING hnsw (embedding vector_cosine_ops);
