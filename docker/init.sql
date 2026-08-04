-- Se ejecuta automáticamente la primera vez que arranca el contenedor.
-- Activa pgvector; el diseño de tablas (fichas de paciente + embeddings)


CREATE EXTENSION IF NOT EXISTS vector;
