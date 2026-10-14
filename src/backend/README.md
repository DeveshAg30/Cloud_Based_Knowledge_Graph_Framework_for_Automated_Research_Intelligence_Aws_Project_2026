# Backend (AWS Lambda, Python 3.12)

Two Lambda functions share this code (`CodeUri: src/backend/` in `template.yaml`):

- `handlers/ingest.py`: triggered by S3 `raw/` uploads. It parses (PDF / OpenAlex / S2ORC), extracts entities (SciBERT on SageMaker
  plus a term dictionary), writes the graph, and records upload status for the dashboard.
- `handlers/api.py`: API Gateway routes `POST /query` (GraphRAG), `GET /graph` (Explore), `GET /connect` (multi-hop paths
  between two entities), `GET /insights` (Research 360 profile), `GET /stats`, `GET /public/stats`, `POST /upload-url` and
  `GET /upload-status` (uploads are for the `members` group only).

`kg/` modules: `parsers` (input formats, PDF text repair), `extractors` (SciBERT / LLM / rules), `aliases` (entity canonicalisation),
`enrich` (OpenAlex metadata for PDFs), `schema` (ontology), `graph_client` (Neo4j / Neptune), `graph_writer` (idempotent MERGE),
`retrieval` (GraphRAG), `insights` (entity profiles and connection paths), `llm` (Groq / Gemini / Bedrock with failover),
`metrics` (CloudWatch EMF). Extraction also records **outcomes**: the value a paper reports next to a metric (e.g. "96.7%")
is stored on its `REPORTS_METRIC` edge.
