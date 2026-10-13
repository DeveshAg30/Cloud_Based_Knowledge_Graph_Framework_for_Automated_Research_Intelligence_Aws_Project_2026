# Architecture

Data flow: papers land in **S3 `raw/`**. That triggers the **ingest Lambda**, which parses them, sends text to the **SciBERT
endpoint on SageMaker**, canonicalises entities and MERGEs them into the **knowledge graph** (Neo4j AuraDB or Amazon Neptune).
Users sign in through **Cognito** on the **Amplify** dashboard. **API Gateway** passes requests to the **API Lambda**, which runs
GraphRAG: entity linking, multi-hop retrieval, ranking, then a grounded LLM answer with citations. **CloudWatch** and **X-Ray**
monitor both paths.

Graph ontology: `(Author)-[:AUTHORED]->(Paper)`, `(Author)-[:AFFILIATED_WITH]->(Institution)`, `(Paper)-[:PUBLISHED_IN]->(Venue)`,
`(Paper)-[:CITES]->(Paper)`, `(Paper)-[:HAS_CONCEPT]->(Concept)`,
`(Paper)-[:USES_METHOD|USES_DATASET|ADDRESSES_TASK|REPORTS_METRIC]->(entity)`,
`(Method)-[:APPLIED_TO]->(Task)`, `(Method)-[:EVALUATED_ON]->(Dataset)`, `(Method)-[:MEASURED_BY]->(Metric)`.

The diagram source is in the main [README](../README.md#architecture).
