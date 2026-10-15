# Architecture

Data flow: papers land in **S3 `raw/`**. That triggers the **ingest Lambda**, which parses them, sends text to the **SciBERT
endpoint on SageMaker**, canonicalises entities and MERGEs them into the **knowledge graph** (Neo4j AuraDB or Amazon Neptune).
Users sign in through **Cognito** on the **Amplify** dashboard. **API Gateway** passes requests to the **API Lambda**, which runs
GraphRAG: entity linking, multi-hop retrieval, ranking, then a grounded LLM answer with citations. **CloudWatch** and **X-Ray**
monitor both paths.

Graph ontology: `(Author)-[:AUTHORED]->(Paper)`, `(Author)-[:AFFILIATED_WITH]->(Institution)`, `(Paper)-[:PUBLISHED_IN]->(Venue)`,
`(Paper)-[:CITES]->(Paper)`, `(Paper)-[:HAS_CONCEPT]->(Concept)`,
`(Paper)-[:USES_METHOD|USES_DATASET|ADDRESSES_TASK|REPORTS_METRIC]->(entity)`,
`(Method)-[:APPLIED_TO]->(Task)`, `(Method|Task)-[:EVALUATED_ON]->(Dataset)`, `(Method|Task)-[:MEASURED_BY]->(Metric)`,
`(Method)-[:COMPARED_WITH]->(Method)`, `(Method|Task)-[:IS_A]->(Method|Task)`, `(Method)-[:PART_OF]->(Method)`.
The last group comes from the SciBERT relation-extraction model. `REPORTS_METRIC` edges carry the reported value (`value: "96.7%"`).

Diagrams (as deployed, used in the project report):

- [cloud_deployment_architecture.png](cloud_deployment_architecture.png): AWS services, external services, data flow, security and monitoring.
- [system_workflow.png](system_workflow.png): the ingestion path and the GraphRAG query path, step by step.

A text version is in the main [README](../README.md#architecture).
