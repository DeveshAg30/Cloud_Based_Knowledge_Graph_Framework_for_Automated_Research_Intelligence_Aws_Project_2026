# Cloud-Based Knowledge Graph Framework for Automated Research Intelligence

**BITE412L Cloud Computing, VIT Vellore.** Dhruv Gupta (23BIT0350), Parth Sharma (23BIT0211), Devesh Agarwal (23BIT0201).

The system turns scientific papers into a knowledge graph of papers, authors, institutions, methods, datasets, tasks and metrics.
Researchers ask questions in plain English. The **GraphRAG** layer retrieves the relevant facts from the graph, and an LLM writes an
answer **only from those facts**, citing the source papers.

| Result | Value |
|---|---|
| SciBERT entity recognition (SciERC test F1) | **0.618** (Methods 0.707) |
| SciBERT relation extraction (SciERC test F1, calibrated) | **0.423** |
| End-to-end test of the deployed system | **13/13** checks (`scripts/smoke_test.py`) |
| GraphRAG answer recall vs. the same LLM without the graph | **100.0%** vs 2.1% |
| Hallucinated answers (confident and mostly wrong) | **0.0%** vs 62.5% without the graph |
| Answers citing their sources / citation precision | **100.0%** / **98.3%** |
| Graph retrieval latency | about 0.6 s |
| Ingestion (NER + RE per paper) | 306 papers in 13.5 min (about 1,350 papers/hour), 0 failures |

Full results: [results/EVALUATION.md](results/EVALUATION.md).

## Architecture

```
Papers (OpenAlex / S2ORC / PDF upload)
   │
   ▼
Amazon S3  raw/ ──event──▶ AWS Lambda (ingest) ──▶ Amazon SageMaker serverless endpoint (fine-tuned SciBERT NER)
                               │   PDF text repair, OpenAlex metadata enrichment, entity canonicalisation
                               ▼
                         Knowledge graph (openCypher): Neo4j AuraDB Free, or Amazon Neptune Analytics
                               ▲
Dashboard (AWS Amplify) ──Cognito JWT──▶ Amazon API Gateway ──▶ AWS Lambda (API) ──GraphRAG──▶ LLM (Groq / Gemini / Bedrock)

Amazon CloudWatch: dashboard, custom metrics, alarms, saved Logs Insights queries. AWS X-Ray: tracing.
```

**The AWS Free plan blocks Amazon Neptune and Amazon Bedrock**, so the deployed system uses Neo4j AuraDB Free (the same openCypher
queries) and the Groq / Gemini free tiers. The SciBERT model was trained on a laptop CPU, because the SageMaker training quota is 0
on this account; it is **served on Amazon SageMaker**. All of these are deploy parameters in [template.yaml](template.yaml):
`GraphBackend=neptune-analytics` and `LlmProvider=bedrock-nova` / `bedrock-claude` switch back to the original AWS design.

## Features

- **Automated ingestion:** OpenAlex, S2ORC and PDF uploads land in S3 and are processed automatically. PDFs are matched to
  OpenAlex for exact authors, venues and citations.
- **NLP on SageMaker:** SciBERT entity recognition (methods, tasks, datasets, metrics) **and** SciBERT relation extraction
  (applied-to, evaluated-on, measured-by, compared-with, is-a, part-of), plus the **outcomes** papers report (e.g. `F1 = 91.4%`).
- **Ask (GraphRAG):** plain-English questions answered only from graph facts, with clickable citations.
- **Explore:** an interactive graph around any paper, author, method or institution.
- **Connect:** the shortest multi-hop chains between any two entities (e.g. *TransE ↔ BERT*).
- **Research 360 insights:** a profile for any entity: papers per year, collaborators and collaborating institutions,
  related methods, tasks and datasets, top authors, reported results.
- **Public sign-up** with email verification; uploads are limited to the `members` group. Live upload tracking.

## How GraphRAG answers a question
1. **Entity linking:** match the question to nodes: an exact name, a paper title inside the question, or phrases of up to 4 words.
2. **Multi-hop expansion:** collect neighbours of each matched node, plus paths of up to 3 hops between pairs of matches (27 graph
   queries run in parallel). Paths reveal links between papers that never cite each other.
3. **Ranking:** score each fact by match quality, relation type and question wording (e.g. "who wrote…" boosts AUTHORED).
4. **Grounded generation:** the top 30 facts go to the LLM, which must answer only from them and cite `[doc:<paper id>]`.
   If no LLM answers in time, the user gets the cited facts instead. A request never fails because of the LLM.

## Repository layout

| Folder | Contents |
|---|---|
| [src/backend/](src/backend/) | Lambda code (Python): `kg/` (parsing, extraction, graph, GraphRAG, LLM) and `handlers/` (ingest + API) |
| [src/ai_model/](src/ai_model/) | SciBERT NER: `train/train.py` (fine-tuning on SciERC) and `inference/inference.py` (SageMaker handler) |
| [src/frontend/](src/frontend/) | Dashboard: landing page, sign-in/up, Ask, Explore, live upload tracking |
| [src/aws/](src/aws/) | IAM policy documents. The AWS SAM template is [template.yaml](template.yaml) at the repo root |
| [scripts/](scripts/) | Dataset fetchers, local ingest, SageMaker deploy, benchmarks, metrics collection |
| [dataset/](dataset/) | Sample papers in the canonical document format |
| [results/](results/) | Evaluation reports and raw metrics |
| [docs/](docs/) | Setup guides: [FREE_TIER_SETUP.md](docs/FREE_TIER_SETUP.md) (as deployed) and [AWS_IMPLEMENTATION.md](docs/AWS_IMPLEMENTATION.md) (full AWS) |
| [tests/](tests/) | 30 unit tests (`python -m pytest tests`) and `scripts/smoke_test.py`, an end-to-end test of the deployed system |

## Quick start

```powershell
pip install -r requirements-dev.txt
python -m pytest tests                       # 30 tests
sam build --use-container                    # needs Docker running
sam deploy --guided                          # see docs/FREE_TIER_SETUP.md for every parameter
python scripts/fetch_openalex.py --search "knowledge graph" --limit 300
aws s3 cp data/openalex "s3://<DataBucketName>/raw/openalex/" --recursive
```

## Work distribution

| Member | Area | Main files |
|---|---|---|
| Devesh Agarwal | Data acquisition, NLP extraction, SciBERT | `scripts/fetch_*.py`, `src/backend/kg/{parsers,extractors,aliases,enrich}.py`, `src/backend/handlers/ingest.py`, `src/ai_model/` |
| Dhruv Gupta | Knowledge graph, GraphRAG, evaluation | `src/backend/kg/{schema,graph_client,graph_writer,retrieval,llm}.py`, `scripts/evaluate_graphrag.py` |
| Parth Sharma | Dashboard, API, AWS infrastructure, monitoring | `src/frontend/`, `src/backend/handlers/api.py`, `template.yaml`, `scripts/collect_metrics.py` |
