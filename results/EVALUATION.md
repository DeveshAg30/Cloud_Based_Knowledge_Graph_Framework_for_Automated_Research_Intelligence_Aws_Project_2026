# Evaluation and monitoring (Phase 7)

This document collects every measured result for the report. All numbers come from the deployed system
(AWS us-east-1, Neo4j AuraDB Free, the Groq and Gemini free tiers) and were reproduced by the scripts named in each section.

## 1. NLP: SciBERT entity and relation extraction (Objective 1)
Source: [SCIBERT_RESULTS.md](SCIBERT_RESULTS.md) (`src/ai_model/train/train.py`, `train_re.py`, `calibrate_re.py`)

| SciERC test set | Precision | Recall | F1 |
|---|---|---|---|
| NER, all entity types | 0.593 | 0.645 | **0.618** |
| NER, Method | 0.699 | 0.714 | **0.707** |
| Relation extraction (7 types, gold entities, calibrated on dev) | 0.358 | 0.516 | **0.423** |

Both models are served from one SageMaker serverless endpoint. Model relations are mapped to typed graph edges
(APPLIED_TO, EVALUATED_ON, MEASURED_BY, COMPARED_WITH, IS_A, PART_OF). Pairs that don't fit the ontology are discarded.
The extractor also records **outcomes**: the value reported next to a metric (e.g. `Precision = 90%`) is stored on the
`REPORTS_METRIC` edge.

Effect on the graph (same corpus): papers with extracted entities went from 189/304 to **305/305**, paper→entity links
from 377 to **6,063**, and method→task/dataset/metric relations from ≈0 to **2,630**. Those were co-occurrence relations;
the model-typed relations are counted in section 2.

## 2. Knowledge graph size

| Nodes | Relationships | Papers | Authors | Institutions | Methods | Tasks | Datasets | Metrics |
|---|---|---|---|---|---|---|---|---|
| 29,320 | 42,015 | 20,746 | 1,674 | 553 | 2,416 | 1,727 | 779 | 262 |

Typed relations from the RE model: 865 in total. APPLIED_TO 399, IS_A 191,
EVALUATED_ON 122, MEASURED_BY 83, COMPARED_WITH 51, PART_OF 19.
Each pair has one direction: there are 0 contradictory two-way IS_A pairs. Reported outcome values stored on edges: 64.
Most Paper nodes are cited references; 306 documents were fully ingested.

## 3. GraphRAG retrieval quality (Objective 3)
Source: [GRAPHRAG_RETRIEVAL_EVAL.md](GRAPHRAG_RETRIEVAL_EVAL.md) (`scripts/evaluate_graphrag.py --retrieval-only`)

51 questions were generated from the graph with verifiable ground truth.

| Metric | Result |
|---|---|
| Context recall (ground-truth items present in the evidence given to the LLM) | **98.3%** |
| Questions with *all* ground-truth items retrieved | **97.8%** |
| Recall: authors of a paper / papers using a method / methods applied to a task | 100.0% / 100.0% / 95.0% |
| Retrieval latency (entity linking plus multi-hop expansion, 27 parallel graph queries) | **0.535 s** average |

Questions about non-existent entities still retrieve loosely related facts (only 16.7% return nothing),
because generic words such as "benchmark" match real nodes. Declining those questions is left to the grounded LLM prompt;
section 6 shows that it does.

## 4. Pipeline performance (Objective 2)
Source: `results/system_metrics.json` (`scripts/collect_metrics.py`, CloudWatch + S3, last 7 days)

| Metric | Result |
|---|---|
| Processing time per document (NER + RE on SageMaker, graph write) | 10.94 s median, 19.85 s p90 |
| Processing time per full-text PDF | 162.67 s mean (OpenAlex lookup, SciBERT NER + RE on every chunk) |
| Final rebuild throughput | 306 documents in 13.5 minutes with 12 Lambdas in parallel: **0 failures, 0 dead-letter messages** |
| SciBERT endpoint | 2,225 invocations, 3528.4 ms mean model latency (NER + RE) |
| API Gateway | 240 requests, **0 server errors (5xx)** |
| Lambda errors in the window | 4. All from one rebuild run *before* the throttling fix (section 5); the documents were recovered by Lambda's automatic retry |

## 5. Reliability engineering done in this phase
- **Parallel graph retrieval:** 44 sequential queries became 27 concurrent ones; retrieval went from about 6 s to about 0.6–1.0 s.
- **Time-boxed LLM calls:** each question has a 22 s budget, so a slow model can't exceed API Gateway's 30 s limit.
- **Model failover:** Groq gpt-oss-120b (about 1 s) → Gemini Flash-Lite → Gemini Flash → **cited graph facts**. The user always gets a grounded answer, never an error.
- **Faster cold starts:** the read-only API no longer re-creates database indexes when a new Lambda instance starts.
- **Live upload tracking:** S3 → Lambda → SageMaker → graph, visible on the dashboard.
- **Throttling-safe ingestion:** endpoint concurrency raised to 10 (the account maximum), plus exponential backoff with
  jitter on `ThrottlingException`. 12 parallel ingest Lambdas ran with 0 failures in the final rebuild.
- **Relation clean-up:** one direction per entity pair (the more confident one) and a canonical direction for symmetric
  relations; leading and trailing function words are removed from entity names.

## 6. GraphRAG vs. LLM only: the hallucination test (Objective 3)
Source: [GRAPHRAG_EVAL.md](GRAPHRAG_EVAL.md) (`scripts/evaluate_graphrag.py`)

28 questions with verifiable ground truth (24 answerable plus 4 about made-up entities). Each was answered by the **same model**
(Groq `openai/gpt-oss-120b`) with and without the knowledge graph, on the final graph (NER + RE).

| Metric | GraphRAG | LLM only |
|---|---|---|
| Answer recall | **100.0%** | 2.1% |
| Fully correct answers | **100.0%** | 0.0% |
| **Hallucination rate** (confident answer that is mostly wrong) | **0.0%** | **62.5%** |
| Said "I don't know" on answerable questions | 0.0% | 37.5% |
| Correctly declined questions about non-existent entities | 100.0% | 100.0% |
| Answers with source citations | **100.0%** | 0% |
| Citation precision | **98.3%** | n/a |
| Average answer latency | 1.69 s | 1.498 s |

**Finding:** without the graph, the model answered 62.5% of questions confidently and wrongly. For example, it invented the
authors, venue and year of a survey paper. With GraphRAG, every answerable question was answered correctly, with no
hallucination, and every answer cited its source papers. Both systems declined to describe made-up entities, so the gain
comes from grounding answers in retrievable facts, not just from the model being cautious.

*Scoring note:* names are matched ignoring hyphen and spacing variants (e.g. "Kolmogorov Arnold" = "Kolmogorov-Arnold").
An earlier version of the scorer counted one correct answer as wrong because of this.

## 7. Monitoring in the AWS console
- **CloudWatch → Dashboards → `research-kg-dev`:** ingestion volume, document processing time, query latency, questions
  answered, SciBERT invocations, latency and cold starts, API Gateway requests, errors and latency, Lambda errors.
- **CloudWatch → Logs Insights → Saved queries:** `research-kg-dev/recent-questions`, `/slow-questions`,
  `/ingested-papers`, `/errors`.
- **Alarms:** ingest Lambda errors and API errors, emailed through SNS.
- **X-Ray → Traces:** per-request timing across Lambda, S3 and SageMaker.

## 8. Cost
The whole evaluation ran on the AWS Free plan: Lambda, API Gateway, S3, Cognito and CloudWatch stayed inside the free tier.
SageMaker serverless inference used a few cents of credits for all rebuilds. AuraDB Free and Gemini cost $0. Card charges: $0.

## 9. End-to-end integration test (shared responsibility: integration testing)
Source: `scripts/smoke_test.py`. It runs against the deployed stack and uses a temporary Cognito user that it deletes afterwards.

| Check | Result |
|---|---|
| Cognito sign-in; unauthenticated API calls rejected (401) | pass |
| Public stats, authenticated stats | pass |
| GraphRAG answer with citations; made-up entity answered without invented facts | pass |
| Explore, Connect (multi-hop paths), Insights (Research 360) | pass |
| Upload URL → S3 PUT → Lambda → SageMaker NER + RE → graph, tracked to completion | pass (about 13 s) |
| **Total** | **13/13 checks pass** |

## 10. Features added after reviewing the proposal
- **Relation extraction model** (proposal: "fine-tune NER and RE models"): trained, calibrated, deployed, and mapped to typed edges.
- **Outcomes** (abstract: "methodologies and outcomes"): reported metric values are stored on `REPORTS_METRIC` edges.
- **Connect** (Objective 5: "visualize multi-hop research connections"): the shortest paths between any two entities, drawn on the dashboard.
- **Research 360 insights** (novelty: "comprehensive insights / 360 view"; problem statement: "cross-disciplinary
  collaborations"): entity profiles with yearly trends, co-authors, collaborating institutions, related methods, tasks and datasets.
- **Retraction filtering** (dataset section: "audit and filter … retraction handling"): OpenAlex fetches exclude retracted works.
