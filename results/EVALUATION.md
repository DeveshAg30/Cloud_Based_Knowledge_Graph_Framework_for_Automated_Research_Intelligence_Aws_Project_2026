# Evaluation and monitoring (Phase 7)

This document collects every measured result for the report. All numbers come from the deployed system
(AWS us-east-1, Neo4j AuraDB Free, the Groq and Gemini free tiers) and were reproduced by the scripts named in each section.

## 1. NLP: SciBERT entity recognition (Objective 1)
Source: [SCIBERT_RESULTS.md](SCIBERT_RESULTS.md) (`src/ai_model/train/train.py`)

| SciERC test set | Precision | Recall | F1 |
|---|---|---|---|
| All entity types | 0.593 | 0.645 | **0.618** |
| Method | 0.699 | 0.714 | **0.707** |

Effect on the graph (same corpus): papers with extracted entities went from 189/304 to **305/305**, paper→entity links
from 377 to **6,063**, and method→task/dataset/metric relations from ≈0 to **2,630**.

## 2. Knowledge graph size

| Nodes | Relationships written | Methods | Tasks | Datasets | Metrics |
|---|---|---|---|---|---|
| 28,973 | 43,967 (incl. updates) | 2,339 | 1,653 | 728 | 226 |

## 3. GraphRAG retrieval quality (Objective 3)
Source: [GRAPHRAG_RETRIEVAL_EVAL.md](GRAPHRAG_RETRIEVAL_EVAL.md) (`scripts/evaluate_graphrag.py --retrieval-only`)

51 questions were generated from the graph with verifiable ground truth.

| Metric | Result |
|---|---|
| Context recall (ground-truth items present in the evidence given to the LLM) | **96.3%** |
| Questions with *all* ground-truth items retrieved | **95.6%** |
| Recall: authors of a paper / methods applied to a task / papers using a method | 100.0% / 100.0% / 88.9% |
| Retrieval latency (entity linking plus multi-hop expansion, 27 parallel graph queries) | **0.57 s** average |

Questions about non-existent entities still retrieve loosely related facts (only 16.7% return nothing),
because generic words such as "benchmark" match real nodes. Declining those questions is left to the grounded LLM prompt
("answer only from the facts"). The GraphRAG-vs-LLM comparison (section 6) measures this.

## 4. Pipeline performance (Objective 2)
Source: `results/system_metrics.json` (`scripts/collect_metrics.py`, CloudWatch + S3)

| Metric | Result |
|---|---|
| Documents ingested (incl. rebuilds), failures | 1,830 ingested, **0 failures** |
| Processing time per document (abstract-length) | 6.07 s median, 8.39 s p90 |
| Processing time per full-text PDF | 88.01 s mean (includes OpenAlex lookup and SciBERT on every chunk) |
| Throughput | 305 documents in about 6 minutes, so about **3,000 documents/hour**, with Lambda processing files in parallel |
| SciBERT endpoint | 1,422 invocations, 1836.9 ms mean model latency, 20 cold starts, 1 errors |
| API Gateway | 153 requests, **0 server errors (5xx)** |
| Lambda errors | 1 |

## 5. Reliability engineering done in this phase
- **Parallel graph retrieval:** 44 sequential queries became 27 concurrent ones; retrieval went from about 6 s to about 0.6–1.0 s.
- **Time-boxed LLM calls:** each question has a 22 s budget, so a slow model can't exceed API Gateway's 30 s limit.
- **Model failover:** Groq gpt-oss-120b (about 1 s) → Gemini Flash-Lite → Gemini Flash → **cited graph facts**. The user always gets a grounded answer, never an error.
- **Faster cold starts:** the read-only API no longer re-creates database indexes when a new Lambda instance starts.
- **Live upload tracking:** S3 → Lambda → SageMaker → graph, visible on the dashboard.

## 6. GraphRAG vs. LLM only: the hallucination test (Objective 3)
Source: [GRAPHRAG_EVAL.md](GRAPHRAG_EVAL.md) (`scripts/evaluate_graphrag.py`)

28 questions with verifiable ground truth (24 answerable plus 4 about made-up entities). Each was answered by the **same model**
(Groq `openai/gpt-oss-120b`) with and without the knowledge graph.

| Metric | GraphRAG | LLM only |
|---|---|---|
| Answer recall | **95.8%** | 4.3% |
| Fully correct answers | **87.5%** | 0.0% |
| **Hallucination rate** (confident answer that is mostly wrong) | **0.0%** | **62.5%** |
| Said "I don't know" on answerable questions | 0.0% | 37.5% |
| Correctly declined questions about non-existent entities | 100.0% | 100.0% |
| Answers with source citations | **100.0%** | 0% |
| Citation precision | **96.5%** | n/a |
| Average answer latency | 1.857 s | 1.691 s |

**Finding:** without the graph, the model answered 62.5% of questions confidently and wrongly. For example, it invented the
authors, venue and year of a survey paper. With GraphRAG, 0% of answers were hallucinated, 100% cited their source papers,
and 96.5% of citations were correct. Both systems correctly declined to describe made-up entities, so the gain comes
from grounding answers in real, retrievable facts, not just from the model being cautious.

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
