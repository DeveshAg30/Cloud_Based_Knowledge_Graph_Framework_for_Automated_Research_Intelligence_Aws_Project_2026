# Results

- [EVALUATION.md](EVALUATION.md): the full evaluation (NLP, graph, retrieval, GraphRAG vs LLM-only, performance, monitoring, cost).
- [SCIBERT_RESULTS.md](SCIBERT_RESULTS.md): NER training setup, test P/R/F1 per entity type, before/after effect on the graph.
- [GRAPHRAG_EVAL.md](GRAPHRAG_EVAL.md): hallucination benchmark (GraphRAG vs the same LLM without the graph).
- [GRAPHRAG_RETRIEVAL_EVAL.md](GRAPHRAG_RETRIEVAL_EVAL.md): retrieval-only context recall.
- `*.json`: raw outputs of `scripts/evaluate_graphrag.py`, `scripts/collect_metrics.py` and `src/ai_model/train/train.py`.
