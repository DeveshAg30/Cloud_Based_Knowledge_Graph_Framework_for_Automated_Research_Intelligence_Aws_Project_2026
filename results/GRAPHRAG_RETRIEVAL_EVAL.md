# GraphRAG retrieval evaluation (no LLM)

51 questions were generated from the knowledge graph. For each one, GraphRAG's retrieval step (entity linking, then 1-hop expansion plus up to 3-hop paths) produced the facts that would be given to the LLM. This measures whether the right evidence reaches the LLM, independent of which LLM writes the answer.

| Metric | Value |
|---|---|
| Context recall (share of ground-truth items present in the retrieved facts) | **96%** |
| Questions with all ground-truth items retrieved | **96%** |
| Source precision (retrieved paper ids that support the answer) | 51% |
| Questions about non-existent entities with no matching facts | **17%** |
| Average retrieval latency | 0.57 s |

| Question type | Context recall |
|---|---|
| Who are the authors of <paper>? | 100% |
| Which papers use <method>? | 89% |
| Which methods have been applied to <task>? | 100% |

Source precision is low by design: retrieval returns up to 60 facts around the matched entities, and the LLM then selects and cites the relevant ones. Context recall is the property that matters at this stage.

Raw results: `results/graphrag_retrieval_eval.json`.
