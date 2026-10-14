# GraphRAG evaluation

28 questions were generated automatically from the knowledge graph, so every answer has a verifiable ground truth. Each was answered twice with the same LLM (Groq openai/gpt-oss-120b): **GraphRAG** (graph retrieval + grounded generation, the system in this project) and **LLM only** (the same model with no graph).

| Metric | GraphRAG | LLM only |
|---|---|---|
| Answer recall (share of ground-truth items in the answer) | **96%** | 4% |
| Fully correct answers (all items found) | **88%** | 0% |
| **Hallucinated: confident answer that is mostly wrong** | **0%** | 62% |
| Said "I don't know" on answerable questions | 0% | 38% |
| Correct "not found" on questions about non-existent entities | **100%** | 100% |
| Answers with source citations | **100%** | 0% (no sources available) |
| Citation precision (cited paper really supports the answer) | **96%** | n/a |
| Average latency per question | 1.857 s | 1.691 s |

## Recall by question type

| Question type | GraphRAG | LLM only |
|---|---|---|
| Who are the authors of <paper>? | 100% | 5% |
| Which papers use <method>? | 96% | 3% |
| Which methods have been applied to <task>? | 92% | 5% |

## Examples

**Q: Who are the authors of the paper 'A Survey on Knowledge Graph Embedding: Approaches, Applications and Benchmarks'?**

*Ground truth:* Yuanfei Dai, Shiping Wang, Naixue N. Xiong, Wenzhong Guo

*GraphRAG:* The paper **“A Survey on Knowledge Graph Embedding: Approaches, Applications and Benchmarks”** was authored by:

- Yuanfei Dai[doc:W3021393704]  
- Shiping Wang[doc:W3021393704]  
- Naixue N. Xiong[doc:W3021393704]  
- Wenzhong Guo[doc:W3021393704]

*LLM only:* **Paper:**  
*“A Survey on Knowledge Graph Embedding: Approaches, Applications and Benchmarks”*  

**Authors:**  
1. **Xiao Wang**  
2. **Yao Ma**  
3. **Zhiyuan Liu**  
4. **Maosong Sun**  

The article was published in 2020 (IEEE Transactions on Knowledge and Data Engineering) and has become one of the most-cited overview papers on knowledge-graph embedding methods, covering translational, tensor-factorization, neural-network-based, and graph-neural-network approaches, as well as a systematic benchmark suite.

**Q: Which papers evaluate the QuantumFrobnicator embedding, and what results did they report?**

*Ground truth:* no such entity exists

*GraphRAG:* The provided facts do not contain any information about a “QuantumFrobnicator” embedding, nor do they mention any papers that evaluate it or report results. Consequently, the answer cannot be derived from the given data.

*LLM only:* I’m sorry, but I don’t have any information about a “QuantumFrobnicator” embedding or any papers that evaluate it. If you can provide more context or clarify the term, I’d be happy to try to help further.

## Method

- Ground truth comes from graph edges (AUTHORED, USES_METHOD, APPLIED_TO). Unanswerable questions name made-up entities.
- An item counts as found if the answer names it: author surname, the first five significant words of a paper title, a cited paper id, or a method name.
- Refusal = the answer says the information isn't available, instead of describing the made-up entity.
- Scores reflect retrieval and grounding. The LLM-only baseline can't see this corpus, so its low recall on corpus-specific questions is expected. The key differences are invented answers versus refusals, and the absence of citations.

Raw per-question results: `results/graphrag_eval.json`.
