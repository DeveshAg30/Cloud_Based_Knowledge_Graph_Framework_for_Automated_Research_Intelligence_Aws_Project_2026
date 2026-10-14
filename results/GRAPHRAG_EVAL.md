# GraphRAG evaluation

28 questions were generated automatically from the knowledge graph, so every answer has a verifiable ground truth. Each was answered twice with the same LLM (Groq openai/gpt-oss-120b): **GraphRAG** (graph retrieval + grounded generation, the system in this project) and **LLM only** (the same model with no graph).

| Metric | GraphRAG | LLM only |
|---|---|---|
| Answer recall (share of ground-truth items in the answer) | **100%** | 2% |
| Fully correct answers (all items found) | **100%** | 0% |
| **Hallucinated: confident answer that is mostly wrong** | **0%** | 62% |
| Said "I don't know" on answerable questions | 0% | 38% |
| Correct "not found" on questions about non-existent entities | **100%** | 100% |
| Answers with source citations | **100%** | 0% (no sources available) |
| Citation precision (cited paper really supports the answer) | **98%** | n/a |
| Average latency per question | 1.69 s | 1.498 s |

## Recall by question type

| Question type | GraphRAG | LLM only |
|---|---|---|
| Who are the authors of <paper>? | 100% | 0% |
| Which papers use <method>? | 100% | 6% |
| Which methods have been applied to <task>? | 100% | 0% |

## Examples

**Q: Who are the authors of the paper 'CMAverse: A Suite of Functions for Reproducible Causal Mediation Analyses'?**

*Ground truth:* Baoyi Shi, Christine Choirat, Brent A. Coull, Tyler J. VanderWeele, Linda Valeri

*GraphRAG:* The paper **“CMAverse: A Suite of Functions for Reproducible Causal Mediation Analyses”** was authored by:

- Baoyi Shi  
- Christine Choirat  
- Brent A. Coull  
- Tyler J. VanderWeele  
- Linda Valeri  

These author-paper links are recorded in facts F1-F5 [doc:W3165488072].

*LLM only:* The paper **“CMAverse: A Suite of Functions for Reproducible Causal Mediation Analyses”** was authored by  

- **Daniel J. McCaffrey**  
- **Michael J. O’Neill**  
- **Christopher J. O’Neill**  

These three researchers are listed as the authors on the original publication (Journal of Statistical Software, 2022).

**Q: Which papers evaluate the QuantumFrobnicator embedding, and what results did they report?**

*Ground truth:* no such entity exists

*GraphRAG:* The provided facts do not contain any information about a “QuantumFrobnicator” embedding—no paper is listed as evaluating it, nor are any results reported. Consequently, the answer cannot be derived from the given data.

*LLM only:* I’m sorry, but I don’t have any information on a “QuantumFrobnicator embedding,” and I’m not aware of any papers that evaluate it. If you can provide more context or clarify the terminology, I’d be happy to try to help further.

## Method

- Ground truth comes from graph edges (AUTHORED, USES_METHOD, APPLIED_TO). Unanswerable questions name made-up entities.
- An item counts as found if the answer names it: author surname, the first five significant words of a paper title, a cited paper id, or a method name.
- Refusal = the answer says the information isn't available, instead of describing the made-up entity.
- Scores reflect retrieval and grounding. The LLM-only baseline can't see this corpus, so its low recall on corpus-specific questions is expected. The key differences are invented answers versus refusals, and the absence of citations.

Raw per-question results: `results/graphrag_eval.json`.
