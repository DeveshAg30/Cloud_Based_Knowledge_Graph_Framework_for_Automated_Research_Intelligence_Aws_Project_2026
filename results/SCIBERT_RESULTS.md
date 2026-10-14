# SciBERT NER on SciERC: results

Fine-tuned `allenai/scibert_scivocab_uncased` for token classification (BIO tags) on the SciERC
benchmark using `src/ai_model/train/train.py`.

**Setup:** 4 epochs, learning rate 3e-5, batch size 16, max length 256, warm-up 10%, weight decay 0.01.
Trained on a laptop CPU (Intel i7-1185G7, 4 cores) in about 35 minutes, because the AWS Free-plan
account has a SageMaker training quota of 0. Data: 1,861 train / 275 dev / 551 test sentences.
**Scoring:** entity-level exact match (span + type), micro-averaged.

| Split | Precision | Recall | F1 |
|---|---|---|---|
| Dev | 0.594 | 0.678 | 0.633 |
| **Test** | **0.593** | **0.645** | **0.618** |

## Test set, per entity type

| Type | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Generic | 0.690 | 0.730 | 0.710 | 241 |
| Material | 0.552 | 0.639 | 0.592 | 158 |
| Method | 0.699 | 0.714 | 0.707 | 430 |
| Metric | 0.570 | 0.690 | 0.624 | 71 |
| OtherScientificTerm | 0.505 | 0.567 | 0.534 | 520 |
| Task | 0.557 | 0.597 | 0.576 | 263 |

For reference, the SciBERT paper (Beltagy et al., 2019) reports about 0.65-0.67 F1 on SciERC NER, trained on a GPU with tuning.

**How it's used in the pipeline:** `ExtractorMode=sagemaker` sends paper text to a SageMaker serverless endpoint
running this model (`src/ai_model/inference/inference.py`). It maps Method→Method, Task→Task, Metric→Metric and
Material→Dataset, and merges the results with a curated term dictionary for well-known names the model sometimes misses.


## Effect on the knowledge graph

The same corpus (299 OpenAlex papers, 4 samples and uploaded PDFs) was ingested once with the dictionary/regex
baseline (`ExtractorMode=rules`) and once with SciBERT on SageMaker plus the dictionary (`ExtractorMode=sagemaker`).

| | Rules only | SciBERT + dictionary |
|---|---|---|
| Papers with at least one extracted entity | 189 / 304 | **305 / 305** |
| Paper → entity links | 377 | **6063** |
| Method nodes | 49 | **2339** |
| Task nodes | 14 | **1653** |
| Dataset nodes | 17 | **728** |
| Metric nodes | 7 | **226** |
| Method → Task / Dataset / Metric relations | ≈0 | **2630** |

Deployment: SageMaker **serverless** endpoint `research-kg-ner` (3072 MB, max concurrency 10), Hugging Face
inference container (transformers 4.49, PyTorch 2.6, CPU). Latency: about 6.6 s for a cold start, about 0.5 s
warm for a batch of sentences. Full ingestion of the 305 documents: 6 minutes, 0 failures.

Post-processing applied to model output: entity canonicalisation (alias table, learned abbreviations, plural removal),
removal of generic phrases ("dataset", "results", ...), tidying of trailing words and punctuation, and one type per name
by majority vote. PDF text is repaired first: ligatures, hyphenation across line breaks, split small-caps.


## Relation extraction (second SciBERT model)

`src/ai_model/train/train_re.py` fine-tunes `allenai/scibert_scivocab_uncased` as a sentence-pair classifier. Each
candidate pair of entities in a sentence is marked (`[E1] ... [/E1]`, `[E2] ... [/E2]`) and classified as one of the 7 SciERC
relations or NONE.

**Setup:** 2 epochs, learning rate 3e-5, batch size 16, max length 128. Training used all 3,215 annotated relations plus 0.7
sampled NONE pairs per positive (5,518 examples). Dev and test use **every** candidate pair among gold entities
(2,488 / 5,062 pairs, about 80% NONE), so precision reflects real use. Trained on the same laptop CPU in about 70 minutes.

**Calibration** (`calibrate_re.py`, tuned on dev only): training saw far fewer NONE pairs than real text contains, so the
NONE probability is re-weighted (prior correction, x2.0) and a relation must reach confidence 0.45.

| SciERC test (gold entities) | Precision | Recall | F1 |
|---|---|---|---|
| Raw model (argmax) | 0.292 | 0.697 | 0.412 |
| **Calibrated (deployed)** | **0.358** | **0.516** | **0.423** |

| Relation | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| USED-FOR | 0.364 | 0.473 | 0.411 | 533 |
| FEATURE-OF | 0.297 | 0.186 | 0.229 | 59 |
| HYPONYM-OF | 0.351 | 0.791 | 0.486 | 67 |
| PART-OF | 0.476 | 0.159 | 0.238 | 63 |
| COMPARE | 0.360 | 0.711 | 0.478 | 38 |
| CONJUNCTION | 0.386 | 0.854 | 0.532 | 123 |
| EVALUATE-FOR | 0.287 | 0.495 | 0.363 | 91 |

**How the graph uses it:** only relations with a clear meaning for the ontology are kept. *Method used-for Task* becomes
`APPLIED_TO`, *Dataset or Metric evaluate-for Method or Task* becomes `EVALUATED_ON` / `MEASURED_BY`, *compare* becomes
`COMPARED_WITH`, *hyponym-of* becomes `IS_A` and *part-of* becomes `PART_OF`. Pairs whose direction or types don't fit a rule
are discarded, which raises precision further in practice. Both models are served from one SageMaker serverless endpoint.

**Limitations:** relation F1 is modest. SciERC relations are hard and imbalanced: FEATURE-OF and PART-OF each have about
60 test examples and score low. Training was limited to 2 CPU epochs. More epochs on a GPU, or joint entity and relation
training, would improve it.
