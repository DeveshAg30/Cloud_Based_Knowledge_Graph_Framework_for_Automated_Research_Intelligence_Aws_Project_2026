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

Deployment: SageMaker **serverless** endpoint `research-kg-ner` (3072 MB, max concurrency 5), Hugging Face
inference container (transformers 4.49, PyTorch 2.6, CPU). Latency: about 6.6 s for a cold start, about 0.5 s
warm for a batch of sentences. Full ingestion of the 305 documents: 6 minutes, 0 failures.

Post-processing applied to model output: entity canonicalisation (alias table, learned abbreviations, plural removal),
removal of generic phrases ("dataset", "results", ...), tidying of trailing words and punctuation, and one type per name
by majority vote. PDF text is repaired first: ligatures, hyphenation across line breaks, split small-caps.
