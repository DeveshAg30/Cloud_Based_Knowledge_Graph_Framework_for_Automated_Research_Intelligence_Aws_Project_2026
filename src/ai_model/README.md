# AI model: SciBERT named-entity recognition

- `train/train.py` fine-tunes `allenai/scibert_scivocab_uncased` on SciERC (BIO tagging, entity-level P/R/F1).
  Test F1 is 0.618 (see `results/SCIBERT_RESULTS.md`). It runs in the SageMaker Hugging Face container, on Colab or on a laptop CPU.
- `train/train_re.py` fine-tunes a second SciBERT model for **relation extraction** on SciERC's relation annotations
  (USED-FOR, EVALUATE-FOR, COMPARE, HYPONYM-OF, PART-OF, FEATURE-OF, CONJUNCTION), using typed entity markers and
  entity-start pooling (`inference/re_model.py`), micro P/R/F1 over all candidate pairs, and the best epoch on dev.
  `train/calibrate_re.py` tunes the decision rule on dev. Test F1 is 0.661 (see `results/SCIBERT_RESULTS.md`).
- `inference/inference.py` is the SageMaker inference handler: `{"sentences": [...]}` →
  `{"entities": [{text, label, sentence}], "relations": [{head, rel, tail, score, ...}]}`. It runs NER, then classifies
  each pair of entities in a sentence with the RE model (the RE model is optional: without `re/` in the archive the
  endpoint returns entities only). It is served from a serverless endpoint (`research-kg-ner`, 3072 MB).
