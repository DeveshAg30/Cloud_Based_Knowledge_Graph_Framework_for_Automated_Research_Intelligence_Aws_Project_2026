# AI model: SciBERT named-entity recognition

- `train/train.py` fine-tunes `allenai/scibert_scivocab_uncased` on SciERC (BIO tagging, entity-level P/R/F1).
  Test F1 is 0.618 (see `results/SCIBERT_RESULTS.md`). It runs in the SageMaker Hugging Face container, on Colab or on a laptop CPU.
- `inference/inference.py` is the SageMaker inference handler: `{"sentences": [...]}` → `{"entities": [{text, label, sentence}]}`.
  It is served from a serverless endpoint (`research-kg-ner`, 3072 MB); deploy it with `scripts/sagemaker_launch.py`.
