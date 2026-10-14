"""Fine-tune SciBERT for relation extraction (RE) on SciERC.

Each example is one ordered pair of entities from the same sentence, marked in the text:
  "We use [E1] BERT [/E1] for [E2] relation extraction [/E2] ."   ->  USED-FOR
Pairs with no annotated relation are labelled NONE (negatives are sampled for training; every
candidate pair is kept for evaluation, so precision reflects real use).

SciERC relation types: USED-FOR, FEATURE-OF, HYPONYM-OF, PART-OF, COMPARE, CONJUNCTION, EVALUATE-FOR.
Metric: micro precision / recall / F1 over the 7 relation types (NONE excluded), given gold entities.

  python train_re.py --data_dir processed_data/json --model_dir re_model --output_data_dir output --epochs 3
"""

import argparse
import inspect
import json
import os
import random

import numpy as np
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments)

RELATIONS = ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "COMPARE", "CONJUNCTION", "EVALUATE-FOR"]
LABELS = ["NONE"] + RELATIONS
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
MARKERS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]


def mark(tokens, head, tail):
    """Insert entity markers around the head (E1) and tail (E2) spans (inclusive word indices)."""
    out = []
    for i, tok in enumerate(tokens):
        if i == head[0]:
            out.append("[E1]")
        if i == tail[0]:
            out.append("[E2]")
        out.append(tok)
        if i == head[1]:
            out.append("[/E1]")
        if i == tail[1]:
            out.append("[/E2]")
    return " ".join(out)


def load(path, neg_ratio, rng):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            offset = 0
            for tokens, spans, rels in zip(doc["sentences"], doc["ner"], doc["relations"]):
                ents = [(s - offset, e - offset) for s, e, _ in spans]
                gold = {((h1 - offset, h2 - offset), (t1 - offset, t2 - offset)): lab for h1, h2, t1, t2, lab in rels}
                positives, negatives = [], []
                for h in ents:
                    for t in ents:
                        if h == t:
                            continue
                        lab = gold.get((h, t))
                        (positives if lab else negatives).append({"text": mark(tokens, h, t), "label": LABEL2ID[lab or "NONE"]})
                rows += positives
                if neg_ratio is None:
                    rows += negatives
                else:
                    rng.shuffle(negatives)
                    rows += negatives[:max(1, int(len(positives) * neg_ratio))] if positives else negatives[:1]
                offset += len(tokens)
    return Dataset.from_list(rows)


def scores(pred, gold):
    """Micro P/R/F1 over relation labels (NONE = 0 excluded), plus per-relation F1."""
    tp = int(np.sum((pred == gold) & (gold != 0)))
    n_pred, n_gold = int(np.sum(pred != 0)), int(np.sum(gold != 0))
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    out = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "per_relation": {}}
    for i, lab in enumerate(LABELS[1:], 1):
        tpi = int(np.sum((pred == i) & (gold == i)))
        pi, gi = int(np.sum(pred == i)), int(np.sum(gold == i))
        pp, rr = (tpi / pi if pi else 0.0), (tpi / gi if gi else 0.0)
        out["per_relation"][lab] = {"precision": pp, "recall": rr, "f1": 2 * pp * rr / (pp + rr) if pp + rr else 0.0,
                                    "support": gi}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--learning_rate", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--neg_ratio", type=float, default=0.7, help="NONE examples per positive in training")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--data_dir", default=os.environ.get("SM_CHANNEL_TRAIN", "data"))
    ap.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "re_model"))
    ap.add_argument("--output_data_dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "output"))
    args = ap.parse_args()
    rng = random.Random(13)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": MARKERS})
    splits = {"train": load(os.path.join(args.data_dir, "train.json"), args.neg_ratio, rng),
              "dev": load(os.path.join(args.data_dir, "dev.json"), None, rng),
              "test": load(os.path.join(args.data_dir, "test.json"), None, rng)}
    print({k: len(v) for k, v in splits.items()})
    tok = {k: v.map(lambda b: tokenizer(b["text"], truncation=True, max_length=args.max_len), batched=True, remove_columns=["text"])
           for k, v in splits.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID)
    model.resize_token_embeddings(len(tokenizer))

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="/tmp/re_checkpoints", num_train_epochs=args.epochs,
                               learning_rate=args.learning_rate, per_device_train_batch_size=args.batch_size,
                               per_device_eval_batch_size=64, weight_decay=0.01, warmup_ratio=0.1,
                               logging_steps=50, save_strategy="no", report_to=[]),
        train_dataset=tok["train"],
        data_collator=DataCollatorWithPadding(tokenizer),
        **({"processing_class": tokenizer} if "processing_class" in inspect.signature(Trainer.__init__).parameters
           else {"tokenizer": tokenizer}),
    )
    trainer.train()

    metrics = {}
    for split in ("dev", "test"):
        out = trainer.predict(tok[split])
        metrics[split] = scores(np.argmax(out.predictions, axis=-1), out.label_ids)
        m = metrics[split]
        print(f"{split}_precision={m['precision']:.4f}; {split}_recall={m['recall']:.4f}; {split}_f1={m['f1']:.4f};")
    for lab, s in metrics["test"]["per_relation"].items():
        print(f"{lab:14}{s['precision']:>8.3f}{s['recall']:>8.3f}{s['f1']:>8.3f}{s['support']:>7}")

    os.makedirs(args.output_data_dir, exist_ok=True)
    json.dump(metrics, open(os.path.join(args.output_data_dir, "re_metrics.json"), "w"), indent=2)
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)


if __name__ == "__main__":
    main()
