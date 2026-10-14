"""Fine-tune SciBERT for scientific NER on SciERC (Method / Task / Metric / Material / ...).

Runs inside the SageMaker Hugging Face training container. Expects the `train` channel to contain the
SciERC processed JSON files (train.json, dev.json, test.json), one document per line:
  {"doc_key": ..., "sentences": [[tokens]], "ner": [[[start, end, label], ...] per sentence], ...}
NER offsets are document-level token indices (inclusive).
"""

import argparse
import inspect
import json
import os

import numpy as np
from datasets import Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments)

ENTITY_TYPES = ["Method", "Task", "Metric", "Material", "OtherScientificTerm", "Generic"]
LABELS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


# Entity-level (exact span + type) scoring, same definition as seqeval's default mode. Implemented here
# because seqeval only ships a setup.py that no longer builds on current Python / setuptools.
def bio_spans(tags):
    spans, start, kind = set(), None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag == "O" or tag.startswith("B-") or (tag.startswith("I-") and tag[2:] != kind):
            if start is not None:
                spans.add((start, i, kind))
            start, kind = (i, tag[2:]) if tag != "O" else (None, None)
    return spans


def entity_scores(y_true, y_pred):
    """Micro precision/recall/F1 overall and per entity type."""
    counts = {}  # type -> [true positives, predicted, gold]
    for t_seq, p_seq in zip(y_true, y_pred):
        gold, pred = bio_spans(t_seq), bio_spans(p_seq)
        for kind in {s[2] for s in gold | pred}:
            g = {s for s in gold if s[2] == kind}
            p = {s for s in pred if s[2] == kind}
            c = counts.setdefault(kind, [0, 0, 0])
            c[0] += len(g & p)
            c[1] += len(p)
            c[2] += len(g)

    def prf(tp, n_pred, n_gold):
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_gold if n_gold else 0.0
        return precision, recall, (2 * precision * recall / (precision + recall) if precision + recall else 0.0)

    total = [sum(c[i] for c in counts.values()) for i in range(3)]
    per_type = {k: dict(zip(("precision", "recall", "f1", "support"), (*prf(*c), c[2]))) for k, c in sorted(counts.items())}
    p, r, f = prf(*total)
    return {"precision": p, "recall": r, "f1": f, "per_type": per_type}


def load_scierc(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            offset = 0
            for tokens, spans in zip(doc["sentences"], doc["ner"]):
                tags = ["O"] * len(tokens)
                for start, end, label in spans:
                    s, e = start - offset, end - offset
                    if 0 <= s <= e < len(tokens):
                        tags[s] = f"B-{label}"
                        for i in range(s + 1, e + 1):
                            tags[i] = f"I-{label}"
                rows.append({"tokens": tokens, "tags": [LABEL2ID[t] for t in tags]})
                offset += len(tokens)
    return Dataset.from_list(rows)


def tokenize_and_align(batch, tokenizer, max_len):
    enc = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, max_length=max_len)
    labels = []
    for i, tags in enumerate(batch["tags"]):
        prev, row = None, []
        for word_id in enc.word_ids(batch_index=i):
            row.append(-100 if word_id is None or word_id == prev else tags[word_id])
            prev = word_id
        labels.append(row)
    enc["labels"] = labels
    return enc


def to_tag_sequences(predictions, label_ids):
    preds = np.argmax(predictions, axis=-1)
    true_seqs, pred_seqs = [], []
    for p_row, l_row in zip(preds, label_ids):
        t, p = [], []
        for pi, li in zip(p_row, l_row):
            if li != -100:
                t.append(LABELS[li])
                p.append(LABELS[pi])
        true_seqs.append(t)
        pred_seqs.append(p)
    return true_seqs, pred_seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--learning_rate", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--data_dir", default=os.environ.get("SM_CHANNEL_TRAIN", "data"))
    ap.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "model"))
    ap.add_argument("--output_data_dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "output"))
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    splits = {name: load_scierc(os.path.join(args.data_dir, f"{name}.json")) for name in ("train", "dev", "test")}
    tokenized = {name: ds.map(lambda b: tokenize_and_align(b, tokenizer, args.max_len), batched=True,
                              remove_columns=ds.column_names) for name, ds in splits.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/checkpoints",
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=50,
            save_strategy="no",
            report_to=[],
        ),
        train_dataset=tokenized["train"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        # transformers >= 4.46 renamed `tokenizer` to `processing_class` (v5 removed the old name)
        **({"processing_class": tokenizer} if "processing_class" in inspect.signature(Trainer.__init__).parameters
           else {"tokenizer": tokenizer}),
    )
    trainer.train()

    metrics = {}
    for split in ("dev", "test"):
        out = trainer.predict(tokenized[split])
        y_true, y_pred = to_tag_sequences(out.predictions, out.label_ids)
        metrics[split] = entity_scores(y_true, y_pred)
        # These lines are parsed by the metric_definitions in scripts/sagemaker_launch.py
        print(f"{split}_precision={metrics[split]['precision']:.4f}; {split}_recall={metrics[split]['recall']:.4f}; "
              f"{split}_f1={metrics[split]['f1']:.4f};")
        if split == "test":
            print(f"{'type':22}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>9}")
            for kind, s in metrics[split]["per_type"].items():
                print(f"{kind:22}{s['precision']:>10.4f}{s['recall']:>10.4f}{s['f1']:>10.4f}{s['support']:>9}")

    os.makedirs(args.output_data_dir, exist_ok=True)
    with open(os.path.join(args.output_data_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)


if __name__ == "__main__":
    main()
