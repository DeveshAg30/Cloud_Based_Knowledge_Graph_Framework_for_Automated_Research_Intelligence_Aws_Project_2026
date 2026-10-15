"""Fine-tune SciBERT for relation extraction (RE) on SciERC: typed entity markers + entity-start pooling.

Each example is one ordered pair of entities from the same sentence, marked with their types:
  "We use [S:Method] BERT [/S:Method] for [O:Task] relation extraction [/O:Task] ."   ->  USED-FOR
The classifier reads the encoder states at the two start markers (model: src/ai_model/inference/re_model.py).
Pairs with no annotated relation are NONE: sampled for training; every candidate pair is kept for dev/test,
so precision reflects real use. The epoch with the best dev F1 is kept (early stopping).

Metric: micro precision / recall / F1 over the 7 SciERC relation types (NONE excluded), given gold entities.

  python train_re.py --data_dir processed_data/json --model_dir re_model --output_data_dir output
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inference"))
import re_model as rm  # noqa: E402

LABEL2ID = {l: i for i, l in enumerate(rm.LABELS)}


def load(path, neg_ratio, rng):
    """[{"text", "label"}] for every ordered entity pair per sentence (NONE pairs sampled if neg_ratio is set)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            offset = 0
            for tokens, spans, rels in zip(doc["sentences"], doc["ner"], doc["relations"]):
                ents = [(s - offset, e - offset + 1, t) for s, e, t in spans]          # end-exclusive
                gold = {((h1 - offset, h2 - offset + 1), (t1 - offset, t2 - offset + 1)): lab
                        for h1, h2, t1, t2, lab in rels}
                positives, negatives = [], []
                for h in ents:
                    for t in ents:
                        if h == t:
                            continue
                        lab = gold.get(((h[0], h[1]), (t[0], t[1])))
                        (positives if lab else negatives).append({"text": rm.mark(tokens, h, t),
                                                                  "label": LABEL2ID[lab or "NONE"]})
                rows += positives
                if neg_ratio is None:
                    rows += negatives
                else:
                    rng.shuffle(negatives)
                    rows += negatives[:max(1, int(len(positives) * neg_ratio))] if positives else negatives[:1]
                offset += len(tokens)
    return rows


def scores(pred, gold):
    """Micro P/R/F1 over relation labels (NONE = 0 excluded), plus per-relation F1."""
    pred, gold = np.asarray(pred), np.asarray(gold)
    tp = int(np.sum((pred == gold) & (gold != 0)))
    n_pred, n_gold = int(np.sum(pred != 0)), int(np.sum(gold != 0))
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    out = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "per_relation": {}}
    for i, lab in enumerate(rm.LABELS[1:], 1):
        tpi = int(np.sum((pred == i) & (gold == i)))
        pi, gi = int(np.sum(pred == i)), int(np.sum(gold == i))
        pp, rr = (tpi / pi if pi else 0.0), (tpi / gi if gi else 0.0)
        out["per_relation"][lab] = {"precision": pp, "recall": rr, "f1": 2 * pp * rr / (pp + rr) if pp + rr else 0.0,
                                    "support": gi}
    return out


def encode(tokenizer, rows, max_len):
    enc = tokenizer([r["text"] for r in rows], truncation=True, max_length=max_len)
    s_pos, o_pos = rm.marker_positions(tokenizer, enc["input_ids"])
    return [{"input_ids": ids, "s_pos": s, "o_pos": o, "label": r["label"]}
            for ids, s, o, r in zip(enc["input_ids"], s_pos, o_pos, rows)]


def collate(pad_id):
    def fn(batch):
        width = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, b in enumerate(batch):
            ids[i, :len(b["input_ids"])] = torch.tensor(b["input_ids"])
            mask[i, :len(b["input_ids"])] = 1
        return (ids, mask, torch.tensor([b["s_pos"] for b in batch]), torch.tensor([b["o_pos"] for b in batch]),
                torch.tensor([b["label"] for b in batch]))
    return fn


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, golds = [], []
    for ids, mask, s, o, y in loader:
        _, logits = model(ids, mask, s, o)
        preds += logits.argmax(-1).tolist()
        golds += y.tolist()
    return np.array(preds), np.array(golds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--neg_ratio", type=float, default=1.0, help="NONE examples per positive in training")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only this many training examples")
    ap.add_argument("--data_dir", default=os.environ.get("SM_CHANNEL_TRAIN", "data"))
    ap.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "re_model"))
    ap.add_argument("--output_data_dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "output"))
    args = ap.parse_args()
    torch.manual_seed(13)
    rng = random.Random(13)

    model, tokenizer = rm.new_model(args.model_name)
    train_rows = load(os.path.join(args.data_dir, "train.json"), args.neg_ratio, rng)
    rng.shuffle(train_rows)
    if args.limit:
        train_rows = train_rows[:args.limit]
    dev_rows = load(os.path.join(args.data_dir, "dev.json"), None, rng)
    test_rows = load(os.path.join(args.data_dir, "test.json"), None, rng)
    if args.limit:
        dev_rows, test_rows = dev_rows[:args.limit], test_rows[:args.limit]
    print({"train": len(train_rows), "dev": len(dev_rows), "test": len(test_rows)}, flush=True)

    pad = collate(tokenizer.pad_token_id)
    train = DataLoader(encode(tokenizer, train_rows, args.max_len), batch_size=args.batch_size, shuffle=True, collate_fn=pad)
    dev = DataLoader(encode(tokenizer, dev_rows, args.max_len), batch_size=64, collate_fn=pad)
    test = DataLoader(encode(tokenizer, test_rows, args.max_len), batch_size=64, collate_fn=pad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total = len(train) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total), total)

    best_f1, history, step, started = -1.0, [], 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for ids, mask, s, o, y in train:
            loss, _ = model(ids, mask, s, o, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 25 == 0:
                rate = (time.time() - started) / step
                print(f"step {step}/{total} loss={loss.item():.4f} eta={rate * (total - step) / 60:.1f} min", flush=True)
        dev_scores = scores(*predict(model, dev))
        history.append({"epoch": epoch, "dev": dev_scores})
        print(f"epoch {epoch}: dev_precision={dev_scores['precision']:.4f}; dev_recall={dev_scores['recall']:.4f}; "
              f"dev_f1={dev_scores['f1']:.4f};", flush=True)
        if dev_scores["f1"] > best_f1:  # early stopping: keep the best epoch on dev
            best_f1 = dev_scores["f1"]
            rm.save(model, tokenizer, args.model_dir, {"best_epoch": epoch})

    best, tokenizer = rm.load(args.model_dir)[:2]
    test_scores = scores(*predict(best, test))
    print(f"test_precision={test_scores['precision']:.4f}; test_recall={test_scores['recall']:.4f}; "
          f"test_f1={test_scores['f1']:.4f};", flush=True)
    for lab, s in test_scores["per_relation"].items():
        print(f"{lab:14}{s['precision']:>8.3f}{s['recall']:>8.3f}{s['f1']:>8.3f}{s['support']:>7}")
    os.makedirs(args.output_data_dir, exist_ok=True)
    json.dump({"history": history, "best_dev_f1": best_f1, "test": test_scores},
              open(os.path.join(args.output_data_dir, "re_metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
