"""Calibrate the relation-extraction decision threshold on the dev set, then score the test set.

Training uses a balanced sample of NONE pairs, but real sentences are mostly unrelated pairs, so the raw model
over-predicts relations (high recall, low precision). Two calibration knobs are tuned on dev F1:
  * none_weight: prior correction, the NONE probability is multiplied by this factor before the argmax;
  * threshold:   the winning relation's probability must reach this value, otherwise NONE.
Both are stored in the model config (`re_none_weight`, `re_threshold`) and read by
src/ai_model/inference/inference.py. The test set is never used for tuning.

  python calibrate_re.py --data_dir processed_data/json --model_dir re_model --output_data_dir output
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_re import load, scores


def probabilities(model, tokenizer, texts, max_len, batch=64):
    out = []
    model.eval()
    for i in range(0, len(texts), batch):
        enc = tokenizer(texts[i:i + batch], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        with torch.no_grad():
            out.append(torch.softmax(model(**enc).logits, dim=-1).numpy())
    return np.concatenate(out)


def decide(probs, t, none_weight=1.0):
    """Prior correction (scale the NONE probability by none_weight), then require the winning relation's
    probability to reach t."""
    adjusted = probs.copy()
    adjusted[:, 0] *= none_weight
    pred = adjusted.argmax(axis=1)
    return np.where((pred != 0) & (probs[np.arange(len(probs)), pred] >= t), pred, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--output_data_dir", required=True)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    rng = random.Random(13)
    data = {s: load(os.path.join(args.data_dir, f"{s}.json"), None, rng) for s in ("dev", "test")}
    probs = {s: probabilities(model, tokenizer, d["text"], args.max_len) for s, d in data.items()}
    gold = {s: np.array(d["label"]) for s, d in data.items()}

    sweep = []
    for w in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0):
        for t in np.round(np.arange(0.0, 0.96, 0.05), 2):
            sweep.append((w, float(t), scores(decide(probs["dev"], t, w), gold["dev"])["f1"]))
    best_w, best_t, best_dev = max(sweep, key=lambda x: x[2])
    test = scores(decide(probs["test"], best_t, best_w), gold["test"])
    uncalibrated = scores(decide(probs["test"], 0.0, 1.0), gold["test"])
    print(f"none_weight={best_w} threshold={best_t} dev_f1={best_dev:.4f}")
    print(f"test_precision={test['precision']:.4f}; test_recall={test['recall']:.4f}; test_f1={test['f1']:.4f};")
    print(f"(uncalibrated test: P={uncalibrated['precision']:.4f} R={uncalibrated['recall']:.4f} F1={uncalibrated['f1']:.4f})")
    for lab, s in test["per_relation"].items():
        print(f"{lab:14}{s['precision']:>8.3f}{s['recall']:>8.3f}{s['f1']:>8.3f}{s['support']:>7}")

    model.config.re_threshold = best_t
    model.config.re_none_weight = best_w
    model.config.save_pretrained(args.model_dir)
    os.makedirs(args.output_data_dir, exist_ok=True)
    json.dump({"none_weight": best_w, "threshold": best_t, "dev_f1": best_dev, "test": test, "test_uncalibrated": uncalibrated,
               "sweep": sweep}, open(os.path.join(args.output_data_dir, "re_metrics_calibrated.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
