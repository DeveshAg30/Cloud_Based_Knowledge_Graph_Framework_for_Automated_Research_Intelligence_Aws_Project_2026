"""Calibrate the relation-extraction decision rule on the dev set, then score the test set.

Training uses a sampled set of NONE pairs, but real sentences are mostly unrelated pairs, so the raw model
over-predicts relations (high recall, low precision). Two knobs are tuned on dev F1:
  * none_weight: prior correction, the NONE probability is multiplied by this factor before the argmax;
  * threshold:   the winning relation's probability must reach this value, otherwise NONE.
Both are written to re_config.json in the model directory and read by src/ai_model/inference/inference.py.
The test set is never used for tuning.

  python calibrate_re.py --data_dir processed_data/json --model_dir re_model --output_data_dir output
"""

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inference"))
import re_model as rm  # noqa: E402
from train_re import load, scores  # noqa: E402


def decide(probs, t, none_weight=1.0):
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

    model, tokenizer, config = rm.load(args.model_dir)
    rng = random.Random(13)
    data = {s: load(os.path.join(args.data_dir, f"{s}.json"), None, rng) for s in ("dev", "test")}
    probs = {s: rm.predict_proba(model, tokenizer, [r["text"] for r in d], args.max_len).numpy() for s, d in data.items()}
    gold = {s: np.array([r["label"] for r in d]) for s, d in data.items()}

    sweep = []
    for w in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
        for t in np.round(np.arange(0.0, 0.96, 0.05), 2):
            sweep.append((w, float(t), scores(decide(probs["dev"], t, w), gold["dev"])["f1"]))
    best_w, best_t, best_dev = max(sweep, key=lambda x: x[2])
    test = scores(decide(probs["test"], best_t, best_w), gold["test"])
    raw = scores(decide(probs["test"], 0.0, 1.0), gold["test"])
    print(f"none_weight={best_w} threshold={best_t} dev_f1={best_dev:.4f}")
    print(f"test_precision={test['precision']:.4f}; test_recall={test['recall']:.4f}; test_f1={test['f1']:.4f};")
    print(f"(raw argmax test: P={raw['precision']:.4f} R={raw['recall']:.4f} F1={raw['f1']:.4f})")
    for lab, s in test["per_relation"].items():
        print(f"{lab:14}{s['precision']:>8.3f}{s['recall']:>8.3f}{s['f1']:>8.3f}{s['support']:>7}")

    config.update({"none_weight": best_w, "threshold": best_t})
    json.dump(config, open(os.path.join(args.model_dir, "re_config.json"), "w"), indent=2)
    os.makedirs(args.output_data_dir, exist_ok=True)
    json.dump({"none_weight": best_w, "threshold": best_t, "dev_f1": best_dev, "test": test, "test_raw": raw,
               "sweep": sweep}, open(os.path.join(args.output_data_dir, "re_metrics_calibrated.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
