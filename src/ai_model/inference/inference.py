"""SageMaker Hugging Face inference handler: SciBERT NER and (optionally) SciBERT relation extraction.

model.tar.gz layout:  <NER model files>  re/<RE model: encoder, tokenizer, head.pt, re_config.json>
                      code/inference.py  code/re_model.py
The RE model is optional; without re/ the endpoint returns entities only.

Request:  {"sentences": ["We apply BERT to relation extraction on SciERC.", ...]}
Response: {"entities":  [{"text": "BERT", "label": "Method", "sentence": "..."}, ...],
           "relations": [{"head": "BERT", "head_label": "Method", "rel": "USED-FOR",
                          "tail": "relation extraction", "tail_label": "Task", "score": 0.93, "sentence": "..."}]}
"""

import json
import os
import re
import sys

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # re_model.py ships next to this file
import re_model as rm  # noqa: E402

WORD = re.compile(r"\w+(?:[-.@/]\w+)*|[^\w\s]")
RE_TYPES = {"Method", "Task", "Metric", "Material"}   # entity types worth relating in the knowledge graph
MAX_PAIRS_PER_SENTENCE = 20


def model_fn(model_dir):
    models = {"ner_tok": AutoTokenizer.from_pretrained(model_dir),
              "ner": AutoModelForTokenClassification.from_pretrained(model_dir).eval()}
    re_dir = os.path.join(model_dir, "re")
    if os.path.isdir(re_dir):
        models["re"], models["re_tok"], models["re_config"] = rm.load(re_dir)
    return models


def input_fn(body, content_type):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type {content_type}")
    return json.loads(body)


def _spans(words, tags):
    """Merge BIO tags into (start, end, type) word spans (end exclusive).

    An I- tag always continues the current span even if its type differs ("large/B-Task language/I-Method
    models/I-Method"); the span takes the type of its last word, which is the head noun in English phrases.
    """
    spans, start, kind = [], None, None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("I-") and start is not None:
            kind = tag[2:]
            continue
        if start is not None:
            spans.append((start, i, kind))
            start, kind = None, None
        if tag != "O":
            start, kind = i, tag[2:]
    return spans


def _ner(models, sentences):
    tok, model = models["ner_tok"], models["ner"]
    per_sentence = []
    for start in range(0, len(sentences), 16):
        batch = sentences[start:start + 16]
        words = [WORD.findall(s) for s in batch]
        enc = tok(words, is_split_into_words=True, truncation=True, max_length=256, padding=True, return_tensors="pt")
        with torch.no_grad():
            pred = model(**enc).logits.argmax(-1).tolist()
        for i, sentence in enumerate(batch):
            tags, prev = ["O"] * len(words[i]), None
            for pos, word_id in enumerate(enc.word_ids(batch_index=i)):
                if word_id is not None and word_id != prev:
                    tags[word_id] = model.config.id2label[pred[i][pos]]
                prev = word_id
            per_sentence.append((sentence, words[i], _spans(words[i], tags)))
    return per_sentence


def _relations(models, per_sentence):
    if "re" not in models:
        return []
    config = models["re_config"]
    candidates = []
    for sentence, words, spans in per_sentence:
        ents = [s for s in spans if s[2] in RE_TYPES]
        for h, t in [(h, t) for h in ents for t in ents if h != t][:MAX_PAIRS_PER_SENTENCE]:
            candidates.append((sentence, words, h, t))
    if not candidates:
        return []
    probs = rm.predict_proba(models["re"], models["re_tok"], [rm.mark(c[1], c[2], c[3]) for c in candidates])
    # Calibrated on the dev set by calibrate_re.py: prior correction for NONE, then a confidence threshold.
    adjusted = probs.clone()
    adjusted[:, 0] *= config.get("none_weight", 1.0)
    indices = adjusted.argmax(dim=-1)
    values = probs.gather(1, indices.unsqueeze(1)).squeeze(1)
    out = []
    for (sentence, words, h, t), score, idx in zip(candidates, values.tolist(), indices.tolist()):
        label = rm.LABELS[idx]
        if label != "NONE" and score >= config.get("threshold", 0.5):
            out.append({"head": " ".join(words[h[0]:h[1]]), "head_label": h[2], "rel": label,
                        "tail": " ".join(words[t[0]:t[1]]), "tail_label": t[2], "score": round(score, 3),
                        "sentence": sentence})
    return out


def predict_fn(data, models):
    sentences = data.get("sentences") or []
    per_sentence = _ner(models, sentences)
    entities = [{"text": " ".join(words[s:e]), "label": kind, "sentence": sentence}
                for sentence, words, spans in per_sentence for s, e, kind in spans]
    return {"entities": entities, "relations": _relations(models, per_sentence)}


def output_fn(prediction, accept):
    return json.dumps(prediction)
