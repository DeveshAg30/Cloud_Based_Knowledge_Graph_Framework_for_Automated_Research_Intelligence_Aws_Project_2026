"""SageMaker Hugging Face inference handler: SciBERT NER and (optionally) SciBERT relation extraction.

model.tar.gz layout:  <NER model files>  re/<RE model files>  code/inference.py
The RE model is optional; without re/ the endpoint returns entities only.

Request:  {"sentences": ["We apply BERT to relation extraction on SciERC.", ...]}
Response: {"entities":  [{"text": "BERT", "label": "Method", "sentence": "..."}, ...],
           "relations": [{"head": "BERT", "head_label": "Method", "rel": "USED-FOR",
                          "tail": "relation extraction", "tail_label": "Task", "score": 0.93, "sentence": "..."}]}
"""

import json
import os
import re

import torch
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

WORD = re.compile(r"\w+(?:[-.@/]\w+)*|[^\w\s]")
RE_TYPES = {"Method", "Task", "Metric", "Material"}   # entity types worth relating in the knowledge graph
MAX_PAIRS_PER_SENTENCE = 20
RE_THRESHOLD = 0.5


def model_fn(model_dir):
    models = {"ner_tok": AutoTokenizer.from_pretrained(model_dir),
              "ner": AutoModelForTokenClassification.from_pretrained(model_dir).eval()}
    re_dir = os.path.join(model_dir, "re")
    if os.path.isdir(re_dir):
        models["re_tok"] = AutoTokenizer.from_pretrained(re_dir)
        models["re"] = AutoModelForSequenceClassification.from_pretrained(re_dir).eval()
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


def _mark(words, head, tail):
    """Same entity-marker format as train_re.py ([E1] head [/E1] ... [E2] tail [/E2]); spans are end-exclusive."""
    out = []
    for i, w in enumerate(words):
        if i == head[0]:
            out.append("[E1]")
        if i == tail[0]:
            out.append("[E2]")
        out.append(w)
        if i == head[1] - 1:
            out.append("[/E1]")
        if i == tail[1] - 1:
            out.append("[/E2]")
    return " ".join(out)


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
    tok, model = models["re_tok"], models["re"]
    candidates = []
    for sentence, words, spans in per_sentence:
        ents = [s for s in spans if s[2] in RE_TYPES]
        pairs = [(h, t) for h in ents for t in ents if h != t][:MAX_PAIRS_PER_SENTENCE]
        for h, t in pairs:
            candidates.append((sentence, words, h, t, _mark(words, h, t)))
    out = []
    for start in range(0, len(candidates), 32):
        batch = candidates[start:start + 32]
        enc = tok([c[4] for c in batch], truncation=True, max_length=128, padding=True, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        # Calibrated on the dev set by calibrate_re.py: prior correction for NONE, then a confidence threshold.
        adjusted = probs.clone()
        adjusted[:, 0] *= getattr(model.config, "re_none_weight", 1.0)
        indices = adjusted.argmax(dim=-1)
        values = probs.gather(1, indices.unsqueeze(1)).squeeze(1)
        threshold = getattr(model.config, "re_threshold", RE_THRESHOLD)
        for (sentence, words, h, t, _), score, idx in zip(batch, values.tolist(), indices.tolist()):
            label = model.config.id2label[idx]
            if label != "NONE" and score >= threshold:
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
