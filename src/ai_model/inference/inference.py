"""SageMaker Hugging Face inference handler for the fine-tuned SciBERT NER model.

Request:  {"sentences": ["We apply BERT to relation extraction on SciERC.", ...]}
Response: {"entities": [{"text": "BERT", "label": "Method", "sentence": "..."}, ...]}
"""

import json
import re

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

WORD = re.compile(r"\w+(?:[-.@/]\w+)*|[^\w\s]")


def model_fn(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def input_fn(body, content_type):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type {content_type}")
    return json.loads(body)


def _spans(words, tags):
    """Merge BIO tags into (start, end, type) word spans.

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


def predict_fn(data, model_and_tokenizer):
    tokenizer, model = model_and_tokenizer
    entities = []
    sentences = data.get("sentences") or []
    for batch_start in range(0, len(sentences), 16):
        batch = sentences[batch_start:batch_start + 16]
        words = [WORD.findall(s) for s in batch]
        enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=256,
                        padding=True, return_tensors="pt")
        with torch.no_grad():
            pred = model(**enc).logits.argmax(-1).tolist()
        for i, sentence in enumerate(batch):
            tags, prev = ["O"] * len(words[i]), None
            for pos, word_id in enumerate(enc.word_ids(batch_index=i)):
                if word_id is not None and word_id != prev:
                    tags[word_id] = model.config.id2label[pred[i][pos]]
                prev = word_id
            for s, e, kind in _spans(words[i], tags):
                entities.append({"text": " ".join(words[i][s:e]), "label": kind, "sentence": sentence})
    return {"entities": entities}


def output_fn(prediction, accept):
    return json.dumps(prediction)
