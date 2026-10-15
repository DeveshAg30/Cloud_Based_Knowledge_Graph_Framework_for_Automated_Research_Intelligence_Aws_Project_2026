"""Relation-extraction model shared by training (src/ai_model/train/train_re.py) and the SageMaker handler.

Design (PURE-style, Zhong & Chen 2021; entity-start pooling, Baldini Soares et al. 2019):
  * typed entity markers: "[S:Method] BERT [/S:Method] is used for [O:Task] parsing [/O:Task]"
  * the representation is the concatenation of the encoder states at the two *start* markers,
    followed by a linear classifier over NONE + the 7 SciERC relations.

Saved layout (a directory): encoder files (save_pretrained), tokenizer files, head.pt, re_config.json.
"""

import json
import os

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

RELATIONS = ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "COMPARE", "CONJUNCTION", "EVALUATE-FOR"]
LABELS = ["NONE"] + RELATIONS
ENTITY_TYPES = ["Method", "Task", "Metric", "Material", "OtherScientificTerm", "Generic"]
MARKERS = [f"[{role}:{t}]" for t in ENTITY_TYPES for role in ("S", "/S", "O", "/O")]


def mark(words, head, tail):
    """Insert typed markers. head/tail = (start, end_exclusive, type) over word indices."""
    out = []
    for i, w in enumerate(words):
        if i == head[0]:
            out.append(f"[S:{head[2]}]")
        if i == tail[0]:
            out.append(f"[O:{tail[2]}]")
        out.append(w)
        if i == head[1] - 1:
            out.append(f"[/S:{head[2]}]")
        if i == tail[1] - 1:
            out.append(f"[/O:{tail[2]}]")
    return " ".join(out)


class RelationClassifier(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(2 * hidden, num_labels)

    def forward(self, input_ids, attention_mask, s_pos, o_pos, labels=None):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        rows = torch.arange(states.size(0), device=states.device)
        pooled = torch.cat([states[rows, s_pos], states[rows, o_pos]], dim=-1)
        logits = self.head(self.dropout(pooled))
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return loss, logits


def marker_positions(tokenizer, input_ids):
    """Index of the first [S:*] and [O:*] start marker in each tokenized example."""
    s_ids = {tokenizer.convert_tokens_to_ids(m) for m in MARKERS if m.startswith("[S:")}
    o_ids = {tokenizer.convert_tokens_to_ids(m) for m in MARKERS if m.startswith("[O:")}
    s_pos, o_pos = [], []
    for ids in input_ids:
        s_pos.append(next((i for i, t in enumerate(ids) if t in s_ids), 0))
        o_pos.append(next((i for i, t in enumerate(ids) if t in o_ids), 0))
    return s_pos, o_pos


def new_model(base="allenai/scibert_scivocab_uncased"):
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.add_special_tokens({"additional_special_tokens": MARKERS})
    encoder = AutoModel.from_pretrained(base)
    encoder.resize_token_embeddings(len(tokenizer))
    return RelationClassifier(encoder, len(LABELS)), tokenizer


def save(model, tokenizer, directory, extra=None):
    os.makedirs(directory, exist_ok=True)
    model.encoder.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    torch.save(model.head.state_dict(), os.path.join(directory, "head.pt"))
    json.dump({"labels": LABELS, "pooling": "entity_start", "markers": "typed", **(extra or {})},
              open(os.path.join(directory, "re_config.json"), "w"), indent=2)


def load(directory):
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = RelationClassifier(AutoModel.from_pretrained(directory), len(LABELS))
    model.head.load_state_dict(torch.load(os.path.join(directory, "head.pt"), map_location="cpu"))
    config = json.load(open(os.path.join(directory, "re_config.json")))
    return model.eval(), tokenizer, config


@torch.no_grad()
def predict_proba(model, tokenizer, texts, max_len=128, batch=32):
    """Class probabilities (NONE first) for marked sentences."""
    out = []
    for i in range(0, len(texts), batch):
        enc = tokenizer(texts[i:i + batch], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        s_pos, o_pos = marker_positions(tokenizer, enc["input_ids"].tolist())
        _, logits = model(enc["input_ids"], enc["attention_mask"], torch.tensor(s_pos), torch.tensor(o_pos))
        out.append(torch.softmax(logits, dim=-1))
    return torch.cat(out) if out else torch.zeros((0, len(LABELS)))
