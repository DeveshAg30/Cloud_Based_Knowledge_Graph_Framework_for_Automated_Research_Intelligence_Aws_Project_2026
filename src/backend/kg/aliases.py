"""Entity name canonicalisation, so "GCN", "GCNs" and "graph convolutional networks" become one node.

Three steps, applied to every extractor's output:
  1. abbreviations a paper defines itself - "graph convolutional networks (GCNs)" - are learned per document;
  2. plurals of multi-word / acronym terms are singularised ("networks" -> "network", "LLMs" -> "LLM");
  3. a curated alias table maps well-known variants to one canonical display name.
"""

import re

from .schema import DATASET, METHOD, METRIC, TASK, normalize_name

# canonical display name -> variants (compared after lower-casing and hyphen -> space)
ALIAS_TABLE = {
    METHOD: {
        "Graph Convolutional Network": ["gcn", "graph convolution network", "graph convolutional net"],
        "Graph Neural Network": ["gnn", "graph neural net"],
        "Graph Attention Network": ["gat"],
        "Relational Graph Convolutional Network": ["r gcn", "rgcn"],
        "Retrieval-Augmented Generation": ["rag", "retrieval augmented generation", "retrieval augmented generator"],
        "GraphRAG": ["graph rag", "graph based rag", "graph retrieval augmented generation"],
        "Large Language Model": ["llm", "large language model", "large scale language model"],
        "Convolutional Neural Network": ["cnn", "convnet", "convolutional network"],
        "Recurrent Neural Network": ["rnn"],
        "Long Short-Term Memory": ["lstm", "long short term memory"],
        "Bidirectional LSTM": ["bilstm", "bi lstm", "bidirectional long short term memory"],
        "Knowledge Graph Embedding": ["kge", "kg embedding", "knowledge graph embedding model"],
        "Generative Adversarial Network": ["gan"],
        "Support Vector Machine": ["svm"],
        "Transformer": ["transformer model", "transformer architecture"],
        "BERT": ["bert model"],
        "Reinforcement Learning": ["rl", "deep reinforcement learning", "drl"],
        "Transfer Learning": [],
        "Meta-Learning": ["meta learning"],
        "Personalized PageRank": ["ppr", "personalised pagerank"],
        "Word2Vec": ["word2vec", "word 2 vec"],
        "TransE": ["trans e"],
    },
    TASK: {
        "Named Entity Recognition": ["ner"],
        "Question Answering": ["qa"],
        "Knowledge Graph Completion": ["kgc", "kg completion"],
        "Knowledge Graph Construction": ["kg construction"],
        "Link Prediction": [],
        "Relation Extraction": ["relation extraction task"],
        "Entity Linking": ["el"],
        "Recommendation": ["recommender system", "recommendation system"],
    },
    METRIC: {
        "F1": ["f1 score", "f1score", "f score", "f measure", "f1 measure"],
        "MRR": ["mean reciprocal rank"],
        "MAP": ["mean average precision"],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "AUC": ["area under the curve", "roc auc", "auroc"],
        "NDCG": ["normalized discounted cumulative gain"],
    },
    DATASET: {
        "FB15k-237": ["fb15k 237", "fb15k237"],
        "WN18RR": ["wn18 rr"],
        "FB15k": [],
        "WN18": [],
    },
}


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_name(text).replace("-", " ")).strip()


# Variants that mean the same thing whatever type the extractor assigned.
ANY_LABEL_ALIASES = {
    "Knowledge Graph": ["kg", "kgs", "knowledge graph", "knowledge graphs"],
    "Natural Language Processing": ["nlp", "natural language processing"],
    "Machine Learning": ["ml", "machine learning"],
    "Deep Learning": ["dl", "deep learning"],
    "Artificial Intelligence": ["ai", "artificial intelligence"],
}

_LOOKUP = {}
for _label, _table in ALIAS_TABLE.items():
    for _canonical, _variants in _table.items():
        for _v in [_canonical] + _variants:
            _LOOKUP[(_label, _key(_v))] = _canonical
for _canonical, _variants in ANY_LABEL_ALIASES.items():
    for _label in (METHOD, TASK, DATASET, METRIC):
        for _v in _variants:
            _LOOKUP.setdefault((_label, _key(_v)), _canonical)

_KEEP_S = ("ss", "is", "us", "ics", "ness")  # analysis, corpus, metrics-like words, robustness


def singular(key: str) -> str:
    """'graph neural networks' -> 'graph neural network', 'llms' -> 'llm'; leaves 'analysis', 'precision' alone."""
    words = key.split()
    last = words[-1]
    if len(last) > 3 and last.endswith("s") and not last.endswith(_KEEP_S) and (len(words) > 1 or last[:-1].isalpha()):
        if last.endswith("ies") and len(last) > 4:
            words[-1] = last[:-3] + "y"
        else:
            words[-1] = last[:-1]
    return " ".join(words)


ABBREV_PATTERN = re.compile(r"([A-Za-z][A-Za-z\- ]{3,80}?)\s*\(\s*([A-Z][A-Za-z0-9\-]{1,10})\s*\)")


def learn_abbreviations(text: str) -> dict:
    """{'gcn': 'graph convolutional network'} from definitions like 'graph convolutional networks (GCNs)'."""
    found = {}
    for long_form, abbr in ABBREV_PATTERN.findall(text or ""):
        core = abbr[:-1] if abbr.endswith("s") and len(abbr) > 2 and abbr[-2].isupper() else abbr
        letters = re.sub(r"[^a-z]", "", core.lower())
        if not 2 <= len(letters) <= 8:
            continue
        words = re.findall(r"[A-Za-z]+", long_form)
        candidate = words[-len(letters):]
        if len(candidate) == len(letters) and "".join(w[0] for w in candidate).lower() == letters:
            found[_key(core)] = singular(_key(" ".join(candidate)))
    return found


_TRAILING_ABBR = re.compile(r"^(.{4,}?)\s*\(\s*([A-Za-z][\w-]{1,10})\s*\)$")


def canonical(label: str, text: str, abbreviations: dict | None = None) -> str:
    """Canonical display name for an entity mention."""
    abbreviations = abbreviations or {}
    m = _TRAILING_ABBR.match(text.strip())
    if m:  # "graph convolutional network (GCN)" -> "graph convolutional network"
        text = m.group(1)
    key = _key(text)
    expanded = abbreviations.get(key) or abbreviations.get(singular(key))
    if expanded:
        key = expanded
    for candidate in (key, singular(key)):
        if (label, candidate) in _LOOKUP:
            return _LOOKUP[(label, candidate)]
    if expanded:  # abbreviation defined in the paper but not in the table
        return singular(key).title()
    original = text.strip()
    if singular(key) != key:  # plural: drop it but keep the author's capitalisation
        return original[:-3] + ("Y" if original[-3:].isupper() else "y") if original.lower().endswith("ies") else original[:-1]
    return original


def canonicalize(extraction: dict, text: str) -> dict:
    abbreviations = learn_abbreviations(text)
    for e in extraction.get("entities", []):
        e["text"] = canonical(e["label"], e["text"], abbreviations)
    for r in extraction.get("relations", []):
        r["head"] = canonical(r["head_label"], r["head"], abbreviations)
        r["tail"] = canonical(r["tail_label"], r["tail"], abbreviations)
    return extraction
