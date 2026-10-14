"""Scientific entity + relation extraction.

Three interchangeable back ends, selected by the EXTRACTOR_MODE env var:
  * ``sagemaker`` - fine-tuned SciBERT NER endpoint (see sagemaker/), the main path in the proposal
  * ``llm``       - zero-shot extraction with the configured LLM_PROVIDER (``bedrock`` is accepted as an alias)
  * ``rules``     - dictionary + regex baseline, no cloud dependency (local dev and unit tests)

All return: {"entities": [{"text", "label", "sentence"}],
             "relations": [{"head", "head_label", "rel", "tail", "tail_label", "evidence"}]}
"""

import json
import os
import random
import re
import time

from .aliases import canonicalize
from .schema import DATASET, METHOD, METRIC, TASK, normalize_name

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])|\n+")  # sentences, plus line breaks (titles, headings)

# Relations inferred from entities that co-occur in one sentence.
COOCCURRENCE_RELS = {
    (METHOD, TASK): "APPLIED_TO",
    (METHOD, DATASET): "EVALUATED_ON",
    (METHOD, METRIC): "MEASURED_BY",
}

# SciERC label set -> our ontology. "Material" in SciERC is mostly datasets/corpora.
SCIERC_LABEL_MAP = {"Method": METHOD, "Task": TASK, "Metric": METRIC, "Material": DATASET}


def split_sentences(text: str):
    return [s.strip() for s in SENTENCE_SPLIT.split(text or "") if len(s.strip()) > 20]


NOT_ENTITIES = {"the", "our", "this", "that", "these", "we", "their", "its", "a", "an", "proposed", "each",
                 "such", "new", "novel", "existing", "baseline", "previous", "same", "other", "first", "second"}
# Lone adjectives the "<Word> network/model" pattern picks out of longer names. Rejected only as a whole
# name: "Graph Convolutional Network" is fine, a bare "Convolutional" is not.
LONE_ADJECTIVES = {"convolutional", "neural", "deep", "graph", "attention", "temporal", "spatial", "recurrent",
                   "generative", "adversarial", "residual", "hierarchical", "dynamic", "multi", "hybrid", "end",
                   "language", "learning", "search", "simple", "standard", "general", "unified", "efficient"}
_ADJECTIVE_SUFFIX = re.compile(r"-(driven|based|powered|aware|enhanced|guided|augmented|specific|oriented|agent)$", re.I)


# Phrases the NER model tags but that carry no information as graph nodes (seen on real OpenAlex data).
FUNCTION_WORDS = {"the", "a", "an", "of", "and", "or", "for", "in", "on", "with", "to", "by", "as", "at", "from", "it", "they", "them",
                  "which", "its", "this", "these", "those", "such", "both", "all", "each"}
GENERIC_TERMS = {
    "dataset", "datasets", "data", "data set", "data sets", "text", "texts", "image", "images", "corpus", "corpora",
    "benchmark", "benchmarks", "benchmark dataset", "benchmark datasets", "real world dataset", "real world datasets",
    "real-world dataset", "real-world datasets", "public dataset", "public datasets", "large-scale dataset",
    "experimental results", "results", "experiments", "performance", "model", "models", "method", "methods",
    "approach", "approaches", "framework", "frameworks", "system", "systems", "technique", "techniques",
    "algorithm", "algorithms", "task", "tasks", "problem", "problems", "information", "knowledge", "features",
    "graph data", "network", "networks", "baseline", "baselines", "state of the art", "state-of-the-art",
}


def plausible(text: str) -> bool:
    """Filters the junk real PDFs produce: pronouns/determiners, compound adjectives, and
    run-together words where the PDF text layer lost its spaces."""
    name = normalize_name(text)
    if len(name) < 2 or len(name) > 80 or name.isdigit() or name in NOT_ENTITIES or name in LONE_ADJECTIVES \
            or name in GENERIC_TERMS or all(w in FUNCTION_WORDS for w in name.split()):
        return False
    if name.split()[0] in NOT_ENTITIES or _ADJECTIVE_SUFFIX.search(name):
        return False
    return not any(len(tok) > 30 for tok in name.split())


_TRAILING_JUNK = re.compile(r"(?:[\s,;:(\[{/-]+|\s+(?:of|versus|vs|and|or|for|in|on|with|to|the|a|an|by|from))+$", re.I)
LABEL_PRIORITY = [METHOD, TASK, DATASET, METRIC]


def tidy(text: str) -> str:
    """'machine reading of' -> 'machine reading', 'Text Classification (' -> 'Text Classification',
    'infrared spectroscopy ( FTIR )' -> 'infrared spectroscopy (FTIR)', 'calorimetry ( DSC' -> 'calorimetry'."""
    text = re.sub(r"([(\[])\s+", r"\1", text.strip())          # tokenizer spacing inside brackets
    text = re.sub(r"\s+([)\],.;:!?])", r"\1", text)
    if text.count("(") > text.count(")"):                      # unbalanced "( DSC" -> drop the dangling part
        text = text[:text.rfind("(")]
    text = _DASHES.sub("-", text)                              # Unicode hyphens/dashes -> "-"
    text = re.sub(r"(?<=\w) - (?=\w)", "-", text)              # "Multi - Agent" (tokenizer spacing) -> "Multi-Agent"
    text = _LEADING_JUNK.sub("", text)                         # "of protein interaction" -> "protein interaction"
    return _TRAILING_JUNK.sub("", text).strip()


_DASHES = re.compile("[‐‑‒–−]")
_LEADING_JUNK = re.compile(r"^(?:(?:of|for|in|on|with|to|by|from|and|or|the|a|an|its|their|our|this|these)\s+)+", re.I)


def dedupe_entities(entities):
    """Clean names, drop junk, and give each name one type: the label it was assigned most often
    (ties broken Method > Task > Dataset > Metric), so 'SciBERT' isn't both a Method and a Dataset."""
    for e in entities:
        e["text"] = tidy(e["text"])
    votes = {}
    for e in entities:
        votes.setdefault(normalize_name(e["text"]), []).append(e["label"])
    best = {name: max(set(labels), key=lambda l: (labels.count(l), -LABEL_PRIORITY.index(l) if l in LABEL_PRIORITY else -9))
            for name, labels in votes.items()}
    seen, out = set(), []
    for e in entities:
        name = normalize_name(e["text"])
        if not plausible(name) or best.get(name) != e["label"]:
            continue
        key = (e["label"], name)
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def cooccurrence_relations(entities):
    by_sentence = {}
    for e in entities:
        by_sentence.setdefault(e.get("sentence", ""), []).append(e)
    rels, seen = [], set()
    for sentence, ents in by_sentence.items():
        for h in ents:
            for t in ents:
                rel = COOCCURRENCE_RELS.get((h["label"], t["label"]))
                if not rel:
                    continue
                key = (normalize_name(h["text"]), rel, normalize_name(t["text"]))
                if key in seen:
                    continue
                seen.add(key)
                rels.append({"head": h["text"], "head_label": h["label"], "rel": rel,
                             "tail": t["text"], "tail_label": t["label"], "evidence": sentence[:500]})
    return rels


# --------------------------------------------------------------------------- rules baseline
GAZETTEER = {
    METHOD: ["transformer", "bert", "scibert", "roberta", "gpt", "lstm", "bilstm", "cnn", "rnn", "gnn",
             "graph neural network", "graph convolutional network", "gcn", "graphsage", "transe", "rotate",
             "retrieval-augmented generation", "rag", "graphrag", "knowledge graph embedding", "svm",
             "random forest", "xgboost", "k-means", "logistic regression", "diffusion model", "gan",
             "meta-learning", "transfer learning", "reinforcement learning", "word2vec", "cbow", "pagerank"],
    DATASET: ["fb15k-237", "fb15k", "wn18rr", "wn18", "nell-995", "wikidata", "imagenet", "squad", "mimic-iii",
              "mimic-iv", "s2orc", "openalex", "pubmedqa", "hotpotqa", "natural questions", "ms marco", "scierc",
              "icews", "dbp15k", "conll-2003"],
    TASK: ["named entity recognition", "relation extraction", "link prediction", "knowledge graph completion",
           "question answering", "entity linking", "text classification", "summarization",
           "machine translation", "entity alignment", "information retrieval", "recommendation",
           "sentiment analysis", "knowledge graph construction", "literature review"],
    METRIC: ["f1", "f1-score", "accuracy", "precision", "recall", "mrr", "hits@1", "hits@3", "hits@10",
             "bleu", "rouge", "auc", "mean average precision", "ndcg", "perplexity", "exact match"],
}
_GAZ_PATTERNS = {
    label: re.compile(r"(?<![\w-])(" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")(?![\w-])", re.I)
    for label, terms in GAZETTEER.items()
}
# "the FooBar dataset", "XYZ-Net model"
_DATASET_PATTERN = re.compile(r"\b([A-Z][\w-]{2,}(?:\s[A-Z][\w-]+)?)\s+(?:dataset|corpus|benchmark)\b")
_METHOD_PATTERN = re.compile(r"\b([A-Z][A-Za-z0-9-]{2,})\s+(?:model|framework|architecture|algorithm|network)\b")


def extract_rules(text: str):
    entities = []
    for sentence in split_sentences(text):
        for label, pattern in _GAZ_PATTERNS.items():
            for m in pattern.finditer(sentence):
                entities.append({"text": m.group(1), "label": label, "sentence": sentence})
        for m in _DATASET_PATTERN.finditer(sentence):
            entities.append({"text": m.group(1), "label": DATASET, "sentence": sentence})
        for m in _METHOD_PATTERN.finditer(sentence):
            entities.append({"text": m.group(1), "label": METHOD, "sentence": sentence})
    entities = dedupe_entities(entities)
    return {"entities": entities, "relations": cooccurrence_relations(entities)}


# --------------------------------------------------------------------------- SageMaker endpoint
_sm_runtime = None


def extract_sagemaker(text: str, endpoint_name: str | None = None):
    """Calls the SciBERT endpoint (src/ai_model/inference/inference.py): NER entities plus, when the RE model is
    deployed, typed relations. Older NER-only endpoints fall back to same-sentence co-occurrence relations."""
    global _sm_runtime
    import boto3

    if _sm_runtime is None:
        from botocore.config import Config

        # Serverless endpoints cold-start (model load) and allow only a few concurrent calls, so parallel
        # ingest Lambdas get throttled: retry adaptively and allow a long read for the first call.
        _sm_runtime = boto3.client("sagemaker-runtime", config=Config(
            read_timeout=90, connect_timeout=10, retries={"max_attempts": 8, "mode": "adaptive"}))
    endpoint_name = endpoint_name or os.environ["SAGEMAKER_ENDPOINT"]
    entities, model_relations, has_re = [], [], False
    # Serverless endpoints have a 60 s limit, so send the document in chunks of sentences.
    chunk, chunks = [], []
    for s in split_sentences(text):
        chunk.append(s)
        if sum(len(c) for c in chunk) > 4000:
            chunks.append(chunk)
            chunk = []
    if chunk:
        chunks.append(chunk)
    for sentences in chunks:
        out = json.loads(_invoke_with_backoff(endpoint_name, json.dumps({"sentences": sentences})))
        for e in out["entities"]:
            label = SCIERC_LABEL_MAP.get(e["label"])
            if label:
                entities.append({"text": e["text"], "label": label, "sentence": e["sentence"]})
        if "relations" in out:
            has_re = True
            model_relations += [r for r in (map_scierc_relation(x) for x in out["relations"]) if r]
    entities = dedupe_entities(entities)
    relations = model_relations if has_re else cooccurrence_relations(entities)
    return {"entities": entities, "relations": relations, "relation_source": "model" if has_re else "co-occurrence"}


def _invoke_with_backoff(endpoint_name, body, attempts=8):
    """Serverless endpoints cap concurrent requests; when many ingest Lambdas run at once they get
    ThrottlingException. Botocore already retries quickly; on top of that wait with exponential backoff and
    jitter (up to about 2 minutes in total) so a busy endpoint delays ingestion instead of failing it."""
    from botocore.exceptions import ClientError

    for attempt in range(attempts):
        try:
            resp = _sm_runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType="application/json", Body=body)
            return resp["Body"].read()
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") != "ThrottlingException" or attempt == attempts - 1:
                raise
            time.sleep(min(30.0, 2 ** attempt) + random.uniform(0, 2))


# SciERC relation (head type, relation, tail type) -> our ontology (head, REL, tail). The model's head/tail order is kept
# unless the ontology reads the other way round (e.g. "Metric EVALUATE-FOR Method" becomes "Method MEASURED_BY Metric").
SCIERC_RELATION_MAP = {
    (METHOD, "USED-FOR", TASK): ("head", "APPLIED_TO", "tail"),
    (DATASET, "USED-FOR", METHOD): ("tail", "EVALUATED_ON", "head"),
    (DATASET, "USED-FOR", TASK): ("tail", "EVALUATED_ON", "head"),
    (METRIC, "EVALUATE-FOR", METHOD): ("tail", "MEASURED_BY", "head"),
    (METRIC, "EVALUATE-FOR", TASK): ("tail", "MEASURED_BY", "head"),
    (DATASET, "EVALUATE-FOR", METHOD): ("tail", "EVALUATED_ON", "head"),
    (DATASET, "EVALUATE-FOR", TASK): ("tail", "EVALUATED_ON", "head"),
    (METHOD, "COMPARE", METHOD): ("head", "COMPARED_WITH", "tail"),
    (METHOD, "HYPONYM-OF", METHOD): ("head", "IS_A", "tail"),
    (TASK, "HYPONYM-OF", TASK): ("head", "IS_A", "tail"),
    (METHOD, "PART-OF", METHOD): ("head", "PART_OF", "tail"),
}


def map_scierc_relation(r):
    head_label, tail_label = SCIERC_LABEL_MAP.get(r["head_label"]), SCIERC_LABEL_MAP.get(r["tail_label"])
    rule = SCIERC_RELATION_MAP.get((head_label, r["rel"], tail_label))
    if not rule:
        return None
    first, rel, second = rule
    side = {"head": (r["head"], head_label), "tail": (r["tail"], tail_label)}
    (h, hl), (t, tl) = side[first], side[second]
    return {"head": h, "head_label": hl, "rel": rel, "tail": t, "tail_label": tl,
            "evidence": r.get("sentence", "")[:500], "score": r.get("score")}


# --------------------------------------------------------------------------- LLM (Gemini, Nova or Claude)
EXTRACTION_PROMPT = """Extract scientific entities from the paper text below.

Return only a JSON object, no prose, with this shape:
{"entities": [{"text": "...", "label": "Method|Dataset|Task|Metric", "sentence": "<the sentence it appears in>"}],
 "relations": [{"head": "<Method name>", "rel": "APPLIED_TO|EVALUATED_ON|MEASURED_BY", "tail": "<Task|Dataset|Metric name>", "evidence": "<sentence>"}]}

Rules:
- Method = algorithms, models, architectures, techniques. Dataset = named datasets, corpora, benchmarks.
  Task = research problems being solved. Metric = evaluation measures.
- Use the shortest canonical surface form (e.g. "BERT", not "the BERT-based model we propose").
- Only include relations the text states; do not infer from background knowledge.
- At most 40 entities and 40 relations.

<paper>
{text}
</paper>"""


def _first_json_object(s: str):
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("model did not return JSON")
    return json.loads(s[start:end + 1])


def extract_llm(text: str):
    from .llm import complete

    raw = complete(EXTRACTION_PROMPT.replace("{text}", text[:30000]), max_tokens=8000, effort="low")
    data = _first_json_object(raw)
    labels = {METHOD, DATASET, TASK, METRIC}
    entities = dedupe_entities([e for e in data.get("entities", []) if e.get("label") in labels and e.get("text")])
    label_of = {normalize_name(e["text"]): e["label"] for e in entities}
    relations = []
    for r in data.get("relations", []):
        h, t = label_of.get(normalize_name(r.get("head", ""))), label_of.get(normalize_name(r.get("tail", "")))
        if h and t and COOCCURRENCE_RELS.get((h, t)) == r.get("rel"):
            relations.append({**r, "head_label": h, "tail_label": t, "evidence": (r.get("evidence") or "")[:500]})
    return {"entities": entities, "relations": relations}


def extract(text: str, mode: str | None = None):
    mode = (mode or os.environ.get("EXTRACTOR_MODE", "rules")).lower()
    if mode == "sagemaker":
        # Hybrid: the fine-tuned model finds new/unseen entities; the curated dictionary guarantees
        # well-known names (BERT, RoBERTa, ...) that a 0.6-F1 model sometimes misses.
        result = extract_sagemaker(text)
        known = extract_rules(text)
        result["entities"] += known["entities"]
        if result.get("relation_source") != "model":  # typed RE relations beat co-occurrence guesses
            result["relations"] += known["relations"]
    elif mode in ("llm", "bedrock"):  # uses whichever LLM_PROVIDER is configured
        from .llm import LLMDisabled

        try:
            result = extract_llm(text)
        except LLMDisabled:
            result = extract_rules(text)
    else:
        result = extract_rules(text)
    # Merge "GCN" / "GCNs" / "graph convolutional networks" into one canonical entity.
    result = canonicalize(result, text)
    result["entities"] = dedupe_entities(result["entities"])
    result["relations"] = resolve_directions(result["relations"])
    attach_metric_values(result["entities"])
    return result


SYMMETRIC_RELS = {"COMPARED_WITH"}


def resolve_directions(relations):
    """One edge per entity pair and relation. The RE model often predicts both A->B and B->A; for one-way relations
    (IS_A, APPLIED_TO, PART_OF, ...) keep the more confident direction, and store symmetric ones in a canonical
    (alphabetical) direction."""
    best = {}
    for r in relations:
        h, t = normalize_name(r["head"]), normalize_name(r["tail"])
        if not h or h == t:
            continue
        if r["rel"] in SYMMETRIC_RELS and h > t:
            r = {**r, "head": r["tail"], "head_label": r["tail_label"], "tail": r["head"], "tail_label": r["head_label"]}
            h, t = t, h
        key = (r["rel"], frozenset((h, t)))
        if key not in best or (r.get("score") or 0) > (best[key].get("score") or 0):
            best[key] = r
    return list(best.values())


# A reported result next to a metric: "96.7%", "0.489", "91.4 F1", "an MRR of .52".
_VALUE = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d+)?\s?%|\d?\.\d{2,4}|\d{1,3}\.\d{1,2})(?![\w.%])")


def metric_value(metric: str, sentence: str, window: int = 60):
    """The numeric value closest to a metric mention in its sentence (within `window` characters), if any."""
    pos = sentence.lower().find(metric.lower())
    if pos < 0:
        return None
    best, best_dist = None, window + 1
    for m in _VALUE.finditer(sentence):
        dist = min(abs(m.start() - (pos + len(metric))), abs(pos - m.end()))
        if dist < best_dist and not re.match(r"(19|20)\d\d$", m.group(1)):
            best, best_dist = m.group(1).replace(" ", ""), dist
    return best


def attach_metric_values(entities):
    """Outcomes: give Metric entities the value the paper reports for them (stored on the REPORTS_METRIC edge)."""
    for e in entities:
        if e["label"] == METRIC and "value" not in e:
            value = metric_value(e["text"], e.get("sentence", ""))
            if value:
                e["value"] = value
