"""Canonical document format and graph ontology.

Every source (PDF, S2ORC, OpenAlex) is converted into a ``Document`` dict with this shape
before extraction, so the rest of the pipeline only deals with one format:

{
  "doc_id": "W2741809807",          # stable id (OpenAlex id, S2ORC paper_id, or sha1 of a PDF)
  "source": "openalex" | "s2orc" | "pdf",
  "title": str, "year": int | None, "venue": str | None, "abstract": str,
  "authors":    [{"id": str | None, "name": str, "institutions": [{"id": str | None, "name": str}]}],
  "references": [{"id": str | None, "title": str}],
  "concepts":   [str],
  "sections":   [{"heading": str, "text": str, "citations": [{"ref_id": str | None, "title": str}]}]
}
"""

import hashlib
import re

# Node labels in the knowledge graph.
PAPER = "Paper"
AUTHOR = "Author"
INSTITUTION = "Institution"
VENUE = "Venue"
CONCEPT = "Concept"
METHOD = "Method"
DATASET = "Dataset"
TASK = "Task"
METRIC = "Metric"

ENTITY_LABELS = {METHOD, DATASET, TASK, METRIC}
ALL_LABELS = {PAPER, AUTHOR, INSTITUTION, VENUE, CONCEPT} | ENTITY_LABELS

# Relationship types: (from_label, REL, to_label)
ONTOLOGY = [
    (AUTHOR, "AUTHORED", PAPER),
    (AUTHOR, "AFFILIATED_WITH", INSTITUTION),
    (PAPER, "PUBLISHED_IN", VENUE),
    (PAPER, "CITES", PAPER),
    (PAPER, "HAS_CONCEPT", CONCEPT),
    (PAPER, "USES_METHOD", METHOD),
    (PAPER, "USES_DATASET", DATASET),
    (PAPER, "ADDRESSES_TASK", TASK),
    (PAPER, "REPORTS_METRIC", METRIC),
    (METHOD, "APPLIED_TO", TASK),
    (METHOD, "EVALUATED_ON", DATASET),
    (METHOD, "MEASURED_BY", METRIC),
]
ALL_RELS = {r for _, r, _ in ONTOLOGY}

# Which paper->entity relation to create for each extracted entity label.
PAPER_REL_FOR_LABEL = {
    METHOD: "USES_METHOD",
    DATASET: "USES_DATASET",
    TASK: "ADDRESSES_TASK",
    METRIC: "REPORTS_METRIC",
}


def normalize_name(name: str) -> str:
    """Lower-case, collapse whitespace and strip punctuation at the ends."""
    name = re.sub(r"\s+", " ", (name or "")).strip().strip(".,;:()[]{}\"'")
    return name.lower()


def make_uid(label: str, key: str) -> str:
    """Deterministic node id so the same entity from two papers merges into one node."""
    norm = normalize_name(key)
    digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]
    return f"{label.lower()}:{digest}"


def paper_uid(doc_id: str) -> str:
    return f"paper:{doc_id}"


def empty_document(doc_id: str, source: str) -> dict:
    return {
        "doc_id": doc_id,
        "source": source,
        "title": "",
        "year": None,
        "venue": None,
        "abstract": "",
        "authors": [],
        "references": [],
        "concepts": [],
        "sections": [],
    }


def document_text(doc: dict, max_chars: int = 60000) -> str:
    """Title + abstract + body, truncated to keep extraction cost bounded."""
    parts = [doc.get("title") or "", doc.get("abstract") or ""]
    for sec in doc.get("sections", []):
        parts.append(sec.get("text") or "")
    text = "\n\n".join(p for p in parts if p)
    return text[:max_chars]
