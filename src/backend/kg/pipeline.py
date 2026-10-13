"""End-to-end processing of one canonical document: extract -> build graph -> upsert.
Shared by the ingest Lambda and scripts/local_ingest.py."""

import time

from .extractors import extract
from .graph_writer import build_graph, upsert
from .schema import document_text


def process_document(doc: dict, graph, mode: str | None = None):
    started = time.time()
    extraction = extract(document_text(doc), mode)
    nodes, stubs, edges = build_graph(doc, extraction)
    stats = upsert(graph, nodes, stubs, edges)
    stats.update({"doc_id": doc["doc_id"], "entities": len(extraction["entities"]),
                  "relations": len(extraction["relations"]), "seconds": round(time.time() - started, 2)})
    return extraction, stats
