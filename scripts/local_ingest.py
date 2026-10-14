"""Run the same ingest pipeline as the Lambda, from your machine, against any graph backend.

  # local Neo4j (docker compose up -d), rule-based extraction, no AWS needed (PowerShell):
  $env:GRAPH_BACKEND="neo4j"; python scripts/local_ingest.py dataset/sample_papers.jsonl --mode rules

  # straight into Neptune Analytics (uses your AWS CLI credentials):
  $env:GRAPH_BACKEND="neptune-analytics"; $env:NEPTUNE_GRAPH_ID="g-xxxx"; python scripts/local_ingest.py data/openalex
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "backend"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # non-Latin titles on Windows consoles
from kg.graph_client import get_graph_client  # noqa: E402
from kg.parsers import from_pdf_bytes, parse_json_payload  # noqa: E402
from kg.pipeline import process_document  # noqa: E402


def iter_files(paths):
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                yield os.path.join(p, name)
        else:
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="PDF / JSON / JSONL files or directories")
    ap.add_argument("--mode", choices=["rules", "bedrock", "sagemaker"], default=None,
                    help="extractor (default: EXTRACTOR_MODE env or rules)")
    args = ap.parse_args()

    graph = get_graph_client()
    total = 0
    for path in iter_files(args.paths):
        if path.lower().endswith(".pdf"):
            with open(path, "rb") as f:
                docs = [from_pdf_bytes(f.read(), os.path.basename(path))]
        elif path.lower().endswith((".json", ".jsonl")):
            with open(path, encoding="utf-8") as f:
                docs = parse_json_payload(f.read())
        else:
            continue
        for doc in docs:
            _, stats = process_document(doc, graph, args.mode)
            total += 1
            print(f"[{total}] {doc['title'][:70]!r}: {stats['entities']} entities, {stats['edges']} edges")
    print(f"Done: {total} documents")


if __name__ == "__main__":
    main()
