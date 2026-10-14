"""S3 ObjectCreated (raw/) -> parse -> NLP extraction -> graph upsert -> processed/<doc_id>.json

Dashboard uploads (raw/uploads/...) also get a small status/<file>.json so the dashboard can show
live progress and what the extractor found.
"""

import gzip
import json
import os
import time
import traceback
import urllib.parse

import boto3

from kg.enrich import enrich_pdf_document
from kg.graph_client import get_graph_client
from kg.metrics import emit
from kg.parsers import from_pdf_bytes, parse_json_payload
from kg.pipeline import process_document

s3 = boto3.client("s3")
PROCESSED_PREFIX = os.environ.get("PROCESSED_PREFIX", "processed/")
UPLOAD_PREFIX = "raw/uploads/"
EXTRACTOR_NAMES = {"sagemaker": "SciBERT on Amazon SageMaker + term dictionary", "llm": "LLM (Gemini)", "bedrock": "LLM",
                   "rules": "rule-based baseline"}


def status_key(upload_key: str) -> str:
    return "status/" + upload_key.rsplit("/", 1)[-1] + ".json"


def write_status(bucket, key, payload):
    if key.startswith(UPLOAD_PREFIX):
        s3.put_object(Bucket=bucket, Key=status_key(key), Body=json.dumps(payload).encode(),
                      ContentType="application/json")


def load_documents(bucket: str, key: str):
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    lower = key.lower()
    if lower.endswith(".pdf"):
        return [enrich_pdf_document(from_pdf_bytes(body, key.rsplit("/", 1)[-1]))]
    if lower.endswith(".gz"):
        body = gzip.decompress(body)
    return parse_json_payload(body.decode("utf-8"))


def handler(event, context):
    graph = get_graph_client()
    mode = os.environ.get("EXTRACTOR_MODE", "rules")
    results = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        started = time.time()
        try:
            docs = load_documents(bucket, key)
        except Exception as e:
            traceback.print_exc()
            emit({"ParseFailures": 1}, Stage="ingest")
            write_status(bucket, key, {"status": "failed", "error": f"Could not read the file ({type(e).__name__})."})
            raise
        for doc in docs:
            try:
                extraction, stats = process_document(doc, graph)
            except Exception as e:
                print(f"Failed doc {doc.get('doc_id')} from s3://{bucket}/{key}")
                traceback.print_exc()
                emit({"DocumentFailures": 1}, Stage="ingest")
                write_status(bucket, key, {"status": "failed", "error": f"Processing failed ({type(e).__name__})."})
                raise
            s3.put_object(
                Bucket=bucket,
                Key=f"{PROCESSED_PREFIX}{doc['doc_id']}.json",
                Body=json.dumps({"source_key": key, "document": doc, "extraction": extraction, "stats": stats}).encode(),
                ContentType="application/json",
            )
            write_status(bucket, key, {
                "status": "done",
                "title": doc.get("title"),
                "authors": [a["name"] for a in doc.get("authors", [])][:8],
                "extractor": EXTRACTOR_NAMES.get(mode, mode),
                "entities": [{"text": e["text"], "label": e["label"], **({"value": e["value"]} if e.get("value") else {})}
                             for e in extraction["entities"][:60]],
                "relations": [{"head": r["head"], "rel": r["rel"], "tail": r["tail"]} for r in extraction["relations"][:20]],
                "edges": stats["edges"],
                "seconds": round(time.time() - started, 1),
            })
            emit({"DocumentsProcessed": 1, "EntitiesExtracted": stats["entities"], "EdgesWritten": stats["edges"]},
                 Stage="ingest")
            emit({"DocumentLatency": stats["seconds"]}, unit="Seconds", Stage="ingest")
            print(json.dumps({"ingested": stats}))
            results.append(stats)
    return {"processed": len(results), "documents": results}
