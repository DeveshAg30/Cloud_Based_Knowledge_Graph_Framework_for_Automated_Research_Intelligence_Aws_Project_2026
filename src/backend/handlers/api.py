"""HTTP API (behind API Gateway + Cognito JWT authorizer).

POST /query       {"question": "..."}         -> GraphRAG answer + citations + subgraph
GET  /graph?name= ...                         -> neighbourhood subgraph for the visualiser
GET  /stats                                   -> node counts per label
GET  /connect?a=...&b=...                     -> shortest multi-hop paths between two entities
GET  /insights?name=...                       -> Research 360 profile of an entity
POST /upload-url  {"filename": "paper.pdf"}   -> presigned S3 PUT URL into raw/uploads/
"""

import json
import os
import re
import time
import traceback
import uuid

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from kg.graph_client import get_graph_client
from kg.llm import LLMRefusal
from kg.metrics import emit
from kg.insights import connect, profile
from kg.retrieval import answer_question, neighbourhood

# SigV4 + regional endpoint: the default legacy SigV2 presigned URL signs the Content-Type, so a
# browser PUT that sends "application/pdf" fails with SignatureDoesNotMatch.
s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"),
                  config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))


def _response(status, body):
    return {"statusCode": status, "headers": {"Content-Type": "application/json"}, "body": json.dumps(body, default=str)}


def _claims(event):
    return (((event.get("requestContext") or {}).get("authorizer") or {}).get("jwt") or {}).get("claims") or {}


def _user(event):
    claims = _claims(event)
    return claims.get("email") or claims.get("username") or claims.get("sub") or "anonymous"


def _is_member(event):
    """API Gateway passes cognito:groups as a string such as "[members]" or "[members admins]"."""
    groups = _claims(event).get("cognito:groups") or ""
    if isinstance(groups, list):
        return "members" in groups
    return "members" in re.findall(r"[\w-]+", groups)


def handle_query(event):
    body = json.loads(event.get("body") or "{}")
    question = (body.get("question") or "").strip()
    if not question or len(question) > 1000:
        return _response(400, {"error": "question must be 1-1000 characters"})
    started = time.time()
    try:
        result = answer_question(get_graph_client(), question)
    except LLMRefusal as e:
        return _response(422, {"error": str(e)})
    latency = time.time() - started
    emit({"QueryLatency": latency}, unit="Seconds", Stage="query")
    emit({"Queries": 1, "FactsRetrieved": len(result["facts"])}, Stage="query")
    print(json.dumps({"user": _user(event), "question": question, "seeds": [s["name"] for s in result["seeds"]],
                      "facts": len(result["facts"]), "latency": round(latency, 2)}))
    return _response(200, result)


def handle_graph(event):
    name = ((event.get("queryStringParameters") or {}).get("name") or "").strip()
    if not name:
        return _response(400, {"error": "name query parameter is required"})
    return _response(200, neighbourhood(get_graph_client(), name))


_stats_cache = {"at": 0.0, "counts": None}


def _counts():
    # Cached for 5 minutes: the public landing page calls this without a login.
    if _stats_cache["counts"] is None or time.time() - _stats_cache["at"] > 300:
        rows = get_graph_client().query("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count")
        _stats_cache.update(at=time.time(), counts={r["label"]: r["count"] for r in rows})
    return _stats_cache["counts"]


def handle_connect(event):
    """Multi-hop discovery: shortest chains linking two entities (GET /connect?a=...&b=...)."""
    q = event.get("queryStringParameters") or {}
    a, b = (q.get("a") or "").strip(), (q.get("b") or "").strip()
    if not a or not b or len(a) > 300 or len(b) > 300:
        return _response(400, {"error": "a and b query parameters are required"})
    return _response(200, connect(get_graph_client(), a, b))


def handle_insights(event):
    """Research 360 profile of an entity (GET /insights?name=...)."""
    name = ((event.get("queryStringParameters") or {}).get("name") or "").strip()
    if not name or len(name) > 300:
        return _response(400, {"error": "name query parameter is required"})
    return _response(200, profile(get_graph_client(), name))


def handle_stats(event):
    return _response(200, {"counts": _counts(), "member": _is_member(event)})


def handle_public_stats(event):
    counts = _counts()
    rels = {"papers": counts.get("Paper", 0), "authors": counts.get("Author", 0),
            "institutions": counts.get("Institution", 0),
            "entities": sum(counts.get(k, 0) for k in ("Method", "Dataset", "Task", "Metric", "Concept"))}
    return _response(200, rels)


def handle_upload_url(event):
    if not _is_member(event):
        return _response(403, {"error": "Uploading papers is limited to project members."})
    body = json.loads(event.get("body") or "{}")
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", body.get("filename") or "upload.pdf")[:120]
    if not filename.lower().endswith((".pdf", ".json", ".jsonl")):
        return _response(400, {"error": "only .pdf, .json and .jsonl uploads are accepted"})
    key = f"raw/uploads/{uuid.uuid4().hex[:8]}-{filename}"
    url = s3.generate_presigned_url("put_object", ExpiresIn=600,
                                    Params={"Bucket": os.environ["DATA_BUCKET"], "Key": key})
    return _response(200, {"url": url, "key": key})


def handle_upload_status(event):
    """Live progress for a dashboard upload: processing until the ingest Lambda writes status/<file>.json."""
    if not _is_member(event):
        return _response(403, {"error": "Uploading papers is limited to project members."})
    key = ((event.get("queryStringParameters") or {}).get("key") or "").strip()
    if not re.fullmatch(r"raw/uploads/[A-Za-z0-9._-]+", key):
        return _response(400, {"error": "invalid upload key"})
    try:
        obj = s3.get_object(Bucket=os.environ["DATA_BUCKET"], Key="status/" + key.rsplit("/", 1)[-1] + ".json")
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return _response(200, {"status": "processing"})
        raise
    return _response(200, json.loads(obj["Body"].read()))


ROUTES = {
    "POST /query": handle_query,
    "GET /graph": handle_graph,
    "GET /stats": handle_stats,
    "POST /upload-url": handle_upload_url,
    "GET /public/stats": handle_public_stats,
    "GET /upload-status": handle_upload_status,
    "GET /connect": handle_connect,
    "GET /insights": handle_insights,
}


def handler(event, context):
    route = ROUTES.get(event.get("routeKey"))
    if not route:
        return _response(404, {"error": f"no route {event.get('routeKey')}"})
    try:
        return route(event)
    except Exception as e:
        traceback.print_exc()
        emit({"ApiErrors": 1}, Stage="api")
        return _response(500, {"error": type(e).__name__})
