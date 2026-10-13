"""Stream real S2ORC full text from the Semantic Scholar Datasets API and keep papers on your topic.

S2ORC shards are 4-6 GB each (30 per release), so this script never stores them: it streams each
gzip shard, keeps records whose title or opening text mention one of your keywords, and stops at
--limit. Matching papers are enriched with year / venue / author ids / abstract from the Graph API.

  # 1. free API key: https://www.semanticscholar.org/product/api#api-key-form
  $env:S2_API_KEY="..."
  python scripts/fetch_s2orc.py --keywords "knowledge graph" "graph neural network" --limit 300
  aws s3 cp data/s2orc s3://<bucket>/raw/s2orc/ --recursive

Tip: the shards live in S3 us-west-2; running this on an EC2 instance there is much faster than from a laptop.
"""

import argparse
import gzip
import io
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "backend"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # non-Latin titles on Windows consoles
from kg.parsers import from_s2orc_current  # noqa: E402

DATASETS = "https://api.semanticscholar.org/datasets/v1"
GRAPH_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"


def get_json(url, api_key, data=None):
    headers = {"x-api-key": api_key}
    if data is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(data).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=120) as resp:
        return json.load(resp)


def shard_urls(api_key, dataset):
    release = get_json(f"{DATASETS}/release/latest", api_key)["release_id"]
    files = get_json(f"{DATASETS}/release/{release}/dataset/{dataset}", api_key)["files"]
    print(f"Release {release}: {len(files)} {dataset} shards")
    return files


def stream_records(url):
    with urllib.request.urlopen(url, timeout=300) as resp:
        for line in io.TextIOWrapper(gzip.GzipFile(fileobj=resp), encoding="utf-8"):
            if line.strip():
                yield json.loads(line)


def matches(doc, keywords):
    head = (doc["title"] + " " + doc["abstract"] + " " + " ".join(s["text"] for s in doc["sections"][:3])).lower()
    return any(k in head for k in keywords)


def enrich(docs, api_key):
    """Fill year, venue, abstract and author ids from the Graph API (500 ids per call)."""
    for i in range(0, len(docs), 500):
        chunk = docs[i:i + 500]
        ids = [f"CorpusId:{d['doc_id'].removeprefix('s2-')}" for d in chunk]
        rows = get_json(f"{GRAPH_BATCH}?fields=title,year,venue,abstract,authors", api_key, {"ids": ids})
        for doc, row in zip(chunk, rows):
            if not row:
                continue
            doc["title"] = doc["title"] or row.get("title") or ""
            doc["year"] = row.get("year")
            doc["venue"] = row.get("venue") or None
            doc["abstract"] = doc["abstract"] or row.get("abstract") or ""
            if row.get("authors"):
                doc["authors"] = [{"id": f"s2a-{a['authorId']}" if a.get("authorId") else None,
                                   "name": a["name"], "institutions": []} for a in row["authors"] if a.get("name")]
        time.sleep(1)  # Graph API: 1 request / second with a key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keywords", nargs="+", required=True, help="kept if title/opening text contains any")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max-shards", type=int, default=2, help="stop after streaming this many shards")
    ap.add_argument("--max-records", type=int, default=2_000_000, help="stop after scanning this many records")
    ap.add_argument("--dataset", default="s2orc_v2", choices=["s2orc_v2", "s2orc"])
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--api-key", default=os.environ.get("S2_API_KEY"))
    ap.add_argument("--out", default="data/s2orc")
    args = ap.parse_args()
    if not args.api_key:
        sys.exit("Set S2_API_KEY or pass --api-key (free key from semanticscholar.org/product/api)")

    keywords = [k.lower() for k in args.keywords]
    kept, scanned, failed = [], 0, 0
    for shard_no, url in enumerate(shard_urls(args.api_key, args.dataset)[:args.max_shards], 1):
        print(f"Streaming shard {shard_no} ...")
        for rec in stream_records(url):
            scanned += 1
            if scanned == 1:
                print(f"  first record keys: {sorted(rec)}")  # shows the schema if parsing ever breaks
            try:
                doc = from_s2orc_current(rec)
            except Exception:
                failed += 1
                continue
            if doc["sections"] and matches(doc, keywords):
                kept.append(doc)
                print(f"  [{len(kept)}] {doc['title'][:80]}")
            if scanned % 100_000 == 0:
                print(f"  scanned {scanned:,} records, kept {len(kept)}")
            if len(kept) >= args.limit or scanned >= args.max_records:
                break
        if len(kept) >= args.limit or scanned >= args.max_records:
            break

    print(f"Scanned {scanned:,} records ({failed} unparseable); enriching {len(kept)} matches")
    enrich(kept, args.api_key)
    os.makedirs(args.out, exist_ok=True)
    for n in range(0, len(kept), args.batch):
        with open(os.path.join(args.out, f"s2orc_{n // args.batch + 1:04d}.jsonl"), "w", encoding="utf-8") as f:
            for d in kept[n:n + args.batch]:
                f.write(json.dumps(d) + "\n")
    print(f"Wrote {len(kept)} documents under {args.out}")


if __name__ == "__main__":
    main()
