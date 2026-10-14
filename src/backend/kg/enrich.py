"""Fill in metadata for uploaded PDFs by matching their title on OpenAlex.

A PDF's text gives us the body, but often no reliable authors, year or reference list. When OpenAlex
has a work with (almost) the same title we copy its authors and institutions, year, venue, concepts
and references, and adopt its id. That id makes the paper merge with an existing node, e.g. one
created earlier as a cited reference, instead of becoming an isolated duplicate.
"""

import difflib
import json
import urllib.parse
import urllib.request

from .parsers import from_openalex_work
from .schema import normalize_name

API = "https://api.openalex.org/works"


def find_work(title: str, timeout: float = 8.0):
    if not title or len(title) < 15:
        return None
    query = urllib.parse.urlencode({"filter": f"title.search:{title[:200].replace(',', ' ')}", "per-page": 5})
    with urllib.request.urlopen(f"{API}?{query}", timeout=timeout) as resp:
        results = json.load(resp).get("results", [])
    wanted = normalize_name(title)
    best, score = None, 0.0
    for work in results:
        s = difflib.SequenceMatcher(None, wanted, normalize_name(work.get("display_name") or "")).ratio()
        if s > score:
            best, score = work, s
    return best if score >= 0.9 else None


def enrich_pdf_document(doc: dict) -> dict:
    """Returns the document, enriched in place when a confident OpenAlex match exists. Never raises."""
    try:
        work = find_work(doc.get("title", ""))
    except Exception as e:  # network or API problems must not block ingestion
        print(f"OpenAlex lookup skipped: {type(e).__name__}")
        return doc
    if not work:
        return doc
    meta = from_openalex_work(work)
    doc["doc_id"] = meta["doc_id"]
    doc["title"] = meta["title"] or doc["title"]
    for field in ("year", "venue", "authors", "references", "concepts"):
        if meta.get(field):
            doc[field] = meta[field]
    doc["abstract"] = doc.get("abstract") or meta.get("abstract", "")
    doc["source"] = "pdf+openalex"
    print(f"Matched upload to OpenAlex {meta['doc_id']}: {len(doc['authors'])} authors, {len(doc['references'])} references")
    return doc
