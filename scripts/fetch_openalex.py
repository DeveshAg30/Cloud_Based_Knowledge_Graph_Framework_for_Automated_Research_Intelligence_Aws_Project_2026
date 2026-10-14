"""Pull a topical slice of real OpenAlex data through its REST API and write canonical JSONL batches.

The full OpenAlex snapshot is ~330 GB compressed; for this project a few thousand works on your
research area is plenty. Examples:

  # metadata + abstracts (authors, institutions, venues, concepts, citation links)
  python scripts/fetch_openalex.py --search "knowledge graph" --from-year 2019 --limit 1000

  # also download each open-access PDF and extract its full text for deep NLP
  python scripts/fetch_openalex.py --search "retrieval augmented generation" --limit 200 --with-fulltext

  aws s3 cp data/openalex s3://<bucket>/raw/openalex/ --recursive

Without an API key OpenAlex allows a small daily budget (about 100 list pages); register for a free
key at openalex.org and pass --api-key for more. Each output file holds --batch documents so one
ingest Lambda invocation stays well under its timeout.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "backend"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # non-Latin titles on Windows consoles
from kg.parsers import from_openalex_work, from_pdf_bytes  # noqa: E402
from kg.schema import normalize_name  # noqa: E402

API = "https://api.openalex.org/works"
MAX_PDF_BYTES = 30 * 1024 * 1024
ARXIV_SOURCE = "S4306400194"  # OpenAlex source id of arXiv


def fetch(args):
    # is_retracted:false: the proposal's data-quality step (OpenAlex retraction handling is inconsistent upstream).
    filters = [f"title_and_abstract.search:{args.search}", "has_abstract:true", "type:article|preprint", "is_retracted:false"]
    if args.from_year:
        filters.append(f"from_publication_date:{args.from_year}-01-01")
    if args.min_references:
        filters.append(f"referenced_works_count:>{args.min_references - 1}")
    if args.with_fulltext:
        filters.append("has_content.pdf:true")  # OpenAlex knows a PDF exists for the work
        if not args.api_key:
            # Without a key, publisher PDFs mostly return 403 and PMC serves a bot check, so keep
            # works that have an arXiv copy; arXiv serves PDFs directly and covers CS/NLP well.
            filters.append(f"locations.source.id:{ARXIV_SOURCE}")
    cursor, fetched = "*", 0
    while cursor and fetched < args.limit:
        params = {"filter": ",".join(filters), "per-page": 200, "cursor": cursor, "sort": args.sort}
        if args.email:
            params["mailto"] = args.email
        if args.api_key:
            params["api_key"] = args.api_key
        with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(params)}", timeout=60) as resp:
            page = json.load(resp)
        if not page["results"]:
            return
        for work in page["results"]:
            yield work
            fetched += 1
            if fetched >= args.limit:
                return
        cursor = page["meta"].get("next_cursor")
        time.sleep(0.2)


def pdf_urls(work, api_key=None):
    """Candidate PDF links: arXiv first (most reliable), then other open-access copies, then
    OpenAlex's own cached PDF, which needs an API key."""
    urls = []
    candidates = [(work.get("open_access") or {}).get("oa_url")]
    candidates += [loc.get("pdf_url") for loc in [work.get("best_oa_location")] + (work.get("locations") or []) if loc]
    for url in candidates:
        if url and url not in urls:
            urls.append(url)
    urls.sort(key=lambda u: "arxiv.org" not in u)
    cached = (work.get("content_urls") or {}).get("pdf")
    if cached and api_key:
        urls.append(f"{cached}?api_key={api_key}")
    return urls


def download_pdf(url, email):
    agent = f"research-kg/1.0 (mailto:{email})" if email else "research-kg/1.0"
    req = urllib.request.Request(url, headers={"User-Agent": agent})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read(MAX_PDF_BYTES + 1)
    if "arxiv.org" in url:
        time.sleep(3)  # arXiv asks automated clients for one request every 3 seconds
    if not data.startswith(b"%PDF") or len(data) > MAX_PDF_BYTES:
        raise ValueError("not a PDF (publisher landing page or paywall)")
    return data


def add_fulltext(doc, work, email, api_key):
    for url in pdf_urls(work, api_key)[:4]:
        shown = url.split("?api_key=")[0]
        try:
            pdf_doc = from_pdf_bytes(download_pdf(url, email))
        except (urllib.error.URLError, ValueError, TimeoutError, OSError) as e:
            print(f"    skip {shown}: {e}")
            continue
        except Exception as e:  # pypdf raises many types on malformed files
            print(f"    skip {shown}: unreadable PDF ({type(e).__name__})")
            continue
        if sum(len(s["text"]) for s in pdf_doc["sections"]) > 2000:
            doc["sections"] = pdf_doc["sections"]
            doc["fulltext_url"] = shown
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", required=True, help="matched against title and abstract")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--from-year", type=int)
    ap.add_argument("--min-references", type=int, default=1,
                    help="skip works with fewer references (0 keeps everything); citation edges need these")
    ap.add_argument("--sort", default="cited_by_count:desc", help="e.g. cited_by_count:desc, publication_date:desc")
    ap.add_argument("--with-fulltext", action="store_true", help="download open-access PDFs and extract full text")
    ap.add_argument("--email", help="contact email: OpenAlex polite pool and PDF hosts' User-Agent")
    ap.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY"))
    ap.add_argument("--out", default="data/openalex")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    batch, n_files, total, with_text, seen_titles = [], 0, 0, 0, set()
    for work in fetch(args):
        doc = from_openalex_work(work)
        title_key = normalize_name(doc["title"])
        if not title_key or title_key in seen_titles:  # preprint + published versions of the same paper
            continue
        seen_titles.add(title_key)
        if args.with_fulltext:
            print(f"[{total + 1}] {doc['title'][:70]}")
            if add_fulltext(doc, work, args.email, args.api_key):
                with_text += 1
        batch.append(doc)
        total += 1
        if len(batch) == args.batch:
            n_files += 1
            _write(args.out, n_files, batch)
            batch = []
    if batch:
        n_files += 1
        _write(args.out, n_files, batch)
    print(f"Wrote {total} documents ({with_text} with full text) into {n_files} files under {args.out}")


def _write(out, n, docs):
    with open(os.path.join(out, f"openalex_{n:04d}.jsonl"), "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    main()
