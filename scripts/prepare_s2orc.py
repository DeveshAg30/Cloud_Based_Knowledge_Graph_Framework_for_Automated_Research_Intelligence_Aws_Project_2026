"""Join S2ORC metadata + pdf_parses shards by paper_id, resolve inline citations, write canonical JSONL.

S2ORC ships metadata and full text in separate shards (metadata_N.jsonl.gz, pdf_parses_N.jsonl.gz).
Only papers with full text are kept, since those are the ones worth deep NLP.

  python scripts/prepare_s2orc.py --metadata metadata_0.jsonl.gz --pdf-parses pdf_parses_0.jsonl.gz \
      --limit 1000 --out data/s2orc
  aws s3 cp data/s2orc s3://<bucket>/raw/s2orc/ --recursive
"""

import argparse
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "backend"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # non-Latin titles on Windows consoles
from kg.parsers import from_s2orc  # noqa: E402


def read_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--pdf-parses", required=True)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--out", default="data/s2orc")
    args = ap.parse_args()

    parses = {}
    for p in read_jsonl(args.pdf_parses):
        if p.get("body_text"):
            parses[str(p["paper_id"])] = p

    os.makedirs(args.out, exist_ok=True)
    batch, n_files, total = [], 0, 0
    for meta in read_jsonl(args.metadata):
        parse = parses.get(str(meta["paper_id"]))
        if not parse or not meta.get("title"):
            continue
        batch.append(from_s2orc(meta, parse))
        total += 1
        if len(batch) == args.batch:
            n_files += 1
            _write(args.out, n_files, batch)
            batch = []
        if total >= args.limit:
            break
    if batch:
        n_files += 1
        _write(args.out, n_files, batch)
    print(f"Wrote {total} full-text documents into {n_files} files under {args.out}")


def _write(out, n, docs):
    with open(os.path.join(out, f"s2orc_{n:04d}.jsonl"), "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    main()
