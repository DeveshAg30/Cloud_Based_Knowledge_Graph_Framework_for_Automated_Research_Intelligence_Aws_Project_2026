"""Convert raw inputs (PDF, OpenAlex work JSON, S2ORC JSON, canonical JSON) into canonical documents."""

import hashlib
import io
import json
import logging
import re
import unicodedata

from .schema import empty_document

logging.getLogger("pypdf").setLevel(logging.ERROR)  # real-world PDFs trigger many harmless xref warnings


def short_openalex_id(value):
    """'https://openalex.org/W123' -> 'W123'."""
    if not value:
        return None
    return value.rsplit("/", 1)[-1]


def rebuild_abstract(inverted_index):
    """OpenAlex stores abstracts as {word: [positions]}; turn it back into text."""
    if not inverted_index:
        return ""
    positions = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    return " ".join(w for _, w in sorted(positions))


def from_openalex_work(work: dict) -> dict:
    doc = empty_document(short_openalex_id(work.get("id")), "openalex")
    doc["title"] = work.get("display_name") or work.get("title") or ""
    doc["year"] = work.get("publication_year")
    source = ((work.get("primary_location") or {}).get("source") or {})
    doc["venue"] = source.get("display_name")
    doc["abstract"] = rebuild_abstract(work.get("abstract_inverted_index"))
    for a in work.get("authorships", []):
        author = a.get("author") or {}
        if not author.get("display_name"):
            continue
        doc["authors"].append({
            "id": short_openalex_id(author.get("id")),
            "name": author["display_name"],
            "institutions": [
                {"id": short_openalex_id(i.get("id")), "name": i["display_name"]}
                for i in a.get("institutions", []) if i.get("display_name")
            ],
        })
    doc["references"] = [{"id": short_openalex_id(r), "title": ""} for r in work.get("referenced_works", [])]
    concepts = [c["display_name"] for c in work.get("concepts", []) if c.get("score", 0) >= 0.4]
    concepts += [t["display_name"] for t in work.get("topics", [])[:3]]
    doc["concepts"] = sorted(set(concepts))
    return doc


def _s2orc_author_name(a: dict) -> str:
    return " ".join(p for p in [a.get("first", ""), " ".join(a.get("middle", [])), a.get("last", "")] if p).strip()


def from_s2orc(metadata: dict, pdf_parse: dict | None) -> dict:
    """Join an S2ORC metadata record with its pdf_parse record (same paper_id).

    Inline citation markers (cite_spans[].ref_id = 'BIBREF3') are resolved against bib_entries,
    so each section knows exactly which papers it cites.
    """
    doc = empty_document(str(metadata["paper_id"]), "s2orc")
    doc["title"] = metadata.get("title") or ""
    doc["year"] = metadata.get("year")
    doc["venue"] = metadata.get("venue") or metadata.get("journal")
    doc["abstract"] = metadata.get("abstract") or ""
    doc["authors"] = [
        {"id": None, "name": _s2orc_author_name(a), "institutions": []}
        for a in metadata.get("authors", []) if _s2orc_author_name(a)
    ]
    doc["concepts"] = metadata.get("mag_field_of_study") or []
    doc["references"] = [{"id": str(r), "title": ""} for r in metadata.get("outbound_citations", [])]

    if pdf_parse:
        bib = pdf_parse.get("bib_entries", {})
        for para in pdf_parse.get("body_text", []):
            cites = []
            for span in para.get("cite_spans", []):
                entry = bib.get(span.get("ref_id") or "")
                if entry:
                    cites.append({"ref_id": entry.get("link"), "title": entry.get("title") or ""})
            doc["sections"].append({"heading": para.get("section") or "", "text": para.get("text") or "", "citations": cites})
        # Resolved bibliography entries give titles for references that metadata only listed by id.
        known = {r["id"] for r in doc["references"]}
        for entry in bib.values():
            link = entry.get("link")
            if link and str(link) not in known:
                doc["references"].append({"id": str(link), "title": entry.get("title") or ""})
            elif not link and entry.get("title"):
                doc["references"].append({"id": None, "title": entry["title"]})
    return doc


# ---------------------------------------------------------------- current Semantic Scholar S2ORC
def _spans(annotations: dict, *names):
    """Annotation spans by any of several key spellings; values may be JSON-encoded strings."""
    for name in names:
        value = (annotations or {}).get(name)
        if value:
            return json.loads(value) if isinstance(value, str) else value
    return []


def _person_name(a) -> str:
    if isinstance(a, str):
        return a
    return a.get("name") or " ".join(p for p in [a.get("first", ""), " ".join(a.get("middle") or []), a.get("last", "")] if p)


def s2_doc_id(corpus_id) -> str:
    return f"s2-{corpus_id}"


def from_s2orc_current(rec: dict) -> dict:
    """Parse a record of the Semantic Scholar Datasets API ``s2orc`` or ``s2orc_v2`` dataset.

    Both store Grobid output as one text string plus annotation spans {start, end, attributes}:
      s2orc:    content.text / content.annotations (bibliography inside the same text)
      s2orc_v2: body.text / body.annotations + bibliography.text / bibliography.annotations
    Inline citations (bib_ref spans, attributes.ref_id) are resolved to bibliography entries
    (bib_entry spans, attributes.id / matched_paper_id = cited paper's corpus id).
    """
    corpus_id = rec.get("corpusid") or ((rec.get("openaccessinfo") or {}).get("externalids") or {}).get("CorpusId")
    doc = empty_document(s2_doc_id(corpus_id), "s2orc")

    if "body" in rec:  # s2orc_v2
        body_text, body_ann = (rec["body"] or {}).get("text") or "", (rec["body"] or {}).get("annotations") or {}
        bib_text, bib_ann = (rec.get("bibliography") or {}).get("text") or "", (rec.get("bibliography") or {}).get("annotations") or {}
    else:  # s2orc
        content = rec.get("content") or {}
        body_text, body_ann = content.get("text") or "", content.get("annotations") or {}
        bib_text, bib_ann = body_text, body_ann

    def cut(text, span):
        return text[int(span["start"]):int(span["end"])].strip()

    doc["title"] = rec.get("title") or next((cut(body_text, s) for s in _spans(body_ann, "title")), "")
    doc["authors"] = [{"id": None, "name": _person_name(a), "institutions": []} for a in rec.get("authors") or []
                      if _person_name(a)]
    doc["abstract"] = " ".join(cut(body_text, s) for s in _spans(body_ann, "abstract"))

    # Bibliography: entry id -> {corpus id, title}
    titles = _spans(bib_ann, "bib_title", "bibtitle")
    bib = {}
    for entry in _spans(bib_ann, "bib_entry", "bibentry"):
        attrs = entry.get("attributes") or {}
        start, end = int(entry["start"]), int(entry["end"])
        title = next((cut(bib_text, t) for t in titles if start <= int(t["start"]) < end), "")
        matched = attrs.get("matched_paper_id")
        bib[attrs.get("id")] = {"id": s2_doc_id(matched) if matched else None, "title": title}
    doc["references"] = [b for b in bib.values() if b["id"] or b["title"]]

    # Body paragraphs with their section heading and resolved inline citations.
    headers = sorted(_spans(body_ann, "section_header", "sectionheader"), key=lambda s: int(s["start"]))
    refs = _spans(body_ann, "bib_ref", "bibref")
    for para in sorted(_spans(body_ann, "paragraph"), key=lambda s: int(s["start"])):
        start, end = int(para["start"]), int(para["end"])
        heading = next((cut(body_text, h) for h in reversed(headers) if int(h["start"]) < start), "")
        cites = []
        for r in refs:
            if start <= int(r["start"]) < end:
                entry = bib.get((r.get("attributes") or {}).get("ref_id"))
                if entry:
                    cites.append({"ref_id": entry["id"], "title": entry["title"]})
        doc["sections"].append({"heading": heading, "text": cut(body_text, para), "citations": cites})
    if not doc["sections"] and body_text:
        doc["sections"] = [{"heading": "body", "text": body_text, "citations": []}]
    return doc


_NOT_TITLE = re.compile(
    r"^(arxiv:|doi|https?://|www\.|preprint|accepted|published|proceedings|copyright|©|journal|vol\.|volume|"
    r"received|submitted|under review|workshop|conference|\d{1,2}\s+\w+\s+\d{4}$)", re.I)


def clean_pdf_text(text: str) -> str:
    """Repair the PDF text layer before NLP:
    - "scien-\\ntific" -> "scientific" (words hyphenated across line breaks);
    - "B ERT" -> "BERT" (small-caps letters split apart; A and I are skipped because they are real words);
    - lines that continue a sentence (next line starts lower-case) are joined, so remaining line breaks
      separate real units such as the title, author lines and headings."""
    text = re.sub(r"(\w)-[ \t]*\n[ \t]*([a-z])", r"\1\2", text)
    text = re.sub(r"\b([B-HJ-Z]) ([A-Z]{2,})\b", r"\1\2", text)
    text = re.sub(r"[ \t]*\n[ \t]*(?=[a-z(])", " ", text)
    return text


def guess_title(text: str) -> str:
    """First plausible title line near the top of page 1 (skips arXiv stamps, DOIs, dates, venue lines)."""
    for line in [ln.strip() for ln in text.splitlines()[:25]]:
        letters = sum(c.isalpha() for c in line)
        if 15 <= len(line) <= 250 and letters / max(len(line), 1) > 0.6 and not _NOT_TITLE.search(line) \
                and "@" not in line and not re.search(r"\[cs\.\w+\]|\d{4}\.\d{4,5}", line):
            return line
    return ""


def from_pdf_bytes(data: bytes, filename: str = "") -> dict:
    from pypdf import PdfReader  # imported lazily: only the ingest Lambda needs it

    reader = PdfReader(io.BytesIO(data))
    pages = [(p.extract_text() or "") for p in reader.pages]
    # NFKC turns typographic ligatures ("ﬁ", "ﬂ") back into plain letters so words match.
    text = clean_pdf_text(unicodedata.normalize("NFKC", "\n".join(pages)))
    doc = empty_document(hashlib.sha1(data).hexdigest()[:20], "pdf")
    meta = reader.metadata or {}
    meta_title = (meta.get("/Title") or "").strip()
    if len(meta_title) < 10 or re.search(r"\.(docx?|tex|dvi)$|^untitled|^microsoft word", meta_title, re.I):
        meta_title = ""  # tools often leave junk like "main.tex" or "Microsoft Word - draft.docx"
    doc["title"] = (meta_title or guess_title(text) or filename)[:300]
    if meta.get("/Author"):
        doc["authors"] = [{"id": None, "name": n.strip(), "institutions": []}
                          for n in re.split(r",| and |;", meta["/Author"]) if n.strip()]

    # Split into Abstract / body on common headings; good enough for text-layer PDFs.
    # Journals label it "Abstract" or "Summary" (e.g. Cell Press); it ends at the Introduction / Keywords.
    m = re.search(r"\b(?:abstract|summary)\b(.{50,3000}?)(?:\n\s*(?:1\.?\s*)?introduction\b|\n\s*keywords|\n\s*key\s*words)",
                  text, re.I | re.S)
    if m:
        doc["abstract"] = re.sub(r"\s+", " ", m.group(1)).strip()
    body = re.split(r"\n\s*references\s*\n", text, flags=re.I)[0]
    doc["sections"] = [{"heading": "body", "text": body, "citations": []}]
    return doc


def is_canonical(obj: dict) -> bool:
    return isinstance(obj, dict) and "doc_id" in obj and "sections" in obj


def parse_json_payload(raw: str):
    """Accept a single canonical doc, a list of them, JSONL, or raw OpenAlex works."""
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        items = json.loads(raw)
    elif raw.startswith("{") and "\n{" not in raw:
        items = [json.loads(raw)]
    else:
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]

    docs = []
    for item in items:
        if is_canonical(item):
            docs.append(item)
        elif str(item.get("id", "")).startswith("https://openalex.org/W"):
            docs.append(from_openalex_work(item))
        elif "paper_id" in item:
            docs.append(from_s2orc(item, item.get("pdf_parse")))
        elif "corpusid" in item and ("body" in item or "content" in item):
            docs.append(from_s2orc_current(item))
        else:
            raise ValueError(f"Unrecognised record with keys {sorted(item)[:8]}")
    return docs
