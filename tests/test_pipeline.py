import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "backend"))

from kg.extractors import extract_rules  # noqa: E402
from kg.graph_writer import build_graph, upsert  # noqa: E402
from kg.parsers import from_openalex_work, from_s2orc, parse_json_payload, rebuild_abstract  # noqa: E402
from kg.retrieval import answer_question, candidate_terms  # noqa: E402
from kg.schema import make_uid  # noqa: E402

SAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "sample_papers.jsonl")


class RecordingGraph:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or []

    def query(self, cypher, params=None):
        self.calls.append((cypher, params))
        for needle, rows in self.responses:
            if needle in cypher:
                return rows
        return []


def test_openalex_conversion():
    work = {
        "id": "https://openalex.org/W1", "display_name": "Graph RAG", "publication_year": 2024,
        "primary_location": {"source": {"display_name": "ACL"}},
        "abstract_inverted_index": {"graphs": [1], "Knowledge": [0], "help": [2]},
        "authorships": [{"author": {"id": "https://openalex.org/A9", "display_name": "Ada"},
                         "institutions": [{"id": "https://openalex.org/I3", "display_name": "VIT"}]}],
        "referenced_works": ["https://openalex.org/W2"],
        "concepts": [{"display_name": "Knowledge graph", "score": 0.9}, {"display_name": "Noise", "score": 0.1}],
    }
    doc = from_openalex_work(work)
    assert doc["doc_id"] == "W1"
    assert doc["abstract"] == "Knowledge graphs help"
    assert doc["authors"][0] == {"id": "A9", "name": "Ada", "institutions": [{"id": "I3", "name": "VIT"}]}
    assert doc["references"] == [{"id": "W2", "title": ""}]
    assert doc["concepts"] == ["Knowledge graph"]


def test_rebuild_abstract_empty():
    assert rebuild_abstract(None) == ""


def test_s2orc_resolves_inline_citations():
    meta = {"paper_id": "77", "title": "T", "authors": [{"first": "Grace", "middle": [], "last": "Hopper"}],
            "outbound_citations": ["88"]}
    parse = {"body_text": [{"text": "We follow [1].", "section": "Method",
                            "cite_spans": [{"start": 10, "end": 13, "ref_id": "BIBREF0"}]}],
             "bib_entries": {"BIBREF0": {"title": "Cited paper", "link": "88"},
                             "BIBREF1": {"title": "Unlinked paper", "link": None}}}
    doc = from_s2orc(meta, parse)
    assert doc["authors"][0]["name"] == "Grace Hopper"
    assert doc["sections"][0]["citations"] == [{"ref_id": "88", "title": "Cited paper"}]
    assert {"id": None, "title": "Unlinked paper"} in doc["references"]


def test_parse_payload_accepts_jsonl():
    with open(SAMPLE, encoding="utf-8") as f:
        docs = parse_json_payload(f.read())
    assert len(docs) == 4 and docs[0]["doc_id"] == "sample-001"


def test_rule_extraction_finds_entities_and_relations():
    out = extract_rules("We evaluate TransE and BERT on WN18RR for link prediction using MRR and Hits@10.")
    found = {(e["label"], e["text"].lower()) for e in out["entities"]}
    assert ("Method", "transe") in found and ("Dataset", "wn18rr") in found
    assert ("Task", "link prediction") in found and ("Metric", "mrr") in found
    rels = {(r["head"].lower(), r["rel"], r["tail"].lower()) for r in out["relations"]}
    assert ("transe", "EVALUATED_ON", "wn18rr") in rels
    assert ("bert", "APPLIED_TO", "link prediction") in rels


def test_build_graph_and_upsert_are_consistent():
    with open(SAMPLE, encoding="utf-8") as f:
        doc = parse_json_payload(f.read())[1]
    nodes, stubs, edges = build_graph(doc, extract_rules(doc["abstract"]))
    uids = {n["uid"] for n in nodes} | {s["uid"] for s in stubs}
    assert all(e["from"] in uids and e["to"] in uids for e in edges)
    assert any(e["rel"] == "USES_DATASET" and e["to"] == make_uid("Dataset", "WN18RR") for e in edges)
    assert any(e["rel"] == "CITES" for e in edges)

    graph = RecordingGraph()
    stats = upsert(graph, nodes, stubs, edges)
    assert stats["edges"] == len(edges)
    assert all("MERGE" in c for c, _ in graph.calls)
    json.dumps([p for _, p in graph.calls])  # parameters must be JSON-serialisable for Neptune


def test_candidate_terms_drop_stopwords():
    terms = candidate_terms("Which methods are evaluated on WN18RR?")
    assert "wn18rr" in terms and "which" not in terms


def test_answer_question_uses_graph_facts():
    graph = RecordingGraph([
        ("n.name_lc IN $terms", [{"uid": "dataset:x", "label": "Dataset", "name": "WN18RR"}]),
        ("<-[r]-(n)", [{"rel": "USES_DATASET", "uid": "paper:p1", "label": "Paper", "name": "KG-CF",
                        "year": 2025, "doc_id": "p1", "evidence": "Experiments on WN18RR", "edge_doc": "p1",
                        "mentions": 2}]),
    ])
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return "KG-CF uses WN18RR [doc:p1]."

    result = answer_question(graph, "Which papers use WN18RR?", generate=fake_llm)
    assert "(Paper) KG-CF -[USES_DATASET]-> (Dataset) WN18RR" in seen["prompt"]
    assert result["citations"] == ["p1"]
    assert {n["id"] for n in result["graph"]["nodes"]} == {"dataset:x", "paper:p1"}


def test_answer_question_without_matches_skips_llm():
    result = answer_question(RecordingGraph(), "zzz qqq", generate=lambda p: 1 / 0)
    assert result["facts"] == [] and "couldn't match" in result["answer"]


def _s2_text():
    body = "Introduction We build on TransE [1] for link prediction."
    bib = "[1] Translating embeddings for modeling multi-relational data. Bordes et al."
    ann_body = {
        "section_header": [{"start": 0, "end": 12}],
        "paragraph": [{"start": 13, "end": len(body)}],
        "bib_ref": [{"start": 35, "end": 38, "attributes": {"ref_id": "b0"}}],
    }
    ann_bib = {
        "bib_entry": [{"start": 0, "end": len(bib), "attributes": {"id": "b0", "matched_paper_id": 14941970}}],
        "bib_title": [{"start": 4, "end": 61}],
    }
    return body, bib, ann_body, ann_bib


def test_s2orc_v2_record():
    body, bib, ann_body, ann_bib = _s2_text()
    rec = {"corpusid": 42, "title": "Paper X", "authors": [{"name": "Ada Lovelace"}],
           "body": {"text": body, "annotations": ann_body}, "bibliography": {"text": bib, "annotations": ann_bib}}
    doc = parse_json_payload(json.dumps(rec))[0]
    assert doc["doc_id"] == "s2-42" and doc["authors"][0]["name"] == "Ada Lovelace"
    assert doc["sections"][0]["heading"] == "Introduction"
    assert doc["sections"][0]["citations"] == [
        {"ref_id": "s2-14941970", "title": "Translating embeddings for modeling multi-relational data"}]
    assert doc["references"][0]["id"] == "s2-14941970"


def test_s2orc_v1_record_with_json_string_annotations():
    body, bib, ann_body, ann_bib = _s2_text()
    text = body + "\n" + bib
    shift = len(body) + 1
    ann = {k: json.dumps(v) for k, v in ann_body.items()}
    ann["bibentry"] = json.dumps([{"start": shift, "end": len(text), "attributes": {"id": "b0", "matched_paper_id": 7}}])
    ann["bibtitle"] = json.dumps([{"start": shift + 4, "end": shift + 61}])
    ann["title"] = json.dumps([{"start": 0, "end": 12}])
    doc = parse_json_payload(json.dumps({"corpusid": 5, "content": {"text": text, "annotations": ann}}))[0]
    assert doc["title"] == "Introduction"
    assert doc["sections"][0]["citations"][0]["ref_id"] == "s2-7"


def test_answer_without_llm_returns_ranked_facts():
    from kg.llm import LLMDisabled

    graph = RecordingGraph([
        ("n.name_lc IN $terms", [{"uid": "dataset:x", "label": "Dataset", "name": "WN18RR"}]),
        ("<-[r]-(n)", [{"rel": "USES_DATASET", "uid": "paper:p1", "label": "Paper", "name": "KG-CF",
                        "year": 2025, "doc_id": "p1", "evidence": "", "edge_doc": "p1", "mentions": 1}]),
    ])

    def disabled(prompt):
        raise LLMDisabled()

    result = answer_question(graph, "Which papers use WN18RR?", generate=disabled)
    assert "No LLM configured" in result["answer"] and "[doc:p1]" in result["answer"]


def test_llm_extractor_falls_back_to_rules_when_disabled(monkeypatch):
    import kg.llm
    from kg.extractors import extract

    monkeypatch.setattr(kg.llm, "PROVIDER", "none")
    out = extract("We evaluate TransE on WN18RR for link prediction.", mode="llm")
    assert any(e["text"].lower() == "transe" for e in out["entities"])


def test_aliases_merge_abbreviations_and_plurals():
    from kg.aliases import canonical

    assert {canonical("Method", t) for t in ["GCN", "GCNs", "graph convolutional networks"]} == {"Graph Convolutional Network"}
    assert canonical("Method", "LLMs") == "Large Language Model"
    assert canonical("Metric", "mean reciprocal rank") == "MRR"
    assert canonical("Task", "sentiment analysis") == "sentiment analysis"  # no false singularisation


def test_abbreviations_learned_from_the_paper():
    from kg.aliases import canonical, learn_abbreviations

    abbr = learn_abbreviations("We introduce temporal knowledge graph networks (TKGNs).")
    assert canonical("Method", "TKGN", abbr) == "Temporal Knowledge Graph Network"


def test_extraction_output_is_canonical_and_deduplicated():
    out = extract_rules_canonical("Graph convolutional networks (GCNs) and the GCN model are evaluated using MRR and mean reciprocal rank.")
    names = [e["text"] for e in out["entities"]]
    assert names.count("Graph Convolutional Network") == 1 and names.count("MRR") == 1


def extract_rules_canonical(text):
    from kg.extractors import extract

    return extract(text, mode="rules")


def test_full_title_matches_the_paper_itself():
    from kg.retrieval import link_entities

    graph = RecordingGraph([("n.name_lc = $q", [{"uid": "paper:W1", "label": "Paper", "name": "A Long Paper Title About Graphs"}])])
    seeds = link_entities(graph, "A long paper title about graphs")
    assert [s["uid"] for s in seeds] == ["paper:W1"]


def test_fragments_of_longer_matches_are_dropped():
    from kg.retrieval import link_entities

    graph = RecordingGraph([("n.name_lc IN $terms", [
        {"uid": "c:graph", "label": "Concept", "name": "Graph", "term": "graph"},
        {"uid": "m:gcn", "label": "Method", "name": "Graph Convolutional Network", "term": "graph convolutional network"},
        {"uid": "c:sol", "label": "Concept", "name": "Solubility", "term": "solubility"},
    ])])
    uids = {s["uid"] for s in link_entities(graph, "solubility with graph convolutional network")}
    assert uids == {"m:gcn", "c:sol"}


def _api_event(route, groups=None, body="{}"):
    claims = {"email": "x@example.com"}
    if groups is not None:
        claims["cognito:groups"] = groups
    return {"routeKey": route, "body": body, "requestContext": {"authorizer": {"jwt": {"claims": claims}}}}


def test_upload_requires_members_group(monkeypatch):
    os.environ.setdefault("AWS_REGION", "us-east-1")
    os.environ.setdefault("DATA_BUCKET", "test-bucket")
    from handlers import api

    assert api.handler(_api_event("POST /upload-url"), None)["statusCode"] == 403
    assert api.handler(_api_event("POST /upload-url", "[admins]"), None)["statusCode"] == 403
    monkeypatch.setattr(api.s3, "generate_presigned_url", lambda *a, **k: "https://signed")
    ok = api.handler(_api_event("POST /upload-url", "[members]", '{"filename": "p.pdf"}'), None)
    assert ok["statusCode"] == 200 and json.loads(ok["body"])["url"] == "https://signed"


def test_upload_status_processing_then_done(monkeypatch):
    os.environ.setdefault("AWS_REGION", "us-east-1")
    os.environ.setdefault("DATA_BUCKET", "test-bucket")
    import io
    from botocore.exceptions import ClientError
    from handlers import api

    def missing(**kw):
        raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
    monkeypatch.setattr(api.s3, "get_object", missing)
    ev = _api_event("GET /upload-status", "[members]")
    ev["queryStringParameters"] = {"key": "raw/uploads/ab12cd34-paper.pdf"}
    assert json.loads(api.handler(ev, None)["body"]) == {"status": "processing"}

    monkeypatch.setattr(api.s3, "get_object", lambda **kw: {"Body": io.BytesIO(b'{"status": "done", "edges": 5}')})
    assert json.loads(api.handler(ev, None)["body"])["status"] == "done"
    ev["queryStringParameters"] = {"key": "processed/../secret.json"}
    assert api.handler(ev, None)["statusCode"] == 400


def test_guess_title_skips_arxiv_stamp_and_venue_lines():
    from kg.parsers import guess_title

    text = "arXiv:1903.10676v3  [cs.CL]  10 Sep 2019\nProceedings of EMNLP 2019\nSCIBERT: A Pretrained Language Model for Scientific Text\nIz Beltagy"
    assert guess_title(text) == "SCIBERT: A Pretrained Language Model for Scientific Text"


def test_clean_pdf_text_repairs_hyphenation_small_caps_and_wrapping():
    from kg.parsers import clean_pdf_text

    raw = "S CIBERT: A Pretrained Model\nIz Beltagy\nWe use scien-\ntific text and improve perfor-\nmance over B ERT on\nmany tasks."
    out = clean_pdf_text(raw)
    assert "SCIBERT: A Pretrained Model\nIz Beltagy" in out
    assert "scientific text" in out and "performance over BERT on many tasks." in out


def test_dedupe_tidies_names_and_picks_one_label():
    from kg.extractors import dedupe_entities

    ents = [{"text": "machine reading of", "label": "Task", "sentence": "s"},
            {"text": "Text Classification (", "label": "Task", "sentence": "s"},
            {"text": "SciBERT", "label": "Method", "sentence": "s"},
            {"text": "SciBERT", "label": "Method", "sentence": "t"},
            {"text": "SciBERT", "label": "Dataset", "sentence": "s"}]
    out = {(e["label"], e["text"]) for e in dedupe_entities(ents)}
    assert out == {("Task", "machine reading"), ("Task", "Text Classification"), ("Method", "SciBERT")}


def test_answer_survives_llm_timeout():
    graph = RecordingGraph([
        ("n.name_lc IN $terms", [{"uid": "dataset:x", "label": "Dataset", "name": "WN18RR"}]),
        ("<-[r]-(n)", [{"rel": "USES_DATASET", "uid": "paper:p1", "label": "Paper", "name": "KG-CF",
                        "year": 2025, "doc_id": "p1", "evidence": "", "edge_doc": "p1", "mentions": 1}]),
    ])

    def slow(prompt):
        raise TimeoutError("The read operation timed out")

    result = answer_question(graph, "Which papers use WN18RR?", generate=slow)
    assert result["degraded"] and "temporarily unavailable" in result["answer"] and result["citations"] == ["p1"]


def test_groq_fallback_when_gemini_fails(monkeypatch):
    import kg.llm as llm

    monkeypatch.setattr(llm, "PROVIDER", "gemini")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_gemini", lambda *a, **k: (_ for _ in ()).throw(TimeoutError()))
    monkeypatch.setattr(llm, "_groq", lambda prompt, system, max_tokens, timeout: "answer from groq")
    assert llm.complete("q") == "answer from groq"


def test_fact_number_citations_are_mapped_to_paper_ids():
    from kg.retrieval import normalize_citations

    facts = [{"doc_id": "W1"}, {"doc_id": None}, {"doc_id": "W3"}]
    out = normalize_citations("GNNs are used 【F1】 and here [F3] and (F1, F3) but not [F9].", facts)
    assert "[doc:W1]" in out and "[doc:W3]" in out and "F9" not in out and "【" not in out


def test_lenticular_doc_citations_are_normalized():
    from kg.retrieval import normalize_citations

    assert normalize_citations("by Alper 【doc:W2951159620】 and (doc:W1)", []) == "by Alper [doc:W2951159620] and [doc:W1]"
