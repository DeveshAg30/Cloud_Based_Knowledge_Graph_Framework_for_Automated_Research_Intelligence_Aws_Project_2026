"""Turn a canonical document + extraction result into nodes/edges and upsert them with openCypher MERGE."""

from collections import Counter, defaultdict

from .schema import (AUTHOR, CONCEPT, INSTITUTION, PAPER, PAPER_REL_FOR_LABEL, VENUE, make_uid,
                     normalize_name, paper_uid)

BATCH = 200


def _node(uid, label, name, **props):
    return {"uid": uid, "label": label, "name": name, "props": {k: v for k, v in props.items() if v is not None}}


def _edge(src, rel, dst, **props):
    return {"from": src["uid"], "from_label": src["label"], "rel": rel,
            "to": dst["uid"], "to_label": dst["label"], "props": props}


def build_graph(doc: dict, extraction: dict):
    """Returns (nodes, stub_nodes, edges). Stub nodes are cited papers we have not ingested yet."""
    nodes, stubs, edges = {}, {}, []

    def add(n, into=nodes):
        if n["uid"] not in nodes:
            into.setdefault(n["uid"], n)
        return n

    paper = add(_node(paper_uid(doc["doc_id"]), PAPER, doc.get("title") or doc["doc_id"],
                      doc_id=doc["doc_id"], year=doc.get("year"), source=doc.get("source"),
                      abstract=(doc.get("abstract") or "")[:2000]))

    for a in doc.get("authors", []):
        author = add(_node(f"author:{a['id']}" if a.get("id") else make_uid(AUTHOR, a["name"]), AUTHOR, a["name"]))
        edges.append(_edge(author, "AUTHORED", paper))
        for inst in a.get("institutions", []):
            i = add(_node(f"institution:{inst['id']}" if inst.get("id") else make_uid(INSTITUTION, inst["name"]),
                          INSTITUTION, inst["name"]))
            edges.append(_edge(author, "AFFILIATED_WITH", i))

    if doc.get("venue"):
        edges.append(_edge(paper, "PUBLISHED_IN", add(_node(make_uid(VENUE, doc["venue"]), VENUE, doc["venue"]))))

    for c in doc.get("concepts", []):
        edges.append(_edge(paper, "HAS_CONCEPT", add(_node(make_uid(CONCEPT, c), CONCEPT, c))))

    # Citation evidence: which section text cites which reference.
    evidence_for = {}
    for sec in doc.get("sections", []):
        for cite in sec.get("citations", []):
            key = cite.get("ref_id") or normalize_name(cite.get("title", ""))
            evidence_for.setdefault(key, f"[{sec.get('heading', '')}] {sec.get('text', '')[:400]}")
    for ref in doc.get("references", []):
        if ref.get("id"):
            uid, key = paper_uid(ref["id"]), ref["id"]
        elif ref.get("title"):
            uid, key = make_uid(PAPER, ref["title"]), normalize_name(ref["title"])
        else:
            continue
        cited = _node(uid, PAPER, ref.get("title") or ref.get("id"), doc_id=ref.get("id"))
        if uid != paper["uid"]:
            add(cited, into=stubs)
            edges.append(_edge(paper, "CITES", cited, evidence=evidence_for.get(key, "")))

    # NLP entities: Paper -> entity, weighted by mention count, plus entity -> entity relations.
    counts = Counter((e["label"], normalize_name(e["text"])) for e in extraction.get("entities", []))
    first_sentence, values = {}, {}
    for e in extraction.get("entities", []):
        first_sentence.setdefault((e["label"], normalize_name(e["text"])), e.get("sentence", ""))
        if e.get("value"):  # reported outcome for a metric, e.g. "96.7%"
            values.setdefault((e["label"], normalize_name(e["text"])), e["value"])
    entity_nodes = {}
    for e in extraction.get("entities", []):
        key = (e["label"], normalize_name(e["text"]))
        if key in entity_nodes:
            continue
        n = add(_node(make_uid(e["label"], e["text"]), e["label"], e["text"].strip()))
        entity_nodes[key] = n
        edges.append(_edge(paper, PAPER_REL_FOR_LABEL[e["label"]], n,
                           mentions=counts[key], evidence=first_sentence[key][:500], value=values.get(key)))
    for r in extraction.get("relations", []):
        h = entity_nodes.get((r["head_label"], normalize_name(r["head"])))
        t = entity_nodes.get((r["tail_label"], normalize_name(r["tail"])))
        if h and t:
            edges.append(_edge(h, r["rel"], t, evidence=r.get("evidence", "")[:500]))

    for e in edges:
        e["props"]["doc_id"] = doc["doc_id"]
    return list(nodes.values()), [s for uid, s in stubs.items() if uid not in nodes], edges


def _chunks(rows):
    for i in range(0, len(rows), BATCH):
        yield rows[i:i + BATCH]


def upsert(graph, nodes, stubs, edges):
    """Idempotent: re-ingesting the same document updates instead of duplicating."""
    if hasattr(graph, "ensure_schema"):
        graph.ensure_schema()
    by_label = defaultdict(list)
    for n in nodes:
        by_label[n["label"]].append({"uid": n["uid"], "name": n["name"], "name_lc": normalize_name(n["name"]),
                                     "props": n["props"]})
    for label, rows in by_label.items():
        for chunk in _chunks(rows):
            graph.query(
                f"UNWIND $rows AS r MERGE (n:{label} {{uid: r.uid}}) "
                f"SET n.name = r.name, n.name_lc = r.name_lc, n.stub = false, n += r.props",
                {"rows": chunk})

    stub_rows = [{"uid": s["uid"], "name": s["name"], "name_lc": normalize_name(s["name"]),
                  "doc_id": s["props"].get("doc_id") or ""} for s in stubs]
    for chunk in _chunks(stub_rows):
        graph.query(
            "UNWIND $rows AS r MERGE (n:Paper {uid: r.uid}) "
            "ON CREATE SET n.name = r.name, n.name_lc = r.name_lc, n.stub = true, n.doc_id = r.doc_id",
            {"rows": chunk})

    by_rel = defaultdict(list)
    for e in edges:
        by_rel[(e["from_label"], e["rel"], e["to_label"])].append(
            {"from": e["from"], "to": e["to"], "evidence": e["props"].get("evidence", ""),
             "mentions": e["props"].get("mentions", 1), "doc_id": e["props"]["doc_id"],
             "value": e["props"].get("value")})
    for (fl, rel, tl), rows in by_rel.items():
        for chunk in _chunks(rows):
            graph.query(
                f"UNWIND $rows AS r MATCH (a:{fl} {{uid: r.from}}) MATCH (b:{tl} {{uid: r.to}}) "
                f"MERGE (a)-[e:{rel}]->(b) SET e.evidence = r.evidence, e.mentions = r.mentions, e.doc_id = r.doc_id, "
                f"e.value = r.value",
                {"rows": chunk})
    return {"nodes": len(nodes), "stubs": len(stub_rows), "edges": len(edges)}
