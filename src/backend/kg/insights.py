"""Research 360: aggregate profiles of an entity, and multi-hop connections between two entities.

profile(graph, name)  -> {"entity", "stats", "timeline", "sections": [{"title", "items": [{"name", "count"|"value"}]}]}
connect(graph, a, b)  -> shortest paths between the best matches for a and b (up to 5 hops), with source papers.
"""

from concurrent.futures import ThreadPoolExecutor

from .retrieval import link_entities

PAPER_LINKS = "USES_METHOD|USES_DATASET|ADDRESSES_TASK|REPORTS_METRIC|HAS_CONCEPT|PUBLISHED_IN"
TOP = 8


def _top(cypher):
    return f"{cypher} RETURN x.name AS name, count(DISTINCT p) AS count ORDER BY count DESC LIMIT {TOP}"


def _queries(label):
    """(section title, cypher) pairs for each node type. $u is the entity's uid."""
    paper_side = {
        "Methods used alongside": _top(f"MATCH (p:Paper)-[:{PAPER_LINKS}]->(e {{uid: $u}}) MATCH (p)-[:USES_METHOD]->(x) WHERE x <> e"),
        "Tasks addressed": _top(f"MATCH (p:Paper)-[:{PAPER_LINKS}]->(e {{uid: $u}}) MATCH (p)-[:ADDRESSES_TASK]->(x) WHERE x <> e"),
        "Datasets": _top(f"MATCH (p:Paper)-[:{PAPER_LINKS}]->(e {{uid: $u}}) MATCH (p)-[:USES_DATASET]->(x) WHERE x <> e"),
        "Top authors": _top(f"MATCH (p:Paper)-[:{PAPER_LINKS}]->(e {{uid: $u}}) MATCH (x:Author)-[:AUTHORED]->(p)"),
        "Top institutions": _top(f"MATCH (p:Paper)-[:{PAPER_LINKS}]->(e {{uid: $u}}) "
                                 "MATCH (a:Author)-[:AUTHORED]->(p) MATCH (a)-[:AFFILIATED_WITH]->(x)"),
        "Related (model-extracted)": "MATCH (e {uid: $u})-[r:APPLIED_TO|EVALUATED_ON|MEASURED_BY|COMPARED_WITH|IS_A|PART_OF]-(x) "
                                     "RETURN type(r) + ': ' + x.name AS name, count(*) AS count ORDER BY count DESC LIMIT 8",
    }
    if label == "Author":
        return {
            "Co-authors (collaborations)": _top("MATCH (e {uid: $u})-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(x:Author)"),
            "Institutions": "MATCH (e {uid: $u})-[:AFFILIATED_WITH]->(x) RETURN x.name AS name, 1 AS count LIMIT 8",
            "Methods used": _top("MATCH (e {uid: $u})-[:AUTHORED]->(p:Paper)-[:USES_METHOD]->(x)"),
            "Tasks addressed": _top("MATCH (e {uid: $u})-[:AUTHORED]->(p:Paper)-[:ADDRESSES_TASK]->(x)"),
            "Papers": "MATCH (e {uid: $u})-[:AUTHORED]->(x:Paper) RETURN x.name AS name, x.year AS count "
                      "ORDER BY count DESC LIMIT 8",
        }
    if label == "Institution":
        return {
            "Top researchers": _top("MATCH (e {uid: $u})<-[:AFFILIATED_WITH]-(x:Author)-[:AUTHORED]->(p:Paper)"),
            "Collaborating institutions": _top("MATCH (e {uid: $u})<-[:AFFILIATED_WITH]-(:Author)-[:AUTHORED]->(p:Paper)"
                                               "<-[:AUTHORED]-(:Author)-[:AFFILIATED_WITH]->(x:Institution) WHERE x <> e"),
            "Research focus (methods)": _top("MATCH (e {uid: $u})<-[:AFFILIATED_WITH]-(:Author)-[:AUTHORED]->(p:Paper)"
                                             "-[:USES_METHOD]->(x)"),
            "Research focus (tasks)": _top("MATCH (e {uid: $u})<-[:AFFILIATED_WITH]-(:Author)-[:AUTHORED]->(p:Paper)"
                                           "-[:ADDRESSES_TASK]->(x)"),
        }
    if label == "Paper":
        return {
            "Authors": "MATCH (x:Author)-[:AUTHORED]->(e {uid: $u}) RETURN x.name AS name, null AS count LIMIT 20",
            "Methods": "MATCH (e {uid: $u})-[:USES_METHOD]->(x) RETURN x.name AS name, null AS count LIMIT 12",
            "Tasks": "MATCH (e {uid: $u})-[:ADDRESSES_TASK]->(x) RETURN x.name AS name, null AS count LIMIT 12",
            "Datasets": "MATCH (e {uid: $u})-[:USES_DATASET]->(x) RETURN x.name AS name, null AS count LIMIT 12",
            "Reported results": "MATCH (e {uid: $u})-[r:REPORTS_METRIC]->(x) RETURN x.name AS name, r.value AS count LIMIT 12",
            "Cited by (in this graph)": "MATCH (x:Paper)-[:CITES]->(e {uid: $u}) RETURN x.name AS name, x.year AS count LIMIT 8",
        }
    sections = dict(paper_side)
    if label == "Metric":
        sections["Reported values"] = ("MATCH (p:Paper)-[r:REPORTS_METRIC]->(e {uid: $u}) WHERE r.value IS NOT NULL "
                                       "RETURN p.name AS name, r.value AS count LIMIT 8")
    return sections


STATS = {
    "Author": "MATCH (e {uid: $u})-[:AUTHORED]->(p:Paper) RETURN count(p) AS papers, min(p.year) AS first, max(p.year) AS last",
    "Institution": "MATCH (e {uid: $u})<-[:AFFILIATED_WITH]-(a:Author) OPTIONAL MATCH (a)-[:AUTHORED]->(p:Paper) "
                   "RETURN count(DISTINCT a) AS researchers, count(DISTINCT p) AS papers",
    "Paper": "MATCH (e {uid: $u}) OPTIONAL MATCH (e)-[:CITES]->(c) OPTIONAL MATCH (v)-[:PUBLISHED_IN]-(e) "
             "RETURN e.year AS year, count(DISTINCT c) AS references",
}
DEFAULT_STATS = (f"MATCH (p:Paper)-[:{PAPER_LINKS}]->(e {{uid: $u}}) "
                 "RETURN count(DISTINCT p) AS papers, min(p.year) AS first, max(p.year) AS last")
TIMELINE = {
    "Author": "MATCH (e {uid: $u})-[:AUTHORED]->(p:Paper) WHERE p.year IS NOT NULL RETURN p.year AS year, count(p) AS count ORDER BY year",
    "Institution": "MATCH (e {uid: $u})<-[:AFFILIATED_WITH]-(:Author)-[:AUTHORED]->(p:Paper) WHERE p.year IS NOT NULL "
                   "RETURN p.year AS year, count(DISTINCT p) AS count ORDER BY year",
}
DEFAULT_TIMELINE = (f"MATCH (p:Paper)-[:{PAPER_LINKS}]->(e {{uid: $u}}) WHERE p.year IS NOT NULL "
                    "RETURN p.year AS year, count(DISTINCT p) AS count ORDER BY year")


def profile(graph, name: str):
    seeds = link_entities(graph, name)
    if not seeds:
        return {"entity": None}
    e = seeds[0]
    label = e["label"]
    jobs = {"stats": STATS.get(label, DEFAULT_STATS), "timeline": None if label == "Paper" else TIMELINE.get(label, DEFAULT_TIMELINE)}
    jobs.update({f"section:{k}": v for k, v in _queries(label).items()})

    def run(cypher):
        try:
            return graph.query(cypher, {"u": e["uid"]}) if cypher else []
        except Exception as exc:  # one failing aggregate must not hide the rest of the profile
            print(f"insights query failed: {type(exc).__name__}: {exc}")
            return []

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = dict(zip(jobs, pool.map(run, jobs.values())))
    sections = [{"title": k.split(":", 1)[1], "items": [r for r in v if r.get("name")]}
                for k, v in results.items() if k.startswith("section:")]
    return {"entity": {"uid": e["uid"], "label": label, "name": e["name"]},
            "stats": (results["stats"] or [{}])[0],
            "timeline": [r for r in results["timeline"] if r.get("year")],
            "sections": [s for s in sections if s["items"]]}


SHORTEST = ("MATCH (x {uid: $a}), (y {uid: $b}) MATCH p = allShortestPaths((x)-[*..5]-(y)) "
            "RETURN [n IN nodes(p) | {uid: n.uid, label: labels(n)[0], name: n.name}] AS nodes, "
            "[r IN relationships(p) | {rel: type(r), doc_id: r.doc_id, from: startNode(r).uid}] AS rels LIMIT 5")
BOUNDED = ("MATCH p = (x {uid: $a})-[*1..4]-(y {uid: $b}) "
           "RETURN [n IN nodes(p) | {uid: n.uid, label: labels(n)[0], name: n.name}] AS nodes, "
           "[r IN relationships(p) | {rel: type(r), doc_id: r.doc_id, from: startNode(r).uid}] AS rels LIMIT 5")


def connect(graph, a: str, b: str):
    """Multi-hop discovery: how are two entities connected through papers, authors, methods...?"""
    sa, sb = link_entities(graph, a)[:1], link_entities(graph, b)[:1]
    if not sa or not sb:
        missing = a if not sa else b
        return {"error": f"No entity in the graph matches '{missing}'.", "paths": [], "nodes": [], "edges": []}
    a_node, b_node = sa[0], sb[0]
    try:
        rows = graph.query(SHORTEST, {"a": a_node["uid"], "b": b_node["uid"]})
    except Exception:  # backends without allShortestPaths: bounded variable-length search
        rows = graph.query(BOUNDED, {"a": a_node["uid"], "b": b_node["uid"]})
    nodes, edges, paths = {}, {}, []
    for row in rows:
        hops, docs = [], []
        for n in row["nodes"]:
            nodes.setdefault(n["uid"], {"id": n["uid"], "label": n["label"], "name": n["name"],
                                        "seed": n["uid"] in (a_node["uid"], b_node["uid"])})
        for k, r in enumerate(row["rels"]):
            left, right = row["nodes"][k], row["nodes"][k + 1]
            src, dst = (left, right) if r["from"] == left["uid"] else (right, left)
            edges.setdefault((src["uid"], r["rel"], dst["uid"]), {"source": src["uid"], "target": dst["uid"], "rel": r["rel"]})
            hops.append(f"({src['label']}) {src['name']} -[{r['rel']}]-> ({dst['label']}) {dst['name']}")
            if r.get("doc_id") and r["doc_id"] not in docs:
                docs.append(r["doc_id"])
        paths.append({"length": len(row["rels"]), "hops": hops, "docs": docs})
    return {"from": a_node, "to": b_node, "paths": paths, "nodes": list(nodes.values()), "edges": list(edges.values())}
