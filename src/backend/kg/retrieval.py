"""GraphRAG: question -> seed entities -> multi-hop graph context -> grounded, cited answer.

1. Entity linking   - n-grams of the question are matched against node names (exact, then substring).
2. Expansion        - 1-hop neighbourhood of each seed, plus shortest connecting paths (<=3 hops)
                      between pairs of seeds, which is what surfaces multi-hop / cross-paper links.
3. Scoring          - fact score = seed match score x hop decay x log(1 + mention count).
4. Generation       - Claude only sees the top-scored facts and must cite paper ids for every claim.
"""

import math
import re
import time
from concurrent.futures import ThreadPoolExecutor

from .schema import normalize_name

STOPWORDS = set("""a an the of in on for to and or with by from at as is are was were be been which what who whom
whose how why when where does do did that this these those it its into about between across using use used
papers paper research researchers work works show find list give me tell any all most more than other
related relation relationship connected connect""".split())

MAX_SEEDS = 8
MAX_PATH_SEEDS = 6  # pairwise path searches grow quadratically; 6 seeds -> 15 searches
MAX_FACTS = 60


def candidate_terms(question: str):
    words = [w for w in re.findall(r"[A-Za-z0-9@][\w@.+-]*", question)]
    grams = set()
    for n in range(1, 5):
        for i in range(len(words) - n + 1):
            gram = words[i:i + n]
            if gram[0].lower() in STOPWORDS or gram[-1].lower() in STOPWORDS:
                continue
            grams.add(normalize_name(" ".join(gram)))
    return sorted((g for g in grams if len(g) >= 2), key=len, reverse=True)


def link_entities(graph, question: str):
    # 1. The whole input names one node (e.g. a pasted paper title or an author): use exactly that.
    whole = normalize_name(question)
    if len(whole) >= 3:
        exact = graph.query(
            "MATCH (n) WHERE n.name_lc = $q RETURN n.uid AS uid, labels(n)[0] AS label, n.name AS name LIMIT 3",
            {"q": whole})
        if exact:
            return [{**row, "score": 1.5} for row in exact]

    # 1b. A paper title mentioned inside a longer question ("Who wrote <title>?").
    titled = [r for r in graph.query(
        "MATCH (p:Paper) WHERE $q CONTAINS p.name_lc RETURN p.uid AS uid, 'Paper' AS label, p.name AS name, "
        "p.name_lc AS term", {"q": whole}) if len(r["term"] or "") >= 25]
    if titled:
        best = max(titled, key=lambda r: len(r["term"]))
        return [{"uid": best["uid"], "label": "Paper", "name": best["name"], "score": 1.4}]

    terms = candidate_terms(question)
    if not terms:
        return []
    seeds = {}
    for row in graph.query(
            "MATCH (n) WHERE n.name_lc IN $terms "
            "RETURN n.uid AS uid, labels(n)[0] AS label, n.name AS name, n.name_lc AS term LIMIT 50", {"terms": terms}):
        seeds[row["uid"]] = {**row, "term": row.get("term") or normalize_name(row["name"]), "score": 1.0}
    # 2. Drop fragments of a longer match: "graph" and "convolutional" inside "graph convolutional network".
    matched = [s["term"] for s in seeds.values()]
    seeds = {uid: s for uid, s in seeds.items()
             if not any(s["term"] != t and re.search(rf"\b{re.escape(s['term'])}\b", t) for t in matched)}
    for s in seeds.values():
        s.pop("term", None)
    if len(seeds) < 3:
        for term in [t for t in terms if len(t) >= 4][:6]:
            for row in graph.query(
                    "MATCH (n) WHERE n.name_lc CONTAINS $t AND n.stub = false "
                    "RETURN n.uid AS uid, labels(n)[0] AS label, n.name AS name LIMIT 10", {"t": term}):
                score = 0.6 * min(1.0, len(term) / max(len(row["name"]), 1))
                if row["uid"] not in seeds or seeds[row["uid"]]["score"] < score:
                    seeds[row["uid"]] = {**row, "score": score}
    return sorted(seeds.values(), key=lambda s: s["score"], reverse=True)[:MAX_SEEDS]


NEIGHBOUR_QUERIES = [
    ("out", "MATCH (s {uid: $uid})-[r]->(n) RETURN type(r) AS rel, n.uid AS uid, labels(n)[0] AS label, "
            "n.name AS name, n.year AS year, n.doc_id AS doc_id, r.evidence AS evidence, "
            "r.doc_id AS edge_doc, r.mentions AS mentions LIMIT 40"),
    ("in", "MATCH (s {uid: $uid})<-[r]-(n) RETURN type(r) AS rel, n.uid AS uid, labels(n)[0] AS label, "
           "n.name AS name, n.year AS year, n.doc_id AS doc_id, r.evidence AS evidence, "
           "r.doc_id AS edge_doc, r.mentions AS mentions LIMIT 40"),
]

PATH_QUERY = (
    "MATCH p = (a {uid: $a})-[*1..3]-(b {uid: $b}) "
    "RETURN [n IN nodes(p) | {uid: n.uid, label: labels(n)[0], name: n.name}] AS nodes, "
    "[r IN relationships(p) | {rel: type(r), doc_id: r.doc_id, from: startNode(r).uid}] AS rels LIMIT 3"
)


# How informative each relation is as evidence; citations and concepts are plentiful but low-signal.
REL_WEIGHT = {"AUTHORED": 1.5, "USES_METHOD": 1.3, "USES_DATASET": 1.3, "ADDRESSES_TASK": 1.3, "REPORTS_METRIC": 1.3,
              "APPLIED_TO": 1.3, "EVALUATED_ON": 1.3, "MEASURED_BY": 1.3, "AFFILIATED_WITH": 1.2, "PUBLISHED_IN": 1.0,
              "HAS_CONCEPT": 0.8, "CITES": 0.6}
# Words in the question that make a relation the likely answer.
QUESTION_BOOSTS = [
    (r"authors?|who|wr[io]te|written|researchers?", ("AUTHORED",), 2.5),
    (r"institution|universit|affiliat|lab", ("AFFILIATED_WITH",), 2.5),
    (r"datasets?|benchmarks?|evaluated on|corpus|corpora", ("USES_DATASET", "EVALUATED_ON"), 2.0),
    (r"methods?|models?|techniques?|approach|algorithms?|architectures?", ("USES_METHOD", "APPLIED_TO"), 1.5),
    (r"tasks?|problems?|used for|applied|applications?", ("ADDRESSES_TASK", "APPLIED_TO"), 1.5),
    (r"metrics?|measured?|accuracy|score|f1|mrr|performance", ("REPORTS_METRIC", "MEASURED_BY"), 2.0),
    (r"cites?|citations?|cited|references?|builds? on", ("CITES",), 3.0),
    (r"venue|journal|conference|published", ("PUBLISHED_IN",), 2.5),
    (r"concepts?|topics?|fields?|areas?", ("HAS_CONCEPT",), 2.0),
]


def relation_weights(question: str = ""):
    weights = dict(REL_WEIGHT)
    for pattern, rels, factor in QUESTION_BOOSTS:
        if re.search(pattern, question or "", re.I):
            for rel in rels:
                weights[rel] = weights.get(rel, 1.0) * factor
    return weights


def expand(graph, seeds, question: str = ""):
    rel_weight = relation_weights(question)
    facts, nodes, edges = [], {}, {}

    def add_node(uid, label, name):
        nodes.setdefault(uid, {"id": uid, "label": label, "name": name, "seed": False})

    def add_edge(src, rel, dst):
        edges.setdefault((src, rel, dst), {"source": src, "target": dst, "rel": rel})

    # The graph queries are independent, so run them concurrently (one network round trip each);
    # results are then merged in a fixed order so answers stay deterministic.
    def safe_query(cypher, params):
        try:
            return graph.query(cypher, params)
        except Exception:  # variable-length paths can time out on very dense graphs; skip them
            return []

    neighbour_jobs = [(s, d, c) for s in seeds for d, c in NEIGHBOUR_QUERIES]
    path_seeds = seeds[:MAX_PATH_SEEDS]
    path_jobs = [(a, b) for i, a in enumerate(path_seeds) for b in path_seeds[i + 1:]]
    with ThreadPoolExecutor(max_workers=12) as pool:
        neighbour_rows = list(pool.map(lambda j: safe_query(j[2], {"uid": j[0]["uid"]}), neighbour_jobs))
        path_rows = list(pool.map(lambda j: safe_query(PATH_QUERY, {"a": j[0]["uid"], "b": j[1]["uid"]}), path_jobs))

    for seed in seeds:
        add_node(seed["uid"], seed["label"], seed["name"])
        nodes[seed["uid"]]["seed"] = True
    for (seed, direction, _), rows in zip(neighbour_jobs, neighbour_rows):
        for row in rows:
            add_node(row["uid"], row["label"], row["name"])
            src, dst = (seed, row) if direction == "out" else (row, seed)
            add_edge(src["uid"], row["rel"], dst["uid"])
            weight = (math.log(1 + (row.get("mentions") or 1)) + 0.5) * rel_weight.get(row["rel"], 1.0)
            facts.append({
                "score": seed["score"] * weight,
                "text": f"({src['label']}) {src['name']} -[{row['rel']}]-> ({dst['label']}) {dst['name']}",
                "doc_id": row.get("edge_doc") or row.get("doc_id"),
                "evidence": row.get("evidence") or "",
            })

    for (a, b), rows in zip(path_jobs, path_rows):
        for row in rows:
            for n in row["nodes"]:
                add_node(n["uid"], n["label"], n["name"])
            hops = []
            for k, r in enumerate(row["rels"]):
                left, right = row["nodes"][k], row["nodes"][k + 1]
                src, dst = (left, right) if r["from"] == left["uid"] else (right, left)
                add_edge(src["uid"], r["rel"], dst["uid"])
                hops.append(f"({src['label']}) {src['name']} -[{r['rel']}]-> ({dst['label']}) {dst['name']}")
            facts.append({
                "score": (a["score"] + b["score"]) * 0.75,
                "text": "PATH: " + " ; ".join(hops),
                "doc_id": next((r["doc_id"] for r in row["rels"] if r.get("doc_id")), None),
                "evidence": "",
            })

    unique = {}
    for f in sorted(facts, key=lambda f: f["score"], reverse=True):
        unique.setdefault(f["text"], f)
    return list(unique.values())[:MAX_FACTS], list(nodes.values()), list(edges.values())


SYSTEM_PROMPT = """You are a research-intelligence assistant answering questions from a scientific knowledge graph.

You will receive numbered FACTS retrieved from the graph. Each fact may carry a paper id and an evidence sentence.
- Answer only from the FACTS. If they do not contain the answer, say what is missing instead of guessing.
- Cite supporting papers inline exactly as [doc:<id>], copying the id shown in the fact (for example [doc:W2984902757]).
  Never cite fact numbers like F3.
- When a PATH fact connects entities across papers, explain the chain step by step.
- Be concise: a short paragraph or a few bullet points. Do not use tables."""


# Facts sent to the LLM. Free tiers cap tokens per minute (Groq: 8k), so the prompt keeps the 30 best-scored
# facts with short evidence; the dashboard still shows all retrieved facts.
LLM_FACTS = 30
EVIDENCE_CHARS = 160


def format_context(facts, evidence_chars=300):
    lines = []
    for i, f in enumerate(facts, 1):
        line = f"F{i}. {f['text']}"
        if f.get("doc_id"):
            line += f"  [doc:{f['doc_id']}]"
        if f.get("evidence"):
            line += f"\n    evidence: {f['evidence'][:evidence_chars]}"
        lines.append(line)
    return "\n".join(lines)


def normalize_citations(answer, facts):
    """Some models cite facts by number ("【F26】", "[F26]", "(F3, F7)") instead of [doc:id]; map them to paper ids."""
    def repl(match):
        ids = []
        for n in re.findall(r"F(\d+)", match.group(0)):
            k = int(n) - 1
            if 0 <= k < len(facts) and facts[k].get("doc_id") and facts[k]["doc_id"] not in ids:
                ids.append(facts[k]["doc_id"])
        return " ".join(f"[doc:{i}]" for i in ids) if ids else ""
    answer = re.sub(r"[【\[(]\s*F\d+(?:\s*[,;–-]\s*F\d+)*\s*[】\])]", repl, answer)
    # "【doc:W123】" / "(doc:W123)" -> "[doc:W123]"
    return re.sub(r"[【(]\s*doc:\s*([^\s】)\]]+)\s*[】)]", r"[doc:\1]", answer)


# API Gateway ends a request at 30 s; leave headroom for the Lambda's own overhead.
REQUEST_BUDGET_S = 22


def answer_question(graph, question: str, generate=None):
    """``generate`` is injectable for tests; defaults to the configured LLM with whatever time is left."""
    started = time.time()
    seeds = link_entities(graph, question)
    if not seeds:
        return {"answer": "I couldn't match any entity in the knowledge graph to this question. "
                          "Try naming a method, dataset, task, author or paper title.",
                "seeds": [], "facts": [], "graph": {"nodes": [], "edges": []}}
    facts, nodes, edges = expand(graph, seeds, question)
    llm_facts = facts[:LLM_FACTS]
    prompt = f"FACTS:\n{format_context(llm_facts, EVIDENCE_CHARS)}\n\nQUESTION: {question}"
    if generate is None:
        from .llm import complete

        def generate(p):
            remaining = REQUEST_BUDGET_S - (time.time() - started)
            return complete(p, system=SYSTEM_PROMPT, max_tokens=900, budget=max(4.0, remaining))
    from .llm import LLMDisabled, LLMRefusal

    top = "\n".join(f"- {f['text']}" + (f" [doc:{f['doc_id']}]" if f.get("doc_id") else "") for f in facts[:10])
    degraded = False
    try:
        answer = generate(prompt)
    except LLMDisabled:
        answer = f"(No LLM configured: showing the most relevant graph facts.)\n{top}"
    except LLMRefusal:
        raise
    except Exception as e:  # timeout / overloaded free tier: never fail the request, show the grounded facts
        print(f"LLM unavailable ({type(e).__name__}: {e}); returning ranked facts")
        degraded = True
        answer = ("The AI summary is temporarily unavailable (the language model didn't respond in time). "
                  f"Here are the most relevant facts from the knowledge graph:\n{top}")
    answer = normalize_citations(answer, llm_facts)
    cited = sorted(set(re.findall(r"\[doc:([^\]]+)\]", answer)))
    return {"answer": answer, "citations": cited, "seeds": seeds, "facts": facts, "degraded": degraded,
            "graph": {"nodes": nodes, "edges": edges}}


def neighbourhood(graph, name: str):
    seeds = link_entities(graph, name)[:3]
    _, nodes, edges = expand(graph, seeds) if seeds else ([], [], [])
    return {"seeds": seeds, "nodes": nodes, "edges": edges}
