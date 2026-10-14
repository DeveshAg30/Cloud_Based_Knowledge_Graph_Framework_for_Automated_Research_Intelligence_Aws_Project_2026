"""Benchmark GraphRAG (knowledge graph + LLM) against the same LLM answering alone.

Questions are generated from the graph itself, so every answer has a verifiable ground truth:
  * paper_authors     "Who are the authors of the paper '<title>'?"      truth: AUTHORED edges
  * method_papers     "Which papers use <method>?"                          truth: USES_METHOD edges
  * task_methods      "Which methods have been applied to <task>?"         truth: APPLIED_TO edges
  * unanswerable      questions about entities that do not exist           truth: the system should decline

Metrics per system:
  * answer recall        share of ground-truth items mentioned in the answer
  * citation precision   (GraphRAG) share of cited paper ids that really support the answer
  * correct refusal      unanswerable questions answered with "not found" instead of invented facts
  * latency              seconds per question (GraphRAG includes graph retrieval)

  python scripts/evaluate_graphrag.py --per-type 8          # writes results/graphrag_eval.{json,md}

Uses the same env vars as the Lambda: GRAPH_BACKEND/NEO4J_* and LLM_PROVIDER/GEMINI_*.
"""

import argparse
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "backend"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from kg.graph_client import get_graph_client  # noqa: E402
from kg.llm import complete  # noqa: E402
from kg.retrieval import answer_question  # noqa: E402
from kg.schema import normalize_name  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE_ENTITIES = ["the QuantumFrobnicator embedding", "the Zelkova-9 benchmark", "HyperLattice Transformer v7",
                 "the Morvane reasoning dataset", "the Pellucid graph distillation method", "the Kestrel-X retrieval model"]
BASELINE_SYSTEM = ("You are a research assistant. Answer the question about scientific literature as accurately as you can. "
                   "If you do not know, say you don't know. Name specific papers, authors or methods when you can.")
REFUSAL = re.compile(r"i'm sorry|couldn't match|could not match|not (?:contain|include|mention|found|aware|able)|no (?:information|facts|"
                     r"mention|record|data|papers?)|don't (?:know|have)|do not (?:know|have)|unable to|cannot (?:find|answer|"
                     r"determine)|there is no|isn't (?:mentioned|in)|unknown|not familiar|no such", re.I)


# ------------------------------------------------------------------ question generation
def build_questions(graph, per_type, seed):
    rng = random.Random(seed)
    qs = []

    papers = graph.query(
        "MATCH (a:Author)-[:AUTHORED]->(p:Paper) WHERE p.stub = false WITH p, collect(a.name) AS authors "
        "WHERE size(authors) >= 2 AND size(authors) <= 6 RETURN p.name AS title, p.doc_id AS doc, authors")
    papers = [p for p in papers if 25 <= len(p["title"]) <= 140]
    for p in rng.sample(papers, min(per_type, len(papers))):
        qs.append({"type": "paper_authors", "question": f"Who are the authors of the paper '{p['title']}'?",
                   "truth": p["authors"], "truth_docs": [p["doc"]], "match": "surname"})

    methods = graph.query(
        "MATCH (p:Paper)-[:USES_METHOD]->(m:Method) WITH m, collect(DISTINCT {t: p.name, d: p.doc_id}) AS ps "
        "WHERE size(ps) >= 3 AND size(ps) <= 8 RETURN m.name AS method, ps")
    for m in rng.sample(methods, min(per_type, len(methods))):
        qs.append({"type": "method_papers", "question": f"Which papers use {m['method']}?",
                   "truth": [x["t"] for x in m["ps"]], "truth_docs": [x["d"] for x in m["ps"]], "match": "title"})

    tasks = graph.query(
        "MATCH (m:Method)-[r:APPLIED_TO]->(t:Task) WITH t, collect(DISTINCT m.name) AS ms, collect(DISTINCT r.doc_id) AS docs "
        "WHERE size(ms) >= 2 AND size(ms) <= 8 AND COUNT { (:Paper)-[:ADDRESSES_TASK]->(t) } >= 2 "
        "RETURN t.name AS task, ms, docs")
    for t in rng.sample(tasks, min(per_type, len(tasks))):
        qs.append({"type": "task_methods", "question": f"Which methods have been applied to {t['task']}?",
                   "truth": t["ms"], "truth_docs": t["docs"], "match": "name"})

    for fake in FAKE_ENTITIES[:max(2, per_type // 2)]:
        qs.append({"type": "unanswerable", "question": f"Which papers evaluate {fake}, and what results did they report?",
                   "truth": [], "truth_docs": [], "match": "refusal"})
    return qs


# ------------------------------------------------------------------ scoring
def mentioned(item, answer, mode):
    a = normalize_name(answer)
    if mode == "surname":
        surname = normalize_name(item).split()[-1] if item.strip() else ""
        return len(surname) > 1 and re.search(rf"\b{re.escape(surname)}\b", a) is not None
    if mode == "title":  # titles get paraphrased/shortened: require the first 5 significant words
        words = [w for w in re.findall(r"[a-z0-9]+", normalize_name(item)) if len(w) > 2][:5]
        return bool(words) and all(w in a for w in words)
    # Names: ignore hyphen/dash variants and spacing ("Multi ‐ Agent" == "Multi-Agent", "Kolmogorov Arnold" == "Kolmogorov-Arnold").
    squash = lambda s: re.sub(r"[^a-z0-9]", "", normalize_name(s))
    return squash(item) in squash(answer)


def score(q, answer, cited):
    answer = answer.replace("’", "'")  # curly apostrophes ("don’t") must still match the refusal patterns
    if q["type"] == "unanswerable":
        return {"recall": None, "refused": bool(REFUSAL.search(answer))}
    hits = [t for t in q["truth"] if mentioned(t, answer, q["match"])]
    # A cited paper id counts as a hit too: the answer pointed at the right source even if it shortened the title.
    if q["match"] == "title":
        hits = [t for t, d in zip(q["truth"], q["truth_docs"]) if t in hits or d in cited]
    out = {"recall": len(hits) / len(q["truth"]), "refused": bool(REFUSAL.search(answer)) and not hits}
    if cited:
        out["citation_precision"] = sum(1 for c in cited if c in set(q["truth_docs"])) / len(cited)
    return out


# ------------------------------------------------------------------ run
def run(args):
    graph = get_graph_client()
    questions = build_questions(graph, args.per_type, args.seed)
    print(f"{len(questions)} questions: " + ", ".join(f"{t}={sum(q['type'] == t for q in questions)}"
                                                       for t in ("paper_authors", "method_papers", "task_methods", "unanswerable")))
    results = []
    if args.retrieval_only:
        # No LLM: score the retrieved facts themselves (context recall / context precision).
        from kg.llm import LLMDisabled

        def no_llm(prompt):
            raise LLMDisabled()

        for i, q in enumerate(questions, 1):
            t = time.time()
            g = answer_question(graph, q["question"], generate=no_llm)
            context = "\n".join(f"{f['text']} [doc:{f.get('doc_id')}]" for f in g["facts"])
            docs = {f["doc_id"] for f in g["facts"] if f.get("doc_id")}
            s = score(q, context if g["facts"] else "no facts found", docs)
            results.append({**q, "retrieval": {"seconds": round(time.time() - t, 2), "facts": len(g["facts"]),
                                               "seeds": [x["name"] for x in g["seeds"]], **s}})
            print(f"[{i:>2}/{len(questions)}] {q['type']:<14} context recall={s.get('recall')} facts={len(g['facts'])}")
        return results
    for i, q in enumerate(questions, 1):
        row = {**q}
        t = time.time()
        try:
            g = answer_question(graph, q["question"])
            row["graphrag"] = {"answer": g["answer"], "citations": g.get("citations", []), "seconds": round(time.time() - t, 2),
                               "facts": len(g["facts"]), **score(q, g["answer"], set(g.get("citations", [])))}
        except Exception as e:
            row["graphrag"] = {"error": f"{type(e).__name__}: {e}"}
        time.sleep(args.pause)
        t = time.time()
        try:
            b = complete(q["question"], system=BASELINE_SYSTEM, max_tokens=800)
            row["baseline"] = {"answer": b, "seconds": round(time.time() - t, 2), **score(q, b, set())}
        except Exception as e:
            row["baseline"] = {"error": f"{type(e).__name__}: {e}"}
        time.sleep(args.pause)
        gr, bl = row["graphrag"], row["baseline"]
        print(f"[{i:>2}/{len(questions)}] {q['type']:<14} graphrag recall={gr.get('recall')} refused={gr.get('refused')} | "
              f"baseline recall={bl.get('recall')} refused={bl.get('refused')}")
        results.append(row)
    return results


def summarize(results):
    def avg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {}
    for system in ("graphrag", "baseline"):
        rows = [r for r in results if "error" not in r[system]]
        answerable = [r for r in rows if r["type"] != "unanswerable"]
        unanswerable = [r for r in rows if r["type"] == "unanswerable"]
        s = {
            "questions": len(rows),
            "errors": sum("error" in r[system] for r in results),
            "answer_recall": avg([r[system]["recall"] for r in answerable]),
            "fully_correct": avg([1.0 if r[system]["recall"] == 1.0 else 0.0 for r in answerable]),
            # Hallucination: an answer that commits to specifics (no "I don't know") but gets most of them wrong.
            "confidently_wrong": avg([1.0 if not r[system]["refused"] and r[system]["recall"] < 0.5 else 0.0 for r in answerable]),
            "said_dont_know": avg([1.0 if r[system]["refused"] else 0.0 for r in answerable]),
            "correct_refusal_on_unanswerable": avg([1.0 if r[system]["refused"] else 0.0 for r in unanswerable]),
            "avg_latency_s": avg([r[system]["seconds"] for r in rows]),
            "by_type": {t: avg([r[system]["recall"] for r in answerable if r["type"] == t])
                        for t in ("paper_authors", "method_papers", "task_methods")},
        }
        if system == "graphrag":
            s["citation_precision"] = avg([r[system].get("citation_precision") for r in answerable])
            s["answers_with_citations"] = avg([1.0 if r[system].get("citations") else 0.0 for r in answerable])
        summary[system] = s
    return summary


def llm_name():
    provider = os.environ.get("LLM_PROVIDER", "gemini")
    if provider == "groq":
        return f"Groq {os.environ.get('GROQ_MODEL', 'openai/gpt-oss-120b')}"
    return f"Gemini {os.environ.get('GEMINI_MODEL', 'gemini-flash-lite-latest')}"


def write_report(results, summary, path_md):
    g, b = summary["graphrag"], summary["baseline"]
    pct = lambda v: "n/a" if v is None else f"{v * 100:.0f}%"
    lines = [
        "# GraphRAG evaluation",
        "",
        f"{g['questions']} questions were generated automatically from the knowledge graph, so every answer has a "
        f"verifiable ground truth. Each was answered twice with the same LLM ({llm_name()}): **GraphRAG** "
        "(graph retrieval + grounded generation, the system in this project) and **LLM only** (the same model with no graph).",
        "",
        "| Metric | GraphRAG | LLM only |",
        "|---|---|---|",
        f"| Answer recall (share of ground-truth items in the answer) | **{pct(g['answer_recall'])}** | {pct(b['answer_recall'])} |",
        f"| Fully correct answers (all items found) | **{pct(g['fully_correct'])}** | {pct(b['fully_correct'])} |",
        f"| **Hallucinated: confident answer that is mostly wrong** | **{pct(g['confidently_wrong'])}** | {pct(b['confidently_wrong'])} |",
        f"| Said \"I don't know\" on answerable questions | {pct(g['said_dont_know'])} | {pct(b['said_dont_know'])} |",
        f"| Correct \"not found\" on questions about non-existent entities | **{pct(g['correct_refusal_on_unanswerable'])}** | "
        f"{pct(b['correct_refusal_on_unanswerable'])} |",
        f"| Answers with source citations | **{pct(g['answers_with_citations'])}** | 0% (no sources available) |",
        f"| Citation precision (cited paper really supports the answer) | **{pct(g['citation_precision'])}** | n/a |",
        f"| Average latency per question | {g['avg_latency_s']} s | {b['avg_latency_s']} s |",
        "",
        "## Recall by question type",
        "",
        "| Question type | GraphRAG | LLM only |",
        "|---|---|---|",
    ]
    labels = {"paper_authors": "Who are the authors of <paper>?", "method_papers": "Which papers use <method>?",
              "task_methods": "Which methods have been applied to <task>?"}
    for t, label in labels.items():
        lines.append(f"| {label} | {pct(g['by_type'][t])} | {pct(b['by_type'][t])} |")
    lines += ["", "## Examples", ""]
    hallucinated = [x for x in results if x["type"] == "paper_authors" and not x["baseline"].get("refused")
                    and x["baseline"].get("recall", 1) < 0.5]
    for r in hallucinated[:1] + [x for x in results if x["type"] == "unanswerable"][:1]:
        truth = f"*Ground truth:* {', '.join(r['truth'])}" if r["truth"] else "*Ground truth:* no such entity exists"
        lines += [f"**Q: {r['question']}**", "", truth, "",
                  f"*GraphRAG:* {r['graphrag'].get('answer', r['graphrag'].get('error', ''))[:600]}", "",
                  f"*LLM only:* {r['baseline'].get('answer', r['baseline'].get('error', ''))[:600]}", ""]
    lines += ["## Method", "",
              "- Ground truth comes from graph edges (AUTHORED, USES_METHOD, APPLIED_TO). Unanswerable questions name made-up entities.",
              "- An item counts as found if the answer names it: author surname, the first five significant words of a paper "
              "title, a cited paper id, or a method name.",
              "- Refusal = the answer says the information isn't available, instead of describing the made-up entity.",
              "- Scores reflect retrieval and grounding. The LLM-only baseline can't see this corpus, so its low recall on "
              "corpus-specific questions is expected. The key differences are invented answers versus refusals, and the absence of citations.",
              "", "Raw per-question results: `results/graphrag_eval.json`."]
    open(path_md, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def write_retrieval_report(results, out_dir):
    pct = lambda v: "n/a" if v is None else f"{v * 100:.0f}%"
    answerable = [r for r in results if r["type"] != "unanswerable"]
    unanswerable = [r for r in results if r["type"] == "unanswerable"]
    avg = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    by = lambda t: avg([r["retrieval"]["recall"] for r in answerable if r["type"] == t])
    summary = {
        "questions": len(results),
        "context_recall": avg([r["retrieval"]["recall"] for r in answerable]),
        "full_context_recall": avg([1.0 if r["retrieval"]["recall"] == 1.0 else 0.0 for r in answerable]),
        "context_precision_of_sources": avg([r["retrieval"]["citation_precision"] for r in answerable
                                             if "citation_precision" in r["retrieval"]]),
        "unanswerable_with_no_matching_facts": avg([1.0 if r["retrieval"]["facts"] == 0 or r["retrieval"]["refused"] else 0.0
                                                    for r in unanswerable]),
        "avg_retrieval_latency_s": avg([r["retrieval"]["seconds"] for r in results]),
        "by_type": {t: by(t) for t in ("paper_authors", "method_papers", "task_methods")},
    }
    json.dump({"summary": summary, "results": results}, open(os.path.join(out_dir, "graphrag_retrieval_eval.json"), "w",
                                                              encoding="utf-8"), indent=2)
    labels = {"paper_authors": "Who are the authors of <paper>?", "method_papers": "Which papers use <method>?",
              "task_methods": "Which methods have been applied to <task>?"}
    lines = ["# GraphRAG retrieval evaluation (no LLM)", "",
             f"{summary['questions']} questions were generated from the knowledge graph. For each one, GraphRAG's retrieval step "
             "(entity linking, then 1-hop expansion plus up to 3-hop paths) produced the facts that would be given to the "
             "LLM. This measures whether the right evidence reaches the LLM, independent of which LLM writes the answer.", "",
             "| Metric | Value |", "|---|---|",
             f"| Context recall (share of ground-truth items present in the retrieved facts) | **{pct(summary['context_recall'])}** |",
             f"| Questions with all ground-truth items retrieved | **{pct(summary['full_context_recall'])}** |",
             f"| Source precision (retrieved paper ids that support the answer) | {pct(summary['context_precision_of_sources'])} |",
             f"| Questions about non-existent entities with no matching facts | **{pct(summary['unanswerable_with_no_matching_facts'])}** |",
             f"| Average retrieval latency | {summary['avg_retrieval_latency_s']} s |", "",
             "| Question type | Context recall |", "|---|---|"]
    lines += [f"| {labels[t]} | {pct(v)} |" for t, v in summary["by_type"].items()]
    lines += ["", "Source precision is low by design: retrieval returns up to 60 facts around the matched entities, "
              "and the LLM then selects and cites the relevant ones. Context recall is the property that matters at this stage.", "",
              "Raw results: `results/graphrag_retrieval_eval.json`."]
    open(os.path.join(out_dir, "GRAPHRAG_RETRIEVAL_EVAL.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pause", type=float, default=4.0, help="seconds between LLM calls (free-tier rate limits)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--retrieval-only", action="store_true", help="score graph retrieval without calling any LLM")
    ap.add_argument("--rescore", action="store_true", help="re-score saved answers in graphrag_eval.json (no LLM calls)")
    args = ap.parse_args()
    if args.rescore:
        from kg.retrieval import normalize_citations

        path = os.path.join(args.out, "graphrag_eval.json")
        results = json.load(open(path, encoding="utf-8"))["results"]
        for r in results:
            for system in ("graphrag", "baseline"):
                ans = r[system].get("answer")
                if ans is None:
                    continue
                if system == "graphrag":
                    ans = r[system]["answer"] = normalize_citations(ans, [])
                    r[system]["citations"] = sorted(set(re.findall(r"\[doc:([^\]]+)\]", ans)))
                cited = set(r[system].get("citations", []))
                for k in ("recall", "refused", "citation_precision"):
                    r[system].pop(k, None)
                r[system].update(score(r, ans, cited))
        summary = summarize(results)
        json.dump({"summary": summary, "results": results}, open(path, "w", encoding="utf-8"), indent=2)
        write_report(results, summary, os.path.join(args.out, "GRAPHRAG_EVAL.md"))
        print(json.dumps(summary, indent=2))
        return
    results = run(args)
    if args.retrieval_only:
        write_retrieval_report(results, args.out)
        return
    summary = summarize(results)
    os.makedirs(args.out, exist_ok=True)
    json.dump({"summary": summary, "results": results}, open(os.path.join(args.out, "graphrag_eval.json"), "w", encoding="utf-8"), indent=2)
    write_report(results, summary, os.path.join(args.out, "GRAPHRAG_EVAL.md"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
