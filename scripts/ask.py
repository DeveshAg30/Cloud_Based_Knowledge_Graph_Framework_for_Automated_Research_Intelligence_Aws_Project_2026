"""Ask the knowledge graph a question from the command line (GraphRAG through Claude on Bedrock).

  python scripts/ask.py "Which methods are evaluated on WN18RR and what tasks do they address?"
  python scripts/ask.py --context-only "..."   # show retrieved facts without calling the LLM
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "backend"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # non-Latin titles on Windows consoles
from kg.graph_client import get_graph_client  # noqa: E402
from kg.retrieval import answer_question, format_context  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--context-only", action="store_true")
    args = ap.parse_args()

    generate = (lambda prompt: "(LLM call skipped)") if args.context_only else None
    result = answer_question(get_graph_client(), args.question, generate=generate)
    print("Seeds:", ", ".join(f"{s['name']} ({s['label']})" for s in result["seeds"]) or "none")
    print("\nRetrieved facts:\n" + format_context(result["facts"]))
    print("\nAnswer:\n" + result["answer"])


if __name__ == "__main__":
    main()
