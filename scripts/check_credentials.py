"""Check that the Neo4j and Gemini credentials work, without printing them.

  # check what is actually deployed (reads the Lambda's environment; nothing is printed)
  python scripts/check_credentials.py --from-lambda research-kg-api-dev

  # or type the values yourself (input is hidden)
  python scripts/check_credentials.py
"""

import argparse
import getpass
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "backend"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def describe(value):
    """Shape of a secret without revealing it: length, and any stray whitespace."""
    if not value:
        return "EMPTY"
    note = " (has leading/trailing spaces!)" if value != value.strip() else ""
    return f"{len(value)} chars, ends with ...{value.strip()[-3:]}{note}"


def check_neo4j(env):
    from neo4j import GraphDatabase

    uri, user, db = env.get("NEO4J_URI", ""), env.get("NEO4J_USER", ""), env.get("NEO4J_DATABASE") or None
    print(f"Neo4j  uri={uri!r} user={user!r} database={db!r} password={describe(env.get('NEO4J_PASSWORD'))}")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, env.get("NEO4J_PASSWORD", "")))
        with driver.session(database=db) as s:
            count = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        driver.close()
        print(f"  OK - connected, graph has {count} nodes")
        return True
    except Exception as e:
        print(f"  FAILED - {type(e).__name__}: {str(e)[:200]}")
        return False


def check_gemini(env):
    key = env.get("GEMINI_API_KEY", "")
    model = env.get("GEMINI_MODEL") or "gemini-flash-lite-latest"
    print(f"Gemini model={model!r} key={describe(key)}")
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps({"contents": [{"parts": [{"text": "Reply with the single word: ok"}]}]}).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key.strip()},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        print(f"  OK - model replied: {text.strip()[:40]!r}")
        return True
    except urllib.error.HTTPError as e:
        print(f"  FAILED - HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
        return False
    except Exception as e:
        print(f"  FAILED - {type(e).__name__}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-lambda", metavar="FUNCTION_NAME", help="test the values deployed in this Lambda")
    args = ap.parse_args()

    if args.from_lambda:
        import boto3

        env = boto3.client("lambda").get_function_configuration(FunctionName=args.from_lambda)["Environment"]["Variables"]
    else:
        env = {
            "NEO4J_URI": input("NEO4J_URI: ").strip(),
            "NEO4J_USER": input("NEO4J_USERNAME: ").strip(),
            "NEO4J_DATABASE": input("NEO4J_DATABASE (Enter to skip): ").strip(),
            "NEO4J_PASSWORD": getpass.getpass("NEO4J_PASSWORD (hidden): "),
            "GEMINI_API_KEY": getpass.getpass("GEMINI_API_KEY (hidden): "),
        }

    ok_graph = check_neo4j(env)
    ok_llm = check_gemini(env) if env.get("LLM_PROVIDER", "gemini") == "gemini" else True
    print("\nAll good." if ok_graph and ok_llm else "\nFix the FAILED item(s) and redeploy.")
    sys.exit(0 if ok_graph and ok_llm else 1)


if __name__ == "__main__":
    main()
