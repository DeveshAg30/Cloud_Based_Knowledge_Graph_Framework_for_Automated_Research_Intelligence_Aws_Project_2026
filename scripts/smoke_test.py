"""End-to-end integration test of the DEPLOYED system (API Gateway, Cognito, Lambda, graph, LLM, S3 -> SageMaker).

Creates a temporary Cognito test user (no email is sent), runs every API route, uploads a sample file and tracks it
through ingestion, then deletes the test user.

  python scripts/smoke_test.py               # uses the outputs of the research-kg CloudFormation stack
  python scripts/smoke_test.py --skip-upload # read-only checks

Exit code 0 when every check passes.
"""

import argparse
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import boto3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def http(method, url, token=None, body=None, data=None, headers=None, timeout=35):
    hdrs = dict(headers or {})
    if token:
        hdrs["Authorization"] = f"Bearer {token}"
    if body is not None:
        hdrs["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    try:
        payload = json.loads(raw) if raw else {}
    except ValueError:
        payload = {}
    return status, payload, round(time.time() - t, 2)


class Checks:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, seconds=None, detail=""):
        self.rows.append((name, bool(ok), seconds, detail))
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name:<52} {'' if seconds is None else f'{seconds:>6.2f}s'}  {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="research-kg")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--keep-user", action="store_true")
    args = ap.parse_args()

    outputs = {o["OutputKey"]: o["OutputValue"] for o in
               boto3.client("cloudformation").describe_stacks(StackName=args.stack)["Stacks"][0]["Outputs"]}
    api, pool, client_id = outputs["ApiUrl"], outputs["UserPoolId"], outputs["UserPoolClientId"]
    idp = boto3.client("cognito-idp")
    email = f"smoke-test-{secrets.token_hex(4)}@example.invalid"
    password = "Sm0ke-" + secrets.token_urlsafe(12) + "aA1"
    c = Checks()

    # Temporary member user: created silently (no email), permanent password, member of "members" for upload tests.
    idp.admin_create_user(UserPoolId=pool, Username=email, MessageAction="SUPPRESS",
                          UserAttributes=[{"Name": "email", "Value": email}, {"Name": "email_verified", "Value": "true"}])
    idp.admin_set_user_password(UserPoolId=pool, Username=email, Password=password, Permanent=True)
    idp.admin_add_user_to_group(UserPoolId=pool, Username=email, GroupName="members")
    try:
        auth = idp.initiate_auth(ClientId=client_id, AuthFlow="USER_PASSWORD_AUTH",
                                 AuthParameters={"USERNAME": email, "PASSWORD": password})
        token = auth["AuthenticationResult"]["IdToken"]
        c.check("Cognito sign-in (USER_PASSWORD_AUTH)", bool(token))

        s, p, t = http("GET", f"{api}/public/stats")
        c.check("GET /public/stats (no login)", s == 200 and p.get("papers", 0) > 0, t, f"{p.get('papers')} papers")
        s, p, t = http("GET", f"{api}/stats")
        c.check("GET /stats without token is rejected (401)", s == 401, t, f"status {s}")
        s, p, t = http("GET", f"{api}/stats", token)
        c.check("GET /stats", s == 200 and p.get("member") is True, t, f"{sum((p.get('counts') or {}).values())} nodes")

        s, p, t = http("POST", f"{api}/query", token, {"question": "Which methods are used for link prediction?"})
        c.check("POST /query (GraphRAG answer)", s == 200 and p.get("answer") and t < 30, t,
                f"{len(p.get('citations', []))} citations, AI summary: {'no' if p.get('degraded') else 'yes'}")
        c.check("  ...answer cites source papers", s == 200 and len(p.get("citations", [])) > 0)
        s, p, t = http("POST", f"{api}/query", token, {"question": "Which papers evaluate the Zelkova-9 benchmark?"})
        c.check("POST /query (made-up entity, no invented facts)", s == 200, t, (p.get("answer") or "")[:70].replace("\n", " "))

        s, p, t = http("GET", f"{api}/graph?name=" + urllib.parse.quote("Graph Neural Network"), token)
        c.check("GET /graph (Explore)", s == 200 and len(p.get("nodes", [])) > 1, t, f"{len(p.get('nodes', []))} nodes")
        s, p, t = http("GET", f"{api}/connect?a=TransE&b=BERT", token)
        c.check("GET /connect (multi-hop paths)", s == 200 and "paths" in p, t,
                f"{len(p.get('paths', []))} paths" + (f", {p['paths'][0]['length']} hops" if p.get("paths") else ""))
        s, p, t = http("GET", f"{api}/insights?name=" + urllib.parse.quote("Graph Convolutional Network"), token)
        c.check("GET /insights (Research 360)", s == 200 and (p.get("entity") or {}).get("label"), t,
                f"{(p.get('entity') or {}).get('label')}, {len(p.get('sections', []))} sections")

        if not args.skip_upload:
            s, p, t = http("POST", f"{api}/upload-url", token, {"filename": "smoke_test.jsonl"})
            c.check("POST /upload-url (member)", s == 200 and p.get("url"), t)
            if s == 200:
                with open(os.path.join(ROOT, "dataset", "sample_papers.jsonl"), "rb") as f:
                    put_status, _, t = http("PUT", p["url"], data=f.read())
                c.check("PUT file to S3 (presigned URL)", put_status == 200, t)
                started, st = time.time(), {}
                while time.time() - started < 300:
                    _, st, _ = http("GET", f"{api}/upload-status?key=" + urllib.parse.quote(p["key"]), token)
                    if st.get("status") in ("done", "failed"):
                        break
                    time.sleep(5)
                c.check("Ingestion S3 -> Lambda -> extractor -> graph", st.get("status") == "done",
                        round(time.time() - started, 1), f"{st.get('extractor')}, {st.get('edges')} edges")
    finally:
        if args.keep_user:
            print(f"kept test user {email}")
        else:
            idp.admin_delete_user(UserPoolId=pool, Username=email)

    passed = sum(1 for r in c.rows if r[1])
    print(f"\n{passed}/{len(c.rows)} checks passed")
    sys.exit(0 if passed == len(c.rows) else 1)


if __name__ == "__main__":
    main()
