"""LLM access for GraphRAG answers and LLM-based extraction. LLM_PROVIDER selects the back end:

  * ``gemini``         - Google Gemini API free tier (default; works on an AWS Free-plan account)
  * ``groq``           - Groq free tier (openai/gpt-oss-120b, ~1 s answers); Gemini is the backup when both keys are set
  * ``bedrock-nova``   - Amazon Nova on Bedrock via the Converse API (paid plan; billed to AWS credits)
  * ``bedrock-claude`` - Claude on Amazon Bedrock (paid plan + Anthropic use-case approval)
  * ``none``           - no LLM: GraphRAG returns the ranked graph facts without a written summary
"""

import json
import os
import time
import urllib.error
import urllib.request

PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")


class LLMRefusal(Exception):
    pass


class LLMDisabled(Exception):
    pass


def provider() -> str:
    return PROVIDER


# --------------------------------------------------------------------------- Gemini (free tier)
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


GEMINI_FALLBACK = "gemini-flash-lite-latest"  # default model: fast and light
GEMINI_MODELS = ["gemini-flash-lite-latest", "gemini-flash-latest"]


def _gemini_call(model, body, timeout):
    req = urllib.request.Request(
        GEMINI_URL.format(model=model),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"].strip()},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _gemini(prompt, system, max_tokens, timeout=18.0):
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.2,
            # Answers only summarise retrieved facts; low thinking keeps us inside API Gateway's 30 s.
            "thinkingConfig": {"thinkingLevel": os.environ.get("GEMINI_THINKING", "low")},
        },
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    # Try the configured model, then the other Flash tier: on the free tier one of them is often
    # overloaded (503 / stalls) while the other answers in seconds.
    primary = os.environ.get("GEMINI_MODEL", GEMINI_FALLBACK).strip()
    models = [primary] + [m for m in GEMINI_MODELS if m != primary]
    started, data, last_error = time.time(), None, None
    for i, model in enumerate(models):
        remaining = timeout - (time.time() - started)
        if remaining < 3:
            break
        try:
            data = _gemini_call(model, body, timeout=remaining / (len(models) - i))
            break
        except (urllib.error.HTTPError, TimeoutError, urllib.error.URLError, OSError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code not in (429, 500, 503, 504):
                raise
            print(f"Gemini {model} unavailable ({type(e).__name__}); trying next model")
            last_error = e
    if data is None:
        raise last_error or TimeoutError("Gemini did not answer in time")
    candidate = (data.get("candidates") or [{}])[0]
    if candidate.get("finishReason") in ("SAFETY", "PROHIBITED_CONTENT", "RECITATION"):
        raise LLMRefusal("The model declined to answer this request.")
    return "".join(p.get("text", "") for p in (candidate.get("content") or {}).get("parts", []))


# --------------------------------------------------------------------------- Amazon Nova (Bedrock)
_bedrock_runtime = None


def _nova(prompt, system, max_tokens):
    global _bedrock_runtime
    import boto3

    if _bedrock_runtime is None:
        _bedrock_runtime = boto3.client("bedrock-runtime")
    kwargs = {
        "modelId": os.environ.get("NOVA_MODEL_ID", "us.amazon.nova-2-lite-v1:0"),
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.2},
    }
    if system:
        kwargs["system"] = [{"text": system}]
    resp = _bedrock_runtime.converse(**kwargs)
    if resp.get("stopReason") == "content_filtered":
        raise LLMRefusal("The model declined to answer this request.")
    return "".join(block.get("text", "") for block in resp["output"]["message"]["content"])


# --------------------------------------------------------------------------- Claude (Bedrock)
_claude_client = None


def _claude(prompt, system, max_tokens, effort):
    global _claude_client
    from anthropic import AnthropicBedrockMantle

    if _claude_client is None:
        region = os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION", "us-east-1")
        _claude_client = AnthropicBedrockMantle(aws_region=region)
    kwargs = {
        "model": os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-opus-5-5"),
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        # API Gateway caps a synchronous request at ~30 s, so query-time calls default to low effort.
        "output_config": {"effort": effort or os.environ.get("BEDROCK_EFFORT", "low")},
    }
    if system:
        kwargs["system"] = system
    message = _claude_client.messages.create(**kwargs)
    if message.stop_reason == "refusal":
        raise LLMRefusal("The model declined to answer this request.")
    return "".join(block.text for block in message.content if block.type == "text")


# --------------------------------------------------------------------------- Groq (optional free fallback)
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _groq(prompt, system, max_tokens, timeout):
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    body = {"model": model, "messages": messages, "max_tokens": min(max_tokens, 4000), "temperature": 0.2}
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"  # summarising retrieved facts needs little reasoning; keeps answers ~1 s
    req = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['GROQ_API_KEY'].strip()}",
                 "User-Agent": "research-kg/1.0"},
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = json.load(resp)["choices"][0]["message"]["content"] or ""
    except urllib.error.HTTPError as e:
        # 429 = free-tier tokens-per-minute limit; wait once if Groq says it frees up within our time budget.
        wait = float(e.headers.get("retry-after") or 0) if e.code == 429 else 0
        if not wait or wait > timeout - (time.time() - started) - 2:
            raise
        print(f"Groq rate limited; retrying in {wait:.0f}s")
        time.sleep(wait)
        with urllib.request.urlopen(req, timeout=max(2.0, timeout - (time.time() - started))) as resp:
            text = json.load(resp)["choices"][0]["message"]["content"] or ""
    # gpt-oss writes typographic hyphens/dashes ("KG‑CF"); plain ASCII keeps names matchable and searchable.
    return text.translate(str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", " ": " "}))


def complete(prompt: str, system: str | None = None, max_tokens: int = 4000, effort: str | None = None,
             budget: float | None = None) -> str:
    """LLM_BUDGET_S caps total time so the API stays inside API Gateway's 30 s limit (retrieval needs a few s).
    With both free providers configured, the other one is tried when the primary fails or times out."""
    budget = budget or float(os.environ.get("LLM_BUDGET_S", "15"))
    has_groq = bool(os.environ.get("GROQ_API_KEY", "").strip())
    has_gemini = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    started = time.time()
    if PROVIDER == "bedrock-nova":
        return _nova(prompt, system, max_tokens)
    if PROVIDER == "bedrock-claude":
        return _claude(prompt, system, max_tokens, effort)
    # Free providers: try the primary, then the other one if it is configured.
    chain = {"groq": ["groq", "gemini"], "gemini": ["gemini", "groq"]}.get(PROVIDER, [])
    chain = [p for p in chain if (p == "groq" and has_groq) or (p == "gemini" and has_gemini)]
    last_error = None
    for i, name in enumerate(chain):
        remaining = budget - (time.time() - started)
        if remaining < 3:
            break
        share = remaining if i == len(chain) - 1 else remaining * (0.45 if name == "groq" else 0.55)
        try:
            return _groq(prompt, system, max_tokens, share) if name == "groq" else _gemini(prompt, system, max_tokens, share)
        except LLMRefusal:
            raise
        except Exception as e:
            print(f"{name} failed ({type(e).__name__}); trying next provider")
            last_error = e
    if last_error:
        raise last_error
    raise LLMDisabled(f"LLM_PROVIDER is '{PROVIDER}'")
