"""
Thin wrapper around the OpenAI-compatible API (chat + embeddings).
Also keeps a running total of the money spent, so the web app can show it.
"""

import json
import re
import requests
from config import API_BASE_URL, API_KEY, LLM_MODEL, EMBEDDING_MODEL

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Running cost counter (USD) for the whole server session
SPENT = {"usd": 0.0, "calls": 0}


def _track(usage: dict) -> float:
    cost = float((usage or {}).get("cost") or 0.0)
    SPENT["usd"] += cost
    SPENT["calls"] += 1
    return cost


def chat(messages: list, tools: list = None, temperature: float = 0.0) -> dict:
    """One chat completion call. Returns {"message": ..., "cost": ...}."""
    payload = {"model": LLM_MODEL, "messages": messages, "temperature": temperature}
    if tools:
        payload["tools"] = tools
    for attempt in range(2):                           # one retry on network timeouts
        try:
            res = requests.post(f"{API_BASE_URL}/chat/completions", headers=HEADERS, json=payload, timeout=60)
            break
        except requests.exceptions.Timeout:
            if attempt == 1:
                raise
    res.raise_for_status()
    data = res.json()
    return {"message": data["choices"][0]["message"], "cost": _track(data.get("usage"))}


def chat_json(system: str, user: str) -> dict:
    """Chat call that expects a JSON object back. Returns {"data": dict, "raw": str, "cost": float}."""
    out = chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
    raw = out["message"].get("content") or ""
    return {"data": parse_json(raw), "raw": raw, "cost": out["cost"]}


def parse_json(text: str) -> dict:
    """Parse JSON even if the model wrapped it in ```json fences or added text around it."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group(0)) if match else {}


def embed(texts: list) -> tuple:
    """Embed a list of texts. Returns (vectors, cost)."""
    res = requests.post(f"{API_BASE_URL}/embeddings", headers=HEADERS,
                        json={"model": EMBEDDING_MODEL, "input": texts}, timeout=90)
    res.raise_for_status()
    data = res.json()
    vectors = [item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"])]
    return vectors, _track(data.get("usage"))
