"""
CineAgent web server (Flask).

  GET  /api/status          models, data sizes, money spent
  POST /api/nl2sql          mode 1: question -> SQL -> guard -> result (no router, no tools)
  POST /api/ask             mode 2: question -> coordinator -> sub-agent (+ calculator) -> answer + trace
  POST /api/sql/check       run any SQL through the SQL Guard (to demo blocked queries)
  GET  /api/schema          tables + the exact schema text the LLM sees
  GET  /api/chunks          indexed document chunks
  GET/POST /api/prompts     read / edit prompts,  POST /api/prompts/reset
  GET  /api/logs            last requests from logs/requests.jsonl

Run:  .venv/bin/python app.py   ->  http://localhost:5050
"""

import os
import json
import logging
from flask import Flask, request, jsonify, send_from_directory

import llm
import database
import doc_agent
import coordinator
from config import CURRENT_PROMPTS, DEFAULT_PROMPTS, LLM_MODEL, EMBEDDING_MODEL, PDF_PAGE_START, PDF_PAGE_END
from sql_guard import run_safe_query, SQLGuardError
from tracelog import LOG_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
app = Flask(__name__, static_folder="web", static_url_path="")

# Prepare data once at startup
database.build_sqlite()
doc_agent.build_index()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/status")
def status():
    return jsonify({
        "models": {"llm": LLM_MODEL, "embedding": EMBEDDING_MODEL},
        "tables": len(database.SCHEMA),
        "chunks": len(doc_agent.CHUNKS),
        "pages": f"1-{PDF_PAGE_END - PDF_PAGE_START}",
        "spent_usd": round(llm.SPENT["usd"], 5),
        "api_calls": llm.SPENT["calls"],
    })


@app.post("/api/ask")
def ask():
    data = request.json or {}
    result = coordinator.ask(data.get("question", ""), data.get("agent", "auto"))
    result["spent_usd"] = round(llm.SPENT["usd"], 5)
    return jsonify(result), (400 if "error" in result else 200)


@app.post("/api/nl2sql")
def nl2sql():
    result = coordinator.nl2sql((request.json or {}).get("question", ""))
    result["spent_usd"] = round(llm.SPENT["usd"], 5)
    return jsonify(result), (400 if "error" in result else 200)


@app.post("/api/sql/check")
def sql_check():
    sql = (request.json or {}).get("sql", "")
    try:
        return jsonify({"ok": True, **run_safe_query(sql)})
    except SQLGuardError as e:
        return jsonify({"ok": False, "error": str(e), "checks": e.checks})


@app.get("/api/schema")
def schema():
    return jsonify({"tables": database.table_overview(), "context": database.schema_context(),
                    "joins": database.JOIN_PATHS})


@app.get("/api/chunks")
def chunks():
    return jsonify([{k: c[k] for k in ("id", "page", "text")} for c in doc_agent.CHUNKS])


@app.route("/api/prompts", methods=["GET", "POST"])
def prompts():
    if request.method == "POST":
        for key, value in (request.json or {}).get("prompts", {}).items():
            if key in CURRENT_PROMPTS:
                CURRENT_PROMPTS[key] = value
    return jsonify({"prompts": CURRENT_PROMPTS, "defaults": DEFAULT_PROMPTS})


@app.post("/api/prompts/reset")
def prompts_reset():
    CURRENT_PROMPTS.update(DEFAULT_PROMPTS)
    return jsonify({"prompts": CURRENT_PROMPTS})


@app.get("/api/logs")
def logs():
    if not LOG_FILE.exists():
        return jsonify([])
    lines = LOG_FILE.read_text(encoding="utf-8").strip().splitlines()[-30:]
    return jsonify([json.loads(line) for line in reversed(lines)])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5050)), debug=False)
