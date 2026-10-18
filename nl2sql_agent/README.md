# CineAgent - NL2SQL + Document RAG + Agentic Router

## Run
See [../README.md](../README.md) (Docker or local). The API key comes from `adan_tmp/.env`.

```bash
.venv/bin/python app.py        # http://localhost:5050
```

## Two parts in the web app
1. **🗄 NL2SQL** - question -> LLM writes SQL -> SQL Guard -> read-only result (no router, no tools).
2. **🤖 Agent & Router** - Coordinator picks the Database or Document Agent; both can call the Calculator tool.

Data lives in `data/` (`imdb_small.sql`, `Cinema-Studies-Key-Concepts-1-289afca.pdf`).

## Flow
```
question -> Coordinator (1 LLM call: database | document, flags write requests)
   database -> Database Agent: LLM writes SQL -> SQL Guard -> run read-only -> (1 repair if error) -> answer
   document -> Document Agent: embed question -> cosine top-4 chunks -> answer with [p. N]
   both agents can call the Calculator tool (real function calling)
```

## Files
| File | Role |
|---|---|
| `config.py` | API, models, paths, editable prompts |
| `database.py` | MySQL dump -> SQLite, schema context given to the LLM |
| `sql_guard.py` | 4 safety layers: static check, EXPLAIN dry run, authorizer, read-only conn + limits |
| `sql_agent.py` | Database sub-agent (NL2SQL + self-repair) |
| `doc_agent.py` | Document sub-agent (simple dense RAG, book pages 1-30) |
| `calculator.py` | Safe AST calculator + tool schema |
| `tool_runner.py` | Tool-calling loop shared by both agents |
| `coordinator.py` | Router / orchestration |
| `tracelog.py` | Per-request trace + `logs/requests.jsonl` |
| `app.py`, `web/` | Flask API + web app |
| `Dockerfile` | container for this project |

## Demo questions
- Filtering: *Which movies were released after 2000 and have a rank above 8?*
- Aggregation: *What is the average rank of movies per genre?*
- Join: *Who directed the movie Aliens?*
- Multi-step (SQL -> Calculator): *What percentage of all movies are Drama movies?*
- Document: *What does auteur theory mean?*
- Safety: *Delete all movies with rank below 5.* (refused) and the **SQL Guard Lab** tab
