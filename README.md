# adan_tmp - two small LLM projects

| Folder | Project | Web app |
|---|---|---|
| [`rag/`](rag/) | **Medical RAG** - KDIGO CKD guideline + UCI patient data: VLM parsing, knowledge graph, BM25 + embeddings + KG retrieval, multi-hop, Actor / Critic | http://localhost:5001 |
| [`nl2sql_agent/`](nl2sql_agent/) | **CineAgent** - (1) NL2SQL on an IMDB database, (2) Agent & Router with a Database Agent, a Document Agent (book RAG) and a Calculator tool | http://localhost:5050 |

Each folder is self-contained: code, data and its own `Dockerfile`.

```
adan_tmp/
├── docker-compose.yml      # runs both projects
├── .env                    # API key (already filled in)
├── rag/                    # project 1 (data: PDF + Chronic_Kidney_Disease/)
└── nl2sql_agent/           # project 2 (data: data/imdb_small.sql + data/Cinema-Studies...pdf)
```

---

## 1. Run with Docker (recommended)

Requirements: [Docker Desktop](https://www.docker.com/products/docker-desktop/) running.

```bash
cd adan_tmp
docker compose up --build
```

Open:
- Medical RAG → http://localhost:5001
- CineAgent → http://localhost:5050

Stop with `Ctrl+C`, or `docker compose down`.

If a port is already in use on your machine:

```bash
RAG_PORT=6001 AGENT_PORT=6050 docker compose up --build
```

---

## 2. Run without Docker

Python 3.10+ is needed.

```bash
# project 1
cd adan_tmp/rag
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python app.py            # http://localhost:5001

# project 2 (in another terminal)
cd adan_tmp/nl2sql_agent
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python app.py            # http://localhost:5050
```

---

## API key

Both apps read `API_BASE_URL` and `API_KEY` from `adan_tmp/.env`:

```
API_BASE_URL=https://api.gapgpt.app/v1
API_KEY=sk-...
```

`.env` is already included and filled in, so nothing needs to be changed to run the projects.

Cached results are already included (RAG pipeline outputs, book embeddings), so starting the apps
costs nothing; only new questions call the API (`gpt-4o-mini`, a fraction of a cent each).
