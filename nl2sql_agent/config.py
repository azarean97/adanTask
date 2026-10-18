"""
CineAgent configuration.
One place for: API settings, model names, data paths and the editable prompts.
Prompts are kept in CURRENT_PROMPTS so the web app can change them at runtime.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_env_file():
    """Read KEY=VALUE lines from a .env file (project folder or adan_tmp/) into os.environ."""
    for env_file in (BASE_DIR / ".env", BASE_DIR.parent / ".env"):
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


_load_env_file()

DATA_DIR = BASE_DIR / "data"                    # imdb_small.sql + the cinema PDF
CACHE_DIR = BASE_DIR / "data_cache"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ---------------- API ----------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.gapgpt.app/v1")
API_KEY = os.getenv("API_KEY", "")                # set in adan_tmp/.env (see .env.example)

# Cheap models on purpose: the whole project must stay well under $1
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")   # same as the main PID project

# ---------------- Data ----------------
SQL_DUMP_PATH = DATA_DIR / "imdb_small.sql"
SQLITE_PATH = CACHE_DIR / "imdb_small.sqlite"
PDF_PATH = DATA_DIR / "Cinema-Studies-Key-Concepts-1-289afca.pdf"

# Only a slice of the 550-page book is indexed (book pages 1-30) to keep embedding cost tiny
PDF_PAGE_START = 21     # 0-based PDF index of book page 1
PDF_PAGE_END = 51       # exclusive

# ---------------- Safety limits ----------------
MAX_QUESTION_CHARS = 500
MAX_RESULT_ROWS = 200
SQL_TIMEOUT_SECONDS = 3

# ---------------- Prompts (editable from the web app) ----------------
DEFAULT_PROMPTS = {
    "router_prompt": """You are the Coordinator of a small assistant with two specialist sub-agents:
- "database": answers questions about the IMDB movie database (movies, years, ranks, genres, actors, roles, directors). Use it for counting, filtering, averages, top-N lists, who acted in / directed what.
- "document": answers conceptual questions from the book "Cinema Studies: The Key Concepts" (film theory terms such as adaptation, animation, auteur theory, avant-garde, art cinema, apparatus, ambiguity).

Pick exactly one sub-agent for the user question.
Set "is_write_request" to true if the user wants to CHANGE data (delete, remove, update, insert, add, drop, rename).
The system is read-only, so such requests will be refused.
Set "needs_calculation" to true if a numeric step (percentage, difference, ratio, average of returned values) is needed after getting the data.

Return only JSON:
{"agent": "database" or "document", "needs_calculation": true or false, "is_write_request": true or false, "reason": "one short sentence"}""",

    "sql_prompt": """You are an expert SQLite analyst. Convert the user question into ONE read-only SQLite query.

Rules:
- Only SELECT (or WITH ... SELECT). Never modify data.
- Use only the tables and columns in the schema below. Follow the listed join paths.
- Text filters: use LIKE with % for names when the exact spelling is unsure.
- Always add LIMIT 50 unless the query returns a single aggregated row.
- "Which / list / show" questions return the rows themselves, not a count.
- When counting movies across a join (genres, roles, directors), use COUNT(DISTINCT movies.id).
- Do NOT compute percentages, ratios or differences in SQL. Return the raw numbers as separate columns
  (e.g. drama_movies, total_movies) - a calculator tool will do the math afterwards.

DATABASE SCHEMA:
{schema}

Return only JSON:
{{"sql": "the SQLite query", "explanation": "one sentence on how the query answers the question"}}""",

    "sql_repair_prompt": """The previous SQLite query failed. Fix it.

Question: {question}
Failed SQL: {sql}
Error: {error}

DATABASE SCHEMA:
{schema}

Return only JSON:
{{"sql": "the corrected SQLite query", "explanation": "what was fixed"}}""",

    "db_answer_prompt": """You are the Database sub-agent. Answer the user question using ONLY the SQL result below.
If a numeric step is needed (percentage, difference, ratio, average), you MUST call the "calculator" tool instead of doing math in your head.
Keep the answer short and mention the key numbers.

SQL that was executed:
{sql}

SQL result (columns + rows):
{result}""",

    "doc_answer_prompt": """You are the Document sub-agent. Answer the user question using ONLY the book excerpts below.
Cite pages like [p. 12]. If the excerpts do not contain the answer, say so clearly.
If a numeric step is needed, call the "calculator" tool.

Book excerpts:
{context}""",
}

CURRENT_PROMPTS = dict(DEFAULT_PROMPTS)
