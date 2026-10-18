"""
Database layer.
1. Converts the MySQL dump (imdb_small.sql) into a local SQLite file, using our own clean schema.
2. Builds the "schema context" text that is shown to the LLM:
   tables + columns + meaning, join paths, row counts, sample rows and known category values.
3. Opens READ-ONLY connections for query execution.
"""

import re
import sqlite3
from config import SQL_DUMP_PATH, SQLITE_PATH

# Our schema design: same data as the dump, but with explicit foreign keys and
# human-readable descriptions (the descriptions are given to the LLM).
SCHEMA = {
    "movies": {
        "ddl": "CREATE TABLE movies (id INTEGER PRIMARY KEY, name TEXT, year INTEGER, rank REAL)",
        "about": "One row per movie.",
        "columns": {"id": "movie id", "name": "movie title", "year": "release year",
                    "rank": "IMDB rating from 0 to 10 (NULL when unknown)"},
    },
    "actors": {
        "ddl": "CREATE TABLE actors (id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT, gender TEXT, film_count INTEGER)",
        "about": "One row per actor/actress.",
        "columns": {"id": "actor id", "first_name": "first name", "last_name": "last name",
                    "gender": "'M' or 'F'", "film_count": "number of films of this actor"},
    },
    "directors": {
        "ddl": "CREATE TABLE directors (id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT)",
        "about": "One row per director.",
        "columns": {"id": "director id", "first_name": "first name", "last_name": "last name"},
    },
    "roles": {
        "ddl": "CREATE TABLE roles (actor_id INTEGER REFERENCES actors(id), movie_id INTEGER REFERENCES movies(id), role TEXT)",
        "about": "Which actor played which character in which movie (actor <-> movie link).",
        "columns": {"actor_id": "-> actors.id", "movie_id": "-> movies.id", "role": "character name"},
    },
    "movies_directors": {
        "ddl": "CREATE TABLE movies_directors (director_id INTEGER REFERENCES directors(id), movie_id INTEGER REFERENCES movies(id))",
        "about": "Which director directed which movie (director <-> movie link).",
        "columns": {"director_id": "-> directors.id", "movie_id": "-> movies.id"},
    },
    "movies_genres": {
        "ddl": "CREATE TABLE movies_genres (movie_id INTEGER REFERENCES movies(id), genre TEXT)",
        "about": "Genres of each movie (a movie can have several genres).",
        "columns": {"movie_id": "-> movies.id", "genre": "genre name"},
    },
    "directors_genres": {
        "ddl": "CREATE TABLE directors_genres (director_id INTEGER REFERENCES directors(id), genre TEXT, prob REAL)",
        "about": "How likely a director works in a genre.",
        "columns": {"director_id": "-> directors.id", "genre": "genre name", "prob": "probability 0..1"},
    },
}

JOIN_PATHS = [
    "movies.id = roles.movie_id AND roles.actor_id = actors.id   (actors in a movie)",
    "movies.id = movies_directors.movie_id AND movies_directors.director_id = directors.id   (director of a movie)",
    "movies.id = movies_genres.movie_id   (genres of a movie)",
    "directors.id = directors_genres.director_id   (genres of a director)",
]

# Columns whose distinct values are listed for the model (helps it write correct filters)
CATEGORY_COLUMNS = [("movies_genres", "genre"), ("actors", "gender")]


def _mysql_to_sqlite_values(statement: str) -> str:
    """Convert MySQL string escapes (\\' \\" \\\\) into SQLite-compatible ones."""
    escapes = {"'": "''", '"': '"', "\\": "\\", "n": "\n", "r": "", "t": "\t", "0": ""}
    return re.sub(r"\\(.)", lambda m: escapes.get(m.group(1), m.group(1)), statement)


def build_sqlite(force: bool = False) -> None:
    """Create the SQLite database from the MySQL dump (only once)."""
    if SQLITE_PATH.exists() and not force:
        return
    SQLITE_PATH.unlink(missing_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    for table in SCHEMA.values():
        conn.execute(table["ddl"])
    dump = SQL_DUMP_PATH.read_text(encoding="utf-8", errors="ignore")
    for line in dump.splitlines():
        match = re.match(r"INSERT INTO `(\w+)` VALUES (.*);$", line)
        if match and match.group(1) in SCHEMA:           # actors2 is skipped (duplicate of actors.film_count)
            conn.execute(f"INSERT INTO {match.group(1)} VALUES {_mysql_to_sqlite_values(match.group(2))}")
    conn.commit()
    conn.close()


def connect_readonly() -> sqlite3.Connection:
    """Open the database in read-only mode: SQLite itself refuses any write."""
    return sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True, check_same_thread=False)


def table_overview() -> list:
    """Structured info about every table (used by the web app)."""
    conn = connect_readonly()
    tables = []
    for name, info in SCHEMA.items():
        count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        cur = conn.execute(f"SELECT * FROM {name} LIMIT 3")
        tables.append({
            "name": name, "about": info["about"], "rows": count,
            "columns": [{"name": c, "about": a} for c, a in info["columns"].items()],
            "sample": [list(r) for r in cur.fetchall()],
        })
    conn.close()
    return tables


def schema_context() -> str:
    """The text given to the LLM so it understands the database."""
    conn = connect_readonly()
    lines = []
    for t in table_overview():
        lines.append(f"TABLE {t['name']}  ({t['rows']} rows) -- {t['about']}")
        for c in t["columns"]:
            lines.append(f"  - {c['name']}: {c['about']}")
        lines.append(f"  sample rows: {t['sample'][:2]}")
    lines.append("\nJOIN PATHS:")
    lines += [f"  {j}" for j in JOIN_PATHS]
    lines.append("\nKNOWN VALUES:")
    for table, col in CATEGORY_COLUMNS:
        values = [r[0] for r in conn.execute(f"SELECT DISTINCT {col} FROM {table} ORDER BY 1")]
        lines.append(f"  {table}.{col}: {values}")
    conn.close()
    return "\n".join(lines)
