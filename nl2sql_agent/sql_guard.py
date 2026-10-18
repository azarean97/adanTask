"""
SQL Guard: every query goes through 4 simple safety layers before/while running.

  1. Static check   - one statement only, must start with SELECT/WITH, no write/admin keywords.
  2. Dry run        - EXPLAIN QUERY PLAN catches syntax errors and unknown tables/columns without running it.
  3. Authorizer     - SQLite calls us for every action; anything that is not a read is denied.
  4. Read-only conn + limits - file opened with mode=ro, row cap and a time limit.

Each layer adds an entry to `checks`, so the web app can show exactly what passed or failed.
"""

import re
import sqlite3
import time
from config import MAX_RESULT_ROWS, SQL_TIMEOUT_SECONDS
from database import connect_readonly

FORBIDDEN = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "TRUNCATE",
             "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX", "GRANT", "REVOKE"]

# SQLite actions that are allowed (everything else is denied by the authorizer)
ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
if hasattr(sqlite3, "SQLITE_RECURSIVE"):
    ALLOWED_ACTIONS.add(sqlite3.SQLITE_RECURSIVE)


class SQLGuardError(Exception):
    """Raised when a query is rejected. Carries the list of checks done so far."""
    def __init__(self, message, checks):
        super().__init__(message)
        self.checks = checks


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    return re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL).strip()


def static_check(sql: str) -> str:
    """Layer 1: text rules. Returns the cleaned SQL or raises ValueError."""
    clean = _strip_comments(sql).rstrip(";").strip()
    if not clean:
        raise ValueError("Empty query.")
    if ";" in clean:
        raise ValueError("Multiple statements are not allowed.")
    if not re.match(r"^(SELECT|WITH)\b", clean, re.IGNORECASE):
        raise ValueError("Only SELECT / WITH queries are allowed.")
    without_strings = re.sub(r"'(?:[^']|'')*'", "''", clean)     # ignore words inside string literals
    for word in FORBIDDEN:
        if re.search(rf"\b{word}\b", without_strings, re.IGNORECASE):
            raise ValueError(f"Forbidden keyword: {word}")
    return clean


def _authorizer(action, *args):
    """Layer 3: SQLite asks permission for each internal action."""
    return sqlite3.SQLITE_OK if action in ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def run_safe_query(sql: str) -> dict:
    """Validate and execute a query. Returns {"sql", "columns", "rows", "row_count", "truncated", "checks"}."""
    checks = []

    # Layer 1
    try:
        clean = static_check(sql)
        checks.append({"name": "Static check", "ok": True, "detail": "single read-only SELECT statement"})
    except ValueError as e:
        checks.append({"name": "Static check", "ok": False, "detail": str(e)})
        raise SQLGuardError(str(e), checks)

    conn = connect_readonly()
    conn.set_authorizer(_authorizer)
    try:
        # Layer 2
        try:
            conn.execute(f"EXPLAIN QUERY PLAN {clean}")
            checks.append({"name": "Dry run (EXPLAIN)", "ok": True, "detail": "syntax, tables and columns are valid"})
        except sqlite3.Error as e:
            checks.append({"name": "Dry run (EXPLAIN)", "ok": False, "detail": str(e)})
            raise SQLGuardError(str(e), checks)

        # Layer 4: time limit (progress handler aborts long queries)
        deadline = time.time() + SQL_TIMEOUT_SECONDS
        conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10_000)
        try:
            cur = conn.execute(clean)
            rows = cur.fetchmany(MAX_RESULT_ROWS + 1)
        except sqlite3.Error as e:
            layer = "Authorizer + read-only" if "not authorized" in str(e) or "readonly" in str(e) else "Execution"
            checks.append({"name": layer, "ok": False, "detail": str(e)})
            raise SQLGuardError(str(e), checks)

        truncated = len(rows) > MAX_RESULT_ROWS
        rows = [list(r) for r in rows[:MAX_RESULT_ROWS]]
        checks.append({"name": "Authorizer + read-only", "ok": True, "detail": "only read actions were performed"})
        checks.append({"name": "Execution", "ok": True,
                       "detail": f"{len(rows)} rows" + (f" (capped at {MAX_RESULT_ROWS})" if truncated else "")})
        return {"sql": clean, "columns": [d[0] for d in cur.description or []], "rows": rows,
                "row_count": len(rows), "truncated": truncated, "checks": checks}
    finally:
        conn.close()
