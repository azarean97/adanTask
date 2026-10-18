"""
Database Sub-Agent (NL2SQL).

  question -> LLM writes SQL (with schema context) -> SQL Guard validates + runs it
           -> if it fails: LLM repairs it once using the error message
           -> LLM writes the answer from the result (can call the calculator tool)
"""

import json
import time
import llm
from config import CURRENT_PROMPTS
from database import schema_context
from sql_guard import run_safe_query, SQLGuardError
from tool_runner import answer_with_tools
from tracelog import Trace

AGENT = "Database Agent"
MAX_REPAIRS = 1


def generate_sql(question: str, trace: Trace) -> str:
    t0 = time.time()
    system = CURRENT_PROMPTS["sql_prompt"].format(schema=schema_context())
    out = llm.chat_json(system, question)
    sql = out["data"].get("sql", "")
    trace.add(AGENT, "llm", "Generate SQL", input=question,
              output={"sql": sql, "explanation": out["data"].get("explanation", "")}, cost=out["cost"], started=t0)
    return sql


def repair_sql(question: str, sql: str, error: str, trace: Trace) -> str:
    t0 = time.time()
    prompt = CURRENT_PROMPTS["sql_repair_prompt"].format(question=question, sql=sql, error=error, schema=schema_context())
    out = llm.chat_json(prompt, "Return the fixed query.")
    fixed = out["data"].get("sql", "")
    trace.add(AGENT, "llm", "Repair SQL", input={"failed_sql": sql, "error": error},
              output={"sql": fixed, "explanation": out["data"].get("explanation", "")}, cost=out["cost"], started=t0)
    return fixed


def execute(sql: str, trace: Trace) -> dict:
    """Run SQL through the guard and log it. Returns the result dict, or {"error", "checks"}."""
    t0 = time.time()
    try:
        result = run_safe_query(sql)
        trace.add(AGENT, "sql", "Validate + execute SQL", input=sql,
                  output={"checks": result["checks"], "row_count": result["row_count"]}, started=t0)
        return result
    except SQLGuardError as e:
        trace.add(AGENT, "sql", "Validate + execute SQL", input=sql,
                  output={"checks": e.checks, "error": str(e)}, ok=False, started=t0)
        return {"sql": sql, "error": str(e), "checks": e.checks}


def run(question: str, trace: Trace, use_tools: bool = True) -> dict:
    sql = generate_sql(question, trace)
    if not sql.strip():                                # model refused (e.g. a write request)
        reason = trace.steps[-1]["output"]["explanation"]
        trace.add(AGENT, "answer", "Refused (read-only)", output=reason, ok=False)
        return {"answer": f"This assistant is read-only and cannot run that request. {reason}"}

    result = execute(sql, trace)

    for _ in range(MAX_REPAIRS):                       # self-correction loop
        if "error" not in result:
            break
        sql = repair_sql(question, sql, result["error"], trace)
        result = execute(sql, trace)

    if "error" in result:
        return {"answer": f"I could not build a valid SQL query for this question. Last error: {result['error']}",
                "sql": result}

    table = json.dumps({"columns": result["columns"], "rows": result["rows"][:50]}, ensure_ascii=False, default=str)
    system = CURRENT_PROMPTS["db_answer_prompt"].format(sql=result["sql"], result=table)
    answer = answer_with_tools(AGENT, system, question, trace, use_tools)
    return {"answer": answer, "sql": result}
