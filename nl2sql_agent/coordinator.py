"""
Coordinator (router) - the simplest possible orchestration:

  1. validate the user input
  2. ONE LLM call decides: "database" or "document" (and flags write requests -> refused)
  3. hand the question to that sub-agent
  4. return answer + full trace (which agent and which tools ran)
"""

import time
import llm
import sql_agent
import doc_agent
from config import CURRENT_PROMPTS, MAX_QUESTION_CHARS
from tracelog import Trace

AGENTS = {"database": sql_agent, "document": doc_agent}


def route(question: str, trace: Trace) -> dict:
    t0 = time.time()
    out = llm.chat_json(CURRENT_PROMPTS["router_prompt"], question)
    decision = out["data"]
    if decision.get("agent") not in AGENTS:            # fallback if the model returns something odd
        decision = {"agent": "document", "needs_calculation": False, "reason": "fallback: unclear router output"}
    trace.add("Coordinator", "router", f"Route -> {decision['agent']}", input=question,
              output=decision, cost=out["cost"], started=t0)
    return decision


def validate_question(question: str):
    """Input validation shared by both modes. Returns an error message or None."""
    if not question:
        return "Please type a question."
    if len(question) > MAX_QUESTION_CHARS:
        return f"Question is too long (max {MAX_QUESTION_CHARS} characters)."
    return None


def nl2sql(question: str) -> dict:
    """Mode 1 - plain NL2SQL: no router, no tools. Question -> SQL -> guard -> result."""
    question = (question or "").strip()
    if error := validate_question(question):
        return {"error": error}
    trace = Trace(question)
    try:
        result = sql_agent.run(question, trace, use_tools=False)
    except Exception as e:
        trace.add("Database Agent", "answer", "Error", output=str(e), ok=False)
        result = {"answer": f"Something went wrong: {e}"}
    response = {"question": question, "route": {"agent": "database", "reason": "NL2SQL mode"},
                **result, "trace": trace.steps, "summary": trace.summary()}
    trace.save(response)
    return response


def ask(question: str, force_agent: str = "auto") -> dict:
    """Mode 2 - agentic: coordinator routes to a sub-agent, sub-agents may call the calculator."""
    question = (question or "").strip()
    if error := validate_question(question):
        return {"error": error}
    trace = Trace(question)

    if force_agent in AGENTS:
        decision = {"agent": force_agent, "needs_calculation": None, "reason": "forced by user"}
        trace.add("Coordinator", "router", f"Route -> {force_agent} (manual)", input=question, output=decision)
    else:
        decision = route(question, trace)

    if decision.get("is_write_request"):              # read-only system: refuse before any sub-agent runs
        trace.add("Coordinator", "answer", "Refused: write request", output=decision.get("reason"), ok=False)
        result = {"answer": "This assistant is read-only. Requests that change data (delete, update, insert...) are refused."}
    else:
        try:
            result = AGENTS[decision["agent"]].run(question, trace)
        except Exception as e:                         # API/network problems etc. never crash the app
            trace.add("Coordinator", "answer", "Error", output=str(e), ok=False)
            result = {"answer": f"Something went wrong: {e}"}

    response = {"question": question, "route": decision, **result, "trace": trace.steps, "summary": trace.summary()}
    trace.save(response)
    return response
