"""
Trace = the step-by-step log of one request.
Every component (coordinator, sub-agents, tools) appends a step, so we can always see
WHICH sub-agent and WHICH tool ran, with its input, output, time and cost.
"""

import json
import logging
import time
from datetime import datetime
from config import LOGS_DIR

logger = logging.getLogger("cineagent")
LOG_FILE = LOGS_DIR / "requests.jsonl"


class Trace:
    def __init__(self, question: str):
        self.question = question
        self.steps = []
        self.started = time.time()

    def add(self, agent: str, kind: str, title: str, input=None, output=None, ok=True, cost=0.0, started=None):
        """Record one step. `agent` = who ran it, `kind` = router | llm | retrieval | sql | tool | answer."""
        step = {
            "n": len(self.steps) + 1, "agent": agent, "kind": kind, "title": title,
            "input": input, "output": output, "ok": ok, "cost": round(cost, 6),
            "ms": int((time.time() - started) * 1000) if started else 0,
        }
        self.steps.append(step)
        logger.info(f"[{agent}] {kind}: {title} ({'ok' if ok else 'FAILED'})")
        return step

    def summary(self) -> dict:
        return {
            "agents": sorted({s["agent"] for s in self.steps}),
            "tools": [s["title"] for s in self.steps if s["kind"] == "tool"],
            "cost": round(sum(s["cost"] for s in self.steps), 6),
            "ms": int((time.time() - self.started) * 1000),
        }

    def save(self, result: dict) -> None:
        """Append the request to logs/requests.jsonl."""
        record = {"time": datetime.now().isoformat(timespec="seconds"), "question": self.question,
                  "route": result.get("route"), "answer": result.get("answer"), **self.summary(),
                  "steps": [{k: s[k] for k in ("n", "agent", "kind", "title", "ok")} for s in self.steps]}
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
