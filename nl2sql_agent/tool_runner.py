"""
Tool-calling loop used by both sub-agents for their final answer.

  LLM  --(wants calculator)-->  we run calculator()  --(result)-->  LLM  ...  --> final text

The loop stops when the LLM answers without asking for a tool (max 4 rounds).
"""

import json
import time
import llm
from calculator import CALCULATOR_TOOL, calculate
from tracelog import Trace

MAX_ROUNDS = 4


def answer_with_tools(agent: str, system_prompt: str, question: str, trace: Trace, use_tools: bool = True) -> str:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]

    for _ in range(MAX_ROUNDS):
        t0 = time.time()
        out = llm.chat(messages, tools=[CALCULATOR_TOOL] if use_tools else None)
        msg = out["message"]
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            answer = msg.get("content") or ""
            trace.add(agent, "answer", "Final answer", input=system_prompt, output=answer, cost=out["cost"], started=t0)
            return answer

        trace.add(agent, "llm", "LLM requested a tool call",
                  output=[c["function"] for c in tool_calls], cost=out["cost"], started=t0)
        messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": tool_calls})

        for call in tool_calls:
            t1 = time.time()
            expression = json.loads(call["function"]["arguments"] or "{}").get("expression", "")
            try:
                result, ok = calculate(expression), True
            except Exception as e:
                result, ok = f"error: {e}", False
            trace.add(agent, "tool", "calculator", input=expression, output=result, ok=ok, started=t1)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(result)})

    return "Stopped: too many tool calls."
