"""
Calculator tool (real tool calling).
The LLM sends an expression like "round(12 / 36 * 100, 2)" and we evaluate it SAFELY
by walking the Python AST - no eval(), only numbers, + - * / ** % and a few functions.
"""

import ast
import math
import operator

OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
FUNCTIONS = {
    "round": round, "abs": abs, "min": min, "max": max, "sqrt": math.sqrt,
    "sum": lambda *x: sum(x), "avg": lambda *x: sum(x) / len(x),
}

# Tool definition sent to the LLM (OpenAI function-calling format)
CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression. Supports + - * / ** %, parentheses and "
                       "round(x, n), abs, min, max, sqrt, sum(a, b, ...), avg(a, b, ...).",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. round(12 / 36 * 100, 2)"}},
            "required": ["expression"],
        },
    },
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS:
        return FUNCTIONS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"Unsupported expression element: {ast.dump(node)[:60]}")


def calculate(expression: str) -> float:
    """Safely evaluate an arithmetic expression string."""
    if len(expression) > 200:
        raise ValueError("Expression too long.")
    return _eval(ast.parse(expression, mode="eval").body)
