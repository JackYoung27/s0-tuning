from __future__ import annotations

import math
import re
from typing import Any

from state_peft.common import apply_chat_template_no_thinking


def extract_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        idx = text.rfind("\\boxed ")
        if idx == -1:
            return None
        rest = text[idx + 7:].strip()
        match = re.match(r"(\S+)", rest)
        return match.group(1) if match else None

    start = idx + 7
    depth = 1
    cursor = start
    while cursor < len(text) and depth > 0:
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
        cursor += 1
    if depth == 0:
        return text[start:cursor - 1]
    return None


def normalize_answer(answer: str | None) -> str:
    if answer is None:
        return ""
    normalized = answer.strip().strip("$")
    normalized = re.sub(r"\\text\w*\{([^}]*)\}", r"\1", normalized)
    normalized = re.sub(r"\\(?:d|t)?frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", normalized)
    normalized = re.sub(r"\\sqrt\{([^}]*)\}", r"sqrt(\1)", normalized)
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = re.sub(r"\\[,;!\s]", "", normalized)
    normalized = normalized.replace("\\cdot", "*")
    normalized = normalized.replace("\\times", "*")
    normalized = normalized.replace("\\div", "/")
    normalized = normalized.replace("\\pi", "pi")
    normalized = normalized.replace("\\infty", "inf")
    normalized = normalized.replace("\\", "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def try_float(expression: str) -> float | None:
    try:
        expression = expression.replace("pi", str(math.pi))
        expression = expression.replace("sqrt", "___sqrt___")
        value = eval(
            expression,
            {"__builtins__": {}},
            {"sqrt": math.sqrt, "___sqrt___": math.sqrt, "pi": math.pi, "inf": math.inf},
        )
        return float(value)
    except Exception:
        return None


def check_answer(predicted: str | None, gold: str | None) -> bool:
    if predicted is None or gold is None:
        return False

    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)
    if pred_norm == gold_norm:
        return True
    if pred_norm.replace(" ", "") == gold_norm.replace(" ", ""):
        return True

    pred_float = try_float(pred_norm)
    gold_float = try_float(gold_norm)
    if pred_float is None or gold_float is None:
        return False
    if gold_float == 0:
        return abs(pred_float) < 1e-6
    return abs(pred_float - gold_float) / max(abs(gold_float), 1e-12) < 1e-4


def format_prompt(problem: str, tokenizer: Any) -> str:
    return apply_chat_template_no_thinking(
        tokenizer,
        [{
            "role": "user",
            "content": (
                "Solve this math problem. Show your work step by step. "
                "Put your final answer in \\boxed{}.\n\n"
                f"{problem}"
            ),
        }],
    )
