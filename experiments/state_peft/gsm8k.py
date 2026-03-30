from __future__ import annotations

import re
from typing import Any

from state_peft.common import apply_chat_template_no_thinking


def extract_answer(text: str) -> str | None:
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def check_answer(predicted: str | None, gold: str) -> bool:
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(gold)) < 1e-3
    except (ValueError, TypeError):
        return predicted.strip() == gold.strip()


def format_prompt(question: str, tokenizer: Any) -> str:
    return apply_chat_template_no_thinking(
        tokenizer,
        [{
            "role": "user",
            "content": (
                "Solve this math problem step by step. "
                "End with the answer after ####.\n\n"
                f"{question}"
            ),
        }],
    )
