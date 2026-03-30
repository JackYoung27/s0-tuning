from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def apply_chat_template_no_thinking(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def strip_generation_artifacts(text: str, *, remove_code_fences: bool = False) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if remove_code_fences:
        cleaned = re.sub(r"^```python\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
    return cleaned


def prepare_decoder_only_padding(tokenizer: Any) -> None:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_json(path: str | Path, payload: Any) -> None:
    import os as _os

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
        handle.flush()
        _os.fsync(handle.fileno())
    _os.rename(tmp_path, output_path)
