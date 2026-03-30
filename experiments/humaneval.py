from __future__ import annotations

import logging
import re
import subprocess
import sys
import time

from datasets import load_dataset

log = logging.getLogger(__name__)


def load_humaneval() -> list[dict]:
    t0 = time.time()
    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = [dict(row) for row in ds]
    log.info(f"Loaded {len(problems)} HumanEval problems in {time.time() - t0:.1f}s")
    return problems


def format_prompt(problem: dict, tokenizer) -> str:
    instruction = (
        "Complete the following Python function. "
        "Only output the function body, no explanation. No markdown.\n\n"
        + problem["prompt"]
    )
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": instruction}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
    return instruction


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^```python\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text.strip())
    return text.strip()


_WARNED_UNSAFE = False

def execute_completion(problem: dict, completion: str, timeout: float = 10.0) -> bool:
    global _WARNED_UNSAFE
    if not _WARNED_UNSAFE:
        log.warning(
            "SECURITY: executing model-generated code in an unsandboxed subprocess. "
            "Run inside a container (Modal/Docker) for safety."
        )
        _WARNED_UNSAFE = True

    check_program = (
        problem["prompt"]
        + completion
        + "\n"
        + problem["test"]
        + "\n"
        + f"check({problem['entry_point']})"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", check_program],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            log.debug(f"  Execution failed for {problem['task_id']}: {result.stderr[:200]}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log.debug(f"  Execution timed out for {problem['task_id']}")
        return False
    except OSError as e:
        log.debug(f"  Execution OSError for {problem['task_id']}: {e}")
        return False


def generate_completions(
    model,
    tokenizer,
    problem: dict,
    n_samples: int = 16,
    temperature: float = 0.7,
    max_new_tokens: int = 1024,
) -> list[str]:
    import torch

    t0 = time.time()
    prompt = format_prompt(problem, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    log.info(f"  [{problem['task_id']}] Generating {n_samples} samples (prompt_len={prompt_len}, max_new={max_new_tokens}, temp={temperature})")

    inputs = {k: v.expand(n_samples, -1) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_time = time.time() - t0
    completions = []
    gen_lengths = []
    for i in range(n_samples):
        gen_ids = outputs[i][prompt_len:]
        gen_lengths.append(len(gen_ids))
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        completions.append(strip_thinking(text))

    avg_gen_len = sum(gen_lengths) / len(gen_lengths)
    log.info(
        f"  [{problem['task_id']}] Generated {n_samples} samples in {gen_time:.1f}s "
        f"(avg_gen_tokens={avg_gen_len:.0f}, tokens/s={sum(gen_lengths) / gen_time:.0f})"
    )
    return completions


def build_full_text(problem: dict, completion: str, tokenizer) -> str:
    prompt = format_prompt(problem, tokenizer)
    return prompt + completion
