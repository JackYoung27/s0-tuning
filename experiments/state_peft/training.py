from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch


def freeze_model_parameters(model: Any) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def train_state_parameters(
    *,
    model: Any,
    tokenizer: Any,
    parameters: Iterable[torch.nn.Parameter],
    correct_data: list[tuple[str, int]],
    device: torch.device | str,
    n_steps: int,
    lr: float,
    l2_lambda: float,
    max_length: int = 2048,
    grad_clip: float = 1.0,
    log_fn: Any | None = None,
    log_every: int = 5,
) -> None:
    params = list(parameters)
    optimizer = torch.optim.Adam(params, lr=lr)
    freeze_model_parameters(model)
    for parameter in params:
        parameter.requires_grad_(True)

    for step in range(n_steps):
        optimizer.zero_grad()
        loss_sum = 0.0
        for text, prompt_len in correct_data:
            tokenized = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )["input_ids"].to(device)
            labels = tokenized.clone()
            labels[0, :prompt_len] = -100
            loss = model(input_ids=tokenized, labels=labels).loss
            (loss / len(correct_data)).backward()
            loss_sum += float(loss.item())

        l2_term = sum(parameter.pow(2).sum() for parameter in params)
        (l2_lambda * l2_term).backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()

        should_log = step % log_every == 0 or step == n_steps - 1
        if log_fn is not None and should_log:
            log_fn(step, loss_sum / len(correct_data), params)


def clone_state_dict(parameters: Mapping[int, torch.nn.Parameter]) -> dict[int, torch.Tensor]:
    return {index: parameter.data.clone() for index, parameter in parameters.items()}


def scale_state_dict(
    states: Mapping[int, torch.Tensor],
    *,
    alpha: float,
    normalize: bool = False,
    warn_zero_norm: Any | None = None,
) -> dict[int, torch.nn.Parameter]:
    scaled: dict[int, torch.nn.Parameter] = {}
    for index, state in states.items():
        if normalize:
            norm = state.norm()
            if float(norm) > 0.0:
                scaled[index] = torch.nn.Parameter(alpha * state / norm)
                continue
            if warn_zero_norm is not None:
                warn_zero_norm(index)
        scaled[index] = torch.nn.Parameter(state * alpha)
    return scaled
