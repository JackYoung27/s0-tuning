# S0 Tuning

S0 tuning adapts a recurrent language model to a task by learning its initial hidden states. These states are the model's working memory: matrices that carry information between tokens. We train their starting values on prompt-completion pairs in place of the usual zero initialization. The model weights stay frozen.

- The learned states give each new prompt a task-specific starting point without adding prompt tokens.
- The released states target Python code generation. Reuse them on new prompts, or train a separate state file for another task.

## Results

Qwen3.5-4B on 84 held-out HumanEval problems, with thinking disabled. The S0 result is averaged across 10 seeds.

| Model | Score |
| --- | --- |
| Base model | 48.8% |
| With S0 tuning | 72.2% |

## Use

```bash
python -m pip install huggingface_hub
hf download JackYoung27/s0-tuning-qwen3.5-4b-humaneval --local-dir s0-tuning
cd s0-tuning
python -m pip install -e .
```

Run from the downloaded directory:

```python
from s0 import S0Trainer

trainer = S0Trainer.from_pretrained("Qwen/Qwen3.5-4B")
trainer.load(".")
trainer.activate()
print(trainer.generate("def fibonacci(n):\n", max_new_tokens=128))
```

The saved states require Qwen3.5-4B. Its weights download separately and must fit in local memory.

[MIT license](LICENSE).
