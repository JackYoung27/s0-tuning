# Reproducing Paper Results

This directory contains the public experiment scripts referenced by the paper.

This directory contains MATH-500 (8 seeds), GSM8K (10 seeds), and Spider
(5 alphas) artifacts. HumanEval 10-seed results are available on request.

## What Is Included

- `state_tuning.py` -- core HumanEval S0 experiment script
- `math500_state_tuning.py` -- MATH-500 transfer script
- `gsm8k_state_tuning.py` -- GSM8K transfer script
- `spider_state_tuning.py` -- Spider boundary-test script
- `humaneval.py` -- HumanEval dataset loading and sandbox execution
- `state_peft/` -- shared helpers for training, prompting, and evaluation
- `results/README.md` -- compact final-result summary for the public repo

## How To Run

These scripts were designed for Modal/RunPod-style GPU execution.
To run locally, you need a GPU with roughly 20 GB of VRAM and the target model
available locally or through Hugging Face.

Each script exposes a `run()` function that can be called from Python:

```bash
cd experiments
```

```python
from math500_state_tuning import run

result = run(seed=42, alpha=0.07, output_dir="./results/math500")
```

You must run from the `experiments/` directory so that local imports
(`state_peft`, `state_tuning`, `humaneval`, etc.) resolve correctly.

## Expected Final Aggregates

- HumanEval (Qwen3.5-4B): `+23.6 pp` over 10 seeds
- MATH-500: `+4.8 pp` over 8 seeds
- GSM8K: `+2.8 pp` over 10 seeds
- Spider: null result across the five-alpha boundary-test sweep

## Results Layout

Result artifacts are grouped by benchmark:

- `math500/seed_<seed>/results.json` contains all 8 MATH-500 seeds
- `gsm8k/seed_<seed>/results.json` contains all 10 GSM8K seeds
- `spider/alpha_<value>/results.json` contains the five-alpha Spider
  boundary-test sweep

For the paper-facing summary numbers, see `results/README.md`.
