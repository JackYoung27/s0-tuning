# S₀ Tuning

**S₀ Tuning: Zero-Overhead Adaptation of Hybrid Recurrent-Attention Models**

Jack Young

[Paper](https://github.com/jackyoung27/s0-tuning/blob/main/paper/s0_tuning.pdf) | [Website](https://www.jackyoung.io/research/s0-tuning)

<p align="center">
  <img src="https://raw.githubusercontent.com/JackYoung27/s0-tuning/main/paper/fig1_hero.svg" alt="S₀ Tuning overview: learned initial states injected into recurrent layers with zero inference overhead" width="80%">
</p>

S₀ Tuning optimizes the initial hidden state (S₀) of recurrent layers in hybrid
architectures (GatedDeltaNet, Mamba-2). The learned states are injected before
each forward pass, adding zero latency at inference since recurrent models
already maintain state. Trained states are 48 MB, training takes ~3 minutes on
one GPU.

## Results

All results on Qwen3.5-4B with 20 optimization steps per task. p-values from
Welch's t-test (unequal variances) across independent seed runs.

| Benchmark  | Base     | + S₀ Tuning | Delta    | Seeds | p-value     |
|------------|----------|-------------|----------|-------|-------------|
| HumanEval  | 48.8%    | 72.2%       | **+23.6pp**  | 10    | < 10⁻¹¹    |
| MATH-500   | 51.4%    | 56.2%       | **+4.8pp**   | 8     | 0.00002     |
| GSM8K      | 85.3%    | 88.1%       | **+2.8pp**   | 10    | 0.0003      |
| Spider (boundary test) | ~72% | ~72% | +0.0pp   | 5 alphas | n.s.     |

The Spider result (no improvement on out-of-distribution SQL) supports the
trajectory-steering mechanism: S₀ Tuning biases the model toward solution
trajectories already in its distribution, rather than injecting new knowledge.

**Scaling.** Gains increase with model size: +2.7pp at 0.8B, +23.6pp at 4B, +44.0pp at 9B (HumanEval).

**Cross-architecture.** On FalconH1-7B (Mamba-2): 71.8% vs 71.4% LoRA baseline (experimental).

## Usage

```python
from s0_tuning import S0Trainer

trainer = S0Trainer.from_pretrained("Qwen/Qwen3.5-4B")

prompt = "Q: What is 2+2?\nA:"
answer = " 4"
full_text = prompt + answer

# prompt_length = number of prompt tokens (for completion-only loss masking)
tokens = trainer.tokenizer(prompt)
prompt_length = len(tokens["input_ids"])

data = [(full_text, prompt_length)]
trainer.train(data)
trainer.activate()

output = trainer.generate("Q: What is 2+2?\nA:")
print(output)

trainer.save("./my_s0_states")
```

Loading saved states on a new instance:

```python
trainer = S0Trainer.from_pretrained("Qwen/Qwen3.5-4B")
trainer.load("./my_s0_states")
trainer.activate()
output = trainer.generate("Q: What is 2+2?\nA:")
```

## Installation

```sh
pip install git+https://github.com/jackyoung27/s0-tuning.git
```

Requires PyTorch 2.0+, `transformers` >= 4.51.0, and a GPU with >= 10 GB VRAM for training.

## Supported Models

| Model Family | Architecture  | Recurrent Layers | Default Alpha | Status       |
|-------------|---------------|------------------|---------------|--------------|
| Qwen3.5     | GatedDeltaNet | `linear_attention` layers | 0.07  | Tested       |
| FalconH1    | Mamba-2       | `mamba` layers    | 0.65          | Experimental |

To add a new hybrid architecture, the model needs (1) identifiable recurrent
layers and (2) an `initial_state` argument in the recurrent kernel. See
`_detect_architecture` and `_patch_gdn` / `_patch_mamba2` in `trainer.py`.

## How It Works

Recurrent models process sequences starting from S₀ = 0. S₀ Tuning creates
learnable state tensors, patches them into the model as `initial_state`, and
optimizes via next-token prediction loss on correct completions. At inference,
learned states are scaled by alpha (e.g. 0.07 for GatedDeltaNet) to prevent
distribution shift. The result: the model starts each sequence from a
task-informed state rather than zeros, biasing generation toward correct
solution trajectories without modifying any model weights.

## Configuration

`S0Config` controls training:

| Parameter   | Default | Description                              |
|-------------|---------|------------------------------------------|
| `n_steps`   | 20      | Optimization steps                       |
| `lr`        | 1e-3    | Learning rate                            |
| `l2_lambda` | 5e-4    | L2 regularization weight                 |
| `alpha`     | None    | Scaling factor (auto-detected per architecture if None) |
| `normalize` | False   | Normalize states before scaling          |
| `grad_clip` | 1.0     | Gradient clipping norm                   |
| `max_length`| 2048    | Maximum sequence length                  |

## Citation

If you use this codebase, or otherwise found our work valuable, please cite:
```bibtex
@article{young2026s0tuning,
  title={S$_0$ Tuning: Zero-Overhead Adaptation of Hybrid Recurrent-Attention Models},
  author={Young, Jack},
  year={2026}
}
```

## License

MIT
