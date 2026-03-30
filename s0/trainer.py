from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from s0.config import S0Config

log = logging.getLogger(__name__)

_DEFAULT_ALPHA = {
    "gated_delta_net": 0.07,
    "mamba2": 0.65,
}


def _detect_architecture(config: Any, model: Any | None = None) -> tuple[str, list[int]]:
    cfg = getattr(config, "text_config", config)

    if hasattr(cfg, "layer_types"):
        indices = [i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"]
        if indices:
            return "gated_delta_net", indices

    if model is not None:
        layers = _get_layers(model)
        indices = [i for i, layer in enumerate(layers) if hasattr(layer, "mamba")]
        if indices:
            return "mamba2", indices

    raise ValueError(
        "No supported recurrent architecture detected. "
        "Expected layer_types with 'linear_attention' (GatedDeltaNet) "
        "or layers with 'mamba' attribute (Mamba-2/FalconH1)."
    )


def _get_state_shape(config: Any, arch: str, model: Any | None = None) -> tuple[int, int, int]:
    cfg = getattr(config, "text_config", config)
    if arch == "gated_delta_net":
        return cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    if arch == "mamba2":
        if model is not None:
            layers = _get_layers(model)
            for layer in layers:
                if hasattr(layer, "mamba"):
                    mixer = layer.mamba
                    return mixer.num_heads, mixer.head_dim, mixer.ssm_state_size
        n_heads = getattr(cfg, "mamba_n_heads", getattr(cfg, "n_mamba_heads", None))
        head_dim = getattr(cfg, "mamba_d_head", getattr(cfg, "mamba_head_dim", None))
        d_state = getattr(cfg, "mamba_d_state", getattr(cfg, "ssm_state_size", None))
        if all(x is not None for x in (n_heads, head_dim, d_state)):
            return n_heads, head_dim, d_state
        raise ValueError(
            "Cannot determine Mamba-2 state shape. Load the model and pass it "
            "to the trainer so dimensions can be read from the mixer modules."
        )
    raise ValueError(f"Unknown architecture: {arch}")


def _get_layers(model: Any) -> Any:
    if hasattr(model.model, "model"):
        return model.model.model.layers
    return model.model.layers


def _fix_torch_fallback() -> bool:
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as mod
    except ImportError:
        return False

    if not hasattr(mod, "torch_chunk_gated_delta_rule"):
        return False

    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk
        if fla_chunk is not None:
            log.info("FLA Triton kernel available, skipping fallback patch")
            return True
    except ImportError:
        pass

    log.info("Patching torch_chunk_gated_delta_rule for autograd-safe backward")

    def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

    def fixed_torch_chunk_gated_delta_rule(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = 64,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        initial_dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            query = _l2norm(query, dim=-1, eps=1e-6)
            key = _l2norm(key, dim=-1, eps=1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (query, key, value, beta, g)
        ]

        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
        total_sequence_length = sequence_length + pad_size
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        query, key, value, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
            for x in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)

        # Gauss-Seidel: out-of-place clones to avoid autograd corruption
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            new_row = row + (row.unsqueeze(-1) * sub).sum(-2)
            attn = attn.clone()
            attn[..., i, :i] = new_row

        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
        last_recurrent_state = (
            torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
            if initial_state is None
            else initial_state.to(value)
        )
        core_attn_out = torch.zeros_like(value)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

        for i in range(0, total_sequence_length // chunk_size):
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill(mask, 0)
            v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
            core_attn_out[:, :, i] = attn_inter + attn @ v_new
            last_recurrent_state = (
                last_recurrent_state * g[:, :, i, -1, None, None].exp()
                + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
            )

        if not output_final_state:
            last_recurrent_state = None
        core_attn_out = core_attn_out.reshape(
            core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
        )
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, last_recurrent_state

    mod.torch_chunk_gated_delta_rule = fixed_torch_chunk_gated_delta_rule
    return True


def _reassign_chunk_fn(model: Any) -> None:
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as mod
        fixed_fn = mod.torch_chunk_gated_delta_rule
    except (ImportError, AttributeError):
        return

    for layer in _get_layers(model):
        if not hasattr(layer, "linear_attn"):
            continue
        gdn = layer.linear_attn
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk
            if gdn.chunk_gated_delta_rule is fla_chunk:
                continue
        except ImportError:
            pass
        gdn.chunk_gated_delta_rule = fixed_fn


def _patch_gdn(model: Any, states: dict[int, torch.nn.Parameter]) -> dict[int, Any]:
    layers = _get_layers(model)
    originals: dict[int, Any] = {}
    for i, state in states.items():
        gdn = layers[i].linear_attn
        originals[i] = gdn.chunk_gated_delta_rule

        def _make_patched(fn: Any, s: torch.nn.Parameter) -> Any:
            def patched(*a: Any, **kw: Any) -> Any:
                bs = a[0].shape[0]
                kw["initial_state"] = s.unsqueeze(0).expand(bs, -1, -1, -1).contiguous()
                return fn(*a, **kw)
            return patched

        gdn.chunk_gated_delta_rule = _make_patched(originals[i], state)
    return originals


def _unpatch_gdn(model: Any, originals: dict[int, Any]) -> None:
    layers = _get_layers(model)
    for i, fn in originals.items():
        layers[i].linear_attn.chunk_gated_delta_rule = fn


def _make_mamba2_patched_forward(
    mixer: Any, state_param: torch.nn.Parameter,
) -> Any:
    import types

    orig_cuda_fwd = mixer.cuda_kernels_forward

    if isinstance(orig_cuda_fwd, types.MethodType):
        globs = orig_cuda_fwd.__func__.__globals__
    else:
        globs = orig_cuda_fwd.__globals__

    real_mamba_split = globs.get("mamba_split_conv1d_scan_combined")
    real_mamba_chunk = globs.get("mamba_chunk_scan_combined")

    def patched_forward(
        hidden_states: torch.Tensor,
        cache_params: Any = None,
        cache_position: Any = None,
        attention_mask: Any = None,
    ) -> Any:
        batch = hidden_states.shape[0]
        s0 = state_param.unsqueeze(0).expand(batch, -1, -1, -1).to(
            dtype=hidden_states.dtype, device=hidden_states.device,
        )

        def split_with_s0(*args: Any, **kwargs: Any) -> Any:
            kwargs["initial_states"] = s0
            return real_mamba_split(*args, **kwargs)

        def chunk_with_s0(*args: Any, **kwargs: Any) -> Any:
            kwargs["initial_states"] = s0
            return real_mamba_chunk(*args, **kwargs)

        globs["mamba_split_conv1d_scan_combined"] = split_with_s0
        globs["mamba_chunk_scan_combined"] = chunk_with_s0
        try:
            result = orig_cuda_fwd(
                hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
        finally:
            globs["mamba_split_conv1d_scan_combined"] = real_mamba_split
            globs["mamba_chunk_scan_combined"] = real_mamba_chunk

        return result

    return patched_forward


def _patch_mamba2(model: Any, states: dict[int, torch.nn.Parameter]) -> dict[int, Any]:
    layers = _get_layers(model)
    originals: dict[int, Any] = {}
    for i, state in states.items():
        mixer = layers[i].mamba
        originals[i] = mixer.cuda_kernels_forward
        mixer.cuda_kernels_forward = _make_mamba2_patched_forward(mixer, state)
    return originals


def _unpatch_mamba2(model: Any, originals: dict[int, Any]) -> None:
    layers = _get_layers(model)
    for i, fn in originals.items():
        layers[i].mamba.cuda_kernels_forward = fn


class S0Trainer:

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: S0Config | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or S0Config()
        self._arch, self._recurrent_layers = _detect_architecture(model.config, model)
        self._state_shape = _get_state_shape(model.config, self._arch, model)
        self._states: dict[int, torch.nn.Parameter] = {}
        self._originals: dict[int, Any] | None = None
        self._active = False

        if self._arch == "gated_delta_net":
            _fix_torch_fallback()
            _reassign_chunk_fn(model)

        log.info(
            "S0Trainer: arch=%s, %d recurrent layers, state_shape=%s",
            self._arch, len(self._recurrent_layers), self._state_shape,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        config: S0Config | None = None,
        **model_kwargs: Any,
    ) -> S0Trainer:
        model_kwargs.setdefault("torch_dtype", torch.bfloat16)
        model_kwargs.setdefault("device_map", "auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()
        return cls(model, tokenizer, config)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _patch(self, states: dict[int, torch.nn.Parameter]) -> dict[int, Any]:
        if self._arch == "gated_delta_net":
            return _patch_gdn(self.model, states)
        if self._arch == "mamba2":
            return _patch_mamba2(self.model, states)
        raise ValueError(f"No patch implementation for {self._arch}")

    def _unpatch(self, originals: dict[int, Any]) -> None:
        if self._arch == "gated_delta_net":
            _unpatch_gdn(self.model, originals)
        elif self._arch == "mamba2":
            _unpatch_mamba2(self.model, originals)
        else:
            raise ValueError(f"No unpatch implementation for {self._arch}")

    def train(self, data: list[tuple[str, int]]) -> dict[str, float]:
        if not data:
            raise ValueError("train_data is empty")

        cfg = self.config
        dev = self.device
        nh, kd, vd = self._state_shape

        self._states = {
            i: torch.nn.Parameter(torch.zeros(nh, kd, vd, device=dev, dtype=torch.float32))
            for i in self._recurrent_layers
        }
        params = list(self._states.values())

        originals = self._patch(self._states)

        try:
            for p in self.model.parameters():
                p.requires_grad_(False)
            for p in params:
                p.requires_grad_(True)

            optimizer = torch.optim.Adam(params, lr=cfg.lr)
            final_loss = 0.0

            for step in range(cfg.n_steps):
                optimizer.zero_grad()
                loss_sum = 0.0

                for text, prompt_len in data:
                    ids = self.tokenizer(
                        text, return_tensors="pt", truncation=True, max_length=cfg.max_length,
                    )["input_ids"].to(dev)
                    labels = ids.clone()
                    labels[0, :prompt_len] = -100
                    loss = self.model(input_ids=ids, labels=labels).loss
                    (loss / len(data)).backward()
                    loss_sum += loss.item()

                l2 = sum(p.pow(2).sum() for p in params)
                (cfg.l2_lambda * l2).backward()

                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                optimizer.step()
                final_loss = loss_sum / len(data)

                if step % 5 == 0 or step == cfg.n_steps - 1:
                    norm = sum(float(p.data.norm()) for p in params) / len(params)
                    log.info("step %d/%d: loss=%.4f avg_norm=%.6f", step, cfg.n_steps, final_loss, norm)
        finally:
            self._unpatch(originals)

        avg_norm = sum(float(p.data.norm()) for p in params) / len(params)
        return {"final_loss": final_loss, "avg_state_norm": avg_norm}

    def _resolve_alpha(self) -> float:
        if self.config.alpha is not None:
            return self.config.alpha
        return _DEFAULT_ALPHA.get(self._arch, 0.07)

    def _scaled_states(self) -> dict[int, torch.nn.Parameter]:
        alpha = self._resolve_alpha()
        scaled: dict[int, torch.nn.Parameter] = {}
        for idx, state in self._states.items():
            raw = state.data.clone()
            if self.config.normalize:
                norm = raw.norm()
                if float(norm) > 0:
                    scaled[idx] = torch.nn.Parameter(alpha * raw / norm)
                    continue
            scaled[idx] = torch.nn.Parameter(raw * alpha)
        return scaled

    def activate(self) -> None:
        if self._active:
            return
        if not self._states:
            raise RuntimeError("No trained states. Call train() first.")
        scaled = self._scaled_states()
        self._originals = self._patch(scaled)
        self._active = True
        log.info("S0 states activated (alpha=%.4f)", self._resolve_alpha())

    def deactivate(self) -> None:
        if not self._active or self._originals is None:
            return
        self._unpatch(self._originals)
        self._originals = None
        self._active = False
        log.info("S0 states deactivated")

    def generate(self, prompt: str, max_new_tokens: int = 512, **kwargs: Any) -> str:
        ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.config.max_length,
        )["input_ids"].to(self.device)
        kwargs.setdefault("pad_token_id", self.tokenizer.eos_token_id)
        with torch.no_grad():
            out = self.model.generate(input_ids=ids, max_new_tokens=max_new_tokens, **kwargs)
        return self.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if not self._states:
            raise RuntimeError("No trained states to save.")

        tensors = {str(k): v.data for k, v in self._states.items()}
        save_file(tensors, path / "states.safetensors")
        self.config.save(path / "config.json")

        meta = {
            "arch": self._arch,
            "recurrent_layers": self._recurrent_layers,
            "state_shape": list(self._state_shape),
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))
        log.info("Saved %d states to %s", len(self._states), path)

    def load(self, path: str | Path) -> None:
        path = Path(path)

        meta_path = path / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            saved_arch = meta.get("arch")
            saved_shape = tuple(meta.get("state_shape", []))
            if saved_arch != self._arch:
                raise ValueError(
                    f"Architecture mismatch: saved states are for '{saved_arch}' "
                    f"but current model is '{self._arch}'"
                )
            if saved_shape != tuple(self._state_shape):
                raise ValueError(
                    f"State shape mismatch: saved states have shape {list(saved_shape)} "
                    f"but current model expects {list(self._state_shape)}"
                )

        if meta_path.exists():
            saved_layers = meta.get("recurrent_layers")
            if saved_layers is not None and sorted(saved_layers) != sorted(self._recurrent_layers):
                raise ValueError(
                    f"Layer index mismatch: saved states cover layers {sorted(saved_layers)} "
                    f"but current model has recurrent layers {sorted(self._recurrent_layers)}"
                )

        config_path = path / "config.json"
        if config_path.exists():
            self.config = S0Config.load(config_path)

        st_path = path / "states.safetensors"
        pt_path = path / "states.pt"
        if st_path.exists():
            raw = load_file(st_path, device=str(self.device))
        elif pt_path.exists():
            raw = torch.load(pt_path, map_location=self.device, weights_only=True)
        else:
            raise FileNotFoundError(f"No states found in {path}")

        self._states = {int(k): torch.nn.Parameter(v) for k, v in raw.items()}
        log.info("Loaded %d states from %s", len(self._states), path)
