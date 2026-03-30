from __future__ import annotations

import logging
import subprocess
import sys
import torch
import torch.nn.functional as F

from datasets import load_dataset
from state_peft import apply_chat_template_no_thinking, ensure_output_dir, strip_generation_artifacts, train_state_parameters, write_json
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)
MODEL_NAME = "Qwen/Qwen3.5-4B"
MAX_LENGTH = 2048
MAX_NEW_TOKENS = 512

def _fix_torch_fallback():
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as mod
    except ImportError:
        return False

    if not hasattr(mod, "torch_chunk_gated_delta_rule"):
        return False

    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk
        if fla_chunk is not None:
            log.info("FLA Triton chunk_gated_delta_rule available — no fallback fix needed")
            return True
    except ImportError:
        pass

    log.info("Patching torch_chunk_gated_delta_rule for autograd-safe backward")

    def _l2norm(x, dim=-1, eps=1e-6):
        inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
        return x * inv_norm

    def fixed_torch_chunk_gated_delta_rule(
        query, key, value, g, beta,
        chunk_size=64, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False, **kwargs,
    ):
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

        # Gauss-Seidel: clone before each indexed write for autograd safety
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


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    _fix_torch_fallback()
    _reassign_chunk_fn(model)
    cfg = getattr(model.config, "text_config", model.config)
    gdn = [i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"]
    log.info(f"Loaded {MODEL_NAME}: {len(gdn)} GDN layers, {cfg.hidden_size}d")
    return model, tok, cfg, gdn


def _reassign_chunk_fn(model):
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as mod
        fixed_fn = mod.torch_chunk_gated_delta_rule
    except (ImportError, AttributeError):
        return
    layers = get_layers(model)
    for layer in layers:
        if hasattr(layer, "linear_attn"):
            gdn = layer.linear_attn
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk
                if gdn.chunk_gated_delta_rule is fla_chunk:
                    continue
            except ImportError:
                pass
            gdn.chunk_gated_delta_rule = fixed_fn


def get_layers(model):
    if hasattr(model.model, "model"):
        return model.model.model.layers
    return model.model.layers


def patch(model, states, gdn_layers):
    layers = get_layers(model)
    originals = {}
    for i in gdn_layers:
        gdn = layers[i].linear_attn
        originals[i] = gdn.chunk_gated_delta_rule

        def make(fn, st):
            def patched(*a, **kw):
                batch_size = a[0].shape[0]
                kw["initial_state"] = st.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
                return fn(*a, **kw)
            return patched

        gdn.chunk_gated_delta_rule = make(originals[i], states[i])
    return originals


def unpatch(model, originals):
    layers = get_layers(model)
    for i, fn in originals.items():
        layers[i].linear_attn.chunk_gated_delta_rule = fn


def prompt_for(problem, tok):
    msg = [{
        "role": "user",
        "content": (
            "Complete the following Python function. Only output the function body, "
            "no explanation. No markdown.\n\n"
            + problem["prompt"]
        ),
    }]
    return apply_chat_template_no_thinking(tok, msg)


def gen_and_eval(model, tok, problem, n=16, temp=0.7, greedy=False):
    p = prompt_for(problem, tok)
    inp = tok(p, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inp = {k: v.to(model.device) for k, v in inp.items()}
    prompt_len = inp["input_ids"].shape[1]
    if greedy:
        n = 1
    inp = {k: v.expand(n, -1) for k, v in inp.items()}
    with torch.no_grad():
        if greedy:
            out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id)
        else:
            out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, temperature=temp, do_sample=True,
                                 top_p=0.95, pad_token_id=tok.eos_token_id)
    comps, results = [], []
    for i in range(n):
        t = strip_generation_artifacts(
            tok.decode(out[i][prompt_len:], skip_special_tokens=True),
            remove_code_fences=True,
        )
        comps.append(t)
        check_code = problem["prompt"] + t + "\n" + problem["test"] + "\n" + f"check({problem['entry_point']})"
        try:
            r = subprocess.run(
                [sys.executable, "-c", check_code],
                capture_output=True,
                text=True,
                timeout=10,
            )
            results.append(r.returncode == 0)
        except Exception:
            results.append(False)
    return sum(results), comps, results, p

def run(max_problems=20, n_steps=50, lr=1e-3, l2_lambda=5e-4, alpha=0.07,
        n_samples=16, output_dir="/results/state_tune"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_output_dir(output_dir)

    model, tok, cfg, gdn_layers = load_model()
    problems = [dict(r) for r in load_dataset("openai/openai_humaneval", split="test")]
    dev = model.device

    log.info(f"State tuning validation: {max_problems} problems, {n_steps} steps, lr={lr}")
    results = []

    for problem in problems:
        if len(results) >= max_problems:
            break
        tid = problem["task_id"]

        n_pass, comps, exec_res, prompt = gen_and_eval(model, tok, problem, n_samples)
        rate = n_pass / n_samples
        if n_pass == 0 or n_pass == n_samples:
            log.info(f"  {tid}: {n_pass}/{n_samples} — skip")
            continue

        prompt_len = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)["input_ids"].shape[1]
        correct_data = [(prompt + c, prompt_len) for c, r in zip(comps, exec_res) if r]
        log.info(f"  {tid}: baseline={rate:.0%} ({n_pass}/{n_samples}), training on {len(correct_data)} correct")

        nh = cfg.linear_num_value_heads
        kd, vd = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        states = {i: torch.nn.Parameter(torch.zeros(nh, kd, vd, device=dev, dtype=torch.float32))
                  for i in gdn_layers}
        originals = patch(model, states, gdn_layers)
        train_state_parameters(
            model=model,
            tokenizer=tok,
            parameters=states.values(),
            correct_data=correct_data,
            device=dev,
            n_steps=n_steps,
            lr=lr,
            l2_lambda=l2_lambda,
            log_every=10,
            log_fn=lambda step, loss, _: log.info("    step %s: loss=%.4f", step, loss),
        )

        for s in states.values():
            s.data.mul_(alpha)

        n_pass_t, _, _, _ = gen_and_eval(model, tok, problem, n_samples)
        rate_t = n_pass_t / n_samples
        delta = rate_t - rate

        unpatch(model, originals)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if delta > 0:
            marker = "IMPROVED"
        elif delta < 0:
            marker = "DEGRADED"
        else:
            marker = "unchanged"
        log.info(f"  {tid}: baseline={rate:.0%} tuned={rate_t:.0%} delta={delta:+.0%} [{marker}]")
        results.append({
            "task_id": tid, "baseline": rate, "tuned": rate_t, "delta": delta,
            "n_correct_train": len(correct_data),
        })

    if results:
        avg_b = sum(r["baseline"] for r in results) / len(results)
        avg_t = sum(r["tuned"] for r in results) / len(results)
        n_improved = sum(1 for r in results if r["delta"] > 0)
        n_degraded = sum(1 for r in results if r["delta"] < 0)
        log.info(f"\nRESULTS: {len(results)} problems | improved={n_improved} degraded={n_degraded}")
        log.info(f"  avg baseline={avg_b:.1%} tuned={avg_t:.1%} delta={avg_t - avg_b:+.1%}")
    write_json(f"{output_dir}/results.json", results)
    return results


if __name__ == "__main__":
    run()
