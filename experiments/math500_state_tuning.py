from __future__ import annotations

import logging
import time
import torch

from state_peft import (
    ensure_output_dir,
    scale_state_dict,
    strip_generation_artifacts,
    train_state_parameters,
    write_json,
)
from state_peft.math500 import check_answer, extract_boxed, format_prompt
from state_tuning import load_model, patch, unpatch

log = logging.getLogger(__name__)

BATCH_SIZE = 16


def gen_and_eval_math(model, tok, problem_text: str, gold_answer: str, n: int = 16, greedy: bool = False):
    prompt = format_prompt(problem_text, tok)
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inp = {k: v.to(model.device) for k, v in inp.items()}
    pl = inp["input_ids"].shape[1]

    if greedy:
        n = 1
    inp = {k: v.expand(n, -1) for k, v in inp.items()}

    with torch.no_grad():
        if greedy:
            out = model.generate(**inp, max_new_tokens=1024, do_sample=False, pad_token_id=tok.eos_token_id)
        else:
            out = model.generate(**inp, max_new_tokens=1024, temperature=0.7, do_sample=True,
                                 top_p=0.95, pad_token_id=tok.eos_token_id)

    comps, results = [], []
    for i in range(out.shape[0]):
        t = strip_generation_artifacts(tok.decode(out[i][pl:], skip_special_tokens=True))
        comps.append(t)
        pred = extract_boxed(t)
        results.append(check_answer(pred, gold_answer))

    return sum(results), comps, results, prompt


def batch_greedy_eval_math(model, tok, problems, batch_size=BATCH_SIZE):
    orig_padding_side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    results = [None] * len(problems)
    t0 = time.time()

    valid_global_indices = []
    valid_prompts = []
    for i, d in enumerate(problems):
        if d["gold"] is None:
            results[i] = {
                "problem": d["problem"][:100], "pass": False, "gold": None,
                "predicted": None,
                "level": d.get("level", "unknown"),
                "subject": d.get("type", d.get("subject", "unknown")),
            }
        else:
            valid_global_indices.append(i)
            valid_prompts.append(format_prompt(d["problem"], tok))

    for batch_start in range(0, len(valid_prompts), batch_size):
        batch_prompts = valid_prompts[batch_start:batch_start + batch_size]
        batch_indices = valid_global_indices[batch_start:batch_start + batch_size]

        inp = tok(batch_prompts, return_tensors="pt", truncation=True, max_length=2048,
                  padding=True)
        inp = {k: v.to(model.device) for k, v in inp.items()}

        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=1024, do_sample=False,
                                 pad_token_id=tok.pad_token_id)

        input_len = inp["input_ids"].shape[1]
        for k in range(len(batch_prompts)):
            t = strip_generation_artifacts(tok.decode(out[k][input_len:], skip_special_tokens=True))
            pred = extract_boxed(t)
            gi = batch_indices[k]
            d = problems[gi]
            correct = check_answer(pred, d["gold"])
            results[gi] = {
                "problem": d["problem"][:100], "pass": correct, "gold": d["gold"],
                "predicted": pred,
                "level": d.get("level", "unknown"),
                "subject": d.get("type", d.get("subject", "unknown")),
            }

        done = batch_start + len(batch_prompts)
        total_done = done + sum(1 for r in results if r is not None and r["gold"] is None)
        if done % 20 < batch_size or done == len(valid_prompts):
            elapsed = time.time() - t0
            filled = [r for r in results if r is not None]
            acc = sum(r["pass"] for r in filled) / len(filled) if filled else 0
            rate = total_done / elapsed if elapsed > 0 else 0
            eta = (len(problems) - total_done) / rate if rate > 0 else 0
            log.info(f"  Progress: {total_done}/{len(problems)}, acc={acc:.1%}, "
                     f"speed={rate:.1f} prob/s, ETA={eta:.0f}s")

    tok.padding_side = orig_padding_side
    return results


def run(n_train: int = 250, n_eval: int = 250, n_steps: int = 20, lr: float = 1e-3,
        l2_lambda: float = 5e-4, alpha: float = 0.07, n_samples: int = 4,
        normalize_alpha: bool = True, output_dir: str = "/results/math500"):
    from datasets import load_dataset
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_output_dir(output_dir)

    model, tok, cfg, gdn_layers = load_model()
    dev = model.device

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    all_problems = [dict(r) for r in ds]
    log.info(f"MATH-500: loaded {len(all_problems)} problems")

    for d in all_problems:
        d["gold"] = extract_boxed(d["solution"])
        if d["gold"] is None:
            d["gold"] = d.get("answer", None)

    train_data = all_problems[:n_train]
    eval_data = all_problems[len(all_problems) - n_eval:]

    for split_name, split_data in [("train", train_data), ("eval", eval_data)]:
        levels = {}
        for d in split_data:
            lvl = d.get("level", "unknown")
            levels[lvl] = levels.get(lvl, 0) + 1
        log.info(f"  {split_name}: {len(split_data)} problems, levels={dict(sorted(levels.items()))}")

    subjects = {}
    for d in eval_data:
        subj = d.get("type", d.get("subject", "unknown"))
        subjects[subj] = subjects.get(subj, 0) + 1
    log.info(f"  Eval subjects: {dict(sorted(subjects.items()))}")

    log.info(f"Collecting correct solutions from {len(train_data)} training problems...")
    correct_data = []
    train_accuracy = 0
    t0_train = time.time()
    for i, d in enumerate(train_data):
        if d["gold"] is None:
            continue
        n_pass, comps, exec_res, prompt = gen_and_eval_math(model, tok, d["problem"], d["gold"], n_samples)
        if n_pass > 0:
            train_accuracy += 1
        for c, r in zip(comps, exec_res):
            if r:
                full = prompt + c
                prompt_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)["input_ids"]
                correct_data.append((full, prompt_ids.shape[1]))
                break  # one correct solution per problem
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0_train
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(train_data) - i - 1) / rate if rate > 0 else 0
            log.info(f"  Processed {i+1}/{len(train_data)}, found {len(correct_data)} correct so far "
                     f"(accuracy: {train_accuracy}/{i+1} = {train_accuracy/(i+1):.1%}), "
                     f"speed={rate:.1f} prob/s, ETA={eta:.0f}s")
    log.info(f"  Found {len(correct_data)} correct solutions from {len(train_data)} problems "
             f"(train accuracy: {train_accuracy}/{len(train_data)} = {train_accuracy/len(train_data):.1%})")

    if len(correct_data) < 5:
        output = {"error": "insufficient_training_data", "n_correct": len(correct_data),
                  "train_accuracy": train_accuracy / len(train_data) if train_data else 0}
        write_json(f"{output_dir}/results.json", output)
        return output

    log.info(f"Evaluating baseline (greedy, batched, batch_size={BATCH_SIZE}) on {len(eval_data)} problems...")
    baseline_greedy = batch_greedy_eval_math(model, tok, eval_data, batch_size=BATCH_SIZE)
    bg_acc = sum(r["pass"] for r in baseline_greedy) / len(baseline_greedy)
    log.info(f"  Baseline greedy: {bg_acc:.1%}")

    log.info(f"State tuning on {len(correct_data)} correct solutions, {n_steps} steps")
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
        log_fn=lambda step, loss, params: log.info(
            "  step %s: loss=%.4f, norm=%.6f",
            step,
            loss,
            sum(parameter.data.norm().item() for parameter in params) / len(states),
        ),
    )

    final_states = {i: s.data.clone() for i, s in states.items()}
    unpatch(model, originals)

    scaled = scale_state_dict(
        final_states,
        alpha=alpha,
        normalize=normalize_alpha,
        warn_zero_norm=lambda index: log.warning(
            "Layer %s: zero-norm state, using raw alpha scaling",
            index,
        ),
    )
    if normalize_alpha:
        log.info(f"Alpha normalization: alpha={alpha}, per-layer state/||state||")
    originals = patch(model, scaled, gdn_layers)

    log.info(f"Evaluating state-tuned (alpha={alpha}, greedy, batched, batch_size={BATCH_SIZE}) "
             f"on {len(eval_data)} problems...")
    tuned_greedy = batch_greedy_eval_math(model, tok, eval_data, batch_size=BATCH_SIZE)
    tg_acc = sum(r["pass"] for r in tuned_greedy) / len(tuned_greedy)
    log.info(f"  Tuned greedy: {tg_acc:.1%}")

    unpatch(model, originals)

    log.info(f"  Training: {len(correct_data)} correct from {len(train_data)} problems")
    log.info(f"  Eval: {len(eval_data)} problems")
    log.info(f"  GREEDY: baseline={bg_acc:.1%}, tuned={tg_acc:.1%}, delta={tg_acc - bg_acc:+.1%}")

    improved = sum(1 for b, t in zip(baseline_greedy, tuned_greedy) if not b["pass"] and t["pass"])
    degraded = sum(1 for b, t in zip(baseline_greedy, tuned_greedy) if b["pass"] and not t["pass"])
    log.info(f"  {improved} newly solved, {degraded} newly broken")

    levels = sorted(set(b["level"] for b in baseline_greedy))
    level_results = {}
    for lvl in levels:
        b_lvl = [b for b in baseline_greedy if b["level"] == lvl]
        t_lvl = [t for t in tuned_greedy if t["level"] == lvl]
        b_acc = sum(r["pass"] for r in b_lvl) / len(b_lvl) if b_lvl else 0
        t_acc = sum(r["pass"] for r in t_lvl) / len(t_lvl) if t_lvl else 0
        level_results[lvl] = {"n": len(b_lvl), "baseline": b_acc, "tuned": t_acc, "delta": t_acc - b_acc}
        log.info(f"  {lvl}: n={len(b_lvl)}, baseline={b_acc:.1%}, tuned={t_acc:.1%}, delta={t_acc - b_acc:+.1%}")

    subjects_list = sorted(set(b["subject"] for b in baseline_greedy))
    subject_results = {}
    for subj in subjects_list:
        b_subj = [b for b in baseline_greedy if b["subject"] == subj]
        t_subj = [t for t in tuned_greedy if t["subject"] == subj]
        b_acc = sum(r["pass"] for r in b_subj) / len(b_subj) if b_subj else 0
        t_acc = sum(r["pass"] for r in t_subj) / len(t_subj) if t_subj else 0
        subject_results[subj] = {"n": len(b_subj), "baseline": b_acc, "tuned": t_acc, "delta": t_acc - b_acc}
        log.info(f"  {subj}: n={len(b_subj)}, baseline={b_acc:.1%}, tuned={t_acc:.1%}, delta={t_acc - b_acc:+.1%}")

    per_problem = []
    for d, b, t in zip(eval_data, baseline_greedy, tuned_greedy):
        per_problem.append({
            "problem": d["problem"][:200],
            "gold": d["gold"],
            "level": d.get("level", "unknown"),
            "subject": d.get("type", d.get("subject", "unknown")),
            "baseline_correct": b["pass"],
            "tuned_correct": t["pass"],
            "baseline_predicted": b.get("predicted"),
            "tuned_predicted": t.get("predicted"),
        })

    output = {
        "benchmark": "math500",
        "n_train": len(train_data),
        "n_eval": len(eval_data),
        "n_correct_train": len(correct_data),
        "train_accuracy": train_accuracy / len(train_data) if train_data else 0,
        "baseline_greedy": bg_acc,
        "tuned_greedy": tg_acc,
        "greedy_delta": tg_acc - bg_acc,
        "greedy_improved": improved,
        "greedy_degraded": degraded,
        "by_level": level_results,
        "by_subject": subject_results,
        "hyperparams": {"n_steps": n_steps, "lr": lr, "l2_lambda": l2_lambda, "alpha": alpha,
                        "n_samples": n_samples},
        "per_problem": per_problem,
    }
    write_json(f"{output_dir}/results.json", output)
    log.info(f"Results saved to {output_dir}/results.json")
    return output


if __name__ == "__main__":
    run()
