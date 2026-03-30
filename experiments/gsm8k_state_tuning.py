from __future__ import annotations

import logging
import torch

from state_peft import (
    ensure_output_dir,
    prepare_decoder_only_padding,
    scale_state_dict,
    strip_generation_artifacts,
    train_state_parameters,
    write_json,
)
from state_peft.gsm8k import check_answer, extract_answer, format_prompt
from state_tuning import load_model, patch, unpatch

log = logging.getLogger(__name__)


def gen_and_eval_math(model, tok, question, gold_answer, n=16, greedy=False):
    prompt = format_prompt(question, tok)
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inp = {k: v.to(model.device) for k, v in inp.items()}
    prompt_len = inp["input_ids"].shape[1]

    if greedy:
        n = 1
    inp = {k: v.expand(n, -1) for k, v in inp.items()}

    gen_kwargs = dict(max_new_tokens=512, pad_token_id=tok.eos_token_id)
    if greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.95)

    with torch.no_grad():
        out = model.generate(**inp, **gen_kwargs)

    comps, results = [], []
    for i in range(out.shape[0]):
        text = strip_generation_artifacts(tok.decode(out[i][prompt_len:], skip_special_tokens=True))
        comps.append(text)
        results.append(check_answer(extract_answer(text), gold_answer))
    return sum(results), comps, results, prompt


def batch_greedy_eval(model, tok, problems, batch_size=8):
    prepare_decoder_only_padding(tok)
    results = []
    for start in range(0, len(problems), batch_size):
        batch = problems[start:start + batch_size]
        prompts = [format_prompt(d["question"], tok) for d in batch]
        golds = [d["gold"] for d in batch]

        inputs = tok(prompts, return_tensors="pt", truncation=True, max_length=2048,
                     padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for i, gold in enumerate(golds):
            text = strip_generation_artifacts(tok.decode(outputs[i][input_len:], skip_special_tokens=True))
            pred = extract_answer(text)
            results.append({"question": batch[i]["question"], "pass": check_answer(pred, gold),
                           "gold": gold, "pred": pred or ""})

        done = min(start + batch_size, len(problems))
        if done % 100 < batch_size or done == len(problems):
            acc = sum(r["pass"] for r in results) / len(results)
            log.info(f"  Progress: {done}/{len(problems)}, acc so far: {acc:.1%}")

    return results


def run(n_train=200, n_eval=0, n_steps=20, lr=1e-3, l2_lambda=5e-4, alpha=0.07,
        output_dir="/results/gsm8k"):
    from datasets import load_dataset
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_output_dir(output_dir)

    model, tok, cfg, gdn_layers = load_model()
    dev = model.device

    prepare_decoder_only_padding(tok)

    ds = load_dataset("openai/gsm8k", "main")
    train_data = [dict(r) for r in ds["train"]][:n_train]
    all_test = [dict(r) for r in ds["test"]]
    test_data = all_test if n_eval == 0 else all_test[:n_eval]

    for d in train_data + test_data:
        d["gold"] = d["answer"].split("####")[-1].strip().replace(",", "")

    log.info(f"GSM8K: Train={len(train_data)}, Eval={len(test_data)}")

    log.info("Collecting correct solutions from training problems...")
    correct_data = []
    for i, d in enumerate(train_data):
        n_pass, comps, exec_res, prompt = gen_and_eval_math(model, tok, d["question"], d["gold"], 4)
        for c, r in zip(comps, exec_res):
            if r:
                full = prompt + c
                prompt_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)["input_ids"]
                correct_data.append((full, prompt_ids.shape[1]))
                break
        if (i + 1) % 50 == 0:
            log.info(f"  Processed {i+1}/{len(train_data)}, found {len(correct_data)} correct so far")
    log.info(f"  Found {len(correct_data)} correct solutions from {len(train_data)} problems")

    if len(correct_data) < 5:
        return {"error": "insufficient_training_data", "n_correct": len(correct_data)}

    log.info("Evaluating baseline (greedy, batched)...")
    baseline_greedy = batch_greedy_eval(model, tok, test_data, batch_size=8)
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
            "  step %s: loss=%.4f, norm=%.3f",
            step,
            loss,
            sum(parameter.data.norm().item() for parameter in params) / len(states),
        ),
    )

    final_states = {i: s.data.clone() for i, s in states.items()}
    unpatch(model, originals)

    scaled = scale_state_dict(final_states, alpha=alpha)
    originals = patch(model, scaled, gdn_layers)

    log.info(f"Evaluating state-tuned (alpha={alpha}, greedy, batched)...")
    tuned_greedy = batch_greedy_eval(model, tok, test_data, batch_size=8)
    tg_acc = sum(r["pass"] for r in tuned_greedy) / len(tuned_greedy)
    log.info(f"  Tuned greedy: {tg_acc:.1%}")

    unpatch(model, originals)

    greedy_improved = sum(1 for b, t in zip(baseline_greedy, tuned_greedy) if not b["pass"] and t["pass"])
    greedy_degraded = sum(1 for b, t in zip(baseline_greedy, tuned_greedy) if b["pass"] and not t["pass"])

    log.info(f"  Training: {len(correct_data)} correct solutions from {n_train} problems")
    log.info(f"  Eval: {len(test_data)} test problems")
    log.info(f"  GREEDY:  baseline={bg_acc:.1%}, tuned={tg_acc:.1%}, delta={tg_acc-bg_acc:+.1%}")
    log.info(f"  Greedy: {greedy_improved} newly solved, {greedy_degraded} newly broken")

    per_problem = [{"question": b["question"], "baseline": b["pass"], "tuned": t["pass"],
                    "gold": b["gold"], "baseline_pred": b.get("pred", ""), "tuned_pred": t.get("pred", "")}
                   for b, t in zip(baseline_greedy, tuned_greedy)]

    output = {
        "benchmark": "gsm8k", "n_train": n_train, "n_eval": len(test_data),
        "n_correct_train": len(correct_data),
        "baseline_greedy": bg_acc, "tuned_greedy": tg_acc, "greedy_delta": tg_acc - bg_acc,
        "greedy_improved": greedy_improved, "greedy_degraded": greedy_degraded,
        "per_problem": per_problem,
    }
    write_json(f"{output_dir}/results.json", output)
    return output


if __name__ == "__main__":
    run()
