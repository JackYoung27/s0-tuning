from __future__ import annotations

import json
import logging
import os
import pickle
import time
import torch


def atomic_write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def atomic_write_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def atomic_save_torch(path, obj):
    torch.save(obj, path)

from state_peft import (
    ensure_output_dir,
    prepare_decoder_only_padding,
    scale_state_dict,
    strip_generation_artifacts,
    train_state_parameters,
    write_json,
)
from state_peft.spider import (
    check_sql_answer,
    extract_sql,
    format_spider_prompt,
    get_schema,
)
from state_tuning import load_model, patch, unpatch

log = logging.getLogger(__name__)

BATCH_SIZE = 8


def _resolve_db_path(db_id: str, db_dir: str) -> str:
    return os.path.join(db_dir, db_id, f"{db_id}.sqlite")


def _get_schema_cached(db_id: str, db_dir: str, cache: dict) -> str:
    if db_id not in cache:
        cache[db_id] = get_schema(_resolve_db_path(db_id, db_dir))
    return cache[db_id]


def gen_and_eval_spider(model, tok, question, db_id, schema, gold_sql, db_path,
                        n=4, greedy=False):
    prompt = format_spider_prompt(question, db_id, schema, tok)
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
        pred_sql = extract_sql(text)
        results.append(check_sql_answer(pred_sql, gold_sql, db_path))
    return sum(results), comps, results, prompt


def batch_greedy_eval_spider(model, tok, problems, db_dir, batch_size=BATCH_SIZE):
    prepare_decoder_only_padding(tok)
    schema_cache = {}
    results = []
    t0 = time.time()

    for start in range(0, len(problems), batch_size):
        batch = problems[start:start + batch_size]
        prompts = []
        for d in batch:
            schema = _get_schema_cached(d["db_id"], db_dir, schema_cache)
            prompts.append(format_spider_prompt(d["question"], d["db_id"], schema, tok))

        inputs = tok(prompts, return_tensors="pt", truncation=True, max_length=2048,
                     padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for i, d in enumerate(batch):
            text = strip_generation_artifacts(
                tok.decode(outputs[i][input_len:], skip_special_tokens=True))
            pred_sql = extract_sql(text)
            db_path = _resolve_db_path(d["db_id"], db_dir)
            correct = check_sql_answer(pred_sql, d["query"], db_path)
            results.append({
                "question": d["question"][:200],
                "db_id": d["db_id"],
                "pass": correct,
                "gold": d["query"],
                "predicted": pred_sql,
            })

        done = min(start + batch_size, len(problems))
        if done % 50 < batch_size or done == len(problems):
            elapsed = time.time() - t0
            acc = sum(r["pass"] for r in results) / len(results) if results else 0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(problems) - done) / rate if rate > 0 else 0
            log.info(f"  Progress: {done}/{len(problems)}, acc={acc:.1%}, "
                     f"speed={rate:.1f} prob/s, ETA={eta:.0f}s")

    return results


def download_spider_dbs(db_dir: str) -> str:
    if os.path.isdir(db_dir) and len(os.listdir(db_dir)) > 100:
        log.info(f"Spider databases already present at {db_dir} ({len(os.listdir(db_dir))} dirs)")
        return db_dir

    log.info("Downloading Spider databases from Google Drive...")
    os.makedirs(db_dir, exist_ok=True)
    import subprocess
    import shutil
    import zipfile

    zip_path = "/tmp/spider.zip"
    extract_dir = "/tmp/spider_extracted"

    subprocess.run([
        "pip", "install", "-q", "gdown"
    ], check=True, capture_output=True)
    subprocess.run([
        "gdown", "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J", "-O", zip_path
    ], check=True, capture_output=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    src = None
    for subdir in ["spider", "spider_data", ""]:
        candidate = os.path.join(extract_dir, subdir, "database") if subdir else os.path.join(extract_dir, "database")
        if os.path.isdir(candidate):
            src = candidate
            break
    if src and os.path.isdir(src):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(db_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        raise RuntimeError(f"Spider database/ not found after extraction. Contents: {os.listdir(extract_dir)}")

    os.remove(zip_path)
    shutil.rmtree(extract_dir, ignore_errors=True)

    n_dbs = len([d for d in os.listdir(db_dir) if os.path.isdir(os.path.join(db_dir, d))])
    log.info(f"Spider databases ready at {db_dir} ({n_dbs} databases)")
    if n_dbs < 100:
        raise RuntimeError(f"Expected 166 Spider databases, only found {n_dbs}")
    return db_dir


def run(seed: int = 42, alpha: float = 0.07, n_steps: int = 20, lr: float = 1e-3,
        l2_lambda: float = 5e-4, n_train: int = 500, n_eval: int = 0,
        normalize_alpha: bool = True, db_dir: str = "/data/spider/database",
        output_dir: str = "/results/spider"):
    from datasets import load_dataset
    import random
    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_output_dir(output_dir)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model, tok, cfg, gdn_layers = load_model()
    dev = model.device
    prepare_decoder_only_padding(tok)

    db_dir = download_spider_dbs(db_dir)

    ds = load_dataset("xlangai/spider")
    train_raw = [dict(r) for r in ds["train"]][:n_train]
    all_dev = [dict(r) for r in ds["validation"]]
    eval_data = all_dev if n_eval == 0 else all_dev[:n_eval]

    log.info(f"Spider: Train={len(train_raw)}, Eval={len(eval_data)}")

    test_db_id = train_raw[0]["db_id"]
    test_db_path = _resolve_db_path(test_db_id, db_dir)
    test_schema = get_schema(test_db_path)
    log.info(f"Schema sanity check: db={test_db_id}, schema_len={len(test_schema)}, "
             f"tables={test_schema.count('CREATE TABLE')}")

    schema_cache = {}

    correct_cache_path = os.path.join(output_dir, "_correct_data.pkl")
    if os.path.exists(correct_cache_path):
        log.info("Loading cached correct solutions...")
        with open(correct_cache_path, "rb") as f:
            correct_data = pickle.load(f)
        log.info(f"  Loaded {len(correct_data)} correct solutions from cache")
    else:
        log.info(f"Collecting correct solutions from {len(train_raw)} training problems...")
        correct_data = []
        train_accuracy = 0
        t0 = time.time()
        for i, d in enumerate(train_raw):
            schema = _get_schema_cached(d["db_id"], db_dir, schema_cache)
            db_path = _resolve_db_path(d["db_id"], db_dir)
            n_pass, comps, exec_res, prompt = gen_and_eval_spider(
                model, tok, d["question"], d["db_id"], schema, d["query"], db_path, n=4)
            if n_pass > 0:
                train_accuracy += 1
            for c, r in zip(comps, exec_res):
                if r:
                    full = prompt + c
                    prompt_ids = tok(prompt, return_tensors="pt", truncation=True,
                                     max_length=2048)["input_ids"]
                    correct_data.append((full, prompt_ids.shape[1]))
                    break
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(train_raw) - i - 1) / rate if rate > 0 else 0
                log.info(f"  Processed {i+1}/{len(train_raw)}, found {len(correct_data)} correct "
                         f"(accuracy: {train_accuracy}/{i+1} = {train_accuracy/(i+1):.1%}), "
                         f"speed={rate:.1f} prob/s, ETA={eta:.0f}s")
        log.info(f"  Found {len(correct_data)} correct solutions from {len(train_raw)} problems "
                 f"(train accuracy: {train_accuracy}/{len(train_raw)} = "
                 f"{train_accuracy/len(train_raw):.1%})")
        atomic_write_pickle(correct_cache_path, correct_data)
        log.info(f"  Cached correct solutions to {correct_cache_path}")

    if len(correct_data) < 5:
        output = {"error": "insufficient_training_data", "n_correct": len(correct_data)}
        write_json(f"{output_dir}/results.json", output)
        return output

    baseline_path = os.path.join(output_dir, "_baseline.json")
    if os.path.exists(baseline_path):
        log.info("Loading cached baseline results...")
        with open(baseline_path) as f:
            baseline_greedy = json.load(f)
        log.info(f"  Loaded {len(baseline_greedy)} baseline results")
    else:
        log.info(f"Evaluating baseline (greedy, batched, batch_size={BATCH_SIZE}) "
                 f"on {len(eval_data)} problems...")
        baseline_greedy = batch_greedy_eval_spider(model, tok, eval_data, db_dir, batch_size=BATCH_SIZE)
        atomic_write_json(baseline_path, baseline_greedy)
        log.info("  Cached baseline results")

    bg_acc = sum(r["pass"] for r in baseline_greedy) / len(baseline_greedy)
    log.info(f"  Baseline greedy: {bg_acc:.1%}")

    state_path = os.path.join(output_dir, "_trained_states.pt")
    if os.path.exists(state_path):
        log.info("Loading cached trained states...")
        final_states = torch.load(state_path, map_location=dev, weights_only=True)
        log.info(f"  Loaded {len(final_states)} trained state tensors")
    else:
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
                sum(p.data.norm().item() for p in params) / len(states),
            ),
        )

        final_states = {i: s.data.clone() for i, s in states.items()}
        unpatch(model, originals)
        atomic_save_torch(state_path, final_states)
        log.info(f"  Saved trained states to {state_path}")

    tuned_path = os.path.join(output_dir, "_tuned.json")
    if os.path.exists(tuned_path):
        log.info("Loading cached tuned results...")
        with open(tuned_path) as f:
            tuned_greedy = json.load(f)
        log.info(f"  Loaded {len(tuned_greedy)} tuned results")
    else:
        scaled = scale_state_dict(
            final_states,
            alpha=alpha,
            normalize=normalize_alpha,
            warn_zero_norm=lambda index: log.warning(
                "Layer %s: zero-norm state, using raw alpha scaling", index),
        )
        if normalize_alpha:
            log.info(f"Alpha normalization: alpha={alpha}, per-layer state/||state||")
        originals = patch(model, scaled, gdn_layers)

        log.info(f"Evaluating state-tuned (alpha={alpha}, greedy, batched, batch_size={BATCH_SIZE}) "
                 f"on {len(eval_data)} problems...")
        tuned_greedy = batch_greedy_eval_spider(model, tok, eval_data, db_dir, batch_size=BATCH_SIZE)
        unpatch(model, originals)

        atomic_write_json(tuned_path, tuned_greedy)
        log.info("  Cached tuned results")

    tg_acc = sum(r["pass"] for r in tuned_greedy) / len(tuned_greedy)
    log.info(f"  Tuned greedy: {tg_acc:.1%}")

    improved = sum(1 for b, t in zip(baseline_greedy, tuned_greedy) if not b["pass"] and t["pass"])
    degraded = sum(1 for b, t in zip(baseline_greedy, tuned_greedy) if b["pass"] and not t["pass"])

    log.info(f"  Training: {len(correct_data)} correct from {len(train_raw)} problems")
    log.info(f"  Eval: {len(eval_data)} problems")
    log.info(f"  GREEDY: baseline={bg_acc:.1%}, tuned={tg_acc:.1%}, delta={tg_acc - bg_acc:+.1%}")
    log.info(f"  {improved} newly solved, {degraded} newly broken")

    db_ids = sorted(set(b["db_id"] for b in baseline_greedy))
    by_db = {}
    for db_id in db_ids:
        b_db = [b for b in baseline_greedy if b["db_id"] == db_id]
        t_db = [t for t in tuned_greedy if t["db_id"] == db_id]
        b_acc = sum(r["pass"] for r in b_db) / len(b_db) if b_db else 0
        t_acc = sum(r["pass"] for r in t_db) / len(t_db) if t_db else 0
        by_db[db_id] = {"n": len(b_db), "baseline": b_acc, "tuned": t_acc, "delta": t_acc - b_acc}

    per_problem = []
    for b, t in zip(baseline_greedy, tuned_greedy):
        per_problem.append({
            "question": b["question"],
            "db_id": b["db_id"],
            "gold": b["gold"],
            "baseline_correct": b["pass"],
            "tuned_correct": t["pass"],
            "baseline_predicted": b.get("predicted"),
            "tuned_predicted": t.get("predicted"),
        })

    output = {
        "benchmark": "spider",
        "seed": seed,
        "n_train": len(train_raw),
        "n_eval": len(eval_data),
        "n_correct_train": len(correct_data),
        "baseline_greedy": bg_acc,
        "tuned_greedy": tg_acc,
        "greedy_delta": tg_acc - bg_acc,
        "greedy_improved": improved,
        "greedy_degraded": degraded,
        "by_db": by_db,
        "hyperparams": {"n_steps": n_steps, "lr": lr, "l2_lambda": l2_lambda, "alpha": alpha,
                        "normalize_alpha": normalize_alpha, "n_train": n_train},
        "per_problem": per_problem,
    }
    write_json(f"{output_dir}/results.json", output)
    log.info(f"Results saved to {output_dir}/results.json")
    return output


if __name__ == "__main__":
    run()
