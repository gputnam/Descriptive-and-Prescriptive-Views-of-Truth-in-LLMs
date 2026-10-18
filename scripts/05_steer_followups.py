#!/usr/bin/env python3
"""Ask each of the 40 Thought Anchors responses "Are you sure?" while steering the
layer-L residual stream (GPU).

Step 0 (--measure-norm): m = mean ||h_L|| over the <think> tokens of the 40 base
responses -> steering/ladder.json. Steering strength is alpha = r * m, added along a
unit direction at every position (re-read prefix and generated tokens alike).

Then, per direction:
  --direction truth      theta, r from config r_values.truth (includes the r=0 baseline)
  --direction unsure     u, cross-fitted per problem (directions/folds.json),
                         r from config r_values.unsure; shares truth's r=0 baseline

PROMPT. Thought Anchors' prompts are raw text and DeepSeek's chat template deletes
the <think> block of earlier assistant turns, so the follow-up is hand-built:
full_cot (byte-identical) + EOS + <|User|> + question + <|Assistant|> + "<think>\\n".
The script asserts that appending the follow-up leaves the prefix tokenization
unchanged.

BATCHING. All rows for one response share the identical prefix, so no padding is
needed. Rows are ordered sample-major, so the r values of one sample share a
generate() call. Batch size is capped by a KV-cache budget
(192 KB/token for this model); the defaults suit an 80 GB GPU.

Output: steering/generations_<direction>/<problem>_<solution_type>.parquet, written
atomically and used as the resume marker.

    python scripts/05_steer_followups.py --measure-norm
    python scripts/05_steer_followups.py --direction truth
    python scripts/05_steer_followups.py --direction unsure
"""
import argparse
import hashlib
import json
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truthsteer import paths  # noqa: E402
from truthsteer.model import load_model, mean_hidden_norm, steer  # noqa: E402

KV_BYTES_PER_TOKEN = 48 * 8 * 128 * 2 * 2      # layers x kv heads x head dim x (k, v) x bf16
THINK_OPEN_ID, THINK_CLOSE_ID = 151648, 151649
LADDER = os.path.join(paths.STEER, "ladder.json")


def iter_responses(c, problem_ids, solution_types):
    for pid in problem_ids:
        for st in solution_types:
            d = paths.ta_response_dir(c, st, pid)
            base = json.load(open(os.path.join(d, "base_solution.json")))
            prob = json.load(open(os.path.join(d, "problem.json")))
            assert base["full_cot"] == base["prompt"] + base["solution"], f"format changed: {pid}/{st}"
            yield pid, st, base, prob


def build_followup(c, full_cot):
    f = c["followup"]
    return f"{full_cot}{f['eos_marker']}{f['user_marker']}{f['question']}{f['assistant_marker']}{f['think_open']}"


def measure_norm(c, model, tok, problem_ids):
    device = next(model.parameters()).device
    tot, n = 0.0, 0
    for pid, st, base, _ in iter_responses(c, problem_ids, c["thought_anchors"]["solution_types"]):
        ids = tok(base["full_cot"], return_tensors="pt", add_special_tokens=False)["input_ids"]
        seq = ids[0].tolist()
        a, b = seq.index(THINK_OPEN_ID) + 1, seq.index(THINK_CLOSE_ID)
        s, k = mean_hidden_norm(model, ids.to(device), c["layer"], start=a, end=b)
        tot, n = tot + s, n + k
        print(f"  {pid}/{st}: {k:5d} think tokens  mean||h||={s / k:7.2f}")
        torch.cuda.empty_cache()
    out = {"layer": c["layer"], "mean_hidden_norm": tot / n, "n_think_tokens": n,
           "problems": list(problem_ids)}
    paths.atomic_write(lambda tmp: json.dump(out, open(tmp, "w"), indent=2), LADDER)
    print(f"[05] m = mean||h_{c['layer']}|| over {n} think tokens = {tot / n:.2f} -> {LADDER}")


def pick_batch_size(prefix_len, max_new, max_batch, kv_budget_gb, n_r):
    b = max(1, min(max_batch, int(kv_budget_gb * 2**30) // ((prefix_len + max_new) * KV_BYTES_PER_TOKEN)))
    return (b // n_r) * n_r if b >= n_r else b     # keep a sample's r values together


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure-norm", action="store_true")
    ap.add_argument("--direction", choices=["truth", "unsure"])
    ap.add_argument("--r-values", type=float, nargs="*", default=None)
    ap.add_argument("--problems", nargs="*", default=None)
    ap.add_argument("--solution-types", nargs="*", default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--kv-budget-gb", type=float, default=36.0)
    args = ap.parse_args()
    c = paths.cfg()
    problem_ids = args.problems or c["thought_anchors"]["problem_ids"]
    solution_types = args.solution_types or c["thought_anchors"]["solution_types"]

    model, tok = load_model(c["model"])
    device = next(model.parameters()).device
    if args.measure_norm:
        measure_norm(c, model, tok, problem_ids)
        return
    if not args.direction:
        ap.error("--direction is required unless --measure-norm")

    m = json.load(open(LADDER))["mean_hidden_norm"]
    r_values = args.r_values if args.r_values is not None else c["r_values"][args.direction]
    k = args.samples or c["sampling"]["samples"]
    max_new = args.max_new_tokens or c["sampling"]["max_new_tokens"]
    if args.direction == "truth":
        theta_map, default_rel = {}, "directions/theta.pt"
    else:
        fj = json.load(open(os.path.join(paths.DIRECTIONS, "folds.json")))
        theta_map, default_rel = fj["problem_to_direction"], fj["default"]
    print(f"[05] {args.direction}: m={m:.2f}, r={r_values}, {len(problem_ids)} problems x "
          f"{len(solution_types)} types x k={k}, max_new={max_new}")

    gen_dir = os.path.join(paths.STEER, f"generations_{args.direction}")
    t_start, n_gen, n_trunc = time.time(), 0, 0
    for pid, st, base, prob in iter_responses(c, problem_ids, solution_types):
        out_path = os.path.join(gen_dir, f"{pid}_{st}.parquet")
        if os.path.exists(out_path):
            print(f"  [done already] {pid}/{st}")
            continue
        rel = theta_map.get(pid, default_rel)
        direction = torch.load(os.path.join(paths.DATA, rel))
        assert abs(float(direction.norm()) - 1) < 1e-4, f"{rel} is not unit-norm"

        ids = tok(build_followup(c, base["full_cot"]), return_tensors="pt",
                  add_special_tokens=False)["input_ids"].to(device)
        pre = tok(base["full_cot"], add_special_tokens=False)["input_ids"]
        assert ids[0, :len(pre)].tolist() == pre, f"{pid}/{st}: follow-up changed the prefix tokens"
        plen = ids.shape[1]
        jobs = [(s, r) for s in range(k) for r in r_values]
        B = pick_batch_size(plen, max_new, args.max_batch, args.kv_budget_gb, len(r_values))
        rows, t0 = [], time.time()
        for j0 in range(0, len(jobs), B):
            chunk = jobs[j0:j0 + B]
            batch = ids.repeat(len(chunk), 1)
            seed = int(hashlib.sha256(f"{pid}|{st}|batch{j0 // B}|B{B}|{args.direction}"
                                      .encode()).hexdigest()[:8], 16)
            torch.manual_seed(seed)
            with steer(model, c["layer"], direction, [r * m for _, r in chunk]), torch.no_grad():
                gen = model.generate(input_ids=batch, attention_mask=torch.ones_like(batch),
                                     max_new_tokens=max_new, do_sample=True,
                                     temperature=c["sampling"]["temperature"],
                                     top_p=c["sampling"]["top_p"], pad_token_id=tok.eos_token_id)
            for i, (s, r) in enumerate(chunk):
                toks = gen[i, plen:].tolist()
                trunc = tok.eos_token_id not in toks
                if not trunc:
                    toks = toks[:toks.index(tok.eos_token_id)]
                n_gen, n_trunc = n_gen + 1, n_trunc + trunc
                rows.append({"problem_id": pid, "solution_type": st, "direction": args.direction,
                             "direction_path": rel, "r": r, "alpha": r * m, "sample_idx": s,
                             "seed": seed, "batch_size": B, "prefix_tokens": plen,
                             "n_new_tokens": len(toks), "truncated": bool(trunc),
                             "response_text": tok.decode(toks, skip_special_tokens=False),
                             "original_answer": str(base["answer"]).strip(),
                             "gt_answer": str(prob["gt_answer"]).strip(),
                             "is_correct_original": bool(base["is_correct"]),
                             "problem": prob["problem"]})
            del gen, batch
            torch.cuda.empty_cache()
        paths.atomic_write(lambda tmp: pd.DataFrame(rows).to_parquet(tmp), out_path)
        ntok = sum(r["n_new_tokens"] for r in rows)
        print(f"  {pid}/{st}: prefix={plen} B={B} -> {len(rows)} gens, {ntok} tokens in "
              f"{(time.time() - t0) / 60:.1f} min, truncated={sum(r['truncated'] for r in rows)}")
    print(f"[05] {n_gen} generations in {(time.time() - t_start) / 60:.1f} min, "
          f"{n_trunc} hit the token cap; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
