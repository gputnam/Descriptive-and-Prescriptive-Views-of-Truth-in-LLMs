#!/usr/bin/env python3
"""Build the answer-swapped chains of thought for the Reasoned Truth direction
(CPU; needs only the tokenizer).

For each of the 20 problems, take up to `swap.per_problem` CORRECT chains of thought
(Thought Anchors' correct base solution, then chunk_0 resamples in a seeded order)
whose <think> block ends its reasoning with \\boxed{X}, cut each right after that box,
and build three conditions:
    control       X kept (the correct answer)
    model_wrong   X := the model's own most frequent wrong answer to this problem
                  (over its 200 chunk_0 rollouts)
    random_wrong  X := another problem's most frequent correct answer (seeded)
Splice check: every token before the '\\boxed{' boundary token must be identical
across the three conditions, or the CoT is dropped.

    python scripts/02_build_swap_prefixes.py
"""
import collections
import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truthsteer import paths  # noqa: E402
from truthsteer.swap import build_prefix, norm_answer  # noqa: E402


def main():
    c = paths.cfg()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained(c["model"])

    def encode(s):
        return tok.encode(s, add_special_tokens=False).ids

    pids = sorted(c["thought_anchors"]["problem_ids"], key=lambda p: int(p.split("_")[1]))
    info = {}
    for pid in pids:
        d = paths.ta_response_dir(c, "correct_base_solution", pid)
        prob = json.load(open(os.path.join(d, "problem.json")))
        ta = json.load(open(os.path.join(d, "base_solution.json")))
        rolls = []
        for st in c["thought_anchors"]["solution_types"]:
            rolls += json.load(open(paths.ta_resample_file(c, st, pid)))
        correct = [r for r in rolls if r.get("is_correct") is True]
        correct_forms = {norm_answer(prob["gt_answer"])} | {norm_answer(r["answer"]) for r in correct}
        wrong = collections.Counter(
            r["answer"] for r in rolls
            if r.get("is_correct") is False and r["answer"]
            and norm_answer(r["answer"]) not in correct_forms)
        info[pid] = dict(prob=prob, ta=ta, correct=correct, correct_forms=correct_forms,
                         model_wrong=wrong.most_common(1)[0][0] if wrong else None,
                         model_wrong_count=wrong.most_common(1)[0][1] if wrong else 0,
                         n_rollouts=len(rolls),
                         modal_correct=collections.Counter(r["answer"] for r in correct).most_common(1)[0][0])

    rng = random.Random(0)
    for pid in pids:
        others = [q for q in pids if q != pid]
        rng.shuffle(others)
        for q in others:
            n = norm_answer(info[q]["modal_correct"])
            if n not in info[pid]["correct_forms"] and n != norm_answer(info[pid]["model_wrong"] or ""):
                info[pid]["random_wrong"], info[pid]["random_wrong_from"] = info[q]["modal_correct"], q
                break

    rows, dropped = [], collections.Counter()
    for pid in pids:
        I = info[pid]
        cands = [("ta_base", I["ta"]["full_cot"])]
        pool = list(enumerate(I["correct"]))
        random.Random(f"select|{pid}").shuffle(pool)
        cands += [(f"resample_{i}", r["full_cot"]) for i, r in pool]
        kept = 0
        for src, cot in cands:
            if kept >= c["swap"]["per_problem"]:
                break
            b = build_prefix(cot, "")
            if b is None:
                dropped["no_think_box"] += 1
                continue
            _, common, box = b
            if norm_answer(box) not in I["correct_forms"]:
                dropped["box_not_final_answer"] += 1
                continue
            answers = {"control": box, "model_wrong": I["model_wrong"],
                       "random_wrong": I.get("random_wrong")}
            if any(v is None for v in answers.values()):
                dropped["no_wrong_answer"] += 1
                continue
            # The '{' of '\boxed{' can merge with the answer's first character, so that
            # boundary token may differ between conditions; everything before it must not.
            stem = encode(common)[:-1]
            built = {}
            for cond, a in answers.items():
                prefix = common + a + "}"
                ids = encode(prefix)
                if ids[:len(stem)] != stem:
                    break
                built[cond] = (prefix, len(ids), len(ids) - len(stem))
            if len(built) < len(answers):
                dropped["splice_changed_prefix"] += 1
                continue
            for cond, (prefix, n_tok, n_tail) in built.items():
                rows.append({"cot_id": f"{pid}__{src}", "problem_id": pid, "source": src,
                             "condition": cond, "inserted_answer": answers[cond],
                             "correct_answer": box, "gt_answer": I["prob"]["gt_answer"],
                             "random_wrong_from": I.get("random_wrong_from"),
                             "prefix": prefix, "n_prefix_tokens": n_tok,
                             "n_answer_tail_tokens": n_tail})
            kept += 1
        print(f"  {pid}: kept {kept} CoTs  model_wrong={I['model_wrong']!r} "
              f"({I['model_wrong_count']}/{I['n_rollouts']})  random_wrong={I.get('random_wrong')!r}")
    df = pd.DataFrame(rows)
    paths.atomic_write(lambda tmp: df.to_parquet(tmp), paths.PREFIXES)
    print(f"[02] {df.cot_id.nunique()} CoTs x {len(c['swap']['conditions'])} conditions = "
          f"{len(df)} prefixes -> {paths.PREFIXES}   (dropped: {dict(dropped)})")


if __name__ == "__main__":
    main()
