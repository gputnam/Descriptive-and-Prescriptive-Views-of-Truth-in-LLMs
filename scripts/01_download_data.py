#!/usr/bin/env python3
"""Download the inputs (CPU, ~100 MB).

1. Thought Anchors (uzaymacar/math-rollouts), DeepSeek-R1-Distill-Qwen-14B, 20 MATH
   problems: problem.json + base_solution.json for the correct and incorrect base
   solutions (the 40 responses of the steering test), and each one's chunk_0
   resamples (100 fresh solutions apiece, the source of extra correct chains of
   thought and of the model's own wrong answers for the answer swap).

2. The full MATH benchmark (EleutherAI/hendrycks_math), turned into true/false
   statements for the Outside Truth direction:
       "Question: {problem}\\nAnswer: The result is {answer}."
   true = the reference answer (last \\boxed{} of the solution); false = a different
   problem's answer, drawn with a seeded RNG.

    python scripts/01_download_data.py [--only ta|math]
"""
import argparse
import os
import random
import sys

import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truthsteer import paths  # noqa: E402
from truthsteer.swap import extract_boxed  # noqa: E402


def download_ta(c):
    t = c["thought_anchors"]
    base = f"{t['model_dir']}/{t['temperature_dir']}"
    n = 0
    for pid in t["problem_ids"]:
        for st in t["solution_types"]:
            for fn in ("problem.json", "base_solution.json"):
                hf_hub_download(t["repo_id"], f"{base}/{st}/{pid}/{fn}", repo_type=t["repo_type"],
                                local_dir=os.path.join(paths.TA_DIR, "problems"))
            hf_hub_download(t["repo_id"], f"{base}/{st}/{pid}/chunk_0/solutions.json",
                            repo_type=t["repo_type"], local_dir=os.path.join(paths.TA_DIR, "resamples"))
            n += 1
    print(f"[01] Thought Anchors: {n} responses + resamples -> {paths.TA_DIR}")


def build_math_statements(c):
    m = c["math_statements"]
    dfs = []
    for cat in m["categories"]:
        for split in ["train", "test"]:
            p = hf_hub_download(m["repo_id"], repo_type="dataset",
                                filename=f"{cat}/{split}-00000-of-00001.parquet")
            df = pd.read_parquet(p)
            df["category"], df["split"] = cat, split
            dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    full = full[full["level"] != "Level ?"].reset_index(drop=True)
    full["answer"] = full["solution"].apply(extract_boxed)
    full = full.dropna(subset=["answer"]).reset_index(drop=True)
    full["problem_id"] = [f"math_{i:06d}" for i in range(len(full))]

    rng = random.Random(m["seed"])
    answers = full["answer"].tolist()
    wrong = []
    for i in range(len(full)):
        j, tries = rng.randrange(len(full)), 0
        while answers[j].strip() == answers[i].strip() and tries < 20:
            j, tries = rng.randrange(len(full)), tries + 1
        wrong.append(answers[j])
    full["wrong_answer"] = wrong

    rows = []
    for _, p in full.iterrows():
        q = p["problem"].strip()
        for label, ans in ((1, p["answer"]), (0, p["wrong_answer"])):
            rows.append({"statement": m["template"].format(problem=q, answer=ans), "label": label,
                         "problem_id": p["problem_id"], "answer": ans, "level": p["level"],
                         "type": p["category"], "split": p["split"]})
    out = pd.DataFrame(rows)
    paths.atomic_write(lambda tmp: out.to_parquet(tmp), paths.STATEMENTS)
    print(f"[01] MATH: {len(out)} statements from {len(full)} problems "
          f"({out.label.sum()} true) -> {paths.STATEMENTS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ta", "math"], default=None)
    args = ap.parse_args()
    c = paths.cfg()
    if args.only in (None, "ta"):
        download_ta(c)
    if args.only in (None, "math"):
        build_math_statements(c)


if __name__ == "__main__":
    main()
