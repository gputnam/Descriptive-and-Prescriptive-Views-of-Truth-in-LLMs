#!/usr/bin/env python3
"""Grade the steered follow-ups with an LLM (needs ANTHROPIC_API_KEY; no GPU).

Only Fig 6 uses the grades (did an initially INCORRECT answer end on the ground
truth?), so by default only incorrect-base responses are graded. Pass
--solution-types correct_base_solution incorrect_base_solution to grade all.

Grades go to steering/grades_<direction>.parquet, keyed by
(problem_id, solution_type, r, sample_idx). The run is resume-safe: keys that are
already graded are skipped. An item the grader cannot answer is logged and retried
on the next run.

    python scripts/06_grade_followups.py --direction truth
    python scripts/06_grade_followups.py --direction unsure
"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truthsteer import paths  # noqa: E402
from truthsteer.grader import grade_one, resolve_auth  # noqa: E402

KEYS = ["problem_id", "solution_type", "r", "sample_idx"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["truth", "unsure"], required=True)
    ap.add_argument("--solution-types", nargs="*", default=["incorrect_base_solution"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    gcfg = paths.cfg()["grader"]
    base, headers = resolve_auth(gcfg)

    files = sorted(glob.glob(os.path.join(paths.STEER, f"generations_{args.direction}", "*.parquet")))
    if not files:
        sys.exit(f"no generations for {args.direction}")
    gens = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    gens = gens[gens.solution_type.isin(args.solution_types)]
    out_path = os.path.join(paths.STEER, f"grades_{args.direction}.parquet")
    done = pd.read_parquet(out_path) if os.path.exists(out_path) else pd.DataFrame(columns=KEYS)
    gens = gens.merge(done[KEYS].assign(_g=1), on=KEYS, how="left")
    gens = gens[gens._g.isna()].drop(columns="_g")
    if args.limit:
        gens = gens.head(args.limit)
    print(f"[06] {len(done)} graded already, {len(gens)} to grade with {gcfg['model']}")
    if gens.empty:
        return

    sess = requests.Session()

    def one(rec):
        try:
            g = grade_one(sess, base, headers, gcfg, rec["problem"], rec["original_answer"],
                          rec["gt_answer"], rec["response_text"])
        except Exception as e:  # noqa: BLE001 - one bad item must not stop the run
            print(f"  [skip] {[rec[k] for k in KEYS]}: {str(e)[:160]}")
            return None
        u = g.get("_usage", {})
        return {**{k: rec[k] for k in KEYS}, "verdict": g["verdict"],
                "final_answer": g.get("final_answer"), "matches_original": g.get("matches_original"),
                "matches_gt": g.get("matches_gt"), "grader_reasoning": g.get("reasoning"),
                "grader_model": gcfg["model"], "input_tokens": u.get("input_tokens"),
                "output_tokens": u.get("output_tokens")}

    def save(rows):
        allg = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) else pd.DataFrame(rows)
        paths.atomic_write(lambda tmp: allg.to_parquet(tmp), out_path)

    out, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=gcfg["concurrency"]) as ex:
        for i, r in enumerate(ex.map(one, gens.to_dict("records")), 1):
            if r is not None:
                out.append(r)
            if i % 100 == 0:
                save(out)
                print(f"  graded {i}/{len(gens)} ({(time.time() - t0) / i:.2f} s/call)")
    save(out)
    new = pd.DataFrame(out)
    print(f"[06] {len(new)} graded, {len(gens) - len(new)} failed (rerun to retry) -> {out_path}")
    if len(new):
        print(new.verdict.value_counts().to_string())


if __name__ == "__main__":
    main()
