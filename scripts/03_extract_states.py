#!/usr/bin/env python3
"""Residual-stream states at the configured layer, LAST token (GPU).

  --dataset truth   the 24,992 MATH true/false statements (tokenized with BOS, as
                    statements given to the model from outside)
  --dataset swap    the 354 answer-swapped chain-of-thought prefixes (raw text, no
                    BOS, exactly as Thought Anchors' prompts); the last token is the
                    '}' closing the (possibly swapped) final \\boxed{}

Writes states/<dataset>_layer<L>.pt ([N, hidden] float32) and
states/<dataset>_meta.parquet (row-aligned).

Statements are batched only with others of identical token length, so no padding
is ever involved.

    python scripts/03_extract_states.py --dataset truth [--limit 200]
    python scripts/03_extract_states.py --dataset swap  [--limit 2]   # limit = CoTs
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truthsteer import paths  # noqa: E402
from truthsteer.model import last_token_states, load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["truth", "swap"], required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="truth: first N statements; swap: first N CoTs (all conditions)")
    ap.add_argument("--max-batch", type=int, default=32)
    args = ap.parse_args()
    c = paths.cfg()
    layer = c["layer"]

    if args.dataset == "truth":
        meta = pd.read_parquet(paths.STATEMENTS)
        if args.limit:
            meta = meta.iloc[:args.limit]
        texts, special = meta["statement"].tolist(), True
    else:
        meta = pd.read_parquet(paths.PREFIXES)
        if args.limit:
            keep = sorted(meta.cot_id.unique())[:args.limit]
            meta = meta[meta.cot_id.isin(keep)]
        meta = meta.sort_values(["cot_id", "condition"])
        texts, special = meta["prefix"].tolist(), False
    meta = meta.reset_index(drop=True)

    model, tok = load_model(c["model"])
    ids = [tok(t, add_special_tokens=special)["input_ids"] for t in texts]
    if args.dataset == "swap":
        n_expected = meta["n_prefix_tokens"].tolist()
        bad = [i for i, (a, n) in enumerate(zip(ids, n_expected)) if len(a) != n]
        assert not bad, f"tokenization differs from 02's for rows {bad[:5]}"

    by_len = defaultdict(list)
    for i, a in enumerate(ids):
        by_len[len(a)].append(i)
    order = [grp[j:j + args.max_batch] for grp in by_len.values()
             for j in range(0, len(grp), args.max_batch)]
    X = torch.empty(len(ids), model.config.hidden_size, dtype=torch.float32)
    t0, done = time.time(), 0
    batches = (torch.tensor([ids[i] for i in rows]) for rows in order)
    for rows, h in zip(order, last_token_states(model, batches, layer)):
        X[rows] = h
        done += len(rows)
        if done // 2000 != (done - len(rows)) // 2000:
            print(f"  {done}/{len(ids)}  {done / (time.time() - t0):.1f}/s")

    out = os.path.join(paths.STATES, f"{args.dataset}_layer{layer}.pt")
    paths.atomic_write(lambda tmp: torch.save(X, tmp), out)
    paths.atomic_write(lambda tmp: meta.drop(columns=["prefix"], errors="ignore").to_parquet(tmp),
                       os.path.join(paths.STATES, f"{args.dataset}_meta.parquet"))
    print(f"[03] {args.dataset}: {len(ids)} states at layer {layer} in "
          f"{time.time() - t0:.0f}s -> {out}")
    print(f"[03] peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
