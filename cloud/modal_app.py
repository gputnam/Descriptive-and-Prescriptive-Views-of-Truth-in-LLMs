"""Run the GPU half of the pipeline on Modal (https://modal.com).

Everything runs against one Modal Volume (default `truthsteer-data`, override with
$TRUTHSTEER_VOLUME), mounted at /data:
    /data/hf           Hugging Face cache (the ~28 GB model weights, downloaded once)
    /data/truthsteer   the pipeline's $TRUTHSTEER_DATA (raw/, states/, directions/, steering/)
    /data/truthsteer-smoke   outputs of --task smoke

Usage, from the repository root:
    modal run cloud/modal_app.py --task prepare      # CPU: weights, Thought Anchors, MATH, swap prefixes
    modal run cloud/modal_app.py --task smoke        # one GPU container, every step at tiny scale (~$0.5)
    modal run --detach cloud/modal_app.py --task all # full run: states -> directions -> norm -> steering
    modal run cloud/modal_app.py --task fetch        # print the commands that copy results back

Stages of --task all (each runs only after the previous one finishes):
    1. GPU, in parallel: 03 --dataset truth  |  03 --dataset swap, then 05 --measure-norm
    2. CPU:              04_build_directions
    3. GPU, in parallel: 05 --direction truth / unsure, `--shards` jobs per direction
Every script resumes where it stopped, so rerunning a task after a preemption or a
timeout only redoes unfinished work. Use --detach for long runs: the remote
`orchestrate` function keeps going if this machine disconnects.

Cost: GPU seconds are logged per job to cloud/ledger.jsonl at approximate list prices
and a launch is refused once the ledger passes --max-usd. `modal billing` remains the
authoritative number.
"""
import json
import os
import subprocess
import time

import modal

APP_ROOT = "/root/truthsteer"
VOL_MOUNT = "/data"
DATA = f"{VOL_MOUNT}/truthsteer"
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
VOLUME = os.environ.get("TRUTHSTEER_VOLUME", "truthsteer-data")
GPU = ["A100-80GB", "H100"]                       # fallback order
GPU_USD_PER_H = {"A100": 2.50, "H100": 3.95}      # approximate list prices
OVERHEAD_USD_PER_H = 4 * 0.047 + 32 * 0.008       # 4 cores + 32 GiB
LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ledger.jsonl")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.14.1", "transformers==5.19.0", "accelerate==1.15.0",
                 "safetensors==0.8.0", "tokenizers==0.23.3", "numpy==2.4.6",
                 "pandas==3.0.6", "pyarrow", "pyyaml", "scipy", "huggingface_hub", "requests")
    .env({"TRUTHSTEER_DATA": DATA, "HF_HOME": f"{VOL_MOUNT}/hf", "HF_HUB_CACHE": f"{VOL_MOUNT}/hf/hub",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONUNBUFFERED": "1"})
    .add_local_dir("truthsteer", f"{APP_ROOT}/truthsteer")
    .add_local_dir("scripts", f"{APP_ROOT}/scripts")
    .add_local_dir("configs", f"{APP_ROOT}/configs")
)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
app = modal.App("truthsteer", image=image)


def _run_cmds(cmds, name, env):
    """Run scripts sequentially, streaming output, committing the volume every 2 min
    so a preempted container loses at most its current response."""
    import threading
    stop = threading.Event()

    def committer():
        while not stop.wait(120):
            try:
                vol.commit()
            except Exception as e:  # noqa: BLE001 - never kill a run over a commit
                print(f"[modal] commit failed: {e}")

    threading.Thread(target=committer, daemon=True).start()
    t0, rc = time.time(), 0
    for argv in cmds:
        print(f"[{name}] $ python {' '.join(argv)}", flush=True)
        rc = subprocess.run(["python", *argv], cwd=APP_ROOT, env={**os.environ, **env}).returncode
        if rc != 0:
            break
    stop.set()
    vol.commit()
    return {"name": name, "rc": rc, "seconds": time.time() - t0, "cmds": cmds}


@app.function(volumes={VOL_MOUNT: vol}, gpu=GPU, cpu=4.0, memory=32768, timeout=6 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def gpu_job(cmds: list, name: str, env: dict):
    import torch
    res = _run_cmds(cmds, name, env)
    res["gpu"] = torch.cuda.get_device_name(0)
    return res


@app.function(volumes={VOL_MOUNT: vol}, cpu=2.0, memory=16384, timeout=3 * 3600)
def cpu_job(cmds: list, name: str, env: dict):
    return {**_run_cmds(cmds, name, env), "gpu": "cpu"}


@app.function(volumes={VOL_MOUNT: vol}, cpu=2.0, memory=8192, timeout=3600)
def download_weights():
    from huggingface_hub import snapshot_download
    t = time.time()
    p = snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])
    vol.commit()
    return f"{p} in {time.time() - t:.0f}s"


@app.function(volumes={VOL_MOUNT: vol}, cpu=0.25, memory=512, timeout=24 * 3600)
def orchestrate(stages: list):
    """Run stages in order, the jobs of each stage in parallel, REMOTELY - so a
    `modal run --detach` survives the client going away."""
    out = []
    for stage in stages:
        calls = [(gpu_job if kind == "gpu" else cpu_job).spawn(cmds, name, env)
                 for kind, cmds, name, env in stage]
        for (kind, cmds, name, _), c in zip(stage, calls):
            try:
                res = c.get()
            except Exception as e:  # noqa: BLE001
                res = {"name": name, "rc": -1, "seconds": 0.0, "cmds": cmds, "gpu": kind, "error": repr(e)}
            print(f"[orchestrate] {name}: rc={res['rc']} {res['seconds'] / 60:.1f} min on {res['gpu']}")
            out.append(res)
        if any(r["rc"] != 0 for r in out[-len(stage):]):
            print("[orchestrate] a job failed - stopping before the next stage")
            break
    return out


# --------------------------------------------------------------------------- #
# local side
# --------------------------------------------------------------------------- #
S03, S04, S05 = "scripts/03_extract_states.py", "scripts/04_build_directions.py", "scripts/05_steer_followups.py"
PREPARE = [["scripts/01_download_data.py"], ["scripts/02_build_swap_prefixes.py"]]
# The smoke test reads the real inputs but writes to its own tree, so it never
# leaves partial outputs that a full run would mistake for finished work.
SMOKE_ENV = {"TRUTHSTEER_DATA": f"{VOL_MOUNT}/truthsteer-smoke", "TRUTHSTEER_RAW": f"{DATA}/raw"}
SMOKE_STEER = ["--problems", "problem_2189", "--solution-types", "incorrect_base_solution",
               "--samples", "1", "--max-new-tokens", "256"]


def _usd(res):
    if res["gpu"] == "cpu":
        return res["seconds"] / 3600 * 0.2
    rate = next((v for k, v in GPU_USD_PER_H.items() if k in res["gpu"]), max(GPU_USD_PER_H.values()))
    return res["seconds"] / 3600 * (rate + OVERHEAD_USD_PER_H)


def _spent():
    if not os.path.exists(LEDGER):
        return 0.0
    return sum(json.loads(line)["usd"] for line in open(LEDGER) if line.strip())


def _shards(items, n):
    return [items[i::n] for i in range(n) if items[i::n]]


@app.local_entrypoint()
def main(task: str, shards: int = 4, max_usd: float = 60.0):
    if task == "fetch":
        print("Copy results to ./data (then run 06-08 locally):")
        for sub in ["states", "directions", "steering"]:
            print(f"  modal volume get {VOLUME} /truthsteer/{sub} data/")
        return
    spent = _spent()
    print(f"[ledger] ~${spent:.2f} logged so far (cap ${max_usd:.2f})")
    if spent >= max_usd:
        raise SystemExit("[ledger] cap reached - raise --max-usd to continue")

    if task == "prepare":
        print(download_weights.remote())
        stages = [[("cpu", PREPARE, "prepare", {})]]
    elif task == "smoke":
        stages = [[("gpu", [
            [S03, "--dataset", "truth", "--limit", "200"],
            [S03, "--dataset", "swap", "--limit", "2"],
            [S04],
            [S05, "--measure-norm"],
            [S05, "--direction", "truth", "--r-values", "0", "-0.4", "0.4", *SMOKE_STEER],
            [S05, "--direction", "unsure", "--r-values", "0.4", *SMOKE_STEER],
        ], "smoke", SMOKE_ENV)]]
    elif task == "all":
        import yaml
        problems = yaml.safe_load(open("configs/experiment.yaml"))["thought_anchors"]["problem_ids"]
        stages = [
            [("gpu", [[S03, "--dataset", "truth"]], "states_truth", {}),
             ("gpu", [[S03, "--dataset", "swap"], [S05, "--measure-norm"]], "states_swap_norm", {})],
            [("cpu", [[S04]], "directions", {})],
            [("gpu", [[S05, "--direction", d, "--problems", *ps]], f"steer_{d}_{i}", {})
             for d in ("truth", "unsure") for i, ps in enumerate(_shards(problems, shards))],
        ]
    else:
        raise SystemExit(f"unknown task {task!r}")

    print(f"[modal] {sum(len(s) for s in stages)} job(s) in {len(stages)} stage(s) on volume {VOLUME}")
    try:
        results = orchestrate.remote(stages)
    except Exception as e:  # noqa: BLE001
        print(f"[modal] lost the orchestrator ({e!r}); with --detach the jobs keep running. "
              f"Check `modal app logs` and `modal billing`.")
        return
    with open(LEDGER, "a") as f:
        for res in results:
            usd = _usd(res)
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "task": task,
                                "name": res["name"], "gpu": res["gpu"], "rc": res["rc"],
                                "seconds": round(res["seconds"], 1), "usd": round(usd, 3)}) + "\n")
            print(f"[modal] {res['name']}: rc={res['rc']} {res['seconds'] / 60:.1f} min ~${usd:.2f}")
    print(f"[ledger] ~${_spent():.2f} logged in total")
