"""Where everything lives. All data goes under $TRUTHSTEER_DATA (default ./data)."""
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.abspath(os.environ.get("TRUTHSTEER_DATA", os.path.join(ROOT, "data")))
# Downloaded/derived inputs; separate so a smoke run can read real inputs but write elsewhere.
RAW = os.path.abspath(os.environ.get("TRUTHSTEER_RAW", os.path.join(DATA, "raw")))


def cfg():
    with open(os.path.join(ROOT, "configs", "experiment.yaml")) as f:
        return yaml.safe_load(f)


def d(*parts):
    """Path under DATA."""
    return os.path.join(DATA, *parts)


TA_DIR = os.path.join(RAW, "thought_anchors")
STATEMENTS = os.path.join(RAW, "math_statements.parquet")
PREFIXES = os.path.join(RAW, "swap_prefixes.parquet")
STATES = d("states")                    # <dataset>_layer<L>.pt + <dataset>_meta.parquet
DIRECTIONS = d("directions")            # theta.pt, unsure.pt, unsure_<fold>.pt, folds.json
STEER = d("steering")                   # ladder.json, generations_<direction>/, grades_<direction>.parquet
REPORT = d("report")                    # figures/, direction_stats.json, steering_strength.csv


def ta_response_dir(c, solution_type, pid):
    t = c["thought_anchors"]
    return os.path.join(TA_DIR, "problems", t["model_dir"], t["temperature_dir"], solution_type, pid)


def ta_resample_file(c, solution_type, pid):
    t = c["thought_anchors"]
    return os.path.join(TA_DIR, "resamples", t["model_dir"], t["temperature_dir"], solution_type,
                        pid, "chunk_0", "solutions.json")


def atomic_write(writer, path):
    """writer(tmp_path), then rename: a preempted job never leaves a half-written file
    that a resumed run would take as finished."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    writer(tmp)
    os.replace(tmp, path)
