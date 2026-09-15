#!/usr/bin/env python3
"""Report one-click SPARC matrix progress without reading prediction outcomes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs/sparc_untouched2k_five_backend_matrix.frozen.json"
DEFAULT_LOG = ROOT / "results/sparc_untouched2k_five_backend_runs/command_log.json"
BACKENDS = ("clip", "tent", "clipartt", "tda", "mint")
# Current native-smoke extrapolation, including a conservative initialization allowance.
FALLBACK_SECONDS_PER_JOB = {
    "clip": 54.0,
    "tent": 176.0,
    "clipartt": 493.0,
    "tda": 58.0,
    "mint": 80.0,
}


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def jobs(matrix: dict) -> list[dict]:
    answer = []
    for condition in matrix.get("conditions", []):
        for side in ("parent", "plugin"):
            answer.append({
                "backend": condition["backend"],
                "scenario": condition["scenario"],
                "side": side,
                "output": Path(condition[side]["output_dir"]),
            })
    return answer


def latest_log_entries(path: Path) -> dict[tuple[str, str, str], dict]:
    if not path.is_file():
        return {}
    entries = load(path).get("commands", [])
    latest = {}
    for entry in entries:
        key = (entry.get("backend"), entry.get("scenario"), entry.get("side"))
        latest[key] = entry
    return latest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--command-log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    matrix_jobs = jobs(load(args.manifest.resolve()))
    if len(matrix_jobs) != 150:
        raise RuntimeError(f"expected 150 jobs, found {len(matrix_jobs)}")
    logs = latest_log_entries(args.command_log.resolve())
    state_counts: Counter[str] = Counter()
    backend_counts: dict[str, Counter[str]] = defaultdict(Counter)
    observed_durations: dict[str, list[float]] = defaultdict(list)
    fractional_done = 0.0

    for job in matrix_jobs:
        key = (job["backend"], job["scenario"], job["side"])
        output = job["output"]
        entry = logs.get(key, {})
        if (output / "complete.json").is_file():
            state, fraction = "complete", 1.0
        elif (output / "progress.json").is_file():
            progress = load(output / "progress.json")
            fraction = min(1.0, max(0.0, float(progress.get("next_batch", 0)) / 100.0))
            state = "running_or_resumable"
        elif (output / "protocol.json").is_file():
            state, fraction = "starting_or_resumable", 0.0
        elif entry.get("status") == "failed":
            state, fraction = "failed", 0.0
        else:
            state, fraction = "pending", 0.0
        state_counts[state] += 1
        backend_counts[job["backend"]][state] += 1
        fractional_done += fraction
        if state == "complete" and isinstance(entry.get("started_unix"), (int, float)) \
                and isinstance(entry.get("completed_unix"), (int, float)):
            duration = float(entry["completed_unix"]) - float(entry["started_unix"])
            if duration > 0:
                observed_durations[job["backend"]].append(duration)

    remaining_seconds = 0.0
    for job in matrix_jobs:
        output = job["output"]
        if (output / "complete.json").is_file():
            continue
        progress_fraction = 0.0
        if (output / "progress.json").is_file():
            progress_fraction = min(
                1.0, max(0.0, float(load(output / "progress.json").get("next_batch", 0)) / 100.0)
            )
        samples = sorted(observed_durations[job["backend"]])
        typical = samples[len(samples) // 2] if samples else FALLBACK_SECONDS_PER_JOB[job["backend"]]
        remaining_seconds += typical * (1.0 - progress_fraction)

    now = time.time()
    payload = {
        "status": "progress_only_no_prediction_outcomes_read",
        "jobs_total": len(matrix_jobs),
        "jobs_complete": state_counts["complete"],
        "equivalent_jobs_complete": round(fractional_done, 2),
        "progress_percent": round(100.0 * fractional_done / len(matrix_jobs), 2),
        "states": dict(sorted(state_counts.items())),
        "by_backend": {backend: dict(sorted(backend_counts[backend].items())) for backend in BACKENDS},
        "estimated_remaining_hours": round(remaining_seconds / 3600.0, 2),
        "estimated_finish_unix": now + remaining_seconds,
        "estimated_finish_local": datetime.fromtimestamp(now + remaining_seconds).astimezone().isoformat(
            timespec="seconds"
        ),
        "estimate_basis": "per-backend completed-job median, otherwise frozen native-smoke extrapolation",
        "command_log": str(args.command_log.resolve()),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
