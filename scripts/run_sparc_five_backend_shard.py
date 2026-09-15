#!/usr/bin/env python3
"""Run an even, pair-aligned slice of the locked 150-job SPARC matrix.

This worker exists for execution on a second machine.  It preserves the full
frozen matrix and external lock, requires the exact locked software/GPU
environment, and writes the same per-job output layout as the primary runner.
It deliberately does not summarize partial evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXECUTOR = ROOT / "scripts/run_sparc_five_backend_matrix.py"


def load_executor():
    spec = importlib.util.spec_from_file_location("sparc_locked_matrix_executor", EXECUTOR)
    if spec is None or spec.loader is None:
        raise ImportError(EXECUTOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def result_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    for condition_index, condition in enumerate(conditions):
        for side in ("parent", "plugin"):
            jobs.append({
                "condition_index": condition_index,
                "backend": condition["backend"],
                "scenario": condition["scenario"],
                "side": side,
                "manifest": condition[side]["manifest"],
                "canonical_output_dir": condition[side]["output_dir"],
            })
    return jobs


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0 <= args.start_job < args.stop_job <= 150:
        raise ValueError("job slice must satisfy 0 <= start < stop <= 150")
    if args.start_job % 2 or args.stop_job % 2:
        raise ValueError("job slice must begin and end on a parent/plugin pair boundary")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"

    locked = load_executor()
    matrix_path, lock_path = args.manifest.resolve(), args.lock.resolve()
    audit = locked.audit_matrix(matrix_path)
    locked.validate_lock(lock_path, matrix_path, audit["runner"], args.python)
    jobs = flatten(audit["conditions"])
    selected = jobs[args.start_job:args.stop_job]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / f"shard_command_log_{args.start_job}_{args.stop_job}.json"
    log: list[dict[str, Any]] = []
    environment = os.environ.copy()

    pair_results: dict[int, dict[str, dict[str, Any]]] = {}
    for absolute_index, job in enumerate(selected, start=args.start_job):
        manifest_path = Path(job["manifest"])
        canonical_name = Path(job["canonical_output_dir"]).name
        output = output_root / canonical_name
        side = str(job["side"])
        action = locked.output_action(manifest_path, output, require_parent=side == "parent")
        command = [args.python, str(audit["runner"]), "--manifest", str(manifest_path),
                   "--output-dir", str(output)]
        entry = {
            "absolute_job_index": absolute_index,
            "backend": job["backend"], "scenario": job["scenario"], "side": side,
            "manifest": str(manifest_path), "output_dir": str(output),
            "canonical_output_dir": job["canonical_output_dir"],
            "action": action, "command": command, "started_unix": time.time(),
            "status": "verified_skip" if action == "verified_skip" else "launched",
        }
        log.append(entry)
        atomic_json(log_path, {"schema_version": 1, "commands": log})
        if action != "verified_skip":
            completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
            if completed.returncode:
                entry.update(status="failed", returncode=completed.returncode,
                             completed_unix=time.time())
                atomic_json(log_path, {"schema_version": 1, "commands": log})
                raise locked.GateError(
                    f"shard job failed: {absolute_index}/{job['backend']}/{job['scenario']}/{side}"
                )
            entry["status"] = "completed"
        result = locked.audit_output(manifest_path, output, require_parent=side == "parent")
        pair_results.setdefault(int(job["condition_index"]), {})[side] = result
        entry["result_sha256"] = result_sha(output / "result.json")
        entry["completed_unix"] = time.time()
        atomic_json(log_path, {"schema_version": 1, "commands": log})

        if side == "plugin":
            pair = pair_results[int(job["condition_index"])]
            left, right = pair["parent"]["records"], pair["plugin"]["records"]
            if [[row[name] for name in locked.PAIR_FIELDS] for row in left] != [
                [row[name] for name in locked.PAIR_FIELDS] for row in right
            ]:
                raise locked.GateError(
                    f"shard pair mismatch: {job['backend']}/{job['scenario']}"
                )

    report = {
        "schema_version": 1,
        "status": "completed_locked_pair_aligned_matrix_shard",
        "start_job_inclusive": args.start_job,
        "stop_job_exclusive": args.stop_job,
        "jobs": len(selected), "pairs": len(selected) // 2,
        "gpu_argument": str(args.gpu),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "matrix": str(matrix_path), "matrix_sha256": locked.sha256(matrix_path),
        "lock": str(lock_path), "lock_sha256": locked.sha256(lock_path),
        "runner_sha256": locked.sha256(audit["runner"]),
        "executor_sha256": locked.sha256(EXECUTOR),
        "environment": locked.probe_environment(args.python),
        "output_root": str(output_root),
        "command_log": str(log_path), "command_log_sha256": result_sha(log_path),
        "outputs": [
            {"absolute_job_index": row["absolute_job_index"],
             "canonical_output_dir": row["canonical_output_dir"],
             "output_dir": row["output_dir"], "result_sha256": row["result_sha256"]}
            for row in log
        ],
        "partial_summary_generated": False,
    }
    atomic_json(output_root / f"shard_complete_{args.start_job}_{args.stop_job}.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "configs/sparc_untouched2k_five_backend_matrix.frozen.json")
    parser.add_argument("--lock", type=Path,
                        default=ROOT / "configs/sparc_untouched2k_five_backend_matrix.lock.json")
    parser.add_argument("--python", default="python")
    parser.add_argument("--gpu", required=True,
                        help="physical GPU index exported through CUDA_VISIBLE_DEVICES")
    parser.add_argument("--start-job", type=int, default=70)
    parser.add_argument("--stop-job", type=int, default=150)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results/sparc_untouched2k_five_backend_runs")
    args = parser.parse_args()
    try:
        report = run(args)
    except (ValueError, RuntimeError) as error:
        print(json.dumps({"status": "shard_failed", "error": str(error)}, indent=2))
        raise SystemExit(2)
    print(json.dumps({key: report[key] for key in (
        "status", "start_job_inclusive", "stop_job_exclusive", "jobs", "pairs",
        "gpu_argument", "output_root",
    )}, indent=2))


if __name__ == "__main__":
    main()
