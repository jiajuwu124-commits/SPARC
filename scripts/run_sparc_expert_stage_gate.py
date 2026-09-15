#!/usr/bin/env python3
"""Run an auditable development -> diagnostic A -> diagnostic B expert gate.

The runner is intentionally evaluator-agnostic.  It invokes one existing
evaluator per split, audits the evaluator's sample-level JSON outputs, freezes
one scale from development evidence, and reuses that scale for every declared
backend and diagnostic split.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("development", "diagnostic_a", "diagnostic_b")
RESERVED_FLAGS = {
    "--ids", "--cache", "--output-dir", "--status", "--backends",
    "--seed", "--subset-seed", "--batch-size",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_structured(path: Path) -> dict[str, Any]:
    text = path.read_text()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("YAML manifest requires PyYAML") from error
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("manifest root must be a mapping")
    return value


def resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def dotted_get(payload: dict[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted)
        value = value[part]
    return value


def scale_token(scale: float) -> str:
    return format(scale, ".12g")


def validate_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    manifest = load_structured(path)
    base = path.resolve().parent
    if manifest.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if manifest.get("status") != "expert_candidate_stage_gate":
        raise ValueError("status must be expert_candidate_stage_gate")
    seed = manifest.get("seed")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    batch_size = manifest.get("batch_size")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    threshold = float(manifest.get("minimum_gain_pp", 1.5))
    if threshold < 1.5:
        raise ValueError("minimum_gain_pp cannot be below 1.5")
    backends = manifest.get("backends")
    if not isinstance(backends, list) or not backends or len(set(backends)) != len(backends):
        raise ValueError("backends must be a non-empty unique list")
    if any(not isinstance(item, str) or not item for item in backends):
        raise ValueError("every backend must be a non-empty string")
    if not isinstance(manifest.get("model_identity"), str) or not manifest["model_identity"]:
        raise ValueError("model_identity is required")

    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty list")
    scales: set[float] = set()
    methods: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("each candidate must be a mapping")
        scale = float(candidate.get("scale"))
        method = candidate.get("method")
        if not math.isfinite(scale) or not isinstance(method, str) or not method:
            raise ValueError("candidate scale/method is invalid")
        if scale in scales or method in methods or method == manifest.get("identity_method", "identity"):
            raise ValueError("candidate scales and methods must be unique")
        scales.add(scale)
        methods.add(method)

    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(STAGES):
        raise ValueError(f"splits must contain exactly {STAGES}")
    for stage in STAGES:
        item = splits[stage]
        if not isinstance(item, dict) or "ids" not in item or "cache" not in item:
            raise ValueError(f"split {stage} requires ids and cache")
        ids = resolve(base, item["ids"])
        cache_protocol = resolve(base, item["cache"]) / "protocol.json"
        if not ids.is_file():
            raise FileNotFoundError(ids)
        if not cache_protocol.is_file():
            raise FileNotFoundError(cache_protocol)
        if "existing_output" in item:
            existing = resolve(base, item["existing_output"])
            if not (existing / "protocol.json").is_file():
                raise FileNotFoundError(existing / "protocol.json")

    evaluator = manifest.get("evaluator")
    if not isinstance(evaluator, dict) or "script" not in evaluator:
        raise ValueError("evaluator.script is required")
    script = resolve(base, evaluator["script"])
    if not script.is_file():
        raise FileNotFoundError(script)
    for key in ("common_args", "development_args", "diagnostic_args"):
        args = evaluator.get(key, [])
        if not isinstance(args, list) or any(not isinstance(value, str) for value in args):
            raise ValueError(f"evaluator.{key} must be a list of strings")
        if RESERVED_FLAGS.intersection(args):
            raise ValueError(f"evaluator.{key} contains a runner-controlled flag")
    diagnostic_text = " ".join(evaluator.get("diagnostic_args", []))
    if "{selected_scale}" not in diagnostic_text and "{frozen_manifest}" not in diagnostic_text:
        raise ValueError("diagnostic_args must consume selected_scale or frozen_manifest")

    stable = manifest.get("cross_stage_protocol_fields", [])
    if not isinstance(stable, list) or any(not isinstance(value, str) for value in stable):
        raise ValueError("cross_stage_protocol_fields must be a list of dotted keys")
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts must be a name-to-path mapping")
    for value in artifacts.values():
        if not resolve(base, value).is_file():
            raise FileNotFoundError(resolve(base, value))
    if manifest.get("selection_rule", "maximin_development_gain") != "maximin_development_gain":
        raise ValueError("only maximin_development_gain is supported")
    return manifest, base


def render_args(values: list[str], context: dict[str, str]) -> list[str]:
    rendered = []
    for value in values:
        try:
            rendered.append(value.format_map(context))
        except KeyError as error:
            raise ValueError(f"unknown evaluator template field: {error.args[0]}") from error
    return rendered


def build_command(
    manifest: dict[str, Any], base: Path, stage: str, output_dir: Path,
    *, selected_scale: float | None = None, frozen_manifest: Path | None = None,
) -> list[str]:
    evaluator = manifest["evaluator"]
    split = manifest["splits"][stage]
    candidates = manifest["candidates"]
    context = {
        "manifest_dir": str(base),
        "project_root": str(ROOT),
        "scales_csv": ",".join(scale_token(float(item["scale"])) for item in candidates),
        "selected_scale": (
            scale_token(selected_scale) if selected_scale is not None else "<selected-after-development>"
        ),
        "frozen_manifest": (
            str(frozen_manifest.resolve()) if frozen_manifest is not None else "<frozen-after-development>"
        ),
    }
    python = evaluator.get("python", sys.executable)
    script = resolve(base, evaluator["script"])
    command = [str(python), str(script)]
    command.extend(render_args(evaluator.get("common_args", []), context))
    command.extend([
        "--ids", str(resolve(base, split["ids"])),
        "--cache", str(resolve(base, split["cache"])),
        "--output-dir", str(output_dir.resolve()),
        "--status", "development" if stage == "development" else "diagnostic",
        "--backends", ",".join(manifest["backends"]),
        "--seed", str(manifest["seed"]),
        "--subset-seed", str(manifest["seed"]),
        "--batch-size", str(manifest["batch_size"]),
    ])
    if "n" in manifest:
        command.extend(["--n", str(int(manifest["n"]))])
    stage_args = (
        evaluator.get("development_args", [])
        if stage == "development"
        else evaluator.get("diagnostic_args", [])
    )
    command.extend(render_args(stage_args, context))
    return command


def input_hashes(manifest_path: Path, manifest: dict[str, Any], base: Path) -> dict[str, str]:
    hashes = {
        "manifest": sha256_file(manifest_path),
        "stage_gate_runner": sha256_file(Path(__file__)),
        "evaluator": sha256_file(resolve(base, manifest["evaluator"]["script"])),
    }
    for stage, split in manifest["splits"].items():
        hashes[f"{stage}.ids"] = sha256_file(resolve(base, split["ids"]))
        hashes[f"{stage}.cache_protocol"] = sha256_file(
            resolve(base, split["cache"]) / "protocol.json"
        )
    for name, value in manifest.get("artifacts", {}).items():
        hashes[f"artifact.{name}"] = sha256_file(resolve(base, value))
    hashes["common_args"] = sha256_json(manifest["evaluator"].get("common_args", []))
    hashes["model_identity"] = sha256_bytes(manifest["model_identity"].encode())
    return hashes


def record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    required = ("sample", "target", "scenario", "batch", "stream_position", "raw_sha256")
    missing = [key for key in required if key not in record]
    if missing:
        raise RuntimeError(f"record lacks pairing fields: {missing}")
    return tuple(record[key] for key in required)


def discover_results(output_dir: Path) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    protocol_path = output_dir / "protocol.json"
    if not protocol_path.is_file():
        raise RuntimeError(f"evaluator did not write {protocol_path}")
    protocol = json.loads(protocol_path.read_text())
    results = []
    for path in sorted(output_dir.glob("*.json")):
        if path == protocol_path:
            continue
        payload = json.loads(path.read_text())
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("protocol"), dict)
            and "accuracy" in payload
            and isinstance(payload.get("records"), list)
        ):
            results.append((path, payload))
    if not results:
        raise RuntimeError("no raw evaluator result JSON files were found")
    return protocol, results


def audit_stage(
    manifest: dict[str, Any], base: Path, stage: str, output_dir: Path,
    *, selected_scale: float | None = None,
) -> dict[str, Any]:
    shared, result_files = discover_results(output_dir)
    expected_status = "development" if stage == "development" else "diagnostic"
    if shared.get("status") != expected_status:
        raise RuntimeError(f"{stage}: evaluator status mismatch")
    if shared.get("seed") != manifest["seed"]:
        raise RuntimeError(f"{stage}: seed mismatch")
    if shared.get("batch_size") != manifest["batch_size"]:
        raise RuntimeError(f"{stage}: batch-size mismatch")
    if list(shared.get("backends", [])) != manifest["backends"]:
        raise RuntimeError(f"{stage}: backend list mismatch")
    split = manifest["splits"][stage]
    cache_protocol = resolve(base, split["cache"]) / "protocol.json"
    cache_payload = json.loads(cache_protocol.read_text())
    if shared.get("cache_protocol_sha256") != sha256_file(cache_protocol):
        raise RuntimeError(f"{stage}: cache protocol hash mismatch")
    if not isinstance(shared.get("ids_sha256"), str):
        raise RuntimeError(f"{stage}: evaluator protocol lacks ids_sha256")
    if shared["ids_sha256"] != cache_payload.get("ids_sha256"):
        raise RuntimeError(f"{stage}: evaluator/cache ID hash mismatch")
    if shared.get("n") != cache_payload.get("n"):
        raise RuntimeError(f"{stage}: evaluator/cache sample count mismatch")
    if "ids_source" in shared:
        if Path(shared["ids_source"]).resolve() != resolve(base, split["ids"]):
            raise RuntimeError(f"{stage}: evaluator used a different ID source")
    expected_protocol_sha = sha256_file(output_dir / "protocol.json")
    for dotted, expected in manifest.get("expected_protocol", {}).items():
        try:
            observed = dotted_get(shared, dotted)
        except KeyError as error:
            raise RuntimeError(f"{stage}: protocol lacks {dotted}") from error
        if observed != expected:
            raise RuntimeError(f"{stage}: protocol field {dotted} mismatch")

    by_backend_method: dict[tuple[str, str], dict[str, Any]] = {}
    files: dict[tuple[str, str], Path] = {}
    for path, payload in result_files:
        details = payload["protocol"]
        backend, method = details.get("backend"), details.get("method")
        if backend not in manifest["backends"]:
            continue
        if details.get("seed") != manifest["seed"]:
            raise RuntimeError(f"{stage}: result seed mismatch in {path.name}")
        if details.get("protocol_sha256") != expected_protocol_sha:
            raise RuntimeError(f"{stage}: result/shared protocol hash mismatch in {path.name}")
        key = (backend, method)
        if key in by_backend_method:
            raise RuntimeError(f"{stage}: duplicate result for {key}")
        records = payload["records"]
        if not records:
            raise RuntimeError(f"{stage}: empty records in {path.name}")
        keys = [record_key(record) for record in records]
        if len(set(keys)) != len(keys):
            raise RuntimeError(f"{stage}: duplicate pairing keys in {path.name}")
        accuracy = 100.0 * sum(bool(record.get("correct")) for record in records) / len(records)
        if not math.isclose(float(payload["accuracy"]), accuracy, abs_tol=1e-10):
            raise RuntimeError(f"{stage}: accuracy does not reconstruct in {path.name}")
        by_backend_method[key] = payload
        files[key] = path

    identity_method = manifest.get("identity_method", "identity")
    selected_method = None
    if selected_scale is not None:
        matches = [
            item["method"] for item in manifest["candidates"]
            if float(item["scale"]) == float(selected_scale)
        ]
        if len(matches) != 1:
            raise RuntimeError("selected scale does not identify exactly one candidate")
        selected_method = matches[0]
    expected_methods = (
        [identity_method] + [item["method"] for item in manifest["candidates"]]
        if stage == "development"
        else [identity_method, selected_method]
    )
    allowed = {(backend, method) for backend in manifest["backends"] for method in expected_methods}
    observed = set(by_backend_method)
    if observed != allowed:
        missing = sorted(allowed - observed)
        extra = sorted(observed - allowed)
        raise RuntimeError(f"{stage}: result matrix mismatch; missing={missing}, extra={extra}")

    reference_keys = None
    rows = []
    for backend in manifest["backends"]:
        identity = by_backend_method[(backend, identity_method)]
        identity_keys = [record_key(record) for record in identity["records"]]
        if reference_keys is None:
            reference_keys = identity_keys
        elif identity_keys != reference_keys:
            raise RuntimeError(f"{stage}: identity streams differ across backends")
        for method in expected_methods[1:]:
            candidate = by_backend_method[(backend, method)]
            candidate_keys = [record_key(record) for record in candidate["records"]]
            if candidate_keys != identity_keys:
                raise RuntimeError(f"{stage}: {backend}/{method} is not paired to identity")
            gain = float(candidate["accuracy"]) - float(identity["accuracy"])
            rows.append({
                "backend": backend,
                "method": method,
                "identity_accuracy": float(identity["accuracy"]),
                "candidate_accuracy": float(candidate["accuracy"]),
                "gain_pp": gain,
                "pairing_sha256": sha256_json(identity_keys),
                "identity_result": str(files[(backend, identity_method)].resolve()),
                "identity_result_sha256": sha256_file(files[(backend, identity_method)]),
                "candidate_result": str(files[(backend, method)].resolve()),
                "candidate_result_sha256": sha256_file(files[(backend, method)]),
            })
    return {
        "schema_version": 1,
        "stage": stage,
        "status": expected_status,
        "shared_protocol": str((output_dir / "protocol.json").resolve()),
        "shared_protocol_sha256": expected_protocol_sha,
        "seed": manifest["seed"],
        "batch_size": manifest["batch_size"],
        "model_identity": manifest["model_identity"],
        "cache_protocol_sha256": sha256_file(cache_protocol),
        "rows": rows,
    }


def choose_development_scale(manifest: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in audit["rows"]:
        by_method.setdefault(row["method"], []).append(row)
    candidates = []
    for item in manifest["candidates"]:
        rows = by_method.get(item["method"], [])
        if {row["backend"] for row in rows} != set(manifest["backends"]):
            raise RuntimeError(f"development candidate {item['method']} lacks a backend")
        gains = {row["backend"]: float(row["gain_pp"]) for row in rows}
        candidates.append({
            "scale": float(item["scale"]),
            "method": item["method"],
            "backend_gain_pp": gains,
            "minimum_backend_gain_pp": min(gains.values()),
            "mean_backend_gain_pp": sum(gains.values()) / len(gains),
        })
    # One global scale is selected.  Never select a different scale per backend.
    selected = sorted(
        candidates,
        key=lambda row: (
            -row["minimum_backend_gain_pp"],
            -row["mean_backend_gain_pp"],
            abs(row["scale"] - 1.0),
            row["scale"],
        ),
    )[0]
    threshold = float(manifest.get("minimum_gain_pp", 1.5))
    if not selected["minimum_backend_gain_pp"] > threshold:
        raise RuntimeError(
            "development gate failed: selected minimum backend gain "
            f"{selected['minimum_backend_gain_pp']:.12g} is not strictly greater than "
            f"{threshold:.12g} pp"
        )
    return {
        "schema_version": 1,
        "status": "frozen_after_development",
        "selection_rule": "maximize development minimum gain across all declared backends",
        "tie_breakers": ["larger mean gain", "scale closest to 1", "smaller scale"],
        "minimum_gain_pp_strictly_greater_than": threshold,
        "selected_scale": selected["scale"],
        "selected_residual_scale": selected["scale"],
        "selected_method": selected["method"],
        "selected_backend_gain_pp": selected["backend_gain_pp"],
        "selected_minimum_backend_gain_pp": selected["minimum_backend_gain_pp"],
        "selected_mean_backend_gain_pp": selected["mean_backend_gain_pp"],
        "all_candidates": candidates,
        "uses_development_only": True,
        "uses_diagnostic_results": False,
        "single_scale_for_all_backends": True,
    }


def enforce_diagnostic_gate(manifest: dict[str, Any], audit: dict[str, Any]) -> None:
    threshold = float(manifest.get("minimum_gain_pp", 1.5))
    failures = [
        (row["backend"], row["gain_pp"])
        for row in audit["rows"] if not float(row["gain_pp"]) > threshold
    ]
    if failures:
        raise RuntimeError(
            f"{audit['stage']} gate failed; every backend gain must be strictly greater "
            f"than {threshold:.12g} pp: {failures}"
        )


def verify_cross_stage_fields(
    manifest: dict[str, Any], completed_audits: list[dict[str, Any]]
) -> None:
    fields = manifest.get("cross_stage_protocol_fields", [])
    protocols = [json.loads(Path(audit["shared_protocol"]).read_text()) for audit in completed_audits]
    for field in fields:
        values = []
        for protocol in protocols:
            try:
                values.append(dotted_get(protocol, field))
            except KeyError as error:
                raise RuntimeError(f"cross-stage protocol field is missing: {field}") from error
        if any(value != values[0] for value in values[1:]):
            raise RuntimeError(f"cross-stage protocol field differs: {field}")


def next_attempt(stage_root: Path) -> tuple[int, Path]:
    indices = []
    if stage_root.exists():
        for path in stage_root.glob("attempt-*"):
            try:
                indices.append(int(path.name.split("-", 1)[1]))
            except (IndexError, ValueError):
                continue
    index = max(indices, default=0) + 1
    return index, stage_root / f"attempt-{index:03d}"


def append_command_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(payload) + "\n")


def verify_completed_stage(entry: dict[str, Any]) -> dict[str, Any]:
    audit_path = Path(entry["audit"])
    if not audit_path.is_file() or sha256_file(audit_path) != entry["audit_sha256"]:
        raise RuntimeError("resume audit file is missing or changed")
    audit = json.loads(audit_path.read_text())
    for row in audit["rows"]:
        for key in ("identity", "candidate"):
            path = Path(row[f"{key}_result"])
            if not path.is_file() or sha256_file(path) != row[f"{key}_result_sha256"]:
                raise RuntimeError("resume result file is missing or changed")
    return audit


def dry_run_plan(
    manifest_path: Path, manifest: dict[str, Any], base: Path, output_root: Path
) -> dict[str, Any]:
    commands = []
    placeholder_scale = None
    for index, stage in enumerate(STAGES, start=1):
        stage_dir = output_root / "stages" / stage / "attempt-001"
        command = build_command(
            manifest, base, stage, stage_dir,
            selected_scale=placeholder_scale,
            frozen_manifest=(output_root / "frozen_scale_manifest.json") if stage != "development" else None,
        )
        commands.append({
            "order": index,
            "stage": stage,
            "argv": command,
            "command_sha256": sha256_json(command),
            "note": (
                "actual diagnostic command is rendered after one development scale is frozen"
                if stage != "development" else "screens every predeclared scale"
            ),
        })
    return {
        "schema_version": 1,
        "mode": "dry_run_no_commands_executed",
        "manifest": str(manifest_path.resolve()),
        "input_hashes": input_hashes(manifest_path, manifest, base),
        "selection_rule": "maximize development minimum gain across declared backends",
        "minimum_gain_pp_strictly_greater_than": float(manifest.get("minimum_gain_pp", 1.5)),
        "commands": commands,
    }


def run_pipeline(
    manifest_path: Path, output_root: Path, *, resume: bool = False,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    manifest, base = validate_manifest(manifest_path)
    current_hashes = input_hashes(manifest_path, manifest, base)
    state_path = output_root / "stage_gate_state.json"
    command_ledger = output_root / "commands.jsonl"
    frozen_path = output_root / "frozen_scale_manifest.json"
    if output_root.exists() and not resume:
        raise FileExistsError(f"output exists; pass --resume: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if not resume:
            raise FileExistsError(f"state exists; pass --resume: {state_path}")
        if state.get("input_hashes") != current_hashes:
            raise RuntimeError("resume refused because manifest/evaluator/input hashes changed")
    else:
        state = {
            "schema_version": 1,
            "status": "running",
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": current_hashes["manifest"],
            "input_hashes": current_hashes,
            "selection_rule": "maximin_development_gain",
            "minimum_gain_pp_strictly_greater_than": float(
                manifest.get("minimum_gain_pp", 1.5)
            ),
            "stages": {},
        }
        write_json_atomic(state_path, state)

    completed_audits: list[dict[str, Any]] = []
    selected_scale: float | None = None
    for stage in STAGES:
        existing = state["stages"].get(stage)
        if existing and existing.get("status") == "complete":
            audit = verify_completed_stage(existing)
            if stage == "development":
                if not frozen_path.is_file() or sha256_file(frozen_path) != existing["frozen_sha256"]:
                    raise RuntimeError("resume frozen scale manifest is missing or changed")
                frozen = json.loads(frozen_path.read_text())
                selected_scale = float(frozen["selected_scale"])
            else:
                enforce_diagnostic_gate(manifest, audit)
            completed_audits.append(audit)
            verify_cross_stage_fields(manifest, completed_audits)
            continue

        attempt_number, output_dir = next_attempt(output_root / "stages" / stage)
        command = build_command(
            manifest, base, stage, output_dir,
            selected_scale=selected_scale,
            frozen_manifest=frozen_path if stage != "development" else None,
        )
        command_record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "attempt": attempt_number,
            "cwd": str(ROOT),
            "argv": command,
            "command_sha256": sha256_json(command),
            "manifest_sha256": current_hashes["manifest"],
            "evaluator_sha256": current_hashes["evaluator"],
            "model_identity_sha256": current_hashes["model_identity"],
        }
        append_command_record(command_ledger, {**command_record, "event": "start"})
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        log_path = output_dir.parent / f"attempt-{attempt_number:03d}.log"
        with log_path.open("wb") as log:
            completed = runner(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        append_command_record(command_ledger, {
            **command_record,
            "event": "finish",
            "returncode": int(completed.returncode),
            "log": str(log_path.resolve()),
            "log_sha256": sha256_file(log_path),
        })
        if completed.returncode != 0:
            state["stages"][stage] = {
                "status": "failed_evaluator", "attempt": attempt_number,
                "returncode": int(completed.returncode), "log": str(log_path.resolve()),
                "log_sha256": sha256_file(log_path),
            }
            write_json_atomic(state_path, state)
            raise RuntimeError(f"evaluator failed at {stage}; see {log_path}")

        audit = audit_stage(
            manifest, base, stage, output_dir, selected_scale=selected_scale
        )
        audit["command_sha256"] = command_record["command_sha256"]
        audit["evaluator_sha256"] = current_hashes["evaluator"]
        audit_path = output_dir / "stage_gate_audit.json"
        write_json_atomic(audit_path, audit)
        stage_state = {
            "status": "complete",
            "attempt": attempt_number,
            "output_dir": str(output_dir.resolve()),
            "audit": str(audit_path.resolve()),
            "audit_sha256": sha256_file(audit_path),
            "command_sha256": command_record["command_sha256"],
        }
        if stage == "development":
            frozen = choose_development_scale(manifest, audit)
            frozen.update({
                "manifest_sha256": current_hashes["manifest"],
                "development_audit": str(audit_path.resolve()),
                "development_audit_sha256": sha256_file(audit_path),
                "input_hashes": current_hashes,
            })
            if "artifact.specialist_checkpoint" in current_hashes:
                frozen["checkpoint_sha256"] = current_hashes["artifact.specialist_checkpoint"]
            write_json_atomic(frozen_path, frozen)
            selected_scale = float(frozen["selected_scale"])
            stage_state["frozen_scale_manifest"] = str(frozen_path.resolve())
            stage_state["frozen_sha256"] = sha256_file(frozen_path)
        else:
            enforce_diagnostic_gate(manifest, audit)
        state["stages"][stage] = stage_state
        completed_audits.append(audit)
        verify_cross_stage_fields(manifest, completed_audits)
        write_json_atomic(state_path, state)

    state["status"] = "complete"
    state["completed_stage_order"] = list(STAGES)
    state["frozen_scale_manifest"] = str(frozen_path.resolve())
    state["frozen_scale_manifest_sha256"] = sha256_file(frozen_path)
    state["commands_ledger"] = str(command_ledger.resolve())
    state["commands_ledger_sha256"] = sha256_file(command_ledger)
    write_json_atomic(state_path, state)
    return state


def import_existing_pipeline(
    manifest_path: Path, output_root: Path, *, resume: bool = False,
) -> dict[str, Any]:
    """Audit pre-existing evaluator directories without invoking the evaluator."""
    manifest, base = validate_manifest(manifest_path)
    missing = [
        stage for stage in STAGES
        if "existing_output" not in manifest["splits"][stage]
    ]
    if missing:
        raise ValueError(f"--import-existing requires existing_output for: {missing}")
    current_hashes = input_hashes(manifest_path, manifest, base)
    state_path = output_root / "stage_gate_state.json"
    ledger_path = output_root / "commands.jsonl"
    frozen_path = output_root / "frozen_scale_manifest.json"
    if output_root.exists() and not resume:
        raise FileExistsError(f"output exists; pass --resume: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if not resume:
            raise FileExistsError(f"state exists; pass --resume: {state_path}")
        if state.get("input_hashes") != current_hashes:
            raise RuntimeError("resume refused because manifest/evaluator/input hashes changed")
        if state.get("status") == "complete":
            for stage in STAGES:
                verify_completed_stage(state["stages"][stage])
            if not frozen_path.is_file() or sha256_file(frozen_path) != state["frozen_scale_manifest_sha256"]:
                raise RuntimeError("resume frozen scale manifest is missing or changed")
            return state
    else:
        state = {
            "schema_version": 1,
            "status": "importing_existing_read_only",
            "evidence_mode": "imported_existing_evaluator_outputs",
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": current_hashes["manifest"],
            "input_hashes": current_hashes,
            "selection_rule": "maximin_development_gain",
            "minimum_gain_pp_strictly_greater_than": float(
                manifest.get("minimum_gain_pp", 1.5)
            ),
            "stages": {},
        }
        write_json_atomic(state_path, state)

    completed_audits: list[dict[str, Any]] = []
    selected_scale: float | None = None
    for stage in STAGES:
        source = resolve(base, manifest["splits"][stage]["existing_output"])
        audit = audit_stage(
            manifest, base, stage, source, selected_scale=selected_scale
        )
        audit.update({
            "evidence_mode": "imported_existing_read_only",
            "source_output_dir": str(source),
            "source_output_protocol_sha256": sha256_file(source / "protocol.json"),
            "evaluator_was_invoked_by_stage_gate": False,
        })
        destination = output_root / "stages" / stage / "imported_stage_gate_audit.json"
        write_json_atomic(destination, audit)
        entry = {
            "status": "complete",
            "evidence_mode": "imported_existing_read_only",
            "source_output_dir": str(source),
            "output_dir": str(source),
            "audit": str(destination.resolve()),
            "audit_sha256": sha256_file(destination),
            "command_sha256": None,
        }
        append_command_record(ledger_path, {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": "import_existing_no_evaluator_command",
            "stage": stage,
            "argv": None,
            "command_sha256": None,
            "source_output_dir": str(source),
            "source_protocol_sha256": sha256_file(source / "protocol.json"),
            "manifest_sha256": current_hashes["manifest"],
            "evaluator_sha256": current_hashes["evaluator"],
        })
        if stage == "development":
            frozen = choose_development_scale(manifest, audit)
            frozen.update({
                "evidence_mode": "imported_existing_read_only",
                "evaluator_was_invoked_by_stage_gate": False,
                "manifest_sha256": current_hashes["manifest"],
                "development_audit": str(destination.resolve()),
                "development_audit_sha256": sha256_file(destination),
                "input_hashes": current_hashes,
            })
            if "artifact.specialist_checkpoint" in current_hashes:
                frozen["checkpoint_sha256"] = current_hashes["artifact.specialist_checkpoint"]
            write_json_atomic(frozen_path, frozen)
            selected_scale = float(frozen["selected_scale"])
            entry["frozen_scale_manifest"] = str(frozen_path.resolve())
            entry["frozen_sha256"] = sha256_file(frozen_path)
        else:
            enforce_diagnostic_gate(manifest, audit)
        state["stages"][stage] = entry
        completed_audits.append(audit)
        verify_cross_stage_fields(manifest, completed_audits)
        write_json_atomic(state_path, state)

    state["status"] = "complete"
    state["completed_stage_order"] = list(STAGES)
    state["frozen_scale_manifest"] = str(frozen_path.resolve())
    state["frozen_scale_manifest_sha256"] = sha256_file(frozen_path)
    state["commands_ledger"] = str(ledger_path.resolve())
    state["commands_ledger_sha256"] = sha256_file(ledger_path)
    write_json_atomic(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--import-existing", action="store_true",
        help="audit split.existing_output directories without invoking evaluator",
    )
    args = parser.parse_args()
    if args.dry_run and args.import_existing:
        parser.error("--dry-run and --import-existing are mutually exclusive")
    manifest, base = validate_manifest(args.manifest)
    if args.dry_run:
        print(json.dumps(dry_run_plan(args.manifest, manifest, base, args.output_dir), indent=2))
        return
    state = (
        import_existing_pipeline(args.manifest, args.output_dir, resume=args.resume)
        if args.import_existing
        else run_pipeline(args.manifest, args.output_dir, resume=args.resume)
    )
    print(json.dumps({
        "status": state["status"],
        "state": str((args.output_dir / "stage_gate_state.json").resolve()),
        "frozen_scale_manifest": state["frozen_scale_manifest"],
    }, indent=2))


if __name__ == "__main__":
    main()
