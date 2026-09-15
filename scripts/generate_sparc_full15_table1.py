#!/usr/bin/env python3
"""Generate the SPARC Full-15 Table 1 from one audited summary JSON.

The script is deliberately a presentation-only consumer: it validates the
complete five-backend by fifteen-corruption evidence matrix, recomputes all
averages, and never accepts hand-entered table values.  It writes a full LaTeX
table, reusable LaTeX rows, a machine-readable CSV, and a provenance manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


SCENARIOS = (
    "gaussian_noise:5", "shot_noise:5", "impulse_noise:5",
    "defocus_blur:5", "glass_blur:5", "motion_blur:5", "zoom_blur:5",
    "snow:5", "frost:5", "fog:5", "brightness:5",
    "contrast:5", "elastic_transform:5", "pixelate:5", "jpeg_compression:5",
)
BACKENDS = ("clip", "tent", "clipartt", "tda", "mint")
BACKEND_LABELS = {
    "clip": "CLIP", "tent": "TENT", "clipartt": "CLIPArTT",
    "tda": "TDA", "mint": "Mint",
}
SCENARIO_LABELS = {
    "gaussian_noise:5": "Gau.", "shot_noise:5": "Shot",
    "impulse_noise:5": "Imp.", "defocus_blur:5": "Def.",
    "glass_blur:5": "Glass", "motion_blur:5": "Mot.",
    "zoom_blur:5": "Zoom", "snow:5": "Snow", "frost:5": "Frost",
    "fog:5": "Fog", "brightness:5": "Bright", "contrast:5": "Contr.",
    "elastic_transform:5": "Elastic", "pixelate:5": "Pixel",
    "jpeg_compression:5": "JPEG",
}
FAMILIES = {
    "Noise": SCENARIOS[:3], "Blur": SCENARIOS[3:7],
    "Weather": SCENARIOS[7:11], "Digital": SCENARIOS[11:],
}
PAIR_FIELDS = (
    "image_id", "sample", "target", "raw_sha256", "descriptor_sha256",
    "stream_position", "batch",
)


class TableInputError(RuntimeError):
    """Raised when a summary cannot support the final Full-15 table."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TableInputError(f"cannot read summary JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise TableInputError("summary JSON root must be an object")
    return value


def finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TableInputError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TableInputError(f"{where} must be finite")
    return result


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)


def validate_summary(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    if payload.get("schema_version") != 1:
        raise TableInputError("summary schema_version must be 1")
    if payload.get("status") != "strictly_paired_frozen_evaluation_summary":
        raise TableInputError("input is not a strictly paired frozen evaluation summary")
    if payload.get("development_test_overlap") != 0:
        raise TableInputError("development/test overlap must be exactly zero")
    if tuple(payload.get("pairing_fields", ())) != PAIR_FIELDS:
        raise TableInputError("summary pairing_fields do not match the frozen contract")

    conditions = payload.get("conditions")
    if not isinstance(conditions, list):
        raise TableInputError("summary.conditions must be a list")
    expected = {(backend, scenario) for backend in BACKENDS for scenario in SCENARIOS}
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(conditions):
        if not isinstance(row, dict):
            raise TableInputError(f"conditions[{index}] must be an object")
        if row.get("split") != "test":
            raise TableInputError(f"conditions[{index}].split must be 'test'")
        key = (str(row.get("backend")), str(row.get("scope")))
        if key in observed:
            raise TableInputError(f"duplicate backend/corruption cell: {key}")
        if key not in expected:
            raise TableInputError(f"unexpected backend/corruption cell: {key}")
        parent = finite_number(row.get("parent_accuracy"), f"conditions[{index}].parent_accuracy")
        plugin = finite_number(row.get("plugin_accuracy"), f"conditions[{index}].plugin_accuracy")
        gain = finite_number(row.get("gain_pp"), f"conditions[{index}].gain_pp")
        if not 0.0 <= parent <= 100.0 or not 0.0 <= plugin <= 100.0:
            raise TableInputError(f"accuracy outside [0,100] at {key}")
        if not close(gain, plugin - parent):
            raise TableInputError(f"gain is not reconstructible at {key}")
        ci = row.get("ci95_pp")
        if not isinstance(ci, list) or len(ci) != 2:
            raise TableInputError(f"ci95_pp must contain two values at {key}")
        low = finite_number(ci[0], f"{key}.ci95_pp[0]")
        high = finite_number(ci[1], f"{key}.ci95_pp[1]")
        if low > high:
            raise TableInputError(f"reversed confidence interval at {key}")
        if not isinstance(row.get("expert_counts"), dict):
            raise TableInputError(f"expert_counts must be an object at {key}")
        observed[key] = row
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if missing or extra or len(conditions) != len(expected):
        raise TableInputError(
            f"summary must contain exactly 5x15=75 unique cells; "
            f"missing={missing}, extra={extra}, rows={len(conditions)}"
        )

    validate_aggregates(payload.get("aggregates"), observed)
    validate_acceptance(payload.get("acceptance"), observed)
    return observed


def validate_aggregates(
    aggregates: Any, cells: dict[tuple[str, str], dict[str, Any]]
) -> None:
    if not isinstance(aggregates, list):
        raise TableInputError("summary.aggregates must be a list")
    expected_scopes = (*FAMILIES.keys(), "Avg")
    expected = {(backend, scope) for backend in BACKENDS for scope in expected_scopes}
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(aggregates):
        if not isinstance(row, dict) or row.get("split") != "test":
            raise TableInputError(f"aggregates[{index}] must be a test-split object")
        key = (str(row.get("backend")), str(row.get("scope")))
        if key in observed:
            raise TableInputError(f"duplicate aggregate cell: {key}")
        if key not in expected:
            raise TableInputError(f"unexpected aggregate cell: {key}")
        scenarios = SCENARIOS if key[1] == "Avg" else FAMILIES[key[1]]
        parent = sum(float(cells[(key[0], scenario)]["parent_accuracy"]) for scenario in scenarios) / len(scenarios)
        plugin = sum(float(cells[(key[0], scenario)]["plugin_accuracy"]) for scenario in scenarios) / len(scenarios)
        for field, reconstructed in (
            ("parent_accuracy", parent), ("plugin_accuracy", plugin),
            ("gain_pp", plugin - parent),
        ):
            recorded = finite_number(row.get(field), f"aggregates[{index}].{field}")
            if not close(recorded, reconstructed):
                raise TableInputError(f"aggregate {field} is not reconstructible at {key}")
        observed[key] = row
    if set(observed) != expected or len(aggregates) != len(expected):
        missing = sorted(expected - set(observed))
        raise TableInputError(
            f"summary must contain exactly 5x5=25 unique aggregates; missing={missing}"
        )


def validate_acceptance(
    acceptance: Any, cells: dict[tuple[str, str], dict[str, Any]]
) -> None:
    if not isinstance(acceptance, dict):
        raise TableInputError("summary.acceptance must be an object")
    gains = [float(row["gain_pp"]) for row in cells.values()]
    checks = {
        "core_cells": 75,
        "positive_core_cells": sum(value > 0.0 for value in gains),
        "preferred_gt_1_5pp_cells": sum(value > 1.5 for value in gains),
        "all_core_cells_positive": all(value > 0.0 for value in gains),
    }
    for key, expected in checks.items():
        if acceptance.get(key) != expected:
            raise TableInputError(f"acceptance.{key} is inconsistent with conditions")
    minimum = finite_number(acceptance.get("minimum_observed_gain_pp"),
                            "acceptance.minimum_observed_gain_pp")
    if not close(minimum, min(gains)):
        raise TableInputError("acceptance.minimum_observed_gain_pp is inconsistent")


def values_for(
    cells: dict[tuple[str, str], dict[str, Any]], backend: str, field: str
) -> list[float]:
    values = [float(cells[(backend, scenario)][field]) for scenario in SCENARIOS]
    values.append(sum(values) / len(values))
    return values


def tex_escape(value: str) -> str:
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def formatted_pair(
    cells: dict[tuple[str, str], dict[str, Any]], backend: str
) -> tuple[str, str]:
    parent_values = values_for(cells, backend, "parent_accuracy")
    plugin_values = values_for(cells, backend, "plugin_accuracy")
    parent_cells: list[str] = []
    plugin_cells: list[str] = []
    for parent, plugin in zip(parent_values, plugin_values, strict=True):
        # Exact ties intentionally emphasize only the lower (SPARC) row.
        parent_cells.append(f"{parent:.2f}" if plugin >= parent else rf"\textbf{{{parent:.2f}}}")
        plugin_cells.append(rf"\textbf{{{plugin:.2f}}}" if plugin >= parent else f"{plugin:.2f}")
    label = BACKEND_LABELS[backend]
    parent_row = tex_escape(label) + " & " + " & ".join(parent_cells) + r" \\"
    plugin_row = tex_escape(f"SPARC + {label}") + " & " + " & ".join(plugin_cells) + r" \\"
    return parent_row, plugin_row


def render_rows(cells: dict[tuple[str, str], dict[str, Any]]) -> str:
    lines = ["% Auto-generated from an audited Full-15 summary; do not edit numbers."]
    for index, backend in enumerate(BACKENDS):
        lines.extend(formatted_pair(cells, backend))
        if index != len(BACKENDS) - 1:
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


def render_table(cells: dict[tuple[str, str], dict[str, Any]]) -> str:
    labels = [SCENARIO_LABELS[scenario] for scenario in SCENARIOS]
    column_header = "Method & " + " & ".join(tex_escape(value) for value in labels) + r" & Avg \\"
    return "\n".join([
        "% Auto-generated by scripts/generate_sparc_full15_table1.py.",
        "% Requires: \\usepackage{booktabs,graphicx}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Top-1 accuracy (\%) on all 15 ImageNet-C severity-5 corruptions. "
        r"Bold marks the better result within each matched parent/SPARC pair; "
        r"for exact ties, only the SPARC row is bold.}",
        r"\label{tab:full15}",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\resizebox{\textwidth}{!}{%",
        # Compact spacing within a corruption family and a visibly larger
        # gap between families (and before the single final Avg column).
        r"\begin{tabular}{l*{3}{c}@{\hspace{7pt}}*{4}{c}@{\hspace{7pt}}*{4}{c}@{\hspace{7pt}}*{4}{c}@{\hspace{7pt}}c}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Noise} & \multicolumn{4}{c}{Blur} & "
        r"\multicolumn{4}{c}{Weather} & \multicolumn{4}{c}{Digital} & \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}\cmidrule(lr){9-12}\cmidrule(lr){13-16}",
        column_header,
        r"\midrule",
        render_rows(cells).rstrip(),
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
        "",
    ])


def render_csv(cells: dict[tuple[str, str], dict[str, Any]]) -> str:
    from io import StringIO

    target = StringIO(newline="")
    fieldnames = ("backend", "role", "method", *SCENARIOS, "Avg")
    writer = csv.DictWriter(target, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for backend in BACKENDS:
        for role, method, field in (
            ("parent", BACKEND_LABELS[backend], "parent_accuracy"),
            ("plugin", f"SPARC + {BACKEND_LABELS[backend]}", "plugin_accuracy"),
        ):
            values = values_for(cells, backend, field)
            row = {"backend": backend, "role": role, "method": method}
            row.update({scenario: f"{value:.10f}" for scenario, value in zip(SCENARIOS, values[:-1], strict=True)})
            row["Avg"] = f"{values[-1]:.10f}"
            writer.writerow(row)
    return target.getvalue()


def run(summary_path: Path, output_dir: Path) -> dict[str, Any]:
    summary_path = summary_path.resolve()
    payload = load_object(summary_path)
    cells = validate_summary(payload)
    rows = render_rows(cells)
    table = render_table(cells)
    csv_text = render_csv(cells)
    manifest = {
        "schema_version": 1,
        "status": "full15_table1_generated_from_strict_audit",
        "source_summary": str(summary_path),
        "source_summary_sha256": sha256(summary_path),
        "backends": list(BACKENDS),
        "scenarios": list(SCENARIOS),
        "input_cells": 75,
        "output_rows": 10,
        "average_definition": "unweighted arithmetic mean of all 15 corruption accuracies",
        "emphasis_rule": "higher within each parent/SPARC pair; exact ties bold lower SPARC row only",
    }
    output_dir = output_dir.resolve()
    atomic_text(output_dir / "table1_full15_rows.tex", rows)
    atomic_text(output_dir / "table1_full15.tex", table)
    atomic_text(output_dir / "table1_full15.csv", csv_text)
    atomic_text(output_dir / "table1_full15_manifest.json",
                json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args.summary, args.output_dir)
    except TableInputError as error:
        print(json.dumps({"status": "blocked_invalid_summary", "error": str(error)}, indent=2))
        raise SystemExit(2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
