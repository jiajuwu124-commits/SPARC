#!/usr/bin/env python3
"""Generate the final SPARC Tables 2--3 and Figures 2--4 from audited records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


BACKENDS = ("clip", "tent", "clipartt", "tda", "mint")
LABELS = {"clip": "CLIP", "tent": "TENT", "clipartt": "CLIPArTT", "tda": "TDA", "mint": "Mint"}
SCENARIOS = (
    "gaussian_noise:5", "shot_noise:5", "impulse_noise:5",
    "defocus_blur:5", "glass_blur:5", "motion_blur:5", "zoom_blur:5",
    "snow:5", "frost:5", "fog:5", "brightness:5",
    "contrast:5", "elastic_transform:5", "pixelate:5", "jpeg_compression:5",
)
SCENARIO_LABELS = (
    "Gaussian", "Shot", "Impulse", "Defocus", "Glass", "Motion", "Zoom",
    "Snow", "Frost", "Fog", "Brightness", "Contrast", "Elastic", "Pixelate", "JPEG",
)
EXPERT_LABELS = (
    "Pass", "Gauss.", "Median", "Unsharp", "AutoContr.", "InvZoom",
    "Defocus-R", "Glass-R", "Motion-R", "Snow-R", "Frost-R", "Fog-R",
    "InvBright", "Bright-R",
)
BLUE = "#165A8A"
TEAL = "#22A6A1"
LIGHT_TEAL = "#7BC6CA"
GRAY = "#6B7785"
GRID = "#D8E1E8"


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


def csv_text(rows: list[dict]) -> str:
    target = StringIO(newline="")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue()


def fmt(value: float) -> str:
    return f"{value:.2f}"


def save_figure(
    fig: plt.Figure, directory: Path, stem: str, *, pad_inches: float | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    options = {"bbox_inches": "tight"}
    if pad_inches is not None:
        options["pad_inches"] = pad_inches
    fig.savefig(directory / f"{stem}.pdf", **options)
    fig.savefig(directory / f"{stem}.png", dpi=300, **options)
    fig.savefig(directory / f"{stem}.svg", **options)
    plt.close(fig)


def table2(output: Path, rows: list[dict]) -> None:
    directory = output / "table2_router_controls"
    atomic_text(directory / "table2_router_controls.csv", csv_text(rows))
    best_delta = max(row["delta_accuracy_pp"] for row in rows)
    best_fix = max(row["fix_pct"] for row in rows)
    best_break = min(row["break_pct"] for row in rows)
    body = []
    for row in rows:
        label = row["router"]
        if label == "SPARC + batch":
            label = rf"\textbf{{{label}}}"
        values = []
        for key, best in (("delta_accuracy_pp", best_delta), ("fix_pct", best_fix), ("break_pct", best_break)):
            value = fmt(row[key])
            if abs(row[key] - best) < 1e-12:
                value = rf"\textbf{{{value}}}"
            values.append(value)
        body.append(
            f"{label} & {row['model']} & {row['active_experts']} & "
            + " & ".join(values) + r" \\"
        )
    tex = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Development router controls with matched data, experts, and nested image-and-batch-disjoint folds. Act. is the number of experts selected at least once. $\Delta$Acc. is the realized per-image top-1 change; Fix and Break are the percentages of all observations changing from wrong to correct and correct to wrong.}",
        r"\label{tab:router-controls}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.0pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Router & Model & Act. & $\Delta$Acc. $\uparrow$ & Fix $\uparrow$ & Break $\downarrow$ \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ])
    atomic_text(directory / "table2_router_controls_fragment.tex", tex)


def display_table3_rows(rows: list[dict]) -> list[dict]:
    selected = [row for row in rows if int(row["experts"]) in (4, 7, 13)]
    if [int(row["experts"]) for row in selected] != [4, 7, 13]:
        raise RuntimeError("expected complete bank sequence to contain 4, 7, and 13 experts")
    return selected


def table3(output: Path, complete_rows: list[dict]) -> None:
    directory = output / "table3_ablations"
    rows = display_table3_rows(complete_rows)
    atomic_text(directory / "table3_ablations.csv", csv_text(rows))
    atomic_text(directory / "table3_nested_expert_bank_complete.csv", csv_text(complete_rows))
    best_delta = max(row["delta_accuracy_pp"] for row in rows)
    best_fix = max(row["fix_pct"] for row in rows)
    best_break = min(row["break_pct"] for row in rows)
    body = []
    for row in rows:
        label = row["expert_bank"]
        if label.startswith("+"):
            label = "$+$" + label[1:]
        if int(row["experts"]) == 13:
            label = rf"\textbf{{{label}}}"
        values = []
        for key, best in (("delta_accuracy_pp", best_delta), ("fix_pct", best_fix), ("break_pct", best_break)):
            value = fmt(row[key])
            if abs(row[key] - best) < 1e-12:
                value = rf"\textbf{{{value}}}"
            values.append(value)
        body.append(f"{label} & {row['experts']} & " + " & ".join(values) + r" \\")
    tex = "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Development expert-bank ablation with matched router architecture and nested image-and-batch-disjoint folds. Each row retrains the router and gate for that bank. $\Delta$Acc., Fix, and Break are realized per-image top-1 transition rates.}",
        r"\label{tab:ablations}", r"\scriptsize", r"\setlength{\tabcolsep}{3.0pt}",
        r"\renewcommand{\arraystretch}{1.15}", r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule", r"Expert bank & Exp. & $\Delta$Acc. $\uparrow$ & Fix $\uparrow$ & Break $\downarrow$ \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ])
    atomic_text(directory / "table3_ablations_fragment.tex", tex)


def gains_figure(output: Path, summary: dict) -> None:
    conditions = {(row["backend"], row["scope"]): row for row in summary["conditions"]}
    aggregates = {(row["backend"], row["scope"]): row for row in summary["aggregates"]}
    data = []
    for backend in BACKENDS:
        row = aggregates[(backend, "Avg")]
        wins = sum(conditions[(backend, scenario)]["gain_pp"] > 0 for scenario in SCENARIOS)
        data.append({
            "backend": LABELS[backend], "parent": row["parent_accuracy"],
            "sparc": row["plugin_accuracy"], "gain": row["gain_pp"],
            "ci_low": row["ci95_pp"][0], "ci_high": row["ci95_pp"][1], "wins": wins,
        })
    directory = output / "figure2_backend_gains"
    atomic_text(directory / "figure2_backend_gains.csv", csv_text(data))
    fig, ax = plt.subplots(figsize=(7.15, 3.55))
    y = np.arange(len(data))[::-1] * 1.25
    for position, row in zip(y, data):
        ax.plot([row["parent"], row["sparc"]], [position, position], color=LIGHT_TEAL,
                linewidth=6, solid_capstyle="round", zorder=1)
        ax.scatter(row["parent"], position, s=88, facecolor="white", edgecolor=GRAY,
                   linewidth=2.1, zorder=3)
        ax.scatter(row["sparc"], position, s=105, marker="D", color=BLUE,
                   edgecolor="white", linewidth=.8, zorder=4)
        ax.text(row["parent"], position + .22, f"{row['parent']:.2f}", ha="center",
                fontsize=8.7, color=GRAY)
        ax.text(row["sparc"] + .16, position + .20, f"{row['sparc']:.2f}", ha="left",
                fontsize=9, color=BLUE, fontweight="bold")
        ax.text((row["parent"] + row["sparc"]) / 2, position - .31,
                f"+{row['gain']:.2f} pp  ·  95% CI [{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
                f"  ·  {row['wins']}/15 wins", ha="center", va="top", fontsize=7.7,
                color="#244B6B")
    ax.set_yticks(y, [row["backend"] for row in data], fontsize=10)
    ax.set_xlim(38, 55.2)
    ax.set_ylim(-.70, y.max() + .72)
    ax.set_xlabel("Mean top-1 accuracy across 15 corruptions (%)", fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8.5)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=GRAY, markeredgewidth=1.8, markersize=7, label="Without SPARC"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=BLUE,
               markeredgecolor="white", markersize=7, label="With SPARC"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(.70, 1.04),
              frameon=False, ncol=2, fontsize=8.5, handletextpad=.4, columnspacing=1.2)
    fig.tight_layout()
    save_figure(fig, directory, "figure2_backend_gains", pad_inches=0.04)


def efficiency_figure(output: Path, rows: list[dict]) -> None:
    directory = output / "figure3_data_efficiency"
    atomic_text(directory / "figure3_data_efficiency.csv", csv_text(rows))
    x = np.asarray([row["available_base_images"] for row in rows])
    y = np.asarray([row["delta_accuracy_pp"] for row in rows])
    previous_style = plt.rcParams.copy()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.35, 2.25), constrained_layout=True)
    ax.plot(x, y, color="#117A8B", linewidth=2.1, marker="o", markersize=5.5,
            markerfacecolor="white", markeredgewidth=1.6)
    for xx, yy in zip(x, y):
        ax.annotate(f"{yy:.2f}", (xx, yy), xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color="#174A5B")
    ax.set_xlabel("Available development images")
    ax.set_ylabel(r"OOF $\Delta$Acc. (pp)")
    ax.set_xticks(x)
    pad = max(.5, .12 * (float(y.max()) - float(y.min()) + 1e-9))
    ax.set_ylim(float(y.min()) - pad, float(y.max()) + 2.4 * pad)
    ax.grid(axis="y", color="#D7E1E5", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, directory, "figure3_data_efficiency")
    plt.rcParams.update(previous_style)


def routing_figure(output: Path, summary: dict) -> None:
    conditions = {(row["backend"], row["scope"]): row for row in summary["conditions"]}
    names = (
        "identity", "gaussian15", "median3", "unsharp2", "autocontrast",
        "inverse_zoom10", "learned_defocus_r1", "learned_glass_r075",
        "learned_motion_r1", "learned_snow_r1", "learned_frost_r075",
        "learned_fog_r1", "inverse_brightness_c055", "deployment_brightness",
    )
    matrix = np.zeros((15, len(names)))
    full = []
    for i, scenario in enumerate(SCENARIOS):
        counts = conditions[("clip", scenario)]["expert_counts"]
        total = sum(int(value) for value in counts.values())
        for j, name in enumerate(names):
            matrix[i, j] = 100 * int(counts.get(name, 0)) / total
            full.append({"scenario": scenario, "expert": name, "selection_pct": matrix[i, j]})
    directory = output / "figure4_routing"
    atomic_text(directory / "figure4_routing_full.csv", csv_text(full))
    representatives = []
    for start, stop, family in ((0, 3, "Noise"), (3, 7, "Blur"), (7, 11, "Weather"), (11, 15, "Digital")):
        value, row_index, expert_index = max(
            (matrix[i, int(np.argmax(matrix[i]))], i, int(np.argmax(matrix[i])))
            for i in range(start, stop)
        )
        representatives.append({
            "family": family, "scenario": SCENARIO_LABELS[row_index],
            "expert": EXPERT_LABELS[expert_index], "selection_pct": float(value),
            "selection_rule": "highest dominant-expert rate within family",
        })
    atomic_text(directory / "figure4_routing.csv", csv_text(representatives))
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    positions = np.arange(len(representatives))[::-1]
    values = [row["selection_pct"] for row in representatives]
    labels = [f"{row['scenario']}  →  {row['expert']}" for row in representatives]
    bars = ax.barh(positions, values, color=[BLUE, "#287F9E", TEAL, "#5AB5A6"], height=.58)
    for bar, value in zip(bars, values):
        ax.text(value - 2.2, bar.get_y() + bar.get_height() / 2, f"{value:.0f}%",
                va="center", ha="right", fontsize=7.6, color="white", fontweight="bold")
    ax.set_yticks(positions, labels, fontsize=7.2)
    ax.set_xlim(0, 103)
    ax.set_xlabel("Dominant-expert selection rate", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", color=GRID, linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save_figure(fig, directory, "figure4_routing", pad_inches=0.04)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    suite = args.suite.resolve()
    summary_path = args.summary.resolve()
    output = args.output_dir.resolve()
    report = json.loads((suite / "report.json").read_text())
    audit = json.loads((suite / "independent_audit.json").read_text())
    summary = json.loads(summary_path.read_text())
    if report.get("status") != "completed_final_per_image_development_suite":
        raise RuntimeError("development suite is not final")
    if audit.get("status") != "passed_independent_final_per_image_suite_audit":
        raise RuntimeError("development suite has not passed independent audit")
    if summary.get("status") != "strictly_paired_frozen_evaluation_summary":
        raise RuntimeError("main summary is not frozen and paired")
    output.mkdir(parents=True, exist_ok=True)
    table2(output, report["table2_router_controls"])
    table3(output, report["table3_nested_expert_bank_complete"])
    gains_figure(output, summary)
    efficiency_figure(output, report["figure3_data_efficiency"])
    routing_figure(output, summary)
    manifest = {
        "schema_version": 1,
        "status": "generated_final_paper_artifacts_from_audited_evidence",
        "development_report_sha256": sha256(suite / "report.json"),
        "development_audit_sha256": sha256(suite / "independent_audit.json"),
        "main_summary_sha256": sha256(summary_path),
        "outputs": {},
    }
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["outputs"][str(path.relative_to(output))] = {
                "bytes": path.stat().st_size, "sha256": sha256(path),
            }
    atomic_text(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "files": len(manifest["outputs"])}, indent=2))


if __name__ == "__main__":
    main()
