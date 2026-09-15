#!/usr/bin/env python3
"""Generate audited SPARC follow-up tables and figures from machine records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from io import StringIO
from pathlib import Path
import statistics
import subprocess

import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pypdfium2 as pdfium


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "results/sparc_development_experiment_suite_v1"
DEFAULT_SUMMARY = ROOT / "results/sparc_untouched2k_five_backend_summary_v1/summary.json"
DEFAULT_OUTPUT = ROOT / "paper_artifacts/sparc_followups_v1"
RUNS = ROOT / "results/sparc_untouched2k_five_backend_runs"
BACKENDS = ("clip", "tent", "clipartt", "tda", "mint")
LABELS = {"clip": "CLIP", "tent": "TENT", "clipartt": "CLIPArTT",
          "tda": "TDA", "mint": "Mint"}
SCENARIOS = (
    "gaussian_noise:5", "shot_noise:5", "impulse_noise:5",
    "defocus_blur:5", "glass_blur:5", "motion_blur:5", "zoom_blur:5",
    "snow:5", "frost:5", "fog:5", "brightness:5",
    "contrast:5", "elastic_transform:5", "pixelate:5", "jpeg_compression:5",
)
SCENARIO_LABELS = ("Gaussian", "Shot", "Impulse", "Defocus", "Glass", "Motion",
                   "Zoom", "Snow", "Frost", "Fog", "Brightness", "Contrast",
                   "Elastic", "Pixelate", "JPEG")
EXPERT_LABELS = ("Pass", "Gauss.", "Median", "Unsharp", "AutoContr.", "InvZoom",
                 "Defocus-R", "Glass-R", "Motion-R", "Snow-R", "Frost-R",
                 "Fog-R", "InvBright", "Bright-R")
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
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue()


def escape(value: object) -> str:
    return (str(value).replace("&", r"\&").replace("_", r"\_")
            .replace("%", r"\%").replace("#", r"\#"))


def compile_table(output: Path, stem: str, body: str, caption: str,
                  columns: str, header: str, paper_width: str = "10.5in",
                  table_number: int = 1, label: str | None = None,
                  float_env: str = "table*", font_command: str = r"\small",
                  tabcolsep: float = 5.0) -> None:
    directory = output / stem
    directory.mkdir(parents=True, exist_ok=True)
    label = label or f"tab:{stem}"
    fragment = rf"""\begin{{{float_env}}}[t]
\centering
\caption{{{caption}}}
\label{{{label}}}
{font_command}
\setlength{{\tabcolsep}}{{{tabcolsep:.1f}pt}}
\renewcommand{{\arraystretch}}{{1.15}}
\begin{{tabular}}{{{columns}}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{{float_env}}}
"""
    atomic_text(directory / f"{stem}_fragment.tex", fragment)
    doc = rf"""\documentclass[10pt]{{article}}
\usepackage[paperwidth={paper_width},paperheight=4.3in,margin=.35in]{{geometry}}
\usepackage{{booktabs,newtxtext}}
\pagestyle{{empty}}
\begin{{document}}
\renewcommand{{\thetable}}{{{table_number}}}
{fragment}
\end{{document}}
"""
    atomic_text(directory / f"{stem}.tex", doc)
    subprocess.run([str(Path.home() / ".local/bin/tectonic"), "-X", "compile",
                    f"{stem}.tex", "--keep-logs"], cwd=directory, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pdf = directory / f"{stem}.pdf"
    document = pdfium.PdfDocument(pdf)
    if len(document) != 1:
        raise RuntimeError(f"{stem} must compile to one page")
    preview = document[0].render(scale=2.4).to_pil()
    preview.save(directory / f"{stem}.png", optimize=True)


def table2(output: Path, suite: dict) -> None:
    source_rows = suite["same_information_controls"]
    display = {
        "Best single fixed expert": ("Best single fixed expert", "--"),
        "Ridge direct utility": ("Ridge direct utility", "Linear"),
        "Ridge direct utility + batch": ("Ridge direct utility + batch", "Linear"),
        "HGB direct utility": ("HGB direct utility", "Nonlinear"),
        "Corruption classifier then map": ("Corruption classifier then map", "Nonlinear"),
        "SPARC direct utility + batch": ("SPARC + batch", "Nonlinear"),
    }
    order = tuple(display)
    by_name = {row["method"]: row for row in source_rows}
    rows = [by_name[name] for name in order]
    atomic_text(output / "table2_router_controls/table2_router_controls.csv", csv_text(rows))
    feasible = [row for row in rows if row["harm_pct"] <= 5.0]
    best_safe_utility = max(row["mean_utility_pp"] for row in feasible)
    best_safe_harm = min(row["harm_pct"] for row in feasible)
    best_safe_benefit = max(row["benefit_pct"] for row in feasible)
    body = []
    for row in rows:
        ours = row["method"] == "SPARC direct utility + batch"
        utility = f"{row['mean_utility_pp']:.2f}"
        harm = f"{row['harm_pct']:.2f}"
        benefit = f"{row['benefit_pct']:.2f}"
        active = "--" if row.get("active_experts") is None else str(row["active_experts"])
        if row["harm_pct"] <= 5.0 and row["mean_utility_pp"] == best_safe_utility:
            utility = rf"\textbf{{{utility}}}"
        if row["harm_pct"] <= 5.0 and row["harm_pct"] == best_safe_harm:
            harm = rf"\textbf{{{harm}}}"
        if row["harm_pct"] <= 5.0 and row["benefit_pct"] == best_safe_benefit:
            benefit = rf"\textbf{{{benefit}}}"
        method, model = display[row["method"]]
        if ours:
            method = rf"\textbf{{{escape(method)}}}"
        else:
            method = escape(method)
        method_cell = rf"\parbox[t]{{1.08in}}{{\raggedright {method}}}"
        body.append(
            f"{method_cell} & {model} & {active} & {utility} & {harm} & {benefit} \\\\" 
        )
    compile_table(
        output, "table2_router_controls", "\n".join(body),
        r"Image-and-batch-disjoint development router controls with matched data and experts. "
        r"Bold marks the best value among methods satisfying Harm $\leq5\%$.",
        r"@{}llrrrr@{}",
        r"Router & Model & Act. & Util. $\uparrow$ & Harm $\downarrow$ & Ben. $\uparrow$ \\",
        paper_width="4.25in", table_number=2, label="tab:router-controls",
        float_env="table", font_command=r"\scriptsize", tabcolsep=1.2,
    )


def table3(output: Path, suite: dict) -> None:
    rows = []
    body = []
    bank = suite["expert_bank_ablation"]
    best_utility = max(row["mean_utility_pp"] for row in bank)
    best_harm = min(row["harm_pct"] for row in bank)
    best_benefit = max(row["benefit_pct"] for row in bank)
    for index, row in enumerate(bank):
        ours = index == len(bank) - 1
        utility = f"{row['mean_utility_pp']:.2f}"
        harm = f"{row['harm_pct']:.2f}"
        benefit = f"{row['benefit_pct']:.2f}"
        if row["mean_utility_pp"] == best_utility:
            utility = rf"\textbf{{{utility}}}"
        if row["harm_pct"] == best_harm:
            harm = rf"\textbf{{{harm}}}"
        if row["benefit_pct"] == best_benefit:
            benefit = rf"\textbf{{{benefit}}}"
        label = "Without brightness" if row["ablation"] == "Full bank without learned brightness" else row["ablation"]
        if ours:
            label = rf"\textbf{{{escape(label)}}}"
        else:
            label = escape(label)
        body.append(f"{label} & {row['experts']} & {utility} & {harm} & "
                    f"{benefit} " + r"\\")
        rows.append({"section": "expert_bank", **row})
    atomic_text(output / "table3_ablations/table3_ablations.csv", csv_text(rows))
    compile_table(
        output, "table3_ablations", "\n".join(body),
        r"Image-and-batch-disjoint development expert-bank ablation with router and folds fixed. "
        r"Utility is mean batch-level top-1 change (pp).",
        "@{}lrrrr@{}",
        r"Expert bank & Exp. & Util. $\uparrow$ & Harm $\downarrow$ & Ben. $\uparrow$ \\",
        "4.25in", table_number=3, label="tab:ablations",
        float_env="table", font_command=r"\scriptsize", tabcolsep=3.0,
    )


def cost_rows(summary: dict) -> tuple[list[dict], list[dict]]:
    result_paths = sorted(RUNS.glob("*/result.json"))
    if len(result_paths) != 150:
        raise RuntimeError(f"expected 150 final result files, found {len(result_paths)}")
    timing_rows = []
    per_backend = {backend: {"parent": [], "plugin": []} for backend in BACKENDS}
    for path in result_paths:
        payload = json.loads(path.read_text())
        backend = payload["protocol"]["backend"]
        role = "plugin" if path.parent.name.endswith("_plugin") else "parent"
        timing = payload["timing"]
        row = {"backend": backend, "role": role, "run": path.parent.name,
               "seconds": float(timing["seconds"]),
               "peak_gpu_gib": float(timing["peak_gpu_gib"]),
               "peak_fraction": float(timing["observed_peak_fraction"])}
        timing_rows.append(row)
        per_backend[backend][role].append(row)

    config = json.loads((ROOT / "configs/sparc_untouched2k_five_backend/"
                         "clip_gaussian_noise_s5_plugin.json").read_text())
    checkpoint_paths = []
    for expert in config["expert_bank"]:
        checkpoint = expert.get("checkpoint", {}).get("path")
        if checkpoint:
            checkpoint_paths.append(Path(checkpoint))
    learned_bytes = sum(path.stat().st_size for path in checkpoint_paths)
    router_path = Path(config["router"]["artifact"]["path"])
    router_bytes = router_path.stat().st_size
    aggregates = {(row["backend"], row["scope"]): row for row in summary["aggregates"]}
    counts = aggregates[("clip", "Avg")]["expert_counts"]
    total = sum(int(value) for value in counts.values())
    identity = int(counts.get("identity", 0))
    active = sum(name != "identity" and int(value) > 0 for name, value in counts.items())
    max_peak = max(row["peak_gpu_gib"] for row in timing_rows if row["role"] == "plugin")
    rows = [
        {"component": "Masked-risk probes", "count": "4 fixed",
         "storage_mib": 0.0, "runtime": "Descriptor only; no backend features"},
        {"component": "Utility router", "count": "13 HGB regressors",
         "storage_mib": router_bytes / 2**20, "runtime": "Scores all 13 experts on 55-D input"},
        {"component": "Restoration bank", "count": "4 fixed + 2 analytic + 7 learned",
         "storage_mib": learned_bytes / 2**20, "runtime": "Executes zero or one expert"},
        {"component": "Observed routing", "count": f"{active}/13 active",
         "storage_mib": math.nan,
         "runtime": f"{100*(total-identity)/total:.1f}% expert; {100*identity/total:.1f}% pass-through"},
        {"component": "Downstream backend", "count": "Unchanged",
         "storage_mib": 0.0, "runtime": "Same backend interface as parent"},
        {"component": "Peak GPU memory", "count": "75 plugin runs",
         "storage_mib": math.nan, "runtime": f"{max_peak:.2f} GiB maximum observed"},
    ]
    return rows, timing_rows


def table4(output: Path, summary: dict) -> None:
    rows, timing_rows = cost_rows(summary)
    atomic_text(output / "table4_cost/table4_cost.csv", csv_text(rows))
    atomic_text(output / "table4_cost/observed_run_timing.csv", csv_text(timing_rows))
    body = []
    for row in rows:
        storage = "--" if math.isnan(row["storage_mib"]) else f"{row['storage_mib']:.2f}"
        body.append(f"{escape(row['component'])} & {escape(row['count'])} & {storage} & "
                    f"{escape(row['runtime'])} " + r"\\")
    compile_table(
        output, "table4_cost", "\n".join(body),
        r"Front-end structure and measured deployment behavior. Storage excludes the unchanged "
        r"CLIP backbone. GPU memory is the maximum allocator observation across all 75 final "
        r"SPARC runs; shared-run wall-clock values are archived but not ranked.",
        "@{}llrl@{}", r"Item & Count & Storage (MiB) & Runtime behavior \\", "9.2in",
        table_number=4, label="tab:cost",
    )


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(directory / f"{stem}.{suffix}", dpi=260 if suffix == "png" else None,
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def gains_figure(output: Path, summary: dict) -> None:
    aggregates = {(row["backend"], row["scope"]): row for row in summary["aggregates"]}
    conditions = {(row["backend"], row["scope"]): row for row in summary["conditions"]}
    data = []
    for backend in BACKENDS:
        avg = aggregates[(backend, "Avg")]
        wins = sum(conditions[(backend, scenario)]["gain_pp"] > 0 for scenario in SCENARIOS)
        data.append({"backend": LABELS[backend], "parent": avg["parent_accuracy"],
                     "sparc": avg["plugin_accuracy"], "gain": avg["gain_pp"],
                     "ci_low": avg["ci95_pp"][0], "ci_high": avg["ci95_pp"][1],
                     "wins": wins})
    atomic_text(output / "figure2_backend_gains/figure2_backend_gains.csv", csv_text(data))
    fig, ax = plt.subplots(figsize=(7.15, 3.55))
    y = np.arange(len(data))[::-1] * 1.25
    for position, row in zip(y, data):
        ax.plot([row["parent"], row["sparc"]], [position, position], color=LIGHT_TEAL,
                linewidth=6, solid_capstyle="round", zorder=1)
        ax.scatter(row["parent"], position, s=88, facecolor="white", edgecolor=GRAY,
                   linewidth=2.1, zorder=3)
        ax.scatter(row["sparc"], position, s=105, marker="D", color=BLUE,
                   edgecolor="white", linewidth=0.8, zorder=4)
        ax.text(row["parent"], position + .22, f"{row['parent']:.2f}", ha="center",
                fontsize=8.7, color=GRAY)
        ax.text(row["sparc"] + .16, position + .20, f"{row['sparc']:.2f}", ha="left",
                fontsize=9.0, color=BLUE, fontweight="bold")
        ax.text((row["parent"] + row["sparc"]) / 2, position - .31,
                f"+{row['gain']:.2f} pp  ·  95% CI [{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
                f"  ·  {row['wins']}/15 wins",
                ha="center", va="top", fontsize=7.7, color="#244B6B")
    ax.set_yticks(y, [row["backend"] for row in data], fontsize=10)
    ax.set_xlim(38.0, 55.2)
    ax.set_ylim(-.70, y.max() + .72)
    ax.set_xlabel("Full-15 top-1 accuracy (%)", fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8.5)
    legend = [Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                     markeredgecolor=GRAY, markeredgewidth=1.8, markersize=7, label="Parent"),
              Line2D([0], [0], marker="D", color="none", markerfacecolor=BLUE,
                     markeredgecolor="white", markersize=7, label="Parent + SPARC")]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(.70, 1.04),
              frameon=False, ncol=2, fontsize=8.5, handletextpad=.4, columnspacing=1.2)
    ax.set_title("One frozen front end improves five unchanged backends", loc="left",
                 fontsize=11.5, fontweight="bold", pad=10)
    fig.tight_layout()
    save_figure(fig, output / "figure2_backend_gains", "figure2_backend_gains")


def efficiency_figure(output: Path, suite: dict) -> None:
    rows = suite["data_efficiency"]
    atomic_text(output / "figure3_data_efficiency/figure3_data_efficiency.csv", csv_text(rows))
    x = np.asarray([row["available_base_images"] for row in rows])
    utility = np.asarray([row["mean_utility_pp"] for row in rows])
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.plot(x, utility, color=BLUE, linewidth=2.35, marker="o", markersize=5.2)
    for xx, yy in zip(x, utility):
        ax.annotate(f"{yy:.2f}", (xx, yy), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7.2, color="#30485C")
    ax.set_title("500 images nearly match full-data utility",
                 fontsize=9.0, fontweight="bold", pad=6)
    ax.set_xlabel("Development base images", fontsize=8.0)
    ax.set_ylabel("Grouped-OOF utility (pp)", fontsize=8.0)
    ax.set_xticks(x)
    ax.tick_params(labelsize=7.2)
    ax.grid(color=GRID, linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(min(utility) - .25, max(utility) + .25)
    fig.tight_layout()
    save_figure(fig, output / "figure3_data_efficiency", "figure3_data_efficiency")


def routing_figure(output: Path, summary: dict) -> None:
    conditions = {(row["backend"], row["scope"]): row for row in summary["conditions"]}
    names = ("identity", "gaussian15", "median3", "unsharp2", "autocontrast",
             "inverse_zoom10", "learned_defocus_r1", "learned_glass_r075",
             "learned_motion_r1", "learned_snow_r1", "learned_frost_r075",
             "learned_fog_r1", "inverse_brightness_c055", "deployment_brightness")
    matrix = np.zeros((15, len(names)), dtype=float)
    data = []
    for i, scenario in enumerate(SCENARIOS):
        counts = conditions[("clip", scenario)]["expert_counts"]
        total = sum(int(value) for value in counts.values())
        for j, name in enumerate(names):
            matrix[i, j] = 100.0 * int(counts.get(name, 0)) / total
            data.append({"scenario": scenario, "expert": name,
                         "selection_pct": matrix[i, j]})
    full_dir = output / "figure4_routing"
    atomic_text(full_dir / "figure4_routing_full.csv", csv_text(data))
    # One representative is selected per predefined ImageNet-C family by the
    # largest dominant-expert frequency. This makes the mechanism legible in a
    # single column without presenting the subset as the complete distribution.
    family_ranges = ((0, 3, "Noise"), (3, 7, "Blur"),
                     (7, 11, "Weather"), (11, 15, "Digital"))
    representatives = []
    for start, stop, family in family_ranges:
        candidates = []
        for i in range(start, stop):
            dominant = int(np.argmax(matrix[i]))
            candidates.append((matrix[i, dominant], i, dominant))
        value, row_idx, expert_idx = max(candidates)
        representatives.append({
            "family": family,
            "scenario": SCENARIO_LABELS[row_idx],
            "expert": EXPERT_LABELS[expert_idx],
            "selection_pct": float(value),
            "selection_rule": "highest dominant-expert rate within family",
        })
    atomic_text(full_dir / "figure4_routing.csv", csv_text(representatives))
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    positions = np.arange(len(representatives))[::-1]
    values = [row["selection_pct"] for row in representatives]
    labels = [f"{row['scenario']}  →  {row['expert']}" for row in representatives]
    colors = [BLUE, "#287F9E", TEAL, "#5AB5A6"]
    bars = ax.barh(positions, values, color=colors, height=.58)
    for bar, value in zip(bars, values):
        ax.text(value - 2.2, bar.get_y() + bar.get_height() / 2, f"{value:.0f}%",
                va="center", ha="right", fontsize=7.6, color="white",
                fontweight="bold")
    ax.set_yticks(positions, labels, fontsize=7.2)
    ax.set_xlim(0, 103)
    ax.set_xlabel("Dominant-expert selection rate", fontsize=8.0)
    ax.set_title("Representative high-confidence routes", fontsize=9.0,
                 fontweight="bold", pad=6)
    ax.tick_params(axis="x", labelsize=7.0)
    ax.grid(axis="x", color=GRID, linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save_figure(fig, output / "figure4_routing", "figure4_routing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-cost-table", action="store_true",
        help="Generate the legacy cost table (not used by the current paper).",
    )
    args = parser.parse_args()
    suite_dir, summary_path, output = (args.suite.resolve(), args.summary.resolve(),
                                       args.output_dir.resolve())
    audit_path = suite_dir / "independent_audit.json"
    audit = json.loads(audit_path.read_text())
    suite = json.loads((suite_dir / "report.json").read_text())
    summary = json.loads(summary_path.read_text())
    if audit.get("status") != "passed_independent_lineage_metric_and_grouping_audit":
        raise RuntimeError("development suite has not passed its independent audit")
    if audit.get("suite_report_sha256") != sha256(suite_dir / "report.json"):
        raise RuntimeError("development audit/report hash mismatch")
    if summary.get("status") != "strictly_paired_frozen_evaluation_summary":
        raise RuntimeError("main summary is not the strict frozen summary")
    output.mkdir(parents=True, exist_ok=True)
    table2(output, suite)
    table3(output, suite)
    if args.include_cost_table:
        table4(output, summary)
    gains_figure(output, summary)
    efficiency_figure(output, suite)
    routing_figure(output, summary)
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                          "sha256": sha256(path)})
    manifest = {
        "schema_version": 1,
        "status": "generated_from_audited_main_and_development_evidence",
        "main_summary": str(summary_path), "main_summary_sha256": sha256(summary_path),
        "development_report": str(suite_dir / "report.json"),
        "development_report_sha256": sha256(suite_dir / "report.json"),
        "development_audit_sha256": sha256(audit_path),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "visual_inspection": "PENDING",
        "files": files,
    }
    atomic_text(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "output": str(output),
                      "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
