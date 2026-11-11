"""Generate the Task 1 and Task 3 figures from saved results (no GPU required).

Reads only files already copied into the local outputs/ tree; missing inputs are
skipped rather than substituted.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T1 = ROOT / "outputs" / "task_01"
T3 = ROOT / "outputs" / "task_03"
FIG1 = T1 / "figures"
FIG3 = T3 / "figures"


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    print("wrote", path.with_suffix(".png"))


def task1_benchmarks() -> None:
    csv = T1 / "tables" / "condition_table.csv"
    if not csv.exists():
        print("missing", csv)
        return
    df = pd.read_csv(csv)
    metrics = ["gsm8k_flexible_extract", "hellaswag_acc_norm", "boolq_acc", "arc_easy_acc_norm"]
    labels = ["GSM8K", "HellaSwag", "BoolQ", "ARC-Easy"]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(metrics))
    width = 0.8 / len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics]
        ax.bar([p + i * width for p in x], vals, width, label=row["condition"])
    ax.set_xticks([p + 0.4 - width / 2 for p in x])
    ax.set_xticklabels(labels)
    ax.set_ylabel("score")
    ax.set_title("Task 1 — seven evaluation conditions")
    ax.legend(fontsize=7, ncol=2)
    save(fig, FIG1 / "benchmarks_by_condition")

    # deltas vs baseline
    base = df[df["condition"] == "baseline"].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (_, row) in enumerate(df.iterrows()):
        if row["condition"] == "baseline":
            continue
        vals = [(row[m] - base[m]) * 100 for m in metrics]
        ax.bar([p + i * width for p in x], vals, width, label=row["condition"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([p + 0.4 - width / 2 for p in x])
    ax.set_xticklabels(labels)
    ax.set_ylabel("change vs baseline (percentage points)")
    ax.set_title("Task 1 — deltas from the shared baseline")
    ax.legend(fontsize=7, ncol=2)
    save(fig, FIG1 / "deltas_vs_baseline")


def task1_learning_curves() -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    any_found = False
    for run in sorted((T1 / "runs").iterdir()):
        log = run / "train" / "train_log.jsonl"
        if not log.exists():
            continue
        rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        if not rows:
            continue
        ax.plot([r["update"] for r in rows], [r["sft_loss"] for r in rows],
                label=run.name, linewidth=1.2)
        any_found = True
    if not any_found:
        print("no training logs found")
        plt.close(fig)
        return
    ax.set_xlabel("optimizer update")
    ax.set_ylabel("SFT loss (token-weighted)")
    ax.set_title("Task 1 — training loss")
    ax.legend(fontsize=7)
    save(fig, FIG1 / "sft_loss_curves")


def task1_diagnostics() -> None:
    rows = []
    for run in sorted((T1 / "runs").iterdir()):
        rep = run / "report.json"
        if not rep.exists():
            continue
        d = json.loads(rep.read_text())
        for w in d.get("weight_diagnostics", []) or []:
            rows.append({"condition": run.name, "layer": w["layer"],
                         "rho_k": w.get("rho_k"), "eta": w.get("eta")})
    if not rows:
        print("no diagnostics found")
        return
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("rho_k", "eta")):
        sub = df.dropna(subset=[metric])
        if sub.empty:
            continue
        for cond, g in sub.groupby("condition"):
            ax.plot(g["layer"], g[metric], marker="o", label=cond)
        ax.set_xlabel("down_proj layer")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend(fontsize=6)
    save(fig, FIG1 / "weight_diagnostics")


def task3_curves() -> None:
    off = T3 / "offline"
    for ds in ("math", "mathvista"):
        path = off / f"summary_{ds}_heldout.json"
        if not path.exists():
            continue
        df = pd.DataFrame(json.loads(path.read_text()))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for method, g in df.groupby("method"):
            g = g.sort_values("budget")
            ax.plot(g["budget"], g["accuracy"], marker="o", label=method, linewidth=1.2)
        oracle = df[df["method"] == "first_ans"].sort_values("budget")
        ax.plot(oracle["budget"], oracle["oracle_coverage"], "k--", label="oracle coverage")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("candidate budget N")
        ax.set_ylabel("accuracy")
        ax.set_title(f"Task 3 — {ds}: accuracy and oracle coverage")
        ax.legend(fontsize=6, ncol=2)
        save(fig, FIG3 / f"accuracy_vs_budget_{ds}")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for method, g in df.groupby("method"):
            g = g.sort_values("mean_cost_tokens")
            ax.plot(g["mean_cost_tokens"], g["accuracy"], marker="o", label=method)
        ax.set_xlabel("mean generated tokens")
        ax.set_ylabel("accuracy")
        ax.set_title(f"Task 3 — {ds}: accuracy vs token cost")
        ax.legend(fontsize=6, ncol=2)
        save(fig, FIG3 / f"accuracy_vs_cost_{ds}")


def task3_live() -> None:
    path = T3 / "live" / "live_summary.json"
    if not path.exists():
        print("missing live summary")
        return
    d = json.loads(path.read_text())
    names = list(d["accuracy"])
    acc = [d["accuracy"][n] for n in names]
    tok = [d["mean_output_tokens"][n] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, acc, color=["#4c72b0", "#55a868", "#c44e52"])
    axes[0].set_ylabel("accuracy (50 MATH questions)")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(names, tok, color=["#4c72b0", "#55a868", "#c44e52"])
    axes[1].set_ylabel("mean generated tokens")
    axes[1].tick_params(axis="x", rotation=15)
    fig.suptitle("Task 3 — live strategies at a 3,072-token allowance")
    save(fig, FIG3 / "live_strategies")


def main() -> None:
    task1_benchmarks()
    task1_learning_curves()
    task1_diagnostics()
    task3_curves()
    task3_live()
    print("FIGURES_DONE")


if __name__ == "__main__":
    main()
