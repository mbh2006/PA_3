"""Generate the Task 2 tables and figures from the saved evaluation artefacts.

Reads only ``results/task_02/eval.json`` and ``results/task_02/training.json``.
No GPU, no model, no regeneration. Every plotted number is copied from those files.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T2 = ROOT / "results" / "task_02"
FIG = T2 / "figures"
TAB = T2 / "tables"

CONDITIONS = ["correct", "complementary", "mismatched", "no_image"]
COND_LABEL = {
    "correct": "correct image",
    "complementary": "complementary",
    "mismatched": "mismatched",
    "no_image": "no image",
}
SUITE_LABEL = {"grounding": "annotation grounding", "natural": "natural VQAv2"}


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


def write_table(df: pd.DataFrame, name: str) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=".4f") + "\n")
    print("wrote", (TAB / f"{name}.csv").relative_to(ROOT))


# --------------------------------------------------------------------------- tables


def condition_table(eval_d: dict) -> pd.DataFrame:
    rows = []
    for suite in ("natural", "grounding"):
        s = eval_d[suite]
        for cond in CONDITIONS:
            rows.append(
                {
                    "suite": suite,
                    "n_pairs": s["n_pairs"],
                    "n_examples": s["n_examples"],
                    "condition": cond,
                    "accuracy": s["conditions"][cond]["accuracy"],
                    "empty_rate": s["conditions"][cond]["empty_rate"],
                    "pair_accuracy": s["pair_accuracy"].get(cond),
                    "identical_answer_rate": s["identical_answer_rate"],
                    "reference_nll": {
                        "correct": s["nll_correct"],
                        "complementary": s["nll_complementary"],
                        "no_image": s["nll_no_image"],
                    }.get(cond),
                }
            )
        b = s["bootstrap_accuracy_advantage"]
        rows.append(
            {
                "suite": suite,
                "n_pairs": s["n_pairs"],
                "n_examples": s["n_examples"],
                "condition": "correct_minus_complementary",
                "accuracy": s["correct_advantage_accuracy"],
                "empty_rate": None,
                "pair_accuracy": None,
                "identical_answer_rate": None,
                "reference_nll": s["swap_penalty"],
                "boot_lo": b["lo"],
                "boot_hi": b["hi"],
                "boot_n": b["n"],
            }
        )
    return pd.DataFrame(rows)


def by_kind_table(eval_d: dict) -> pd.DataFrame:
    rows = []
    for suite in ("natural", "grounding"):
        for kind, v in eval_d[suite]["by_kind"].items():
            rows.append(
                {
                    "suite": suite,
                    "subtype": kind,
                    "n": v["n"],
                    "correct_accuracy": v["correct_accuracy"],
                    "complementary_accuracy": v["complementary_accuracy"],
                    "no_image_accuracy": v["no_image_accuracy"],
                }
            )
    return pd.DataFrame(rows)


def stage_table(training: dict) -> pd.DataFrame:
    rows = []
    for st in training["stages"]:
        rows.append(
            {
                "stage": st["stage"],
                "updates": st["updates"],
                "seconds": st["seconds"],
                "lora_enabled": st["lora_enabled"],
                "trainable_connector": st["trainable_connector"],
                "trainable_lora": st["trainable_lora"],
                "dev_points": len(st.get("dev", [])),
                "mix_realized": json.dumps(st.get("mix_realized", {})),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures


def fig_conditions(eval_d: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, suite in zip(axes, ("natural", "grounding")):
        s = eval_d[suite]
        vals = [s["conditions"][c]["accuracy"] for c in CONDITIONS]
        bars = ax.bar(
            range(len(CONDITIONS)),
            vals,
            color=["#2b6cb0", "#c05621", "#718096", "#a0aec0"],
        )
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
        b = s["bootstrap_accuracy_advantage"]
        ax.set_title(
            f"{SUITE_LABEL[suite]} ({s['n_pairs']} pairs, {s['n_examples']} examples)\n"
            f"correct − complementary = {s['correct_advantage_accuracy']:+.3f} "
            f"[{b['lo']:.3f}, {b['hi']:.3f}]",
            fontsize=10,
        )
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([COND_LABEL[c] for c in CONDITIONS], rotation=15)
        ax.set_ylim(0, 0.65)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("accuracy")
    save(fig, FIG / "conditions_accuracy")


def fig_nll(eval_d: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.35
    conds = ["correct", "complementary", "no_image"]
    for i, suite in enumerate(("natural", "grounding")):
        s = eval_d[suite]
        vals = [s["nll_correct"], s["nll_complementary"], s["nll_no_image"]]
        bars = ax.bar(
            [x + (i - 0.5) * width for x in range(len(conds))],
            vals,
            width,
            label=SUITE_LABEL[suite],
        )
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(["correct", "complementary", "no image"])
    ax.set_ylabel("reference-answer NLL (nats)")
    ax.set_title("Reference NLL by image condition (lower is better)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save(fig, FIG / "nll_by_condition")


def fig_by_kind(eval_d: dict) -> None:
    kinds, correct, comp, suite_of = [], [], [], []
    for suite in ("natural", "grounding"):
        for kind, v in eval_d[suite]["by_kind"].items():
            kinds.append(f"{kind}\n({suite[:4]})")
            correct.append(v["correct_accuracy"])
            comp.append(v["complementary_accuracy"])
            suite_of.append(suite)
    x = range(len(kinds))
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar([i - 0.2 for i in x], correct, 0.4, label="correct image")
    ax.bar([i + 0.2 for i in x], comp, 0.4, label="complementary")
    for i, (c, k) in enumerate(zip(correct, comp)):
        ax.text(i - 0.2, c + 0.012, f"{c:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, k + 0.012, f"{k:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(kinds, fontsize=9)
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy by question subtype", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save(fig, FIG / "by_kind")


def fig_dev_curves(training: dict) -> None:
    """Dev NLL against cumulative main updates, with the stage boundary marked.

    ``training.json`` covers Stages 2-3 only (this run resumed from the Stage 1
    checkpoint), so the axis is offset by the 2,250 Stage 1 updates to show each point at
    its true position in the 4,500-update curriculum.
    """
    stage1_updates = 2250
    points = []
    cumulative = stage1_updates
    for st in training["stages"]:
        for p in st.get("dev", []):
            points.append(
                {
                    "stage": st["stage"],
                    "cumulative_update": cumulative + p["at_update"],
                    **{f"g_{k}": v for k, v in p["grounding"].items()},
                    **{f"n_{k}": v for k, v in p["natural"].items()},
                }
            )
        cumulative += st["updates"]
    df = pd.DataFrame(points).sort_values("cumulative_update")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, prefix, title in (
        (axes[0], "g_", "annotation grounding (dev, 24 pairs)"),
        (axes[1], "n_", "natural VQAv2 (dev, 32 pairs)"),
    ):
        ax.plot(df["cumulative_update"], df[f"{prefix}nll_correct"], "o-", label="correct image")
        ax.plot(
            df["cumulative_update"],
            df[f"{prefix}nll_swapped"],
            "s--",
            label="complementary swap",
        )
        ax.plot(
            df["cumulative_update"],
            df[f"{prefix}nll_no_image"],
            "^:",
            label="no image",
            alpha=0.7,
        )
        boundary = stage1_updates + training["stages"][0]["updates"]
        ax.axvline(boundary, color="k", ls=":", lw=1)
        ax.text(boundary, ax.get_ylim()[1], " stage 2 → 3", va="top", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("cumulative main updates (Stage 1 = 2,250, no dev points recorded)")
        ax.set_ylabel("reference-answer NLL (nats)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    save(fig, FIG / "dev_nll_curves")


def fig_train_loss(training: dict) -> None:
    stage1_updates = 2250
    log = pd.DataFrame(training["log"])
    offsets = {
        "stage2_instruction": stage1_updates,
        "stage3_natural": stage1_updates + training["stages"][0]["updates"],
    }
    log["cumulative_update"] = log["update"] + log["stage"].map(offsets).fillna(0)
    log = log.sort_values("cumulative_update")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(log["cumulative_update"], log["loss"], lw=1)
    boundary = stage1_updates + training["stages"][0]["updates"]
    ax.axvline(boundary, color="k", ls=":", lw=1)
    ax.text(boundary, ax.get_ylim()[1], " stage 2 → 3", va="top", fontsize=8)
    ax.set_xlabel("cumulative main updates (Stages 2–3 of the 4,500-update curriculum)")
    ax.set_ylabel("training loss (example-weighted CE)")
    ax.set_title("Stages 2–3 training loss", fontsize=10)
    ax.grid(alpha=0.3)
    save(fig, FIG / "train_loss")


def main() -> None:
    eval_d = json.loads((T2 / "eval.json").read_text())
    training = json.loads((T2 / "training.json").read_text())

    write_table(condition_table(eval_d), "condition_table")
    write_table(by_kind_table(eval_d), "by_kind")
    write_table(stage_table(training), "stages")

    fig_conditions(eval_d)
    fig_nll(eval_d)
    fig_by_kind(eval_d)
    fig_dev_curves(training)
    fig_train_loss(training)


if __name__ == "__main__":
    main()
