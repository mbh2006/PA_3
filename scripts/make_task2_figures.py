"""Generate the Task 2 tables and figures from the saved evaluation artefacts.

Final evaluation uses saved Kaggle batch-8 metrics. Historical training
and development curves use the Vast training records.
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
    svg = path.with_suffix(".svg")
    fig.savefig(svg, metadata={"Date": None})
    lines = svg.read_text(encoding="utf-8").splitlines()
    svg.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


def write_table(df: pd.DataFrame, name: str) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=".4f") + "\n")
    print("wrote", (TAB / f"{name}.csv").relative_to(ROOT))


# --------------------------------------------------------------------------- tables


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


def fig_final(metrics: dict, metric: str, filename: str, ylabel: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, suite in zip(axes, ("grounding", "natural")):
        conditions = metrics[suite]["overall"]["conditions"]
        values = [conditions[name][metric] for name in CONDITIONS]
        means = [v["mean"] for v in values]
        errors = [[max(0, v["mean"] - v["lo"]) for v in values],
                  [max(0, v["hi"] - v["mean"]) for v in values]]
        ax.bar(range(4), means, yerr=errors, capsize=4,
               color=["#225c75", "#9a522e", "#727347", "#6b548a"])
        ax.set_xticks(range(4), [COND_LABEL[name] for name in CONDITIONS], rotation=18)
        ax.set_title(SUITE_LABEL[suite])
    axes[0].set_ylabel(ylabel)
    fig.suptitle("Kaggle batch-8 evaluation · 95% complete-pair bootstrap intervals")
    save(fig, FIG / filename)


def fig_by_kind(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, suite in zip(axes, ("grounding", "natural")):
        rows = table[(table.suite == suite) & (table.question_type != "overall")]
        pivot = rows.pivot(index="question_type", columns="condition", values="accuracy")
        pivot[CONDITIONS].rename(columns=COND_LABEL).plot.bar(ax=ax, legend=False)
        ax.set(xlabel="", title=SUITE_LABEL[suite])
        ax.tick_params(axis="x", rotation=0)
    axes[0].set_ylabel("Accuracy / soft score")
    axes[1].legend(fontsize=8)
    fig.suptitle("Kaggle batch-8 evaluation by question type")
    save(fig, FIG / "by_kind")


def fig_caption_endpoints(endpoints: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    updates = [row["update"] for row in endpoints]
    ax.scatter(updates, [row["mean_caption_loss"] for row in endpoints], color="#225c75")
    ax.set(xlabel="Main update", ylabel="Caption development loss", xticks=updates,
           title="Kaggle checkpoint measurements · 96 captions per point")
    save(fig, FIG / "caption_dev_endpoints")


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
        ax.set_title(f"Vast · {title}", fontsize=10)
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
    ax.set_title("Vast · Stages 2–3 training loss", fontsize=10)
    ax.grid(alpha=0.3)
    save(fig, FIG / "train_loss")


def main() -> None:
    recovered = T2 / "recovered_kaggle"
    metrics = json.loads((recovered / "metrics.json").read_text())
    training = json.loads((T2 / "training.json").read_text())
    endpoints = json.loads((recovered / "caption_dev_endpoints.json").read_text())
    final = pd.read_csv(recovered / "condition_table.csv")

    write_table(final[final.question_type == "overall"], "condition_table")
    write_table(final[final.question_type != "overall"], "by_kind")
    write_table(stage_table(training), "stages")
    fig_final(metrics, "accuracy", "conditions_accuracy", "Accuracy / soft score")
    fig_final(metrics, "reference_nll", "nll_by_condition", "Reference answer NLL")
    fig_by_kind(final)
    fig_caption_endpoints(endpoints)
    fig_dev_curves(training)
    fig_train_loss(training)


if __name__ == "__main__":
    main()
