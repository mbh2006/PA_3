"""Task 2 losses and the seeded two-example mixture sampler (PA pp. 8-9)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


def example_weighted_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """L = -1/|B| Σ_i (1/|R_i|) Σ_t log p(y_it | ...).

    Each example contributes equally regardless of answer length; this is deliberately
    different from Task 1's token-weighted objective. Examples with no supervised tokens
    are excluded from the batch mean.
    """
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    vocab = shift_logits.size(-1)
    per_token = F.cross_entropy(
        shift_logits.reshape(-1, vocab), shift_labels.reshape(-1),
        ignore_index=-100, reduction="none",
    ).view(shift_labels.shape)
    mask = shift_labels != -100
    n_valid = mask.sum(dim=1)
    keep = n_valid > 0
    if not bool(keep.any()):
        return torch.zeros((), device=logits.device, dtype=logits.dtype, requires_grad=True)
    per_example = (per_token * mask).sum(dim=1)[keep] / n_valid[keep]
    return per_example.mean()


def reference_answer_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-example mean NLL of the reference answer, excluding termination."""
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    vocab = shift_logits.size(-1)
    per_token = F.cross_entropy(
        shift_logits.reshape(-1, vocab), shift_labels.reshape(-1),
        ignore_index=-100, reduction="none",
    ).view(shift_labels.shape)
    mask = shift_labels != -100
    n_valid = mask.sum(dim=1).clamp(min=1)
    return (per_token * mask).sum(dim=1) / n_valid


STAGE2_MIX = {"presence": 0.5, "counting": 0.25, "caption": 0.25}
STAGE3_MIX = {"natural": 0.7, "presence": 0.1, "counting": 0.1, "caption": 0.1}


@dataclass
class SamplerState:
    draws: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"draws": self.draws, "counts": dict(self.counts)}


class MixtureSampler:
    """Seeded sampler; each draw yields two examples (a full question pair or two captions).

    Probabilities are used as-is: no exact quotas are enforced, and pools may repeat.
    """

    def __init__(self, stage: str, pools: dict[str, list], seed: int = 6304):
        self.stage = stage
        self.mix = {"stage2": STAGE2_MIX, "stage3": STAGE3_MIX}[stage]
        self.pools = pools
        self.rng = np.random.default_rng(seed if stage == "stage2" else [seed, 3])
        self.state = SamplerState(counts={k: 0 for k in self.mix})
        missing = [k for k in self.mix if k not in pools or len(pools[k]) == 0]
        if missing:
            raise ValueError(f"missing pools for {missing}")

    def draw(self) -> tuple[str, list]:
        """Return ``(kind, [example_a, example_b])``; never a single example."""
        names = list(self.mix)
        probs = np.array([self.mix[n] for n in names], dtype=float)
        probs /= probs.sum()
        kind = str(self.rng.choice(names, p=probs))
        pool = self.pools[kind]

        if kind == "caption":
            # replay contributes two caption examples per draw
            idxs = self.rng.integers(0, len(pool), size=2)
            examples = [pool[int(i)] for i in idxs]
        else:
            item = pool[int(self.rng.integers(0, len(pool)))]
            examples = list(item)
            if len(examples) != 2:
                raise ValueError(f"{kind} pool items must be complete pairs, got {len(examples)}")

        self.state.draws += 1
        self.state.counts[kind] += 1
        return kind, examples

    def save_state(self) -> dict:
        return {
            "stage": self.stage,
            "draws": self.state.draws,
            "counts": dict(self.state.counts),
            "rng": self.rng.bit_generator.state,
        }

    def load_state(self, payload: dict) -> None:
        if payload["stage"] != self.stage:
            raise ValueError("stage mismatch")
        self.state = SamplerState(draws=payload["draws"], counts=dict(payload["counts"]))
        self.rng.bit_generator.state = payload["rng"]


def realized_fractions(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}
