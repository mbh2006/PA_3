"""Visual connector: 768 -> 1024 -> 1024 MLP with GELU (PA p. 6).

Trained in FP32 while the language backbone stays FP16 and frozen. Output magnitude is
calibrated against text embeddings using training data only; the rule and the resulting
scale are recorded in the run configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

VISION_DIM = 768
HIDDEN_DIM = 1024
N_VISUAL_TOKENS = 196


class Connector(nn.Module):
    def __init__(self, in_dim: int = VISION_DIM, hidden: int = HIDDEN_DIM, out_dim: int = HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(features)))


@dataclass
class MagnitudeCalibration:
    rule: str
    text_rms: float
    raw_visual_rms: float
    scale: float

    def as_dict(self) -> dict:
        return {"rule": self.rule, "text_rms": self.text_rms,
                "raw_visual_rms": self.raw_visual_rms, "scale": self.scale}


@torch.no_grad()
def calibrate_magnitude(
    connector: Connector,
    image_features: torch.Tensor,
    text_embeddings: torch.Tensor,
    max_images: int = 256,
) -> MagnitudeCalibration:
    """Scale the connector output so its RMS matches text embeddings.

    Rule: ``scale = rms(text embeddings) / rms(raw connector output)`` computed over the
    first ``max_images`` training images and the provided training text embeddings.
    Uses training data only.
    """
    feats = image_features[:max_images].float()
    visual = connector(feats).reshape(-1)
    text = text_embeddings.float()
    text_rms = float(torch.sqrt(torch.mean(text.pow(2))))
    visual_rms = float(torch.sqrt(torch.mean(visual.pow(2))))
    if visual_rms == 0.0:
        raise ValueError("connector output is all zeros")
    return MagnitudeCalibration(
        rule="rms_match: scale = rms(text) / rms(connector_output), training data only",
        text_rms=text_rms,
        raw_visual_rms=visual_rms,
        scale=text_rms / visual_rms,
    )


def apply_scale(connector_out: torch.Tensor, scale: float) -> torch.Tensor:
    return connector_out * scale
