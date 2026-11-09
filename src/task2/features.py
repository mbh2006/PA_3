"""SigLIP feature extraction and caching (PA p. 7).

* encoder frozen, `google/siglip-base-patch16-224`
* 196 final patch features of width 768 in 14x14 spatial order
* SigLIP's own final layer norm is applied exactly once (it is the last op of the
  vision tower, so `last_hidden_state` is already normalised - asserted in tests)
* deterministic preprocessing, cached in FP16, encoder unloaded before LM training
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

SIGLIP_ID = "google/siglip-base-patch16-224"
N_PATCHES = 196
FEATURE_DIM = 768
IMAGE_SIZE = 224


def load_vision_encoder(device: str = "cuda"):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(SIGLIP_ID, dtype=torch.float16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_processor():
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(SIGLIP_ID)


@torch.no_grad()
def extract_features(model, processor, images, *, device="cuda") -> torch.Tensor:
    """Return ``(n, 196, 768)`` FP16 features; the final layer norm is applied once."""
    inputs = processor(images=images, return_tensors="pt", size={"height": IMAGE_SIZE,
                                                                 "width": IMAGE_SIZE})
    pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.float16)
    out = model.vision_model(pixel_values=pixel_values)
    feats = out.last_hidden_state
    if feats.shape[1:] != (N_PATCHES, FEATURE_DIM):
        raise ValueError(f"unexpected feature shape {tuple(feats.shape)}")
    return feats.detach().to(torch.float16).cpu()


def build_cache(image_paths: dict[int, Path], out_dir: Path, *, batch_size: int = 32,
                device: str = "cuda", log_every: int = 500) -> Path:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_vision_encoder(device)
    processor = load_processor()

    ids = sorted(image_paths)
    chunks = []
    meta = []
    for start in range(0, len(ids), batch_size):
        chunk_ids = ids[start:start + batch_size]
        images = [Image.open(image_paths[i]).convert("RGB") for i in chunk_ids]
        feats = extract_features(model, processor, images, device=device)
        chunks.append(feats)
        meta.extend(chunk_ids)
        if log_every and (start // batch_size) % max(1, log_every // batch_size) == 0:
            print(f"  cached {start + len(chunk_ids)}/{len(ids)}", flush=True)

    all_feats = torch.cat(chunks, dim=0)
    path = out_dir / "siglip_features.pt"
    torch.save({"features": all_feats, "image_ids": meta}, path)
    (out_dir / "feature_cache.json").write_text(json.dumps({
        "encoder": SIGLIP_ID,
        "n_images": len(meta),
        "shape": list(all_feats.shape[1:]),
        "dtype": str(all_feats.dtype),
        "spatial_order": "14x14 row-major",
        "final_layer_norm": "applied once by the SigLIP vision tower",
    }, indent=2) + "\n")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return path


def load_cache(path: Path) -> tuple[dict[int, torch.Tensor], dict]:
    payload = torch.load(path, map_location="cpu")
    feats = payload["features"]
    ids = payload["image_ids"]
    table = {int(i): feats[k] for k, i in enumerate(ids)}
    meta = json.loads((path.parent / "feature_cache.json").read_text())
    return table, meta


def pool_2x2(features: torch.Tensor) -> torch.Tensor:
    """Optional 49-token variant: average each 2x2 block of the 14x14 grid."""
    n, p, d = features.shape
    if p != N_PATCHES:
        raise ValueError("expected 196 patches")
    grid = features.reshape(n, 14, 14, d)
    pooled = grid.reshape(n, 7, 2, 7, 2, d).mean(dim=(2, 4))
    return pooled.reshape(n, 49, d)
