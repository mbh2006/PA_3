"""Task 2 three-stage curriculum (PA pp. 8-9).

Stage 1: connector only, two passes over 18,000 captions (2,250 updates).
Stage 2: + LoRA (q/k/v/o), presence 50% / counting 25% / caption 25%, 1,500 updates.
Stage 3: natural VQA 70% / presence 10% / counting 10% / caption 10%, 750 updates.

Per-stage 5% warm-up then cosine to 10% of the peak. Micro-batch 8 x accumulation 2.
Loss is the example-weighted mean, deliberately different from Task 1's token-weighted
objective. Every draw contributes two examples.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from task2.loss_sampler import MixtureSampler, SamplerState, realized_fractions


class CaptionCycler:
    """Deterministic caption sampler: two complete passes in order, no quota randomness."""

    def __init__(self, pool: list, seed: int = 6304):
        if len(pool) < 2:
            raise ValueError("caption pool too small")
        self.pool = pool
        self.order = list(range(len(pool)))
        self.pos = 0
        self.state = SamplerState(draws=0, counts={"caption": 0})

    def draw(self) -> tuple[str, list]:
        i = self.order[self.pos]
        j = self.order[(self.pos + 1) % len(self.order)]
        self.pos = (self.pos + 2) % len(self.order)
        self.state.draws += 1
        self.state.counts["caption"] += 1
        return "caption", [self.pool[i], self.pool[j]]

    def save_state(self) -> dict:
        return {"stage": "caption_cycle", "draws": self.state.draws,
                "counts": dict(self.state.counts), "pos": self.pos}

    def load_state(self, payload: dict) -> None:
        self.state = SamplerState(draws=payload["draws"], counts=dict(payload["counts"]))
        self.pos = int(payload.get("pos", 0))
from task2.model import forward_loss

MICRO_BATCH = 8
GRAD_ACCUM = 2
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM


@dataclass
class StageConfig:
    name: str
    updates: int
    connector_lr: float
    lora_lr: float | None
    mix: str


STAGES = [
    StageConfig("stage1_caption", 2250, 1e-3, None, "captions_only"),
    StageConfig("stage2_instruction", 1500, 1e-4, 2e-5, "stage2"),
    StageConfig("stage3_natural", 750, 5e-5, 1e-5, "stage3"),
]


def lr_at(step: int, peak: float, total: int) -> float:
    """5% warm-up, then cosine to 10% of the peak."""
    warmup = max(1, int(round(0.05 * total)))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    floor = 0.1 * peak
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


class CurriculumTrainer:
    def __init__(self, lm, connector, tokenizer, pools, out_dir: Path, *,
                 scale: float = 1.0, seed: int = 6304, device: str = "cuda"):
        self.lm = lm
        self.connector = connector
        self.tokenizer = tokenizer
        self.pools = pools
        self.out_dir = out_dir
        self.scale = scale
        self.device = device
        self.optimizer = None
        self.scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
        self.sampler = None
        self.log: list[dict] = []

    def _build_optimizer(self, stage: StageConfig):
        groups = [{"params": [p for p in self.connector.parameters() if p.requires_grad],
                   "lr": stage.connector_lr, "weight_decay": 0.0}]
        lora = [p for p in self.lm.parameters() if p.requires_grad]
        if stage.lora_lr is not None and lora:
            groups.append({"params": lora, "lr": stage.lora_lr, "weight_decay": 0.01})
        self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)

    def _set_lrs(self, stage: StageConfig, step: int):
        self.optimizer.param_groups[0]["lr"] = lr_at(step, stage.connector_lr, stage.updates)
        if len(self.optimizer.param_groups) > 1 and stage.lora_lr is not None:
            self.optimizer.param_groups[1]["lr"] = lr_at(step, stage.lora_lr, stage.updates)

    def _set_adapter_trainable(self, enabled: bool) -> int:
        """Task 2 prescribes connector-only Stage 1, then connector + LoRA in Stages 2-3."""
        n = 0
        for name, p in self.lm.named_parameters():
            if name.endswith((".A", ".B")):
                p.requires_grad_(enabled)
                n += p.numel()
        return n

    def run_stage(self, stage: StageConfig, *, max_updates: int | None = None,
                  dev_hook=None) -> dict:
        from task2.loss_sampler import MixtureSampler as _S

        updates = stage.updates if max_updates is None else min(stage.updates, max_updates)
        # a fresh sampler at every stage boundary, with that stage's mixture
        if stage.mix == "captions_only":
            self.sampler = CaptionCycler(self.pools["caption"], seed=6304)
        else:
            self.sampler = _S(stage.mix, self.pools, seed=6304)

        lora_enabled = stage.lora_lr is not None
        lora_params = self._set_adapter_trainable(lora_enabled)
        connector_params = sum(p.numel() for p in self.connector.parameters()
                               if p.requires_grad)
        self._build_optimizer(stage)
        self.lm.train()
        self.connector.train()
        print(f"[{stage.name}] connector={connector_params:,} "
              f"lora={lora_params if lora_enabled else 0:,} "
              f"lora_enabled={lora_enabled} updates={updates}", flush=True)

        start_step = self._resume_from(stage)
        dev_log = []
        if dev_hook is not None:
            dev_log.append({"stage": stage.name, "at_update": start_step,
                            **dev_hook(stage.name)})

        t0 = time.time()
        for step in range(start_step, updates):
            self._set_lrs(stage, step)
            self.optimizer.zero_grad(set_to_none=True)
            batch, kinds = [], []
            draws = EFFECTIVE_BATCH // 2
            for _ in range(draws):
                kind, examples = self.sampler.draw()
                kinds.append(kind)
                batch.extend(examples)
            for start in range(0, len(batch), MICRO_BATCH):
                chunk = batch[start:start + MICRO_BATCH]
                with torch.amp.autocast("cuda", dtype=torch.float16,
                                        enabled=(self.device == "cuda")):
                    loss, _ = forward_loss(self.lm, self.connector, self.tokenizer,
                                           chunk, scale=self.scale)
                self.scaler.scale(loss / GRAD_ACCUM).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g["params"]], 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if (step + 1) % 25 == 0 or step == 0:
                entry = {"stage": stage.name, "update": step + 1,
                         "loss": float(loss.detach()),
                         "connector_lr": self.optimizer.param_groups[0]["lr"],
                         "elapsed_s": round(time.time() - t0, 1)}
                self.log.append(entry)
                print(json.dumps(entry), flush=True)
            if (step + 1) % 250 == 0:
                self.save_checkpoint(stage, step + 1)
            if dev_hook is not None and (step + 1) % 500 == 0:
                dev_log.append({"stage": stage.name, "at_update": step + 1,
                                **dev_hook(stage.name)})

        self.save_stage(stage.name)
        if dev_hook is not None and (not dev_log or dev_log[-1]["at_update"] != updates):
            dev_log.append({"stage": stage.name, "at_update": updates,
                            **dev_hook(stage.name)})
        return {"stage": stage.name, "updates": updates,
                "dev": dev_log,
                "seconds": round(time.time() - t0, 1),
                "trainable_connector": connector_params,
                "trainable_lora": lora_params if lora_enabled else 0,
                "lora_enabled": lora_enabled,
                "mix_realized": realized_fractions(self.sampler.state.counts)}

    def save_checkpoint(self, stage: StageConfig, update: int) -> Path:
        ckpt_dir = self.out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        from task1.lora import adapter_state

        path = ckpt_dir / f"{stage.name}_{update:04d}.pt"
        torch.save({
            "stage": stage.name,
            "update": update,
            "connector": self.connector.state_dict(),
            "adapter": adapter_state(self.lm),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "sampler": self.sampler.save_state() if self.sampler else None,
            "rng": {"torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }, path)
        return path

    def _latest_checkpoint(self, stage: "StageConfig"):
        files = sorted((self.out_dir / "checkpoints").glob(f"{stage.name}_*.pt"))
        return files[-1] if files else None

    def _resume_from(self, stage: "StageConfig") -> int:
        """Restore connector, adapter, optimizer, scaler, RNG and sampler position."""
        ckpt = self._latest_checkpoint(stage)
        if ckpt is None:
            return 0
        from task1.lora import load_adapter_state

        payload = torch.load(ckpt, map_location="cuda")
        self.connector.load_state_dict(payload["connector"])
        if payload.get("adapter"):
            load_adapter_state(self.lm, payload["adapter"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload["scaler"])
        if payload.get("sampler") and self.sampler is not None:
            self.sampler.load_state(payload["sampler"])
        if payload.get("rng", {}).get("torch") is not None:
            torch.set_rng_state(payload["rng"]["torch"].cpu())
            if payload["rng"].get("cuda") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in payload["rng"]["cuda"]])
        print(f"[resume] {stage.name}: continuing from update {payload['update']} "
              f"({ckpt.name})", flush=True)
        return int(payload["update"])

    def save_stage(self, name: str) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{name}_weights.pt"
        from task1.lora import adapter_state

        torch.save({"connector": self.connector.state_dict(),
                    "lora": adapter_state(self.lm)}, path)
        return path


def run_curriculum(lm, connector, tokenizer, pools, out_dir: Path, *,
                   scale: float = 1.0, max_updates_per_stage: int | None = None,
                   dev_hook=None, stages=None) -> dict:
    trainer = CurriculumTrainer(lm, connector, tokenizer, pools, out_dir, scale=scale)
    summary = {"stages": []}
    for stage in (stages if stages is not None else STAGES):
        summary["stages"].append(trainer.run_stage(stage, max_updates=max_updates_per_stage,
                                                   dev_hook=dev_hook))
    summary["log"] = trainer.log
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
