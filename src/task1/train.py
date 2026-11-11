"""Task 1 training loop.

Implements the prescribed recipe exactly:

* token-weighted SFT loss over the **effective batch** (not a mean of per-micro-batch means)
* FP32 master parameters/adapters, FP16 autocast compute, dynamic gradient scaling
* unscale -> add L2-SP gradient -> global-norm clip -> step, in that order
* cosine schedule with 38 warm-up updates, one scheduler step per optimizer update
* recovery checkpoints carrying optimizer/scheduler/scaler/RNG/data position
* skipped updates are logged and retried so the 1,250-update budget is honoured
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from task1.data import MathExample, collate
from task1.l2sp import L2SPReference
from task1.optim import make_optimizer, make_scheduler

TOTAL_UPDATES = 1250
WARMUP_UPDATES = 38
MICRO_BATCH = 2
GRAD_ACCUM = 16
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM
MAX_RETRIES_PER_UPDATE = 3


@dataclass
class TrainConfig:
    variant: str
    peak_lr: float
    lambda_l2sp: float = 0.0
    updates: int = TOTAL_UPDATES
    micro_batch: int = MICRO_BATCH
    grad_accum: int = GRAD_ACCUM
    warmup: int = WARMUP_UPDATES
    clip_grad_norm: float = 1.0
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    checkpoint_updates: tuple[int, ...] = (312, 625, 937, 1250)
    seed: int = 6304
    log_every: int = 10
    # Padded-token budget for one micro-batch. Two 2,048-token sequences do not fit on a
    # 16 GB T4 (measured), so an over-budget pair is processed as two single-example
    # micro-batches. The effective batch, the token-weighted objective and therefore the
    # gradient are unchanged; only the grouping differs.
    max_micro_tokens: int = 1600
    # Loss mode. The standard HF loss is faster for ordinary batch lengths; the chunked
    # vocabulary projection is used automatically once a micro-batch is longer than
    # ``chunked_above_tokens``, where the (batch x seq x 151936) logit matrix would
    # otherwise dominate memory. Both paths compute the same objective.
    chunked_loss: bool = False
    loss_chunk_size: int = 256
    chunked_above_tokens: int = 512


def micro_batch_scale(n_valid_micro: int, n_valid_total: int) -> float:
    """Weight applied to a micro-batch mean loss so the sum equals the token-weighted mean."""
    if n_valid_total <= 0:
        raise ValueError("effective batch has no supervised tokens")
    return n_valid_micro / n_valid_total


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        examples: list[MathExample],
        tokenizer,
        config: TrainConfig,
        *,
        device: str = "cuda",
        out_dir: Path | None = None,
        reference: L2SPReference | None = None,
        is_adapter: bool = False,
    ) -> None:
        self.model = model
        self.examples = examples
        self.tokenizer = tokenizer
        self.cfg = config
        self.device = device
        self.out_dir = Path(out_dir) if out_dir else None
        self.reference = reference
        self.is_adapter = is_adapter

        self.params = [p for p in model.parameters() if p.requires_grad]
        if not self.params:
            raise ValueError("no trainable parameters")
        self.optimizer = make_optimizer(
            model, config.peak_lr, weight_decay=config.weight_decay,
            betas=config.betas, eps=config.eps,
        )
        self.scheduler = make_scheduler(self.optimizer, warmup=config.warmup, total=config.updates)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        self.log_path = (self.out_dir / "train_log.jsonl") if self.out_dir else None
        self.checkpoint_dir = (self.out_dir / "checkpoints") if self.out_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # -- batch preparation ---------------------------------------------------------
    def _micro_batches(self, batch_indices: list[int]):
        cfg = self.cfg
        for start in range(0, len(batch_indices), cfg.micro_batch):
            chunk = [self.examples[i] for i in batch_indices[start:start + cfg.micro_batch]]
            max_len = max(ex.n_tokens for ex in chunk)
            if len(chunk) > 1 and len(chunk) * max_len > cfg.max_micro_tokens:
                # over the measured memory budget: one example per micro-batch
                for ex in chunk:
                    yield collate([ex], self.pad_token_id)
            else:
                yield collate(chunk, self.pad_token_id)

    def _to_device(self, batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}

    # -- one optimizer update ------------------------------------------------------
    def update_once(self, batch_indices: list[int]) -> dict:
        cfg = self.cfg
        total_valid = sum(self.examples[i].n_supervised for i in batch_indices)
        self.optimizer.zero_grad(set_to_none=True)

        sft_sum = 0.0
        for batch in self._micro_batches(batch_indices):
            tensors = self._to_device(batch)
            n_valid = int((tensors["labels"][:, 1:] != -100).sum())
            seq_len = int(tensors["input_ids"].shape[1])
            use_chunked = cfg.chunked_loss or seq_len > cfg.chunked_above_tokens
            if use_chunked:
                from task1.chunked_loss import backward_chunked_lm_loss

                # GradScaler lazily creates its scale on the first scale() call; the
                # chunked path scales the loss itself, so initialise it explicitly once.
                self.scaler.scale(torch.zeros((), device=self.device))
                sum_ce, _ = backward_chunked_lm_loss(
                    self.model,
                    tensors["input_ids"],
                    tensors["attention_mask"],
                    tensors["labels"],
                    total_valid=total_valid,
                    loss_scale=float(self.scaler.get_scale()),
                    chunk_size=cfg.loss_chunk_size,
                )
                sft_sum += sum_ce / total_valid
            else:
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(self.device == "cuda")):
                    out = self.model(
                        input_ids=tensors["input_ids"],
                        attention_mask=tensors["attention_mask"],
                        labels=tensors["labels"],
                    )
                mean_loss = out.loss
                scaled = mean_loss * micro_batch_scale(n_valid, total_valid)
                self.scaler.scale(scaled).backward()
                sft_sum += float(mean_loss.detach()) * micro_batch_scale(n_valid, total_valid)

        scale_before = self.scaler.get_scale()
        self.scaler.unscale_(self.optimizer)

        lambda_omega = torch.zeros((), device=self.device)
        if cfg.lambda_l2sp > 0.0:
            if self.reference is None:
                raise ValueError("lambda_l2sp > 0 requires an L2SPReference")
            lambda_omega = self.reference.add_gradient(self.model, cfg.lambda_l2sp)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, cfg.clip_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        skipped = self.scaler.get_scale() < scale_before
        return {
            "sft_loss": sft_sum,
            "lambda_omega": float(lambda_omega.detach()),
            "total_loss": sft_sum + float(lambda_omega.detach()),
            "grad_norm": float(grad_norm.detach()),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "skipped": skipped,
            "scale": float(self.scaler.get_scale()),
        }

    # -- training loop -------------------------------------------------------------
    def train(self, *, max_updates: int | None = None) -> dict:
        cfg = self.cfg
        n_updates = cfg.updates if max_updates is None else min(cfg.updates, max_updates)
        positions = list(range(0, len(self.examples) - EFFECTIVE_BATCH + 1, EFFECTIVE_BATCH))
        if n_updates > len(positions):
            raise ValueError(f"{n_updates} updates need {n_updates * EFFECTIVE_BATCH} examples, "
                             f"have {len(self.examples)}")

        t0 = time.time()
        update = 0
        cursor = 0
        skipped_total = 0
        retries = 0
        history: list[dict] = []
        peak_mem = 0.0

        while update < n_updates:
            batch_indices = positions[cursor]
            batch_indices = list(range(batch_indices, batch_indices + EFFECTIVE_BATCH))
            stats = self.update_once(batch_indices)

            if stats["skipped"]:
                skipped_total += 1
                retries += 1
                if retries > MAX_RETRIES_PER_UPDATE * n_updates:
                    raise RuntimeError("too many skipped updates")
                # do not advance the data cursor: the same effective batch is retried
                continue

            retries = 0
            cursor += 1
            update += 1
            stats.update({"update": update, "examples_seen": update * EFFECTIVE_BATCH,
                          "elapsed_s": round(time.time() - t0, 1)})
            history.append(stats)

            if self.device == "cuda":
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1e9)

            if self.log_path and (update % cfg.log_every == 0 or update in cfg.checkpoint_updates):
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(stats) + "\n")

            if update in cfg.checkpoint_updates:
                self.save_checkpoint(update, examples_seen=update * EFFECTIVE_BATCH)

        summary = {
            "variant": cfg.variant,
            "updates": update,
            "examples_seen": update * EFFECTIVE_BATCH,
            "skipped_updates": skipped_total,
            "elapsed_s": round(time.time() - t0, 1),
            "peak_mem_gb": peak_mem,
            "final_sft_loss": history[-1]["sft_loss"] if history else None,
            "final_lr": history[-1]["lr"] if history else None,
            "history": history,
        }
        if self.out_dir:
            (self.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    # -- checkpoints ---------------------------------------------------------------
    def _rng_state(self) -> dict:
        state = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if self.device == "cuda" and torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def save_checkpoint(self, update: int, *, examples_seen: int) -> Path:
        if self.checkpoint_dir is None:
            raise RuntimeError("no checkpoint directory")
        payload = {
            "update": update,
            "examples_seen": examples_seen,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": self._rng_state(),
            "config": {
                "variant": self.cfg.variant,
                "peak_lr": self.cfg.peak_lr,
                "lambda_l2sp": self.cfg.lambda_l2sp,
                "updates": self.cfg.updates,
                "is_adapter": self.is_adapter,
            },
        }
        if self.is_adapter:
            from task1.lora import adapter_state

            payload["adapter"] = adapter_state(self.model)
        else:
            payload["model"] = self.model.state_dict()
        path = self.checkpoint_dir / f"checkpoint_{update:04d}.pt"
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: Path) -> dict:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if self.is_adapter:
            from task1.lora import load_adapter_state

            load_adapter_state(self.model, payload["adapter"])
        else:
            self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        # RNG states must be restored on the CPU regardless of the checkpoint map_location
        torch.set_rng_state(payload["rng"]["torch"].cpu())
        np.random.set_state(payload["rng"]["numpy"])
        random.setstate(payload["rng"]["python"])
        if "cuda" in payload["rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in payload["rng"]["cuda"]])
        return payload


def checkpoint_size_bytes(path: Path) -> int:
    return os.path.getsize(path)


def displacement(model: nn.Module, reference: L2SPReference) -> float:
    """||θ − θ0||_2 over unique parameters, in FP32."""
    total = 0.0
    for name, p in model.named_parameters():
        # the reference may live on CPU while the model is on the accelerator
        diff = p.detach().float().cpu() - reference.tensors[name].float().cpu()
        total += float(diff.pow(2).sum())
    return math.sqrt(total)
