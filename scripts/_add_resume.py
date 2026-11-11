#!/usr/bin/env python3
"""Add per-stage resume from the latest checkpoint and persist adapter weights."""

import pathlib

p = pathlib.Path("src/task2/train.py")
s = p.read_text(encoding="utf-8")

# persist adapter weights inside each 250-update checkpoint
s = s.replace(
'''        path = ckpt_dir / f"{stage.name}_{update:04d}.pt"
        torch.save({
            "stage": stage.name,
            "update": update,
            "connector": self.connector.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "sampler": self.sampler.save_state() if self.sampler else None,
        }, path)
        return path''',
'''        from task1.lora import adapter_state

        path = ckpt_dir / f"{stage.name}_{update:04d}.pt"
        torch.save({
            "stage": stage.name,
            "update": update,
            "connector": self.connector.state_dict(),
            "adapter": adapter_state(self.lm),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "sampler": self.sampler.save_state() if self.sampler else None,
        }, path)
        return path

    def _latest_checkpoint(self, stage: StageConfig):
        files = sorted((self.out_dir / "checkpoints").glob(f"{stage.name}_*.pt"))
        return files[-1] if files else None

    def _resume_from(self, stage: StageConfig) -> int:
        """Restore connector, adapter, optimizer, scaler and sampler from the last
        checkpoint of this stage. Returns the next update index."""
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
        print(f"[resume] {stage.name}: continuing from update {payload['update']} "
              f"({ckpt.name})", flush=True)
        return int(payload["update"])''')

# resume at stage start
s = s.replace(
'''        self._build_optimizer(stage)
        self.lm.train()
        self.connector.train()

        t0 = time.time()
        for step in range(updates):''',
'''        self._build_optimizer(stage)
        self.lm.train()
        self.connector.train()

        start_step = self._resume_from(stage)
        t0 = time.time()
        for step in range(start_step, updates):''')

p.write_text(s, encoding="utf-8")
print("resume + adapter checkpointing added")
